"""Tests for Batch 2 fixes (H2, M3) and Batch 3 M2 (resolve_retrieval_param)."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestLibraryScannerWarnings:
    """H2: Silent catches in library_scanner now log warnings."""

    def test_read_failure_warns(self, tmp_path, caplog):
        """Failed file read in _auto_resolve logs a warning."""
        from clean_room_agent.indexer.library_scanner import _auto_resolve

        # Create a .py file that will fail to read
        py_file = tmp_path / "broken.py"
        py_file.write_text("import os\n")

        mock_parser = MagicMock()
        # Make read_bytes fail
        with patch("pathlib.Path.rglob") as mock_rglob:
            failing_path = MagicMock(spec=Path)
            failing_path.relative_to.return_value = Path("broken.py")
            failing_path.read_bytes.side_effect = OSError("Permission denied")
            mock_rglob.return_value = [failing_path]

            with caplog.at_level(logging.WARNING, logger="clean_room_agent.indexer.library_scanner"):
                with patch("clean_room_agent.indexer.library_scanner.get_parser"):
                    _auto_resolve(tmp_path)

        assert any("Failed to read" in r.message for r in caplog.records)

    def test_parse_failure_warns(self, tmp_path, caplog):
        """Failed parse in _auto_resolve logs a warning."""
        from clean_room_agent.indexer.library_scanner import _auto_resolve

        py_file = tmp_path / "bad_syntax.py"
        py_file.write_text("def ???\n")

        mock_parser = MagicMock()
        mock_parser.parse.side_effect = SyntaxError("invalid syntax")

        with caplog.at_level(logging.WARNING, logger="clean_room_agent.indexer.library_scanner"):
            with patch("clean_room_agent.indexer.library_scanner.get_parser", return_value=mock_parser):
                _auto_resolve(tmp_path)

        assert any("Failed to parse" in r.message for r in caplog.records)

    def test_resolve_failure_warns(self, caplog):
        """Failed find_spec in _auto_resolve logs a warning."""
        from clean_room_agent.indexer.library_scanner import _auto_resolve

        mock_parser = MagicMock()
        mock_result = MagicMock()
        mock_result.imports = [MagicMock(is_relative=False, module="nonexistent_pkg")]
        mock_parser.parse.return_value = mock_result

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "test.py").write_text("import nonexistent_pkg\n")

            with caplog.at_level(logging.WARNING, logger="clean_room_agent.indexer.library_scanner"):
                with patch("clean_room_agent.indexer.library_scanner.get_parser", return_value=mock_parser):
                    with patch("importlib.util.find_spec", side_effect=ImportError("No module")):
                        _auto_resolve(td_path)

        assert any("Failed to resolve library" in r.message for r in caplog.records)


class TestPrecisionMissingSignature:
    """M3: Missing signature in precision stage logs at DEBUG level."""

    def test_missing_signature_debug_log(self, caplog):
        """Symbol with no signature logs a debug message."""
        from clean_room_agent.retrieval.precision_stage import extract_precision_symbols
        from clean_room_agent.retrieval.dataclasses import TaskQuery

        mock_kb = MagicMock()
        mock_file = MagicMock()
        mock_file.path = "src/main.py"
        mock_file.language = "python"
        mock_file.file_source = "project"
        mock_kb.get_file_by_id.return_value = mock_file

        # Symbol with no signature
        mock_sym = MagicMock()
        mock_sym.id = 1
        mock_sym.name = "MY_CONSTANT"
        mock_sym.kind = "variable"
        mock_sym.start_line = 10
        mock_sym.end_line = 10
        mock_sym.signature = None  # No signature
        mock_kb.get_symbols_for_file.return_value = [mock_sym]
        mock_kb.get_symbol_neighbors.return_value = []

        task = MagicMock(spec=TaskQuery)
        task.keywords = []
        task.mentioned_symbols = []

        with caplog.at_level(logging.DEBUG, logger="clean_room_agent.retrieval.precision_stage"):
            candidates = extract_precision_symbols({1}, task, mock_kb)

        assert len(candidates) == 1
        assert candidates[0]["signature"] == ""
        assert any(
            "MY_CONSTANT" in r.message and "no signature" in r.message
            for r in caplog.records
        )


class TestResolveRetrievalParam:
    """M2: resolve_retrieval_param with documented defaults."""

    def test_returns_default_when_absent(self):
        from clean_room_agent.retrieval.stage import resolve_retrieval_param
        assert resolve_retrieval_param({}, "max_deps") == 30

    def test_returns_override_when_present(self):
        from clean_room_agent.retrieval.stage import resolve_retrieval_param
        assert resolve_retrieval_param({"max_deps": 50}, "max_deps") == 50

    def test_unknown_key_raises(self):
        from clean_room_agent.retrieval.stage import resolve_retrieval_param
        with pytest.raises(KeyError, match="Unknown retrieval parameter"):
            resolve_retrieval_param({}, "nonexistent_param")

    def test_bad_type_raises(self):
        from clean_room_agent.retrieval.stage import resolve_retrieval_param
        with pytest.raises(TypeError, match="must be numeric"):
            resolve_retrieval_param({"max_deps": "thirty"}, "max_deps")
