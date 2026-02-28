"""Tests for transparency audit fixes A1-A12."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clean_room_agent.llm.client import LLMResponse, ModelConfig


# -- A3: documentation pass logging --

class TestA3DocumentationLogging:
    """A3: Documentation pass logs when LLM indicates no changes or no edits returned."""

    def test_no_changes_needed_logged(self, caplog):
        from clean_room_agent.execute.documentation import execute_documentation_file

        mock_llm = MagicMock()
        mock_llm.complete.return_value = LLMResponse(
            text="No changes needed for this file.",
            thinking=None, prompt_tokens=50, completion_tokens=20, latency_ms=5,
        )
        model_config = ModelConfig(
            model="test", base_url="http://test", context_window=32768, max_tokens=4096,
        )

        repo_path = Path("/tmp/test_repo")
        with patch.object(Path, "read_text", return_value="def foo(): pass\n"):
            with patch("clean_room_agent.execute.documentation.build_documentation_prompt",
                       return_value=("sys", "user")):
                with caplog.at_level(logging.INFO):
                    result = execute_documentation_file(
                        "test.py", repo_path, "task desc", "part desc",
                        mock_llm, model_config,
                    )

        assert result.success
        assert not result.edits
        assert any("no changes needed" in r.message for r in caplog.records)

    def test_no_edits_logged_debug(self, caplog):
        from clean_room_agent.execute.documentation import run_documentation_pass

        # Mock execute_documentation_file to return success with empty edits
        with patch("clean_room_agent.execute.documentation.execute_documentation_file") as mock_exec:
            mock_exec.return_value = MagicMock(success=True, edits=[], error_info=None)
            with caplog.at_level(logging.DEBUG):
                results = run_documentation_pass(
                    ["test.py"], Path("/tmp/repo"), "task", "part",
                    MagicMock(), MagicMock(),
                )

        assert results == []
        assert any("no edits returned" in r.message for r in caplog.records)


# -- A4: refilter unknown paths logging --

class TestA4RefilterLogging:
    """A4: Refilter logs when LLM suggests unknown paths."""

    def test_unknown_paths_logged(self, caplog):
        from clean_room_agent.retrieval.context_assembly import _refilter_files

        rendered_files = [
            {"path": "valid.py", "detail": "primary", "tokens": 100, "file_id": 1},
        ]
        context = MagicMock()
        context.task.raw_task = "some task"

        llm = MagicMock()
        llm.config = ModelConfig(
            model="test", base_url="http://test", context_window=32768, max_tokens=4096,
        )
        llm.complete.return_value = LLMResponse(
            text=json.dumps(["valid.py", "hallucinated.py"]),
            thinking=None, prompt_tokens=50, completion_tokens=20, latency_ms=5,
        )

        with caplog.at_level(logging.WARNING):
            keep = _refilter_files(rendered_files, 500, context, llm)

        assert "valid.py" in keep
        assert "hallucinated.py" not in keep
        assert any("unknown paths" in r.message for r in caplog.records)


# -- A9: full response in parse_implement_response error --

class TestA9FullResponseInError:
    """A9: parse_implement_response includes full response text in error, not truncated."""

    def test_full_text_in_error(self):
        from clean_room_agent.execute.parsers import parse_implement_response

        long_text = "x" * 1000  # > 500 chars
        with pytest.raises(ValueError) as exc_info:
            parse_implement_response(long_text)

        # The full text should be present, not truncated to 500 chars
        assert long_text in str(exc_info.value)


# -- A11: config constant extraction --

class TestA11ConfigConstant:
    """A11: _DEFAULT_CODING_STYLE constant used instead of hardcoded strings."""

    def test_constant_value(self):
        from clean_room_agent.config import _DEFAULT_CODING_STYLE
        assert _DEFAULT_CODING_STYLE == "development"

    def test_require_environment_config_none_returns_default(self):
        from clean_room_agent.config import _DEFAULT_CODING_STYLE, require_environment_config
        result = require_environment_config(None)
        assert result["coding_style"] == _DEFAULT_CODING_STYLE

    def test_require_environment_config_with_explicit_default(self):
        from clean_room_agent.config import _DEFAULT_CODING_STYLE, require_environment_config
        result = require_environment_config({"environment": {"coding_style": _DEFAULT_CODING_STYLE}})
        assert result["coding_style"] == _DEFAULT_CODING_STYLE


# -- A12: similarity deny at WARNING level --

class TestA12SimilarityDenyLogLevel:
    """A12: Similarity pair denial logs at WARNING level (R2 pattern)."""

    def test_deny_logged_at_warning(self, caplog):
        from clean_room_agent.retrieval.similarity_stage import judge_similarity

        mock_llm = MagicMock()
        mock_llm.config = ModelConfig(
            model="test", base_url="http://test", context_window=32768, max_tokens=4096,
        )

        # LLM returns a single pair with keep=False
        mock_llm.complete.return_value = LLMResponse(
            text=json.dumps([{"pair_id": 0, "keep": False, "reason": "not similar"}]),
            thinking=None, prompt_tokens=50, completion_tokens=20, latency_ms=5,
        )

        sym_a = MagicMock()
        sym_a.name = "func_a"
        sym_a.kind = "function"
        sym_a.start_line = 1
        sym_a.end_line = 10
        sym_b = MagicMock()
        sym_b.name = "func_b"
        sym_b.kind = "function"
        sym_b.start_line = 20
        sym_b.end_line = 30

        pairs = [{
            "pair_id": 0,
            "sym_a": sym_a, "sym_b": sym_b,
            "score": 0.5,
            "signals": {"line_ratio": 1.0, "callee_jaccard": 0.0, "name_lcs": 0.5, "same_parent": False},
        }]
        task = MagicMock()
        task.raw_task = "test task"
        task.intent_summary = "test"

        with caplog.at_level(logging.WARNING):
            result = judge_similarity(pairs, task, mock_llm)

        assert result == []
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("denied" in m.lower() for m in warning_msgs)


# -- A1: enrichment uses LoggedLLMClient --

class TestA1EnrichmentLoggedClient:
    """A1: enrich_repository uses LoggedLLMClient and flushes records."""

    def test_flush_called_per_file(self, tmp_path):
        from clean_room_agent.llm.enrichment import enrich_repository

        # curated_conn: needs to handle multiple execute() calls:
        #   1. SELECT id FROM repos
        #   2. SELECT * FROM files
        #   3+ SELECT name... FROM symbols, SELECT content FROM docstrings (per file)
        mock_curated_conn = MagicMock()
        repo_result = MagicMock()
        repo_result.fetchone.return_value = {"id": 1}
        files_result = MagicMock()
        files_result.fetchall.return_value = [
            {"id": 10, "path": "src/main.py", "repo_id": 1},
        ]
        # Symbols and docstrings queries return empty
        empty_result = MagicMock()
        empty_result.fetchall.return_value = []

        mock_curated_conn.execute.side_effect = [
            repo_result, files_result, empty_result, empty_result,
        ]

        # raw_conn: no existing enrichment
        mock_raw_conn = MagicMock()
        mock_raw_conn.execute.return_value.fetchone.return_value = None

        def mock_get_conn(role, *, repo_path, read_only=False):
            if role == "curated":
                return mock_curated_conn
            return mock_raw_conn

        mock_client_instance = MagicMock()
        mock_client_instance.complete.return_value = LLMResponse(
            text=json.dumps({
                "purpose": "test", "module": "core", "domain": "test",
                "concepts": ["a"], "public_api_surface": [], "complexity_notes": "",
            }),
            thinking=None, prompt_tokens=100, completion_tokens=50, latency_ms=10,
        )
        mock_client_instance.flush.return_value = []

        models_config = {
            "provider": "ollama",
            "reasoning": "qwen3:4b",
            "base_url": "http://localhost:11434",
            "context_window": 32768,
            "overrides": {},
            "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0},
        }

        # Create source file
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("def hello(): pass\n")

        with patch("clean_room_agent.llm.enrichment.get_connection", side_effect=mock_get_conn):
            with patch("clean_room_agent.llm.enrichment.LoggedLLMClient", return_value=mock_client_instance):
                with patch("clean_room_agent.llm.enrichment.insert_enrichment_output"):
                    result = enrich_repository(tmp_path, models_config)

        assert result.files_enriched == 1
        # flush called once per enriched file + once in finally block
        assert mock_client_instance.flush.call_count >= 2

    def test_finally_logs_unflushed(self, caplog):
        from clean_room_agent.llm.enrichment import enrich_repository

        mock_curated_conn = MagicMock()
        mock_curated_conn.execute.return_value.fetchone.return_value = None

        mock_raw_conn = MagicMock()

        def mock_get_conn(role, *, repo_path, read_only=False):
            if role == "curated":
                return mock_curated_conn
            return mock_raw_conn

        mock_client_instance = MagicMock()
        # flush returns remaining records in finally block
        mock_client_instance.flush.return_value = [{"prompt": "x", "response": "y"}]

        models_config = {
            "provider": "ollama",
            "reasoning": "qwen3:4b",
            "base_url": "http://localhost:11434",
            "context_window": 32768,
            "overrides": {},
            "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0},
        }

        with patch("clean_room_agent.llm.enrichment.get_connection", side_effect=mock_get_conn):
            with patch("clean_room_agent.llm.enrichment.LoggedLLMClient", return_value=mock_client_instance):
                with caplog.at_level(logging.WARNING):
                    with pytest.raises(RuntimeError, match="No indexed repo"):
                        enrich_repository(Path("/tmp/test"), models_config)

        assert any("unflushed" in r.message for r in caplog.records)


