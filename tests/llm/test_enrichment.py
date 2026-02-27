"""Tests for llm/enrichment.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clean_room_agent.db.connection import get_connection
from clean_room_agent.db.queries import (
    insert_docstring,
    insert_symbol,
    upsert_file,
    upsert_repo,
)
from clean_room_agent.llm.enrichment import (
    _build_prompt,
    _parse_enrichment_response,
    enrich_repository,
)


class TestParseEnrichmentResponse:
    def test_valid_json(self):
        text = '{"purpose": "utils", "module": "core", "domain": "general"}'
        result = _parse_enrichment_response(text)
        assert result["purpose"] == "utils"

    def test_strips_markdown_fencing(self):
        text = '```json\n{"purpose": "utils"}\n```'
        result = _parse_enrichment_response(text)
        assert result["purpose"] == "utils"

    def test_strips_plain_markdown_fencing(self):
        text = '```\n{"purpose": "utils"}\n```'
        result = _parse_enrichment_response(text)
        assert result["purpose"] == "utils"

    def test_preserves_inner_content_with_fencing(self):
        """Fencing removal should only strip outer fences, not inner content."""
        inner = '{"purpose": "handles ```code``` blocks"}'
        text = f"```json\n{inner}\n```"
        result = _parse_enrichment_response(text)
        assert result["purpose"] == "handles ```code``` blocks"

    def test_whitespace_around_json(self):
        text = '  \n  {"purpose": "utils"}  \n  '
        result = _parse_enrichment_response(text)
        assert result["purpose"] == "utils"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            _parse_enrichment_response("not json at all")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            _parse_enrichment_response("")


class TestBuildPrompt:
    def test_includes_file_path(self, tmp_path):
        conn = get_connection("curated", repo_path=tmp_path)
        rid = upsert_repo(conn, str(tmp_path), None)
        fid = upsert_file(conn, rid, "src/main.py", "python", "abc", 50)
        conn.commit()

        prompt = _build_prompt("src/main.py", fid, conn, tmp_path)
        assert "File: src/main.py" in prompt
        conn.close()

    def test_includes_symbols(self, tmp_path):
        conn = get_connection("curated", repo_path=tmp_path)
        rid = upsert_repo(conn, str(tmp_path), None)
        fid = upsert_file(conn, rid, "src/main.py", "python", "abc", 50)
        insert_symbol(conn, fid, "process", "function", 1, 10, "def process(data):")
        conn.commit()

        prompt = _build_prompt("src/main.py", fid, conn, tmp_path)
        assert "Symbols:" in prompt
        assert "function process" in prompt
        assert "def process(data):" in prompt
        conn.close()

    def test_includes_docstrings(self, tmp_path):
        conn = get_connection("curated", repo_path=tmp_path)
        rid = upsert_repo(conn, str(tmp_path), None)
        fid = upsert_file(conn, rid, "src/main.py", "python", "abc", 50)
        insert_docstring(conn, fid, "Module for processing data.", "plain")
        conn.commit()

        prompt = _build_prompt("src/main.py", fid, conn, tmp_path)
        assert "Docstrings:" in prompt
        assert "Module for processing data." in prompt
        conn.close()

    def test_includes_source_preview(self, tmp_path):
        conn = get_connection("curated", repo_path=tmp_path)
        rid = upsert_repo(conn, str(tmp_path), None)
        fid = upsert_file(conn, rid, "src/main.py", "python", "abc", 50)
        conn.commit()

        # Create the actual source file
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")

        prompt = _build_prompt("src/main.py", fid, conn, tmp_path)
        assert "Source:" in prompt
        assert "def hello():" in prompt
        conn.close()

    def test_no_source_file_still_works(self, tmp_path):
        conn = get_connection("curated", repo_path=tmp_path)
        rid = upsert_repo(conn, str(tmp_path), None)
        fid = upsert_file(conn, rid, "missing.py", "python", "abc", 50)
        conn.commit()

        prompt = _build_prompt("missing.py", fid, conn, tmp_path)
        assert "File: missing.py" in prompt
        assert "Source:" not in prompt
        conn.close()


class TestEnrichRepository:
    """Tests for enrich_repository orchestration logic."""

    def test_no_indexed_repo_raises(self, tmp_path):
        """enrich_repository raises RuntimeError when no repo is indexed."""
        mock_curated_conn = MagicMock()
        mock_curated_conn.execute.return_value.fetchone.return_value = None
        mock_raw_conn = MagicMock()

        def mock_get_conn(role, *, repo_path, read_only=False):
            if role == "curated":
                return mock_curated_conn
            return mock_raw_conn

        models_config = {
            "provider": "ollama",
            "reasoning": "qwen3:4b",
            "base_url": "http://localhost:11434",
            "context_window": 32768,
            "overrides": {},
            "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0},
        }

        with patch("clean_room_agent.llm.enrichment.get_connection", side_effect=mock_get_conn):
            with patch("clean_room_agent.llm.enrichment.LoggedLLMClient"):
                with pytest.raises(RuntimeError, match="No indexed repo.*Run 'cra index' first"):
                    enrich_repository(tmp_path, models_config)

    def test_skips_already_enriched_files(self, tmp_path):
        """enrich_repository skips files that already have enrichment_outputs."""
        # curated_conn.execute() is called twice:
        #   1. SELECT id FROM repos → fetchone → {"id": 1}
        #   2. SELECT * FROM files  → fetchall → [file row]
        repo_result = MagicMock()
        repo_result.fetchone.return_value = {"id": 1}
        files_result = MagicMock()
        files_result.fetchall.return_value = [
            {"id": 10, "path": "src/main.py", "repo_id": 1},
        ]
        mock_curated_conn = MagicMock()
        mock_curated_conn.execute.side_effect = [repo_result, files_result]

        # raw_conn.execute() checks for existing enrichment → found
        mock_raw_conn = MagicMock()
        mock_raw_conn.execute.return_value.fetchone.return_value = {"id": 99}

        def mock_get_conn(role, *, repo_path, read_only=False):
            if role == "curated":
                return mock_curated_conn
            return mock_raw_conn

        models_config = {
            "provider": "ollama",
            "reasoning": "qwen3:4b",
            "base_url": "http://localhost:11434",
            "context_window": 32768,
            "overrides": {},
            "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0},
        }

        with patch("clean_room_agent.llm.enrichment.get_connection", side_effect=mock_get_conn):
            with patch("clean_room_agent.llm.enrichment.LoggedLLMClient"):
                result = enrich_repository(tmp_path, models_config)

        assert result.files_skipped == 1
        assert result.files_enriched == 0


class TestEnrichmentFailureRecording:
    """H1: Enrichment failure records error to raw DB for audit trail."""

    @staticmethod
    def _make_curated_conn():
        """Create a mock curated conn that handles repo, files, symbols, docstrings queries."""
        mock = MagicMock()
        # Use a callable side_effect so it handles arbitrary numbers of calls
        def execute_handler(query, params=None):
            result = MagicMock()
            if "FROM repos" in query:
                result.fetchone.return_value = {"id": 1}
            elif "FROM files" in query:
                result.fetchall.return_value = [
                    {"id": 10, "path": "src/main.py", "repo_id": 1},
                ]
            elif "FROM symbols" in query:
                result.fetchall.return_value = []
            elif "FROM docstrings" in query:
                result.fetchall.return_value = []
            else:
                result.fetchall.return_value = []
                result.fetchone.return_value = None
            return result
        mock.execute.side_effect = execute_handler
        return mock

    def test_llm_failure_recorded_to_raw_db(self, tmp_path):
        """When LLM call fails, error is recorded in enrichment_outputs."""
        from clean_room_agent.llm.enrichment import enrich_repository

        mock_curated_conn = self._make_curated_conn()

        # Mock raw: no existing enrichment
        mock_raw_conn = MagicMock()
        mock_raw_conn.execute.return_value.fetchone.return_value = None

        def mock_get_conn(role, *, repo_path, read_only=False):
            if role == "curated":
                return mock_curated_conn
            return mock_raw_conn

        # Client that fails on complete()
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("LLM connection lost")
        mock_client.flush.return_value = []
        mock_client.close.return_value = None

        models_config = {
            "provider": "ollama",
            "reasoning": "qwen3:4b",
            "base_url": "http://localhost:11434",
            "context_window": 32768,
            "overrides": {},
            "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0},
        }

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def hello(): pass\n")

        with patch("clean_room_agent.llm.enrichment.get_connection", side_effect=mock_get_conn):
            with patch("clean_room_agent.llm.enrichment.LoggedLLMClient", return_value=mock_client):
                with patch("clean_room_agent.llm.enrichment.insert_enrichment_output") as mock_insert:
                    with pytest.raises(RuntimeError, match="LLM connection lost"):
                        enrich_repository(tmp_path, models_config)

        # Verify error was recorded before re-raise
        assert mock_insert.call_count == 1
        call_kwargs = mock_insert.call_args
        assert "[ERROR] RuntimeError: LLM connection lost" in str(call_kwargs)
        mock_client.flush.assert_called()

    def test_parse_failure_recorded_to_raw_db(self, tmp_path):
        """When JSON parse fails, error is recorded in enrichment_outputs."""
        from clean_room_agent.llm.enrichment import enrich_repository

        mock_curated_conn = self._make_curated_conn()

        mock_raw_conn = MagicMock()
        mock_raw_conn.execute.return_value.fetchone.return_value = None

        def mock_get_conn(role, *, repo_path, read_only=False):
            if role == "curated":
                return mock_curated_conn
            return mock_raw_conn

        # Client returns unparseable response
        mock_response = MagicMock()
        mock_response.text = "not valid json"
        mock_response.thinking = None
        mock_response.prompt_tokens = 100
        mock_response.completion_tokens = 50
        mock_response.latency_ms = 10
        mock_client = MagicMock()
        mock_client.complete.return_value = mock_response
        mock_client.flush.return_value = []
        mock_client.close.return_value = None

        models_config = {
            "provider": "ollama",
            "reasoning": "qwen3:4b",
            "base_url": "http://localhost:11434",
            "context_window": 32768,
            "overrides": {},
            "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0},
        }

        (tmp_path / "src").mkdir(exist_ok=True)
        (tmp_path / "src" / "main.py").write_text("def hello(): pass\n")

        with patch("clean_room_agent.llm.enrichment.get_connection", side_effect=mock_get_conn):
            with patch("clean_room_agent.llm.enrichment.LoggedLLMClient", return_value=mock_client):
                with patch("clean_room_agent.llm.enrichment.insert_enrichment_output") as mock_insert:
                    with pytest.raises(ValueError, match="Failed to parse"):
                        enrich_repository(tmp_path, models_config)

        # Parse failure is caught by except Exception, so error is recorded
        assert mock_insert.call_count == 1
        call_kwargs = mock_insert.call_args
        assert "[ERROR] ValueError" in str(call_kwargs)
