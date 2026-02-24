"""Tests for llm/enrichment.py."""

import pytest

from clean_room_agent.db.connection import get_connection
from clean_room_agent.db.queries import (
    insert_docstring,
    insert_symbol,
    upsert_file,
    upsert_repo,
)
from clean_room_agent.llm.enrichment import _build_prompt, _parse_enrichment_response


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
        assert "Source preview" in prompt
        assert "def hello():" in prompt
        conn.close()

    def test_no_source_file_still_works(self, tmp_path):
        conn = get_connection("curated", repo_path=tmp_path)
        rid = upsert_repo(conn, str(tmp_path), None)
        fid = upsert_file(conn, rid, "missing.py", "python", "abc", 50)
        conn.commit()

        prompt = _build_prompt("missing.py", fid, conn, tmp_path)
        assert "File: missing.py" in prompt
        assert "Source preview" not in prompt
        conn.close()
