"""Tests for knowledge base indexer orchestrator."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from clean_room_agent.db.queries import get_ref_section_content_by_file_id
from clean_room_agent.db.schema import create_curated_schema
from clean_room_agent.knowledge_base.indexer import (
    SOURCE_REGISTRY,
    _insert_source,
    _parse_source,
    index_knowledge_base,
)
from clean_room_agent.knowledge_base.models import RefSection


_KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base" / "c_references"


class TestSourceRegistry:
    """Verify source registry entries are well-formed."""

    def test_all_entries_have_format(self):
        for name, config in SOURCE_REGISTRY.items():
            assert "format" in config, f"{name} missing format"
            assert "type" in config, f"{name} missing type"

    def test_known_formats(self):
        valid_formats = {"text", "pdf", "html", "markdown"}
        for name, config in SOURCE_REGISTRY.items():
            assert config["format"] in valid_formats, f"{name} has unknown format"


class TestParseSource:
    """Test _parse_source for available sources."""

    @pytest.mark.skipif(not (_KB_DIR / "knr2").exists(), reason="K&R not available")
    def test_parse_knr2(self):
        sections = _parse_source("knr2", SOURCE_REGISTRY["knr2"], _KB_DIR / "knr2")
        assert len(sections) > 100
        chapters = [s for s in sections if s.section_type == "chapter"]
        assert len(chapters) == 8

    @pytest.mark.skipif(not (_KB_DIR / "cert_c").exists(), reason="CERT C not available")
    def test_parse_cert_c(self):
        sections = _parse_source("cert_c", SOURCE_REGISTRY["cert_c"], _KB_DIR / "cert_c")
        assert len(sections) > 100
        rules = [s for s in sections if s.section_type == "rule"]
        assert len(rules) > 100


class TestInsertSource:
    """Test _insert_source with a temp DB."""

    def _make_temp_db(self, tmp_path):
        """Create a temp curated DB with schema."""
        db_dir = tmp_path / ".clean_room"
        db_dir.mkdir(parents=True)
        db_path = db_dir / "curated.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        create_curated_schema(conn)
        conn.commit()
        conn.close()
        return tmp_path

    def test_insert_synthetic_sections(self, tmp_path):
        repo_path = self._make_temp_db(tmp_path)
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        sections = [
            RefSection(
                title="Test Chapter",
                section_path="ch1",
                content="Chapter content here.",
                section_type="chapter",
                ordering=0,
            ),
            RefSection(
                title="Test Section",
                section_path="ch1/1.1",
                content="Section content here.",
                section_type="section",
                ordering=1,
                parent_section_path="ch1",
            ),
        ]

        config = {"type": "book", "format": "text"}
        n_sections, n_bridge = _insert_source(
            "test_source", config, sections, kb_path, repo_path,
        )
        assert n_sections == 2
        assert n_bridge == 2

        # Verify data in DB
        conn = sqlite3.connect(str(repo_path / ".clean_room" / "curated.sqlite"))
        conn.row_factory = sqlite3.Row

        # Check ref_sources
        source = conn.execute("SELECT * FROM ref_sources WHERE name = 'test_source'").fetchone()
        assert source is not None

        # Check ref_sections
        secs = conn.execute("SELECT * FROM ref_sections").fetchall()
        assert len(secs) == 2

        # Check bridge files
        files = conn.execute(
            "SELECT * FROM files WHERE file_source = 'knowledge_base'"
        ).fetchall()
        assert len(files) == 2
        paths = {f["path"] for f in files}
        assert "kb/test_source/ch1" in paths
        assert "kb/test_source/ch1/1.1" in paths

        # Check file_metadata
        meta = conn.execute("SELECT * FROM file_metadata").fetchall()
        assert len(meta) == 2

        conn.close()

    def test_bridge_file_content_lookup(self, tmp_path):
        """Test get_ref_section_content_by_file_id via bridge."""
        repo_path = self._make_temp_db(tmp_path)
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        sections = [
            RefSection(
                title="Malloc Reference",
                section_path="memory/malloc",
                content="void *malloc(size_t size); — allocates memory.",
                section_type="function_ref",
                ordering=0,
            ),
        ]

        _insert_source("test_ref", {"type": "api_reference", "format": "html"},
                        sections, kb_path, repo_path)

        conn = sqlite3.connect(str(repo_path / ".clean_room" / "curated.sqlite"))
        conn.row_factory = sqlite3.Row

        # Find the bridge file
        file_row = conn.execute(
            "SELECT id FROM files WHERE path = 'kb/test_ref/memory/malloc'"
        ).fetchone()
        assert file_row is not None

        # Look up content via bridge
        content = get_ref_section_content_by_file_id(conn, file_row["id"])
        assert content is not None
        assert "malloc" in content
        assert "allocates memory" in content

        conn.close()

    def test_reindex_clears_old_data(self, tmp_path):
        """Test that re-indexing a source clears previous data."""
        repo_path = self._make_temp_db(tmp_path)
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        sections_v1 = [
            RefSection(
                title="Old Section",
                section_path="s1",
                content="Old content.",
                section_type="section",
                ordering=0,
            ),
        ]
        sections_v2 = [
            RefSection(
                title="New Section",
                section_path="s1",
                content="New content.",
                section_type="section",
                ordering=0,
            ),
            RefSection(
                title="Extra Section",
                section_path="s2",
                content="Extra content.",
                section_type="section",
                ordering=1,
            ),
        ]

        config = {"type": "book", "format": "text"}
        _insert_source("test", config, sections_v1, kb_path, repo_path)
        _insert_source("test", config, sections_v2, kb_path, repo_path)

        conn = sqlite3.connect(str(repo_path / ".clean_room" / "curated.sqlite"))
        conn.row_factory = sqlite3.Row

        secs = conn.execute("SELECT * FROM ref_sections").fetchall()
        assert len(secs) == 2

        files = conn.execute(
            "SELECT * FROM files WHERE file_source = 'knowledge_base'"
        ).fetchall()
        assert len(files) == 2

        conn.close()


@pytest.mark.skipif(
    not (_KB_DIR / "knr2").exists() or not (_KB_DIR / "cert_c").exists(),
    reason="Reference sources not available",
)
class TestIndexKnowledgeBaseIntegration:
    """Full integration test: index real sources into temp DB."""

    def test_index_knr2_and_cert_c(self, tmp_path):
        # Set up temp repo with .clean_room dir
        db_dir = tmp_path / ".clean_room"
        db_dir.mkdir(parents=True)
        db_path = db_dir / "curated.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        create_curated_schema(conn)
        conn.commit()
        conn.close()

        result = index_knowledge_base(
            _KB_DIR,
            tmp_path,
            sources=["knr2", "cert_c"],
        )

        assert result.sources_indexed == 2
        assert result.sections_total > 200
        assert result.bridge_files_created > 200
        assert len(result.errors) == 0

        # Verify we can query back
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        sources = conn.execute("SELECT * FROM ref_sources").fetchall()
        assert len(sources) == 2

        # Verify bridge files are searchable via file_metadata
        kb_files = conn.execute(
            "SELECT f.path, m.domain FROM files f "
            "JOIN file_metadata m ON f.id = m.file_id "
            "WHERE f.file_source = 'knowledge_base' AND m.domain IS NOT NULL "
            "LIMIT 5"
        ).fetchall()
        assert len(kb_files) > 0

        conn.close()
