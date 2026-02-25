"""Tests for KnowledgeBase file metadata methods (Feature 1: metadata surfacing)."""

import pytest

from clean_room_agent.db.queries import (
    upsert_file,
    upsert_file_metadata,
    upsert_repo,
)
from clean_room_agent.query.api import KnowledgeBase
from clean_room_agent.query.models import FileMetadata


@pytest.fixture
def kb_with_metadata(curated_conn):
    """KnowledgeBase with files that have metadata."""
    repo_id = upsert_repo(curated_conn, "/test/repo", None)
    f1 = upsert_file(curated_conn, repo_id, "src/foo.py", "python", "abc123", 100)
    f2 = upsert_file(curated_conn, repo_id, "src/bar.py", "python", "def456", 200)
    f3 = upsert_file(curated_conn, repo_id, "src/baz.py", "python", "ghi789", 150)

    upsert_file_metadata(
        curated_conn, f1,
        purpose="Helper utilities",
        domain="core",
        concepts="parsing,validation",
        module="utils",
        public_api_surface='["parse", "validate"]',
        complexity_notes="Low complexity",
    )
    upsert_file_metadata(
        curated_conn, f2,
        purpose="Database access layer",
        domain="persistence",
        concepts="sql,orm",
    )
    # f3 intentionally has no metadata

    curated_conn.commit()
    kb = KnowledgeBase(curated_conn)
    return kb, repo_id, f1, f2, f3


class TestGetFileMetadata:
    def test_get_file_metadata_returns_data(self, kb_with_metadata):
        """Insert metadata via upsert_file_metadata, then get_file_metadata returns
        FileMetadata with correct fields."""
        kb, repo_id, f1, f2, f3 = kb_with_metadata

        meta = kb.get_file_metadata(f1)
        assert meta is not None
        assert isinstance(meta, FileMetadata)
        assert meta.file_id == f1
        assert meta.purpose == "Helper utilities"
        assert meta.domain == "core"
        assert meta.concepts == "parsing,validation"
        assert meta.module == "utils"
        assert meta.public_api_surface == '["parse", "validate"]'
        assert meta.complexity_notes == "Low complexity"

    def test_get_file_metadata_returns_none(self, kb_with_metadata):
        """get_file_metadata for non-existent file returns None."""
        kb, repo_id, f1, f2, f3 = kb_with_metadata

        # f3 has no metadata
        meta = kb.get_file_metadata(f3)
        assert meta is None

    def test_get_file_metadata_nonexistent_file_id(self, kb_with_metadata):
        """get_file_metadata for a file_id not in the DB returns None."""
        kb, *_ = kb_with_metadata

        meta = kb.get_file_metadata(99999)
        assert meta is None


class TestGetFileMetadataBatch:
    def test_get_file_metadata_batch_returns_map(self, kb_with_metadata):
        """Insert metadata for multiple files, batch returns dict[int, FileMetadata]."""
        kb, repo_id, f1, f2, f3 = kb_with_metadata

        result = kb.get_file_metadata_batch([f1, f2])
        assert isinstance(result, dict)
        assert len(result) == 2
        assert f1 in result
        assert f2 in result
        assert isinstance(result[f1], FileMetadata)
        assert isinstance(result[f2], FileMetadata)
        assert result[f1].purpose == "Helper utilities"
        assert result[f2].purpose == "Database access layer"

    def test_get_file_metadata_batch_empty(self, kb_with_metadata):
        """Empty file_ids list returns empty dict."""
        kb, *_ = kb_with_metadata

        result = kb.get_file_metadata_batch([])
        assert result == {}

    def test_get_file_metadata_batch_partial(self, kb_with_metadata):
        """Some files have metadata, some don't -- only returns those with metadata."""
        kb, repo_id, f1, f2, f3 = kb_with_metadata

        # f1 and f2 have metadata, f3 does not
        result = kb.get_file_metadata_batch([f1, f2, f3])
        assert len(result) == 2
        assert f1 in result
        assert f2 in result
        assert f3 not in result

    def test_get_file_metadata_batch_all_missing(self, kb_with_metadata):
        """All requested file_ids have no metadata -- returns empty dict."""
        kb, repo_id, f1, f2, f3 = kb_with_metadata

        result = kb.get_file_metadata_batch([f3, 99999])
        assert result == {}

    def test_get_file_metadata_batch_preserves_all_fields(self, kb_with_metadata):
        """Batch returns FileMetadata with all fields populated correctly."""
        kb, repo_id, f1, f2, f3 = kb_with_metadata

        result = kb.get_file_metadata_batch([f1])
        meta = result[f1]
        assert meta.file_id == f1
        assert meta.purpose == "Helper utilities"
        assert meta.module == "utils"
        assert meta.domain == "core"
        assert meta.concepts == "parsing,validation"
        assert meta.public_api_surface == '["parse", "validate"]'
        assert meta.complexity_notes == "Low complexity"

    def test_get_file_metadata_batch_single_item(self, kb_with_metadata):
        """Batch with a single file_id works correctly."""
        kb, repo_id, f1, f2, f3 = kb_with_metadata

        result = kb.get_file_metadata_batch([f2])
        assert len(result) == 1
        assert result[f2].domain == "persistence"


class TestSearchFilesByMetadata:
    """Tests for search_files_by_metadata (1-P2-5)."""

    def test_search_by_module(self, kb_with_metadata):
        """Search by module field returns matching files."""
        kb, repo_id, f1, f2, f3 = kb_with_metadata
        results = kb.search_files_by_metadata(repo_id, module="utils")
        assert len(results) == 1
        assert results[0].id == f1

    def test_search_by_domain(self, kb_with_metadata):
        """Search by domain field returns matching files."""
        kb, repo_id, f1, f2, f3 = kb_with_metadata
        results = kb.search_files_by_metadata(repo_id, domain="persistence")
        assert len(results) == 1
        assert results[0].id == f2

    def test_search_by_concepts(self, kb_with_metadata):
        """Search by concepts with LIKE matching."""
        kb, repo_id, f1, f2, f3 = kb_with_metadata
        results = kb.search_files_by_metadata(repo_id, concepts="pars")
        assert len(results) == 1
        assert results[0].id == f1

    def test_search_combined_filters(self, kb_with_metadata):
        """Search with multiple filters (AND logic)."""
        kb, repo_id, f1, f2, f3 = kb_with_metadata
        results = kb.search_files_by_metadata(repo_id, domain="core", module="utils")
        assert len(results) == 1
        assert results[0].id == f1

    def test_search_no_results(self, kb_with_metadata):
        """Search with no matching files returns empty list."""
        kb, repo_id, *_ = kb_with_metadata
        results = kb.search_files_by_metadata(repo_id, domain="nonexistent")
        assert results == []

    def test_search_no_filters_raises(self, kb_with_metadata):
        """Search with all None filters raises ValueError."""
        kb, repo_id, *_ = kb_with_metadata
        with pytest.raises(ValueError, match="at least one filter"):
            kb.search_files_by_metadata(repo_id)
