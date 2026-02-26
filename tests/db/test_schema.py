"""Tests for db/schema.py."""

import pytest

from clean_room_agent.db.connection import get_connection


def _table_names(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r["name"] for r in rows]


class TestCuratedSchema:
    def test_tables_created(self, curated_conn):
        tables = _table_names(curated_conn)
        expected = [
            "adapter_metadata", "co_changes", "commits", "dependencies",
            "docstrings", "file_commits", "file_metadata", "files",
            "inline_comments", "repos", "symbol_references", "symbols",
        ]
        for t in expected:
            assert t in tables, f"Missing table: {t}"

    def test_idempotent(self, curated_conn):
        from clean_room_agent.db.schema import create_curated_schema
        # Running again should not error
        create_curated_schema(curated_conn)

    def test_file_source_column(self, curated_conn):
        """Verify the file_source column exists with default 'project'."""
        # Insert a file without specifying file_source
        curated_conn.execute(
            "INSERT INTO repos (path, indexed_at) VALUES (?, datetime('now'))",
            ("/test",),
        )
        repo_id = curated_conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        curated_conn.execute(
            "INSERT INTO files (repo_id, path, language, content_hash, size_bytes) "
            "VALUES (?, 'test.py', 'python', 'abc123', 100)",
            (repo_id,),
        )
        curated_conn.commit()

        row = curated_conn.execute(
            "SELECT file_source FROM files WHERE path = 'test.py'"
        ).fetchone()
        assert row is not None
        assert row["file_source"] == "project"

    def test_file_source_migration_idempotent(self, curated_conn):
        """Running create_curated_schema twice doesn't error (migration is idempotent)."""
        from clean_room_agent.db.schema import create_curated_schema
        # First call already happened via fixture; call twice more
        create_curated_schema(curated_conn)
        create_curated_schema(curated_conn)
        # Verify file_source column still works
        curated_conn.execute(
            "INSERT INTO repos (path, indexed_at) VALUES (?, datetime('now'))",
            ("/test_migration",),
        )
        repo_id = curated_conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        curated_conn.execute(
            "INSERT INTO files (repo_id, path, language, content_hash, size_bytes, file_source) "
            "VALUES (?, 'lib.py', 'python', 'def456', 200, 'library')",
            (repo_id,),
        )
        curated_conn.commit()
        row = curated_conn.execute(
            "SELECT file_source FROM files WHERE path = 'lib.py'"
        ).fetchone()
        assert row["file_source"] == "library"


    def test_migration_reraises_non_duplicate_error(self, curated_conn):
        """A2: non-duplicate OperationalError re-raises instead of being silently swallowed."""
        import sqlite3
        from clean_room_agent.db.schema import _migrate_add_column

        # Try to add a column to a non-existent table — should re-raise
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            _migrate_add_column(curated_conn, "nonexistent_table", "col TEXT")


class TestRawSchema:
    def test_tables_created(self, raw_conn):
        tables = _table_names(raw_conn)
        expected = [
            "adapter_registry", "enrichment_outputs", "index_runs",
            "orchestrator_passes", "orchestrator_runs",
            "retrieval_decisions", "retrieval_llm_calls", "run_attempts",
            "session_archives", "task_runs", "training_datasets",
            "training_plans", "validation_results",
        ]
        for t in expected:
            assert t in tables, f"Missing table: {t}"


class TestSessionSchema:
    def test_kv_table(self, session_conn):
        tables = _table_names(session_conn)
        assert "kv" in tables
