"""Tests for db/connection.py."""

import sqlite3

import pytest

from clean_room_agent.db.connection import get_connection


class TestGetConnection:
    def test_curated_creates_file(self, tmp_repo):
        conn = get_connection("curated", repo_path=tmp_repo)
        assert (tmp_repo / ".clean_room" / "curated.sqlite").exists()
        conn.close()

    def test_raw_creates_file(self, tmp_repo):
        conn = get_connection("raw", repo_path=tmp_repo)
        assert (tmp_repo / ".clean_room" / "raw.sqlite").exists()
        conn.close()

    def test_session_creates_file(self, tmp_repo):
        conn = get_connection("session", repo_path=tmp_repo, task_id="abc-123")
        assert (tmp_repo / ".clean_room" / "sessions" / "session_abc-123.sqlite").exists()
        conn.close()

    def test_session_requires_task_id(self, tmp_repo):
        with pytest.raises(ValueError, match="task_id is required"):
            get_connection("session", repo_path=tmp_repo)

    def test_invalid_role(self, tmp_repo):
        with pytest.raises(ValueError, match="Invalid role"):
            get_connection("invalid", repo_path=tmp_repo)

    def test_wal_mode(self, tmp_repo):
        conn = get_connection("curated", repo_path=tmp_repo)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_foreign_keys_enabled(self, tmp_repo):
        conn = get_connection("curated", repo_path=tmp_repo)
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
        conn.close()

    def test_row_factory(self, tmp_repo):
        conn = get_connection("curated", repo_path=tmp_repo)
        assert conn.row_factory is sqlite3.Row
        conn.close()

    def test_read_only_existing_db(self, tmp_repo):
        # First create the DB
        conn = get_connection("curated", repo_path=tmp_repo)
        conn.close()
        # Now open read-only
        ro = get_connection("curated", repo_path=tmp_repo, read_only=True)
        # Writing should fail
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("INSERT INTO repos (path, indexed_at) VALUES ('x', 'now')")
        ro.close()

    def test_read_only_nonexistent_db_fails(self, tmp_repo):
        with pytest.raises(sqlite3.OperationalError):
            get_connection("curated", repo_path=tmp_repo, read_only=True)

    def test_schema_applied_on_creation(self, tmp_repo):
        conn = get_connection("curated", repo_path=tmp_repo)
        # Should be able to query tables that the schema creates
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "repos" in table_names
        assert "files" in table_names
        assert "symbols" in table_names
        conn.close()

    def test_schema_idempotent(self, tmp_repo):
        """Opening the same DB twice should not error (CREATE IF NOT EXISTS)."""
        conn1 = get_connection("curated", repo_path=tmp_repo)
        count1 = conn1.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        conn1.close()
        conn2 = get_connection("curated", repo_path=tmp_repo)
        count2 = conn2.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        assert count2 > 0
        assert count1 == count2
        conn2.close()


class TestEnsureSchema:
    def test_ensure_schema_applies_tables(self, tmp_repo):
        from clean_room_agent.db.connection import ensure_schema

        conn = get_connection("curated", repo_path=tmp_repo)
        # Schema already applied by get_connection, but ensure_schema should be idempotent
        ensure_schema(conn, "curated")
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "repos" in table_names
        assert "files" in table_names
        assert "symbols" in table_names
        conn.close()

    def test_ensure_schema_invalid_role(self, tmp_repo):
        from clean_room_agent.db.connection import ensure_schema

        conn = get_connection("curated", repo_path=tmp_repo)
        with pytest.raises(ValueError, match="Invalid role"):
            ensure_schema(conn, "bogus")
        conn.close()

    def test_ensure_schema_raw(self, tmp_repo):
        from clean_room_agent.db.connection import ensure_schema

        conn = get_connection("raw", repo_path=tmp_repo)
        ensure_schema(conn, "raw")
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "index_runs" in table_names
        assert "enrichment_outputs" in table_names
        conn.close()

    def test_ensure_schema_session(self, tmp_repo):
        from clean_room_agent.db.connection import ensure_schema

        conn = get_connection("session", repo_path=tmp_repo, task_id="t1")
        ensure_schema(conn, "session")
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "kv" in table_names
        conn.close()
