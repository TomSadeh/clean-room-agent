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
