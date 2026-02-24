"""Shared test fixtures."""

import sqlite3
from pathlib import Path

import pytest

from clean_room_agent.db.connection import get_connection


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temporary directory acting as a repo root."""
    return tmp_path


@pytest.fixture
def curated_conn(tmp_repo):
    """Return a curated DB connection in a temp repo."""
    conn = get_connection("curated", repo_path=tmp_repo)
    yield conn
    conn.close()


@pytest.fixture
def raw_conn(tmp_repo):
    """Return a raw DB connection in a temp repo."""
    conn = get_connection("raw", repo_path=tmp_repo)
    yield conn
    conn.close()


@pytest.fixture
def session_conn(tmp_repo):
    """Return a session DB connection in a temp repo."""
    conn = get_connection("session", repo_path=tmp_repo, task_id="test-task-001")
    yield conn
    conn.close()
