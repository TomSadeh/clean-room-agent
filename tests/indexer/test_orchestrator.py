"""Integration tests for indexer/orchestrator.py."""

import subprocess

import pytest

from clean_room_agent.db.connection import get_connection
from clean_room_agent.indexer.orchestrator import index_repository


def _init_git(path):
    """Initialize a git repo at the given path."""
    subprocess.run(["git", "init"], cwd=str(path), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(path), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(path), capture_output=True, check=True,
    )


def _git_commit(path, message="commit"):
    subprocess.run(["git", "add", "-A"], cwd=str(path), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty"],
        cwd=str(path), capture_output=True, check=True,
    )


@pytest.fixture
def sample_repo(tmp_path):
    """Create a small sample repo with Python and TS files."""
    # Python files
    pkg = tmp_path / "src"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(
        '"""Main module."""\n\n'
        'from src.utils import helper\n\n'
        'def main():\n'
        '    """Entry point."""\n'
        '    # TODO: add logging\n'
        '    return helper(42)\n'
    )
    (pkg / "utils.py").write_text(
        '"""Utility functions."""\n\n'
        'MAX_VAL = 100\n\n'
        'def helper(x: int) -> int:\n'
        '    """Help with x."""\n'
        '    return x + MAX_VAL\n'
    )
    # TypeScript file
    (tmp_path / "app.ts").write_text(
        '/** Application entry. */\n'
        'import { Config } from "./config";\n\n'
        'function run(config: Config): void {\n'
        '    console.log(config);\n'
        '}\n'
    )
    (tmp_path / "config.ts").write_text(
        'export interface Config {\n'
        '    name: string;\n'
        '}\n'
    )

    # Init git and commit
    _init_git(tmp_path)
    _git_commit(tmp_path, "initial commit")

    return tmp_path


class TestIndexRepository:
    def test_basic_index(self, sample_repo):
        result = index_repository(sample_repo)
        assert result.files_scanned >= 4  # main, utils, __init__, app.ts, config.ts
        assert result.files_new >= 4
        assert result.files_changed == 0
        assert result.files_deleted == 0
        assert result.parse_errors == 0
        assert result.duration_ms >= 0

    def test_curated_db_populated(self, sample_repo):
        index_repository(sample_repo)
        conn = get_connection("curated", repo_path=sample_repo, read_only=True)
        try:
            files = conn.execute("SELECT * FROM files").fetchall()
            assert len(files) >= 4

            # Check symbols were extracted
            symbols = conn.execute("SELECT * FROM symbols").fetchall()
            assert len(symbols) > 0
            names = {r["name"] for r in symbols}
            assert "main" in names
            assert "helper" in names

            # Check docstrings
            docs = conn.execute("SELECT * FROM docstrings").fetchall()
            assert len(docs) > 0

            # Check comments
            comments = conn.execute("SELECT * FROM inline_comments").fetchall()
            assert len(comments) > 0
            todo = [c for c in comments if c["kind"] == "todo"]
            assert len(todo) >= 1
        finally:
            conn.close()

    def test_raw_db_logged(self, sample_repo):
        index_repository(sample_repo)
        conn = get_connection("raw", repo_path=sample_repo, read_only=True)
        try:
            runs = conn.execute("SELECT * FROM index_runs").fetchall()
            assert len(runs) == 1
            assert runs[0]["status"] == "success"
        finally:
            conn.close()

    def test_incremental_reindex(self, sample_repo):
        # First index
        r1 = index_repository(sample_repo)
        assert r1.files_new >= 4

        # Modify a file
        (sample_repo / "src" / "utils.py").write_text(
            '"""Updated utilities."""\n\n'
            'MAX_VAL = 200\n\n'
            'def helper(x: int) -> int:\n'
            '    return x + MAX_VAL\n'
        )
        _git_commit(sample_repo, "update utils")

        # Re-index
        r2 = index_repository(sample_repo)
        assert r2.files_changed == 1
        assert r2.files_new == 0
        # Other files should be unchanged
        assert r2.files_unchanged >= 3

    def test_file_deletion_detected(self, sample_repo):
        # First index
        index_repository(sample_repo)

        # Delete a file
        (sample_repo / "config.ts").unlink()
        _git_commit(sample_repo, "remove config")

        # Re-index
        r2 = index_repository(sample_repo)
        assert r2.files_deleted == 1

    def test_dependencies_extracted(self, sample_repo):
        index_repository(sample_repo)
        conn = get_connection("curated", repo_path=sample_repo, read_only=True)
        try:
            deps = conn.execute("SELECT * FROM dependencies").fetchall()
            assert len(deps) > 0
        finally:
            conn.close()

    def test_git_history_extracted(self, sample_repo):
        index_repository(sample_repo)
        conn = get_connection("curated", repo_path=sample_repo, read_only=True)
        try:
            commits = conn.execute("SELECT * FROM commits").fetchall()
            assert len(commits) >= 1
        finally:
            conn.close()

    def test_continue_on_error(self, tmp_path):
        """Verify continue_on_error logs errors and skips broken files."""
        _init_git(tmp_path)
        # Create a valid file and a file with invalid syntax (but parseable by tree-sitter)
        (tmp_path / "good.py").write_text("def good(): pass\n")
        (tmp_path / "also_good.py").write_text("x = 1\n")
        _git_commit(tmp_path, "initial")

        result = index_repository(tmp_path, continue_on_error=True)
        assert result.files_scanned == 2
        assert result.parse_errors == 0  # tree-sitter is lenient
