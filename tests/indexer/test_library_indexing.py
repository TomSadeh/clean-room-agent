"""Integration tests for library indexing via orchestrator.index_libraries."""

import subprocess

import pytest

from clean_room_agent.db.connection import get_connection
from clean_room_agent.indexer.orchestrator import index_libraries, index_repository


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
def repo_with_library(tmp_path):
    """Create a repo with a project file plus a fake 'library' directory.

    The library dir lives OUTSIDE the repo to avoid index_repository treating
    it as project files (the real use case: libraries live in site-packages).
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    # Project file
    src = repo_dir / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text(
        '"""Main module."""\n\n'
        'def main():\n'
        '    """Entry point."""\n'
        '    return 42\n'
    )

    # Fake library directory OUTSIDE the repo
    lib_dir = tmp_path / "fake_lib"
    lib_dir.mkdir()
    (lib_dir / "__init__.py").write_text('"""Fake library."""\n')
    (lib_dir / "core.py").write_text(
        '"""Core module of fake library."""\n\n'
        'class Widget:\n'
        '    """A widget class."""\n\n'
        '    def render(self):\n'
        '        """Render the widget."""\n'
        '        return "<widget/>"\n\n\n'
        'def create_widget(name: str) -> Widget:\n'
        '    """Factory function for widgets."""\n'
        '    return Widget()\n'
    )

    # Init git and commit so index_repository works
    _init_git(repo_dir)
    _git_commit(repo_dir, "initial commit")

    return repo_dir, lib_dir


class TestLibraryIndexing:
    def test_library_file_source_in_db(self, repo_with_library):
        """After library indexing, files in DB have file_source='library'."""
        repo_path, lib_dir = repo_with_library

        # First, index the project (required for repo_id to exist)
        index_repository(repo_path)

        # Then index libraries via explicit path config
        config = {
            "library_sources": [],
            "library_paths": [
                {"name": "fake_lib", "path": str(lib_dir)},
            ],
        }
        result = index_libraries(repo_path, indexer_config=config)
        assert result.files_scanned >= 2  # __init__.py + core.py

        # Verify file_source in DB
        conn = get_connection("curated", repo_path=repo_path, read_only=True)
        try:
            rows = conn.execute(
                "SELECT path, file_source FROM files WHERE file_source = 'library'"
            ).fetchall()
            assert len(rows) >= 2
            for row in rows:
                assert row["file_source"] == "library"
                assert row["path"].startswith("fake_lib/")
        finally:
            conn.close()

    def test_library_no_git_history(self, repo_with_library):
        """Library indexing does not extract git history (no commits for library files)."""
        repo_path, lib_dir = repo_with_library

        index_repository(repo_path)

        config = {
            "library_sources": [],
            "library_paths": [
                {"name": "fake_lib", "path": str(lib_dir)},
            ],
        }
        index_libraries(repo_path, indexer_config=config)

        conn = get_connection("curated", repo_path=repo_path, read_only=True)
        try:
            # Get library file IDs
            lib_files = conn.execute(
                "SELECT id FROM files WHERE file_source = 'library'"
            ).fetchall()
            lib_file_ids = {r["id"] for r in lib_files}

            # Check no file_commits rows for library files
            for fid in lib_file_ids:
                fc = conn.execute(
                    "SELECT * FROM file_commits WHERE file_id = ?", (fid,)
                ).fetchall()
                assert len(fc) == 0, f"Library file {fid} should have no commit history"
        finally:
            conn.close()

    def test_library_symbols_extracted(self, repo_with_library):
        """Symbols and docstrings are extracted for library files."""
        repo_path, lib_dir = repo_with_library

        index_repository(repo_path)

        config = {
            "library_sources": [],
            "library_paths": [
                {"name": "fake_lib", "path": str(lib_dir)},
            ],
        }
        index_libraries(repo_path, indexer_config=config)

        conn = get_connection("curated", repo_path=repo_path, read_only=True)
        try:
            # Get the core.py library file
            core_file = conn.execute(
                "SELECT id FROM files WHERE path = 'fake_lib/core.py'"
            ).fetchone()
            assert core_file is not None

            # Check symbols extracted
            symbols = conn.execute(
                "SELECT name, kind FROM symbols WHERE file_id = ?",
                (core_file["id"],),
            ).fetchall()
            sym_names = {r["name"] for r in symbols}
            assert "Widget" in sym_names
            assert "create_widget" in sym_names

            # Check docstrings extracted
            docstrings = conn.execute(
                "SELECT content FROM docstrings WHERE file_id = ?",
                (core_file["id"],),
            ).fetchall()
            assert len(docstrings) > 0
            doc_texts = {r["content"] for r in docstrings}
            assert any("widget" in d.lower() for d in doc_texts)
        finally:
            conn.close()

    def test_library_incremental(self, repo_with_library):
        """Re-indexing unchanged library files is a no-op (files_unchanged count)."""
        repo_path, lib_dir = repo_with_library

        index_repository(repo_path)

        config = {
            "library_sources": [],
            "library_paths": [
                {"name": "fake_lib", "path": str(lib_dir)},
            ],
        }

        # First library index
        r1 = index_libraries(repo_path, indexer_config=config)
        assert r1.files_new >= 2

        # Second library index without changes
        r2 = index_libraries(repo_path, indexer_config=config)
        assert r2.files_unchanged >= 2
        assert r2.files_new == 0
