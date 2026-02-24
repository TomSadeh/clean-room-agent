"""Tests for extractors/git_extractor.py."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from clean_room_agent.extractors.git_extractor import (
    CommitInfo,
    GitHistory,
    _check_git_available,
    extract_git_history,
    get_remote_url,
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command in the given repo directory."""
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(repo),
    )


def _init_repo(repo: Path) -> None:
    """Initialize a git repo with config suitable for testing."""
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test Author")
    _git(repo, "config", "user.email", "test@example.com")


def _write_file(repo: Path, rel_path: str, content: str) -> None:
    """Write content to a file inside the repo (creating parent dirs)."""
    full = repo / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)


def _commit(repo: Path, message: str) -> None:
    """Stage all changes and commit."""
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)


class TestCheckGitAvailable:
    def test_git_not_on_path(self):
        """When git is not on PATH, _check_git_available raises RuntimeError."""
        import clean_room_agent.extractors.git_extractor as mod
        # Reset the cached state so _check_git_available actually runs
        original = mod._git_available
        mod._git_available = None
        try:
            with patch("clean_room_agent.extractors.git_extractor.subprocess.run",
                        side_effect=FileNotFoundError("git not found")):
                with pytest.raises(RuntimeError, match="git not found on PATH"):
                    _check_git_available()
        finally:
            mod._git_available = original

    def test_git_nonzero_returncode(self):
        """When git --version returns non-zero, _check_git_available raises RuntimeError."""
        import clean_room_agent.extractors.git_extractor as mod
        original = mod._git_available
        mod._git_available = None
        try:
            mock_result = subprocess.CompletedProcess(args=["git", "--version"], returncode=1,
                                                       stdout="", stderr="error")
            with patch("clean_room_agent.extractors.git_extractor.subprocess.run",
                        return_value=mock_result):
                with pytest.raises(RuntimeError, match="git not found on PATH"):
                    _check_git_available()
        finally:
            mod._git_available = original


class TestExtractGitHistoryNonzeroReturncode:
    def test_nonzero_returncode_raises(self):
        """extract_git_history raises RuntimeError when git log fails with non-zero exit."""
        import clean_room_agent.extractors.git_extractor as mod
        original = mod._git_available
        mod._git_available = True  # Skip the git availability check
        try:
            mock_result = subprocess.CompletedProcess(
                args=["git", "log"], returncode=128,
                stdout="", stderr="fatal: not a git repository",
            )
            with patch("clean_room_agent.extractors.git_extractor.subprocess.run",
                        return_value=mock_result):
                with pytest.raises(RuntimeError, match="git log failed"):
                    extract_git_history(Path("/fake/repo"), set(), max_commits=10)
        finally:
            mod._git_available = original


class TestExtractGitHistory:
    def test_basic_commit_extraction(self, tmp_path: Path) -> None:
        """Commits are extracted with correct metadata."""
        _init_repo(tmp_path)
        _write_file(tmp_path, "main.py", "print('hello')\n")
        _commit(tmp_path, "Initial commit")

        _write_file(tmp_path, "main.py", "print('hello world')\n")
        _commit(tmp_path, "Update greeting")

        file_index = {"main.py"}
        history = extract_git_history(tmp_path, file_index, max_commits=100)

        assert isinstance(history, GitHistory)
        assert len(history.commits) == 2
        # Most recent commit comes first in git log
        assert history.commits[0].message == "Update greeting"
        assert history.commits[1].message == "Initial commit"
        # All commits have required fields
        for c in history.commits:
            assert c.hash and len(c.hash) == 40
            assert c.author == "Test Author"
            assert c.timestamp  # ISO format string

    def test_file_commit_map(self, tmp_path: Path) -> None:
        """file_commit_map correctly associates files with their commits."""
        _init_repo(tmp_path)
        _write_file(tmp_path, "a.py", "# a\n")
        _write_file(tmp_path, "b.py", "# b\n")
        _commit(tmp_path, "Add a and b")

        _write_file(tmp_path, "a.py", "# a updated\n")
        _commit(tmp_path, "Update a")

        _write_file(tmp_path, "b.py", "# b updated\n")
        _commit(tmp_path, "Update b")

        file_index = {"a.py", "b.py"}
        history = extract_git_history(tmp_path, file_index, max_commits=100)

        # a.py was changed in "Add a and b" and "Update a" => 2 commits
        assert len(history.file_commit_map["a.py"]) == 2
        # b.py was changed in "Add a and b" and "Update b" => 2 commits
        assert len(history.file_commit_map["b.py"]) == 2

    def test_file_commit_map_filters_by_file_index(self, tmp_path: Path) -> None:
        """Only files in file_index appear in file_commit_map."""
        _init_repo(tmp_path)
        _write_file(tmp_path, "tracked.py", "# tracked\n")
        _write_file(tmp_path, "untracked.py", "# untracked\n")
        _commit(tmp_path, "Add files")

        file_index = {"tracked.py"}
        history = extract_git_history(tmp_path, file_index, max_commits=100)

        assert "tracked.py" in history.file_commit_map
        assert "untracked.py" not in history.file_commit_map

    def test_co_change_counts(self, tmp_path: Path) -> None:
        """Co-change pairs are counted correctly with min count >= 2."""
        _init_repo(tmp_path)
        _write_file(tmp_path, "a.py", "# a v1\n")
        _write_file(tmp_path, "b.py", "# b v1\n")
        _write_file(tmp_path, "c.py", "# c v1\n")
        _commit(tmp_path, "Commit 1: a, b, c")

        _write_file(tmp_path, "a.py", "# a v2\n")
        _write_file(tmp_path, "b.py", "# b v2\n")
        _commit(tmp_path, "Commit 2: a, b")

        _write_file(tmp_path, "a.py", "# a v3\n")
        _write_file(tmp_path, "b.py", "# b v3\n")
        _commit(tmp_path, "Commit 3: a, b")

        _write_file(tmp_path, "a.py", "# a v4\n")
        _write_file(tmp_path, "c.py", "# c v2\n")
        _commit(tmp_path, "Commit 4: a, c")

        file_index = {"a.py", "b.py", "c.py"}
        history = extract_git_history(tmp_path, file_index, max_commits=100)

        # (a.py, b.py) changed together in commits 1, 2, 3 => count 3
        assert history.co_change_counts[("a.py", "b.py")] == 3
        # (a.py, c.py) changed together in commits 1 and 4 => count 2
        assert history.co_change_counts[("a.py", "c.py")] == 2
        # (b.py, c.py) changed together only in commit 1 => count 1 => filtered out
        assert ("b.py", "c.py") not in history.co_change_counts

    def test_co_change_pair_ordering(self, tmp_path: Path) -> None:
        """Co-change pairs always have a < b alphabetically."""
        _init_repo(tmp_path)
        _write_file(tmp_path, "zebra.py", "# z v1\n")
        _write_file(tmp_path, "alpha.py", "# a v1\n")
        _commit(tmp_path, "Commit 1")

        _write_file(tmp_path, "zebra.py", "# z v2\n")
        _write_file(tmp_path, "alpha.py", "# a v2\n")
        _commit(tmp_path, "Commit 2")

        file_index = {"zebra.py", "alpha.py"}
        history = extract_git_history(tmp_path, file_index, max_commits=100)

        assert ("alpha.py", "zebra.py") in history.co_change_counts
        assert ("zebra.py", "alpha.py") not in history.co_change_counts

    def test_co_change_skips_large_commits(self, tmp_path: Path) -> None:
        """Commits with >50 tracked files are excluded from co-change calculation."""
        _init_repo(tmp_path)

        # Create a commit that touches 51 tracked files
        file_names = [f"file_{i:03d}.py" for i in range(51)]
        for name in file_names:
            _write_file(tmp_path, name, f"# {name} v1\n")
        _commit(tmp_path, "Bulk commit with 51 files")

        # Create another commit with the same files
        for name in file_names:
            _write_file(tmp_path, name, f"# {name} v2\n")
        _commit(tmp_path, "Another bulk commit with 51 files")

        file_index = set(file_names)
        history = extract_git_history(tmp_path, file_index, max_commits=100)

        # Both commits have >50 tracked files, so no co-change pairs should exist
        assert len(history.co_change_counts) == 0

    def test_co_change_includes_at_boundary(self, tmp_path: Path) -> None:
        """Commits with exactly 50 tracked files ARE included in co-change."""
        _init_repo(tmp_path)

        file_names = [f"file_{i:03d}.py" for i in range(50)]
        for name in file_names:
            _write_file(tmp_path, name, f"# {name} v1\n")
        _commit(tmp_path, "Commit with exactly 50 files")

        for name in file_names:
            _write_file(tmp_path, name, f"# {name} v2\n")
        _commit(tmp_path, "Another commit with exactly 50 files")

        file_index = set(file_names)
        history = extract_git_history(tmp_path, file_index, max_commits=100)

        # 50 files => within limit => co-change pairs should be generated
        # C(50, 2) = 1225 pairs, each appearing twice => all have count 2
        assert len(history.co_change_counts) == 1225

    def test_insertions_and_deletions(self, tmp_path: Path) -> None:
        """Insertion and deletion counts are parsed from numstat output."""
        _init_repo(tmp_path)
        _write_file(tmp_path, "main.py", "line1\nline2\nline3\n")
        _commit(tmp_path, "Add 3 lines")

        _write_file(tmp_path, "main.py", "line1\nmodified\n")
        _commit(tmp_path, "Modify file")

        history = extract_git_history(tmp_path, {"main.py"}, max_commits=100)

        # Most recent commit: replaced 2 lines with 1
        latest = history.commits[0]
        assert latest.message == "Modify file"
        assert latest.insertions > 0 or latest.deletions > 0
        assert latest.files_changed == 1

    def test_max_commits_limits_output(self, tmp_path: Path) -> None:
        """max_commits parameter limits the number of commits returned."""
        _init_repo(tmp_path)
        for i in range(5):
            _write_file(tmp_path, "main.py", f"# version {i}\n")
            _commit(tmp_path, f"Commit {i}")

        history = extract_git_history(tmp_path, {"main.py"}, max_commits=3)
        assert len(history.commits) == 3

    def test_empty_repo_no_commits(self, tmp_path: Path) -> None:
        """An empty repo (no commits) returns empty history."""
        _init_repo(tmp_path)

        history = extract_git_history(tmp_path, set(), max_commits=100)

        assert history.commits == []
        assert history.file_commit_map == {}
        assert history.co_change_counts == {}

    def test_subdirectory_files(self, tmp_path: Path) -> None:
        """Files in subdirectories are tracked correctly with forward-slash paths."""
        _init_repo(tmp_path)
        _write_file(tmp_path, "src/pkg/module.py", "# module\n")
        _write_file(tmp_path, "src/pkg/utils.py", "# utils\n")
        _commit(tmp_path, "Add nested files")

        _write_file(tmp_path, "src/pkg/module.py", "# module updated\n")
        _write_file(tmp_path, "src/pkg/utils.py", "# utils updated\n")
        _commit(tmp_path, "Update nested files")

        file_index = {"src/pkg/module.py", "src/pkg/utils.py"}
        history = extract_git_history(tmp_path, file_index, max_commits=100)

        assert "src/pkg/module.py" in history.file_commit_map
        assert "src/pkg/utils.py" in history.file_commit_map
        assert len(history.file_commit_map["src/pkg/module.py"]) == 2


class TestGetRemoteUrl:
    def test_remote_url_detected(self, tmp_path: Path) -> None:
        """Remote URL is returned when a remote is configured."""
        _init_repo(tmp_path)
        _write_file(tmp_path, "readme.txt", "hello\n")
        _commit(tmp_path, "Init")
        _git(tmp_path, "remote", "add", "origin", "https://github.com/test/repo.git")

        url = get_remote_url(tmp_path)
        assert url == "https://github.com/test/repo.git"

    def test_no_remote_returns_none(self, tmp_path: Path) -> None:
        """Returns None when no remote is configured."""
        _init_repo(tmp_path)
        _write_file(tmp_path, "readme.txt", "hello\n")
        _commit(tmp_path, "Init")

        url = get_remote_url(tmp_path)
        assert url is None

    def test_remote_url_via_extract(self, tmp_path: Path) -> None:
        """extract_git_history includes remote_url in the returned GitHistory."""
        _init_repo(tmp_path)
        _write_file(tmp_path, "main.py", "# main\n")
        _commit(tmp_path, "Init")
        _git(tmp_path, "remote", "add", "origin", "git@github.com:user/project.git")

        history = extract_git_history(tmp_path, {"main.py"}, max_commits=100)
        assert history.remote_url == "git@github.com:user/project.git"
