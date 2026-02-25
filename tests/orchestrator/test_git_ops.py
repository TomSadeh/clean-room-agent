"""Tests for orchestrator/git_ops.py GitWorkflow."""

import subprocess

import pytest

from clean_room_agent.orchestrator.git_ops import GitWorkflow


def _init_git_repo(path):
    """Initialize a git repo with an initial commit."""
    subprocess.run(
        ["git", "init", str(path)], check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True, capture_output=True,
    )
    (path / "initial.txt").write_text("hello")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        check=True, capture_output=True,
    )


def _current_branch(path):
    """Get the current branch name."""
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _current_sha(path):
    """Get the current HEAD SHA."""
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _is_file_tracked(path, filename):
    """Check if a file is committed (tracked and clean)."""
    result = subprocess.run(
        ["git", "-C", str(path), "ls-files", filename],
        capture_output=True, text=True, check=True,
    )
    return filename in result.stdout


class TestCreateTaskBranch:
    def test_create_task_branch(self, tmp_path):
        """Creates a branch with the expected name, HEAD moves to it."""
        _init_git_repo(tmp_path)
        original_branch = _current_branch(tmp_path)

        gw = GitWorkflow(tmp_path, task_id="abc123def456extra")
        branch = gw.create_task_branch()

        assert branch == "cra/task/abc123def456"
        assert _current_branch(tmp_path) == "cra/task/abc123def456"
        assert gw.base_ref is not None
        assert gw.original_branch == original_branch


class TestDirtyRepoRejected:
    def test_dirty_repo_rejected(self, tmp_path):
        """Uncommitted changes raise RuntimeError."""
        _init_git_repo(tmp_path)

        # Create an uncommitted modification
        (tmp_path / "dirty.txt").write_text("uncommitted")
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", "dirty.txt"],
            check=True, capture_output=True,
        )

        gw = GitWorkflow(tmp_path, task_id="task-dirty")
        with pytest.raises(RuntimeError, match="dirty"):
            gw.create_task_branch()


class TestCommitCheckpoint:
    def test_commit_checkpoint(self, tmp_path):
        """After writing a file, commit_checkpoint returns a SHA, file is committed."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-commit")
        gw.create_task_branch()

        # Write a new file
        (tmp_path / "new_file.py").write_text("print('hello')")

        sha = gw.commit_checkpoint("add new file")

        assert len(sha) == 40  # full SHA
        assert _is_file_tracked(tmp_path, "new_file.py")
        # HEAD should be at the new commit
        assert _current_sha(tmp_path) == sha


class TestCumulativeDiff:
    def test_cumulative_diff(self, tmp_path):
        """After commit, get_cumulative_diff shows the changes from base ref."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-diff")
        gw.create_task_branch()

        (tmp_path / "added.py").write_text("x = 42\n")
        gw.commit_checkpoint("add file")

        diff = gw.get_cumulative_diff()

        assert "added.py" in diff
        assert "+x = 42" in diff

    def test_diff_capping(self, tmp_path):
        """get_cumulative_diff with max_chars truncates oldest content."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-cap")
        gw.create_task_branch()

        # Create a large diff with multiple files
        for i in range(20):
            (tmp_path / f"file_{i:03d}.py").write_text(
                f"# File {i}\n" + f"line = {i}\n" * 50
            )
        gw.commit_checkpoint("add many files")

        full_diff = gw.get_cumulative_diff()
        capped_diff = gw.get_cumulative_diff(max_chars=500)

        # Full diff should be larger
        assert len(full_diff) > 500
        # Capped diff should include the truncation notice
        assert "[earlier changes truncated]" in capped_diff
        # Capped diff should be smaller than or close to the limit
        # (it may go slightly over due to the header and block alignment)
        assert len(capped_diff) <= 600


class TestRollbackPart:
    def test_rollback_part(self, tmp_path):
        """After writing a file without committing, rollback_part discards it."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-rollback")
        gw.create_task_branch()

        # Write file without committing
        new_file = tmp_path / "uncommitted.py"
        new_file.write_text("should be discarded")

        gw.rollback_part()

        assert not new_file.exists()


class TestRollbackToCheckpoint:
    def test_rollback_to_checkpoint(self, tmp_path):
        """After multiple commits, rollback_to_checkpoint goes back to base."""
        _init_git_repo(tmp_path)
        base_sha = _current_sha(tmp_path)

        gw = GitWorkflow(tmp_path, task_id="task-reset")
        gw.create_task_branch()

        # Make two commits
        (tmp_path / "first.py").write_text("first")
        gw.commit_checkpoint("first commit")

        (tmp_path / "second.py").write_text("second")
        gw.commit_checkpoint("second commit")

        assert (tmp_path / "first.py").exists()
        assert (tmp_path / "second.py").exists()

        # Rollback to base (no commit_sha = use base_ref)
        gw.rollback_to_checkpoint()

        assert _current_sha(tmp_path) == base_sha
        assert not (tmp_path / "first.py").exists()
        assert not (tmp_path / "second.py").exists()


class TestReturnToOriginalBranch:
    def test_return_to_original_branch(self, tmp_path):
        """After creating task branch, return_to_original gets back."""
        _init_git_repo(tmp_path)
        original = _current_branch(tmp_path)

        gw = GitWorkflow(tmp_path, task_id="task-return")
        gw.create_task_branch()
        assert _current_branch(tmp_path) != original

        gw.return_to_original_branch()
        assert _current_branch(tmp_path) == original
