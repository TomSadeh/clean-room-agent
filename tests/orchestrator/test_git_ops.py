"""Tests for orchestrator/git_ops.py GitWorkflow."""

import subprocess

import pytest

from clean_room_agent.orchestrator.git_ops import GitWorkflow, cleanup_task_branches


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

        assert branch == "cra/task/abc123def456extra"
        assert _current_branch(tmp_path) == "cra/task/abc123def456extra"
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

    def test_return_without_create_raises(self, tmp_path):
        """return_to_original_branch before create_task_branch raises RuntimeError (3-P2-4)."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-no-create")
        with pytest.raises(RuntimeError, match="No original branch"):
            gw.return_to_original_branch()


class TestTaskIdValidation:
    def test_empty_task_id_raises(self, tmp_path):
        """Empty task_id raises ValueError (3-P2-1)."""
        with pytest.raises(ValueError, match="Invalid task_id"):
            GitWorkflow(tmp_path, task_id="")

    def test_none_task_id_raises(self, tmp_path):
        """None task_id raises ValueError (3-P2-1)."""
        with pytest.raises(ValueError, match="Invalid task_id"):
            GitWorkflow(tmp_path, task_id=None)


class TestGetCumulativeDiffBeforeBranch:
    def test_diff_before_branch_raises(self, tmp_path):
        """get_cumulative_diff before create_task_branch raises RuntimeError (3-P2-5)."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-no-branch")
        with pytest.raises(RuntimeError, match="before create_task_branch"):
            gw.get_cumulative_diff()


class TestRollbackToCheckpointExplicitSha:
    def test_rollback_to_explicit_sha(self, tmp_path):
        """rollback_to_checkpoint with explicit commit_sha resets to that commit (3-P2-6)."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-explicit-sha")
        gw.create_task_branch()

        # First commit
        (tmp_path / "first.py").write_text("first")
        sha1 = gw.commit_checkpoint("first commit")

        # Second commit
        (tmp_path / "second.py").write_text("second")
        gw.commit_checkpoint("second commit")

        assert (tmp_path / "second.py").exists()

        # Rollback to first commit (not base_ref)
        gw.rollback_to_checkpoint(commit_sha=sha1)

        assert _current_sha(tmp_path) == sha1
        assert (tmp_path / "first.py").exists()
        assert not (tmp_path / "second.py").exists()

    def test_rollback_no_ref_raises(self, tmp_path):
        """rollback_to_checkpoint with no base_ref and no commit_sha raises RuntimeError."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-no-ref")
        with pytest.raises(RuntimeError, match="No base_ref"):
            gw.rollback_to_checkpoint()


class TestCleanUntracked:
    def test_clean_untracked_removes_new_files(self, tmp_path):
        """Untracked files are removed by clean_untracked()."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-clean")
        gw.create_task_branch()

        untracked = tmp_path / "junk.txt"
        untracked.write_text("should be removed")
        assert untracked.exists()

        gw.clean_untracked()
        assert not untracked.exists()

    def test_clean_untracked_preserves_clean_room(self, tmp_path):
        """.clean_room/ directory survives clean_untracked()."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-clean-cr")
        gw.create_task_branch()

        cr_dir = tmp_path / ".clean_room"
        cr_dir.mkdir()
        (cr_dir / "config.toml").write_text("[models]")

        gw.clean_untracked()
        assert (cr_dir / "config.toml").exists()

    def test_return_after_reset_with_conflicting_untracked(self, tmp_path):
        """Full rollback+clean+return sequence works when untracked files conflict."""
        _init_git_repo(tmp_path)
        original = _current_branch(tmp_path)

        gw = GitWorkflow(tmp_path, task_id="task-full-cleanup")
        gw.create_task_branch()

        # Make a committed change and an untracked file
        (tmp_path / "committed.py").write_text("x = 1")
        gw.commit_checkpoint("add committed.py")
        (tmp_path / "untracked.py").write_text("leftover")

        gw.rollback_to_checkpoint()
        gw.clean_untracked()
        gw.return_to_original_branch()

        assert _current_branch(tmp_path) == original
        assert not (tmp_path / "committed.py").exists()
        assert not (tmp_path / "untracked.py").exists()


class TestDeleteTaskBranch:
    def test_delete_task_branch(self, tmp_path):
        """Branch is removed after return_to_original + delete."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-del")
        gw.create_task_branch()

        gw.return_to_original_branch()
        gw.delete_task_branch()

        # Branch should no longer exist
        result = subprocess.run(
            ["git", "-C", str(tmp_path), "branch", "--list", "cra/task/task-del"],
            capture_output=True, text=True,
        )
        assert "cra/task/task-del" not in result.stdout

    def test_delete_while_on_branch_raises(self, tmp_path):
        """Cannot delete the branch you're currently on."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-del-on")
        gw.create_task_branch()

        with pytest.raises(RuntimeError, match="Cannot delete branch"):
            gw.delete_task_branch()


class TestCleanupTaskBranches:
    def test_cleanup_deletes_all_task_branches(self, tmp_path):
        """Bulk cleanup removes all cra/task/* branches (not current)."""
        _init_git_repo(tmp_path)

        # Create several task branches, then return to main
        for tid in ["aaa", "bbb", "ccc"]:
            gw = GitWorkflow(tmp_path, task_id=tid)
            gw.create_task_branch()
            gw.return_to_original_branch()

        deleted = cleanup_task_branches(tmp_path)
        assert sorted(deleted) == ["cra/task/aaa", "cra/task/bbb", "cra/task/ccc"]

        result = subprocess.run(
            ["git", "-C", str(tmp_path), "branch", "--list", "cra/task/*"],
            capture_output=True, text=True,
        )
        assert result.stdout.strip() == ""

    def test_cleanup_skips_current_branch(self, tmp_path):
        """Current branch is preserved during cleanup."""
        _init_git_repo(tmp_path)

        gw1 = GitWorkflow(tmp_path, task_id="keep")
        gw1.create_task_branch()
        gw1.return_to_original_branch()

        gw2 = GitWorkflow(tmp_path, task_id="current")
        gw2.create_task_branch()
        # Stay on cra/task/current

        deleted = cleanup_task_branches(tmp_path)
        assert "cra/task/keep" in deleted
        assert "cra/task/current" not in deleted
        assert _current_branch(tmp_path) == "cra/task/current"


class TestGetHeadSha:
    def test_get_head_sha(self, tmp_path):
        """Returns current SHA matching rev-parse HEAD."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-sha")
        gw.create_task_branch()

        sha = gw.get_head_sha()
        assert sha == _current_sha(tmp_path)
        assert len(sha) == 40

    def test_get_head_sha_advances_after_commit(self, tmp_path):
        """SHA changes after a commit."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-sha-adv")
        gw.create_task_branch()

        sha_before = gw.get_head_sha()
        (tmp_path / "new.py").write_text("x = 1")
        gw.commit_checkpoint("advance")
        sha_after = gw.get_head_sha()

        assert sha_before != sha_after


class TestMergeToOriginal:
    def test_merge_ff_success(self, tmp_path):
        """Linear history merges via fast-forward, lands on original branch."""
        _init_git_repo(tmp_path)
        original = _current_branch(tmp_path)

        gw = GitWorkflow(tmp_path, task_id="task-merge")
        gw.create_task_branch()
        (tmp_path / "feature.py").write_text("feature = True")
        commit_sha = gw.commit_checkpoint("add feature")

        result = gw.merge_to_original()

        assert result is True
        assert _current_branch(tmp_path) == original
        assert _current_sha(tmp_path) == commit_sha
        assert (tmp_path / "feature.py").exists()

    def test_merge_ff_fail_diverged(self, tmp_path):
        """Diverged branches return False, lands on original branch."""
        _init_git_repo(tmp_path)
        original = _current_branch(tmp_path)

        gw = GitWorkflow(tmp_path, task_id="task-merge-div")
        gw.create_task_branch()
        (tmp_path / "task.py").write_text("task = True")
        gw.commit_checkpoint("task commit")

        # Go back to original and create a diverging commit
        subprocess.run(
            ["git", "-C", str(tmp_path), "checkout", original],
            check=True, capture_output=True,
        )
        (tmp_path / "diverge.py").write_text("diverge = True")
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", "."],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", "diverge"],
            check=True, capture_output=True,
        )

        # Go back to task branch for merge attempt
        subprocess.run(
            ["git", "-C", str(tmp_path), "checkout", "cra/task/task-merge-div"],
            check=True, capture_output=True,
        )

        result = gw.merge_to_original()

        assert result is False
        assert _current_branch(tmp_path) == original

    def test_merge_without_create_raises(self, tmp_path):
        """merge_to_original before create_task_branch raises RuntimeError."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-merge-nc")
        with pytest.raises(RuntimeError, match="No original branch"):
            gw.merge_to_original()

    def test_merge_then_delete(self, tmp_path):
        """Merge + delete leaves no task branch."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="task-md")
        gw.create_task_branch()
        (tmp_path / "f.py").write_text("x")
        gw.commit_checkpoint("add f")

        gw.merge_to_original()
        gw.delete_task_branch()

        result = subprocess.run(
            ["git", "-C", str(tmp_path), "branch", "--list", "cra/task/task-md"],
            capture_output=True, text=True,
        )
        assert "cra/task/task-md" not in result.stdout


class TestBranchNameEdgeCases:
    def test_short_task_id(self, tmp_path):
        """Short task_id (< 12 chars) works correctly in branch name (3-P2-7)."""
        _init_git_repo(tmp_path)
        gw = GitWorkflow(tmp_path, task_id="abc")
        branch = gw.create_task_branch()
        assert branch == "cra/task/abc"
        assert _current_branch(tmp_path) == "cra/task/abc"
