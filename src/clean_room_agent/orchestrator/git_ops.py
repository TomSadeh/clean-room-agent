"""Git workflow: branch-per-task lifecycle for the orchestrator."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def truncate_diff(diff: str, max_chars: int) -> str:
    """Truncate a diff to max_chars, aligning to a diff block boundary.

    Preserves the most recent changes (tail of the diff).
    """
    if len(diff) <= max_chars:
        return diff
    truncated = diff[-max_chars:]
    # Align to the next complete diff block boundary
    first_block = truncated.find("\ndiff --git ")
    if first_block >= 0:
        truncated = truncated[first_block + 1:]
    return f"[earlier changes truncated]\n{truncated}"


class GitWorkflow:
    """Manages the task branch lifecycle: create, commit, diff, rollback."""

    def __init__(self, repo_path: Path, task_id: str):
        if not task_id or not isinstance(task_id, str):
            raise ValueError(f"Invalid task_id: {task_id!r}")
        self._repo_path = repo_path
        self._task_id = task_id
        self._branch_name = f"cra/task/{task_id}"
        self._base_ref: str | None = None
        self._original_branch: str | None = None

    @property
    def branch_name(self) -> str:
        return self._branch_name

    @property
    def base_ref(self) -> str | None:
        return self._base_ref

    @property
    def original_branch(self) -> str | None:
        return self._original_branch

    def create_task_branch(self) -> str:
        """Create and checkout a task branch. Returns branch name.

        Raises RuntimeError if the working tree is dirty.
        """
        # Check repo is clean
        status = self._run_git("status", "--porcelain")
        if status.stdout.strip():
            raise RuntimeError(
                "Working tree is dirty. Commit or stash changes before running cra solve."
            )

        # Save original branch
        result = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        self._original_branch = result.stdout.strip()

        if self._original_branch == "HEAD":
            raise RuntimeError(
                "Cannot create task branch from detached HEAD — checkout a branch first"
            )

        # Record base SHA
        result = self._run_git("rev-parse", "HEAD")
        self._base_ref = result.stdout.strip()

        # Create and checkout task branch
        self._run_git("checkout", "-b", self._branch_name)
        logger.info("Created task branch %s from %s", self._branch_name, self._base_ref[:12])

        return self._branch_name

    def commit_checkpoint(self, message: str, files: list[str] | None = None) -> str:
        """Stage and commit. Returns the commit SHA.

        Args:
            message: Commit message.
            files: Specific files to stage. If None, stages all modified files.
        """
        if files:
            for f in files:
                self._run_git("add", f)
        else:
            self._run_git("add", "-A")

        # Check if there's anything staged (avoids polluting history with empty commits)
        status = self._run_git("diff", "--cached", "--name-only")
        if not status.stdout.strip():
            logger.debug("No staged changes to commit for checkpoint: %s", message)
            result = self._run_git("rev-parse", "HEAD")
            return result.stdout.strip()

        self._run_git("commit", "-m", message)

        result = self._run_git("rev-parse", "HEAD")
        sha = result.stdout.strip()
        logger.debug("Committed checkpoint %s: %s", sha[:12], message)
        return sha

    def get_cumulative_diff(self, max_chars: int | None = None) -> str:
        """Get unified diff from base ref to HEAD.

        Args:
            max_chars: Maximum characters. Truncates oldest hunks first to preserve
                recent changes.
        """
        if self._base_ref is None:
            raise RuntimeError("get_cumulative_diff called before create_task_branch")

        result = self._run_git("diff", f"{self._base_ref}..HEAD")
        diff = result.stdout

        if max_chars is not None and len(diff) > max_chars:
            diff = truncate_diff(diff, max_chars)

        return diff

    def rollback_part(self) -> None:
        """Discard uncommitted changes within the current part."""
        self._run_git("checkout", "--", ".")
        # Clean untracked files, excluding .clean_room/ data
        self._run_git("clean", "-fd", "--exclude=.clean_room")
        logger.debug("Rolled back uncommitted changes")

    def rollback_to_checkpoint(self, commit_sha: str | None = None) -> None:
        """Hard reset to a specific commit or the base ref."""
        target = commit_sha or self._base_ref
        if target is None:
            raise RuntimeError("No base_ref or commit_sha to rollback to")
        self._run_git("reset", "--hard", target)
        logger.info("Rolled back to %s", target[:12])

    def clean_untracked(self) -> None:
        """Remove untracked files and directories, preserving .clean_room/."""
        self._run_git("clean", "-fd", "--exclude=.clean_room")

    def return_to_original_branch(self) -> None:
        """Checkout the original branch.

        Raises RuntimeError if no original branch was recorded.
        """
        if self._original_branch is None:
            raise RuntimeError(
                "No original branch recorded — call create_task_branch() first"
            )
        self._run_git("checkout", self._original_branch)
        logger.info("Returned to original branch %s", self._original_branch)

    def delete_task_branch(self) -> None:
        """Delete the task branch. Must not be on the branch when called."""
        current = self._run_git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        if current == self._branch_name:
            raise RuntimeError(
                f"Cannot delete branch {self._branch_name} while checked out on it"
            )
        # -D (force): intentional — task branches may contain unmerged failed work
        self._run_git("branch", "-D", self._branch_name)
        logger.info("Deleted task branch %s", self._branch_name)

    def get_head_sha(self) -> str:
        """Return the current HEAD SHA."""
        return self._run_git("rev-parse", "HEAD").stdout.strip()

    def merge_to_original(self) -> bool:
        """Fast-forward merge the task branch into the original branch.

        Returns True on success, False if ff-only fails (diverged history).
        Raises RuntimeError if no original branch was recorded.
        """
        if self._original_branch is None:
            raise RuntimeError(
                "No original branch recorded — call create_task_branch() first"
            )
        self._run_git("checkout", self._original_branch)
        merge_result = subprocess.run(
            ["git", "merge", "--ff-only", self._branch_name],
            cwd=str(self._repo_path),
            capture_output=True,
            text=True,
        )
        if merge_result.returncode == 0:
            logger.info(
                "Merged %s into %s (fast-forward)",
                self._branch_name, self._original_branch,
            )
            return True
        else:
            logger.warning(
                "Fast-forward merge of %s into %s failed — branches diverged. "
                "Task branch preserved for manual inspection.",
                self._branch_name, self._original_branch,
            )
            return False

    def _run_git(self, *args: str) -> subprocess.CompletedProcess:
        """Run a git command with repo_path as cwd."""
        cmd = ["git"] + list(args)
        result = subprocess.run(
            cmd,
            cwd=str(self._repo_path),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        return result


def cleanup_task_branches(repo_path: Path) -> list[str]:
    """Delete all cra/task/* branches except the currently checked-out one.

    Returns the list of deleted branch names. Logs warnings on individual failures.
    """
    result = subprocess.run(
        ["git", "branch", "--list", "cra/task/*"],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning("Failed to list task branches: %s", result.stderr.strip())
        return []

    # Get current branch to skip
    current = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
    )
    current_branch = current.stdout.strip() if current.returncode == 0 else ""

    deleted = []
    for line in result.stdout.splitlines():
        branch = line.strip().removeprefix("*").strip()
        if not branch or branch == current_branch:
            continue
        try:
            subprocess.run(
                ["git", "branch", "-D", branch],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=True,
            )
            deleted.append(branch)
        except subprocess.CalledProcessError as e:
            logger.warning(
                "Failed to delete branch %s: %s",
                branch, e.stderr.strip() if e.stderr else str(e),
            )

    if deleted:
        logger.info("Cleaned up %d task branch(es): %s", len(deleted), deleted)
    return deleted
