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
