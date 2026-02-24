"""Git history extraction: commits, file-commit associations, co-change pairs."""

import subprocess
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


COMMIT_SEP = "---CRA_COMMIT_8f14e45f---"

# Commits touching more than this many tracked files are excluded from
# co-change calculation (bulk renames, reformats, etc.).
CO_CHANGE_MAX_FILES = 50

# Co-change pairs must appear in at least this many commits to be kept.
CO_CHANGE_MIN_COUNT = 2


@dataclass
class CommitInfo:
    hash: str
    author: str | None
    message: str | None
    timestamp: str  # ISO format
    files_changed: int
    insertions: int
    deletions: int
    changed_files: list[str]  # relative paths


@dataclass
class GitHistory:
    commits: list[CommitInfo]
    file_commit_map: dict[str, list[str]]  # file_path -> list of commit hashes
    co_change_counts: dict[tuple[str, str], int]  # (file_a, file_b) -> count, a < b
    remote_url: str | None


_git_available: bool | None = None


def _check_git_available() -> None:
    """Raise RuntimeError if git is not on PATH. Caches result after first success."""
    global _git_available
    if _git_available:
        return
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "git not found on PATH. Install git to use indexing features."
        ) from e
    if result.returncode != 0:
        raise RuntimeError(
            "git not found on PATH. Install git to use indexing features."
        )
    _git_available = True


def get_remote_url(repo_path: Path) -> str | None:
    """Return the remote.origin.url for the repo, or None if no remote is configured.

    Raises RuntimeError if git is not available (delegates to _check_git_available).
    """
    _check_git_available()
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(repo_path),
    )
    if result.returncode != 0:
        return None
    url = result.stdout.strip()
    return url if url else None


def _parse_git_log(raw_output: str) -> list[CommitInfo]:
    """Parse the output of git log with --pretty and --numstat into CommitInfo list."""
    commits: list[CommitInfo] = []

    # Split on the separator marker. The first chunk before the first COMMIT_SEP
    # is empty or whitespace, so skip it.
    chunks = raw_output.split(COMMIT_SEP)

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        lines = chunk.split("\n")
        # Expected header: hash, author, subject, ISO timestamp (4 lines)
        if len(lines) < 4:
            continue

        commit_hash = lines[0].strip()
        author = lines[1].strip() or None
        message = lines[2].strip() or None
        timestamp = lines[3].strip()

        # Remaining lines are --numstat output (possibly with blank lines)
        changed_files: list[str] = []
        insertions = 0
        deletions = 0

        for line in lines[4:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            ins_str, del_str, file_path = parts
            # Binary files show "-" for insertions/deletions
            ins = int(ins_str) if ins_str != "-" else 0
            del_ = int(del_str) if del_str != "-" else 0
            insertions += ins
            deletions += del_
            # Normalize path separators to forward slashes
            changed_files.append(file_path.replace("\\", "/"))

        commits.append(
            CommitInfo(
                hash=commit_hash,
                author=author,
                message=message,
                timestamp=timestamp,
                files_changed=len(changed_files),
                insertions=insertions,
                deletions=deletions,
                changed_files=changed_files,
            )
        )

    return commits


def extract_git_history(
    repo_path: Path,
    file_index: set[str],
    max_commits: int = 500,
    remote_url: str | None = None,
) -> GitHistory:
    """Extract git history, file-commit associations, and co-change pairs.

    Args:
        repo_path: Path to the repository root.
        file_index: Set of relative file paths currently tracked in the repo.
            Only these files are included in file_commit_map and co-change analysis.
        max_commits: Maximum number of commits to examine.
        remote_url: Pre-fetched remote URL (avoids redundant subprocess call).
            If None, fetched internally.

    Returns:
        GitHistory with commits, file associations, co-change pairs, and remote URL.

    Raises:
        RuntimeError: If git is not found on PATH or if the git log command fails.
    """
    _check_git_available()

    # Normalize file_index paths to forward slashes for consistent comparison
    normalized_index = {p.replace("\\", "/") for p in file_index}

    format_str = f"{COMMIT_SEP}%n%H%n%aN%n%s%n%aI"
    result = subprocess.run(
        [
            "git", "log",
            f"--pretty=format:{format_str}",
            "--numstat",
            f"--max-count={max_commits}",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(repo_path),
    )
    if result.returncode != 0:
        # Empty repo (no commits yet) is not an error — return empty history
        if "does not have any commits" in result.stderr:
            return GitHistory(
                commits=[],
                file_commit_map={},
                co_change_counts={},
                remote_url=remote_url if remote_url is not None else get_remote_url(repo_path),
            )
        raise RuntimeError(
            f"git log failed (exit {result.returncode}): {result.stderr.strip()}"
        )

    commits = _parse_git_log(result.stdout)

    # Build file_commit_map: only files present in file_index
    file_commit_map: dict[str, list[str]] = {}
    for commit in commits:
        for file_path in commit.changed_files:
            if file_path in normalized_index:
                file_commit_map.setdefault(file_path, []).append(commit.hash)

    # Build co_change_counts: pairs of tracked files changed in the same commit
    co_change_counts: dict[tuple[str, str], int] = {}
    for commit in commits:
        tracked_files = sorted(
            f for f in commit.changed_files if f in normalized_index
        )
        if len(tracked_files) > CO_CHANGE_MAX_FILES:
            continue
        if len(tracked_files) < 2:
            continue
        for a, b in combinations(tracked_files, 2):
            # a < b guaranteed by sorted() + combinations()
            pair = (a, b)
            co_change_counts[pair] = co_change_counts.get(pair, 0) + 1

    # Filter out pairs below the minimum count threshold
    co_change_counts = {
        pair: count
        for pair, count in co_change_counts.items()
        if count >= CO_CHANGE_MIN_COUNT
    }

    if remote_url is None:
        remote_url = get_remote_url(repo_path)

    return GitHistory(
        commits=commits,
        file_commit_map=file_commit_map,
        co_change_counts=co_change_counts,
        remote_url=remote_url,
    )
