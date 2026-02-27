"""Patch application: validate, apply, and rollback search/replace edits."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from clean_room_agent.execute.dataclasses import PatchEdit, PatchResult

logger = logging.getLogger(__name__)


def _check_path_traversal(file_path: str, repo_path: Path) -> None:
    """Raise ValueError if file_path escapes repo_path."""
    resolved = (repo_path / file_path).resolve()
    repo_resolved = repo_path.resolve()
    # Ensure the resolved path is under repo_path
    try:
        resolved.relative_to(repo_resolved)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: {file_path!r} resolves to "
            f"{resolved} which is outside {repo_resolved}"
        )


def _validate_and_simulate(
    edits: list[PatchEdit],
    repo_path: Path,
    *,
    track_originals: bool = False,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Simulate sequential edit application and collect errors.

    Args:
        track_originals: If True, record original file contents for rollback.

    Returns:
        (simulated_contents, original_contents, errors)
    """
    simulated: dict[str, str] = {}
    originals: dict[str, str] = {}
    errors: list[str] = []

    for i, edit in enumerate(edits):
        _check_path_traversal(edit.file_path, repo_path)
        file_path = repo_path / edit.file_path
        if not file_path.is_file():
            errors.append(f"Edit {i}: file does not exist: {edit.file_path}")
            continue

        if edit.file_path not in simulated:
            content = file_path.read_text(encoding="utf-8")
            simulated[edit.file_path] = content
            if track_originals:
                originals[edit.file_path] = content

        content = simulated[edit.file_path]
        count = content.count(edit.search)
        if count == 0:
            errors.append(f"Edit {i}: search string not found in {edit.file_path}")
        elif count > 1:
            errors.append(
                f"Edit {i}: search string found {count} times in {edit.file_path} "
                f"(must be exactly once)"
            )
        else:
            simulated[edit.file_path] = content.replace(edit.search, edit.replacement, 1)

    return simulated, originals, errors


def validate_edits(edits: list[PatchEdit], repo_path: Path) -> list[str]:
    """Validate edits against the current filesystem state.

    Simulates sequential application: each edit validates against the file
    content after prior edits to the same file have been applied in memory.

    Returns list of error strings (empty = all valid).
    """
    _, _, errors = _validate_and_simulate(edits, repo_path)
    return errors


def apply_edits(edits: list[PatchEdit], repo_path: Path) -> PatchResult:
    """Read files once, validate and apply edits atomically.

    Reads each file exactly once (fixing TOCTOU between validate and apply).
    Saves original contents for rollback. On any failure during application,
    rolls back all changes and returns a failure result.
    """
    simulated, original_contents, errors = _validate_and_simulate(
        edits, repo_path, track_originals=True,
    )

    if errors:
        return PatchResult(success=False, error_info="; ".join(errors))

    # Phase 2: Write modified files atomically
    result = PatchResult(
        success=True,
        files_modified=sorted(original_contents.keys()),
        original_contents=original_contents,
    )

    try:
        for rel_path, content in simulated.items():
            file_path = repo_path / rel_path
            _atomic_write(file_path, content)
    except Exception as e:
        rollback_edits(result, repo_path)
        raise RuntimeError(
            f"Failed to apply edits (rolled back): {e}"
        ) from e

    return result


def rollback_edits(patch_result: PatchResult, repo_path: Path) -> None:
    """Restore files from original_contents.

    Attempts all files before raising. If any rollback fails, raises
    RuntimeError with all failure details after attempting the rest.
    """
    rollback_errors: list[str] = []
    for rel_path, content in patch_result.original_contents.items():
        file_path = repo_path / rel_path
        try:
            file_path.write_text(content, encoding="utf-8")
        except Exception as e:
            logger.error("Failed to rollback %s: %s", rel_path, e)
            rollback_errors.append(f"{rel_path}: {e}")
    if rollback_errors:
        raise RuntimeError(
            f"Rollback failed for {len(rollback_errors)} file(s): "
            + "; ".join(rollback_errors)
        )


_WIN32 = sys.platform == "win32"
_ATOMIC_RETRIES = 5 if _WIN32 else 0
_ATOMIC_RETRY_DELAY = 0.1  # seconds


def _atomic_write(file_path: Path, content: str) -> None:
    """Write content to file via temp file + rename for atomicity.

    Uses binary mode to avoid Python's text-mode line ending conversion
    (\\n -> \\r\\n on Windows), which would mutate files with Unix-style
    endings in git repos.

    On Windows, os.replace() can fail with PermissionError when the target
    file is momentarily locked (antivirus, search indexer, editor).  Retry
    a few times with a short sleep before giving up.  The temp file is
    always cleaned up on failure.
    """
    dir_path = file_path.parent
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=dir_path,
        suffix=".tmp", delete=False,
    ) as tmp:
        tmp.write(content.encode("utf-8"))
        tmp_path = Path(tmp.name)

    try:
        last_err: OSError | None = None
        for attempt in range(_ATOMIC_RETRIES + 1):
            try:
                tmp_path.replace(file_path)
                return
            except PermissionError as e:
                last_err = e
                if attempt < _ATOMIC_RETRIES:
                    time.sleep(_ATOMIC_RETRY_DELAY)
        # Exhausted retries (Windows) or first attempt failed (POSIX)
        raise last_err  # type: ignore[misc]
    except BaseException:
        # Clean up the orphaned temp file before propagating
        try:
            os.unlink(tmp_path)
        except OSError as cleanup_err:
            logger.warning("Failed to clean up temp file %s: %s", tmp_path, cleanup_err)
        raise
