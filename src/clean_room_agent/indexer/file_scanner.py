"""Scan a repository for indexable source files."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pathspec

from clean_room_agent.constants import LANGUAGE_MAP

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 1_048_576  # 1 MB

SKIP_DIRS: set[str] = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".clean_room",
    "dist",
    "build",
    ".eggs",
}

SKIP_SUFFIXES: set[str] = {
    ".pyc",
    ".min.js",
    ".min.css",
    ".map",
}


@dataclass(frozen=True)
class FileInfo:
    """Metadata for a single scanned source file."""

    path: str  # relative to repo root, forward-slash separated
    abs_path: Path
    language: str
    content_hash: str  # SHA-256 hex
    size_bytes: int


def _load_gitignore_spec(repo_path: Path) -> pathspec.PathSpec | None:
    """Parse .gitignore at repo root. Returns None if absent."""
    gitignore = repo_path / ".gitignore"
    if not gitignore.is_file():
        return None
    text = gitignore.read_text(encoding="utf-8", errors="replace")
    return pathspec.PathSpec.from_lines("gitignore", text.splitlines())


def _should_skip_dir(name: str) -> bool:
    """Check if a directory name matches a skip pattern (including *.egg-info)."""
    if name in SKIP_DIRS:
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def _should_skip_file(name: str) -> bool:
    """Check if a filename matches a skip-file pattern."""
    for suffix in SKIP_SUFFIXES:
        if name.endswith(suffix):
            return True
    return False


def _hash_file(file_path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _walk_repo(repo_path: Path, gitignore_spec: pathspec.PathSpec | None):
    """Yield (relative_posix_path, abs_path) for every candidate file."""
    stack: list[Path] = [repo_path]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except PermissionError as e:
            raise PermissionError(f"Cannot read directory {current}: {e}") from e

        dirs: list[Path] = []
        files: list[Path] = []
        for entry in entries:
            if entry.is_dir():
                dirs.append(entry)
            elif entry.is_file():
                files.append(entry)

        for d in dirs:
            if _should_skip_dir(d.name):
                continue
            rel = d.relative_to(repo_path).as_posix()
            if gitignore_spec is not None and gitignore_spec.match_file(rel + "/"):
                continue
            stack.append(d)

        for f in files:
            rel = f.relative_to(repo_path).as_posix()
            if gitignore_spec is not None and gitignore_spec.match_file(rel):
                continue
            yield rel, f


def scan_repo(repo_path: Path, *, max_file_size: int = MAX_FILE_SIZE) -> list[FileInfo]:
    """Scan a repository and return FileInfo for every indexable source file.

    Files are filtered by language (extension), skip patterns, .gitignore,
    and max file size. Results are sorted by relative path.
    """
    repo_path = repo_path.resolve()
    gitignore_spec = _load_gitignore_spec(repo_path)
    results: list[FileInfo] = []

    for rel_path, abs_path in _walk_repo(repo_path, gitignore_spec):
        if _should_skip_file(abs_path.name):
            continue

        suffix = abs_path.suffix
        language = LANGUAGE_MAP.get(suffix)
        if language is None:
            continue

        size = abs_path.stat().st_size
        if size > max_file_size:
            logger.warning("Skipping oversized file (%d bytes): %s", size, rel_path)
            continue

        content_hash = _hash_file(abs_path)
        results.append(
            FileInfo(
                path=rel_path,
                abs_path=abs_path,
                language=language,
                content_hash=content_hash,
                size_bytes=size,
            )
        )

    results.sort(key=lambda fi: fi.path)
    return results
