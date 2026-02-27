"""Library scanner: resolve and scan library source files for indexing."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path

from clean_room_agent.parsers.registry import get_parser

logger = logging.getLogger(__name__)

# Directories to skip when scanning library packages
_SKIP_DIRS = {"tests", "test", "_vendor", "examples", "docs", "__pycache__", ".git"}

# Default max file size for library files (512KB — libraries are denser)
_DEFAULT_LIBRARY_MAX_FILE_SIZE = 524_288


@dataclass
class LibrarySource:
    """A resolved library package ready for scanning."""
    package_name: str
    package_path: Path


@dataclass
class LibraryFileInfo:
    """A scanned library file ready for indexing."""
    relative_path: str  # prefixed with package name, e.g. "numpy/core/numeric.py"
    absolute_path: Path
    size_bytes: int


def resolve_library_sources(
    repo_path: Path,
    config: dict | None = None,
) -> list[LibrarySource]:
    """Resolve library sources to scan.

    If config has library_sources = ["auto"], scans project Python files for imports
    and resolves package names to site-packages paths.
    If config has explicit library_paths, uses those directly.
    """
    config = config or {}
    sources = config["library_sources"]
    explicit_paths = config["library_paths"]

    if explicit_paths:
        result = []
        for entry in explicit_paths:
            if isinstance(entry, dict):
                name = entry.get("name")
                if not name:
                    raise ValueError(
                        f"library_paths entry missing or empty 'name': {entry!r}"
                    )
                path_str = entry.get("path")
                if not path_str:
                    raise ValueError(
                        f"library_paths entry missing or empty 'path': {entry!r}"
                    )
                path = Path(path_str)
            else:
                path = Path(entry)
                name = path.name
            if path.is_dir():
                result.append(LibrarySource(package_name=name, package_path=path))
            else:
                logger.warning("Library path does not exist: %s", path)
        return result

    if sources == ["auto"]:
        return _auto_resolve(repo_path)

    return []


def _auto_resolve(repo_path: Path) -> list[LibrarySource]:
    """Auto-resolve by scanning project Python files for imports."""
    # Collect unique top-level import names from project .py files
    import_names: set[str] = set()
    parser = get_parser("python")
    for py_file in repo_path.rglob("*.py"):
        # Skip hidden dirs and common non-source dirs
        parts = py_file.relative_to(repo_path).parts
        if any(p.startswith(".") or p in ("node_modules", "__pycache__", ".git", "venv", "env", ".tox") for p in parts):
            continue
        try:
            source = py_file.read_bytes()
        except (OSError, IOError) as e:
            logger.warning("Failed to read %s: %s", py_file, e)
            continue
        try:
            result = parser.parse(source, str(py_file.relative_to(repo_path)))
        except (OSError, ValueError, SyntaxError) as e:
            logger.warning("Failed to parse %s for import scanning: %s", py_file, e)
            continue
        for imp in result.imports:
            if not imp.is_relative:
                top_level = imp.module.split(".")[0]
                if top_level:
                    import_names.add(top_level)

    # Resolve each to site-packages
    result = []
    seen_paths: set[str] = set()
    unresolvable: list[tuple[str, str]] = []  # (name, reason)
    for name in sorted(import_names):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError, ModuleNotFoundError) as e:
            # Boundary: external system state (installed packages) — track but don't fail.
            unresolvable.append((name, str(e)))
            continue
        if spec is None or spec.origin is None:
            continue

        # Get package directory or single file
        origin = Path(spec.origin)
        if origin.name == "__init__.py":
            pkg_path = origin.parent
        else:
            # Single-file module (e.g. six.py) — use the file itself, not its parent
            pkg_path = origin

        # Skip stdlib and project-local modules
        pkg_str = str(pkg_path)
        if "site-packages" not in pkg_str:
            continue
        if pkg_str in seen_paths:
            continue
        seen_paths.add(pkg_str)
        result.append(LibrarySource(package_name=name, package_path=pkg_path))

    if unresolvable:
        logger.warning(
            "Library auto-resolve: %d import(s) could not be resolved (namespace packages, "
            "C extensions, or not installed): %s",
            len(unresolvable),
            ", ".join(f"{name} ({reason})" for name, reason in unresolvable),
        )

    return result


def scan_library(
    library: LibrarySource,
    max_file_size: int = _DEFAULT_LIBRARY_MAX_FILE_SIZE,
) -> list[LibraryFileInfo]:
    """Walk a library directory, collecting .py files for indexing.

    Skips tests/, _vendor/, examples/, docs/, __pycache__/.
    """
    result = []
    root = library.package_path

    # Handle single-file modules (e.g. six.py)
    if root.is_file():
        if root.suffix == ".py":
            try:
                size = root.stat().st_size
            except OSError as e:
                raise RuntimeError(
                    f"Failed to stat library file {root}: {e}. "
                    f"File exists but stat failed — check permissions."
                ) from e
            if size <= max_file_size:
                result.append(LibraryFileInfo(
                    relative_path=f"{library.package_name}/{root.name}",
                    absolute_path=root,
                    size_bytes=size,
                ))
        return result

    if not root.is_dir():
        logger.warning("Library path does not exist: %s", root)
        return result

    for py_file in root.rglob("*.py"):
        rel = py_file.relative_to(root)

        # Skip excluded directories
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue

        # Skip oversized files
        try:
            size = py_file.stat().st_size
        except OSError as e:
            raise RuntimeError(
                f"Failed to stat library file {py_file}: {e}. "
                f"File discovered by rglob but stat failed — check permissions."
            ) from e
        if size > max_file_size:
            continue

        # Prefix with package name
        prefixed_path = f"{library.package_name}/{rel.as_posix()}"
        result.append(LibraryFileInfo(
            relative_path=prefixed_path,
            absolute_path=py_file,
            size_bytes=size,
        ))

    return result
