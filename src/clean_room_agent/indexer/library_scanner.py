"""Library scanner: resolve and scan library source files for indexing."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path

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
    sources = config.get("library_sources", ["auto"])
    explicit_paths = config.get("library_paths", [])

    if explicit_paths:
        result = []
        for entry in explicit_paths:
            if isinstance(entry, dict):
                name = entry.get("name", "")
                path = Path(entry.get("path", ""))
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
    for py_file in repo_path.rglob("*.py"):
        # Skip hidden dirs and common non-source dirs
        parts = py_file.relative_to(repo_path).parts
        if any(p.startswith(".") or p in ("node_modules", "__pycache__", ".git") for p in parts):
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
        except (OSError, IOError):
            continue
        for line in source.splitlines():
            line = line.strip()
            if line.startswith("import "):
                # "import foo" or "import foo.bar"
                module = line.split()[1].split(".")[0].split(",")[0]
                import_names.add(module)
            elif line.startswith("from "):
                # "from foo import bar" or "from foo.bar import baz"
                parts_split = line.split()
                if len(parts_split) >= 2:
                    module = parts_split[1].split(".")[0]
                    if not module.startswith("."):
                        import_names.add(module)

    # Resolve each to site-packages
    result = []
    seen_paths: set[str] = set()
    for name in sorted(import_names):
        try:
            spec = importlib.util.find_spec(name)
        except (ModuleNotFoundError, ValueError):
            continue
        if spec is None or spec.origin is None:
            continue

        # Get package directory
        origin = Path(spec.origin)
        if origin.name == "__init__.py":
            pkg_path = origin.parent
        else:
            pkg_path = origin.parent

        # Skip stdlib and project-local modules
        pkg_str = str(pkg_path)
        if "site-packages" not in pkg_str:
            continue
        if pkg_str in seen_paths:
            continue
        seen_paths.add(pkg_str)
        result.append(LibrarySource(package_name=name, package_path=pkg_path))

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

    if not root.is_dir():
        logger.warning("Library path is not a directory: %s", root)
        return result

    for py_file in root.rglob("*.py"):
        rel = py_file.relative_to(root)

        # Skip excluded directories
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue

        # Skip oversized files
        try:
            size = py_file.stat().st_size
        except OSError:
            continue
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
