"""Import resolution to file-level dependency edges."""

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from clean_room_agent.parsers.base import ExtractedImport


@dataclass
class ResolvedDep:
    source_path: str  # relative path of importing file
    target_path: str  # relative path of imported file
    kind: str  # "import" or "type_ref"


def resolve_dependencies(
    imports: list[ExtractedImport],
    file_path: str,
    language: str,
    file_index: set[str],
    repo_path: Path,
    library_file_index: dict[str, int] | None = None,
) -> list[ResolvedDep]:
    """Resolve imports to file-level dependency edges.

    Args:
        imports: Extracted imports from the parser.
        file_path: Relative path of the source file.
        language: Language of the source file.
        file_index: Set of all relative paths in the repo.
        repo_path: Root of the repository.
        library_file_index: Optional mapping of library file paths to file_ids.
            When an absolute import doesn't resolve against project file_index,
            falls back to library_file_index.

    Returns:
        List of resolved dependencies (intra-repo + library).
    """
    deps = []
    for imp in imports:
        if language == "python":
            targets = _resolve_python_import(imp, file_path, file_index)
            # Fall back to library resolution for unresolved absolute imports
            if not targets and not imp.is_relative and library_file_index:
                lib_targets = _resolve_python_absolute(imp, set(library_file_index.keys()))
                kind = "library_import"
                for target in lib_targets:
                    deps.append(ResolvedDep(source_path=file_path, target_path=target, kind=kind))
                continue
        elif language in ("typescript", "javascript"):
            targets = _resolve_ts_js_import(imp, file_path, file_index, repo_path)
        else:
            raise ValueError(f"Unsupported language for dependency resolution: {language!r}")

        kind = "type_ref" if imp.is_type_only else "import"
        for target in targets:
            deps.append(ResolvedDep(source_path=file_path, target_path=target, kind=kind))

    return deps


def _resolve_python_import(
    imp: ExtractedImport,
    file_path: str,
    file_index: set[str],
) -> list[str]:
    """Resolve a Python import to file paths."""
    if imp.is_relative:
        return _resolve_python_relative(imp, file_path, file_index)
    return _resolve_python_absolute(imp, file_index)


def _resolve_python_absolute(imp: ExtractedImport, file_index: set[str]) -> list[str]:
    """Resolve an absolute Python import (e.g. `from foo.bar import baz`)."""
    # Convert module path to file path candidates
    parts = imp.module.split(".")
    candidates = [
        "/".join(parts) + ".py",
        "/".join(parts) + "/__init__.py",
    ]
    # If we have specific names, also try module.name as a file
    for name in imp.names:
        candidates.append("/".join(parts + [name]) + ".py")
        candidates.append("/".join(parts + [name]) + "/__init__.py")

    return [c for c in candidates if c in file_index]


def _resolve_python_relative(
    imp: ExtractedImport,
    file_path: str,
    file_index: set[str],
) -> list[str]:
    """Resolve a relative Python import (e.g. `from ..foo import bar`)."""
    # level=1 means current package (same directory), level=2 means parent package, etc.
    current = PurePosixPath(file_path).parent
    for _ in range(imp.level - 1):
        current = current.parent

    if imp.module:
        parts = imp.module.split(".")
        base = current / "/".join(parts)
    else:
        base = current

    candidates = [
        str(base) + ".py",
        str(base / "__init__.py"),
    ]
    for name in imp.names:
        candidates.append(str(base / name) + ".py")
        candidates.append(str(base / name / "__init__.py"))

    # Normalize paths (remove leading ./)
    normalized = []
    for c in candidates:
        c = c.replace("\\", "/")
        if c.startswith("./"):
            c = c[2:]
        normalized.append(c)

    return [c for c in normalized if c in file_index]


def _resolve_ts_js_import(
    imp: ExtractedImport,
    file_path: str,
    file_index: set[str],
    repo_path: Path,
) -> list[str]:
    """Resolve a TS/JS import to file paths."""
    if not imp.is_relative and not _has_base_url(repo_path):
        # Non-relative import without tsconfig baseUrl -> external package
        return []

    if imp.is_relative:
        return _resolve_ts_js_relative(imp, file_path, file_index)

    # Non-relative with tsconfig baseUrl
    return _resolve_ts_js_baseurl(imp, file_index, repo_path)


def _resolve_ts_js_relative(
    imp: ExtractedImport,
    file_path: str,
    file_index: set[str],
) -> list[str]:
    """Resolve a relative TS/JS import."""
    current_dir = PurePosixPath(file_path).parent
    target = str(current_dir / imp.module).replace("\\", "/")
    if target.startswith("./"):
        target = target[2:]

    candidates = []
    extensions = [".ts", ".tsx", ".js", ".jsx"]

    # Handle explicit extension imports, e.g. import "./foo.ts"
    if any(target.endswith(ext) for ext in extensions):
        candidates.append(target)

    # Direct file with extension
    for ext in extensions:
        candidates.append(target + ext)
    # Index file
    for ext in extensions:
        candidates.append(target + "/index" + ext)

    return [c for c in candidates if c in file_index]


def _resolve_ts_js_baseurl(
    imp: ExtractedImport,
    file_index: set[str],
    repo_path: Path,
) -> list[str]:
    """Resolve a non-relative TS/JS import using tsconfig baseUrl."""
    base_url, paths = _read_tsconfig_paths(repo_path)
    if not base_url:
        return []

    module = imp.module

    # Check paths mappings first
    for pattern, targets in paths.items():
        prefix = pattern.replace("/*", "")
        if module.startswith(prefix):
            suffix = module[len(prefix):]
            for target_pattern in targets:
                target = target_pattern.replace("/*", suffix)
                resolved = str(PurePosixPath(base_url) / target).replace("\\", "/")
                extensions = [".ts", ".tsx", ".js", ".jsx"]
                if any(resolved.endswith(ext) for ext in extensions):
                    if resolved in file_index:
                        return [resolved]
                for ext in extensions:
                    candidate = resolved + ext
                    if candidate in file_index:
                        return [candidate]
                for ext in extensions:
                    candidate = resolved + "/index" + ext
                    if candidate in file_index:
                        return [candidate]

    # Try baseUrl + module
    target = str(PurePosixPath(base_url) / module).replace("\\", "/")
    extensions = [".ts", ".tsx", ".js", ".jsx"]
    candidates = []
    if any(target.endswith(ext) for ext in extensions):
        candidates.append(target)
    for ext in extensions:
        candidates.append(target + ext)
    for ext in extensions:
        candidates.append(target + "/index" + ext)

    return [c for c in candidates if c in file_index]


def _has_base_url(repo_path: Path) -> bool:
    """Check if tsconfig.json has a baseUrl.

    Raises on malformed tsconfig.json (fail-fast).
    """
    base_url, _ = _read_tsconfig_paths(repo_path)
    return bool(base_url)


def _read_tsconfig_paths(repo_path: Path) -> tuple[str, dict]:
    """Read baseUrl and paths from tsconfig.json.

    Raises on malformed tsconfig.json (fail-fast).
    """
    tsconfig = repo_path / "tsconfig.json"
    if not tsconfig.exists():
        return "", {}
    data = json.loads(tsconfig.read_text())
    opts = data.get("compilerOptions", {})
    base_url = opts.get("baseUrl", "")
    paths = opts.get("paths", {})
    return base_url, paths
