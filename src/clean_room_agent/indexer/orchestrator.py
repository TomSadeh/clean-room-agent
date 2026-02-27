"""Indexing orchestrator: coordinates scanning, parsing, and DB population."""

import hashlib
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from clean_room_agent.db.connection import get_connection
from clean_room_agent.db import queries, raw_queries
from clean_room_agent.extractors.dependencies import resolve_dependencies
from clean_room_agent.extractors.git_extractor import extract_git_history, get_remote_url
from clean_room_agent.indexer.file_scanner import scan_repo
from clean_room_agent.indexer.library_scanner import _DEFAULT_LIBRARY_MAX_FILE_SIZE
from clean_room_agent.constants import language_from_extension
from clean_room_agent.parsers.registry import get_parser

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    files_scanned: int
    files_new: int
    files_changed: int
    files_deleted: int
    files_unchanged: int
    parse_errors: int
    duration_ms: int


def _find_symbol_id_for_attachment(
    symbol_records: list[dict],
    *,
    symbol_name: str | None,
    line: int | None,
) -> int | None:
    """Resolve a symbol id for docstring/comment attachment.

    Prefer line-range matching when line is provided, then narrow to the smallest span.
    If line is unavailable, only attach when the symbol name is unique in the file.
    """
    if not symbol_name:
        return None

    candidates = [s for s in symbol_records if s["name"] == symbol_name]
    if not candidates:
        return None

    if line is not None:
        line_candidates = [s for s in candidates if s["start_line"] <= line <= s["end_line"]]
        if line_candidates:
            best = min(line_candidates, key=lambda s: (s["end_line"] - s["start_line"], s["id"]))
            return best["id"]

    if len(candidates) == 1:
        return candidates[0]["id"]
    return None


def _insert_file_symbols(
    conn: sqlite3.Connection,
    file_id: int,
    parsed_result,
) -> list[dict]:
    """Insert symbols from a parsed result into the DB (two-pass: parents then children).

    Returns list of {id, name, start_line, end_line} dicts for attachment lookups.
    """
    symbol_records: list[dict] = []

    # First pass: top-level symbols (no parent)
    parent_id_map: dict[str, int] = {}
    for sym in parsed_result.symbols:
        if sym.parent_name is None:
            sid = queries.insert_symbol(
                conn, file_id, sym.name, sym.kind,
                sym.start_line, sym.end_line, sym.signature,
            )
            parent_id_map[sym.name] = sid
            symbol_records.append({
                "id": sid, "name": sym.name,
                "start_line": sym.start_line, "end_line": sym.end_line,
            })

    # Second pass: child symbols
    for sym in parsed_result.symbols:
        if sym.parent_name is not None:
            parent_id = parent_id_map.get(sym.parent_name)
            sid = queries.insert_symbol(
                conn, file_id, sym.name, sym.kind,
                sym.start_line, sym.end_line, sym.signature,
                parent_symbol_id=parent_id,
            )
            symbol_records.append({
                "id": sid, "name": sym.name,
                "start_line": sym.start_line, "end_line": sym.end_line,
            })

    return symbol_records


def _attach_docstrings_comments(
    conn: sqlite3.Connection,
    file_id: int,
    parsed_result,
    symbol_records: list[dict],
    *,
    include_comments: bool = True,
) -> None:
    """Attach docstrings (and optionally comments) to symbols in the DB."""
    for doc in parsed_result.docstrings:
        sym_id = _find_symbol_id_for_attachment(
            symbol_records, symbol_name=doc.symbol_name, line=doc.line,
        )
        queries.insert_docstring(
            conn, file_id, doc.content, doc.format,
            doc.parsed_fields, sym_id,
        )

    if include_comments:
        for comment in parsed_result.comments:
            sym_id = _find_symbol_id_for_attachment(
                symbol_records, symbol_name=comment.symbol_name, line=comment.line,
            )
            queries.insert_inline_comment(
                conn, file_id, comment.line, comment.content,
                comment.kind, comment.is_rationale, sym_id,
            )


def _resolve_deps_and_refs(
    conn: sqlite3.Connection,
    file_id_map: dict[str, int],
    all_parse_results: dict,
    paths: set[str],
    scanned_map: dict,
    repo_path: Path,
) -> None:
    """Resolve file dependencies and symbol references for parsed files."""
    file_index = set(file_id_map.keys())

    # Dependencies
    for p in paths:
        if p not in all_parse_results:
            continue
        result = all_parse_results[p]
        fi = scanned_map[p]
        deps = resolve_dependencies(
            result.imports, fi.path, fi.language, file_index, repo_path,
        )
        for dep in deps:
            source_id = file_id_map.get(dep.source_path)
            target_id = file_id_map.get(dep.target_path)
            if source_id and target_id:
                queries.insert_dependency(conn, source_id, target_id, dep.kind)
    conn.commit()

    # Symbol references (Python only)
    for p in paths:
        if p not in all_parse_results:
            continue
        result = all_parse_results[p]
        file_id = file_id_map[p]

        # Build local symbol name -> id map for this file.
        # Skip ambiguous names (e.g., __init__ in multiple classes) to avoid
        # silently linking references to the wrong symbol.
        file_symbols = conn.execute(
            "SELECT id, name FROM symbols WHERE file_id = ?", (file_id,)
        ).fetchall()
        local_sym_map: dict[str, int] = {}
        _ambiguous: set[str] = set()
        for r in file_symbols:
            name = r["name"]
            if name in local_sym_map:
                _ambiguous.add(name)
            else:
                local_sym_map[name] = r["id"]
        for name in _ambiguous:
            del local_sym_map[name]

        for ref in result.references:
            caller_id = local_sym_map.get(ref.caller_symbol)
            callee_id = local_sym_map.get(ref.callee_symbol)
            if caller_id and callee_id:
                queries.insert_symbol_reference(
                    conn, caller_id, callee_id,
                    ref.reference_kind,
                )
    conn.commit()


def _index_git_history(
    conn: sqlite3.Connection,
    repo_id: int,
    repo_path: Path,
    file_id_map: dict[str, int],
    remote_url: str | None,
    indexer_config: dict,
) -> None:
    """Extract and store git history (commits, co-changes)."""
    file_index = set(file_id_map.keys())
    git_kwargs: dict = {}
    if "max_commits" in indexer_config:
        git_kwargs["max_commits"] = indexer_config["max_commits"]
    if "co_change_max_files" in indexer_config:
        git_kwargs["co_change_max_files"] = indexer_config["co_change_max_files"]
    if "co_change_min_count" in indexer_config:
        git_kwargs["co_change_min_count"] = indexer_config["co_change_min_count"]
    git_history = extract_git_history(repo_path, file_index, remote_url=remote_url, **git_kwargs)

    if git_history:
        for commit in git_history.commits:
            # Only insert commits that touch at least one indexed file (T23).
            # Commits touching only excluded files would create orphan rows.
            tracked_files = [fp for fp in commit.changed_files if fp in file_id_map]
            if not tracked_files:
                continue
            commit_id = queries.insert_commit(
                conn, repo_id, commit.hash, commit.timestamp,
                commit.author, commit.message, commit.files_changed,
                commit.insertions, commit.deletions,
            )
            for fp in tracked_files:
                queries.insert_file_commit(conn, file_id_map[fp], commit_id)

        for (fa, fb), count in git_history.co_change_counts.items():
            fa_id = file_id_map.get(fa)
            fb_id = file_id_map.get(fb)
            if fa_id and fb_id:
                queries.upsert_co_change(
                    conn, fa_id, fb_id, count=count,
                )
    conn.commit()


def index_repository(
    repo_path: Path,
    continue_on_error: bool = False,
    indexer_config: dict | None = None,
) -> IndexResult:
    """Index a repository into the curated and raw databases.

    Pipeline:
    1. Open DBs, apply schemas
    2. Register repo
    3. Scan files
    4. Incremental diff (content_hash comparison)
    5. Delete changed/removed file data
    6. Upsert files
    7. Parse new/changed files
    8. Resolve dependencies
    9. Extract git history
    10. Log to raw DB
    """
    start = time.monotonic()
    repo_path = repo_path.resolve()

    curated_conn = None
    raw_conn = None

    try:
        curated_conn = get_connection("curated", repo_path=repo_path)
        raw_conn = get_connection("raw", repo_path=repo_path)
        return _do_index(repo_path, curated_conn, raw_conn, continue_on_error, start, indexer_config)
    finally:
        if raw_conn is not None:
            raw_conn.close()
        if curated_conn is not None:
            curated_conn.close()


def _do_index(
    repo_path: Path,
    curated_conn: sqlite3.Connection,
    raw_conn: sqlite3.Connection,
    continue_on_error: bool,
    start: float,
    indexer_config: dict | None = None,
) -> IndexResult:
    ic = indexer_config or {}

    # Register repo
    remote_url = get_remote_url(repo_path)
    repo_id = queries.upsert_repo(curated_conn, str(repo_path), remote_url)
    curated_conn.commit()

    # Scan files
    scan_kwargs = {}
    if "max_file_size" in ic:
        scan_kwargs["max_file_size"] = ic["max_file_size"]
    scanned = scan_repo(repo_path, **scan_kwargs)
    scanned_map = {fi.path: fi for fi in scanned}

    # Get existing files from DB
    existing = curated_conn.execute(
        "SELECT id, path, content_hash FROM files WHERE repo_id = ?", (repo_id,)
    ).fetchall()
    existing_map = {r["path"]: (r["id"], r["content_hash"]) for r in existing}

    # Compute diffs
    new_paths = set(scanned_map.keys()) - set(existing_map.keys())
    removed_paths = set(existing_map.keys()) - set(scanned_map.keys())
    common_paths = set(scanned_map.keys()) & set(existing_map.keys())

    changed_paths = set()
    for p in common_paths:
        if scanned_map[p].content_hash != existing_map[p][1]:
            changed_paths.add(p)

    unchanged_paths = common_paths - changed_paths

    # Delete removed files entirely, clear child data for changed files
    for p in removed_paths:
        file_id = existing_map[p][0]
        queries.delete_file_data(curated_conn, file_id)
    for p in changed_paths:
        file_id = existing_map[p][0]
        queries.clear_file_children(curated_conn, file_id)
    curated_conn.commit()

    # Upsert all scanned files and parse new/changed
    file_id_map: dict[str, int] = {}  # relative path -> file_id
    parse_errors = 0

    # Re-insert unchanged files into the map
    for p in unchanged_paths:
        file_id_map[p] = existing_map[p][0]

    # Insert/update new and changed files
    for p in new_paths | changed_paths:
        fi = scanned_map[p]
        file_id = queries.upsert_file(
            curated_conn, repo_id, fi.path, fi.language, fi.content_hash, fi.size_bytes
        )
        file_id_map[p] = file_id
    curated_conn.commit()

    # Parse new/changed files
    all_parse_results = {}
    for p in new_paths | changed_paths:
        fi = scanned_map[p]
        try:
            parser = get_parser(fi.language)
            source = fi.abs_path.read_bytes()
            # T24: Verify hash matches scan to detect file changes between
            # scan and parse (TOCTOU).  Fail-fast rather than silently parsing
            # content that doesn't match the recorded hash.
            actual_hash = hashlib.sha256(source).hexdigest()
            if actual_hash != fi.content_hash:
                raise RuntimeError(
                    f"File {fi.path} changed between scan and parse "
                    f"(hash {fi.content_hash[:12]}→{actual_hash[:12]}). "
                    f"Re-run indexing."
                )
            result = parser.parse(source, fi.path)
            all_parse_results[p] = result
        except Exception as e:
            parse_errors += 1
            if continue_on_error:
                logger.exception("Parse error for %s", p)
                continue
            raise RuntimeError(f"Failed to parse {p}: {e}") from e

        file_id = file_id_map[p]

        symbol_records = _insert_file_symbols(curated_conn, file_id, result)
        _attach_docstrings_comments(curated_conn, file_id, result, symbol_records)

    curated_conn.commit()

    # Resolve dependencies and symbol references for all new/changed files
    _resolve_deps_and_refs(
        curated_conn, file_id_map, all_parse_results,
        new_paths | changed_paths, scanned_map, repo_path,
    )

    # Extract and store git history
    _index_git_history(
        curated_conn, repo_id, repo_path, file_id_map, remote_url, ic,
    )

    elapsed_ms = int((time.monotonic() - start) * 1000)

    # T17: Curated is committed first here because indexing data IS the curated DB —
    # both are deterministic and rebuildable from source via `cra index`. The raw DB
    # entry is metadata only (run stats). A crash between commits loses the audit trail
    # for this run but curated data is consistent and re-indexing will produce the same result.
    raw_queries.insert_index_run(
        raw_conn,
        repo_path=str(repo_path),
        files_scanned=len(scanned),
        files_changed=len(new_paths) + len(changed_paths),
        duration_ms=elapsed_ms,
        status="partial" if parse_errors > 0 and continue_on_error else "success",
    )
    raw_conn.commit()

    return IndexResult(
        files_scanned=len(scanned),
        files_new=len(new_paths),
        files_changed=len(changed_paths),
        files_deleted=len(removed_paths),
        files_unchanged=len(unchanged_paths),
        parse_errors=parse_errors,
        duration_ms=elapsed_ms,
    )


@dataclass
class LibraryIndexResult:
    libraries_found: int
    files_scanned: int
    files_new: int
    files_changed: int
    files_unchanged: int
    parse_errors: int
    read_errors: int
    duration_ms: int


def index_libraries(
    repo_path: Path,
    indexer_config: dict | None = None,
) -> LibraryIndexResult:
    """Index library/dependency source files into the curated DB.

    Uses the same repo_id as the project (files differentiated by file_source='library').
    Parses symbols, signatures, docstrings. Skips git history, co-changes, inline
    comments, symbol_references (not useful for libraries).
    Incremental via content_hash comparison.
    """
    from clean_room_agent.indexer.library_scanner import (
        resolve_library_sources,
        scan_library,
    )

    start = time.monotonic()
    repo_path = repo_path.resolve()
    ic = indexer_config or {}

    curated_conn = get_connection("curated", repo_path=repo_path)
    raw_conn = get_connection("raw", repo_path=repo_path)

    try:
        # Look up repo_id (must exist from prior `cra index`)
        repo_row = curated_conn.execute(
            "SELECT id FROM repos WHERE path = ?", (str(repo_path),)
        ).fetchone()
        if not repo_row:
            raise RuntimeError(
                f"No indexed repo at {repo_path}. Run 'cra index' before 'cra index-libraries'."
            )
        repo_id = repo_row["id"]

        # Resolve library sources
        sources = resolve_library_sources(repo_path, ic)
        max_file_size = ic.get("library_max_file_size", _DEFAULT_LIBRARY_MAX_FILE_SIZE)

        total_scanned = 0
        total_new = 0
        total_changed = 0
        total_unchanged = 0
        total_parse_errors = 0
        total_read_errors = 0
        scanned_paths: set[str] = set()  # track all scanned relative paths for stale cleanup

        # Get existing library files from DB
        existing = curated_conn.execute(
            "SELECT id, path, content_hash FROM files WHERE repo_id = ? AND file_source = 'library'",
            (repo_id,),
        ).fetchall()
        existing_map = {r["path"]: (r["id"], r["content_hash"]) for r in existing}

        for lib in sources:
            lib_files = scan_library(lib, max_file_size=max_file_size)
            total_scanned += len(lib_files)

            for lf in lib_files:
                scanned_paths.add(lf.relative_path)
                # Compute content hash
                try:
                    content = lf.absolute_path.read_bytes()
                except (OSError, IOError) as e:
                    logger.warning("Failed to read library file %s: %s", lf.absolute_path, e)
                    total_read_errors += 1
                    continue
                content_hash = hashlib.sha256(content).hexdigest()

                # Check if unchanged
                is_changed = False
                if lf.relative_path in existing_map:
                    if existing_map[lf.relative_path][1] == content_hash:
                        total_unchanged += 1
                        continue
                    # Changed: clear children
                    queries.clear_file_children(curated_conn, existing_map[lf.relative_path][0])
                    is_changed = True

                # Detect language from extension
                language = language_from_extension(lf.relative_path)
                if language is None:
                    logger.warning("Unsupported extension for library file %s, skipping", lf.relative_path)
                    total_parse_errors += 1
                    continue

                # Upsert file record
                file_id = queries.upsert_file(
                    curated_conn, repo_id, lf.relative_path, language,
                    content_hash, lf.size_bytes, file_source="library",
                )

                # Parse symbols and docstrings
                try:
                    parser = get_parser(language)
                    parsed = parser.parse(content, lf.relative_path)
                except Exception as e:
                    total_parse_errors += 1
                    logger.warning("Parse error for library file %s: %s", lf.relative_path, e)
                    continue

                if is_changed:
                    total_changed += 1
                else:
                    total_new += 1

                symbol_records = _insert_file_symbols(curated_conn, file_id, parsed)
                _attach_docstrings_comments(
                    curated_conn, file_id, parsed, symbol_records,
                    include_comments=False,
                )

            curated_conn.commit()

        # Clean up stale library files no longer present in any scanned library
        stale_paths = set(existing_map.keys()) - scanned_paths
        for stale_path in stale_paths:
            stale_id = existing_map[stale_path][0]
            queries.delete_file_data(curated_conn, stale_id)
        if stale_paths:
            curated_conn.commit()
            logger.info("Removed %d stale library files", len(stale_paths))

        elapsed_ms = int((time.monotonic() - start) * 1000)

        raw_queries.insert_index_run(
            raw_conn,
            repo_path=str(repo_path),
            files_scanned=total_scanned,
            files_changed=total_new + total_changed,
            duration_ms=elapsed_ms,
            status="library_index",
        )
        raw_conn.commit()

        return LibraryIndexResult(
            libraries_found=len(sources),
            files_scanned=total_scanned,
            files_new=total_new,
            files_changed=total_changed,
            files_unchanged=total_unchanged,
            parse_errors=total_parse_errors,
            read_errors=total_read_errors,
            duration_ms=elapsed_ms,
        )
    finally:
        curated_conn.close()
        raw_conn.close()
