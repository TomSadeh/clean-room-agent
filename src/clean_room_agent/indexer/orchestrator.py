"""Indexing orchestrator: coordinates scanning, parsing, and DB population."""

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


def index_repository(repo_path: Path, continue_on_error: bool = False) -> IndexResult:
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

    curated_conn = get_connection("curated", repo_path=repo_path)
    raw_conn = get_connection("raw", repo_path=repo_path)

    try:
        return _do_index(repo_path, curated_conn, raw_conn, continue_on_error, start)
    finally:
        curated_conn.close()
        raw_conn.close()


def _do_index(
    repo_path: Path,
    curated_conn: sqlite3.Connection,
    raw_conn: sqlite3.Connection,
    continue_on_error: bool,
    start: float,
) -> IndexResult:
    # Register repo
    remote_url = get_remote_url(repo_path)
    repo_id = queries.upsert_repo(curated_conn, str(repo_path), remote_url)
    curated_conn.commit()

    # Scan files
    scanned = scan_repo(repo_path)
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
            result = parser.parse(source, fi.path)
            all_parse_results[p] = result
        except Exception as e:
            parse_errors += 1
            if continue_on_error:
                logger.exception("Parse error for %s", p)
                continue
            raise RuntimeError(f"Failed to parse {p}: {e}") from e

        file_id = file_id_map[p]

        # Insert symbols and keep full records for later line-aware attachment.
        symbol_name_to_id: dict[str, int] = {}
        symbol_records: list[dict] = []
        # First pass: non-child symbols
        for sym in result.symbols:
            if sym.parent_name is None:
                sid = queries.insert_symbol(
                    curated_conn, file_id, sym.name, sym.kind,
                    sym.start_line, sym.end_line, sym.signature,
                )
                symbol_name_to_id[sym.name] = sid
                symbol_records.append(
                    {
                        "id": sid,
                        "name": sym.name,
                        "start_line": sym.start_line,
                        "end_line": sym.end_line,
                    }
                )

        # Second pass: child symbols
        for sym in result.symbols:
            if sym.parent_name is not None:
                parent_id = symbol_name_to_id.get(sym.parent_name)
                sid = queries.insert_symbol(
                    curated_conn, file_id, sym.name, sym.kind,
                    sym.start_line, sym.end_line, sym.signature, parent_id,
                )
                symbol_name_to_id[sym.name] = sid
                symbol_records.append(
                    {
                        "id": sid,
                        "name": sym.name,
                        "start_line": sym.start_line,
                        "end_line": sym.end_line,
                    }
                )

        # Insert docstrings
        for doc in result.docstrings:
            sym_id = _find_symbol_id_for_attachment(
                symbol_records, symbol_name=doc.symbol_name, line=doc.line
            )
            queries.insert_docstring(
                curated_conn, file_id, doc.content, doc.format,
                doc.parsed_fields, sym_id,
            )

        # Insert comments
        for comment in result.comments:
            sym_id = _find_symbol_id_for_attachment(
                symbol_records, symbol_name=comment.symbol_name, line=comment.line
            )
            queries.insert_inline_comment(
                curated_conn, file_id, comment.line, comment.content,
                comment.kind, comment.is_rationale, sym_id,
            )

    curated_conn.commit()

    # Resolve dependencies for all new/changed files
    file_index = set(file_id_map.keys())
    for p in new_paths | changed_paths:
        if p not in all_parse_results:
            continue
        result = all_parse_results[p]
        fi = scanned_map[p]
        deps = resolve_dependencies(
            result.imports, fi.path, fi.language, file_index, repo_path
        )
        for dep in deps:
            source_id = file_id_map.get(dep.source_path)
            target_id = file_id_map.get(dep.target_path)
            if source_id and target_id:
                queries.insert_dependency(curated_conn, source_id, target_id, dep.kind)
    curated_conn.commit()

    # Insert symbol references (Python only)
    for p in new_paths | changed_paths:
        if p not in all_parse_results:
            continue
        result = all_parse_results[p]
        file_id = file_id_map[p]

        # Build local symbol name -> id map for this file.
        # Skip ambiguous names (e.g., __init__ in multiple classes) to avoid
        # silently linking references to the wrong symbol.
        file_symbols = curated_conn.execute(
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
                    curated_conn, caller_id, callee_id,
                    ref.reference_kind,
                )
    curated_conn.commit()

    # Extract git history (pass remote_url to avoid redundant subprocess call)
    try:
        git_history = extract_git_history(repo_path, file_index, remote_url=remote_url)
    except RuntimeError:
        logger.warning("Git history extraction failed, skipping")
        git_history = None

    if git_history:
        for commit in git_history.commits:
            commit_id = queries.insert_commit(
                curated_conn, repo_id, commit.hash, commit.timestamp,
                commit.author, commit.message, commit.files_changed,
                commit.insertions, commit.deletions,
            )
            for fp in commit.changed_files:
                fid = file_id_map.get(fp)
                if fid:
                    queries.insert_file_commit(curated_conn, fid, commit_id)

        for (fa, fb), count in git_history.co_change_counts.items():
            fa_id = file_id_map.get(fa)
            fb_id = file_id_map.get(fb)
            if fa_id and fb_id:
                queries.upsert_co_change(
                    curated_conn, fa_id, fb_id, count=count,
                )
    curated_conn.commit()

    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Log to raw DB
    raw_queries.insert_index_run(
        raw_conn,
        repo_path=str(repo_path),
        files_scanned=len(scanned),
        files_changed=len(new_paths) + len(changed_paths),
        duration_ms=elapsed_ms,
        status="success",
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
