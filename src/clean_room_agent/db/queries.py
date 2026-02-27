"""Curated DB insert/upsert/delete helpers."""

import sqlite3

from clean_room_agent.db.helpers import _insert_row, _now


def upsert_repo(conn: sqlite3.Connection, path: str, remote_url: str | None) -> int:
    """Insert or update a repo. Returns the repo id."""
    now = _now()
    row = conn.execute("SELECT id FROM repos WHERE path = ?", (path,)).fetchone()
    if row:
        conn.execute(
            "UPDATE repos SET remote_url = ?, indexed_at = ? WHERE id = ?",
            (remote_url, now, row["id"]),
        )
        return row["id"]
    return _insert_row(conn, "repos",
        ["path", "remote_url", "indexed_at"],
        [path, remote_url, now],
    )


def upsert_file(
    conn: sqlite3.Connection,
    repo_id: int,
    path: str,
    language: str,
    content_hash: str,
    size_bytes: int,
    file_source: str = "project",
) -> int:
    """Insert or update a file record. Returns the file id."""
    row = conn.execute(
        "SELECT id FROM files WHERE repo_id = ? AND path = ?",
        (repo_id, path),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE files SET language = ?, content_hash = ?, size_bytes = ?, file_source = ? WHERE id = ?",
            (language, content_hash, size_bytes, file_source, row["id"]),
        )
        return row["id"]
    cursor = conn.execute(
        "INSERT INTO files (repo_id, path, language, content_hash, size_bytes, file_source) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (repo_id, path, language, content_hash, size_bytes, file_source),
    )
    return cursor.lastrowid


def insert_symbol(
    conn: sqlite3.Connection,
    file_id: int,
    name: str,
    kind: str,
    start_line: int,
    end_line: int,
    signature: str | None = None,
    parent_symbol_id: int | None = None,
) -> int:
    """Insert a symbol. Returns the symbol id."""
    return _insert_row(conn, "symbols",
        ["file_id", "name", "kind", "start_line", "end_line", "signature", "parent_symbol_id"],
        [file_id, name, kind, start_line, end_line, signature, parent_symbol_id],
    )


def insert_docstring(
    conn: sqlite3.Connection,
    file_id: int,
    content: str,
    format_: str | None = None,
    parsed_fields: str | None = None,
    symbol_id: int | None = None,
) -> int:
    """Insert a docstring. Returns the docstring id."""
    return _insert_row(conn, "docstrings",
        ["symbol_id", "file_id", "content", "format", "parsed_fields"],
        [symbol_id, file_id, content, format_, parsed_fields],
    )


def insert_inline_comment(
    conn: sqlite3.Connection,
    file_id: int,
    line: int,
    content: str,
    kind: str | None = None,
    is_rationale: bool = False,
    symbol_id: int | None = None,
) -> int:
    """Insert an inline comment. Returns the comment id."""
    return _insert_row(conn, "inline_comments",
        ["file_id", "symbol_id", "line", "content", "kind", "is_rationale"],
        [file_id, symbol_id, line, content, kind, int(is_rationale)],
    )


def insert_dependency(
    conn: sqlite3.Connection,
    source_file_id: int,
    target_file_id: int,
    kind: str,
) -> int:
    """Insert a file-level dependency edge. Returns the dependency id."""
    return _insert_row(conn, "dependencies",
        ["source_file_id", "target_file_id", "kind"],
        [source_file_id, target_file_id, kind],
    )


def insert_symbol_reference(
    conn: sqlite3.Connection,
    caller_symbol_id: int,
    callee_symbol_id: int,
    reference_kind: str,
) -> int:
    """Insert a symbol-level reference edge. Returns the reference id."""
    return _insert_row(conn, "symbol_references",
        ["caller_symbol_id", "callee_symbol_id", "reference_kind"],
        [caller_symbol_id, callee_symbol_id, reference_kind],
    )


def insert_commit(
    conn: sqlite3.Connection,
    repo_id: int,
    hash_: str,
    timestamp: str,
    author: str | None = None,
    message: str | None = None,
    files_changed: int | None = None,
    insertions: int | None = None,
    deletions: int | None = None,
) -> int:
    """Insert or update a git commit. Returns the commit id.

    Commits are keyed by (repo_id, hash_) to avoid duplication across re-index runs.
    """
    existing = conn.execute(
        "SELECT id FROM commits WHERE repo_id = ? AND hash = ?",
        (repo_id, hash_),
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE commits SET author = ?, message = ?, timestamp = ?, files_changed = ?, "
            "insertions = ?, deletions = ? WHERE id = ?",
            (author, message, timestamp, files_changed, insertions, deletions, existing["id"]),
        )
        return existing["id"]

    cursor = conn.execute(
        "INSERT INTO commits (repo_id, hash, author, message, timestamp, files_changed, insertions, deletions) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (repo_id, hash_, author, message, timestamp, files_changed, insertions, deletions),
    )
    return cursor.lastrowid


def insert_file_commit(
    conn: sqlite3.Connection,
    file_id: int,
    commit_id: int,
) -> None:
    """Associate a file with a commit."""
    conn.execute(
        "INSERT OR IGNORE INTO file_commits (file_id, commit_id) VALUES (?, ?)",
        (file_id, commit_id),
    )


def upsert_co_change(
    conn: sqlite3.Connection,
    file_a_id: int,
    file_b_id: int,
    last_commit_hash: str | None = None,
    *,
    count: int | None = None,
) -> None:
    """Upsert a co-change pair. Enforces a < b ordering.

    If count is None (default), increments by 1.
    If count is provided, sets the count to that absolute value.
    """
    lo, hi = (file_a_id, file_b_id) if file_a_id < file_b_id else (file_b_id, file_a_id)
    if count is not None:
        conn.execute(
            "INSERT INTO co_changes (file_a_id, file_b_id, count, last_commit_hash) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(file_a_id, file_b_id) DO UPDATE SET "
            "count = excluded.count, last_commit_hash = excluded.last_commit_hash",
            (lo, hi, count, last_commit_hash),
        )
    else:
        conn.execute(
            "INSERT INTO co_changes (file_a_id, file_b_id, count, last_commit_hash) "
            "VALUES (?, ?, 1, ?) "
            "ON CONFLICT(file_a_id, file_b_id) DO UPDATE SET "
            "count = count + 1, last_commit_hash = excluded.last_commit_hash",
            (lo, hi, last_commit_hash),
        )


def upsert_file_metadata(
    conn: sqlite3.Connection,
    file_id: int,
    purpose: str | None = None,
    module: str | None = None,
    domain: str | None = None,
    concepts: str | None = None,
    public_api_surface: str | None = None,
    complexity_notes: str | None = None,
) -> None:
    """Insert or update file metadata (from enrichment promotion)."""
    conn.execute(
        "INSERT INTO file_metadata "
        "(file_id, purpose, module, domain, concepts, public_api_surface, complexity_notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(file_id) DO UPDATE SET "
        "purpose = excluded.purpose, module = excluded.module, domain = excluded.domain, "
        "concepts = excluded.concepts, public_api_surface = excluded.public_api_surface, "
        "complexity_notes = excluded.complexity_notes",
        (file_id, purpose, module, domain, concepts, public_api_surface, complexity_notes),
    )


def upsert_ref_source(
    conn: sqlite3.Connection,
    name: str,
    source_type: str,
    path: str,
    format_: str,
) -> int:
    """Insert or update a reference source. Returns the source id."""
    now = _now()
    row = conn.execute("SELECT id FROM ref_sources WHERE name = ?", (name,)).fetchone()
    if row:
        conn.execute(
            "UPDATE ref_sources SET source_type = ?, path = ?, format = ?, indexed_at = ? WHERE id = ?",
            (source_type, path, format_, now, row["id"]),
        )
        return row["id"]
    return _insert_row(conn, "ref_sources",
        ["name", "source_type", "path", "format", "indexed_at"],
        [name, source_type, path, format_, now],
    )


def insert_ref_section(
    conn: sqlite3.Connection,
    source_id: int,
    title: str,
    section_path: str,
    content: str,
    content_hash: str,
    size_bytes: int,
    section_type: str,
    ordering: int,
    parent_section_id: int | None = None,
    source_file: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> int:
    """Insert a reference section. Returns the section id."""
    return _insert_row(conn, "ref_sections",
        ["source_id", "title", "section_path", "content", "content_hash",
         "size_bytes", "section_type", "parent_section_id", "source_file",
         "start_line", "end_line", "ordering"],
        [source_id, title, section_path, content, content_hash,
         size_bytes, section_type, parent_section_id, source_file,
         start_line, end_line, ordering],
    )


def insert_ref_section_metadata(
    conn: sqlite3.Connection,
    section_id: int,
    domain: str | None = None,
    concepts: str | None = None,
    c_standard: str | None = None,
    header: str | None = None,
    related_functions: str | None = None,
) -> None:
    """Insert metadata for a reference section."""
    conn.execute(
        "INSERT INTO ref_section_metadata "
        "(section_id, domain, concepts, c_standard, header, related_functions) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (section_id, domain, concepts, c_standard, header, related_functions),
    )


def delete_ref_sections_for_source(conn: sqlite3.Connection, source_id: int) -> int:
    """Delete all sections (and their metadata) for a source. Returns count deleted."""
    conn.execute(
        "DELETE FROM ref_section_metadata WHERE section_id IN "
        "(SELECT id FROM ref_sections WHERE source_id = ?)",
        (source_id,),
    )
    cursor = conn.execute(
        "DELETE FROM ref_sections WHERE source_id = ?", (source_id,),
    )
    return cursor.rowcount


def delete_bridge_files_for_source(conn: sqlite3.Connection, source_name: str) -> int:
    """Delete bridge files and their metadata for a knowledge base source.

    Deletes child-first (file_metadata before files) to maintain referential integrity.
    Scoped to file_source='knowledge_base' with path prefix 'kb/{source_name}/'.
    Returns the number of files rows deleted.
    """
    pattern = f"kb/{source_name}/%"
    conn.execute(
        "DELETE FROM file_metadata WHERE file_id IN "
        "(SELECT id FROM files WHERE file_source = 'knowledge_base' AND path LIKE ?)",
        (pattern,),
    )
    cursor = conn.execute(
        "DELETE FROM files WHERE file_source = 'knowledge_base' AND path LIKE ?",
        (pattern,),
    )
    return cursor.rowcount


def get_ref_section_content_by_file_id(
    conn: sqlite3.Connection,
    file_id: int,
) -> str | None:
    """Get reference section content by looking up the bridge file's virtual path.

    The bridge links files.path (kb/{source}/{section_path}) to ref_sections.
    Returns the section content, or None if not found.
    """
    file_row = conn.execute(
        "SELECT path FROM files WHERE id = ?", (file_id,),
    ).fetchone()
    if not file_row:
        return None
    path = file_row["path"]
    if not path.startswith("kb/"):
        return None

    # Parse: kb/{source_name}/{section_path}
    parts = path.split("/", 2)
    if len(parts) < 3:
        return None
    source_name = parts[1]
    section_path = parts[2]

    row = conn.execute(
        "SELECT rs.content FROM ref_sections rs "
        "JOIN ref_sources src ON rs.source_id = src.id "
        "WHERE src.name = ? AND rs.section_path = ?",
        (source_name, section_path),
    ).fetchone()
    return row["content"] if row else None


def clear_file_children(conn: sqlite3.Connection, file_id: int) -> None:
    """Delete child data owned by a file, keeping the file row itself.

    Use this when re-indexing a changed file so the file_id is preserved.
    Only deletes data that "belongs" to this file:
    - Outgoing dependencies (source_file_id), NOT incoming from other files
    - Symbol references involving this file's symbols
    - Co-changes, file_commits, docstrings, comments, metadata, symbols
    """
    conn.execute(
        "DELETE FROM symbol_references WHERE caller_symbol_id IN "
        "(SELECT id FROM symbols WHERE file_id = ?) "
        "OR callee_symbol_id IN (SELECT id FROM symbols WHERE file_id = ?)",
        (file_id, file_id),
    )
    conn.execute("DELETE FROM docstrings WHERE file_id = ?", (file_id,))
    conn.execute("DELETE FROM inline_comments WHERE file_id = ?", (file_id,))
    conn.execute("DELETE FROM dependencies WHERE source_file_id = ?", (file_id,))
    conn.execute("DELETE FROM file_commits WHERE file_id = ?", (file_id,))
    conn.execute(
        "DELETE FROM co_changes WHERE file_a_id = ? OR file_b_id = ?",
        (file_id, file_id),
    )
    conn.execute("DELETE FROM file_metadata WHERE file_id = ?", (file_id,))
    conn.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))


def delete_file_data(conn: sqlite3.Connection, file_id: int) -> None:
    """Delete a file and all its associated data (full cascade).

    Use this when a file has been removed from the repository.
    Also cleans up incoming dependencies from other files (target side).
    """
    clear_file_children(conn, file_id)
    # Clean up incoming deps from other files that reference this file
    conn.execute("DELETE FROM dependencies WHERE target_file_id = ?", (file_id,))
    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
