"""Raw DB insert helpers."""

import sqlite3
from datetime import datetime, timezone


def insert_index_run(
    conn: sqlite3.Connection,
    repo_path: str,
    files_scanned: int,
    files_changed: int,
    duration_ms: int,
    status: str,
) -> int:
    """Log an indexing run to the raw DB. Returns the run id."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO index_runs (repo_path, files_scanned, files_changed, duration_ms, status, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (repo_path, files_scanned, files_changed, duration_ms, status, now),
    )
    return cursor.lastrowid


def insert_enrichment_output(
    conn: sqlite3.Connection,
    file_id: int,
    model: str,
    raw_prompt: str,
    raw_response: str,
    purpose: str | None = None,
    module: str | None = None,
    domain: str | None = None,
    concepts: str | None = None,
    public_api_surface: str | None = None,
    complexity_notes: str | None = None,
    promoted: bool = False,
) -> int:
    """Log an enrichment output to the raw DB. Returns the output id."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO enrichment_outputs "
        "(file_id, model, purpose, module, domain, concepts, public_api_surface, "
        "complexity_notes, raw_prompt, raw_response, promoted, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            file_id, model, purpose, module, domain, concepts,
            public_api_surface, complexity_notes, raw_prompt, raw_response,
            int(promoted), now,
        ),
    )
    return cursor.lastrowid
