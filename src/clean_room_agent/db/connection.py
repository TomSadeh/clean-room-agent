"""Connection factory for all three databases."""

import sqlite3
from pathlib import Path

from clean_room_agent.db.schema import (
    create_curated_schema,
    create_raw_schema,
    create_session_schema,
)

_VALID_ROLES = ("curated", "raw", "session")

_SCHEMA_CREATORS = {
    "curated": create_curated_schema,
    "raw": create_raw_schema,
    "session": create_session_schema,
}

def _db_path(repo_path: Path, role: str, task_id: str | None) -> Path:
    base = repo_path / ".clean_room"
    if role == "curated":
        return base / "curated.sqlite"
    if role == "raw":
        return base / "raw.sqlite"
    # session
    return base / "sessions" / f"session_{task_id}.sqlite"


def get_connection(
    role: str,
    *,
    repo_path: Path,
    task_id: str | None = None,
    read_only: bool = False,
) -> sqlite3.Connection:
    """Open a connection to the specified database.

    Args:
        role: One of "curated", "raw", or "session".
        repo_path: Root of the target repository.
        task_id: Required when role is "session".
        read_only: Open in read-only mode (URI ?mode=ro).

    Returns:
        A configured sqlite3.Connection with WAL mode, foreign keys, and Row factory.
    """
    if role not in _VALID_ROLES:
        raise ValueError(f"Invalid role {role!r}. Must be one of {_VALID_ROLES}")
    if role == "session" and not task_id:
        raise ValueError("task_id is required for session connections")

    db_file = _db_path(repo_path, role, task_id)
    if not read_only:
        db_file.parent.mkdir(parents=True, exist_ok=True)

    if read_only:
        if not db_file.exists():
            raise FileNotFoundError(
                f"{role} database not found at {db_file}. Run 'cra index' first."
            )
        # RFC 8089: file:///C:/... for Windows paths
        posix = db_file.as_posix()
        if len(posix) >= 2 and posix[1] == ":":
            posix = "/" + posix
        uri = f"file://{posix}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(str(db_file))

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Apply schema (idempotent CREATE TABLE IF NOT EXISTS)
    if not read_only:
        _SCHEMA_CREATORS[role](conn)
        conn.commit()

    return conn
