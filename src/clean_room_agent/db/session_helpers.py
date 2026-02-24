"""Session DB key-value helpers."""

import json
import sqlite3
from datetime import datetime, timezone


def set_state(conn: sqlite3.Connection, key: str, value: object) -> None:
    """JSON-serialize value and UPSERT into the session kv table."""
    now = datetime.now(timezone.utc).isoformat()
    serialized = json.dumps(value)
    conn.execute(
        "INSERT INTO kv (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (key, serialized, now),
    )
    conn.commit()


def get_state(conn: sqlite3.Connection, key: str) -> object | None:
    """Return deserialized Python object for key, or None if not found."""
    row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    return json.loads(row["value"])


def delete_state(conn: sqlite3.Connection, key: str) -> None:
    """Delete a key from the session kv table. Warns if key did not exist."""
    cursor = conn.execute("DELETE FROM kv WHERE key = ?", (key,))
    if cursor.rowcount == 0:
        import logging
        logging.getLogger(__name__).warning("delete_state: key %r did not exist", key)
    conn.commit()


def list_keys(conn: sqlite3.Connection, prefix: str | None = None) -> list[str]:
    """Return all keys, optionally filtered by prefix."""
    if prefix is not None:
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = conn.execute(
            "SELECT key FROM kv WHERE key LIKE ? ESCAPE '\\'",
            (f"{escaped}%",),
        ).fetchall()
    else:
        rows = conn.execute("SELECT key FROM kv").fetchall()
    return [r["key"] for r in rows]
