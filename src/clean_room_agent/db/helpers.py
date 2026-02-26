"""Shared DB helper utilities used by both raw and curated query modules."""

import sqlite3
from datetime import datetime, timezone


def _now() -> str:
    """UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _insert_row(
    conn: sqlite3.Connection,
    table: str,
    columns: list[str],
    values: list,
) -> int:
    """Insert a row and return lastrowid. Caller is responsible for timestamp."""
    placeholders = ", ".join(["?"] * len(values))
    cursor = conn.execute(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
    return cursor.lastrowid


def _build_update_clause(fields: dict) -> tuple[str, list]:
    """Build a SET clause from a dict of {column: value}, skipping None values.

    Returns:
        (set_clause, params) where set_clause is "col1 = ?, col2 = ?" and
        params is [val1, val2].

    Raises:
        ValueError: If no non-None fields are provided.
    """
    sets = []
    params = []
    for col, val in fields.items():
        if val is not None:
            sets.append(f"{col} = ?")
            params.append(val)
    if not sets:
        raise ValueError("no fields to update")
    return ", ".join(sets), params
