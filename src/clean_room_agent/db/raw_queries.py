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


def insert_retrieval_llm_call(
    conn: sqlite3.Connection,
    task_id: str,
    call_type: str,
    model: str,
    prompt: str,
    response: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    latency_ms: int,
    stage_name: str | None = None,
    system_prompt: str | None = None,
) -> int:
    """Log a retrieval LLM call to the raw DB. Returns the call id."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO retrieval_llm_calls "
        "(task_id, call_type, stage_name, model, system_prompt, prompt, response, "
        "prompt_tokens, completion_tokens, latency_ms, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (task_id, call_type, stage_name, model, system_prompt, prompt, response,
         prompt_tokens, completion_tokens, latency_ms, now),
    )
    return cursor.lastrowid


def insert_retrieval_decision(
    conn: sqlite3.Connection,
    task_id: str,
    stage: str,
    file_id: int,
    included: bool,
    tier: str | None = None,
    reason: str | None = None,
    symbol_id: int | None = None,
    detail_level: str | None = None,
) -> int:
    """Log a retrieval file or symbol decision. Returns the decision id."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO retrieval_decisions "
        "(task_id, stage, file_id, symbol_id, detail_level, tier, included, reason, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (task_id, stage, file_id, symbol_id, detail_level, tier, int(included), reason, now),
    )
    return cursor.lastrowid


def insert_task_run(
    conn: sqlite3.Connection,
    task_id: str,
    repo_path: str,
    mode: str,
    execute_model: str,
    context_window: int,
    reserved_tokens: int,
    stages: str,
    plan_artifact: str | None = None,
) -> int:
    """Log a task run to the raw DB. Returns the task_run id."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO task_runs "
        "(task_id, repo_path, mode, execute_model, context_window, reserved_tokens, "
        "stages, plan_artifact, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (task_id, repo_path, mode, execute_model, context_window,
         reserved_tokens, stages, plan_artifact, now),
    )
    return cursor.lastrowid


def update_task_run(
    conn: sqlite3.Connection,
    task_run_id: int,
    success: bool,
    total_tokens: int | None = None,
    total_latency_ms: int | None = None,
    final_diff: str | None = None,
    final_plan: str | None = None,
) -> None:
    """Update a task run with completion data."""
    cursor = conn.execute(
        "UPDATE task_runs SET success = ?, total_tokens = ?, total_latency_ms = ?, "
        "final_diff = ?, final_plan = ? WHERE id = ?",
        (int(success), total_tokens, total_latency_ms, final_diff, final_plan, task_run_id),
    )
    if cursor.rowcount == 0:
        raise RuntimeError(f"No task_run with id={task_run_id} to update")


def insert_session_archive(
    conn: sqlite3.Connection,
    task_id: str,
    session_blob: bytes,
) -> int:
    """Archive a session DB blob to the raw DB. Returns the archive id."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO session_archives (task_id, session_blob, archived_at) "
        "VALUES (?, ?, ?)",
        (task_id, session_blob, now),
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
    system_prompt: str | None = None,
    task_id: str | None = None,
) -> int:
    """Log an enrichment output to the raw DB. Returns the output id."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO enrichment_outputs "
        "(task_id, file_id, model, purpose, module, domain, concepts, public_api_surface, "
        "complexity_notes, system_prompt, raw_prompt, raw_response, promoted, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            task_id, file_id, model, purpose, module, domain, concepts,
            public_api_surface, complexity_notes, system_prompt, raw_prompt, raw_response,
            int(promoted), now,
        ),
    )
    return cursor.lastrowid
