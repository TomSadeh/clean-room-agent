"""Raw DB insert helpers."""

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


def insert_index_run(
    conn: sqlite3.Connection,
    repo_path: str,
    files_scanned: int,
    files_changed: int,
    duration_ms: int,
    status: str,
) -> int:
    """Log an indexing run to the raw DB. Returns the run id."""
    return _insert_row(conn, "index_runs",
        ["repo_path", "files_scanned", "files_changed", "duration_ms", "status", "timestamp"],
        [repo_path, files_scanned, files_changed, duration_ms, status, _now()],
    )


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
    return _insert_row(conn, "retrieval_llm_calls",
        ["task_id", "call_type", "stage_name", "model", "system_prompt", "prompt",
         "response", "prompt_tokens", "completion_tokens", "latency_ms", "timestamp"],
        [task_id, call_type, stage_name, model, system_prompt, prompt,
         response, prompt_tokens, completion_tokens, latency_ms, _now()],
    )


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
    return _insert_row(conn, "retrieval_decisions",
        ["task_id", "stage", "file_id", "symbol_id", "detail_level",
         "tier", "included", "reason", "timestamp"],
        [task_id, stage, file_id, symbol_id, detail_level,
         tier, int(included), reason, _now()],
    )


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
    return _insert_row(conn, "task_runs",
        ["task_id", "repo_path", "mode", "execute_model", "context_window",
         "reserved_tokens", "stages", "plan_artifact", "timestamp"],
        [task_id, repo_path, mode, execute_model, context_window,
         reserved_tokens, stages, plan_artifact, _now()],
    )


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
    return _insert_row(conn, "session_archives",
        ["task_id", "session_blob", "archived_at"],
        [task_id, session_blob, _now()],
    )


def insert_run_attempt(
    conn: sqlite3.Connection,
    task_run_id: int,
    attempt: int,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    latency_ms: int,
    raw_response: str,
    patch_applied: bool,
) -> int:
    """Log a run attempt to the raw DB. Returns the attempt id."""
    return _insert_row(conn, "run_attempts",
        ["task_run_id", "attempt", "prompt_tokens", "completion_tokens",
         "latency_ms", "raw_response", "patch_applied", "timestamp"],
        [task_run_id, attempt, prompt_tokens, completion_tokens,
         latency_ms, raw_response, int(patch_applied), _now()],
    )


def update_run_attempt_patch(conn: sqlite3.Connection, attempt_id: int, patch_applied: bool) -> None:
    """Update the patch_applied flag on a run_attempt after edits are applied."""
    conn.execute(
        "UPDATE run_attempts SET patch_applied = ? WHERE id = ?",
        (int(patch_applied), attempt_id),
    )


def insert_validation_result(
    conn: sqlite3.Connection,
    attempt_id: int,
    success: bool,
    test_output: str | None = None,
    lint_output: str | None = None,
    type_check_output: str | None = None,
    failing_tests: str | None = None,
) -> int:
    """Log a validation result to the raw DB. Returns the result id."""
    return _insert_row(conn, "validation_results",
        ["attempt_id", "success", "test_output", "lint_output",
         "type_check_output", "failing_tests"],
        [attempt_id, int(success), test_output, lint_output,
         type_check_output, failing_tests],
    )


def insert_orchestrator_run(
    conn: sqlite3.Connection,
    task_id: str,
    repo_path: str,
    task_description: str,
    git_branch: str | None = None,
    git_base_ref: str | None = None,
) -> int:
    """Log an orchestrator run to the raw DB. Returns the run id."""
    columns = ["task_id", "repo_path", "task_description", "status", "timestamp"]
    values: list = [task_id, repo_path, task_description, "running", _now()]
    if git_branch is not None:
        columns.append("git_branch")
        values.append(git_branch)
    if git_base_ref is not None:
        columns.append("git_base_ref")
        values.append(git_base_ref)
    return _insert_row(conn, "orchestrator_runs", columns, values)


def update_orchestrator_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    total_parts: int | None = None,
    total_steps: int | None = None,
    parts_completed: int | None = None,
    steps_completed: int | None = None,
    status: str | None = None,
    completed_at: str | None = None,
) -> None:
    """Update an orchestrator run with progress/completion data."""
    sets = []
    params = []
    if total_parts is not None:
        sets.append("total_parts = ?")
        params.append(total_parts)
    if total_steps is not None:
        sets.append("total_steps = ?")
        params.append(total_steps)
    if parts_completed is not None:
        sets.append("parts_completed = ?")
        params.append(parts_completed)
    if steps_completed is not None:
        sets.append("steps_completed = ?")
        params.append(steps_completed)
    if status is not None:
        sets.append("status = ?")
        params.append(status)
    if completed_at is not None:
        sets.append("completed_at = ?")
        params.append(completed_at)
    if not sets:
        raise ValueError("update_orchestrator_run called with no fields to update")
    params.append(run_id)
    cursor = conn.execute(
        f"UPDATE orchestrator_runs SET {', '.join(sets)} WHERE id = ?",
        params,
    )
    if cursor.rowcount == 0:
        raise RuntimeError(f"No orchestrator_run with id={run_id} to update")


def insert_orchestrator_pass(
    conn: sqlite3.Connection,
    orchestrator_run_id: int,
    task_run_id: int,
    pass_type: str,
    sequence_order: int,
    *,
    part_id: str | None = None,
    step_id: str | None = None,
) -> int:
    """Log an orchestrator pass to the raw DB. Returns the pass id."""
    return _insert_row(conn, "orchestrator_passes",
        ["orchestrator_run_id", "task_run_id", "pass_type", "part_id",
         "step_id", "sequence_order", "timestamp"],
        [orchestrator_run_id, task_run_id, pass_type, part_id,
         step_id, sequence_order, _now()],
    )


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
    file_path: str | None = None,
) -> int:
    """Log an enrichment output to the raw DB. Returns the output id."""
    return _insert_row(conn, "enrichment_outputs",
        ["task_id", "file_id", "file_path", "model", "purpose", "module",
         "domain", "concepts", "public_api_surface", "complexity_notes",
         "system_prompt", "raw_prompt", "raw_response", "promoted", "timestamp"],
        [task_id, file_id, file_path, model, purpose, module,
         domain, concepts, public_api_surface, complexity_notes,
         system_prompt, raw_prompt, raw_response, int(promoted), _now()],
    )
