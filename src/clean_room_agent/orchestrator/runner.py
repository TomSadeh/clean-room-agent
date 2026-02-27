"""Orchestrator: deterministic sequencer for plan + implement passes."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from clean_room_agent.config import require_config_section, require_models_config

if TYPE_CHECKING:
    from clean_room_agent.llm.client import ModelConfig
    from clean_room_agent.orchestrator.git_ops import GitWorkflow
    from clean_room_agent.trace import TraceLogger
from clean_room_agent.db.connection import _db_path, get_connection
from clean_room_agent.db.raw_queries import (
    insert_orchestrator_pass,
    insert_orchestrator_run,
    insert_retrieval_llm_call,
    insert_run_attempt,
    insert_session_archive,
    insert_task_run,
    mark_part_attempts_rolled_back,
    update_orchestrator_pass_sha,
    update_orchestrator_run,
    update_run_attempt_patch,
)
from clean_room_agent.db.session_helpers import set_state
from clean_room_agent.execute.dataclasses import (
    MetaPlan,
    OrchestratorResult,
    PartPlan,
    PassResult,
    PatchResult,
    PlanArtifact,
    PlanStep,
    StepResult,
)
from clean_room_agent.execute.documentation import run_documentation_pass
from clean_room_agent.execute.implement import execute_implement, execute_test_implement
from clean_room_agent.execute.patch import apply_edits, rollback_edits
from clean_room_agent.execute.plan import execute_plan
from clean_room_agent.execute.scaffold import (
    execute_function_implement,
    execute_scaffold,
    extract_function_stubs,
    is_c_part,
    validate_scaffold_compilation,
)
from clean_room_agent.llm.client import LoggedLLMClient
from clean_room_agent.llm.router import ModelRouter
from clean_room_agent.orchestrator.validator import require_testing_config, run_validation
from clean_room_agent.retrieval.dataclasses import BudgetConfig
from clean_room_agent.retrieval.pipeline import run_pipeline

logger = logging.getLogger(__name__)

# Cap cumulative diff at ~12,500 tokens (4 chars/token).  Oldest entries are
# truncated first so the most recent changes remain visible to the model.
_MAX_CUMULATIVE_DIFF_CHARS = 50_000


def _cap_cumulative_diff(diff: str, *, max_chars: int = _MAX_CUMULATIVE_DIFF_CHARS) -> str:
    """Truncate cumulative diff to prevent unbounded context growth."""
    from clean_room_agent.orchestrator.git_ops import truncate_diff
    return truncate_diff(diff, max_chars)


def _resolve_budget(config: dict, role: str) -> BudgetConfig:
    """Resolve budget for a given model role. Hard error if missing."""
    models_config = require_models_config(config)
    router = ModelRouter(models_config)
    model_config = router.resolve(role)
    budget_config = require_config_section(config, "budget")
    rt = budget_config.get("reserved_tokens")
    if rt is None:
        raise RuntimeError(
            "Missing reserved_tokens in [budget] section of config.toml."
        )
    return BudgetConfig(context_window=model_config.context_window, reserved_tokens=rt)


def _resolve_stages(config: dict) -> list[str]:
    """Resolve stage names from config. Hard error if missing."""
    stages_config = require_config_section(config, "stages")
    default_stages = stages_config.get("default")
    if not default_stages:
        raise RuntimeError(
            "Missing default in [stages] section of config.toml."
        )
    return [s.strip() for s in default_stages.split(",")]


def _flush_llm_calls(
    llm: LoggedLLMClient, raw_conn, task_id: str, call_type: str, stage_name: str,
    trace_logger: "TraceLogger | None" = None,
) -> tuple[int | None, int | None, int]:
    """Flush LLM call records to raw DB.

    Returns (total_prompt_tokens, total_completion_tokens, total_latency_ms).
    Token counts are None if any individual call reported None.
    """
    prompt_tokens_list: list[int | None] = []
    completion_tokens_list: list[int | None] = []
    total_latency = 0
    calls = llm.flush()
    if trace_logger is not None:
        trace_logger.log_calls(stage_name, call_type, calls, llm.config.model)
    for call in calls:
        insert_retrieval_llm_call(
            raw_conn, task_id, call_type, llm.config.model,
            call["prompt"], call["response"],
            call["prompt_tokens"], call["completion_tokens"],
            call["elapsed_ms"],
            stage_name=stage_name,
            system_prompt=call["system"],
            thinking=call.get("thinking"),
        )
        prompt_tokens_list.append(call["prompt_tokens"])
        completion_tokens_list.append(call["completion_tokens"])
        total_latency += call["elapsed_ms"]
    raw_conn.commit()
    return (
        _accumulate_optional_tokens(prompt_tokens_list),
        _accumulate_optional_tokens(completion_tokens_list),
        total_latency,
    )


def _write_temp_plan(affected_files: list[str], tmp_dir: Path) -> Path:
    """Write a minimal plan JSON for Tier 0 seeding in retrieval pipeline."""
    plan_data = {"affected_files": [{"path": f} for f in affected_files]}
    tmp_file = tmp_dir / f"plan_{uuid.uuid4().hex[:8]}.json"
    tmp_file.write_text(json.dumps(plan_data), encoding="utf-8")
    return tmp_file


def _get_task_run_id(conn, task_id: str) -> int:
    """Look up task_run_id from raw DB. Raises RuntimeError if not found."""
    row = conn.execute(
        "SELECT id FROM task_runs WHERE task_id = ?", (task_id,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"task_run not found for {task_id}")
    return row["id"]


def _build_diff_text(edits) -> str:
    """Build unified-diff-style text from a list of PatchEdit objects."""
    parts = []
    for edit in edits:
        parts.append(f"--- {edit.file_path}\n-{edit.search}\n+{edit.replacement}\n")
    return "".join(parts)


def _update_cumulative_diff(
    git: GitWorkflow | None,
    current_diff: str,
    edits,
    commit_msg: str,
    max_diff_chars: int,
) -> tuple[str, str | None]:
    """Commit edits (if git enabled) and update cumulative diff.

    Returns (new_cumulative_diff, commit_sha).
    commit_sha is None when git is disabled.
    """
    if git is not None:
        sha = git.commit_checkpoint(commit_msg)
        new_diff = git.get_cumulative_diff(max_chars=max_diff_chars)
        return new_diff, sha
    new_diff = _cap_cumulative_diff(
        current_diff + _build_diff_text(edits),
        max_chars=max_diff_chars,
    )
    return new_diff, None


def _accumulate_optional_tokens(values: list[int | None]) -> int | None:
    """Sum token counts, returning None if any individual value is None."""
    total = 0
    for v in values:
        if v is None:
            return None
        total += v
    return total


def _archive_session(raw_conn, repo_path: Path, task_id: str) -> None:
    """Archive session DB to raw DB and delete the file.

    Steps 1-2 (read + insert) are critical — failure propagates.
    Step 3 (delete) is best-effort — archive already succeeded.
    """
    session_file = _db_path(repo_path, "session", task_id)
    if not session_file.exists():
        return

    # Step 1: Read file — failure means broken archival
    try:
        session_blob = session_file.read_bytes()
    except OSError as e:
        raise RuntimeError(
            f"Failed to read session DB for archival: {session_file}"
        ) from e

    # Step 2: Insert into raw DB — failure means audit trail loss
    try:
        insert_session_archive(raw_conn, task_id, session_blob)
        raw_conn.commit()
    except sqlite3.Error as e:
        raise RuntimeError(
            f"Failed to insert session archive into raw DB for task {task_id}"
        ) from e

    # Step 3: Delete file — best-effort, archive already succeeded
    try:
        session_file.unlink()
        logger.debug("Archived and removed session DB for task %s", task_id)
    except OSError as e:
        logger.warning("Session archived but failed to delete file: %s", e)


def _finalize_orchestrator_run(
    raw_conn, orch_run_id: int, status: str,
    error_message: str | None = None, **kwargs,
) -> None:
    """Update orchestrator run record with final status. Failure propagates."""
    now = datetime.now(timezone.utc).isoformat()
    try:
        update_orchestrator_run(
            raw_conn, orch_run_id,
            status=status,
            completed_at=now,
            error_message=error_message,
            **kwargs,
        )
        raw_conn.commit()
    except Exception as e:
        raise RuntimeError(
            f"Failed to update orchestrator run {orch_run_id} to status={status!r}"
        ) from e


def _git_cleanup(git, status: str) -> None:
    """Run git end-of-task cleanup: rollback+delete on failure, merge+delete on success.

    Critical operations (rollback, return-to-original, merge) propagate exceptions.
    Branch deletion is best-effort — logged warning on failure.
    """
    if status != "complete":
        # Failure path: rollback, clean, return — all critical, let propagate
        git.rollback_to_checkpoint()
        git.clean_untracked()
        git.return_to_original_branch()
        # Branch delete is best-effort
        try:
            git.delete_task_branch()
        except Exception as e:
            logger.warning("Failed to delete task branch after rollback: %s", e)
    else:
        # Success path: merge is critical, let propagate
        merged = git.merge_to_original()
        if merged:
            # Branch delete is best-effort
            try:
                git.delete_task_branch()
            except Exception as e:
                logger.warning("Failed to delete task branch after merge: %s", e)


def _rollback_part(
    *,
    git: GitWorkflow | None,
    repo_path: Path,
    part_id: str,
    part_start_sha: str | None,
    code_patches: list,
    doc_patches: list,
    test_patches: list,
    raw_conn,
    task_id: str,
) -> None:
    """Rollback all patches for a part after validation failure (A6).

    Uses git reset if git is enabled, otherwise LIFO rollback (test → doc → code).
    Marks affected run_attempts as patch_applied=False in raw DB.
    """
    if git is not None:
        git.rollback_to_checkpoint(commit_sha=part_start_sha)
    else:
        rollback_errors = []
        for tp in reversed(test_patches):
            try:
                rollback_edits(tp, repo_path)
            except (RuntimeError, OSError) as e:
                rollback_errors.append(str(e))
        for dp in reversed(doc_patches):
            try:
                rollback_edits(dp, repo_path)
            except (RuntimeError, OSError) as e:
                rollback_errors.append(str(e))
        for cp in reversed(code_patches):
            try:
                rollback_edits(cp, repo_path)
            except (RuntimeError, OSError) as e:
                rollback_errors.append(str(e))

    # A6: Mark affected run_attempts as rolled back in raw DB
    # Must happen before any raise — DB should reflect reality even on partial failure
    mark_part_attempts_rolled_back(raw_conn, task_id, part_id)
    raw_conn.commit()
    logger.info("Rollback: marked run_attempts as rolled back for part %s", part_id)

    if git is None and rollback_errors:
        raise RuntimeError(f"Rollback partially failed: {rollback_errors}")


def _topological_sort(items, get_id, get_deps):
    """Topological sort items by depends_on. Falls back to original order on cycles."""
    try:
        id_to_item = {get_id(item): item for item in items}
    except (AttributeError, TypeError, KeyError) as e:
        # Find the offending item for a useful error message
        for item in items:
            try:
                get_id(item)
            except (AttributeError, TypeError, KeyError):
                raise RuntimeError(
                    f"_topological_sort: get_id failed on item {item!r}: {e}"
                ) from e
        raise  # pragma: no cover — shouldn't reach here
    in_degree = {get_id(item): 0 for item in items}
    adjacency = {get_id(item): [] for item in items}
    valid_ids = set(id_to_item.keys())

    for item in items:
        try:
            deps = get_deps(item)
        except (AttributeError, TypeError, KeyError) as e:
            raise RuntimeError(
                f"_topological_sort: get_deps failed on item {item!r}: {e}"
            ) from e
        for dep in deps:
            if dep in valid_ids:
                adjacency[dep].append(get_id(item))
                in_degree[get_id(item)] += 1

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    result = []
    while queue:
        node = queue.popleft()
        result.append(id_to_item[node])
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(items):
        raise RuntimeError(
            "Cycle detected in dependency graph — validate_plan should have "
            "caught this. This indicates a bug or unvalidated plan input."
        )
    return result


@dataclass
class _OrchestratorContext:
    """Shared state for orchestrator helpers."""

    task_id: str
    repo_path: Path
    config: dict
    raw_conn: sqlite3.Connection
    session_conn: sqlite3.Connection
    git: GitWorkflow | None
    trace_logger: TraceLogger | None
    orch_run_id: int
    sequence_order: int
    cumulative_diff: str
    max_diff_chars: int
    pass_results: list[PassResult]
    stage_names: list[str]
    impl_budget: BudgetConfig
    coding_config: ModelConfig
    plan_budget: BudgetConfig | None
    reasoning_config: ModelConfig | None
    tmp_dir: Path
    router: ModelRouter
    max_retries: int
    max_test_retries: int
    max_adj_rounds: int
    doc_pass_enabled: bool
    scaffold_enabled: bool
    scaffold_compiler: str
    scaffold_compiler_flags: str


def _init_orchestrator(
    task: str,
    repo_path: Path,
    config: dict,
    trace_logger: TraceLogger | None,
    *,
    needs_reasoning: bool = True,
) -> _OrchestratorContext:
    """Create orchestrator context with config resolution, DB, and git.

    Config validation errors raise immediately (fail-fast).
    Caller must use try/finally with _cleanup_orchestrator().
    """
    task_id = str(uuid.uuid4())
    if trace_logger is not None:
        trace_logger.update_task_id(task_id)

    models_config = require_models_config(config)
    router = ModelRouter(models_config)
    coding_config = router.resolve("coding")
    reasoning_config = router.resolve("reasoning") if needs_reasoning else None
    stage_names = _resolve_stages(config)

    orch_config = require_config_section(config, "orchestrator")
    max_retries = orch_config.get("max_retries_per_step")
    if max_retries is None:
        raise RuntimeError(
            "Missing max_retries_per_step in [orchestrator] section of config.toml."
        )
    max_test_retries = orch_config.get("max_retries_per_test_step")
    if max_test_retries is None:
        max_test_retries = max_retries
        logger.info(
            "max_retries_per_test_step not set — using max_retries_per_step (%d)",
            max_retries,
        )

    # A11: require max_adjustment_rounds — no hardcoded fallback
    max_adj_rounds = orch_config.get("max_adjustment_rounds")
    if max_adj_rounds is None:
        raise RuntimeError(
            "Missing max_adjustment_rounds in [orchestrator] section of config.toml."
        )

    # A11: require max_cumulative_diff_chars — no hardcoded fallback
    max_diff_chars = orch_config.get("max_cumulative_diff_chars")
    if max_diff_chars is None:
        raise RuntimeError(
            "Missing max_cumulative_diff_chars in [orchestrator] section of config.toml."
        )
    # T64: bounds-check
    if not isinstance(max_diff_chars, int) or max_diff_chars <= 0:
        raise RuntimeError(
            f"max_cumulative_diff_chars must be a positive integer, got {max_diff_chars!r}"
        )

    # documentation_pass: intentional default True — cosmetic feature, not behavioral
    doc_pass_enabled = orch_config.get("documentation_pass", True)

    # scaffold: Optional, default false. When true and target is C, scaffold pass runs.
    scaffold_enabled = orch_config.get("scaffold_enabled", False)
    scaffold_compiler = orch_config.get("scaffold_compiler", "gcc")
    scaffold_compiler_flags = orch_config.get("scaffold_compiler_flags", "-c -fsyntax-only -Wall")

    # Fail-fast: if scaffold enabled but compiler not found, error at init time
    if scaffold_enabled:
        import shutil as _shutil
        if _shutil.which(scaffold_compiler) is None:
            raise RuntimeError(
                f"scaffold_enabled=true but compiler {scaffold_compiler!r} not found on PATH. "
                f"Install it or set scaffold_enabled = false in [orchestrator] config."
            )

    use_git = orch_config.get("git_workflow")
    if use_git is None:
        raise RuntimeError(
            "Missing git_workflow in [orchestrator] section of config.toml."
        )

    impl_budget = _resolve_budget(config, "coding")
    plan_budget = _resolve_budget(config, "reasoning") if needs_reasoning else None

    raw_conn = get_connection("raw", repo_path=repo_path)
    session_conn = get_connection("session", repo_path=repo_path, task_id=task_id)

    tmp_dir = repo_path / ".clean_room" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    git = None
    if use_git and (repo_path / ".git").exists():
        from clean_room_agent.orchestrator.git_ops import GitWorkflow as _GitWorkflow

        git = _GitWorkflow(repo_path, task_id)
        git.create_task_branch()
    elif use_git:
        logger.info(
            "git_workflow enabled but no .git directory found — falling back to LIFO"
        )

    orch_run_id = insert_orchestrator_run(
        raw_conn, task_id, str(repo_path), task,
        git_branch=git.branch_name if git else None,
        git_base_ref=git.base_ref if git else None,
    )
    raw_conn.commit()

    return _OrchestratorContext(
        task_id=task_id,
        repo_path=repo_path,
        config=config,
        raw_conn=raw_conn,
        session_conn=session_conn,
        git=git,
        trace_logger=trace_logger,
        orch_run_id=orch_run_id,
        sequence_order=0,
        cumulative_diff="",
        max_diff_chars=max_diff_chars,
        pass_results=[],
        stage_names=stage_names,
        impl_budget=impl_budget,
        coding_config=coding_config,
        plan_budget=plan_budget,
        reasoning_config=reasoning_config,
        tmp_dir=tmp_dir,
        router=router,
        max_retries=max_retries,
        max_test_retries=max_test_retries,
        max_adj_rounds=max_adj_rounds,
        doc_pass_enabled=doc_pass_enabled,
        scaffold_enabled=scaffold_enabled,
        scaffold_compiler=scaffold_compiler,
        scaffold_compiler_flags=scaffold_compiler_flags,
    )


def _cleanup_orchestrator(
    ctx: _OrchestratorContext,
    status: str,
    *,
    error_message: str | None = None,
    cumulative_diff: str = "",
    parts_completed: int = 0,
    steps_completed: int = 0,
    total_parts: int | None = None,
    total_steps: int = 0,
) -> None:
    """Run orchestrator finally block: finalize, session, git, archive, close."""
    _finalize_orchestrator_run(
        ctx.raw_conn, ctx.orch_run_id, status,
        error_message=error_message,
        total_parts=total_parts,
        total_steps=total_steps,
        parts_completed=parts_completed,
        steps_completed=steps_completed,
    )

    set_state(ctx.session_conn, "cumulative_diff", cumulative_diff)
    set_state(ctx.session_conn, "orchestrator_progress", {
        "parts_completed": parts_completed,
        "steps_completed": steps_completed,
        "status": status,
    })
    ctx.session_conn.close()

    if ctx.git is not None:
        _git_cleanup(ctx.git, status)

    _archive_session(ctx.raw_conn, ctx.repo_path, ctx.task_id)
    ctx.raw_conn.close()

    try:
        for f in ctx.tmp_dir.glob("plan_*.json"):
            f.unlink()
    except Exception as cleanup_err:
        logger.warning(
            "Failed to clean up temp files in %s: %s", ctx.tmp_dir, cleanup_err
        )


def run_orchestrator(
    task: str, repo_path: Path, config: dict,
    trace_logger: "TraceLogger | None" = None,
) -> OrchestratorResult:
    """Run the full orchestrator loop.

    meta_plan -> parts (topo-sorted) -> steps -> implement -> test -> retry -> adjust
    """
    ctx = _init_orchestrator(task, repo_path, config, trace_logger, needs_reasoning=True)

    # Unpack context for body code
    task_id = ctx.task_id
    raw_conn = ctx.raw_conn
    session_conn = ctx.session_conn
    git = ctx.git
    orch_run_id = ctx.orch_run_id
    stage_names = ctx.stage_names
    plan_budget = ctx.plan_budget
    impl_budget = ctx.impl_budget
    reasoning_config = ctx.reasoning_config
    coding_config = ctx.coding_config
    max_retries = ctx.max_retries
    max_test_retries = ctx.max_test_retries
    max_adj_rounds = ctx.max_adj_rounds
    max_diff_chars = ctx.max_diff_chars
    doc_pass_enabled = ctx.doc_pass_enabled
    scaffold_enabled = ctx.scaffold_enabled
    scaffold_compiler = ctx.scaffold_compiler
    scaffold_compiler_flags = ctx.scaffold_compiler_flags
    tmp_dir = ctx.tmp_dir
    pass_results = ctx.pass_results

    step_final_outcomes: dict[str, bool] = {}  # step_key -> final success
    adjustment_counts: dict[str, int] = {}  # part_id -> adjustment count
    cumulative_diff = ""
    parts_completed = 0
    steps_completed = 0
    all_steps_count = 0
    sequence_order = 0
    status = "failed"
    error_msg: str | None = None

    try:
        # === META-PLAN PASS ===
        sub_task_id = f"{task_id}:meta_plan"
        context = run_pipeline(
            raw_task=task,
            repo_path=repo_path,
            stage_names=stage_names,
            budget=plan_budget,
            mode="plan",
            task_id=sub_task_id,
            config=config,
            trace_logger=trace_logger,
        )

        with LoggedLLMClient(reasoning_config) as llm:
            meta_plan = execute_plan(
                context, task, llm,
                pass_type="meta_plan",
            )
            _flush_llm_calls(llm, raw_conn, sub_task_id, "execute_plan", "execute_plan", trace_logger)

        meta_task_run_id = _get_task_run_id(raw_conn, sub_task_id)

        insert_orchestrator_pass(
            raw_conn, orch_run_id, meta_task_run_id, "meta_plan", sequence_order,
        )
        raw_conn.commit()
        sequence_order += 1

        pass_results.append(PassResult(
            pass_type="meta_plan", task_run_id=meta_task_run_id,
            success=True, artifact=meta_plan,
        ))

        set_state(session_conn, "meta_plan", meta_plan.to_dict())

        update_orchestrator_run(
            raw_conn, orch_run_id,
            total_parts=len(meta_plan.parts),
            status="running",
        )
        raw_conn.commit()

        # === PART LOOP (topological order) ===
        sorted_parts = _topological_sort(
            meta_plan.parts, lambda p: p.id, lambda p: p.depends_on,
        )

        for part in sorted_parts:
            part_start_sha = git.get_head_sha() if git else None

            # --- PART-PLAN PASS ---
            temp_plan_path = _write_temp_plan(part.affected_files, tmp_dir)
            sub_task_id = f"{task_id}:part_plan:{part.id}"

            try:
                part_context = run_pipeline(
                    raw_task=part.description,
                    repo_path=repo_path,
                    stage_names=stage_names,
                    budget=plan_budget,
                    mode="plan",
                    task_id=sub_task_id,
                    config=config,
                    plan_artifact_path=temp_plan_path,
                    trace_logger=trace_logger,
                )

                with LoggedLLMClient(reasoning_config) as llm:
                    part_plan = execute_plan(
                        part_context, part.description, llm,
                        pass_type="part_plan",
                        cumulative_diff=cumulative_diff or None,
                    )
                    _flush_llm_calls(llm, raw_conn, sub_task_id, "execute_plan", "execute_plan", trace_logger)

                part_task_run_id = _get_task_run_id(raw_conn, sub_task_id)

                insert_orchestrator_pass(
                    raw_conn, orch_run_id, part_task_run_id, "part_plan", sequence_order,
                    part_id=part.id,
                )
                raw_conn.commit()
                sequence_order += 1

                pass_results.append(PassResult(
                    pass_type="part_plan", task_run_id=part_task_run_id,
                    success=True, artifact=part_plan,
                ))

                set_state(session_conn, f"part_plan:{part.id}", part_plan.to_dict())

            except (ValueError, RuntimeError, OSError) as e:
                logger.error("Part plan failed for %s: %s", part.id, e)
                pass_results.append(PassResult(
                    pass_type="part_plan", task_run_id=None, success=False,
                ))
                continue

            # --- SCAFFOLD PASS (C projects only) ---
            scaffold_active = False
            if scaffold_enabled and is_c_part(part_plan):
                logger.info("Scaffold pass: C part detected for %s", part.id)
                sub_task_id = f"{task_id}:scaffold:{part.id}"

                scaffold_context = run_pipeline(
                    raw_task=part.description,
                    repo_path=repo_path,
                    stage_names=stage_names,
                    budget=plan_budget,
                    mode="plan",
                    task_id=sub_task_id,
                    config=config,
                    trace_logger=trace_logger,
                )

                with LoggedLLMClient(reasoning_config) as scaffold_llm:
                    scaffold_result = execute_scaffold(
                        scaffold_context, part_plan, scaffold_llm,
                        cumulative_diff=cumulative_diff or None,
                    )
                    _flush_llm_calls(
                        scaffold_llm, raw_conn, sub_task_id,
                        "execute_scaffold", "execute_scaffold", trace_logger,
                    )

                # execute_scaffold raises ValueError on parse failure — no success check.
                if not scaffold_result.edits:
                    raise RuntimeError(
                        f"Scaffold generation produced no edits for part {part.id}"
                    )

                # Apply scaffold edits (raises on failure — no success flag check needed)
                scaffold_patch = apply_edits(scaffold_result.edits, repo_path)

                # Validate compilation
                all_scaffold_files = scaffold_result.header_files + scaffold_result.source_files
                compiled, comp_output = validate_scaffold_compilation(
                    all_scaffold_files, repo_path,
                    compiler=scaffold_compiler,
                    flags=scaffold_compiler_flags,
                )
                scaffold_result.compilation_output = comp_output

                if not compiled:
                    rollback_edits(scaffold_patch, repo_path)
                    raise RuntimeError(
                        f"Scaffold compilation failed for part {part.id}:\n{comp_output}"
                    )

                logger.info("Scaffold compiled successfully for %s", part.id)

                # Extract function stubs via tree-sitter
                stubs = extract_function_stubs(
                    scaffold_result.source_files, repo_path,
                )
                scaffold_result.function_stubs = stubs
                if not stubs:
                    raise RuntimeError(
                        f"Scaffold compiled but no function stubs extracted for part {part.id}"
                    )

                scaffold_active = True
                logger.info(
                    "Scaffold extracted %d function stubs for %s",
                    len(stubs), part.id,
                )

                scaffold_task_run_id = _get_task_run_id(raw_conn, sub_task_id)
                insert_orchestrator_pass(
                    raw_conn, orch_run_id, scaffold_task_run_id, "scaffold",
                    sequence_order, part_id=part.id,
                )
                raw_conn.commit()
                sequence_order += 1

                pass_results.append(PassResult(
                    pass_type="scaffold",
                    task_run_id=scaffold_task_run_id,
                    success=True,
                    artifact=scaffold_result,
                ))

                set_state(session_conn, f"scaffold:{part.id}", scaffold_result.to_dict())

            # --- CODE STEP LOOP or PER-FUNCTION IMPLEMENT (within part) ---
            # Track applied code patches for LIFO rollback
            code_patch_results = []
            code_step_results = []
            all_code_steps_ok = True

            if scaffold_active:
                # Per-function implement loop: each stub gets an independent LLM call
                # against the compact scaffold context (~3-5K tokens vs 20K+)
                scaffold_content: dict[str, str] = {}
                for sf in scaffold_result.header_files + scaffold_result.source_files:
                    sf_path = repo_path / sf
                    if sf_path.exists():
                        scaffold_content[sf] = sf_path.read_text(encoding="utf-8", errors="replace")

                for stub_idx, stub in enumerate(scaffold_result.function_stubs):
                    sub_task_id = f"{task_id}:func_impl:{part.id}:{stub.name}"
                    step_key = f"{part.id}:func:{stub.name}"
                    func_result = None

                    try:
                        with LoggedLLMClient(coding_config) as func_llm:
                            func_result = execute_function_implement(
                                stub, scaffold_content, func_llm,
                            )
                            _flush_llm_calls(
                                func_llm, raw_conn, sub_task_id,
                                "execute_function_implement", "execute_function_implement",
                                trace_logger,
                            )

                        # Both raise on failure — no success flag checks.
                        patch_result = apply_edits(func_result.edits, repo_path)
                        code_patch_results.append(patch_result)
                        steps_completed += 1
                        step_final_outcomes[step_key] = True

                        # Update scaffold_content with the new implementation
                        for edit in func_result.edits:
                            if edit.file_path in scaffold_content:
                                sf_path = repo_path / edit.file_path
                                if sf_path.exists():
                                    scaffold_content[edit.file_path] = sf_path.read_text(
                                        encoding="utf-8", errors="replace"
                                    )

                        cumulative_diff, commit_sha = _update_cumulative_diff(
                            git, cumulative_diff, func_result.edits,
                            f"cra: {part.id}:func:{stub.name}",
                            max_diff_chars,
                        )
                    except (ValueError, RuntimeError, OSError) as e:
                        all_code_steps_ok = False
                        step_final_outcomes[step_key] = False
                        logger.error(
                            "Function implement failed for %s: %s",
                            stub.name, e,
                        )

                    if func_result is not None:
                        code_step_results.append(func_result)

                    pass_results.append(PassResult(
                        pass_type="function_implement",
                        task_run_id=None,
                        success=step_final_outcomes.get(step_key, False),
                        artifact=func_result,
                    ))

                all_steps_count += len(scaffold_result.function_stubs)

            else:
                # Normal code step loop (existing behavior)
                sorted_steps = _topological_sort(
                    part_plan.steps, lambda s: s.id, lambda s: s.depends_on,
                )

                # Use index-based iteration so adjustment can splice revised
                # steps into the remaining portion of the list (T25).
                step_idx = 0
                while step_idx < len(sorted_steps):
                    step = sorted_steps[step_idx]
                    step_success = False
                    step_result = None
                    attempt_num = 0

                    for attempt in range(max_retries + 1):
                        attempt_num = attempt + 1

                        # IMPLEMENT
                        suffix = f":retry_{attempt}" if attempt > 0 else ""
                        sub_task_id = f"{task_id}:impl:{part.id}:{step.id}{suffix}"
                        temp_plan_path = _write_temp_plan(step.target_files, tmp_dir)

                        try:
                            impl_context = run_pipeline(
                                raw_task=step.description,
                                repo_path=repo_path,
                                stage_names=stage_names,
                                budget=impl_budget,
                                mode="implement",
                                task_id=sub_task_id,
                                config=config,
                                plan_artifact_path=temp_plan_path,
                                trace_logger=trace_logger,
                            )

                            with LoggedLLMClient(coding_config) as impl_llm:
                                step_result = execute_implement(
                                    impl_context, step, impl_llm,
                                    plan=part_plan,
                                    cumulative_diff=cumulative_diff or None,
                                )
                                impl_pt, impl_ct, impl_ms = _flush_llm_calls(
                                    impl_llm, raw_conn, sub_task_id,
                                    "execute_implement", "execute_implement",
                                    trace_logger,
                                )

                            impl_task_run_id = _get_task_run_id(raw_conn, sub_task_id)

                            # Log run_attempt with actual token counts (T40) and
                            # patch_applied=False; updated after apply succeeds (T27).
                            attempt_id = insert_run_attempt(
                                raw_conn, impl_task_run_id, attempt_num,
                                impl_pt, impl_ct, impl_ms,
                                step_result.raw_response,
                                False,
                            )
                            raw_conn.commit()

                            impl_pass_id = insert_orchestrator_pass(
                                raw_conn, orch_run_id, impl_task_run_id, "step_implement", sequence_order,
                                part_id=part.id, step_id=step.id,
                            )
                            raw_conn.commit()
                            sequence_order += 1

                            # APPLY EDITS (raises on failure — no success flag check needed)
                            patch_result = apply_edits(step_result.edits, repo_path)

                            # Patch actually applied — update the DB record (T27)
                            update_run_attempt_patch(raw_conn, attempt_id, True)
                            raw_conn.commit()

                            step_success = True
                            code_patch_results.append(patch_result)
                            pass_results.append(PassResult(
                                pass_type="step_implement", task_run_id=impl_task_run_id,
                                success=True, artifact=step_result,
                            ))
                            break

                        except (ValueError, RuntimeError, OSError) as e:
                            logger.error("Step implement failed for %s:%s: %s", part.id, step.id, e)
                            pass_results.append(PassResult(
                                pass_type="step_implement", task_run_id=None, success=False,
                            ))
                            # Continue to next retry attempt (don't break the loop)

                    # Track final outcome for this code step
                    step_key = f"{part.id}:{step.id}"
                    step_final_outcomes[step_key] = step_success
                    code_step_results.append(step_result)

                    # Update state
                    if step_success and step_result:
                        cumulative_diff, commit_sha = _update_cumulative_diff(
                            git, cumulative_diff, step_result.edits,
                            f"cra: {part.id}:{step.id} — {step.description[:60].rsplit(' ', 1)[0]}",
                            max_diff_chars,
                        )
                        if commit_sha is not None:
                            update_orchestrator_pass_sha(raw_conn, impl_pass_id, commit_sha)
                            raw_conn.commit()
                        set_state(session_conn, f"step_result:{part.id}:{step.id}", {
                            "success": True, "edits_count": len(step_result.edits),
                        })

                        # Only count successfully completed steps (T26)
                        steps_completed += 1
                    else:
                        all_code_steps_ok = False

                    # ADJUSTMENT PASS (after every code step)
                    adj_count = adjustment_counts.get(part.id, 0)
                    if adj_count >= max_adj_rounds:
                        logger.info(
                            "Adjustment limit reached for part %s (%d/%d)",
                            part.id, adj_count, max_adj_rounds,
                        )
                    else:
                        try:
                            adj_sub_task_id = f"{task_id}:adjust:{part.id}:after_{step.id}"
                            remaining_desc = f"Remaining steps after {step.id} in part {part.id}"
                            adj_context = run_pipeline(
                                raw_task=remaining_desc,
                                repo_path=repo_path,
                                stage_names=stage_names,
                                budget=plan_budget,
                                mode="plan",
                                task_id=adj_sub_task_id,
                                config=config,
                                trace_logger=trace_logger,
                            )

                            with LoggedLLMClient(reasoning_config) as adj_llm:
                                adjustment = execute_plan(
                                    adj_context, remaining_desc, adj_llm,
                                    pass_type="adjustment",
                                    prior_results=[step_result] if step_result else None,
                                    cumulative_diff=cumulative_diff or None,
                                )
                                _flush_llm_calls(adj_llm, raw_conn, adj_sub_task_id, "execute_adjust", "execute_adjust", trace_logger)

                            adj_task_run_id = _get_task_run_id(raw_conn, adj_sub_task_id)

                            insert_orchestrator_pass(
                                raw_conn, orch_run_id, adj_task_run_id, "adjustment", sequence_order,
                                part_id=part.id, step_id=step.id,
                            )
                            raw_conn.commit()
                            sequence_order += 1

                            set_state(session_conn, f"adjustment:{part.id}:after_{step.id}", adjustment.to_dict())

                            # Replace remaining steps with revised ones (T25).
                            # Safe because we use index-based iteration.
                            if adjustment.revised_steps:
                                sorted_steps[step_idx + 1:] = adjustment.revised_steps

                            pass_results.append(PassResult(
                                pass_type="adjustment", task_run_id=adj_task_run_id,
                                success=True, artifact=adjustment,
                            ))

                            adjustment_counts[part.id] = adj_count + 1

                        except (ValueError, RuntimeError, OSError) as e:
                            logger.warning("Adjustment pass failed for %s:after_%s: %s", part.id, step.id, e)
                            # T66: Record adjustment failure in session state
                            set_state(session_conn, f"adjustment:{part.id}:after_{step.id}", {
                                "success": False, "error": str(e),
                            })
                            pass_results.append(PassResult(
                                pass_type="adjustment", success=False,
                            ))

                    step_idx += 1

                all_steps_count += len(part_plan.steps)

            # --- DOCUMENTATION PASS (after code steps, before tests) ---
            doc_patch_results: list[PatchResult] = []

            if doc_pass_enabled and all_code_steps_ok and cumulative_diff:
                try:
                    # Collect unique files modified by code steps
                    doc_modified_files: list[str] = []
                    for cp in code_patch_results:
                        for f in cp.files_modified:
                            if f not in doc_modified_files:
                                doc_modified_files.append(f)

                    doc_model_config = ctx.router.resolve("reasoning", stage_name="documentation")
                    with LoggedLLMClient(doc_model_config) as doc_llm:
                        doc_patch_results = run_documentation_pass(
                            doc_modified_files, repo_path,
                            task, part.description,
                            doc_llm, doc_model_config,
                            part_context.environment_brief or None,
                        )
                        doc_pt, doc_ct, doc_ms = _flush_llm_calls(
                            doc_llm, raw_conn, f"{task_id}:doc:{part.id}",
                            "documentation", "documentation",
                            trace_logger,
                        )

                    # Commit + update cumulative diff after doc edits
                    doc_sha: str | None = None
                    if doc_patch_results:
                        cumulative_diff, doc_sha = _update_cumulative_diff(
                            git, cumulative_diff, [],
                            f"cra: {part.id}:docs", max_diff_chars,
                        )

                    doc_task_run_id = insert_task_run(
                        raw_conn, f"{task_id}:doc:{part.id}",
                        str(repo_path), "documentation",
                        doc_model_config.model,
                        doc_model_config.context_window,
                        doc_model_config.max_tokens,
                        "",
                    )
                    insert_orchestrator_pass(
                        raw_conn, orch_run_id, doc_task_run_id, "documentation", sequence_order,
                        part_id=part.id, commit_sha=doc_sha,
                    )
                    raw_conn.commit()
                    sequence_order += 1

                    pass_results.append(PassResult(
                        pass_type="documentation", success=True,
                    ))

                    set_state(session_conn, f"doc_pass:{part.id}", {
                        "files_processed": len(doc_modified_files),
                        "files_patched": len(doc_patch_results),
                    })

                except (ValueError, RuntimeError, OSError) as e:
                    logger.warning("Documentation pass failed for %s: %s", part.id, e)
                    pass_results.append(PassResult(
                        pass_type="documentation", success=False,
                    ))

            # --- TESTING PHASE (after all code steps for this part) ---
            test_plan = None
            test_patch_results = []

            if all_code_steps_ok and cumulative_diff:
                # TEST PLAN
                try:
                    test_plan_task_id = f"{task_id}:test_plan:{part.id}"
                    test_plan_desc = (
                        f"Plan tests for part {part.id}: {part.description}"
                    )
                    test_affected = []
                    for sr in code_step_results:
                        if sr and sr.success:
                            for edit in sr.edits:
                                if edit.file_path not in test_affected:
                                    test_affected.append(edit.file_path)
                    temp_plan_path = _write_temp_plan(test_affected, tmp_dir)

                    test_plan_context = run_pipeline(
                        raw_task=test_plan_desc,
                        repo_path=repo_path,
                        stage_names=stage_names,
                        budget=plan_budget,
                        mode="plan",
                        task_id=test_plan_task_id,
                        config=config,
                        plan_artifact_path=temp_plan_path,
                        trace_logger=trace_logger,
                    )

                    with LoggedLLMClient(reasoning_config) as tp_llm:
                        test_plan = execute_plan(
                            test_plan_context, test_plan_desc, tp_llm,
                            pass_type="test_plan",
                            cumulative_diff=cumulative_diff or None,
                        )
                        _flush_llm_calls(tp_llm, raw_conn, test_plan_task_id, "execute_plan", "execute_plan", trace_logger)

                    tp_task_run_id = _get_task_run_id(raw_conn, test_plan_task_id)

                    insert_orchestrator_pass(
                        raw_conn, orch_run_id, tp_task_run_id, "test_plan", sequence_order,
                        part_id=part.id,
                    )
                    raw_conn.commit()
                    sequence_order += 1

                    pass_results.append(PassResult(
                        pass_type="test_plan", task_run_id=tp_task_run_id,
                        success=True, artifact=test_plan,
                    ))

                    set_state(session_conn, f"test_plan:{part.id}", test_plan.to_dict())

                except (ValueError, RuntimeError, OSError) as e:
                    logger.error("Test plan failed for %s: %s", part.id, e)
                    pass_results.append(PassResult(
                        pass_type="test_plan", task_run_id=None, success=False,
                    ))
                    test_plan = None

            # TEST STEP LOOP
            if test_plan is not None:
                sorted_test_steps = _topological_sort(
                    test_plan.steps, lambda s: s.id, lambda s: s.depends_on,
                )

                for test_step in sorted_test_steps:
                    test_step_success = False
                    test_step_result = None

                    for test_attempt in range(max_test_retries + 1):
                        suffix = f":retry_{test_attempt}" if test_attempt > 0 else ""
                        test_sub_task_id = f"{task_id}:test_impl:{part.id}:{test_step.id}{suffix}"
                        test_target_files = test_step.target_files or test_affected
                        temp_plan_path = _write_temp_plan(test_target_files, tmp_dir)

                        try:
                            test_impl_context = run_pipeline(
                                raw_task=test_step.description,
                                repo_path=repo_path,
                                stage_names=stage_names,
                                budget=impl_budget,
                                mode="implement",
                                task_id=test_sub_task_id,
                                config=config,
                                plan_artifact_path=temp_plan_path,
                                trace_logger=trace_logger,
                            )

                            with LoggedLLMClient(coding_config) as test_llm:
                                test_step_result = execute_test_implement(
                                    test_impl_context, test_step, test_llm,
                                    test_plan=test_plan,
                                    cumulative_diff=cumulative_diff or None,
                                )
                                ti_pt, ti_ct, ti_ms = _flush_llm_calls(
                                    test_llm, raw_conn, test_sub_task_id,
                                    "execute_test_implement", "execute_test_implement",
                                    trace_logger,
                                )

                            ti_task_run_id = _get_task_run_id(raw_conn, test_sub_task_id)

                            attempt_id = insert_run_attempt(
                                raw_conn, ti_task_run_id, test_attempt + 1,
                                ti_pt, ti_ct, ti_ms,
                                test_step_result.raw_response,
                                False,
                            )
                            raw_conn.commit()

                            ti_pass_id = insert_orchestrator_pass(
                                raw_conn, orch_run_id, ti_task_run_id, "test_implement", sequence_order,
                                part_id=part.id, step_id=test_step.id,
                            )
                            raw_conn.commit()
                            sequence_order += 1

                            # Both raise on failure — no success flag checks.
                            test_patch_result = apply_edits(test_step_result.edits, repo_path)

                            update_run_attempt_patch(raw_conn, attempt_id, True)
                            raw_conn.commit()

                            test_step_success = True
                            test_patch_results.append(test_patch_result)
                            pass_results.append(PassResult(
                                pass_type="test_implement", task_run_id=ti_task_run_id,
                                success=True, artifact=test_step_result,
                            ))
                            break

                        except (ValueError, RuntimeError, OSError) as e:
                            logger.error("Test step implement failed for %s:%s: %s", part.id, test_step.id, e)
                            pass_results.append(PassResult(
                                pass_type="test_implement", task_run_id=None, success=False,
                            ))
                            # Continue to next retry attempt (don't break the loop)

                    step_key = f"{part.id}:test:{test_step.id}"
                    step_final_outcomes[step_key] = test_step_success

                    if test_step_success and test_step_result:
                        cumulative_diff, commit_sha = _update_cumulative_diff(
                            git, cumulative_diff, test_step_result.edits,
                            f"cra: {part.id}:test:{test_step.id} — {test_step.description[:60].rsplit(' ', 1)[0]}",
                            max_diff_chars,
                        )
                        if commit_sha is not None:
                            update_orchestrator_pass_sha(raw_conn, ti_pass_id, commit_sha)
                            raw_conn.commit()
                        set_state(session_conn, f"test_step_result:{part.id}:{test_step.id}", {
                            "success": True, "edits_count": len(test_step_result.edits),
                        })

            # --- VALIDATION (code + doc + tests together) ---
            validation = None
            if code_patch_results or doc_patch_results or test_patch_results:
                # We need an attempt_id for validation logging. Use the last
                # code step's attempt_id if available.
                last_attempt_id = raw_conn.execute(
                    "SELECT id FROM run_attempts ORDER BY id DESC LIMIT 1"
                ).fetchone()
                val_attempt_id = last_attempt_id["id"] if last_attempt_id else None

                if val_attempt_id is not None:
                    validation = run_validation(repo_path, config, raw_conn, val_attempt_id)
                else:
                    validation = run_validation(repo_path, config, raw_conn, 0)

                if not validation.success:
                    _rollback_part(
                        git=git, repo_path=repo_path,
                        part_id=part.id, part_start_sha=part_start_sha,
                        code_patches=code_patch_results,
                        doc_patches=doc_patch_results,
                        test_patches=test_patch_results,
                        raw_conn=raw_conn, task_id=task_id,
                    )

                    # Mark all steps for this part as failed since validation failed
                    for key in list(step_final_outcomes):
                        if key.startswith(f"{part.id}:"):
                            if step_final_outcomes[key]:
                                step_final_outcomes[key] = False
                                steps_completed -= 1

            parts_completed += 1

        # Determine final status from per-step final outcomes (not individual attempts)
        if not step_final_outcomes:
            status = "failed"
        elif all(step_final_outcomes.values()):
            status = "complete"
        elif any(step_final_outcomes.values()):
            status = "partial"
        else:
            status = "failed"

    except (ValueError, RuntimeError, OSError) as e:
        logger.error("Orchestrator failed: %s", e)
        status = "failed"
        error_msg = str(e)

    finally:
        _cleanup_orchestrator(
            ctx, status,
            error_message=error_msg,
            cumulative_diff=cumulative_diff,
            parts_completed=parts_completed,
            steps_completed=steps_completed,
            total_steps=all_steps_count,
        )

    return OrchestratorResult(
        task_id=task_id,
        status=status,
        parts_completed=parts_completed,
        steps_completed=steps_completed,
        cumulative_diff=cumulative_diff,
        pass_results=pass_results,
    )


def run_single_pass(
    task: str,
    repo_path: Path,
    config: dict,
    *,
    plan_path: Path,
    trace_logger: "TraceLogger | None" = None,
) -> OrchestratorResult:
    """Run a single atomic implement pass from a pre-computed plan.

    Creates orchestrator_run/pass DB records and session DB matching
    the full orchestrator path (T67).
    """
    ctx = _init_orchestrator(task, repo_path, config, trace_logger, needs_reasoning=False)

    # Unpack context
    task_id = ctx.task_id
    raw_conn = ctx.raw_conn
    git = ctx.git
    orch_run_id = ctx.orch_run_id
    stage_names = ctx.stage_names
    impl_budget = ctx.impl_budget
    coding_config = ctx.coding_config
    max_retries = ctx.max_retries
    max_diff_chars = ctx.max_diff_chars
    pass_results = ctx.pass_results

    # Load plan artifact
    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    artifact = PlanArtifact.from_dict(plan_data)

    # Create a synthetic step from the plan
    target_files = [f["path"] if isinstance(f, dict) else f for f in artifact.affected_files]
    step = PlanStep(
        id="single_pass",
        description=artifact.task_summary,
        target_files=target_files,
    )

    cumulative_diff = ""
    last_validation = None
    status = "failed"
    error_msg: str | None = None
    steps_completed = 0
    sequence_order = 0

    try:
        for attempt in range(max_retries + 1):
            suffix = f":retry_{attempt}" if attempt > 0 else ""
            sub_task_id = f"{task_id}:impl{suffix}"

            context = run_pipeline(
                raw_task=task,
                repo_path=repo_path,
                stage_names=stage_names,
                budget=impl_budget,
                mode="implement",
                task_id=sub_task_id,
                config=config,
                plan_artifact_path=plan_path,
                trace_logger=trace_logger,
            )

            with LoggedLLMClient(coding_config) as impl_llm:
                step_result = execute_implement(
                    context, step, impl_llm,
                    failure_context=last_validation,
                )
                impl_pt, impl_ct, impl_ms = _flush_llm_calls(
                    impl_llm, raw_conn, sub_task_id,
                    "execute_implement", "execute_implement",
                    trace_logger,
                )

            impl_task_run_id = _get_task_run_id(raw_conn, sub_task_id)

            # Log run_attempt with actual token counts (T40) and
            # patch_applied=False; updated after apply succeeds (T27).
            attempt_id = insert_run_attempt(
                raw_conn, impl_task_run_id, attempt + 1,
                impl_pt, impl_ct, impl_ms,
                step_result.raw_response,
                False,
            )
            raw_conn.commit()

            # T67: Log orchestrator_pass for each attempt
            sp_pass_id = insert_orchestrator_pass(
                raw_conn, orch_run_id, impl_task_run_id, "step_implement", sequence_order,
                step_id="single_pass",
            )
            raw_conn.commit()
            sequence_order += 1

            if not step_result.success:
                pass_results.append(PassResult(
                    pass_type="step_implement", task_run_id=impl_task_run_id,
                    success=False, artifact=step_result,
                ))
                continue

            patch_result = apply_edits(step_result.edits, repo_path)
            if not patch_result.success:
                pass_results.append(PassResult(
                    pass_type="step_implement", task_run_id=impl_task_run_id,
                    success=False, artifact=step_result,
                ))
                continue

            # Patch actually applied — update the DB record (T27)
            update_run_attempt_patch(raw_conn, attempt_id, True)
            raw_conn.commit()

            last_validation = run_validation(repo_path, config, raw_conn, attempt_id)

            if last_validation.success:
                cumulative_diff, commit_sha = _update_cumulative_diff(
                    git, cumulative_diff, step_result.edits,
                    "cra: single_pass", max_diff_chars,
                )
                if commit_sha is not None:
                    update_orchestrator_pass_sha(raw_conn, sp_pass_id, commit_sha)
                    raw_conn.commit()
                pass_results.append(PassResult(
                    pass_type="step_implement", task_run_id=impl_task_run_id,
                    success=True, artifact=step_result,
                ))
                status = "complete"
                steps_completed = 1
                break
            else:
                if git is not None:
                    git.rollback_part()
                else:
                    rollback_edits(patch_result, repo_path)
                # A6: mark this attempt as rolled back
                update_run_attempt_patch(raw_conn, attempt_id, False)
                raw_conn.commit()
                pass_results.append(PassResult(
                    pass_type="step_implement", task_run_id=impl_task_run_id,
                    success=False, artifact=step_result,
                ))

    except (ValueError, RuntimeError, OSError) as e:
        logger.error("Single pass failed: %s", e)
        status = "failed"
        error_msg = str(e)

    finally:
        _cleanup_orchestrator(
            ctx, status,
            error_message=error_msg,
            cumulative_diff=cumulative_diff,
            parts_completed=1 if steps_completed else 0,
            steps_completed=steps_completed,
            total_parts=1,
            total_steps=1,
        )

    return OrchestratorResult(
        task_id=task_id,
        status=status,
        parts_completed=1 if steps_completed else 0,
        steps_completed=steps_completed,
        cumulative_diff=cumulative_diff,
        pass_results=pass_results,
    )
