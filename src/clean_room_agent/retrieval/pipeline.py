"""Pipeline runner: orchestrates task analysis, retrieval stages, and context assembly."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from clean_room_agent.config import require_models_config

if TYPE_CHECKING:
    from clean_room_agent.trace import TraceLogger
from clean_room_agent.environment import build_environment_brief, build_repo_file_tree
from clean_room_agent.db.connection import _db_path, get_connection
from clean_room_agent.db.raw_queries import (
    insert_retrieval_decision,
    insert_retrieval_llm_call,
    insert_session_archive,
    insert_task_run,
    update_task_run,
)
from clean_room_agent.db.session_helpers import get_state, set_state
from clean_room_agent.llm.client import EnvironmentLLMClient, LoggedLLMClient
from clean_room_agent.llm.router import ModelRouter
from clean_room_agent.query.api import KnowledgeBase
from clean_room_agent.retrieval.context_assembly import assemble_context
from clean_room_agent.retrieval.dataclasses import (
    BudgetConfig,
    ContextPackage,
    RefinementRequest,
    ScopedFile,
    TaskQuery,
)
from clean_room_agent.retrieval.preflight import run_preflight_checks
from clean_room_agent.retrieval.routing import route_stages
from clean_room_agent.retrieval.stage import StageContext, get_stage, get_stage_descriptions
from clean_room_agent.retrieval.task_analysis import analyze_task

logger = logging.getLogger(__name__)


def run_pipeline(
    raw_task: str,
    repo_path: Path,
    stage_names: list[str],
    budget: BudgetConfig,
    mode: str,
    task_id: str,
    config: dict,
    plan_artifact_path: Path | None = None,
    refinement_request: RefinementRequest | None = None,
    trace_logger: "TraceLogger | None" = None,
) -> ContextPackage:
    """Run the full retrieval pipeline.

    1. Preflight checks
    2. Validate stages
    3. Open DB connections
    4. Task analysis (or resume from session for refinement)
    5. Stage loop
    6. Context assembly
    7. Logging and cleanup
    """
    # 1. Preflight
    run_preflight_checks(config, repo_path, mode)

    # 2. Validate all stage names and instantiate once
    stages = {name: get_stage(name) for name in stage_names}

    models_config = require_models_config(config)
    router = ModelRouter(models_config)

    # Resolve execute model for logging — plan mode uses reasoning, implement uses coding
    execute_model_config = router.resolve("reasoning" if mode == "plan" else "coding")

    # 3. Open connections
    # T17: Pipeline writes raw DB (audit trail) and session DB (ephemeral).
    # Curated DB is read-only here — no cross-DB atomicity risk in Phase 2/3.
    curated_conn = get_connection("curated", repo_path=repo_path, read_only=True)
    raw_conn = get_connection("raw", repo_path=repo_path)

    # For refinement re-entry, restore session from the source task's archive
    # before opening the session connection (which would create an empty one).
    if refinement_request is not None:
        _restore_session_from_archive(raw_conn, repo_path, task_id, refinement_request.source_task_id)

    session_conn = get_connection("session", repo_path=repo_path, task_id=task_id)

    total_start = time.monotonic()
    task_run_id = None

    try:
        kb = KnowledgeBase(curated_conn)

        # 5. Resolve repo_id
        repo_row = curated_conn.execute(
            "SELECT id FROM repos WHERE path = ?", (str(repo_path),)
        ).fetchone()
        if not repo_row:
            raise RuntimeError(
                f"No indexed repo at {repo_path}. Run 'cra index' first."
            )
        repo_id = repo_row["id"]

        # 5b. Compute environment brief + repo file tree
        env_brief = build_environment_brief(config, kb, repo_id)
        repo_file_tree = build_repo_file_tree(kb, repo_id)
        brief_text = env_brief.to_prompt_text()

        # 6. Log task_run
        task_run_id = insert_task_run(
            raw_conn,
            task_id=task_id,
            repo_path=str(repo_path),
            mode=mode,
            execute_model=execute_model_config.model,
            context_window=budget.context_window,
            reserved_tokens=budget.reserved_tokens,
            stages=",".join(stage_names),
            plan_artifact=str(plan_artifact_path) if plan_artifact_path else None,
        )
        raw_conn.commit()

        # 7. Task analysis
        if refinement_request is not None:
            # Resume from session — log the deterministic merge decision
            task_query = _resume_task_from_session(session_conn, refinement_request, raw_task, task_id, mode, repo_id)
            insert_retrieval_llm_call(
                raw_conn, task_id, "refinement_merge", "deterministic",
                prompt=json.dumps({
                    "refinement_request": {
                        "missing_files": refinement_request.missing_files,
                        "missing_symbols": refinement_request.missing_symbols,
                        "error_patterns": refinement_request.error_patterns,
                    },
                }),
                response=json.dumps({
                    "intent_summary": task_query.intent_summary,
                    "task_type": task_query.task_type,
                    "mentioned_files": task_query.mentioned_files,
                    "mentioned_symbols": task_query.mentioned_symbols,
                    "keywords": task_query.keywords,
                    "error_patterns": task_query.error_patterns,
                }),
                prompt_tokens=None,
                completion_tokens=None,
                latency_ms=0,
                stage_name="task_analysis",
            )
            raw_conn.commit()
            # Log to trace so the refinement merge is visible in the trace output
            if trace_logger is not None:
                trace_logger.log_calls("task_analysis", "refinement_merge", [{
                    "system": None,
                    "prompt": f"refinement_merge: {refinement_request.missing_files}, {refinement_request.missing_symbols}",
                    "response": f"intent={task_query.intent_summary}, type={task_query.task_type}",
                    "thinking": "",
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "elapsed_ms": 0,
                    "error": "",
                }], model="deterministic")
        else:
            reasoning_config = router.resolve("reasoning", "task_analysis")
            with LoggedLLMClient(reasoning_config) as llm:
                start = time.monotonic()
                task_query = analyze_task(
                    raw_task, task_id, mode, kb, repo_id, llm,
                    repo_file_tree=repo_file_tree,
                    environment_brief=brief_text,
                    retrieval_params=config.get("retrieval", {}),
                )
                elapsed = int((time.monotonic() - start) * 1000)

                calls = llm.flush()
                if trace_logger is not None:
                    trace_logger.log_calls("task_analysis", "task_analysis", calls, reasoning_config.model)
                for call in calls:
                    insert_retrieval_llm_call(
                        raw_conn, task_id, "task_analysis", reasoning_config.model,
                        call["prompt"], call["response"],
                        call["prompt_tokens"], call["completion_tokens"],
                        call["elapsed_ms"],
                        stage_name="task_analysis",
                        system_prompt=call["system"],
                        thinking=call.get("thinking"),
                    )
                raw_conn.commit()

            set_state(session_conn, "task_query", {
                "raw_task": task_query.raw_task,
                "intent_summary": task_query.intent_summary,
                "task_type": task_query.task_type,
                "mentioned_files": task_query.mentioned_files,
                "mentioned_symbols": task_query.mentioned_symbols,
                "keywords": task_query.keywords,
                "seed_file_ids": task_query.seed_file_ids,
                "seed_symbol_ids": task_query.seed_symbol_ids,
                "error_patterns": task_query.error_patterns,
            })

        # 7b. Stage routing
        available_for_routing = {
            name: desc for name, desc in get_stage_descriptions().items()
            if name in stages  # only stages the config authorized
        }
        routing_config = router.resolve("reasoning", "stage_routing")
        with LoggedLLMClient(routing_config) as base_routing_llm:
            routing_llm = EnvironmentLLMClient(base_routing_llm, brief_text)
            selected, reasoning = route_stages(
                task_query, available_for_routing, routing_llm,
            )

            calls = routing_llm.flush()
            if trace_logger is not None:
                trace_logger.log_calls("stage_routing", "stage_routing", calls, routing_config.model)
            for call in calls:
                insert_retrieval_llm_call(
                    raw_conn, task_id, "stage_routing", routing_config.model,
                    call["prompt"], call["response"],
                    call["prompt_tokens"], call["completion_tokens"],
                    call["elapsed_ms"],
                    stage_name="stage_routing",
                    system_prompt=call["system"],
                    thinking=call.get("thinking"),
                )
            raw_conn.commit()

        set_state(session_conn, "routing_decision", {
            "selected_stages": selected,
            "reasoning": reasoning,
            "available_stages": list(available_for_routing.keys()),
        })

        stage_names = selected
        stages = {name: stages[name] for name in selected}

        # 8. Plan artifact
        plan_file_ids = []
        if plan_artifact_path and plan_artifact_path.exists():
            plan_data = json.loads(plan_artifact_path.read_text(encoding="utf-8"))
            affected_files = plan_data.get("affected_files", [])
            for fp in affected_files:
                fp_path = fp["path"] if isinstance(fp, dict) else fp
                f = kb.get_file_by_path(repo_id, fp_path)
                if f:
                    plan_file_ids.append(f.id)

        # Initialize stage context
        context = StageContext(
            task=task_query,
            repo_id=repo_id,
            repo_path=str(repo_path),
        )
        context.retrieval_params = config.get("retrieval", {})
        # Seed tier 0 files from plan
        for fid in plan_file_ids:
            f = kb.get_file_by_id(fid)
            if f:
                context.scoped_files.append(ScopedFile(
                    file_id=fid, path=f.path, language=f.language,
                    tier=0, relevance="relevant", reason="plan artifact",
                ))

        # 9. Stage loop
        logged_file_ids: set[int] = set()
        logged_symbol_ids: set[int] = set()
        for stage_name in stage_names:
            stage = stages[stage_name]
            stage_config = router.resolve("reasoning", stage_name)

            with LoggedLLMClient(stage_config) as base_llm:
                llm = EnvironmentLLMClient(base_llm, brief_text)
                start = time.monotonic()
                context = stage.run(context, kb, task_query, llm)
                elapsed = int((time.monotonic() - start) * 1000)
                context.stage_timings[stage_name] = elapsed

                calls = llm.flush()
                if trace_logger is not None:
                    trace_logger.log_calls(stage_name, stage_name, calls, stage_config.model)
                for call in calls:
                    insert_retrieval_llm_call(
                        raw_conn, task_id, stage_name, stage_config.model,
                        call["prompt"], call["response"],
                        call["prompt_tokens"], call["completion_tokens"],
                        call["elapsed_ms"],
                        stage_name=stage_name,
                        system_prompt=call["system"],
                        thinking=call.get("thinking"),
                    )

            # Log decisions only for files new or changed in this stage
            for sf in context.scoped_files:
                if sf.file_id not in logged_file_ids:
                    insert_retrieval_decision(
                        raw_conn, task_id, stage_name, sf.file_id,
                        included=(sf.relevance == "relevant"),
                        tier=str(sf.tier), reason=sf.reason,
                    )
                    logged_file_ids.add(sf.file_id)

            # T16: Log symbol-level decisions if precision stage populated them
            for sym in context.classified_symbols:
                if sym.symbol_id not in logged_symbol_ids:
                    insert_retrieval_decision(
                        raw_conn, task_id, stage_name, sym.file_id,
                        included=(sym.detail_level != "excluded"),
                        reason=sym.reason,
                        symbol_id=sym.symbol_id,
                        detail_level=sym.detail_level,
                    )
                    logged_symbol_ids.add(sym.symbol_id)
            raw_conn.commit()

            # Save stage output to session
            set_state(session_conn, f"stage_output_{stage_name}", context.to_dict())
            set_state(session_conn, "stage_progress", {
                "completed": stage_names[:stage_names.index(stage_name) + 1],
                "remaining": stage_names[stage_names.index(stage_name) + 1:],
            })

        # 10. Context assembly (pass LLM for R1 re-filter if budget exceeded)
        assembly_config = router.resolve("reasoning")
        with LoggedLLMClient(assembly_config) as base_assembly_llm:
            assembly_llm = EnvironmentLLMClient(base_assembly_llm, brief_text)
            package = assemble_context(context, budget, repo_path, llm=assembly_llm, kb=kb)

            calls = assembly_llm.flush()
            if trace_logger is not None:
                trace_logger.log_calls("assembly", "assembly_refilter", calls, assembly_config.model)
            for call in calls:
                insert_retrieval_llm_call(
                    raw_conn, task_id, "assembly_refilter", assembly_config.model,
                    call["prompt"], call["response"],
                    call["prompt_tokens"], call["completion_tokens"],
                    call["elapsed_ms"],
                    stage_name="assembly",
                    system_prompt=call["system"],
                    thinking=call.get("thinking"),
                )

        # Log assembly-stage file decisions to raw DB
        for decision in package.metadata.get("assembly_decisions", []):
            insert_retrieval_decision(
                raw_conn, task_id, "assembly", decision["file_id"],
                included=decision["included"],
                reason=decision["reason"],
            )
        raw_conn.commit()

        package.environment_brief = brief_text
        package.metadata["stage_timings"] = dict(context.stage_timings)

        # 11. Save final context to session
        set_state(session_conn, "final_context", context.to_dict())

        # 12. Update task_run
        total_elapsed = int((time.monotonic() - total_start) * 1000)
        update_task_run(
            raw_conn, task_run_id,
            success=True,
            total_tokens=package.total_token_estimate,
            total_latency_ms=total_elapsed,
        )
        raw_conn.commit()

        return package

    except Exception:
        if task_run_id is not None:
            try:
                total_elapsed = int((time.monotonic() - total_start) * 1000)
                update_task_run(raw_conn, task_run_id, success=False, total_latency_ms=total_elapsed)
                raw_conn.commit()
            except Exception as db_err:
                logger.warning("Failed to log error to raw DB: %s", db_err)
        raise
    finally:
        curated_conn.close()
        session_conn.close()

        # Archive session DB to raw, then delete the file
        session_file = _db_path(repo_path, "session", task_id)
        if session_file.exists():
            try:
                session_blob = session_file.read_bytes()
                insert_session_archive(raw_conn, task_id, session_blob)
                raw_conn.commit()
                session_file.unlink()
                logger.debug("Archived and removed session DB for task %s", task_id)
            except Exception as archive_err:
                logger.warning("Failed to archive session DB: %s", archive_err)

        raw_conn.close()


def _restore_session_from_archive(
    raw_conn,
    repo_path: Path,
    task_id: str,
    source_task_id: str,
) -> None:
    """Restore session DB from a previous task's archive for refinement re-entry.

    The previous pipeline run archived and deleted its session DB.  This
    function retrieves the most recent archive blob for ``source_task_id``
    and writes it as the session file for ``task_id`` so that
    ``_resume_task_from_session`` can read the stored task_query.
    """
    session_file = _db_path(repo_path, "session", task_id)
    if session_file.exists():
        # Session file already exists (e.g. from a prior interrupted run) — nothing to restore
        return

    archive = raw_conn.execute(
        "SELECT session_blob FROM session_archives WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (source_task_id,),
    ).fetchone()
    if not archive:
        raise RuntimeError(
            f"Refinement requested but no session archive found for source task {source_task_id!r}."
        )

    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_bytes(archive["session_blob"])
    logger.debug(
        "Restored session from archive (source_task_id=%s) for refinement task %s",
        source_task_id, task_id,
    )


def _resume_task_from_session(
    session_conn,
    refinement: RefinementRequest,
    raw_task: str,
    task_id: str,
    mode: str,
    repo_id: int,
) -> TaskQuery:
    """Resume a TaskQuery from session DB, merging refinement request."""
    data = get_state(session_conn, "task_query")
    if data is None:
        raise RuntimeError("No task_query in session DB for refinement re-entry.")

    # Merge refinement info
    extra_files = list(data.get("mentioned_files", []))
    extra_symbols = list(data.get("mentioned_symbols", []))
    extra_keywords = list(data.get("keywords", []))

    extra_files.extend(refinement.missing_files)
    extra_symbols.extend(refinement.missing_symbols)

    return TaskQuery(
        raw_task=raw_task,
        task_id=task_id,
        mode=mode,
        repo_id=repo_id,
        mentioned_files=extra_files,
        mentioned_symbols=extra_symbols,
        keywords=extra_keywords,
        error_patterns=list(data.get("error_patterns", [])) + list(refinement.error_patterns),
        task_type=data.get("task_type", "unknown"),
        intent_summary=data.get("intent_summary", ""),
        seed_file_ids=data.get("seed_file_ids", []),
        seed_symbol_ids=data.get("seed_symbol_ids", []),
    )
