"""Pipeline runner: orchestrates task analysis, retrieval stages, and context assembly."""

import json
import logging
import time
from pathlib import Path

from clean_room_agent.config import require_models_config
from clean_room_agent.db.connection import get_connection
from clean_room_agent.db.raw_queries import (
    insert_retrieval_decision,
    insert_retrieval_llm_call,
    insert_task_run,
    update_task_run,
)
from clean_room_agent.db.session_helpers import get_state, set_state
from clean_room_agent.llm.client import LLMClient
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
from clean_room_agent.retrieval.stage import StageContext, get_stage
from clean_room_agent.retrieval.task_analysis import analyze_task

logger = logging.getLogger(__name__)


class _LoggingLLMClient:
    """LLM client wrapper that records calls for pipeline logging."""

    def __init__(self, client: LLMClient):
        self._client = client
        self.calls: list[dict] = []

    @property
    def config(self):
        return self._client.config

    def complete(self, prompt: str, system: str | None = None):
        start = time.monotonic()
        response = self._client.complete(prompt, system=system)
        elapsed = int((time.monotonic() - start) * 1000)
        self.calls.append({
            "prompt": prompt,
            "system": system,
            "response": response.text,
            "elapsed_ms": elapsed,
        })
        return response


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
    curated_conn = get_connection("curated", repo_path=repo_path, read_only=True)
    raw_conn = get_connection("raw", repo_path=repo_path)
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
            # Resume from session
            task_query = _resume_task_from_session(session_conn, refinement_request, raw_task, task_id, mode, repo_id)
        else:
            reasoning_config = router.resolve("reasoning", "task_analysis")
            with LLMClient(reasoning_config) as llm:
                logging_llm = _LoggingLLMClient(llm)
                start = time.monotonic()
                task_query = analyze_task(raw_task, task_id, mode, kb, repo_id, logging_llm)
                elapsed = int((time.monotonic() - start) * 1000)

                # Log the actual prompt/response (not just raw_task/intent_summary)
                for call in logging_llm.calls:
                    insert_retrieval_llm_call(
                        raw_conn, task_id, "task_analysis", reasoning_config.model,
                        call["prompt"], call["response"],
                        None, None, call["elapsed_ms"],
                        stage_name="task_analysis",
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
        for stage_name in stage_names:
            stage = stages[stage_name]
            stage_config = router.resolve("reasoning", stage_name)

            with LLMClient(stage_config) as llm:
                logging_llm = _LoggingLLMClient(llm)
                start = time.monotonic()
                context = stage.run(context, kb, task_query, logging_llm)
                elapsed = int((time.monotonic() - start) * 1000)
                context.stage_timings[stage_name] = elapsed

                # Log all stage LLM calls to raw DB
                for call in logging_llm.calls:
                    insert_retrieval_llm_call(
                        raw_conn, task_id, stage_name, stage_config.model,
                        call["prompt"], call["response"],
                        None, None, call["elapsed_ms"],
                        stage_name=stage_name,
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
            raw_conn.commit()

            # Save stage output to session
            set_state(session_conn, f"stage_output_{stage_name}", context.to_dict())
            set_state(session_conn, "stage_progress", {
                "completed": stage_names[:stage_names.index(stage_name) + 1],
                "remaining": stage_names[stage_names.index(stage_name) + 1:],
            })

        # 10. Context assembly (pass LLM for R1 re-filter if budget exceeded)
        assembly_config = router.resolve("reasoning")
        with LLMClient(assembly_config) as assembly_llm:
            package = assemble_context(context, budget, repo_path, llm=assembly_llm, kb=kb)
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
        raw_conn.close()
        session_conn.close()


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
