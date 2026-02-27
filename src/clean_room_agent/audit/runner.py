"""Audit runner: execute retrieval pipeline on reference tasks and compute metrics."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from clean_room_agent.audit.dataclasses import (
    AuditResult,
    AuditScores,
    AuditSuiteResult,
    ReferenceTask,
)
from clean_room_agent.audit.loader import load_all_reference_tasks, load_reference_task
from clean_room_agent.audit.scorer import score_retrieval
from clean_room_agent.audit.serializer import save_audit_result
from clean_room_agent.retrieval.dataclasses import BudgetConfig, ContextPackage

logger = logging.getLogger(__name__)


def _get_pipeline_version() -> str:
    """Get current git commit hash as pipeline version identifier."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _get_parse_success_rate(raw_conn, task_id: str) -> float:
    """Compute parse success rate from raw DB LLM call logs.

    Counts calls where the response was successfully parsed (no error recorded)
    vs total LLM calls for this task_id.
    """
    row = raw_conn.execute(
        "SELECT COUNT(*) as total FROM retrieval_llm_calls WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    total = row["total"] if row else 0
    if total == 0:
        return 1.0  # No calls = nothing to fail

    # Calls that produced parseable output: response is non-empty and non-null
    good_row = raw_conn.execute(
        "SELECT COUNT(*) as good FROM retrieval_llm_calls "
        "WHERE task_id = ? AND response IS NOT NULL AND response != ''",
        (task_id,),
    ).fetchone()
    good = good_row["good"] if good_row else 0
    return good / total


def run_single_audit(
    ref_task: ReferenceTask,
    repo_path: Path,
    config: dict,
    budget: BudgetConfig,
    stage_names: list[str],
    model_name: str = "",
    trace_flag: bool = False,
    trace_output: Path | None = None,
) -> AuditResult:
    """Run the retrieval pipeline on a single reference task and score it.

    Returns an AuditResult with scores and metadata. Pipeline failures
    are captured in AuditResult.error rather than raised.
    """
    from clean_room_agent.commands import make_trace_logger
    from clean_room_agent.db.connection import get_connection
    from clean_room_agent.retrieval.pipeline import run_pipeline

    audit_id = f"RA-{ref_task.id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    task_id = str(uuid.uuid4())
    pipeline_version = _get_pipeline_version()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    trace_logger = None
    if trace_flag or trace_output:
        trace_logger = make_trace_logger(
            repo_path, task_id, ref_task.description,
            trace_flag, str(trace_output) if trace_output else None,
        )

    start = time.monotonic()
    package = None
    error = ""
    stages_selected: list[str] = []
    routing_reasoning = ""

    try:
        package = run_pipeline(
            raw_task=ref_task.description,
            repo_path=repo_path,
            stage_names=stage_names,
            budget=budget,
            mode="plan",
            task_id=task_id,
            config=config,
            trace_logger=trace_logger,
        )
        stages_selected = list(package.metadata.get("stage_timings", {}).keys())
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.error("Pipeline failed for %s: %s", ref_task.id, error)
    finally:
        if trace_logger is not None:
            trace_logger.finalize()

    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Compute scores
    if package is not None:
        # Get parse success rate from raw DB
        parse_rate = 1.0
        try:
            raw_conn = get_connection("raw", repo_path=repo_path)
            parse_rate = _get_parse_success_rate(raw_conn, task_id)
            raw_conn.close()
        except Exception as e:
            logger.warning("Failed to compute parse_success_rate: %s", e)

        scores = score_retrieval(ref_task, package, parse_success_rate=parse_rate)
    else:
        # Pipeline failed — zero scores
        scores = AuditScores(
            must_contain_recall=0.0,
            should_contain_recall=0.0,
            exclusion_accuracy=0.0,
            budget_utilization=0.0,
            budget_in_range=False,
            parse_success_rate=0.0,
            must_contain_missing=list(ref_task.must_contain_files),
        )

    return AuditResult(
        audit_id=audit_id,
        task_id=ref_task.id,
        date=date,
        pipeline_version=pipeline_version,
        model=model_name,
        scores=scores,
        stages_selected=stages_selected,
        routing_reasoning=routing_reasoning,
        total_latency_ms=elapsed_ms,
        error=error,
    )


def run_audit_suite(
    repo_path: Path,
    config: dict,
    budget: BudgetConfig,
    stage_names: list[str],
    model_name: str = "",
    tasks_dir: Path | None = None,
    task_filter: str | None = None,
    trace_flag: bool = False,
    save_findings: bool = True,
) -> AuditSuiteResult:
    """Run the full audit suite on all reference tasks.

    Args:
        repo_path: Repository root.
        config: Loaded config dict.
        budget: Token budget configuration.
        stage_names: Configured stage names to make available.
        model_name: Model identifier for logging.
        tasks_dir: Override path to reference tasks directory.
        task_filter: If set, only run tasks whose ID matches this string.
        trace_flag: Enable trace output for each run.
        save_findings: Whether to save findings to TOML files.

    Returns:
        AuditSuiteResult with all individual results and aggregate metrics.
    """
    if task_filter:
        # Load single task by filter
        all_tasks = load_all_reference_tasks(tasks_dir=tasks_dir, repo_path=repo_path)
        tasks = [t for t in all_tasks if task_filter in t.id]
        if not tasks:
            raise ValueError(f"No reference tasks matching filter: {task_filter!r}")
    else:
        tasks = load_all_reference_tasks(tasks_dir=tasks_dir, repo_path=repo_path)

    pipeline_version = _get_pipeline_version()
    suite = AuditSuiteResult(
        pipeline_version=pipeline_version,
        model=model_name,
    )

    for ref_task in tasks:
        logger.info("Running audit: %s — %s", ref_task.id, ref_task.description)
        result = run_single_audit(
            ref_task=ref_task,
            repo_path=repo_path,
            config=config,
            budget=budget,
            stage_names=stage_names,
            model_name=model_name,
            trace_flag=trace_flag,
        )
        suite.results.append(result)

        if save_findings:
            path = save_audit_result(result, repo_path=repo_path)
            logger.info("Saved findings: %s", path)

        # Log per-task summary
        if result.error:
            logger.warning("  %s: FAILED — %s", ref_task.id, result.error)
        else:
            logger.info(
                "  %s: score=%.2f must_recall=%.2f excl_acc=%.2f budget=%.0f%%",
                ref_task.id, result.scores.task_score,
                result.scores.must_contain_recall,
                result.scores.exclusion_accuracy,
                result.scores.budget_utilization * 100,
            )

    return suite
