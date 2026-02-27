"""Business logic for `cra plan`."""

import json
import logging
import uuid
from pathlib import Path

import click

from clean_room_agent.commands import make_trace_logger, resolve_budget, resolve_stages
from clean_room_agent.config import load_config, require_models_config


def run_plan(
    task: str,
    repo_path: str,
    stages: str | None,
    output: str | None,
    trace_flag: bool,
    trace_output: str | None,
    verbose: bool,
) -> None:
    """Run retrieval + plan generation and print/save the result."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    from clean_room_agent.db.connection import get_connection
    from clean_room_agent.db.raw_queries import insert_retrieval_llm_call
    from clean_room_agent.execute.dataclasses import PlanArtifact
    from clean_room_agent.execute.decomposed_plan import decomposed_meta_plan
    from clean_room_agent.execute.plan import execute_plan
    from clean_room_agent.llm.client import LoggedLLMClient
    from clean_room_agent.llm.router import ModelRouter
    from clean_room_agent.retrieval.dataclasses import BudgetConfig
    from clean_room_agent.retrieval.pipeline import run_pipeline

    repo = Path(repo_path).resolve()
    config = load_config(repo)
    if config is None:
        raise click.UsageError(
            "No config file found. Run 'cra init' to create .clean_room/config.toml"
        )

    models_config = require_models_config(config)
    router = ModelRouter(models_config)
    reasoning_config = router.resolve("reasoning")

    cw, rt = resolve_budget(config)
    budget = BudgetConfig(context_window=cw, reserved_tokens=rt)
    stage_names = resolve_stages(config, stages)
    task_id = str(uuid.uuid4())

    trace_logger = make_trace_logger(repo, task_id, task, trace_flag, trace_output)

    try:
        # Phase 2: retrieval
        package = run_pipeline(
            raw_task=task,
            repo_path=repo,
            stage_names=stage_names,
            budget=budget,
            mode="plan",
            task_id=task_id,
            config=config,
            trace_logger=trace_logger,
        )

        # Phase 3: execute plan
        # Flush LLM records even if execute_plan raises (T28 traceability)
        decomposed = bool(config.get("orchestrator", {}).get("decomposed_planning", False))
        with LoggedLLMClient(reasoning_config) as llm:
            try:
                if decomposed:
                    meta_plan = decomposed_meta_plan(package, task, llm)
                else:
                    meta_plan = execute_plan(
                        package, task, llm,
                        pass_type="meta_plan",
                    )
            finally:
                raw_conn = get_connection("raw", repo_path=repo)
                try:
                    calls = llm.flush()
                    if trace_logger is not None:
                        trace_logger.log_calls("execute_plan", "execute_plan", calls, reasoning_config.model)
                    for call in calls:
                        insert_retrieval_llm_call(
                            raw_conn, task_id, "execute_plan", reasoning_config.model,
                            call["prompt"], call["response"],
                            call["prompt_tokens"], call["completion_tokens"],
                            call["elapsed_ms"],
                            stage_name="execute_plan",
                            system_prompt=call["system"],
                            thinking=call.get("thinking"),
                        )
                    raw_conn.commit()
                finally:
                    raw_conn.close()
    finally:
        if trace_logger is not None:
            trace_path = trace_logger.finalize()
            click.echo(f"Trace written to {trace_path}")

    # Convert to user-facing PlanArtifact
    artifact = PlanArtifact.from_meta_plan(meta_plan)
    plan_json = json.dumps(artifact.to_dict(), indent=2)

    if output:
        out_path = Path(output)
        out_path.write_text(plan_json, encoding="utf-8")
        click.echo(f"Plan written to {out_path}")
    else:
        click.echo(plan_json)

    click.echo(f"Task ID: {task_id}")
    click.echo(f"Parts: {len(meta_plan.parts)}")
