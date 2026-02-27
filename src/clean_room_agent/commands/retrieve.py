"""Business logic for `cra retrieve`."""

import logging
import uuid
from pathlib import Path

import click

from clean_room_agent.commands import _require_cli_section, make_trace_logger
from clean_room_agent.config import load_config
from clean_room_agent.retrieval.dataclasses import BudgetConfig


def run_retrieve(
    task: str,
    repo_path: str,
    stages: str | None,
    context_window: int | None,
    reserved_tokens: int | None,
    plan_path: str | None,
    trace_flag: bool,
    trace_output: str | None,
    verbose: bool,
) -> None:
    """Run the retrieval pipeline and print results."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    from clean_room_agent.retrieval.pipeline import run_pipeline

    repo = Path(repo_path).resolve()
    config = load_config(repo)

    if config is None:
        raise click.UsageError(
            "No config file found. Run 'cra init' to create .clean_room/config.toml"
        )

    # Resolve budget: CLI flag -> [budget] config -> hard error
    cw = context_window
    rt = reserved_tokens
    if cw is None or rt is None:
        budget_config = _require_cli_section(config, "budget")
        if cw is None:
            cw = budget_config.get("context_window")
        if rt is None:
            rt = budget_config.get("reserved_tokens")
    if cw is None or rt is None:
        raise click.UsageError(
            "Budget not configured. Provide --context-window and --reserved-tokens, "
            "or set [budget] in .clean_room/config.toml."
        )
    budget = BudgetConfig(context_window=cw, reserved_tokens=rt)

    # Resolve stages: CLI flag -> config.toml -> hard error
    if stages:
        stage_names = [s.strip() for s in stages.split(",")]
    else:
        stages_config = _require_cli_section(config, "stages")
        default_stages = stages_config.get("default")
        if not default_stages:
            raise click.UsageError(
                "Stages not configured. Provide --stages or set [stages] default in config.toml."
            )
        stage_names = [s.strip() for s in default_stages.split(",")]

    task_id = str(uuid.uuid4())
    plan_artifact = Path(plan_path) if plan_path else None

    trace_logger = make_trace_logger(repo, task_id, task, trace_flag, trace_output)

    try:
        package = run_pipeline(
            raw_task=task,
            repo_path=repo,
            stage_names=stage_names,
            budget=budget,
            mode="plan",
            task_id=task_id,
            config=config,
            plan_artifact_path=plan_artifact,
            trace_logger=trace_logger,
        )
    finally:
        if trace_logger is not None:
            trace_path = trace_logger.finalize()
            click.echo(f"Trace written to {trace_path}")

    click.echo(f"Retrieval complete (task_id={task_id})")
    click.echo(f"  Files:       {len(package.files)}")
    click.echo(f"  Tokens:      {package.total_token_estimate}/{budget.effective_budget}")
    click.echo(f"  Timings:     {package.metadata.get('stage_timings', {})}")
    for fc in package.files:
        click.echo(f"    {fc.path} [{fc.detail_level}] ~{fc.token_estimate} tokens")
