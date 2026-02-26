"""Business logic for `cra solve`."""

import logging
import uuid
from pathlib import Path

import click

from clean_room_agent.commands import make_trace_logger
from clean_room_agent.config import load_config, require_models_config
from clean_room_agent.orchestrator.validator import require_testing_config


def run_solve(
    task: str,
    repo_path: str,
    plan_path: str | None,
    trace_flag: bool,
    trace_output: str | None,
    verbose: bool,
) -> None:
    """Run the orchestrator (or single pass) and print results."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    repo = Path(repo_path).resolve()
    config = load_config(repo)
    if config is None:
        raise click.UsageError(
            "No config file found. Run 'cra init' to create .clean_room/config.toml"
        )

    # Validate required config sections
    require_models_config(config)
    require_testing_config(config)

    # Use a temporary task_id for trace logger (the orchestrator generates the real one)
    trace_task_id = str(uuid.uuid4())
    trace_logger = make_trace_logger(repo, trace_task_id, task, trace_flag, trace_output)

    try:
        if plan_path:
            from clean_room_agent.orchestrator.runner import run_single_pass
            result = run_single_pass(task, repo, config, plan_path=Path(plan_path), trace_logger=trace_logger)
        else:
            from clean_room_agent.orchestrator.runner import run_orchestrator
            result = run_orchestrator(task, repo, config, trace_logger=trace_logger)
    finally:
        if trace_logger is not None:
            trace_path = trace_logger.finalize()
            click.echo(f"Trace written to {trace_path}")

    click.echo(f"Status: {result.status}")
    click.echo(f"Task ID: {result.task_id}")
    click.echo(f"Parts completed: {result.parts_completed}")
    click.echo(f"Steps completed: {result.steps_completed}")
    if result.pass_results:
        passes_ok = sum(1 for pr in result.pass_results if pr.success)
        click.echo(f"Passes: {passes_ok}/{len(result.pass_results)} successful")
