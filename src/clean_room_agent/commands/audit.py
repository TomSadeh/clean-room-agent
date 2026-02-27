"""Business logic for `cra audit`."""

import logging
from pathlib import Path

import click

from clean_room_agent.commands import resolve_budget, resolve_stages
from clean_room_agent.config import load_config, require_models_config


def run_audit(
    repo_path: str,
    task_filter: str | None,
    stages: str | None,
    trace_flag: bool,
    verbose: bool,
    save_findings: bool,
) -> None:
    """Run the retrieval audit suite."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    from clean_room_agent.audit.runner import run_audit_suite
    from clean_room_agent.audit.serializer import format_suite_summary
    from clean_room_agent.llm.router import ModelRouter
    from clean_room_agent.retrieval.dataclasses import BudgetConfig

    repo = Path(repo_path).resolve()
    config = load_config(repo)
    if config is None:
        raise click.UsageError(
            "No config file found. Run 'cra init' to create .clean_room/config.toml"
        )

    cw, rt = resolve_budget(config)
    budget = BudgetConfig(context_window=cw, reserved_tokens=rt)
    stage_names = resolve_stages(config, stages)

    models_config = require_models_config(config)
    router = ModelRouter(models_config)
    model_name = router.resolve("reasoning").model

    suite = run_audit_suite(
        repo_path=repo,
        config=config,
        budget=budget,
        stage_names=stage_names,
        model_name=model_name,
        task_filter=task_filter,
        trace_flag=trace_flag,
        save_findings=save_findings,
    )

    click.echo("")
    click.echo(format_suite_summary(suite))
