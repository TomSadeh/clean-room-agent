"""Click CLI for the Clean Room Agent."""

import click


@click.group()
def cli():
    """cra — Clean Room Agent CLI."""


@cli.command()
@click.argument("repo_path", default=".", type=click.Path(exists=True))
def init(repo_path):
    """Initialize a .clean_room directory with config.toml."""
    from pathlib import Path

    from clean_room_agent.config import create_default_config

    repo = Path(repo_path).resolve()
    config_path = create_default_config(repo)
    click.echo(f"Created {config_path}")

    # Ensure .clean_room/ is in .gitignore
    gitignore = repo / ".gitignore"
    marker = ".clean_room/"
    if gitignore.exists():
        content = gitignore.read_text()
        if marker not in content:
            with open(gitignore, "a") as f:
                f.write(f"\n{marker}\n")
            click.echo(f"Added {marker} to .gitignore")
    else:
        gitignore.write_text(f"{marker}\n")
        click.echo(f"Created .gitignore with {marker}")


@cli.command()
@click.argument("repo_path", default=".", type=click.Path(exists=True))
@click.option("--continue-on-error", is_flag=True, help="Log parse errors and continue.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def index(repo_path, continue_on_error, verbose):
    """Index a repository into the knowledge base."""
    import logging
    from pathlib import Path

    from clean_room_agent.indexer.orchestrator import index_repository

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    repo = Path(repo_path).resolve()
    result = index_repository(repo, continue_on_error=continue_on_error)

    click.echo(f"Indexed {repo}")
    click.echo(f"  Files scanned:   {result.files_scanned}")
    click.echo(f"  New:             {result.files_new}")
    click.echo(f"  Changed:         {result.files_changed}")
    click.echo(f"  Deleted:         {result.files_deleted}")
    click.echo(f"  Unchanged:       {result.files_unchanged}")
    click.echo(f"  Parse errors:    {result.parse_errors}")
    click.echo(f"  Duration:        {result.duration_ms}ms")


@cli.command()
@click.argument("repo_path", default=".", type=click.Path(exists=True))
@click.option("--promote", is_flag=True, help="Copy enrichment data to curated DB.")
def enrich(repo_path, promote):
    """Run LLM enrichment on indexed files."""
    from pathlib import Path

    from clean_room_agent.config import load_config, require_models_config
    from clean_room_agent.llm.enrichment import enrich_repository

    repo = Path(repo_path).resolve()
    config = load_config(repo)
    models_config = require_models_config(config)
    result = enrich_repository(repo, models_config, promote=promote)
    click.echo(f"Enriched {result.files_enriched} files ({result.files_skipped} skipped)")
    if promote:
        click.echo(f"Promoted {result.files_promoted} to curated DB")


@cli.command()
@click.argument("task")
@click.option("--repo", "repo_path", default=".", type=click.Path(exists=True), help="Repository path.")
@click.option("--stages", default=None, help="Comma-separated stage names (e.g. scope,precision).")
@click.option("--context-window", type=int, default=None, help="Context window size in tokens.")
@click.option("--reserved-tokens", type=int, default=None, help="Tokens reserved for execute stage.")
@click.option("--plan", "plan_path", default=None, type=click.Path(exists=True), help="Plan artifact JSON path.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def retrieve(task, repo_path, stages, context_window, reserved_tokens, plan_path, verbose):
    """Run the retrieval pipeline to produce a context package."""
    import logging
    import uuid
    from pathlib import Path

    from clean_room_agent.config import load_config
    from clean_room_agent.retrieval.dataclasses import BudgetConfig
    from clean_room_agent.retrieval.pipeline import run_pipeline

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    repo = Path(repo_path).resolve()
    config = load_config(repo)

    # Resolve budget: CLI flag -> [budget] config -> [models].context_window -> hard error
    cw = context_window
    rt = reserved_tokens
    if cw is None or rt is None:
        budget_config = (config or {}).get("budget", {})
        if cw is None:
            cw = budget_config.get("context_window")
        if cw is None:
            # Fall back to models.context_window (single source of truth)
            cw = (config or {}).get("models", {}).get("context_window")
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
        stages_config = (config or {}).get("stages", {})
        default_stages = stages_config.get("default")
        if not default_stages:
            raise click.UsageError(
                "Stages not configured. Provide --stages or set [stages] default in config.toml."
            )
        stage_names = [s.strip() for s in default_stages.split(",")]

    task_id = str(uuid.uuid4())
    plan_artifact = Path(plan_path) if plan_path else None

    if config is None:
        raise click.UsageError(
            "No config file found. Run 'cra init' to create .clean_room/config.toml"
        )

    package = run_pipeline(
        raw_task=task,
        repo_path=repo,
        stage_names=stage_names,
        budget=budget,
        mode="plan",
        task_id=task_id,
        config=config,
        plan_artifact_path=plan_artifact,
    )

    click.echo(f"Retrieval complete (task_id={task_id})")
    click.echo(f"  Files:       {len(package.files)}")
    click.echo(f"  Tokens:      {package.total_token_estimate}/{budget.effective_budget}")
    click.echo(f"  Timings:     {package.metadata.get('stage_timings', {})}")
    for fc in package.files:
        click.echo(f"    {fc.path} [{fc.detail_level}] ~{fc.token_estimate} tokens")
