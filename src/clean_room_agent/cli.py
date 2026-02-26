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

    from clean_room_agent.config import load_config
    from clean_room_agent.indexer.orchestrator import index_repository

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    repo = Path(repo_path).resolve()
    config = load_config(repo)
    result = index_repository(
        repo,
        continue_on_error=continue_on_error,
        indexer_config=(config or {}).get("indexer"),
    )

    click.echo(f"Indexed {repo}")
    click.echo(f"  Files scanned:   {result.files_scanned}")
    click.echo(f"  New:             {result.files_new}")
    click.echo(f"  Changed:         {result.files_changed}")
    click.echo(f"  Deleted:         {result.files_deleted}")
    click.echo(f"  Unchanged:       {result.files_unchanged}")
    click.echo(f"  Parse errors:    {result.parse_errors}")
    click.echo(f"  Duration:        {result.duration_ms}ms")


@cli.command("index-libraries")
@click.argument("repo_path", default=".", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def index_libraries(repo_path, verbose):
    """Index library/dependency source files (separate from project indexing)."""
    import logging
    from pathlib import Path

    from clean_room_agent.config import load_config
    from clean_room_agent.indexer.orchestrator import index_libraries as _index_libraries

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    repo = Path(repo_path).resolve()
    config = load_config(repo)
    result = _index_libraries(
        repo,
        indexer_config=(config or {}).get("indexer"),
    )

    click.echo(f"Library indexing complete for {repo}")
    click.echo(f"  Libraries found: {result.libraries_found}")
    click.echo(f"  Files scanned:   {result.files_scanned}")
    click.echo(f"  New/changed:     {result.files_new}")
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

    repo = Path(repo_path).resolve()

    # T63: Preflight check — curated DB must exist (requires prior `cra index`)
    curated_db = repo / ".clean_room" / "curated.sqlite"
    if not curated_db.exists():
        raise click.UsageError(
            "Curated database not found. Run 'cra index' before 'cra enrich'."
        )

    from clean_room_agent.llm.enrichment import enrich_repository

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
@click.option("--trace", "trace_flag", is_flag=True, help="Enable pipeline trace log.")
@click.option("--trace-output", type=click.Path(), default=None, help="Custom trace output path (implies --trace).")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def retrieve(task, repo_path, stages, context_window, reserved_tokens, plan_path, trace_flag, trace_output, verbose):
    """Run the retrieval pipeline to produce a context package."""
    from clean_room_agent.commands.retrieve import run_retrieve

    run_retrieve(task, repo_path, stages, context_window, reserved_tokens, plan_path, trace_flag, trace_output, verbose)


@cli.command()
@click.argument("task")
@click.option("--repo", "repo_path", default=".", type=click.Path(exists=True), help="Repository path.")
@click.option("--stages", default=None, help="Comma-separated stage names.")
@click.option("--output", type=click.Path(), default=None, help="Save plan to file.")
@click.option("--trace", "trace_flag", is_flag=True, help="Enable pipeline trace log.")
@click.option("--trace-output", type=click.Path(), default=None, help="Custom trace output path (implies --trace).")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def plan(task, repo_path, stages, output, trace_flag, trace_output, verbose):
    """Generate a plan for a task (meta-plan decomposition)."""
    from clean_room_agent.commands.plan import run_plan

    run_plan(task, repo_path, stages, output, trace_flag, trace_output, verbose)


@cli.command()
@click.argument("task")
@click.option("--repo", "repo_path", default=".", type=click.Path(exists=True), help="Repository path.")
@click.option("--plan", "plan_path", default=None, type=click.Path(exists=True),
              help="Pre-computed plan file (single atomic pass, skips orchestrator).")
@click.option("--trace", "trace_flag", is_flag=True, help="Enable pipeline trace log.")
@click.option("--trace-output", type=click.Path(), default=None, help="Custom trace output path (implies --trace).")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def solve(task, repo_path, plan_path, trace_flag, trace_output, verbose):
    """Solve a task by generating and applying code changes."""
    from clean_room_agent.commands.solve import run_solve

    run_solve(task, repo_path, plan_path, trace_flag, trace_output, verbose)
