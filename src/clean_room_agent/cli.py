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
