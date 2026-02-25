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

    trace_logger = _make_trace_logger(repo, task_id, task, trace_flag, trace_output)

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


def _make_trace_logger(repo_path, task_id, task, trace_flag, trace_output):
    """Create a TraceLogger if tracing is enabled, else return None."""
    if not trace_flag and not trace_output:
        return None
    from pathlib import Path

    from clean_room_agent.trace import TraceLogger

    if trace_output:
        output_path = Path(trace_output)
    else:
        output_path = repo_path / ".clean_room" / "traces" / f"trace_{task_id}.md"
    return TraceLogger(output_path, task_id, task)


def _resolve_budget(config: dict | None, role: str = "reasoning") -> tuple[int, int]:
    """Resolve (context_window, reserved_tokens) from config. Raises on missing.

    Uses ModelRouter to resolve context_window so per-role dicts are handled
    correctly (same as orchestrator/runner.py _resolve_budget).
    """
    if config is None:
        raise click.UsageError(
            "Budget not configured. Set [budget] in .clean_room/config.toml."
        )
    from clean_room_agent.config import require_models_config
    from clean_room_agent.llm.router import ModelRouter

    models_config = require_models_config(config)
    router = ModelRouter(models_config)
    model_config = router.resolve(role)
    cw = model_config.context_window

    budget_config = config.get("budget", {})
    rt = budget_config.get("reserved_tokens")
    if rt is None:
        raise click.UsageError(
            "Budget not configured. Set reserved_tokens in [budget] in .clean_room/config.toml."
        )
    return cw, rt


def _resolve_stages(config: dict | None, stages_flag: str | None) -> list[str]:
    """Resolve stage names from CLI flag or config."""
    if stages_flag:
        return [s.strip() for s in stages_flag.split(",")]
    stages_config = (config or {}).get("stages", {})
    default_stages = stages_config.get("default")
    if not default_stages:
        raise click.UsageError(
            "Stages not configured. Provide --stages or set [stages] default in config.toml."
        )
    return [s.strip() for s in default_stages.split(",")]


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
    import json
    import logging
    import uuid
    from pathlib import Path

    from clean_room_agent.config import load_config, require_models_config
    from clean_room_agent.db.connection import get_connection
    from clean_room_agent.db.raw_queries import insert_retrieval_llm_call
    from clean_room_agent.execute.dataclasses import PlanArtifact
    from clean_room_agent.execute.plan import execute_plan
    from clean_room_agent.llm.client import LoggedLLMClient
    from clean_room_agent.llm.router import ModelRouter
    from clean_room_agent.retrieval.dataclasses import BudgetConfig
    from clean_room_agent.retrieval.pipeline import run_pipeline

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

    models_config = require_models_config(config)
    router = ModelRouter(models_config)
    reasoning_config = router.resolve("reasoning")

    cw, rt = _resolve_budget(config)
    budget = BudgetConfig(context_window=cw, reserved_tokens=rt)
    stage_names = _resolve_stages(config, stages)
    task_id = str(uuid.uuid4())

    trace_logger = _make_trace_logger(repo, task_id, task, trace_flag, trace_output)

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
        with LoggedLLMClient(reasoning_config) as llm:
            try:
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
    import logging
    from pathlib import Path

    from clean_room_agent.config import load_config, require_models_config
    from clean_room_agent.orchestrator.validator import require_testing_config

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
    import uuid
    trace_task_id = str(uuid.uuid4())
    trace_logger = _make_trace_logger(repo, trace_task_id, task, trace_flag, trace_output)

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
