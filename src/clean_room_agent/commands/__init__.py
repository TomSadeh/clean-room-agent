"""Shared helpers for CLI commands."""

import click


def resolve_budget(config: dict | None, role: str = "reasoning") -> tuple[int, int]:
    """Resolve (context_window, reserved_tokens) from config. Raises on missing.

    Uses ModelRouter to resolve context_window so per-role dicts are handled
    correctly (same as orchestrator/runner.py).
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


def resolve_stages(config: dict | None, stages_flag: str | None) -> list[str]:
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


def make_trace_logger(repo_path, task_id, task, trace_flag, trace_output):
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
