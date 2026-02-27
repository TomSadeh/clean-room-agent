"""Load and validate reference tasks from TOML files."""

from __future__ import annotations

import logging
from pathlib import Path

from clean_room_agent.audit.dataclasses import ReferenceTask

logger = logging.getLogger(__name__)

# Default location relative to repo root
DEFAULT_TASKS_DIR = Path("protocols/retrieval_audit/reference_tasks")


def load_reference_task(path: Path) -> ReferenceTask:
    """Load a single reference task from a TOML file.

    Raises ValueError on schema violations. Missing required sections
    are hard errors — a reference task without context requirements is
    not a reference task.
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text)

    task_section = data.get("task")
    if task_section is None:
        raise ValueError(f"{path}: missing [task] section")

    ctx = data.get("context_requirements")
    if ctx is None:
        raise ValueError(f"{path}: missing [context_requirements] section")

    task_id = task_section.get("id")
    if not task_id:
        raise ValueError(f"{path}: missing task.id")

    description = task_section.get("description")
    if not description:
        raise ValueError(f"{path}: missing task.description")

    task_type = task_section.get("task_type")
    if not task_type:
        raise ValueError(f"{path}: missing task.task_type")

    budget_range_raw = ctx["budget_range"]
    if not isinstance(budget_range_raw, list) or len(budget_range_raw) != 2:
        raise ValueError(
            f"{path}: budget_range must be a two-element array, got {budget_range_raw!r}"
        )

    routing = data["routing_notes"]

    return ReferenceTask(
        id=task_id,
        description=description,
        task_type=task_type,
        must_contain_files=ctx["must_contain_files"],
        should_contain_files=ctx["should_contain_files"],
        must_not_contain=ctx["must_not_contain"],
        must_contain_information=ctx["must_contain_information"],
        budget_range=tuple(budget_range_raw),
        routing_reasoning=routing["reasoning"],
    )


def load_all_reference_tasks(
    tasks_dir: Path | None = None,
    repo_path: Path | None = None,
) -> list[ReferenceTask]:
    """Load all reference tasks from the tasks directory.

    Args:
        tasks_dir: Explicit path to reference_tasks/ directory.
        repo_path: Repository root. Used to resolve DEFAULT_TASKS_DIR if
                   tasks_dir is not provided.

    Returns list sorted by task ID. Raises on any invalid task file —
    a broken reference task set is worse than no reference tasks.
    """
    if tasks_dir is None:
        if repo_path is None:
            raise ValueError("Either tasks_dir or repo_path must be provided")
        tasks_dir = repo_path / DEFAULT_TASKS_DIR

    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"Reference tasks directory not found: {tasks_dir}")

    toml_files = sorted(tasks_dir.glob("*.toml"))
    if not toml_files:
        raise FileNotFoundError(f"No .toml files found in {tasks_dir}")

    tasks = []
    for path in toml_files:
        task = load_reference_task(path)
        tasks.append(task)

    # Check for duplicate IDs
    ids = [t.id for t in tasks]
    dupes = [tid for tid in ids if ids.count(tid) > 1]
    if dupes:
        raise ValueError(f"Duplicate reference task IDs: {sorted(set(dupes))}")

    return sorted(tasks, key=lambda t: t.id)
