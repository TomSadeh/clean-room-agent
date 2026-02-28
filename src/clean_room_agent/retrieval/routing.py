"""Stage routing: per-stage binary LLM selection of which retrieval stages to run.

Binary decomposition: with N registered stages, the decision space is 2^N
possibilities. Each stage gets an independent binary call: "Should this stage
run?" The reasoning field (logged but never consumed downstream) is dropped.
"""

from clean_room_agent.llm.client import LLMClient
from clean_room_agent.retrieval.batch_judgment import run_binary_judgment
from clean_room_agent.retrieval.dataclasses import TaskQuery

ROUTING_BINARY_SYSTEM = (
    "You are a retrieval stage router. Given a task summary and one retrieval stage, "
    "determine if this stage should run to gather useful context for the task. "
    "Respond with ONLY \"yes\" or \"no\"."
)


def route_stages(
    task_query: TaskQuery,
    available_stages: dict[str, str],
    llm: LLMClient,
) -> tuple[list[str], str]:
    """Select which retrieval stages to run via per-stage binary LLM calls.

    Each available stage gets an independent yes/no call. This replaces the
    previous single-call JSON set-selection pattern.

    Args:
        llm: Must be a LoggedLLMClient (or wrapper around one) so that each
             routing decision is recorded in the raw DB audit trail.

    Returns (selected_stage_names, reasoning). reasoning is always "" (dropped).
    Empty list is valid (0 stages).
    """
    if not available_stages:
        return [], ""

    task_context = (
        f"Task: {task_query.intent_summary}\n"
        f"Type: {task_query.task_type}\n"
        f"Seed files: {len(task_query.seed_file_ids)}\n"
        f"Seed symbols: {len(task_query.seed_symbol_ids)}\n"
        f"Explicit file paths: {len(task_query.mentioned_files)}\n\n"
    )

    stage_items = list(available_stages.items())  # [(name, description), ...]

    verdict_map, _omitted = run_binary_judgment(
        stage_items,
        system_prompt=ROUTING_BINARY_SYSTEM,
        task_context=task_context,
        llm=llm,
        format_item=lambda item: f"Stage: {item[0]}\nDescription: {item[1]}",
        stage_name="routing",
        item_key=lambda item: item[0],
        default_action="skipped",
    )

    selected = [name for name, _ in stage_items if verdict_map.get(name, False)]
    return selected, ""
