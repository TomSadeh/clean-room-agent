"""Stage routing: LLM-based selection of which retrieval stages to run."""

from clean_room_agent.llm.client import LLMClient, LoggedLLMClient
from clean_room_agent.retrieval.budget import estimate_tokens_conservative
from clean_room_agent.retrieval.dataclasses import TaskQuery
from clean_room_agent.retrieval.utils import parse_json_response

ROUTING_SYSTEM = (
    "You are Jane, a retrieval stage router. Given a task summary and "
    "available retrieval stages, select which stages to run.\n"
    "Respond with ONLY a JSON object:\n"
    '{"stages": ["stage_name", ...], "reasoning": "one sentence justification"}\n'
    "Select only stages that would contribute useful context for this task. "
    "If none are needed, use an empty list."
)


def build_routing_prompt(
    task_query: TaskQuery,
    available_stages: dict[str, str],
) -> str:
    """Build the user prompt for stage routing.

    Deliberately minimal — only the distilled summary from task analysis,
    not the full task text, file tree, or environment brief.
    """
    lines = [
        f"Task: {task_query.intent_summary}",
        f"Type: {task_query.task_type}",
        f"Seed files: {len(task_query.seed_file_ids)}",
        f"Seed symbols: {len(task_query.seed_symbol_ids)}",
        f"Explicit file paths: {len(task_query.mentioned_files)}",
        "",
        "Available stages:",
    ]
    for name, description in available_stages.items():
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)


def parse_routing_response(text: str) -> tuple[list[str], str]:
    """Parse routing LLM response into (selected_stages, reasoning).

    Raises ValueError on any parse failure (default-deny, R2).
    """
    data = parse_json_response(text, "routing")
    if not isinstance(data, dict):
        raise ValueError(f"Routing response must be a JSON object, got {type(data).__name__}")
    if "stages" not in data:
        raise ValueError("Routing response missing 'stages' key")
    stages = data["stages"]
    if not isinstance(stages, list):
        raise ValueError(f"'stages' must be a list, got {type(stages).__name__}")
    for item in stages:
        if not isinstance(item, str):
            raise ValueError(f"Each stage must be a string, got {type(item).__name__}: {item!r}")
    reasoning = data.get("reasoning", "")
    return stages, reasoning


def route_stages(
    task_query: TaskQuery,
    available_stages: dict[str, str],
    llm: LLMClient,
) -> tuple[list[str], str]:
    """Select which retrieval stages to run via LLM.

    Args:
        llm: Must be a LoggedLLMClient (or wrapper around one) so that the
             routing decision is recorded in the raw DB audit trail.

    Returns (selected_stage_names, reasoning). Empty list is valid (0 stages).
    Raises ValueError if LLM returns unparseable response or unknown stage names.
    """
    # T79: Enforce that the LLM client supports audit logging.
    # The caller must pass a client with flush() capability (LoggedLLMClient or
    # a wrapper like EnvironmentLLMClient that delegates to one).
    if not hasattr(llm, "flush"):
        raise TypeError(
            f"route_stages() requires a logging-capable LLM client (with flush()), "
            f"got {type(llm).__name__}"
        )
    prompt = build_routing_prompt(task_query, available_stages)

    input_tokens = estimate_tokens_conservative(prompt) + estimate_tokens_conservative(ROUTING_SYSTEM)
    available = llm.config.context_window - llm.config.max_tokens
    if input_tokens > available:
        raise ValueError(
            f"R3: routing prompt too large ({input_tokens} tokens, available {available})"
        )

    response = llm.complete(prompt, system=ROUTING_SYSTEM)
    selected, reasoning = parse_routing_response(response.text)

    # Validate all selected stages exist
    unknown = [s for s in selected if s not in available_stages]
    if unknown:
        raise ValueError(
            f"Routing selected unknown stages: {unknown}. "
            f"Available: {sorted(available_stages)}"
        )

    return selected, reasoning
