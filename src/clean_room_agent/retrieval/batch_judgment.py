"""Shared batching utilities for LLM judgment stages (T83).

Used by scope, precision, and similarity stages to calculate batch sizes,
validate prompts against the model's context window (R3), and run the
shared batch loop (R6).
"""

import logging
from collections.abc import Callable, Hashable, Sequence

from clean_room_agent.retrieval.budget import estimate_tokens_conservative
from clean_room_agent.retrieval.utils import parse_json_response

logger = logging.getLogger(__name__)


def calculate_judgment_batch_size(
    system_prompt: str,
    task_header: str,
    context_window: int,
    max_tokens: int,
    tokens_per_candidate: int,
) -> int:
    """Calculate safe batch size for LLM judgment stages.

    Determines how many candidates to pack into each LLM batch while
    respecting the model's context window limit. Uses conservative
    token estimation (3 chars/token) to match the LLMClient safety gate.

    Returns at least 1 to ensure progress even with very large overhead.
    """
    system_overhead = estimate_tokens_conservative(system_prompt)
    header_overhead = estimate_tokens_conservative(task_header)
    available = context_window - max_tokens - system_overhead - header_overhead
    return max(1, available // tokens_per_candidate)


def validate_judgment_batch(
    prompt: str,
    system_prompt: str,
    stage_name: str,
    context_window: int,
    max_tokens: int,
) -> None:
    """Validate a judgment batch prompt fits within the context window (R3).

    Raises ValueError if the prompt exceeds the available tokens.
    """
    actual_tokens = estimate_tokens_conservative(prompt) + estimate_tokens_conservative(system_prompt)
    available = context_window - max_tokens
    if actual_tokens > available:
        raise ValueError(
            f"R3: {stage_name} batch prompt too large ({actual_tokens} tokens, "
            f"available {available})"
        )


def log_r2_omission(label: str, stage_name: str, default_action: str) -> None:
    """Log an R2 default-deny omission (R22: consistent across stages)."""
    logger.warning(
        "R2: LLM omitted %s from %s judgment — defaulting to %s",
        label, stage_name, default_action,
    )


def run_batched_judgment(
    items: Sequence,
    *,
    system_prompt: str,
    task_header: str,
    llm,
    tokens_per_item: int,
    format_item: Callable,
    extract_key: Callable[[dict], Hashable | None],
    stage_name: str,
    item_key: Callable | None = None,
    default_action: str = "excluded",
) -> tuple[dict[Hashable, dict], set[Hashable]]:
    """Run the shared batch→validate→LLM→parse loop for judgment stages.

    Args:
        items: Candidates to judge.
        system_prompt: System prompt for the LLM.
        task_header: Task description header prepended to each batch prompt.
        llm: LLM client with .config.context_window/.config.max_tokens.
        tokens_per_item: Estimated tokens per candidate line.
        format_item: Callback (item) -> str that formats one item as a prompt line.
        extract_key: Callback (json_dict) -> key that extracts a lookup key from
            each parsed JSON result. Return None to skip the entry.
        stage_name: Name for logging and R3 validation.
        item_key: Optional callback (item) -> key that extracts a lookup key from
            each input item. When provided, omitted items are tracked and logged.
        default_action: Stage-specific R2 default label for omission logging.

    Returns:
        Tuple of (result_map, omitted_keys):
        - result_map: dict mapping key -> judgment dict for all parsed LLM results.
        - omitted_keys: set of input item keys not found in result_map (empty if
          item_key is None).
    """
    if not items:
        return {}, set()

    batch_size = calculate_judgment_batch_size(
        system_prompt, task_header,
        llm.config.context_window, llm.config.max_tokens,
        tokens_per_item,
    )

    result_map: dict[Hashable, dict] = {}
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        candidate_lines = [format_item(item) for item in batch]
        prompt = task_header + "\n".join(candidate_lines)

        validate_judgment_batch(
            prompt, system_prompt, stage_name,
            llm.config.context_window, llm.config.max_tokens,
        )

        response = llm.complete(prompt, system=system_prompt)
        try:
            judgments = parse_json_response(response.text, f"{stage_name} judgment")
        except ValueError:
            logger.warning("R2: %s judgment parse failed — denying all items in batch", stage_name)
            continue

        if not isinstance(judgments, list):
            logger.warning("R2: %s judgment returned non-list — denying all items in batch", stage_name)
            continue

        for j in judgments:
            if not isinstance(j, dict):
                logger.warning("R2: %s judgment returned non-dict item (type=%s) — skipping", stage_name, type(j).__name__)
                continue
            key = extract_key(j)
            if key is None:
                logger.warning("R2: %s judgment item missing extractable key — skipping: %s", stage_name, str(j)[:200])
                continue
            result_map[key] = j

    # Track and log omissions (R2 centralized)
    omitted_keys: set[Hashable] = set()
    if item_key is not None:
        all_keys = {item_key(item) for item in items}
        omitted_keys = all_keys - result_map.keys()
        for key in omitted_keys:
            log_r2_omission(str(key), stage_name, default_action)

    return result_map, omitted_keys
