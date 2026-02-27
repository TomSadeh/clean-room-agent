"""Shared batching utilities for LLM judgment stages (T83).

Used by scope, precision, and similarity stages to calculate batch sizes,
validate prompts against the model's context window (R3), and run the
shared batch loop (R6).
"""

import logging
from collections.abc import Callable, Hashable, Sequence

from clean_room_agent.retrieval.budget import estimate_tokens_conservative
from clean_room_agent.retrieval.utils import parse_json_response
from clean_room_agent.token_estimation import validate_prompt_budget

logger = logging.getLogger(__name__)


def _require_logged_client(llm, caller: str) -> None:
    """Enforce LoggedLLMClient at runtime so LLM I/O logging cannot be forgotten."""
    if not hasattr(llm, "flush"):
        raise TypeError(
            f"{caller} requires a logging-capable LLM client (with flush()), "
            f"got {type(llm).__name__}"
        )


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
        llm: Logged LLM client with .config.context_window/.config.max_tokens.
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

    _require_logged_client(llm, "run_batched_judgment")

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

        validate_prompt_budget(
            prompt, system_prompt,
            llm.config.context_window, llm.config.max_tokens, stage_name,
        )

        response = llm.complete(prompt, system=system_prompt)
        try:
            judgments = parse_json_response(response.text, f"{stage_name} judgment")
        except ValueError as e:
            raise ValueError(
                f"{stage_name} judgment batch {i // batch_size + 1} returned "
                f"unparseable JSON. Raw response (first 500 chars): "
                f"{response.text[:500]!r}"
            ) from e

        if not isinstance(judgments, list):
            raise ValueError(
                f"{stage_name} judgment batch {i // batch_size + 1} returned "
                f"{type(judgments).__name__} instead of list. Raw response "
                f"(first 500 chars): {response.text[:500]!r}"
            )

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


def run_binary_judgment(
    items: Sequence,
    *,
    system_prompt: str,
    task_context: str,
    llm,
    format_item: Callable,
    stage_name: str,
    item_key: Callable | None = None,
    default_action: str = "excluded",
) -> tuple[dict[Hashable, bool], set[Hashable]]:
    """Run independent binary (yes/no) LLM judgment on each item.

    Unlike run_batched_judgment(), each item gets its own LLM call with a
    single yes/no question. Designed for small classifier models (0.6B)
    where the context window is too small for batching.

    Args:
        items: Candidates to judge.
        system_prompt: System prompt for the binary classifier.
        task_context: Task description prepended to each item's prompt.
        llm: LLM client with .config.context_window/.config.max_tokens.
        format_item: Callback (item) -> str that formats one item as a prompt block.
        stage_name: Name for logging and R3 validation.
        item_key: Optional callback (item) -> key that extracts a lookup key from
            each input item. When None, uses the item's index as key.
        default_action: Stage-specific R2 default label for omission logging.

    Returns:
        Tuple of (verdict_map, omitted_keys):
        - verdict_map: dict mapping key -> bool (True = relevant/yes, False = irrelevant/no).
        - omitted_keys: set of input item keys that had parse failures (R2 default-deny).
    """
    if not items:
        return {}, set()

    _require_logged_client(llm, "run_binary_judgment")

    verdict_map: dict[Hashable, bool] = {}
    omitted_keys: set[Hashable] = set()

    for idx, item in enumerate(items):
        key = item_key(item) if item_key is not None else idx
        item_text = format_item(item)
        prompt = task_context + item_text

        validate_prompt_budget(
            prompt, system_prompt,
            llm.config.context_window, llm.config.max_tokens,
            f"{stage_name}_binary",
        )

        response = llm.complete(prompt, system=system_prompt)
        answer = response.text.strip().lower()

        # Parse binary response: accept "yes"/"no" or minimal JSON
        if answer in ("yes", "no"):
            verdict_map[key] = answer == "yes"
        else:
            # Try JSON: {"verdict": "relevant"/"irrelevant"} or {"keep": true/false}
            try:
                parsed = parse_json_response(answer, f"{stage_name}_binary")
            except (ValueError, TypeError):
                # R2 default-deny: unparseable response for a single item.
                # This is expected with small classifiers — default to exclude.
                # Intentional: individual item parse failures are not fatal.
                logger.warning(
                    "R2: %s binary judgment unparseable for %s — defaulting to %s",
                    stage_name, str(key)[:100], default_action,
                )
                verdict_map[key] = False
                omitted_keys.add(key)
                continue

            if not isinstance(parsed, dict):
                # R2 default-deny: non-dict response (e.g. list or string).
                logger.warning(
                    "R2: %s binary judgment returned %s for %s — defaulting to %s",
                    stage_name, type(parsed).__name__, str(key)[:100], default_action,
                )
                verdict_map[key] = False
                omitted_keys.add(key)
                continue

            if "verdict" in parsed:
                verdict_map[key] = parsed["verdict"] in ("relevant", "yes", True)
            elif "keep" in parsed:
                verdict_map[key] = bool(parsed["keep"])
            else:
                # R2 default-deny: parsed JSON but missing expected keys.
                logger.warning(
                    "R2: %s binary judgment missing verdict/keep for %s — defaulting to %s",
                    stage_name, str(key)[:100], default_action,
                )
                verdict_map[key] = False
                omitted_keys.add(key)

    return verdict_map, omitted_keys
