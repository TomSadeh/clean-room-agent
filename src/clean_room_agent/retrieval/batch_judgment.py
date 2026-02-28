"""Binary judgment utilities for LLM classification stages.

All judgment stages use per-item binary calls via run_binary_judgment().
Each item gets its own LLM call producing a yes/no verdict, enabling
small classifier models (0.6B) and clean per-item training data.
"""

import logging
from collections.abc import Callable, Hashable, Sequence

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
