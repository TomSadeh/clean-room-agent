"""Token estimation constants shared between LLM client and retrieval budget.

Centralizes chars-per-token ratios so the LLM input validation gate and
retrieval batch sizing always agree.  Zero internal imports — safe to import
from any layer without circular dependencies.
"""

import logging

logger = logging.getLogger(__name__)

# Planning estimate: ~4 chars per token.  Used for budget tracking in context
# assembly.  The 0.9 safety margin in BudgetTracker provides additional buffer.
CHARS_PER_TOKEN = 4

# Safety-gate estimate: ~3 chars per token.  Used for input validation in
# LLMClient.complete() and for batch sizing in retrieval stages.  Batching
# must use this ratio so constructed prompts are not rejected at the gate.
CHARS_PER_TOKEN_CONSERVATIVE = 3


def check_prompt_budget(
    prompt: str, system: str, context_window: int, max_tokens: int,
) -> tuple[int, int, bool]:
    """Estimate tokens and check fit.  Returns (estimated, available, fits)."""
    estimated = (len(prompt) + len(system)) // CHARS_PER_TOKEN_CONSERVATIVE
    available = context_window - max_tokens
    return estimated, available, estimated <= available


def validate_prompt_budget(
    prompt: str, system: str, context_window: int, max_tokens: int, stage_name: str,
) -> None:
    """R3: Raise ValueError if prompt exceeds context budget."""
    estimated, available, fits = check_prompt_budget(prompt, system, context_window, max_tokens)
    if not fits:
        raise ValueError(
            f"R3: {stage_name} prompt too large ({estimated} tokens, available {available})"
        )


def budget_truncate(
    content: str,
    context_window: int,
    max_tokens: int,
    *,
    max_content_fraction: float,
    stage_name: str,
    keep: str = "head",
) -> str:
    """Truncate content to fit within a fraction of the available context budget.

    Computes max_chars from (context_window - max_tokens) * CHARS_PER_TOKEN_CONSERVATIVE
    * max_content_fraction. Logs a warning when truncation occurs.

    Args:
        content: The text to potentially truncate.
        context_window: Model context window in tokens.
        max_tokens: Reserved output tokens.
        max_content_fraction: Fraction of available input budget for this content.
        stage_name: For logging which stage triggered truncation.
        keep: "head" keeps the beginning, "tail" keeps the end.

    Returns:
        The (possibly truncated) content string.

    Raises:
        ValueError: If the computed budget is non-positive.
    """
    available_tokens = context_window - max_tokens
    max_chars = int(available_tokens * CHARS_PER_TOKEN_CONSERVATIVE * max_content_fraction)

    if max_chars <= 0:
        raise ValueError(
            f"budget_truncate: non-positive budget for {stage_name} "
            f"(context_window={context_window}, max_tokens={max_tokens}, "
            f"fraction={max_content_fraction})"
        )

    if len(content) <= max_chars:
        return content

    logger.warning(
        "budget_truncate: %s content truncated from %d to %d chars "
        "(keep=%s, fraction=%.2f, available_tokens=%d)",
        stage_name, len(content), max_chars, keep, max_content_fraction, available_tokens,
    )

    if keep == "tail":
        return content[-max_chars:]
    return content[:max_chars]
