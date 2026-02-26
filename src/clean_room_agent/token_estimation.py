"""Token estimation constants shared between LLM client and retrieval budget.

Centralizes chars-per-token ratios so the LLM input validation gate and
retrieval batch sizing always agree.  Zero internal imports — safe to import
from any layer without circular dependencies.
"""

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
