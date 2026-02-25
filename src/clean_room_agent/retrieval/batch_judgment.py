"""Shared batching utilities for LLM judgment stages (T83).

Used by scope, precision, and similarity stages to calculate batch sizes
and validate prompts against the model's context window (R3).
"""

from clean_room_agent.retrieval.budget import estimate_tokens_conservative


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
