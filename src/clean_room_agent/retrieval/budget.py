"""Token budget estimation and tracking."""

from clean_room_agent.retrieval.dataclasses import BudgetConfig
from clean_room_agent.token_estimation import (
    CHARS_PER_TOKEN,
    CHARS_PER_TOKEN_CONSERVATIVE,
)

SAFETY_MARGIN = 0.9


def estimate_tokens(text: str) -> int:
    """Estimate token count from text. ~4 chars per token for budget planning."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_tokens_conservative(text: str) -> int:
    """Conservative token estimate for input validation and batch sizing.

    Uses ~3 chars per token to match LLMClient.complete() rejection threshold.
    Use this when computing batch sizes or validating prompt size against
    context window limits, to ensure prompts won't be rejected downstream.
    """
    return max(1, len(text) // CHARS_PER_TOKEN_CONSERVATIVE)


def estimate_framing_tokens(path: str, language: str, detail_level: str) -> int:
    """R5: Estimate token overhead for file framing in prompt output.

    Accounts for: header line, opening/closing code tags, newlines.
    """
    header = f"## {path} [{language}] ({detail_level})\n"
    open_tag = f"<code lang=\"{language}\">\n"
    close_tag = "</code>\n"
    framing = header + open_tag + close_tag
    return len(framing) // CHARS_PER_TOKEN


class BudgetTracker:
    """Tracks token budget consumption during context assembly."""

    def __init__(self, config: BudgetConfig):
        self._effective = int(config.effective_budget * SAFETY_MARGIN)
        self._used = 0

    @property
    def remaining(self) -> int:
        return max(0, self._effective - self._used)

    @property
    def used(self) -> int:
        return self._used

    @property
    def effective_limit(self) -> int:
        return self._effective

    def can_fit(self, tokens: int) -> bool:
        return tokens <= self.remaining

    def consume(self, tokens: int) -> None:
        if tokens < 0:
            raise ValueError(f"Cannot consume negative tokens: {tokens}")
        self._used += tokens
