"""Token budget estimation and tracking."""

from clean_room_agent.retrieval.dataclasses import BudgetConfig

SAFETY_MARGIN = 0.9


def estimate_tokens(text: str) -> int:
    """Estimate token count from text. Conservative: ~4 chars per token."""
    return max(1, len(text) // 4)


def estimate_framing_tokens(path: str, language: str, detail_level: str) -> int:
    """R5: Estimate token overhead for file framing in prompt output.

    Accounts for: header line, opening/closing code tags, newlines.
    """
    header = f"## {path} [{language}] ({detail_level})\n"
    open_tag = f"<code lang=\"{language}\">\n"
    close_tag = "</code>\n"
    framing = header + open_tag + close_tag
    return len(framing) // 4


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
