"""Tests for budget module."""

import pytest

from clean_room_agent.retrieval.budget import (
    CHARS_PER_TOKEN,
    CHARS_PER_TOKEN_CONSERVATIVE,
    SAFETY_MARGIN,
    BudgetTracker,
    estimate_framing_tokens,
    estimate_tokens,
    estimate_tokens_conservative,
)
from clean_room_agent.retrieval.dataclasses import BudgetConfig


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 1  # max(1, 0)

    def test_short_string(self):
        assert estimate_tokens("ab") == 1  # max(1, 0)

    def test_four_chars(self):
        assert estimate_tokens("abcd") == 1

    def test_eight_chars(self):
        assert estimate_tokens("abcdefgh") == 2

    def test_typical_code(self):
        code = "def hello_world():\n    print('hello')\n"
        tokens = estimate_tokens(code)
        assert tokens == len(code) // 4

    def test_large_text(self):
        text = "x" * 4000
        assert estimate_tokens(text) == 1000


class TestBudgetConfigValidation:
    def test_valid(self):
        BudgetConfig(context_window=32768, reserved_tokens=4096)

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="context_window must be > 0"):
            BudgetConfig(context_window=0, reserved_tokens=0)

    def test_reserved_exceeds_window(self):
        with pytest.raises(ValueError, match="reserved_tokens"):
            BudgetConfig(context_window=1000, reserved_tokens=1000)


class TestBudgetTracker:
    def test_initial_state(self):
        config = BudgetConfig(context_window=10000, reserved_tokens=2000)
        tracker = BudgetTracker(config)
        expected = int(8000 * SAFETY_MARGIN)
        assert tracker.effective_limit == expected
        assert tracker.remaining == expected
        assert tracker.used == 0

    def test_can_fit(self):
        config = BudgetConfig(context_window=10000, reserved_tokens=2000)
        tracker = BudgetTracker(config)
        assert tracker.can_fit(100)
        assert tracker.can_fit(tracker.effective_limit)
        assert not tracker.can_fit(tracker.effective_limit + 1)

    def test_consume(self):
        config = BudgetConfig(context_window=10000, reserved_tokens=2000)
        tracker = BudgetTracker(config)
        tracker.consume(1000)
        assert tracker.used == 1000
        assert tracker.remaining == tracker.effective_limit - 1000

    def test_consume_to_zero(self):
        config = BudgetConfig(context_window=10000, reserved_tokens=2000)
        tracker = BudgetTracker(config)
        tracker.consume(tracker.effective_limit)
        assert tracker.remaining == 0
        assert not tracker.can_fit(1)

    def test_consume_past_limit(self):
        config = BudgetConfig(context_window=10000, reserved_tokens=2000)
        tracker = BudgetTracker(config)
        tracker.consume(tracker.effective_limit + 100)
        assert tracker.remaining == 0  # clamped to 0

    def test_safety_margin_applied(self):
        config = BudgetConfig(context_window=10000, reserved_tokens=0)
        tracker = BudgetTracker(config)
        # effective = 10000 * 0.9 = 9000
        assert tracker.effective_limit == 9000

    def test_consume_negative_raises(self):
        config = BudgetConfig(context_window=10000, reserved_tokens=2000)
        tracker = BudgetTracker(config)
        with pytest.raises(ValueError, match="negative"):
            tracker.consume(-1)


class TestEstimateTokensConservative:
    """T5: conservative estimator for batch sizing and input validation."""

    def test_empty_string(self):
        assert estimate_tokens_conservative("") == 1

    def test_three_chars(self):
        assert estimate_tokens_conservative("abc") == 1

    def test_six_chars(self):
        assert estimate_tokens_conservative("abcdef") == 2

    def test_always_gte_planning_estimate(self):
        """Conservative estimate must always be >= planning estimate."""
        for text in ["short", "x" * 100, "y" * 4000, "z" * 12000]:
            assert estimate_tokens_conservative(text) >= estimate_tokens(text)

    def test_uses_conservative_ratio(self):
        text = "x" * 3000
        assert estimate_tokens_conservative(text) == 3000 // CHARS_PER_TOKEN_CONSERVATIVE

    def test_conservative_is_stricter(self):
        """Conservative ratio produces larger token estimates (fewer chars per token)."""
        assert CHARS_PER_TOKEN_CONSERVATIVE < CHARS_PER_TOKEN


class TestTokenEstimationConsistency:
    """T5: verify batch sizing and LLM client validation use the same threshold.

    The invariant: a prompt sized to fit at the conservative estimate must not
    be rejected by LLMClient.complete()'s input validation, which uses the
    same conservative ratio.
    """

    def test_constants_match_client(self):
        """Budget constants re-exported from token_estimation match client's."""
        from clean_room_agent.token_estimation import (
            CHARS_PER_TOKEN_CONSERVATIVE as SOURCE_CONST,
        )
        # budget.py re-exports the same constant
        assert CHARS_PER_TOKEN_CONSERVATIVE == SOURCE_CONST
        # The constant that client.py uses comes from the same source module
        from clean_room_agent.llm.client import CHARS_PER_TOKEN_CONSERVATIVE as CLIENT_CONST
        assert CLIENT_CONST == SOURCE_CONST

    def test_overhead_estimation_uses_conservative_ratio(self):
        """Batch sizing overhead (system + header) uses the same ratio as LLMClient.

        Before T5 fix, batch sizing used chars//4 for overhead while
        LLMClient.complete() validated at chars//3, so a batch sized to fit
        at //4 could be rejected at //3.
        """
        system_text = "You are a code retrieval judge." * 10  # realistic system prompt
        header_text = "Task: refactor authentication\nIntent: split module\n\nCandidates:\n"

        # The overhead estimates used by batch sizing
        system_overhead = estimate_tokens_conservative(system_text)
        header_overhead = estimate_tokens_conservative(header_text)

        # These must be >= what the client would compute
        # Client computes: len(total) // CHARS_PER_TOKEN_CONSERVATIVE
        # Since estimate_tokens_conservative = max(1, len//3), the sum of
        # individual estimates may differ from total estimate by at most
        # 1 per component (from max(1,...) and integer truncation).
        # But crucially, the RATIO is the same.
        client_system = len(system_text) // CHARS_PER_TOKEN_CONSERVATIVE
        client_header = len(header_text) // CHARS_PER_TOKEN_CONSERVATIVE

        # Conservative overhead should be >= what client would compute
        # (max(1,...) can only increase, never decrease)
        assert system_overhead >= client_system
        assert header_overhead >= client_header

    def test_old_ratio_would_underestimate(self):
        """Demonstrate that the old chars//4 ratio underestimates vs client's chars//3.

        This is the bug T5 fixed: batch sizing at //4 created larger batches
        than the client would accept at //3.
        """
        text = "x" * 12000  # large enough to see the gap clearly

        old_estimate = len(text) // 4      # 3000 tokens (what batch sizing used before)
        new_estimate = len(text) // CHARS_PER_TOKEN_CONSERVATIVE  # 4000 tokens (what client uses)

        # The old estimate is 25% lower — batches sized at //4 could exceed
        # the client's //3 threshold by up to 33%
        assert old_estimate < new_estimate
        assert new_estimate == estimate_tokens_conservative(text)


class TestEstimateFramingTokens:
    """R5: Framing overhead estimation."""

    def test_framing_tokens_positive(self):
        tokens = estimate_framing_tokens("src/auth.py", "python", "primary")
        assert tokens > 0

    def test_framing_includes_header_and_tags(self):
        tokens = estimate_framing_tokens("src/auth.py", "python", "primary")
        # Header "## src/auth.py [python] (primary)\n" + open/close tags
        # Should be roughly len("## src/auth.py [python] (primary)\n<code lang=\"python\">\n</code>\n") // 4
        expected_text = '## src/auth.py [python] (primary)\n<code lang="python">\n</code>\n'
        assert tokens == len(expected_text) // 4
