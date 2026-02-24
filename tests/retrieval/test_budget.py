"""Tests for budget module."""

import pytest

from clean_room_agent.retrieval.budget import (
    SAFETY_MARGIN,
    BudgetTracker,
    estimate_framing_tokens,
    estimate_tokens,
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
