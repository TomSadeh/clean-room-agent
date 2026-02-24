"""Tests for token_estimation.py constants."""

from clean_room_agent.token_estimation import (
    CHARS_PER_TOKEN,
    CHARS_PER_TOKEN_CONSERVATIVE,
)


class TestTokenEstimationConstants:
    """Verify the chars-per-token constants maintain their safety invariant."""

    def test_conservative_is_stricter(self):
        """Conservative estimate must produce MORE tokens for the same input,
        so batch sizing never exceeds the LLM client validation gate."""
        assert CHARS_PER_TOKEN_CONSERVATIVE < CHARS_PER_TOKEN

    def test_planning_estimate_value(self):
        assert CHARS_PER_TOKEN == 4

    def test_conservative_estimate_value(self):
        assert CHARS_PER_TOKEN_CONSERVATIVE == 3

    def test_planning_estimate_positive(self):
        assert CHARS_PER_TOKEN > 0

    def test_conservative_estimate_positive(self):
        assert CHARS_PER_TOKEN_CONSERVATIVE > 0

    def test_token_counts_for_typical_code(self):
        """Verify estimates produce reasonable token counts for code."""
        code = "def hello_world():\n    return 'hello'\n"
        planning_tokens = len(code) // CHARS_PER_TOKEN
        conservative_tokens = len(code) // CHARS_PER_TOKEN_CONSERVATIVE
        # Conservative always >= planning
        assert conservative_tokens >= planning_tokens

    def test_empty_string_zero_tokens(self):
        assert len("") // CHARS_PER_TOKEN == 0
        assert len("") // CHARS_PER_TOKEN_CONSERVATIVE == 0

    def test_batch_gate_consistency(self):
        """If batch sizing (conservative) says N items fit, the client gate
        (also conservative) must agree. This is the core invariant from T5."""
        input_text = "x" * 3000  # 3000 chars
        # Both use CHARS_PER_TOKEN_CONSERVATIVE for safety gate
        gate_tokens = len(input_text) // CHARS_PER_TOKEN_CONSERVATIVE
        assert gate_tokens == 1000
        # Planning estimate is smaller (fewer tokens estimated)
        planning_tokens = len(input_text) // CHARS_PER_TOKEN
        assert planning_tokens == 750
        assert planning_tokens < gate_tokens
