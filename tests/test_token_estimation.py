"""Tests for token_estimation.py constants and budget validation (A15)."""

import pytest

from clean_room_agent.token_estimation import (
    CHARS_PER_TOKEN,
    CHARS_PER_TOKEN_CONSERVATIVE,
    check_prompt_budget,
    validate_prompt_budget,
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


class TestCheckPromptBudget:
    """A15: check_prompt_budget returns (estimated, available, fits) tuple."""

    def test_fits_when_under_budget(self):
        prompt = "a" * 30
        system = "b" * 30
        # 60 chars // 3 = 20 tokens estimated; available = 100 - 30 = 70
        estimated, available, fits = check_prompt_budget(prompt, system, 100, 30)
        assert estimated == 20
        assert available == 70
        assert fits is True

    def test_does_not_fit_when_over_budget(self):
        prompt = "a" * 300
        system = "b" * 300
        # 600 chars // 3 = 200 tokens estimated; available = 100 - 30 = 70
        estimated, available, fits = check_prompt_budget(prompt, system, 100, 30)
        assert estimated == 200
        assert available == 70
        assert fits is False

    def test_exact_boundary_fits(self):
        chars = 70 * CHARS_PER_TOKEN_CONSERVATIVE  # 210 chars total
        prompt = "a" * chars
        estimated, available, fits = check_prompt_budget(prompt, "", 100, 30)
        assert estimated == 70
        assert available == 70
        assert fits is True


class TestValidatePromptBudget:
    """A15: validate_prompt_budget raises ValueError on budget exceeded."""

    def test_passes_when_under_budget(self):
        validate_prompt_budget("short", "sys", 10000, 100, "test_stage")

    def test_raises_with_stage_name(self):
        prompt = "a" * 3000
        system = "b" * 3000
        with pytest.raises(ValueError, match=r"R3: my_stage prompt too large"):
            validate_prompt_budget(prompt, system, 100, 30, "my_stage")

    def test_error_message_includes_token_counts(self):
        prompt = "a" * 300
        system = "b" * 300
        with pytest.raises(ValueError, match=r"200 tokens.*available 70"):
            validate_prompt_budget(prompt, system, 100, 30, "x")
