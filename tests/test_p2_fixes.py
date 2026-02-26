"""Tests for P2 fixes: T62-T71."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from clean_room_agent.llm.client import (
    LoggedLLMClient,
    strip_thinking,
)


# ── T63: cra enrich preflight check ─────────────────────────────────


class TestEnrichPreflightCheck:
    """T63: cra enrich should fail fast if curated DB doesn't exist."""

    def test_missing_curated_db_raises(self, tmp_path):
        from click.testing import CliRunner
        from clean_room_agent.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["enrich", str(tmp_path)])
        assert result.exit_code != 0
        assert "cra index" in result.output or "cra index" in str(result.exception)

    def test_existing_curated_db_proceeds(self, tmp_path):
        """If curated DB exists, enrich should proceed past preflight."""
        from click.testing import CliRunner
        from clean_room_agent.cli import cli

        # Create minimal curated DB
        clean_room = tmp_path / ".clean_room"
        clean_room.mkdir()
        (clean_room / "curated.sqlite").write_text("")

        runner = CliRunner()
        # Will fail later (no config.toml) but should pass the preflight check
        result = runner.invoke(cli, ["enrich", str(tmp_path)])
        assert "cra index" not in (result.output or "")


# ── T64: Bounds-check config values ─────────────────────────────────


class TestBoundsChecks:
    """T64: max_cumulative_diff_chars and timeout must be positive."""

    def test_negative_max_diff_chars_raises(self):
        from clean_room_agent.orchestrator.runner import run_orchestrator

        config = {
            "models": {
                "provider": "ollama",
                "coding": "m", "reasoning": "m",
                "base_url": "http://localhost:11434",
                "context_window": 32768,
            },
            "budget": {"reserved_tokens": 4096},
            "stages": {"default": "scope,precision"},
            "orchestrator": {
                "max_retries_per_step": 1,
                "max_adjustment_rounds": 3,
                "git_workflow": False,
                "max_cumulative_diff_chars": -1,
            },
        }
        with pytest.raises(RuntimeError, match="positive integer"):
            run_orchestrator("task", MagicMock(), config)

    def test_zero_max_diff_chars_raises(self):
        from clean_room_agent.orchestrator.runner import run_orchestrator

        config = {
            "models": {
                "provider": "ollama",
                "coding": "m", "reasoning": "m",
                "base_url": "http://localhost:11434",
                "context_window": 32768,
            },
            "budget": {"reserved_tokens": 4096},
            "stages": {"default": "scope,precision"},
            "orchestrator": {
                "max_retries_per_step": 1,
                "max_adjustment_rounds": 3,
                "git_workflow": False,
                "max_cumulative_diff_chars": 0,
            },
        }
        with pytest.raises(RuntimeError, match="positive integer"):
            run_orchestrator("task", MagicMock(), config)

    def test_negative_timeout_raises(self):
        from clean_room_agent.orchestrator.validator import run_validation

        config = {"testing": {"test_command": "echo ok", "timeout": -5}}
        with pytest.raises(RuntimeError, match="positive integer"):
            run_validation(MagicMock(), config, MagicMock(), 1)

    def test_zero_timeout_raises(self):
        from clean_room_agent.orchestrator.validator import run_validation

        config = {"testing": {"test_command": "echo ok", "timeout": 0}}
        with pytest.raises(RuntimeError, match="positive integer"):
            run_validation(MagicMock(), config, MagicMock(), 1)


# ── T65: Validate test_command as non-empty ──────────────────────────


class TestValidateTestCommand:
    """T65: Empty test_command should fail validation."""

    def test_empty_string_raises(self):
        from clean_room_agent.orchestrator.validator import require_testing_config

        with pytest.raises(RuntimeError, match="Missing or empty test_command"):
            require_testing_config({"testing": {"test_command": ""}})

    def test_valid_command_passes(self):
        from clean_room_agent.orchestrator.validator import require_testing_config

        result = require_testing_config({"testing": {"test_command": "pytest"}})
        assert result["test_command"] == "pytest"


# ── T66: Adjustment failure saved to session state ──────────────────


class TestAdjustmentFailureState:
    """T66: Failed adjustment should record to session state."""

    def test_failure_recorded(self):
        """The adjustment except block now calls set_state with success=False."""
        from clean_room_agent.db.session_helpers import set_state

        # This is tested indirectly via the orchestrator. We verify
        # the code structure by checking set_state is called.
        # Direct structural test: the except block has set_state.
        import inspect
        from clean_room_agent.orchestrator import runner

        source = inspect.getsource(runner.run_orchestrator)
        # The except block after adjustment should contain set_state with success: False
        assert '"success": False, "error": str(e)' in source


# ── T67: run_single_pass creates DB records ─────────────────────────


class TestSinglePassDBRecords:
    """T67: run_single_pass should create orchestrator_run/pass records."""

    def test_has_orchestrator_records(self):
        """Verify the function creates orchestrator run/pass records (via helpers)."""
        import inspect
        from clean_room_agent.orchestrator import runner

        # run_single_pass delegates init/cleanup to shared helpers
        source = inspect.getsource(runner.run_single_pass)
        assert "_init_orchestrator" in source
        assert "_cleanup_orchestrator" in source
        assert "insert_orchestrator_pass" in source
        # Verify the shared helpers contain the DB operations
        init_src = inspect.getsource(runner._init_orchestrator)
        assert "insert_orchestrator_run" in init_src
        cleanup_src = inspect.getsource(runner._cleanup_orchestrator)
        assert "_archive_session" in cleanup_src
        assert "_finalize_orchestrator_run" in cleanup_src

    def test_has_session_db(self):
        """Verify the function creates and archives a session DB (via helpers)."""
        import inspect
        from clean_room_agent.orchestrator import runner

        init_src = inspect.getsource(runner._init_orchestrator)
        assert 'get_connection("session"' in init_src
        cleanup_src = inspect.getsource(runner._cleanup_orchestrator)
        assert "session_conn.close()" in cleanup_src


# ── T68: strip_thinking handles edge cases ──────────────────────────


class TestStripThinking:
    """T68: strip_thinking should handle nested and malformed tags."""

    def test_simple_thinking(self):
        text = "<think>reasoning here</think>answer"
        clean, thinking = strip_thinking(text)
        assert clean == "answer"
        assert thinking == "reasoning here"

    def test_no_thinking(self):
        text = "just an answer"
        clean, thinking = strip_thinking(text)
        assert clean == "just an answer"
        assert thinking is None

    def test_nested_think_tags(self):
        """Outermost block should be extracted."""
        text = "<think>outer <think>inner</think> still outer</think>answer"
        clean, thinking = strip_thinking(text)
        assert clean == "answer"
        # rfind finds outermost </think>, so everything between first <think> and last </think>
        assert "outer" in thinking
        assert "inner" in thinking

    def test_unclosed_think_tag(self):
        """Unclosed tag: everything after <think> is thinking content."""
        text = "<think>my reasoning without closing"
        clean, thinking = strip_thinking(text)
        assert clean == ""
        assert thinking == "my reasoning without closing"

    def test_whitespace_stripping(self):
        text = "<think>\n  reasoning  \n</think>\n  answer  "
        clean, thinking = strip_thinking(text)
        assert clean == "answer"
        assert thinking == "reasoning"

    def test_empty_thinking(self):
        text = "<think></think>answer"
        clean, thinking = strip_thinking(text)
        assert clean == "answer"
        # Empty thinking stripped to None
        assert thinking is None

    def test_text_before_think_tag(self):
        text = "prefix <think>reasoning</think> suffix"
        clean, thinking = strip_thinking(text)
        assert "prefix" in clean
        assert "suffix" in clean
        assert thinking == "reasoning"


# ── T69: num_predict clamped to positive ─────────────────────────────


class TestNumPredictClamp:
    """T69: num_predict should be clamped to at least 1."""

    def test_clamp_in_source(self):
        """Verify the max(1, ...) clamp exists."""
        import inspect
        from clean_room_agent.llm import client

        source = inspect.getsource(client.LLMClient.complete)
        assert "max(1," in source


# ── T70: LoggedLLMClient logs failed calls ──────────────────────────


class TestLoggedLLMClientFailedCalls:
    """T70: Failed LLM calls should be recorded for traceability."""

    def test_failed_call_recorded(self):
        config = MagicMock()
        config.provider = "ollama"
        config.context_window = 32768
        config.max_tokens = 4096

        logged = LoggedLLMClient(config)
        # Make the inner client's complete method raise
        logged._client.complete = MagicMock(side_effect=RuntimeError("connection refused"))

        with pytest.raises(RuntimeError, match="connection refused"):
            logged.complete("test prompt", system="test system")

        # The failed call should be recorded
        calls = logged.flush()
        assert len(calls) == 1
        assert calls[0]["prompt"] == "test prompt"
        assert calls[0]["system"] == "test system"
        assert "[ERROR]" in calls[0]["response"]
        assert calls[0]["prompt_tokens"] is None
        assert calls[0]["completion_tokens"] is None
        assert "error" in calls[0]

    def test_successful_call_still_works(self):
        config = MagicMock()
        config.provider = "ollama"
        config.context_window = 32768
        config.max_tokens = 4096

        logged = LoggedLLMClient(config)
        mock_response = MagicMock()
        mock_response.text = "result"
        mock_response.thinking = None
        mock_response.prompt_tokens = 10
        mock_response.completion_tokens = 5
        mock_response.latency_ms = 100
        logged._client.complete = MagicMock(return_value=mock_response)

        result = logged.complete("prompt")
        assert result.text == "result"

        calls = logged.flush()
        assert len(calls) == 1
        assert calls[0]["response"] == "result"
        assert "error" not in calls[0]


# ── T71: Null checks after .fetchone() ──────────────────────────────


class TestFetchoneNullChecks:
    """T71: All .fetchone() calls should have null checks."""

    def test_null_checks_present(self):
        """Verify task_run fetchone null check exists (T80: consolidated in _get_task_run_id)."""
        import inspect
        from clean_room_agent.orchestrator import runner

        # T80 extracted null check into _get_task_run_id helper.
        # Verify the helper exists and contains the null check.
        helper_source = inspect.getsource(runner._get_task_run_id)
        assert "is None" in helper_source
        assert "RuntimeError" in helper_source

        # Verify all 7 call sites use the helper
        full_source = inspect.getsource(runner)
        count = full_source.count("_get_task_run_id(")
        # 7 sites: meta_plan, part_plan, impl, adjust, test_plan, test_impl, single_pass
        assert count >= 7, f"Expected at least 7 _get_task_run_id calls, found {count}"
