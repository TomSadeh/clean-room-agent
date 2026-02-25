"""Tests for pipeline trace log (TraceLogger)."""

from pathlib import Path

import pytest

from clean_room_agent.trace import TraceLogger


def _make_call(
    *,
    system="You are helpful",
    prompt="What is 2+2?",
    response="4",
    thinking="",
    prompt_tokens=10,
    completion_tokens=5,
    elapsed_ms=100,
    error="",
):
    """Build a call dict with sensible defaults, allowing per-field overrides."""
    return {
        "system": system,
        "prompt": prompt,
        "response": response,
        "thinking": thinking,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_ms": elapsed_ms,
        "error": error,
    }


class TestTraceLogger:
    def test_format_basic_call(self, tmp_path):
        """Log a single call, verify stage name, response, and system prompt details tag."""
        out = tmp_path / "trace.md"
        logger = TraceLogger(out, task_id="task-001", task_description="basic test")
        logger.log_calls("scope", "retrieval", [_make_call()])
        path = logger.finalize()

        content = path.read_text(encoding="utf-8")
        # Stage name appears in call header
        assert "scope" in content
        # Response text rendered inline
        assert "4" in content
        # System prompt inside a <details> tag
        assert "<details>" in content
        assert "You are helpful" in content

    def test_format_thinking_content(self, tmp_path):
        """Log a call with thinking content, verify thinking section rendered."""
        out = tmp_path / "trace.md"
        logger = TraceLogger(out, task_id="task-002", task_description="thinking test")
        logger.log_calls(
            "precision",
            "classification",
            [_make_call(thinking="Let me reason step by step about this problem")],
        )
        path = logger.finalize()

        content = path.read_text(encoding="utf-8")
        assert "### Thinking" in content
        assert "Let me reason step by step about this problem" in content

    def test_format_error_call(self, tmp_path):
        """Log a call with error info, verify error section appears."""
        out = tmp_path / "trace.md"
        logger = TraceLogger(out, task_id="task-003", task_description="error test")
        logger.log_calls(
            "scope",
            "retrieval",
            [_make_call(error="Connection timed out after 30s")],
        )
        path = logger.finalize()

        content = path.read_text(encoding="utf-8")
        assert "### Error" in content
        assert "Connection timed out after 30s" in content

    def test_empty_trace(self, tmp_path):
        """Finalize with no calls produces header-only file with task_id and 0 calls."""
        out = tmp_path / "trace.md"
        logger = TraceLogger(out, task_id="task-004", task_description="empty trace")
        path = logger.finalize()

        content = path.read_text(encoding="utf-8")
        assert "task-004" in content
        assert "0 calls" in content

    def test_summary_stats(self, tmp_path):
        """Log multiple calls with known token counts and latencies, verify aggregates."""
        out = tmp_path / "trace.md"
        logger = TraceLogger(out, task_id="task-005", task_description="stats test")
        logger.log_calls(
            "scope",
            "retrieval",
            [
                _make_call(prompt_tokens=100, completion_tokens=50, elapsed_ms=200),
                _make_call(prompt_tokens=150, completion_tokens=75, elapsed_ms=300),
            ],
        )
        logger.log_calls(
            "precision",
            "classification",
            [_make_call(prompt_tokens=80, completion_tokens=40, elapsed_ms=150)],
        )
        path = logger.finalize()

        content = path.read_text(encoding="utf-8")
        # 3 total calls
        assert "3 calls" in content
        # Prompt tokens: 100 + 150 + 80 = 330
        assert "Prompt tokens: 330" in content
        # Completion tokens: 50 + 75 + 40 = 165
        assert "Completion tokens: 165" in content
        # Total latency: 200 + 300 + 150 = 650
        assert "Total latency: 650ms" in content

    def test_missing_tokens_annotated(self, tmp_path):
        """When token counts are None, annotated as N/A in call section and summary."""
        out = tmp_path / "trace.md"
        logger = TraceLogger(out, task_id="task-006", task_description="na test")
        logger.log_calls(
            "scope",
            "retrieval",
            [_make_call(prompt_tokens=None, completion_tokens=None)],
        )
        path = logger.finalize()

        content = path.read_text(encoding="utf-8")
        # Summary header shows N/A for both token types
        assert "Prompt tokens: N/A" in content
        assert "Completion tokens: N/A" in content

    def test_output_file_created(self, tmp_path):
        """Finalize creates file and parent directories for nested path."""
        out = tmp_path / "deep" / "nested" / "dir" / "trace.md"
        logger = TraceLogger(out, task_id="task-007", task_description="mkdir test")
        logger.log_calls("scope", "retrieval", [_make_call()])
        path = logger.finalize()

        assert path.exists()
        assert path.is_file()
        assert path == out
        # Verify parent directories were created
        assert (tmp_path / "deep" / "nested" / "dir").is_dir()

    def test_multiple_stages(self, tmp_path):
        """Log calls from different stages, verify they appear in order with correct headers."""
        out = tmp_path / "trace.md"
        logger = TraceLogger(out, task_id="task-008", task_description="multi-stage")

        logger.log_calls(
            "task_analysis",
            "analysis",
            [_make_call(response="Parsed task intent")],
        )
        logger.log_calls(
            "scope",
            "retrieval",
            [_make_call(response="Found 5 relevant files")],
        )
        logger.log_calls(
            "precision",
            "classification",
            [_make_call(response="Classified symbols at detail levels")],
        )
        path = logger.finalize()

        content = path.read_text(encoding="utf-8")

        # All three stage names present
        assert "task_analysis" in content
        assert "scope" in content
        assert "precision" in content

        # Verify ordering: task_analysis before scope before precision
        pos_analysis = content.index("task_analysis")
        pos_scope = content.index("scope", pos_analysis + 1)
        pos_precision = content.index("precision", pos_scope + 1)
        assert pos_analysis < pos_scope < pos_precision

        # All responses present
        assert "Parsed task intent" in content
        assert "Found 5 relevant files" in content
        assert "Classified symbols at detail levels" in content

    def test_content_injection_details_tag(self, tmp_path):
        """Content with </details> must not corrupt the trace output (4-P0-1 regression)."""
        out = tmp_path / "trace.md"
        logger = TraceLogger(out, task_id="task-inject-1", task_description="injection test")
        logger.log_calls(
            "scope",
            "retrieval",
            [_make_call(system="sys</details>tem", prompt="pro</details>mpt", response="res</details>ponse")],
        )
        path = logger.finalize()

        content = path.read_text(encoding="utf-8")
        # The injected </details> should be inside a code fence, not breaking structure
        # All 3 section headers must be present and intact
        assert "### System Prompt" in content
        assert "### User Prompt" in content
        assert "### Response" in content
        # The </details> in content must not prematurely close the real <details> blocks
        # Count opening vs closing tags — each <details> should have exactly one </details>
        # With 2 collapsible sections (system + prompt), we expect 2 pairs + injected ones in fences
        assert content.count("<details>") == 2  # system + prompt sections

    def test_content_injection_backticks(self, tmp_path):
        """Content with triple backticks must not break fences (4-P0-2 regression)."""
        out = tmp_path / "trace.md"
        logger = TraceLogger(out, task_id="task-inject-2", task_description="backtick test")
        response_with_backticks = "Here is code:\n```python\nprint('hello')\n```\nDone."
        logger.log_calls(
            "scope",
            "retrieval",
            [_make_call(response=response_with_backticks)],
        )
        path = logger.finalize()

        content = path.read_text(encoding="utf-8")
        # The response must be fully contained — check the whole response text is present
        assert "print('hello')" in content
        assert "Here is code:" in content
        assert "Done." in content
        # The fence wrapping the response should be longer than 3 backticks
        assert "````" in content

    def test_none_system_prompt(self, tmp_path):
        """Calls with system=None should render without a system prompt section (4-P1-6)."""
        out = tmp_path / "trace.md"
        logger = TraceLogger(out, task_id="task-none-sys", task_description="none system test")
        logger.log_calls(
            "scope",
            "retrieval",
            [_make_call(system=None)],
        )
        path = logger.finalize()

        content = path.read_text(encoding="utf-8")
        # No system prompt section should be rendered
        assert "### System Prompt" not in content
        # But the response should still be present
        assert "### Response" in content
        assert "4" in content

    def test_double_finalize_raises(self, tmp_path):
        """Calling finalize() twice should raise RuntimeError (4-P1-4)."""
        out = tmp_path / "trace.md"
        logger = TraceLogger(out, task_id="task-double", task_description="double finalize")
        logger.log_calls("scope", "retrieval", [_make_call()])
        logger.finalize()
        with pytest.raises(RuntimeError, match="already called"):
            logger.finalize()

    def test_update_task_id(self, tmp_path):
        """update_task_id should change the task_id in the trace header (4-P0-3)."""
        out = tmp_path / "trace.md"
        logger = TraceLogger(out, task_id="temp-id", task_description="update test")
        logger.update_task_id("real-task-id-123")
        logger.log_calls("scope", "retrieval", [_make_call()])
        path = logger.finalize()

        content = path.read_text(encoding="utf-8")
        assert "real-task-id-123" in content
        assert "temp-id" not in content
