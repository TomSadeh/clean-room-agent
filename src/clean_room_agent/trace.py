"""Pipeline trace log: human-readable markdown trace of LLM calls."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


import re


def _safe_fence(content: str) -> str:
    """Return a backtick fence longer than any backtick run in content."""
    longest = 0
    for m in re.finditer(r"`+", content):
        longest = max(longest, len(m.group()))
    return "`" * max(longest + 1, 3)


class TraceLogger:
    """Accumulates LLM call records and writes a human-readable markdown file."""

    def __init__(self, output_path: Path, task_id: str, task_description: str):
        self._output_path = output_path
        self._task_id = task_id
        self._task_description = task_description
        self._calls: list[dict] = []
        self._finalized = False

    def update_task_id(self, new_id: str) -> None:
        """Update the task_id (called by orchestrator after generating the real one)."""
        self._task_id = new_id

    def log_calls(
        self, stage_name: str, call_type: str, calls: list[dict], model: str,
    ) -> None:
        """Record a batch of flushed call dicts from LoggedLLMClient.flush()."""
        for call in calls:
            self._calls.append({
                "stage_name": stage_name,
                "call_type": call_type,
                "model": model,
                "system": call.get("system") or "",
                "prompt": call.get("prompt") or "",
                "response": call.get("response") or "",
                "thinking": call.get("thinking") or "",
                "prompt_tokens": call.get("prompt_tokens"),
                "completion_tokens": call.get("completion_tokens"),
                "elapsed_ms": call.get("elapsed_ms"),
                "error": call.get("error") or "",
            })

    def finalize(self) -> Path:
        """Write markdown trace to disk. Returns the output path.

        Raises RuntimeError if called more than once.
        """
        if self._finalized:
            raise RuntimeError("TraceLogger.finalize() already called")
        self._finalized = True
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        parts = [self._build_header()]
        for i, call in enumerate(self._calls, 1):
            parts.append(self._format_call(i, call))

        self._output_path.write_text("\n".join(parts), encoding="utf-8")
        return self._output_path

    def _build_header(self) -> str:
        """Render summary header with aggregate stats."""
        total_calls = len(self._calls)
        total_prompt = 0
        total_completion = 0
        total_latency = 0
        has_none_prompt = False
        has_none_completion = False
        has_none_latency = False

        for call in self._calls:
            pt = call.get("prompt_tokens")
            ct = call.get("completion_tokens")
            ms = call.get("elapsed_ms")
            if pt is None:
                has_none_prompt = True
            else:
                total_prompt += pt
            if ct is None:
                has_none_completion = True
            else:
                total_completion += ct
            if ms is None:
                has_none_latency = True
            else:
                total_latency += ms

        prompt_str = "N/A" if has_none_prompt else str(total_prompt)
        completion_str = "N/A" if has_none_completion else str(total_completion)
        latency_str = "N/A" if has_none_latency else f"{total_latency}ms"

        timestamp = datetime.now(timezone.utc).isoformat()

        return (
            f"# Pipeline Trace: {self._task_id}\n\n"
            f"**Generated:** {timestamp}\n\n"
            f"**Task:** {self._task_description}\n\n"
            f"**Summary:** {total_calls} calls | "
            f"Prompt tokens: {prompt_str} | "
            f"Completion tokens: {completion_str} | "
            f"Total latency: {latency_str}\n\n"
            f"---\n"
        )

    def _format_call(self, call_number: int, call: dict) -> str:
        """Render one call as a markdown section."""
        stage = call["stage_name"]
        call_type = call["call_type"]
        model = call.get("model", "")
        pt = call.get("prompt_tokens")
        ct = call.get("completion_tokens")
        latency = call.get("elapsed_ms")

        pt_str = str(pt) if pt is not None else "N/A"
        ct_str = str(ct) if ct is not None else "N/A"
        latency_str = f"{latency}ms" if latency is not None else "N/A"

        parts = [
            f"\n## Call {call_number}: {stage} ({call_type})",
            f"Model: {model} | Prompt tokens: {pt_str} | "
            f"Completion tokens: {ct_str} | Latency: {latency_str}\n",
        ]

        # System prompt (collapsible)
        system = call.get("system", "")
        if system:
            fence = _safe_fence(system)
            parts.append(
                f"### System Prompt\n"
                f"<details><summary>System prompt ({len(system)} chars)</summary>\n\n"
                f"{fence}\n{system}\n{fence}\n\n"
                f"</details>\n"
            )

        # User prompt (collapsible)
        prompt = call.get("prompt", "")
        if prompt:
            fence = _safe_fence(prompt)
            parts.append(
                f"### User Prompt\n"
                f"<details><summary>User prompt ({len(prompt)} chars)</summary>\n\n"
                f"{fence}\n{prompt}\n{fence}\n\n"
                f"</details>\n"
            )

        # Thinking (collapsible, if present)
        thinking = call.get("thinking", "")
        if thinking:
            fence = _safe_fence(thinking)
            parts.append(
                f"### Thinking\n"
                f"<details><summary>Thinking ({len(thinking)} chars)</summary>\n\n"
                f"{fence}\n{thinking}\n{fence}\n\n"
                f"</details>\n"
            )

        # Response (inline — primary audit target)
        response = call.get("response", "")
        fence = _safe_fence(response)
        parts.append(
            f"### Response\n"
            f"{fence}\n{response}\n{fence}\n"
        )

        # Error (if present)
        error = call.get("error", "")
        if error:
            fence = _safe_fence(error)
            parts.append(f"### Error\n{fence}\n{error}\n{fence}\n")

        return "\n".join(parts)
