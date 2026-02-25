"""Pipeline trace log: human-readable markdown trace of LLM calls."""

from __future__ import annotations

from pathlib import Path


class TraceLogger:
    """Accumulates LLM call records and writes a human-readable markdown file."""

    def __init__(self, output_path: Path, task_id: str, task_description: str):
        self._output_path = output_path
        self._task_id = task_id
        self._task_description = task_description
        self._calls: list[dict] = []

    def log_calls(
        self, stage_name: str, call_type: str, calls: list[dict], model: str = "",
    ) -> None:
        """Record a batch of flushed call dicts from LoggedLLMClient.flush()."""
        for call in calls:
            self._calls.append({
                "stage_name": stage_name,
                "call_type": call_type,
                "model": model,
                "system": call.get("system", ""),
                "prompt": call.get("prompt", ""),
                "response": call.get("response", ""),
                "thinking": call.get("thinking", ""),
                "prompt_tokens": call.get("prompt_tokens"),
                "completion_tokens": call.get("completion_tokens"),
                "elapsed_ms": call.get("elapsed_ms", 0),
                "error": call.get("error", ""),
            })

    def finalize(self) -> Path:
        """Write markdown trace to disk. Returns the output path."""
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

        for call in self._calls:
            pt = call.get("prompt_tokens")
            ct = call.get("completion_tokens")
            if pt is None:
                has_none_prompt = True
            else:
                total_prompt += pt
            if ct is None:
                has_none_completion = True
            else:
                total_completion += ct
            total_latency += call.get("elapsed_ms", 0)

        prompt_str = "N/A" if has_none_prompt else str(total_prompt)
        completion_str = "N/A" if has_none_completion else str(total_completion)

        return (
            f"# Pipeline Trace: {self._task_id}\n\n"
            f"**Task:** {self._task_description}\n\n"
            f"**Summary:** {total_calls} calls | "
            f"Prompt tokens: {prompt_str} | "
            f"Completion tokens: {completion_str} | "
            f"Total latency: {total_latency}ms\n\n"
            f"---\n"
        )

    def _format_call(self, call_number: int, call: dict) -> str:
        """Render one call as a markdown section."""
        stage = call["stage_name"]
        call_type = call["call_type"]
        model = call.get("model", "")
        pt = call.get("prompt_tokens")
        ct = call.get("completion_tokens")
        latency = call.get("elapsed_ms", 0)

        pt_str = str(pt) if pt is not None else "N/A"
        ct_str = str(ct) if ct is not None else "N/A"

        parts = [
            f"\n## Call {call_number}: {stage} ({call_type})",
            f"Model: {model} | Prompt tokens: {pt_str} | "
            f"Completion tokens: {ct_str} | Latency: {latency}ms\n",
        ]

        # System prompt (collapsible)
        system = call.get("system", "")
        if system:
            parts.append(
                f"### System Prompt\n"
                f"<details><summary>System prompt ({len(system)} chars)</summary>\n\n"
                f"{system}\n\n"
                f"</details>\n"
            )

        # User prompt (collapsible)
        prompt = call.get("prompt", "")
        if prompt:
            parts.append(
                f"### User Prompt\n"
                f"<details><summary>User prompt ({len(prompt)} chars)</summary>\n\n"
                f"{prompt}\n\n"
                f"</details>\n"
            )

        # Thinking (collapsible, if present)
        thinking = call.get("thinking", "")
        if thinking:
            parts.append(
                f"### Thinking\n"
                f"<details><summary>Thinking ({len(thinking)} chars)</summary>\n\n"
                f"{thinking}\n\n"
                f"</details>\n"
            )

        # Response (inline — primary audit target)
        response = call.get("response", "")
        parts.append(
            f"### Response\n"
            f"```\n{response}\n```\n"
        )

        # Error (if present)
        error = call.get("error", "")
        if error:
            parts.append(f"### Error\n```\n{error}\n```\n")

        return "\n".join(parts)
