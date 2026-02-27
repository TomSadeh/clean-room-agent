"""Serialize audit results to TOML findings files."""

from __future__ import annotations

from pathlib import Path

from clean_room_agent.audit.dataclasses import AuditResult, AuditSuiteResult

# Default location relative to repo root
DEFAULT_FINDINGS_DIR = Path("protocols/retrieval_audit/findings")


def serialize_audit_result(result: AuditResult) -> str:
    """Serialize an AuditResult to TOML format."""
    lines = []

    lines.append("[audit]")
    lines.append(f'id = "{result.audit_id}"')
    lines.append(f'task_id = "{result.task_id}"')
    lines.append(f'date = "{result.date}"')
    lines.append(f'pipeline_version = "{result.pipeline_version}"')
    lines.append(f'model = "{result.model}"')
    if result.error:
        lines.append(f'error = "{_escape_toml(result.error)}"')
    if result.stages_selected:
        lines.append(f'stages_selected = {_toml_list(result.stages_selected)}')
    if result.routing_reasoning:
        lines.append(f'routing_reasoning = "{_escape_toml(result.routing_reasoning)}"')
    lines.append(f"total_latency_ms = {result.total_latency_ms}")
    lines.append("")

    lines.append("[scores]")
    s = result.scores
    lines.append(f"must_contain_recall = {s.must_contain_recall:.2f}")
    lines.append(f"should_contain_recall = {s.should_contain_recall:.2f}")
    lines.append(f"exclusion_accuracy = {s.exclusion_accuracy:.2f}")
    lines.append(f"budget_utilization = {s.budget_utilization:.2f}")
    lines.append(f"budget_in_range = {'true' if s.budget_in_range else 'false'}")
    lines.append(f"parse_success_rate = {s.parse_success_rate:.2f}")
    lines.append(f"task_score = {s.task_score:.2f}")
    lines.append("")

    if s.must_contain_missing:
        lines.append(f"must_contain_missing = {_toml_list(s.must_contain_missing)}")
    if s.should_contain_missing:
        lines.append(f"should_contain_missing = {_toml_list(s.should_contain_missing)}")
    if s.exclusion_violations:
        lines.append(f"exclusion_violations = {_toml_list(s.exclusion_violations)}")
    if s.must_contain_missing or s.should_contain_missing or s.exclusion_violations:
        lines.append("")

    for finding in result.findings:
        lines.append("[[findings]]")
        lines.append(f'description = "{_escape_toml(finding.description)}"')
        lines.append(f'severity = "{finding.severity}"')
        lines.append(f'root_cause = "{finding.root_cause}"')
        lines.append(f'detail = "{_escape_toml(finding.detail)}"')
        lines.append(f'action = "{_escape_toml(finding.action)}"')
        lines.append("")

    lines.append("[assessment]")
    lines.append(f'summary = "{_escape_toml(result.summary)}"')
    lines.append(f"blocking = {'true' if result.blocking else 'false'}")
    lines.append("")

    return "\n".join(lines)


def save_audit_result(
    result: AuditResult,
    findings_dir: Path | None = None,
    repo_path: Path | None = None,
) -> Path:
    """Save an audit result as a TOML file in the findings directory.

    Returns the path to the saved file.
    """
    if findings_dir is None:
        if repo_path is None:
            raise ValueError("Either findings_dir or repo_path must be provided")
        findings_dir = repo_path / DEFAULT_FINDINGS_DIR

    findings_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result.audit_id}.toml"
    path = findings_dir / filename
    path.write_text(serialize_audit_result(result), encoding="utf-8")
    return path


def format_suite_summary(suite: AuditSuiteResult) -> str:
    """Format a human-readable summary of a full audit suite run."""
    lines = []
    lines.append(f"Retrieval Audit Suite — {suite.pipeline_version}")
    lines.append(f"Model: {suite.model}")
    lines.append(f"Tasks: {len(suite.results)} total, {len(suite.failed_tasks)} failed")
    lines.append("")

    lines.append("Aggregate Scores:")
    lines.append(f"  Mean task score:         {suite.mean_task_score:.2f}")
    lines.append(f"  Mean must-contain recall: {suite.mean_must_recall:.2f}")
    lines.append(f"  Mean exclusion accuracy:  {suite.mean_exclusion_accuracy:.2f}")
    lines.append("")

    if suite.worst_task:
        w = suite.worst_task
        lines.append(f"Worst task: {w.task_id} (score={w.scores.task_score:.2f})")
        if w.scores.must_contain_missing:
            lines.append(f"  Missing: {', '.join(w.scores.must_contain_missing)}")
        lines.append("")

    if suite.blocking_findings:
        lines.append("Blocking findings:")
        for r in suite.blocking_findings:
            lines.append(f"  {r.task_id}: {r.summary}")
        lines.append("")

    if suite.failed_tasks:
        lines.append("Pipeline failures:")
        for r in suite.failed_tasks:
            lines.append(f"  {r.task_id}: {r.error}")
        lines.append("")

    lines.append("Per-task scores:")
    for r in sorted(suite.results, key=lambda x: x.task_id):
        status = "FAIL" if r.error else f"{r.scores.task_score:.2f}"
        lines.append(f"  {r.task_id}: {status}")

    return "\n".join(lines)


def _escape_toml(s: str) -> str:
    """Escape a string for TOML basic string representation."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _toml_list(items: list[str]) -> str:
    """Format a list of strings as a TOML array."""
    escaped = [f'"{_escape_toml(item)}"' for item in items]
    return f"[{', '.join(escaped)}]"
