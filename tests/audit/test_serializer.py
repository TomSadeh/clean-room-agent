"""Tests for audit result serializer."""

import pytest
from pathlib import Path

from clean_room_agent.audit.dataclasses import (
    AuditResult,
    AuditScores,
    AuditSuiteResult,
    Finding,
)
from clean_room_agent.audit.serializer import (
    serialize_audit_result,
    save_audit_result,
    format_suite_summary,
)


def _make_scores(**overrides):
    defaults = dict(
        must_contain_recall=0.75,
        should_contain_recall=0.50,
        exclusion_accuracy=1.0,
        budget_utilization=0.55,
        budget_in_range=True,
        parse_success_rate=0.85,
    )
    defaults.update(overrides)
    return AuditScores(**defaults)


def _make_result(**overrides):
    defaults = dict(
        audit_id="RA-RT-001-20260228120000",
        task_id="RT-001",
        date="2026-02-28",
        pipeline_version="f89f708",
        model="qwen3-4b",
        scores=_make_scores(),
    )
    defaults.update(overrides)
    return AuditResult(**defaults)


class TestSerializeAuditResult:
    def test_basic_output(self):
        result = _make_result()
        text = serialize_audit_result(result)

        assert '[audit]' in text
        assert 'id = "RA-RT-001-20260228120000"' in text
        assert 'task_id = "RT-001"' in text
        assert '[scores]' in text
        assert 'must_contain_recall = 0.75' in text
        assert 'task_score = 0.75' in text  # min(0.75, 1.0)
        assert '[assessment]' in text

    def test_with_findings(self):
        result = _make_result(findings=[
            Finding(
                description="Missing db/queries.py",
                severity="high",
                root_cause="data",
                detail="Dependency resolution missed re-export",
                action="Fix dependency resolver",
            ),
        ])
        text = serialize_audit_result(result)
        assert '[[findings]]' in text
        assert 'severity = "high"' in text
        assert 'root_cause = "data"' in text

    def test_with_error(self):
        result = _make_result(error="RuntimeError: No indexed repo")
        text = serialize_audit_result(result)
        assert 'error = "RuntimeError: No indexed repo"' in text

    def test_with_stages(self):
        result = _make_result(stages_selected=["scope", "precision"])
        text = serialize_audit_result(result)
        assert 'stages_selected = ["scope", "precision"]' in text

    def test_blocking_flag(self):
        result = _make_result(blocking=True)
        text = serialize_audit_result(result)
        assert 'blocking = true' in text

    def test_escapes_special_chars(self):
        result = _make_result(summary='Has "quotes" and\nnewlines')
        text = serialize_audit_result(result)
        assert r'\"quotes\"' in text
        assert r'\n' in text

    def test_missing_scores_breakdown(self):
        scores = _make_scores(
            must_contain_missing=["a.py", "b.py"],
            exclusion_violations=["c.py"],
        )
        result = _make_result(scores=scores)
        text = serialize_audit_result(result)
        assert 'must_contain_missing = ["a.py", "b.py"]' in text
        assert 'exclusion_violations = ["c.py"]' in text


class TestSaveAuditResult:
    def test_saves_to_file(self, tmp_path):
        result = _make_result()
        path = save_audit_result(result, findings_dir=tmp_path)
        assert path.exists()
        assert path.name == "RA-RT-001-20260228120000.toml"
        content = path.read_text(encoding="utf-8")
        assert '[audit]' in content

    def test_creates_directory(self, tmp_path):
        findings_dir = tmp_path / "sub" / "findings"
        result = _make_result()
        path = save_audit_result(result, findings_dir=findings_dir)
        assert path.exists()
        assert findings_dir.is_dir()

    def test_repo_path_resolution(self, tmp_path):
        result = _make_result()
        path = save_audit_result(result, repo_path=tmp_path)
        assert path.parent == tmp_path / "protocols" / "retrieval_audit" / "findings"

    def test_no_path_raises(self):
        result = _make_result()
        with pytest.raises(ValueError, match="Either findings_dir or repo_path"):
            save_audit_result(result)


class TestFormatSuiteSummary:
    def _make_suite_result(self, task_id, must_recall, excl_acc, error=""):
        return AuditResult(
            audit_id=f"RA-{task_id}",
            task_id=task_id,
            date="2026-02-28",
            pipeline_version="abc123",
            model="qwen3-4b",
            error=error,
            scores=_make_scores(
                must_contain_recall=must_recall,
                exclusion_accuracy=excl_acc,
            ),
        )

    def test_basic_summary(self):
        suite = AuditSuiteResult(
            pipeline_version="abc123",
            model="qwen3-4b",
            results=[
                self._make_suite_result("RT-001", 0.8, 1.0),
                self._make_suite_result("RT-002", 0.6, 0.9),
            ],
        )
        text = format_suite_summary(suite)
        assert "abc123" in text
        assert "qwen3-4b" in text
        assert "2 total" in text
        assert "0 failed" in text
        assert "RT-001" in text
        assert "RT-002" in text

    def test_with_failures(self):
        suite = AuditSuiteResult(
            pipeline_version="abc123",
            model="qwen3-4b",
            results=[
                self._make_suite_result("RT-001", 1.0, 1.0),
                self._make_suite_result("RT-002", 0.0, 0.0, error="crash"),
            ],
        )
        text = format_suite_summary(suite)
        assert "1 failed" in text
        assert "Pipeline failures" in text
        assert "crash" in text

    def test_empty_suite(self):
        suite = AuditSuiteResult(pipeline_version="x", model="y")
        text = format_suite_summary(suite)
        assert "0 total" in text
