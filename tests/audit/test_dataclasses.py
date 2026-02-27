"""Tests for audit protocol dataclasses."""

import pytest

from clean_room_agent.audit.dataclasses import (
    AuditResult,
    AuditScores,
    AuditSuiteResult,
    Finding,
    ReferenceTask,
)


class TestReferenceTask:
    def test_valid_minimal(self):
        rt = ReferenceTask(
            id="RT-001",
            description="Fix a bug",
            task_type="bug_fix",
            must_contain_files=["src/foo.py"],
        )
        assert rt.id == "RT-001"
        assert rt.budget_range == (20, 80)

    def test_all_fields(self):
        rt = ReferenceTask(
            id="RT-002",
            description="Add feature",
            task_type="feature",
            must_contain_files=["src/a.py"],
            should_contain_files=["src/b.py"],
            must_not_contain=["src/c/*"],
            must_contain_information=["Full source of foo()"],
            budget_range=(30, 70),
            routing_reasoning="Needs scope + precision",
        )
        assert rt.should_contain_files == ["src/b.py"]
        assert rt.must_contain_information == ["Full source of foo()"]

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="id must be non-empty"):
            ReferenceTask(id="", description="x", task_type="bug_fix", must_contain_files=["a.py"])

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description must be non-empty"):
            ReferenceTask(id="RT-001", description="", task_type="bug_fix", must_contain_files=["a.py"])

    def test_invalid_task_type_raises(self):
        with pytest.raises(ValueError, match="task_type must be one of"):
            ReferenceTask(id="RT-001", description="x", task_type="invalid", must_contain_files=["a.py"])

    def test_valid_task_types(self):
        for tt in ("bug_fix", "feature", "refactor", "test", "docs", "performance"):
            rt = ReferenceTask(id="RT-001", description="x", task_type=tt, must_contain_files=["a.py"])
            assert rt.task_type == tt

    def test_no_must_contain_files_raises(self):
        with pytest.raises(ValueError, match="must have at least one must_contain_files"):
            ReferenceTask(id="RT-001", description="x", task_type="bug_fix", must_contain_files=[])

    def test_invalid_budget_range_inverted(self):
        with pytest.raises(ValueError, match="budget_range"):
            ReferenceTask(id="RT-001", description="x", task_type="bug_fix",
                          must_contain_files=["a.py"], budget_range=(80, 30))

    def test_invalid_budget_range_over_100(self):
        with pytest.raises(ValueError, match="budget_range"):
            ReferenceTask(id="RT-001", description="x", task_type="bug_fix",
                          must_contain_files=["a.py"], budget_range=(50, 120))

    def test_invalid_budget_range_negative(self):
        with pytest.raises(ValueError, match="budget_range"):
            ReferenceTask(id="RT-001", description="x", task_type="bug_fix",
                          must_contain_files=["a.py"], budget_range=(-10, 50))


class TestAuditScores:
    def test_task_score_is_min(self):
        scores = AuditScores(
            must_contain_recall=0.8,
            should_contain_recall=0.5,
            exclusion_accuracy=0.6,
            budget_utilization=0.45,
            budget_in_range=True,
            parse_success_rate=0.9,
        )
        assert scores.task_score == 0.6  # min(0.8, 0.6)

    def test_task_score_perfect(self):
        scores = AuditScores(
            must_contain_recall=1.0,
            should_contain_recall=1.0,
            exclusion_accuracy=1.0,
            budget_utilization=0.5,
            budget_in_range=True,
            parse_success_rate=1.0,
        )
        assert scores.task_score == 1.0

    def test_task_score_zero_recall(self):
        scores = AuditScores(
            must_contain_recall=0.0,
            should_contain_recall=0.0,
            exclusion_accuracy=1.0,
            budget_utilization=0.3,
            budget_in_range=True,
            parse_success_rate=1.0,
        )
        assert scores.task_score == 0.0

    def test_detail_breakdowns(self):
        scores = AuditScores(
            must_contain_recall=0.5,
            should_contain_recall=0.0,
            exclusion_accuracy=1.0,
            budget_utilization=0.4,
            budget_in_range=True,
            parse_success_rate=1.0,
            must_contain_present=["a.py"],
            must_contain_missing=["b.py"],
            should_contain_missing=["c.py"],
        )
        assert scores.must_contain_present == ["a.py"]
        assert scores.must_contain_missing == ["b.py"]


class TestFinding:
    def test_valid(self):
        f = Finding(
            description="Missing file",
            severity="high",
            root_cause="data",
            detail="Dependency not indexed",
            action="Fix indexer",
        )
        assert f.severity == "high"

    def test_invalid_severity(self):
        with pytest.raises(ValueError, match="severity must be one of"):
            Finding(description="x", severity="critical", root_cause="data",
                    detail="x", action="x")

    def test_invalid_root_cause(self):
        with pytest.raises(ValueError, match="root_cause must be one of"):
            Finding(description="x", severity="high", root_cause="magic",
                    detail="x", action="x")

    def test_valid_severities(self):
        for sev in ("high", "medium", "low"):
            f = Finding(description="x", severity=sev, root_cause="prompt",
                        detail="x", action="x")
            assert f.severity == sev

    def test_valid_root_causes(self):
        for rc in ("prompt", "data", "architecture", "model"):
            f = Finding(description="x", severity="high", root_cause=rc,
                        detail="x", action="x")
            assert f.root_cause == rc


class TestAuditSuiteResult:
    def _make_result(self, task_id, must_recall, excl_acc, error=""):
        return AuditResult(
            audit_id=f"RA-{task_id}",
            task_id=task_id,
            date="2026-02-28",
            pipeline_version="abc123",
            model="qwen3-4b",
            error=error,
            scores=AuditScores(
                must_contain_recall=must_recall,
                should_contain_recall=0.5,
                exclusion_accuracy=excl_acc,
                budget_utilization=0.5,
                budget_in_range=True,
                parse_success_rate=1.0,
            ),
        )

    def test_empty_suite(self):
        suite = AuditSuiteResult()
        assert suite.mean_task_score == 0.0
        assert suite.worst_task is None
        assert suite.failed_tasks == []

    def test_mean_scores(self):
        suite = AuditSuiteResult(results=[
            self._make_result("RT-001", 0.8, 1.0),
            self._make_result("RT-002", 0.6, 0.8),
        ])
        # task scores: min(0.8,1.0)=0.8, min(0.6,0.8)=0.6
        assert suite.mean_task_score == pytest.approx(0.7)
        assert suite.mean_must_recall == pytest.approx(0.7)

    def test_worst_task(self):
        suite = AuditSuiteResult(results=[
            self._make_result("RT-001", 0.8, 1.0),
            self._make_result("RT-002", 0.2, 0.5),
            self._make_result("RT-003", 0.9, 0.9),
        ])
        assert suite.worst_task.task_id == "RT-002"

    def test_failed_tasks_excluded_from_means(self):
        suite = AuditSuiteResult(results=[
            self._make_result("RT-001", 1.0, 1.0),
            self._make_result("RT-002", 0.0, 0.0, error="pipeline crashed"),
        ])
        assert suite.mean_task_score == 1.0  # only RT-001 counted
        assert len(suite.failed_tasks) == 1
        assert suite.failed_tasks[0].task_id == "RT-002"

    def test_blocking_findings(self):
        r1 = self._make_result("RT-001", 0.8, 1.0)
        r2 = self._make_result("RT-002", 0.5, 0.5)
        r2.blocking = True
        suite = AuditSuiteResult(results=[r1, r2])
        assert len(suite.blocking_findings) == 1
        assert suite.blocking_findings[0].task_id == "RT-002"
