"""Tests for audit metrics scorer."""

import pytest

from clean_room_agent.audit.scorer import score_retrieval, _path_in_context, _matches_exclusion
from clean_room_agent.audit.dataclasses import ReferenceTask
from clean_room_agent.retrieval.dataclasses import (
    BudgetConfig,
    ContextPackage,
    FileContent,
    TaskQuery,
)


def _make_task_query():
    return TaskQuery(raw_task="test task", task_id="t1", mode="plan", repo_id=1)


def _make_package(file_paths, budget=None, total_tokens=500):
    files = [
        FileContent(
            file_id=i, path=p, language="python",
            content="...", token_estimate=100, detail_level="primary",
        )
        for i, p in enumerate(file_paths)
    ]
    return ContextPackage(
        task=_make_task_query(),
        files=files,
        total_token_estimate=total_tokens,
        budget=budget,
    )


def _make_ref_task(**overrides):
    defaults = dict(
        id="RT-001",
        description="Test task",
        task_type="bug_fix",
        must_contain_files=["src/foo.py"],
        should_contain_files=[],
        must_not_contain=[],
        budget_range=(20, 80),
    )
    defaults.update(overrides)
    return ReferenceTask(**defaults)


class TestPathInContext:
    def test_exact_match(self):
        assert _path_in_context("src/foo.py", {"src/foo.py", "src/bar.py"})

    def test_no_match(self):
        assert not _path_in_context("src/baz.py", {"src/foo.py", "src/bar.py"})

    def test_glob_match(self):
        assert _path_in_context("src/*.py", {"src/foo.py", "src/bar.py"})

    def test_glob_no_match(self):
        assert not _path_in_context("tests/*.py", {"src/foo.py"})


class TestMatchesExclusion:
    def test_exact_match(self):
        assert _matches_exclusion("src/foo.py", "src/foo.py")

    def test_glob_star(self):
        assert _matches_exclusion("src/indexer/scanner.py", "src/indexer/*")

    def test_glob_no_match(self):
        assert not _matches_exclusion("src/retrieval/scope.py", "src/indexer/*")

    def test_recursive_glob(self):
        assert _matches_exclusion("src/a/b/c.py", "src/a/*")


class TestScoreRetrieval:
    def test_perfect_scores(self):
        ref = _make_ref_task(
            must_contain_files=["src/foo.py", "src/bar.py"],
            should_contain_files=["src/baz.py"],
            must_not_contain=["src/indexer/*"],
        )
        pkg = _make_package(
            ["src/foo.py", "src/bar.py", "src/baz.py"],
            budget=BudgetConfig(context_window=2000, reserved_tokens=200),
            total_tokens=900,
        )
        scores = score_retrieval(ref, pkg)
        assert scores.must_contain_recall == 1.0
        assert scores.should_contain_recall == 1.0
        assert scores.exclusion_accuracy == 1.0
        assert scores.task_score == 1.0

    def test_partial_must_recall(self):
        ref = _make_ref_task(
            must_contain_files=["src/foo.py", "src/bar.py"],
        )
        pkg = _make_package(["src/foo.py"])
        scores = score_retrieval(ref, pkg)
        assert scores.must_contain_recall == 0.5
        assert scores.must_contain_present == ["src/foo.py"]
        assert scores.must_contain_missing == ["src/bar.py"]

    def test_zero_must_recall(self):
        ref = _make_ref_task(must_contain_files=["src/foo.py"])
        pkg = _make_package(["src/bar.py"])
        scores = score_retrieval(ref, pkg)
        assert scores.must_contain_recall == 0.0
        assert scores.must_contain_missing == ["src/foo.py"]

    def test_partial_should_recall(self):
        ref = _make_ref_task(
            should_contain_files=["src/a.py", "src/b.py", "src/c.py"],
        )
        pkg = _make_package(["src/foo.py", "src/a.py"])
        scores = score_retrieval(ref, pkg)
        assert scores.should_contain_recall == pytest.approx(1 / 3)
        assert scores.should_contain_present == ["src/a.py"]
        assert len(scores.should_contain_missing) == 2

    def test_exclusion_violation(self):
        ref = _make_ref_task(
            must_not_contain=["src/indexer/*", "src/execute/*"],
        )
        pkg = _make_package(["src/foo.py", "src/indexer/scanner.py"])
        scores = score_retrieval(ref, pkg)
        assert scores.exclusion_accuracy == 0.5  # 1 of 2 patterns violated
        assert "src/indexer/scanner.py" in scores.exclusion_violations

    def test_multiple_exclusion_violations_same_pattern(self):
        ref = _make_ref_task(
            must_not_contain=["src/indexer/*"],
        )
        pkg = _make_package(["src/indexer/a.py", "src/indexer/b.py"])
        scores = score_retrieval(ref, pkg)
        # Only 1 pattern, it's violated
        assert scores.exclusion_accuracy == 0.0
        assert len(scores.exclusion_violations) == 2

    def test_no_exclusion_patterns(self):
        ref = _make_ref_task(must_not_contain=[])
        pkg = _make_package(["src/indexer/a.py"])
        scores = score_retrieval(ref, pkg)
        assert scores.exclusion_accuracy == 1.0

    def test_budget_utilization(self):
        ref = _make_ref_task(budget_range=(30, 70))
        budget = BudgetConfig(context_window=2000, reserved_tokens=200)
        pkg = _make_package(["src/foo.py"], budget=budget, total_tokens=900)
        scores = score_retrieval(ref, pkg)
        # effective = 1800, utilization = 900/1800 = 0.5 = 50%
        assert scores.budget_utilization == pytest.approx(0.5)
        assert scores.budget_in_range  # 50% is within [30, 70]

    def test_budget_below_range(self):
        ref = _make_ref_task(budget_range=(30, 70))
        budget = BudgetConfig(context_window=2000, reserved_tokens=200)
        pkg = _make_package(["src/foo.py"], budget=budget, total_tokens=100)
        scores = score_retrieval(ref, pkg)
        # utilization = 100/1800 ≈ 5.6%
        assert not scores.budget_in_range

    def test_budget_above_range(self):
        ref = _make_ref_task(budget_range=(30, 70))
        budget = BudgetConfig(context_window=2000, reserved_tokens=200)
        pkg = _make_package(["src/foo.py"], budget=budget, total_tokens=1700)
        scores = score_retrieval(ref, pkg)
        # utilization = 1700/1800 ≈ 94.4%
        assert not scores.budget_in_range

    def test_no_budget(self):
        ref = _make_ref_task()
        pkg = _make_package(["src/foo.py"])
        scores = score_retrieval(ref, pkg)
        assert scores.budget_utilization == 0.0

    def test_parse_success_rate_passthrough(self):
        ref = _make_ref_task()
        pkg = _make_package(["src/foo.py"])
        scores = score_retrieval(ref, pkg, parse_success_rate=0.75)
        assert scores.parse_success_rate == 0.75

    def test_task_score_min_recall_exclusion(self):
        ref = _make_ref_task(
            must_contain_files=["src/foo.py", "src/bar.py"],
            must_not_contain=["src/indexer/*"],
        )
        pkg = _make_package(["src/foo.py", "src/indexer/a.py"])
        scores = score_retrieval(ref, pkg)
        # must_recall = 0.5, exclusion = 0.0
        assert scores.task_score == 0.0

    def test_empty_should_contain(self):
        ref = _make_ref_task(should_contain_files=[])
        pkg = _make_package(["src/foo.py"])
        scores = score_retrieval(ref, pkg)
        assert scores.should_contain_recall == 1.0  # no requirements = 1.0

    def test_glob_in_must_contain(self):
        ref = _make_ref_task(must_contain_files=["src/retrieval/*.py"])
        pkg = _make_package(["src/retrieval/scope_stage.py"])
        scores = score_retrieval(ref, pkg)
        assert scores.must_contain_recall == 1.0
