"""Tests for retrieval dataclasses."""

import pytest

from clean_room_agent.retrieval.dataclasses import (
    BudgetConfig,
    ClassifiedSymbol,
    ContextPackage,
    FileContent,
    RefinementRequest,
    ScopedFile,
    TaskQuery,
)


class TestBudgetConfig:
    def test_valid(self):
        b = BudgetConfig(context_window=32768, reserved_tokens=4096)
        assert b.effective_budget == 32768 - 4096

    def test_zero_reserved(self):
        b = BudgetConfig(context_window=32768, reserved_tokens=0)
        assert b.effective_budget == 32768

    def test_negative_window_raises(self):
        with pytest.raises(ValueError, match="context_window must be > 0"):
            BudgetConfig(context_window=-1, reserved_tokens=0)

    def test_zero_window_raises(self):
        with pytest.raises(ValueError, match="context_window must be > 0"):
            BudgetConfig(context_window=0, reserved_tokens=0)

    def test_negative_reserved_raises(self):
        with pytest.raises(ValueError, match="reserved_tokens must be >= 0"):
            BudgetConfig(context_window=32768, reserved_tokens=-1)

    def test_reserved_equals_window_raises(self):
        with pytest.raises(ValueError, match="reserved_tokens.*must be <"):
            BudgetConfig(context_window=100, reserved_tokens=100)

    def test_reserved_exceeds_window_raises(self):
        with pytest.raises(ValueError, match="reserved_tokens.*must be <"):
            BudgetConfig(context_window=100, reserved_tokens=200)


class TestTaskQuery:
    def test_valid_defaults(self):
        tq = TaskQuery(raw_task="fix bug", task_id="t1", mode="plan", repo_id=1)
        assert tq.task_type == "unknown"
        assert tq.mentioned_files == []
        assert tq.seed_file_ids == []

    def test_valid_task_types(self):
        for tt in ("bug_fix", "feature", "refactor", "test", "docs", "unknown"):
            tq = TaskQuery(raw_task="x", task_id="t", mode="plan", repo_id=1, task_type=tt)
            assert tq.task_type == tt

    def test_invalid_task_type(self):
        with pytest.raises(ValueError, match="task_type must be one of"):
            TaskQuery(raw_task="x", task_id="t", mode="plan", repo_id=1, task_type="invalid")


class TestScopedFile:
    def test_construction(self):
        sf = ScopedFile(file_id=1, path="a.py", language="python", tier=1)
        assert sf.relevance == "pending"
        assert sf.reason == ""

    def test_custom_values(self):
        sf = ScopedFile(file_id=2, path="b.ts", language="typescript", tier=3,
                        relevance="relevant", reason="co-changed")
        assert sf.tier == 3
        assert sf.relevance == "relevant"


class TestClassifiedSymbol:
    def test_valid_detail_levels(self):
        for dl in ("primary", "supporting", "type_context", "excluded"):
            cs = ClassifiedSymbol(
                symbol_id=1, file_id=1, name="test", kind="function",
                start_line=1, end_line=5, detail_level=dl,
            )
            assert cs.detail_level == dl

    def test_invalid_detail_level(self):
        with pytest.raises(ValueError, match="detail_level must be one of"):
            ClassifiedSymbol(
                symbol_id=1, file_id=1, name="test", kind="function",
                start_line=1, end_line=5, detail_level="invalid",
            )


class TestFileContent:
    def test_construction(self):
        fc = FileContent(
            file_id=1, path="a.py", language="python",
            content="def foo(): pass", token_estimate=4, detail_level="primary",
        )
        assert fc.included_symbols == []


class TestContextPackage:
    def test_to_prompt_text(self):
        tq = TaskQuery(raw_task="fix bug", task_id="t1", mode="plan",
                        repo_id=1, intent_summary="Fix the login bug")
        fc = FileContent(
            file_id=1, path="auth.py", language="python",
            content="def login(): pass", token_estimate=5, detail_level="primary",
        )
        pkg = ContextPackage(
            task=tq, files=[fc], total_token_estimate=5,
            budget=BudgetConfig(context_window=32768, reserved_tokens=4096),
        )
        text = pkg.to_prompt_text()
        assert "fix bug" in text
        assert "Fix the login bug" in text
        assert "auth.py" in text
        assert "def login(): pass" in text

    def test_empty_package(self):
        tq = TaskQuery(raw_task="task", task_id="t1", mode="plan", repo_id=1)
        pkg = ContextPackage(task=tq)
        text = pkg.to_prompt_text()
        assert "task" in text


class TestRefinementRequest:
    def test_construction(self):
        rr = RefinementRequest(reason="missing test files", source_task_id="t1", missing_tests=["test_auth.py"])
        assert rr.reason == "missing test files"
        assert rr.source_task_id == "t1"
        assert rr.missing_tests == ["test_auth.py"]
        assert rr.missing_files == []
