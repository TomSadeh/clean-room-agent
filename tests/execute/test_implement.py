"""Tests for execute_implement function."""

from unittest.mock import MagicMock

import pytest

from clean_room_agent.execute.dataclasses import PartPlan, PlanStep, StepResult, ValidationResult
from clean_room_agent.execute.implement import execute_implement, execute_test_implement
from clean_room_agent.llm.client import LLMResponse, ModelConfig
from clean_room_agent.retrieval.dataclasses import (
    BudgetConfig,
    ContextPackage,
    FileContent,
    TaskQuery,
)


@pytest.fixture
def context_package():
    task = TaskQuery(
        raw_task="Fix bug",
        task_id="test-002",
        mode="implement",
        repo_id=1,
    )
    return ContextPackage(
        task=task,
        files=[
            FileContent(
                file_id=1, path="src/main.py", language="python",
                content="def hello(): pass", token_estimate=10,
                detail_level="primary",
            ),
        ],
        total_token_estimate=10,
        budget=BudgetConfig(context_window=32768, reserved_tokens=4096),
    )


def _make_llm(response_text):
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(
        text=response_text,
        thinking=None,
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=500,
    )
    llm.config = ModelConfig(
        model="test-model",
        base_url="http://localhost:11434",
        context_window=32768,
        max_tokens=4096,
    )
    return llm


class TestExecuteImplement:
    def test_valid_response(self, context_package):
        response = '''<edit file="src/main.py">
<search>def hello(): pass</search>
<replacement>def hello(): return "world"</replacement>
</edit>'''
        llm = _make_llm(response)
        step = PlanStep(id="s1", description="Fix hello")
        result = execute_implement(
            context_package, step, llm,         )
        assert isinstance(result, StepResult)
        assert result.success is True
        assert len(result.edits) == 1
        assert result.edits[0].file_path == "src/main.py"

    def test_parse_failure_returns_failed_result(self, context_package):
        llm = _make_llm("I don't know how to do that")
        step = PlanStep(id="s1", description="Fix bug")
        result = execute_implement(
            context_package, step, llm,         )
        assert result.success is False
        assert result.error_info is not None
        assert "No valid <edit>" in result.error_info
        assert result.raw_response == "I don't know how to do that"

    def test_multiple_edits(self, context_package):
        response = '''<edit file="a.py">
<search>old_a</search>
<replacement>new_a</replacement>
</edit>
<edit file="b.py">
<search>old_b</search>
<replacement>new_b</replacement>
</edit>'''
        llm = _make_llm(response)
        step = PlanStep(id="s1", description="Multi-file change")
        result = execute_implement(
            context_package, step, llm,         )
        assert result.success is True
        assert len(result.edits) == 2

    def test_prompt_includes_step(self, context_package):
        response = '''<edit file="a.py">
<search>x</search>
<replacement>y</replacement>
</edit>'''
        llm = _make_llm(response)
        step = PlanStep(id="s1", description="Fix the bug in handler")
        execute_implement(
            context_package, step, llm,         )
        user_prompt = llm.complete.call_args[0][0]
        assert "s1" in user_prompt
        assert "Fix the bug in handler" in user_prompt

    def test_with_failure_context(self, context_package):
        response = '''<edit file="a.py">
<search>x</search>
<replacement>y</replacement>
</edit>'''
        llm = _make_llm(response)
        step = PlanStep(id="s1", description="Fix")
        failure = ValidationResult(
            success=False,
            test_output="AssertionError",
            failing_tests=["test_foo"],
        )
        result = execute_implement(
            context_package, step, llm,             failure_context=failure,
        )
        assert result.success is True
        user_prompt = llm.complete.call_args[0][0]
        assert "AssertionError" in user_prompt

    def test_raw_response_preserved(self, context_package):
        raw = '''<edit file="a.py">
<search>x</search>
<replacement>y</replacement>
</edit>'''
        llm = _make_llm(raw)
        step = PlanStep(id="s1", description="d")
        result = execute_implement(
            context_package, step, llm,         )
        assert result.raw_response == raw

    def test_with_plan_context(self, context_package):
        """execute_implement with plan=PartPlan includes plan constraints in prompt."""
        response = '''<edit file="a.py">
<search>x</search>
<replacement>y</replacement>
</edit>'''
        llm = _make_llm(response)
        step = PlanStep(id="s1", description="Implement handler")
        plan = PartPlan(
            part_id="p1",
            task_summary="Add REST endpoint",
            steps=[step, PlanStep(id="s2", description="Add tests")],
            rationale="Standard pattern",
        )
        result = execute_implement(
            context_package, step, llm,
            plan=plan,
        )
        assert result.success is True
        user_prompt = llm.complete.call_args[0][0]
        assert "<plan_constraints>" in user_prompt
        assert "Add REST endpoint" in user_prompt
        assert "(current)" in user_prompt
        assert "s2" in user_prompt


class TestExecuteImplementBudgetOverflow:
    def test_budget_overflow_raises(self):
        """Prompt exceeding model budget raises ValueError."""
        task = TaskQuery(
            raw_task="Fix bug",
            task_id="test-budget",
            mode="implement",
            repo_id=1,
        )
        large_content = "y = 2\n" * 5000
        context = ContextPackage(
            task=task,
            files=[
                FileContent(
                    file_id=1, path="src/big.py", language="python",
                    content=large_content, token_estimate=10000,
                    detail_level="primary",
                ),
            ],
            total_token_estimate=10000,
            budget=BudgetConfig(context_window=1024, reserved_tokens=256),
        )
        response = '''<edit file="a.py">
<search>x</search>
<replacement>y</replacement>
</edit>'''
        llm = MagicMock()
        llm.complete.return_value = LLMResponse(
            text=response, thinking=None, prompt_tokens=100, completion_tokens=50, latency_ms=500,
        )
        llm.config = ModelConfig(
            model="test-model",
            base_url="http://localhost:11434",
            context_window=512,
            max_tokens=256,
        )
        step = PlanStep(id="s1", description="Fix the bug")
        with pytest.raises(ValueError, match="R3.*prompt too large"):
            execute_implement(context, step, llm)


class TestExecuteTestImplement:
    def test_valid_response(self, context_package):
        response = '''<edit file="tests/test_main.py">
<search>def test_placeholder(): pass</search>
<replacement>def test_hello():
    assert hello() == "world"</replacement>
</edit>'''
        llm = _make_llm(response)
        step = PlanStep(id="t1", description="Test hello function")
        result = execute_test_implement(context_package, step, llm)
        assert isinstance(result, StepResult)
        assert result.success is True
        assert len(result.edits) == 1
        assert result.edits[0].file_path == "tests/test_main.py"

    def test_parse_failure_returns_failed_result(self, context_package):
        llm = _make_llm("I can't write tests for that")
        step = PlanStep(id="t1", description="Test something")
        result = execute_test_implement(context_package, step, llm)
        assert result.success is False
        assert result.error_info is not None
        assert "No valid <edit>" in result.error_info

    def test_prompt_includes_test_step_header(self, context_package):
        response = '''<edit file="tests/test_a.py">
<search>x</search>
<replacement>y</replacement>
</edit>'''
        llm = _make_llm(response)
        step = PlanStep(id="t1", description="Test the handler")
        execute_test_implement(context_package, step, llm)
        user_prompt = llm.complete.call_args[0][0]
        assert "Test Step to Implement" in user_prompt
        assert "t1" in user_prompt

    def test_with_test_plan_context(self, context_package):
        response = '''<edit file="tests/test_a.py">
<search>x</search>
<replacement>y</replacement>
</edit>'''
        llm = _make_llm(response)
        step = PlanStep(id="t1", description="Test handler")
        test_plan = PartPlan(
            part_id="p1_tests",
            task_summary="Test coverage for p1",
            steps=[step, PlanStep(id="t2", description="Test edge cases")],
            rationale="Full coverage",
        )
        result = execute_test_implement(
            context_package, step, llm,
            test_plan=test_plan,
        )
        assert result.success is True
        user_prompt = llm.complete.call_args[0][0]
        assert "<plan_constraints>" in user_prompt
        assert "Test coverage for p1" in user_prompt

    def test_raw_response_preserved(self, context_package):
        raw = '''<edit file="tests/test_a.py">
<search>x</search>
<replacement>y</replacement>
</edit>'''
        llm = _make_llm(raw)
        step = PlanStep(id="t1", description="d")
        result = execute_test_implement(context_package, step, llm)
        assert result.raw_response == raw
