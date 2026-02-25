"""Tests for Phase 3 prompt templates."""

import pytest

from clean_room_agent.execute.dataclasses import (
    PartPlan,
    PlanStep,
    StepResult,
    ValidationResult,
)
from clean_room_agent.execute.prompts import (
    ADJUSTMENT_SYSTEM,
    IMPLEMENT_SYSTEM,
    META_PLAN_SYSTEM,
    PART_PLAN_SYSTEM,
    SYSTEM_PROMPTS,
    TEST_IMPLEMENT_SYSTEM,
    TEST_PLAN_SYSTEM,
    build_implement_prompt,
    build_plan_prompt,
)
from clean_room_agent.llm.client import ModelConfig
from clean_room_agent.retrieval.dataclasses import (
    BudgetConfig,
    ContextPackage,
    FileContent,
    TaskQuery,
)


@pytest.fixture
def model_config():
    return ModelConfig(
        model="test-model",
        base_url="http://localhost:11434",
        context_window=32768,
        max_tokens=4096,
    )


@pytest.fixture
def small_model_config():
    """Small context window for budget overflow testing."""
    return ModelConfig(
        model="test-model",
        base_url="http://localhost:11434",
        context_window=1024,
        max_tokens=256,
    )


@pytest.fixture
def context_package():
    task = TaskQuery(
        raw_task="Add validation",
        task_id="test-001",
        mode="plan",
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


class TestBuildPlanPrompt:
    def test_meta_plan_system(self, context_package, model_config):
        system, user = build_plan_prompt(
            context_package, "Add validation",
            pass_type="meta_plan", model_config=model_config,
        )
        assert system == META_PLAN_SYSTEM
        assert "Add validation" in user
        assert "src/main.py" in user

    def test_part_plan_system(self, context_package, model_config):
        system, user = build_plan_prompt(
            context_package, "Implement part 1",
            pass_type="part_plan", model_config=model_config,
        )
        assert system == PART_PLAN_SYSTEM

    def test_adjustment_system(self, context_package, model_config):
        system, user = build_plan_prompt(
            context_package, "Adjust plan",
            pass_type="adjustment", model_config=model_config,
        )
        assert system == ADJUSTMENT_SYSTEM

    def test_unknown_pass_type_raises(self, context_package, model_config):
        with pytest.raises(ValueError, match="Unknown plan pass_type"):
            build_plan_prompt(
                context_package, "task",
                pass_type="invalid", model_config=model_config,
            )

    def test_with_cumulative_diff(self, context_package, model_config):
        _, user = build_plan_prompt(
            context_package, "task",
            pass_type="adjustment", model_config=model_config,
            cumulative_diff="--- a/file.py\n+++ b/file.py",
        )
        assert "<prior_changes>" in user
        assert "--- a/file.py" in user

    def test_with_prior_results(self, context_package, model_config):
        results = [
            StepResult(success=True, raw_response="ok"),
            StepResult(success=False, error_info="Parse failed", raw_response="bad"),
        ]
        _, user = build_plan_prompt(
            context_package, "task",
            pass_type="adjustment", model_config=model_config,
            prior_results=results,
        )
        assert "<completed_steps>" in user
        assert "success" in user
        assert "Parse failed" in user

    def test_with_test_results(self, context_package, model_config):
        test_results = [
            ValidationResult(success=False, test_output="FAILED test_foo", failing_tests=["test_foo"]),
        ]
        _, user = build_plan_prompt(
            context_package, "task",
            pass_type="adjustment", model_config=model_config,
            test_results=test_results,
        )
        assert "<test_results>" in user
        assert "test_foo" in user

    def test_budget_overflow_raises(self, context_package, small_model_config):
        with pytest.raises(ValueError, match="Prompt too large"):
            build_plan_prompt(
                context_package, "task" * 1000,
                pass_type="meta_plan", model_config=small_model_config,
            )

    def test_build_plan_prompt_empty_files(self, model_config):
        """build_plan_prompt with empty ContextPackage.files list."""
        task = TaskQuery(
            raw_task="Analyze codebase",
            task_id="test-empty",
            mode="plan",
            repo_id=1,
        )
        context = ContextPackage(
            task=task,
            files=[],
            total_token_estimate=0,
            budget=BudgetConfig(context_window=32768, reserved_tokens=4096),
        )
        system, user = build_plan_prompt(
            context, "Analyze codebase",
            pass_type="meta_plan", model_config=model_config,
        )
        assert system == META_PLAN_SYSTEM
        # Should still contain the task description
        assert "Analyze codebase" in user
        # Should not contain any file sections (no ## header)
        assert "## " not in user


class TestBuildImplementPrompt:
    def test_basic(self, context_package, model_config):
        step = PlanStep(id="s1", description="Add hello endpoint")
        system, user = build_implement_prompt(
            context_package, step, model_config=model_config,
        )
        assert system == IMPLEMENT_SYSTEM
        assert "s1" in user
        assert "Add hello endpoint" in user

    def test_with_target_files(self, context_package, model_config):
        step = PlanStep(
            id="s1", description="Fix",
            target_files=["a.py", "b.py"],
            target_symbols=["foo", "bar"],
        )
        _, user = build_implement_prompt(
            context_package, step, model_config=model_config,
        )
        assert "a.py" in user
        assert "foo" in user

    def test_with_plan_context(self, context_package, model_config):
        step = PlanStep(id="s1", description="Step 1")
        plan = PartPlan(
            part_id="p1", task_summary="Part 1 goal",
            steps=[step, PlanStep(id="s2", description="Step 2")],
            rationale="r",
        )
        _, user = build_implement_prompt(
            context_package, step, model_config=model_config, plan=plan,
        )
        assert "<plan_constraints>" in user
        assert "Part 1 goal" in user
        assert "(current)" in user

    def test_with_failure_context(self, context_package, model_config):
        step = PlanStep(id="s1", description="Fix bug")
        failure = ValidationResult(
            success=False,
            test_output="AssertionError in test_foo",
            failing_tests=["test_foo"],
            lint_output="E501",
            type_check_output="type error",
        )
        _, user = build_implement_prompt(
            context_package, step, model_config=model_config,
            failure_context=failure,
        )
        assert "<test_failures>" in user
        assert "AssertionError" in user
        assert "E501" in user

    def test_with_cumulative_diff(self, context_package, model_config):
        step = PlanStep(id="s1", description="d")
        _, user = build_implement_prompt(
            context_package, step, model_config=model_config,
            cumulative_diff="diff content",
        )
        assert "<prior_changes>" in user
        assert "diff content" in user

    def test_budget_overflow_raises(self, context_package, small_model_config):
        step = PlanStep(id="s1", description="d" * 5000)
        with pytest.raises(ValueError, match="Prompt too large"):
            build_implement_prompt(
                context_package, step, model_config=small_model_config,
            )


class TestBuildPlanPromptTestPlan:
    def test_test_plan_system(self, context_package, model_config):
        system, user = build_plan_prompt(
            context_package, "Plan tests for part p1",
            pass_type="test_plan", model_config=model_config,
        )
        assert system == TEST_PLAN_SYSTEM
        assert "Plan tests" in user

    def test_test_plan_with_cumulative_diff(self, context_package, model_config):
        _, user = build_plan_prompt(
            context_package, "Plan tests",
            pass_type="test_plan", model_config=model_config,
            cumulative_diff="--- a.py\n+new_func()",
        )
        assert "<prior_changes>" in user
        assert "new_func()" in user


class TestBuildTestImplementPrompt:
    def test_basic(self, context_package, model_config):
        step = PlanStep(id="t1", description="Test hello endpoint")
        system, user = build_implement_prompt(
            context_package, step,
            pass_type="test_implement", model_config=model_config,
        )
        assert system == TEST_IMPLEMENT_SYSTEM
        assert "t1" in user
        assert "Test hello endpoint" in user
        assert "Test Step to Implement" in user

    def test_with_test_plan(self, context_package, model_config):
        step = PlanStep(id="t1", description="Test step 1")
        test_plan = PartPlan(
            part_id="p1_tests", task_summary="Test coverage for p1",
            steps=[step, PlanStep(id="t2", description="Test step 2")],
            rationale="r",
        )
        _, user = build_implement_prompt(
            context_package, step,
            pass_type="test_implement", model_config=model_config,
            plan=test_plan,
        )
        assert "<plan_constraints>" in user
        assert "Test coverage for p1" in user
        assert "(current)" in user

    def test_with_cumulative_diff(self, context_package, model_config):
        step = PlanStep(id="t1", description="d")
        _, user = build_implement_prompt(
            context_package, step,
            pass_type="test_implement", model_config=model_config,
            cumulative_diff="diff content",
        )
        assert "<prior_changes>" in user

    def test_with_failure_context(self, context_package, model_config):
        step = PlanStep(id="t1", description="Fix test")
        failure = ValidationResult(
            success=False,
            test_output="ImportError",
            failing_tests=["test_bar"],
        )
        _, user = build_implement_prompt(
            context_package, step,
            pass_type="test_implement", model_config=model_config,
            failure_context=failure,
        )
        assert "<test_failures>" in user
        assert "ImportError" in user

    def test_budget_overflow_raises(self, context_package, small_model_config):
        step = PlanStep(id="t1", description="d" * 5000)
        with pytest.raises(ValueError, match="Prompt too large"):
            build_implement_prompt(
                context_package, step,
                pass_type="test_implement", model_config=small_model_config,
            )

    def test_unknown_implement_pass_type_raises(self, context_package, model_config):
        step = PlanStep(id="s1", description="d")
        with pytest.raises(ValueError, match="Unknown implement pass_type"):
            build_implement_prompt(
                context_package, step,
                pass_type="invalid", model_config=model_config,
            )

    def test_system_prompts_dict_complete(self):
        """SYSTEM_PROMPTS dict covers all expected pass types."""
        expected = {"meta_plan", "part_plan", "test_plan", "adjustment", "implement", "test_implement"}
        assert set(SYSTEM_PROMPTS.keys()) == expected
