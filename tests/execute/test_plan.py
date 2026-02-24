"""Tests for execute_plan function."""

import json
from unittest.mock import MagicMock

import pytest

from clean_room_agent.execute.dataclasses import MetaPlan, PartPlan, PlanAdjustment
from clean_room_agent.execute.plan import execute_plan
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


def _make_llm(response_text):
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(
        text=response_text,
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


class TestExecutePlanMetaPlan:
    def test_valid_meta_plan(self, context_package):
        response = json.dumps({
            "task_summary": "Add validation",
            "parts": [{"id": "p1", "description": "Add input checks", "affected_files": ["a.py"]}],
            "rationale": "Simple task",
        })
        llm = _make_llm(response)
        result = execute_plan(
            context_package, "Add validation", llm,
            pass_type="meta_plan",         )
        assert isinstance(result, MetaPlan)
        assert result.task_summary == "Add validation"
        assert llm.complete.call_count == 1

    def test_parse_failure_raises(self, context_package):
        llm = _make_llm("not valid json")
        with pytest.raises(ValueError, match="Failed to parse"):
            execute_plan(
                context_package, "task", llm,
                pass_type="meta_plan",             )

    def test_validation_failure_raises(self, context_package):
        response = json.dumps({
            "task_summary": "t",
            "parts": [
                {"id": "p1", "description": "d", "depends_on": ["p2"]},
                {"id": "p2", "description": "d", "depends_on": ["p1"]},
            ],
            "rationale": "r",
        })
        llm = _make_llm(response)
        with pytest.raises(ValueError, match="Circular"):
            execute_plan(
                context_package, "task", llm,
                pass_type="meta_plan",             )

    def test_prompt_includes_context(self, context_package):
        response = json.dumps({
            "task_summary": "t",
            "parts": [{"id": "p1", "description": "d"}],
            "rationale": "r",
        })
        llm = _make_llm(response)
        execute_plan(
            context_package, "Add validation", llm,
            pass_type="meta_plan",         )
        # Verify prompt was passed to LLM
        user_prompt = llm.complete.call_args[0][0]
        assert "src/main.py" in user_prompt
        assert "Add validation" in user_prompt


class TestExecutePlanPartPlan:
    def test_valid_part_plan(self, context_package):
        response = json.dumps({
            "part_id": "p1",
            "task_summary": "Add checks",
            "steps": [{"id": "s1", "description": "Add validator"}],
            "rationale": "r",
        })
        llm = _make_llm(response)
        result = execute_plan(
            context_package, "Implement part", llm,
            pass_type="part_plan",         )
        assert isinstance(result, PartPlan)
        assert result.part_id == "p1"


class TestExecutePlanAdjustment:
    def test_valid_adjustment(self, context_package):
        response = json.dumps({
            "revised_steps": [{"id": "s2", "description": "New step"}],
            "rationale": "Tests showed issue",
            "changes_made": ["Replaced s1 with s2"],
        })
        llm = _make_llm(response)
        result = execute_plan(
            context_package, "Adjust plan", llm,
            pass_type="adjustment",         )
        assert isinstance(result, PlanAdjustment)
        assert result.rationale == "Tests showed issue"

    def test_adjustment_with_cyclic_steps_raises(self, context_package):
        """T35: Adjustment with cyclic revised_steps raises ValueError."""
        response = json.dumps({
            "revised_steps": [
                {"id": "s1", "description": "Step 1", "depends_on": ["s2"]},
                {"id": "s2", "description": "Step 2", "depends_on": ["s1"]},
            ],
            "rationale": "Cyclic adjustment",
            "changes_made": ["Added cycle"],
        })
        llm = _make_llm(response)
        with pytest.raises(ValueError, match="Adjustment validation failed"):
            execute_plan(
                context_package, "Adjust plan", llm,
                pass_type="adjustment",             )

    def test_adjustment_with_duplicate_ids_raises(self, context_package):
        """T35: Adjustment with duplicate step IDs raises ValueError."""
        response = json.dumps({
            "revised_steps": [
                {"id": "s1", "description": "Step 1"},
                {"id": "s1", "description": "Step 1 duplicate"},
            ],
            "rationale": "Duplicate IDs",
            "changes_made": ["Duplicated"],
        })
        llm = _make_llm(response)
        with pytest.raises(ValueError, match="Adjustment validation failed"):
            execute_plan(
                context_package, "Adjust plan", llm,
                pass_type="adjustment",             )

    def test_adjustment_empty_revised_steps_skips_validation(self, context_package):
        """T35: Adjustment with empty revised_steps does NOT trigger validation."""
        response = json.dumps({
            "revised_steps": [],
            "rationale": "No changes needed",
            "changes_made": [],
        })
        llm = _make_llm(response)
        result = execute_plan(
            context_package, "Adjust plan", llm,
            pass_type="adjustment",         )
        assert isinstance(result, PlanAdjustment)
        assert result.revised_steps == []

    def test_adjustment_valid_dependencies_accepted(self, context_package):
        """T35: Adjustment with valid (non-cyclic) dependencies passes validation."""
        response = json.dumps({
            "revised_steps": [
                {"id": "s1", "description": "Step 1"},
                {"id": "s2", "description": "Step 2", "depends_on": ["s1"]},
                {"id": "s3", "description": "Step 3", "depends_on": ["s1", "s2"]},
            ],
            "rationale": "Valid chain",
            "changes_made": ["Reordered"],
        })
        llm = _make_llm(response)
        result = execute_plan(
            context_package, "Adjust plan", llm,
            pass_type="adjustment",         )
        assert isinstance(result, PlanAdjustment)
        assert len(result.revised_steps) == 3
