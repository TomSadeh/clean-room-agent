"""Tests for orchestrator runner."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clean_room_agent.execute.dataclasses import (
    MetaPlan,
    MetaPlanPart,
    OrchestratorResult,
    PartPlan,
    PatchEdit,
    PlanAdjustment,
    PlanArtifact,
    PlanStep,
    StepResult,
    ValidationResult,
)
from clean_room_agent.retrieval.dataclasses import (
    BudgetConfig,
    ContextPackage,
    FileContent,
    TaskQuery,
)


def _make_context(task_id="test-001"):
    task = TaskQuery(
        raw_task="Test task", task_id=task_id, mode="plan", repo_id=1,
    )
    return ContextPackage(
        task=task,
        files=[FileContent(
            file_id=1, path="src/main.py", language="python",
            content="def hello(): pass", token_estimate=10,
            detail_level="primary",
        )],
        total_token_estimate=10,
        budget=BudgetConfig(context_window=32768, reserved_tokens=4096),
    )


def _make_meta_plan():
    return MetaPlan(
        task_summary="Test task",
        parts=[MetaPlanPart(id="p1", description="Part 1", affected_files=["a.py"])],
        rationale="Simple task",
    )


def _make_part_plan():
    return PartPlan(
        part_id="p1", task_summary="Part 1",
        steps=[PlanStep(id="s1", description="Step 1", target_files=["a.py"])],
        rationale="One step",
    )


def _make_step_result(success=True):
    if success:
        return StepResult(
            success=True,
            edits=[PatchEdit(file_path="a.py", search="old", replacement="new")],
            raw_response="<edit>...</edit>",
        )
    return StepResult(success=False, error_info="Parse failed", raw_response="garbage")


def _make_adjustment(revised_steps=None):
    return PlanAdjustment(
        revised_steps=revised_steps or [],
        rationale="No changes needed",
        changes_made=[],
    )


def _make_validation(success=True):
    return ValidationResult(
        success=success,
        test_output="3 passed" if success else "1 failed",
        failing_tests=[] if success else ["test_foo"],
    )


def _make_config():
    return {
        "models": {
            "provider": "ollama",
            "coding": "qwen2.5-coder:3b",
            "reasoning": "qwen3:4b",
            "base_url": "http://localhost:11434",
            "context_window": 32768,
        },
        "budget": {"reserved_tokens": 4096},
        "stages": {"default": "scope,precision"},
        "testing": {"test_command": "pytest tests/"},
        "orchestrator": {"max_retries_per_step": 1},
    }


class TestRunOrchestrator:
    @patch("clean_room_agent.orchestrator.runner.run_validation")
    @patch("clean_room_agent.orchestrator.runner.apply_edits")
    @patch("clean_room_agent.orchestrator.runner.execute_implement")
    @patch("clean_room_agent.orchestrator.runner.execute_plan")
    @patch("clean_room_agent.orchestrator.runner.run_pipeline")
    @patch("clean_room_agent.orchestrator.runner.get_connection")
    def test_successful_run(
        self, mock_get_conn, mock_pipeline, mock_exec_plan,
        mock_exec_impl, mock_apply, mock_validate, tmp_path,
    ):
        # Setup mock DB connections
        mock_raw_conn = MagicMock()
        mock_raw_conn.execute.return_value.lastrowid = 1
        mock_raw_conn.execute.return_value.fetchone.return_value = {"id": 1}
        mock_session_conn = MagicMock()
        mock_get_conn.side_effect = lambda role, **kw: (
            mock_raw_conn if role == "raw" else mock_session_conn
        )

        mock_pipeline.return_value = _make_context()

        # execute_plan returns: meta_plan, then part_plan, then adjustment
        mock_exec_plan.side_effect = [
            _make_meta_plan(),
            _make_part_plan(),
            _make_adjustment(),
        ]

        mock_exec_impl.return_value = _make_step_result(True)
        from clean_room_agent.execute.dataclasses import PatchResult
        mock_apply.return_value = PatchResult(success=True, files_modified=["a.py"])
        mock_validate.return_value = _make_validation(True)

        # Create required dirs
        (tmp_path / ".clean_room" / "tmp").mkdir(parents=True)

        from clean_room_agent.orchestrator.runner import run_orchestrator
        result = run_orchestrator("Test task", tmp_path, _make_config())

        assert isinstance(result, OrchestratorResult)
        assert result.status == "complete"
        assert result.parts_completed == 1
        assert result.steps_completed == 1
        assert len(result.pass_results) > 0

    @patch("clean_room_agent.orchestrator.runner.run_validation")
    @patch("clean_room_agent.orchestrator.runner.apply_edits")
    @patch("clean_room_agent.orchestrator.runner.execute_implement")
    @patch("clean_room_agent.orchestrator.runner.execute_plan")
    @patch("clean_room_agent.orchestrator.runner.run_pipeline")
    @patch("clean_room_agent.orchestrator.runner.get_connection")
    def test_failed_implementation(
        self, mock_get_conn, mock_pipeline, mock_exec_plan,
        mock_exec_impl, mock_apply, mock_validate, tmp_path,
    ):
        mock_raw_conn = MagicMock()
        mock_raw_conn.execute.return_value.lastrowid = 1
        mock_raw_conn.execute.return_value.fetchone.return_value = {"id": 1}
        mock_session_conn = MagicMock()
        mock_get_conn.side_effect = lambda role, **kw: (
            mock_raw_conn if role == "raw" else mock_session_conn
        )

        mock_pipeline.return_value = _make_context()
        mock_exec_plan.side_effect = [
            _make_meta_plan(),
            _make_part_plan(),
            _make_adjustment(),
        ]
        # Implementation always fails
        mock_exec_impl.return_value = _make_step_result(False)

        (tmp_path / ".clean_room" / "tmp").mkdir(parents=True)

        from clean_room_agent.orchestrator.runner import run_orchestrator
        result = run_orchestrator("Test task", tmp_path, _make_config())

        assert result.status == "failed"

    @patch("clean_room_agent.orchestrator.runner.run_validation")
    @patch("clean_room_agent.orchestrator.runner.apply_edits")
    @patch("clean_room_agent.orchestrator.runner.execute_implement")
    @patch("clean_room_agent.orchestrator.runner.execute_plan")
    @patch("clean_room_agent.orchestrator.runner.run_pipeline")
    @patch("clean_room_agent.orchestrator.runner.get_connection")
    def test_retry_on_test_failure(
        self, mock_get_conn, mock_pipeline, mock_exec_plan,
        mock_exec_impl, mock_apply, mock_validate, tmp_path,
    ):
        mock_raw_conn = MagicMock()
        mock_raw_conn.execute.return_value.lastrowid = 1
        mock_raw_conn.execute.return_value.fetchone.return_value = {"id": 1}
        mock_session_conn = MagicMock()
        mock_get_conn.side_effect = lambda role, **kw: (
            mock_raw_conn if role == "raw" else mock_session_conn
        )

        mock_pipeline.return_value = _make_context()
        mock_exec_plan.side_effect = [
            _make_meta_plan(),
            _make_part_plan(),
            _make_adjustment(),
        ]
        mock_exec_impl.return_value = _make_step_result(True)
        from clean_room_agent.execute.dataclasses import PatchResult
        mock_apply.return_value = PatchResult(success=True, files_modified=["a.py"])

        # First test fails, retry succeeds
        mock_validate.side_effect = [
            _make_validation(False),
            _make_validation(True),
        ]

        (tmp_path / ".clean_room" / "tmp").mkdir(parents=True)

        from clean_room_agent.orchestrator.runner import run_orchestrator
        result = run_orchestrator("Test task", tmp_path, _make_config())

        assert result.status == "complete"
        # Implementation called twice (initial + retry)
        assert mock_exec_impl.call_count == 2


    @patch("clean_room_agent.orchestrator.runner.run_validation")
    @patch("clean_room_agent.orchestrator.runner.apply_edits")
    @patch("clean_room_agent.orchestrator.runner.execute_implement")
    @patch("clean_room_agent.orchestrator.runner.execute_plan")
    @patch("clean_room_agent.orchestrator.runner.run_pipeline")
    @patch("clean_room_agent.orchestrator.runner.get_connection")
    def test_adjustment_revises_remaining_steps(
        self, mock_get_conn, mock_pipeline, mock_exec_plan,
        mock_exec_impl, mock_apply, mock_validate, tmp_path,
    ):
        """T25: Adjustment pass revised_steps replace remaining steps."""
        mock_raw_conn = MagicMock()
        mock_raw_conn.execute.return_value.lastrowid = 1
        mock_raw_conn.execute.return_value.fetchone.return_value = {"id": 1}
        mock_session_conn = MagicMock()
        mock_get_conn.side_effect = lambda role, **kw: (
            mock_raw_conn if role == "raw" else mock_session_conn
        )

        mock_pipeline.return_value = _make_context()

        # Part plan has TWO steps: s1 and s2
        two_step_plan = PartPlan(
            part_id="p1", task_summary="Part 1",
            steps=[
                PlanStep(id="s1", description="Step 1", target_files=["a.py"]),
                PlanStep(id="s2", description="Step 2", target_files=["b.py"]),
            ],
            rationale="Two steps",
        )

        # After s1 completes, the adjustment replaces remaining [s2] with [s3]
        revised_step = PlanStep(id="s3", description="Revised step 3", target_files=["c.py"])
        adj_with_revision = PlanAdjustment(
            revised_steps=[revised_step],
            rationale="Replacing s2 with s3",
            changes_made=["replaced s2"],
        )
        # After s3 completes, second adjustment has no revisions
        adj_no_revision = _make_adjustment()

        mock_exec_plan.side_effect = [
            _make_meta_plan(),
            two_step_plan,
            adj_with_revision,
            adj_no_revision,
        ]

        mock_exec_impl.return_value = _make_step_result(True)
        from clean_room_agent.execute.dataclasses import PatchResult
        mock_apply.return_value = PatchResult(success=True, files_modified=["a.py"])
        mock_validate.return_value = _make_validation(True)

        (tmp_path / ".clean_room" / "tmp").mkdir(parents=True)

        from clean_room_agent.orchestrator.runner import run_orchestrator
        result = run_orchestrator("Test task", tmp_path, _make_config())

        assert result.status == "complete"
        # Both s1 and s3 should have been executed (s2 was replaced)
        assert result.steps_completed == 2
        assert mock_exec_impl.call_count == 2

        # Verify that the second implement call used the revised step's
        # target_files (c.py), not the original s2's target_files (b.py)
        second_impl_call = mock_exec_impl.call_args_list[1]
        step_arg = second_impl_call[0][1]  # second positional arg = step
        assert step_arg.id == "s3"
        assert step_arg.target_files == ["c.py"]

    @patch("clean_room_agent.orchestrator.runner.run_validation")
    @patch("clean_room_agent.orchestrator.runner.apply_edits")
    @patch("clean_room_agent.orchestrator.runner.execute_implement")
    @patch("clean_room_agent.orchestrator.runner.execute_plan")
    @patch("clean_room_agent.orchestrator.runner.run_pipeline")
    @patch("clean_room_agent.orchestrator.runner.get_connection")
    def test_steps_completed_counts_only_successes(
        self, mock_get_conn, mock_pipeline, mock_exec_plan,
        mock_exec_impl, mock_apply, mock_validate, tmp_path,
    ):
        """T26: steps_completed should only count successful steps."""
        mock_raw_conn = MagicMock()
        mock_raw_conn.execute.return_value.lastrowid = 1
        mock_raw_conn.execute.return_value.fetchone.return_value = {"id": 1}
        mock_session_conn = MagicMock()
        mock_get_conn.side_effect = lambda role, **kw: (
            mock_raw_conn if role == "raw" else mock_session_conn
        )

        mock_pipeline.return_value = _make_context()
        mock_exec_plan.side_effect = [
            _make_meta_plan(),
            _make_part_plan(),
            _make_adjustment(),
        ]
        # Implementation always fails
        mock_exec_impl.return_value = _make_step_result(False)

        (tmp_path / ".clean_room" / "tmp").mkdir(parents=True)

        from clean_room_agent.orchestrator.runner import run_orchestrator
        result = run_orchestrator("Test task", tmp_path, _make_config())

        assert result.status == "failed"
        assert result.steps_completed == 0  # T26: failed steps not counted


class TestRunSinglePass:
    @patch("clean_room_agent.orchestrator.runner.run_validation")
    @patch("clean_room_agent.orchestrator.runner.apply_edits")
    @patch("clean_room_agent.orchestrator.runner.execute_implement")
    @patch("clean_room_agent.orchestrator.runner.run_pipeline")
    @patch("clean_room_agent.orchestrator.runner.get_connection")
    def test_successful_single_pass(
        self, mock_get_conn, mock_pipeline, mock_exec_impl,
        mock_apply, mock_validate, tmp_path,
    ):
        mock_raw_conn = MagicMock()
        mock_raw_conn.execute.return_value.lastrowid = 1
        mock_raw_conn.execute.return_value.fetchone.return_value = {"id": 1}
        mock_get_conn.return_value = mock_raw_conn

        mock_pipeline.return_value = _make_context()
        mock_exec_impl.return_value = _make_step_result(True)
        from clean_room_agent.execute.dataclasses import PatchResult
        mock_apply.return_value = PatchResult(success=True, files_modified=["a.py"])
        mock_validate.return_value = _make_validation(True)

        # Write plan file
        plan = PlanArtifact(
            task_summary="Fix bug",
            affected_files=[{"path": "a.py", "role": "modified", "changes": "fix"}],
            execution_order=["p1"],
            rationale="test",
        )
        plan_file = tmp_path / "plan.json"
        plan_file.write_text(json.dumps(plan.to_dict()))

        from clean_room_agent.orchestrator.runner import run_single_pass
        result = run_single_pass("Fix bug", tmp_path, _make_config(), plan_path=plan_file)

        assert result.status == "complete"
        assert result.steps_completed == 1

    @patch("clean_room_agent.orchestrator.runner.run_validation")
    @patch("clean_room_agent.orchestrator.runner.apply_edits")
    @patch("clean_room_agent.orchestrator.runner.execute_implement")
    @patch("clean_room_agent.orchestrator.runner.run_pipeline")
    @patch("clean_room_agent.orchestrator.runner.get_connection")
    def test_single_pass_failure(
        self, mock_get_conn, mock_pipeline, mock_exec_impl,
        mock_apply, mock_validate, tmp_path,
    ):
        mock_raw_conn = MagicMock()
        mock_raw_conn.execute.return_value.lastrowid = 1
        mock_raw_conn.execute.return_value.fetchone.return_value = {"id": 1}
        mock_get_conn.return_value = mock_raw_conn

        mock_pipeline.return_value = _make_context()
        mock_exec_impl.return_value = _make_step_result(False)

        plan = PlanArtifact(
            task_summary="Fix bug",
            affected_files=[{"path": "a.py"}],
            execution_order=["p1"],
            rationale="test",
        )
        plan_file = tmp_path / "plan.json"
        plan_file.write_text(json.dumps(plan.to_dict()))

        from clean_room_agent.orchestrator.runner import run_single_pass
        result = run_single_pass("Fix bug", tmp_path, _make_config(), plan_path=plan_file)

        assert result.status == "failed"

    @patch("clean_room_agent.orchestrator.runner.run_validation")
    @patch("clean_room_agent.orchestrator.runner.apply_edits")
    @patch("clean_room_agent.orchestrator.runner.execute_implement")
    @patch("clean_room_agent.orchestrator.runner.run_pipeline")
    @patch("clean_room_agent.orchestrator.runner.get_connection")
    def test_single_pass_retry(
        self, mock_get_conn, mock_pipeline, mock_exec_impl,
        mock_apply, mock_validate, tmp_path,
    ):
        mock_raw_conn = MagicMock()
        mock_raw_conn.execute.return_value.lastrowid = 1
        mock_raw_conn.execute.return_value.fetchone.return_value = {"id": 1}
        mock_get_conn.return_value = mock_raw_conn

        mock_pipeline.return_value = _make_context()
        mock_exec_impl.return_value = _make_step_result(True)
        from clean_room_agent.execute.dataclasses import PatchResult
        mock_apply.return_value = PatchResult(success=True, files_modified=["a.py"])
        # First validation fails, second succeeds
        mock_validate.side_effect = [_make_validation(False), _make_validation(True)]

        plan = PlanArtifact(
            task_summary="Fix bug",
            affected_files=[{"path": "a.py"}],
            execution_order=["p1"],
            rationale="test",
        )
        plan_file = tmp_path / "plan.json"
        plan_file.write_text(json.dumps(plan.to_dict()))

        from clean_room_agent.orchestrator.runner import run_single_pass
        result = run_single_pass("Fix bug", tmp_path, _make_config(), plan_path=plan_file)

        assert result.status == "complete"
        assert mock_exec_impl.call_count == 2


class TestTopologicalSort:
    def test_simple_chain(self):
        from clean_room_agent.orchestrator.runner import _topological_sort
        items = [
            PlanStep(id="s2", description="d", depends_on=["s1"]),
            PlanStep(id="s1", description="d"),
        ]
        sorted_items = _topological_sort(items, lambda s: s.id, lambda s: s.depends_on)
        assert [s.id for s in sorted_items] == ["s1", "s2"]

    def test_no_dependencies(self):
        from clean_room_agent.orchestrator.runner import _topological_sort
        items = [
            PlanStep(id="s1", description="d"),
            PlanStep(id="s2", description="d"),
        ]
        sorted_items = _topological_sort(items, lambda s: s.id, lambda s: s.depends_on)
        assert len(sorted_items) == 2

    def test_diamond_dependency(self):
        from clean_room_agent.orchestrator.runner import _topological_sort
        items = [
            PlanStep(id="s4", description="d", depends_on=["s2", "s3"]),
            PlanStep(id="s2", description="d", depends_on=["s1"]),
            PlanStep(id="s3", description="d", depends_on=["s1"]),
            PlanStep(id="s1", description="d"),
        ]
        sorted_items = _topological_sort(items, lambda s: s.id, lambda s: s.depends_on)
        ids = [s.id for s in sorted_items]
        assert ids.index("s1") < ids.index("s2")
        assert ids.index("s1") < ids.index("s3")
        assert ids.index("s2") < ids.index("s4")
        assert ids.index("s3") < ids.index("s4")

    def test_cycle_raises(self):
        from clean_room_agent.orchestrator.runner import _topological_sort
        items = [
            PlanStep(id="s1", description="d", depends_on=["s2"]),
            PlanStep(id="s2", description="d", depends_on=["s1"]),
        ]
        with pytest.raises(RuntimeError, match="Cycle detected"):
            _topological_sort(items, lambda s: s.id, lambda s: s.depends_on)


class TestResolveBudgetMissingConfig:
    """T36: _resolve_budget raises RuntimeError when reserved_tokens is missing."""

    def test_missing_reserved_tokens_raises(self):
        from clean_room_agent.orchestrator.runner import _resolve_budget
        config = {
            "models": {
                "provider": "ollama",
                "coding": "qwen2.5-coder:3b",
                "reasoning": "qwen3:4b",
                "base_url": "http://localhost:11434",
                "context_window": 32768,
            },
            "budget": {},  # reserved_tokens missing
        }
        with pytest.raises(RuntimeError, match="Missing reserved_tokens"):
            _resolve_budget(config, "coding")

    def test_missing_budget_section_raises(self):
        from clean_room_agent.orchestrator.runner import _resolve_budget
        config = {
            "models": {
                "provider": "ollama",
                "coding": "qwen2.5-coder:3b",
                "reasoning": "qwen3:4b",
                "base_url": "http://localhost:11434",
                "context_window": 32768,
            },
            # No budget section at all
        }
        with pytest.raises(RuntimeError, match="Missing reserved_tokens"):
            _resolve_budget(config, "coding")

    def test_valid_budget_succeeds(self):
        from clean_room_agent.orchestrator.runner import _resolve_budget
        config = {
            "models": {
                "provider": "ollama",
                "coding": "qwen2.5-coder:3b",
                "reasoning": "qwen3:4b",
                "base_url": "http://localhost:11434",
                "context_window": 32768,
            },
            "budget": {"reserved_tokens": 4096},
        }
        budget = _resolve_budget(config, "coding")
        assert budget.context_window == 32768
        assert budget.reserved_tokens == 4096


class TestResolveStagesMissingConfig:
    """T36: _resolve_stages raises RuntimeError when default stages missing."""

    def test_missing_default_stages_raises(self):
        from clean_room_agent.orchestrator.runner import _resolve_stages
        config = {"stages": {}}  # no "default" key
        with pytest.raises(RuntimeError, match="Missing default"):
            _resolve_stages(config)

    def test_missing_stages_section_raises(self):
        from clean_room_agent.orchestrator.runner import _resolve_stages
        config = {}  # no stages section at all
        with pytest.raises(RuntimeError, match="Missing default"):
            _resolve_stages(config)

    def test_empty_default_stages_raises(self):
        from clean_room_agent.orchestrator.runner import _resolve_stages
        config = {"stages": {"default": ""}}  # empty string
        with pytest.raises(RuntimeError, match="Missing default"):
            _resolve_stages(config)

    def test_valid_stages_succeeds(self):
        from clean_room_agent.orchestrator.runner import _resolve_stages
        config = {"stages": {"default": "scope,precision"}}
        stages = _resolve_stages(config)
        assert stages == ["scope", "precision"]


class TestOrchestratorMissingMaxRetries:
    """T36: Orchestrator raises RuntimeError when max_retries_per_step is missing."""

    @patch("clean_room_agent.orchestrator.runner.run_pipeline")
    @patch("clean_room_agent.orchestrator.runner.get_connection")
    def test_missing_max_retries_raises(self, mock_get_conn, mock_pipeline, tmp_path):
        from clean_room_agent.orchestrator.runner import run_orchestrator
        config = {
            "models": {
                "provider": "ollama",
                "coding": "qwen2.5-coder:3b",
                "reasoning": "qwen3:4b",
                "base_url": "http://localhost:11434",
                "context_window": 32768,
            },
            "budget": {"reserved_tokens": 4096},
            "stages": {"default": "scope,precision"},
            "orchestrator": {},  # max_retries_per_step missing
        }
        mock_raw_conn = MagicMock()
        mock_raw_conn.execute.return_value.lastrowid = 1
        mock_raw_conn.execute.return_value.fetchone.return_value = {"id": 1}
        mock_session_conn = MagicMock()
        mock_get_conn.side_effect = lambda role, **kw: (
            mock_raw_conn if role == "raw" else mock_session_conn
        )

        (tmp_path / ".clean_room" / "tmp").mkdir(parents=True)

        with pytest.raises(RuntimeError, match="Missing max_retries_per_step"):
            run_orchestrator("Test task", tmp_path, config)

    @patch("clean_room_agent.orchestrator.runner.run_pipeline")
    @patch("clean_room_agent.orchestrator.runner.get_connection")
    def test_single_pass_missing_max_retries_raises(self, mock_get_conn, mock_pipeline, tmp_path):
        from clean_room_agent.orchestrator.runner import run_single_pass
        config = {
            "models": {
                "provider": "ollama",
                "coding": "qwen2.5-coder:3b",
                "reasoning": "qwen3:4b",
                "base_url": "http://localhost:11434",
                "context_window": 32768,
            },
            "budget": {"reserved_tokens": 4096},
            "stages": {"default": "scope,precision"},
            "orchestrator": {},  # max_retries_per_step missing
        }
        mock_raw_conn = MagicMock()
        mock_get_conn.return_value = mock_raw_conn

        plan = PlanArtifact(
            task_summary="Fix bug",
            affected_files=[{"path": "a.py"}],
            execution_order=["p1"],
            rationale="test",
        )
        plan_file = tmp_path / "plan.json"
        plan_file.write_text(json.dumps(plan.to_dict()))

        with pytest.raises(RuntimeError, match="Missing max_retries_per_step"):
            run_single_pass("Fix bug", tmp_path, config, plan_path=plan_file)


class TestCapCumulativeDiff:
    """T52: Cumulative diff is bounded to prevent unbounded context growth."""

    def test_short_diff_unchanged(self):
        from clean_room_agent.orchestrator.runner import _cap_cumulative_diff
        diff = "--- a.py\n-old\n+new\n"
        assert _cap_cumulative_diff(diff) == diff

    def test_long_diff_truncated(self):
        from clean_room_agent.orchestrator.runner import (
            _MAX_CUMULATIVE_DIFF_CHARS,
            _cap_cumulative_diff,
        )
        # Build a diff that exceeds the cap
        block = "--- file.py\n-old line\n+new line\n"
        repeats = (_MAX_CUMULATIVE_DIFF_CHARS // len(block)) + 10
        long_diff = block * repeats
        assert len(long_diff) > _MAX_CUMULATIVE_DIFF_CHARS

        result = _cap_cumulative_diff(long_diff)
        assert len(result) <= _MAX_CUMULATIVE_DIFF_CHARS + 100  # allow header
        assert result.startswith("[earlier changes truncated]\n")
        assert "--- file.py" in result

    def test_truncation_aligns_to_block_boundary(self):
        from clean_room_agent.orchestrator.runner import (
            _MAX_CUMULATIVE_DIFF_CHARS,
            _cap_cumulative_diff,
        )
        block = "--- file.py\n-old line content\n+new line content\n"
        repeats = (_MAX_CUMULATIVE_DIFF_CHARS // len(block)) + 10
        long_diff = block * repeats

        result = _cap_cumulative_diff(long_diff)
        # After the header, should start at a block boundary
        lines = result.split("\n")
        # First line is the truncation notice
        assert lines[0] == "[earlier changes truncated]"
        # Second line should be a block start
        assert lines[1].startswith("--- ")
