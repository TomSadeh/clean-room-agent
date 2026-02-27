"""Tests for Phase 3 execute-stage dataclasses."""

import pytest

from clean_room_agent.execute.dataclasses import (
    ChangePoint,
    ChangePointEnumeration,
    MetaPlan,
    MetaPlanPart,
    OrchestratorResult,
    PartGroup,
    PartGrouping,
    PartPlan,
    PatchEdit,
    PatchResult,
    PlanAdjustment,
    PlanArtifact,
    PlanStep,
    PassResult,
    StepResult,
    SymbolTarget,
    SymbolTargetEnumeration,
    ValidationResult,
)


class TestPlanStep:
    def test_basic_construction(self):
        step = PlanStep(id="s1", description="Add tests")
        assert step.id == "s1"
        assert step.target_files == []
        assert step.depends_on == []

    def test_full_construction(self):
        step = PlanStep(
            id="s1", description="Fix bug",
            target_files=["a.py"], target_symbols=["foo"],
            depends_on=["s0"],
        )
        assert step.target_files == ["a.py"]
        assert step.depends_on == ["s0"]

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="id must be non-empty"):
            PlanStep(id="", description="test")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description must be non-empty"):
            PlanStep(id="s1", description="")

    def test_round_trip(self):
        step = PlanStep(id="s1", description="Fix bug", target_files=["a.py"], depends_on=["s0"])
        d = step.to_dict()
        restored = PlanStep.from_dict(d)
        assert restored.id == step.id
        assert restored.description == step.description
        assert restored.target_files == step.target_files
        assert restored.depends_on == step.depends_on

    def test_from_dict_missing_key(self):
        with pytest.raises(ValueError, match="missing required key"):
            PlanStep.from_dict({"id": "s1"})

    def test_from_dict_defaults(self):
        step = PlanStep.from_dict({"id": "s1", "description": "test"})
        assert step.target_files == []
        assert step.target_symbols == []
        assert step.depends_on == []


class TestMetaPlanPart:
    def test_basic(self):
        part = MetaPlanPart(id="p1", description="Part 1")
        assert part.id == "p1"
        assert part.affected_files == []

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="id must be non-empty"):
            MetaPlanPart(id="", description="test")

    def test_round_trip(self):
        part = MetaPlanPart(id="p1", description="Part 1", affected_files=["a.py"], depends_on=["p0"])
        restored = MetaPlanPart.from_dict(part.to_dict())
        assert restored.id == part.id
        assert restored.affected_files == part.affected_files

    def test_from_dict_missing_key(self):
        with pytest.raises(ValueError, match="missing required key"):
            MetaPlanPart.from_dict({"id": "p1"})


class TestMetaPlan:
    def _make_plan(self):
        return MetaPlan(
            task_summary="Add validation",
            parts=[MetaPlanPart(id="p1", description="Part 1", affected_files=["a.py"])],
            rationale="Because it's needed",
        )

    def test_basic(self):
        plan = self._make_plan()
        assert plan.task_summary == "Add validation"
        assert len(plan.parts) == 1

    def test_empty_parts_raises(self):
        with pytest.raises(ValueError, match="parts must be non-empty"):
            MetaPlan(task_summary="t", parts=[], rationale="r")

    def test_empty_summary_raises(self):
        with pytest.raises(ValueError, match="task_summary must be non-empty"):
            MetaPlan(task_summary="", parts=[MetaPlanPart(id="p1", description="d")], rationale="r")

    def test_round_trip(self):
        plan = self._make_plan()
        d = plan.to_dict()
        restored = MetaPlan.from_dict(d)
        assert restored.task_summary == plan.task_summary
        assert len(restored.parts) == len(plan.parts)
        assert restored.parts[0].id == plan.parts[0].id

    def test_from_dict_missing_key(self):
        with pytest.raises(ValueError, match="missing required key"):
            MetaPlan.from_dict({"task_summary": "t", "rationale": "r"})


class TestPartPlan:
    def _make_plan(self):
        return PartPlan(
            part_id="p1",
            task_summary="Implement part 1",
            steps=[PlanStep(id="s1", description="Step 1")],
            rationale="Needed",
        )

    def test_basic(self):
        plan = self._make_plan()
        assert plan.part_id == "p1"
        assert len(plan.steps) == 1

    def test_empty_steps_raises(self):
        with pytest.raises(ValueError, match="steps must be non-empty"):
            PartPlan(part_id="p1", task_summary="t", steps=[], rationale="r")

    def test_round_trip(self):
        plan = self._make_plan()
        restored = PartPlan.from_dict(plan.to_dict())
        assert restored.part_id == plan.part_id
        assert restored.steps[0].id == plan.steps[0].id

    def test_from_dict_missing_key(self):
        with pytest.raises(ValueError, match="missing required key"):
            PartPlan.from_dict({"part_id": "p1"})


class TestPlanAdjustment:
    def test_basic(self):
        adj = PlanAdjustment(
            revised_steps=[PlanStep(id="s2", description="Revised step")],
            rationale="Test failed",
            changes_made=["Removed step s1"],
        )
        assert len(adj.revised_steps) == 1
        assert adj.changes_made == ["Removed step s1"]

    def test_empty_rationale_raises(self):
        with pytest.raises(ValueError, match="rationale must be non-empty"):
            PlanAdjustment(revised_steps=[], rationale="", changes_made=[])

    def test_empty_steps_allowed(self):
        adj = PlanAdjustment(revised_steps=[], rationale="All done", changes_made=["Cleared"])
        assert adj.revised_steps == []

    def test_round_trip(self):
        adj = PlanAdjustment(
            revised_steps=[PlanStep(id="s2", description="Step 2")],
            rationale="Adjusted", changes_made=["Changed s1 to s2"],
        )
        restored = PlanAdjustment.from_dict(adj.to_dict())
        assert restored.rationale == adj.rationale
        assert restored.changes_made == adj.changes_made
        assert restored.revised_steps[0].id == "s2"


class TestPlanArtifact:
    def test_basic(self):
        pa = PlanArtifact(
            task_summary="Add feature",
            affected_files=[{"path": "a.py", "role": "modified", "changes": "Add function"}],
            execution_order=["p1"],
            rationale="Because",
        )
        assert pa.task_summary == "Add feature"

    def test_empty_summary_raises(self):
        with pytest.raises(ValueError, match="task_summary must be non-empty"):
            PlanArtifact(task_summary="", affected_files=[], execution_order=[], rationale="r")

    def test_round_trip(self):
        pa = PlanArtifact(
            task_summary="t", affected_files=[{"path": "a.py"}],
            execution_order=["p1"], rationale="r",
        )
        restored = PlanArtifact.from_dict(pa.to_dict())
        assert restored.task_summary == pa.task_summary
        assert restored.affected_files == pa.affected_files

    def test_from_meta_plan(self):
        plan = MetaPlan(
            task_summary="Add validation",
            parts=[
                MetaPlanPart(id="p1", description="Part 1", affected_files=["a.py", "b.py"]),
                MetaPlanPart(id="p2", description="Part 2", affected_files=["b.py", "c.py"]),
            ],
            rationale="Because",
        )
        artifact = PlanArtifact.from_meta_plan(plan)
        assert artifact.task_summary == "Add validation"
        assert artifact.execution_order == ["p1", "p2"]
        # b.py should appear only once (deduped)
        paths = [f["path"] for f in artifact.affected_files]
        assert paths == ["a.py", "b.py", "c.py"]


class TestPatchEdit:
    def test_basic(self):
        edit = PatchEdit(file_path="a.py", search="old", replacement="new")
        assert edit.file_path == "a.py"

    def test_empty_file_path_raises(self):
        with pytest.raises(ValueError, match="file_path must be non-empty"):
            PatchEdit(file_path="", search="old", replacement="new")

    def test_empty_search_raises(self):
        with pytest.raises(ValueError, match="search must be non-empty"):
            PatchEdit(file_path="a.py", search="", replacement="new")

    def test_empty_replacement_allowed(self):
        edit = PatchEdit(file_path="a.py", search="old", replacement="")
        assert edit.replacement == ""

    def test_round_trip(self):
        edit = PatchEdit(file_path="a.py", search="old", replacement="new")
        restored = PatchEdit.from_dict(edit.to_dict())
        assert restored.file_path == edit.file_path
        assert restored.search == edit.search
        assert restored.replacement == edit.replacement


class TestStepResult:
    def test_success(self):
        sr = StepResult(
            success=True,
            edits=[PatchEdit(file_path="a.py", search="x", replacement="y")],
            raw_response="<edit>...</edit>",
        )
        assert sr.success is True
        assert len(sr.edits) == 1

    def test_failure(self):
        sr = StepResult(success=False, error_info="Parse failed", raw_response="garbage")
        assert sr.success is False
        assert sr.error_info == "Parse failed"

    def test_round_trip(self):
        sr = StepResult(
            success=True,
            edits=[PatchEdit(file_path="a.py", search="x", replacement="y")],
            raw_response="resp",
        )
        restored = StepResult.from_dict(sr.to_dict())
        assert restored.success == sr.success
        assert len(restored.edits) == 1
        assert restored.edits[0].file_path == "a.py"

    def test_from_dict_missing_success(self):
        with pytest.raises(ValueError, match="missing required key"):
            StepResult.from_dict({"edits": []})


class TestPatchResult:
    def test_success(self):
        pr = PatchResult(
            success=True, files_modified=["a.py", "b.py"],
            original_contents={"a.py": "old_a", "b.py": "old_b"},
        )
        assert pr.files_modified == ["a.py", "b.py"]

    def test_to_dict_excludes_originals(self):
        pr = PatchResult(
            success=True, files_modified=["a.py"],
            original_contents={"a.py": "old"},
        )
        d = pr.to_dict()
        assert "original_contents" not in d

    def test_round_trip(self):
        pr = PatchResult(success=False, error_info="File not found")
        restored = PatchResult.from_dict(pr.to_dict())
        assert restored.success is False
        assert restored.error_info == "File not found"


class TestValidationResult:
    def test_success(self):
        vr = ValidationResult(success=True, test_output="3 passed")
        assert vr.success is True
        assert vr.failing_tests == []

    def test_failure(self):
        vr = ValidationResult(
            success=False, test_output="1 failed",
            failing_tests=["test_foo"],
        )
        assert vr.failing_tests == ["test_foo"]

    def test_round_trip(self):
        vr = ValidationResult(
            success=False, test_output="1 failed",
            lint_output="E501", type_check_output="error",
            failing_tests=["test_foo"],
        )
        restored = ValidationResult.from_dict(vr.to_dict())
        assert restored.success == vr.success
        assert restored.test_output == vr.test_output
        assert restored.lint_output == vr.lint_output
        assert restored.failing_tests == vr.failing_tests


class TestPassResult:
    def test_basic(self):
        pr = PassResult(pass_type="meta_plan", task_run_id=1, success=True)
        assert pr.pass_type == "meta_plan"

    def test_empty_pass_type_raises(self):
        with pytest.raises(ValueError, match="pass_type must be non-empty"):
            PassResult(pass_type="", task_run_id=1, success=True)

    def test_to_dict_with_artifact(self):
        step = PlanStep(id="s1", description="d")
        adj = PlanAdjustment(
            revised_steps=[step], rationale="r", changes_made=["c"],
        )
        pr = PassResult(pass_type="adjustment", task_run_id=1, success=True, artifact=adj)
        d = pr.to_dict()
        assert "artifact" in d
        assert d["artifact"]["rationale"] == "r"

    def test_to_dict_without_artifact(self):
        pr = PassResult(pass_type="meta_plan", task_run_id=1, success=True)
        d = pr.to_dict()
        assert "artifact" not in d

    def test_task_run_id_none(self):
        """T38: PassResult can be created with task_run_id=None."""
        pr = PassResult(pass_type="step_implement", success=False, task_run_id=None)
        assert pr.task_run_id is None
        assert pr.pass_type == "step_implement"
        assert pr.success is False

    def test_task_run_id_none_to_dict(self):
        """T38: PassResult with task_run_id=None serializes correctly."""
        pr = PassResult(pass_type="adjustment", success=False, task_run_id=None)
        d = pr.to_dict()
        assert d["task_run_id"] is None
        assert d["pass_type"] == "adjustment"

    def test_task_run_id_default_is_none(self):
        """T38: PassResult defaults task_run_id to None when not specified."""
        pr = PassResult(pass_type="meta_plan", success=True)
        assert pr.task_run_id is None

    def test_from_dict(self):
        """T48: PassResult.from_dict round-trips correctly."""
        pr = PassResult(pass_type="step_implement", success=True, task_run_id=42)
        d = pr.to_dict()
        restored = PassResult.from_dict(d)
        assert restored.pass_type == "step_implement"
        assert restored.success is True
        assert restored.task_run_id == 42

    def test_from_dict_missing_key(self):
        """T48: PassResult.from_dict raises on missing required key."""
        with pytest.raises(ValueError, match="missing required key"):
            PassResult.from_dict({"pass_type": "meta_plan"})

    def test_from_dict_none_task_run_id(self):
        """T48: PassResult.from_dict handles None task_run_id."""
        restored = PassResult.from_dict({
            "pass_type": "adjustment", "success": False, "task_run_id": None,
        })
        assert restored.task_run_id is None


class TestOrchestratorResult:
    def test_basic(self):
        result = OrchestratorResult(task_id="t1", status="complete")
        assert result.task_id == "t1"
        assert result.parts_completed == 0

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="status must be one of"):
            OrchestratorResult(task_id="t1", status="invalid")

    def test_empty_task_id_raises(self):
        with pytest.raises(ValueError, match="task_id must be non-empty"):
            OrchestratorResult(task_id="", status="complete")

    def test_round_trip(self):
        result = OrchestratorResult(
            task_id="t1", status="partial",
            parts_completed=2, steps_completed=5,
            cumulative_diff="diff here",
        )
        restored = OrchestratorResult.from_dict(result.to_dict())
        assert restored.task_id == result.task_id
        assert restored.status == result.status
        assert restored.parts_completed == 2
        assert restored.steps_completed == 5
        assert restored.cumulative_diff == "diff here"

    def test_round_trip_with_pass_results(self):
        """T48: OrchestratorResult round-trip preserves pass_results."""
        result = OrchestratorResult(
            task_id="t1", status="partial",
            pass_results=[
                PassResult(pass_type="meta_plan", success=True, task_run_id=1),
                PassResult(pass_type="step_implement", success=False),
            ],
        )
        d = result.to_dict()
        assert len(d["pass_results"]) == 2
        restored = OrchestratorResult.from_dict(d)
        assert len(restored.pass_results) == 2
        assert restored.pass_results[0].pass_type == "meta_plan"
        assert restored.pass_results[0].success is True
        assert restored.pass_results[0].task_run_id == 1
        assert restored.pass_results[1].pass_type == "step_implement"
        assert restored.pass_results[1].success is False

    def test_from_dict_missing_key(self):
        with pytest.raises(ValueError, match="missing required key"):
            OrchestratorResult.from_dict({"task_id": "t1"})


# -- Decomposed planning dataclass tests --


class TestChangePoint:
    def test_basic(self):
        cp = ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="Fix bug")
        assert cp.file_path == "a.py"
        assert cp.change_type == "modify"

    def test_empty_file_path_raises(self):
        with pytest.raises(ValueError, match="file_path must be non-empty"):
            ChangePoint(file_path="", symbol="foo", change_type="modify", rationale="r")

    def test_empty_symbol_raises(self):
        with pytest.raises(ValueError, match="symbol must be non-empty"):
            ChangePoint(file_path="a.py", symbol="", change_type="modify", rationale="r")

    def test_round_trip(self):
        cp = ChangePoint(file_path="a.py", symbol="foo", change_type="add", rationale="New function")
        restored = ChangePoint.from_dict(cp.to_dict())
        assert restored.file_path == cp.file_path
        assert restored.symbol == cp.symbol
        assert restored.change_type == cp.change_type
        assert restored.rationale == cp.rationale

    def test_from_dict_missing_key(self):
        with pytest.raises(ValueError, match="missing required key"):
            ChangePoint.from_dict({"file_path": "a.py", "symbol": "foo"})


class TestChangePointEnumeration:
    def _make_enum(self):
        return ChangePointEnumeration(
            task_summary="Fix validation",
            change_points=[
                ChangePoint(file_path="a.py", symbol="validate", change_type="modify", rationale="Fix"),
            ],
        )

    def test_basic(self):
        enum = self._make_enum()
        assert enum.task_summary == "Fix validation"
        assert len(enum.change_points) == 1

    def test_empty_change_points_raises(self):
        with pytest.raises(ValueError, match="change_points must be non-empty"):
            ChangePointEnumeration(task_summary="t", change_points=[])

    def test_empty_summary_raises(self):
        with pytest.raises(ValueError, match="task_summary must be non-empty"):
            ChangePointEnumeration(
                task_summary="",
                change_points=[ChangePoint(file_path="a.py", symbol="f", change_type="modify", rationale="r")],
            )

    def test_round_trip(self):
        enum = self._make_enum()
        restored = ChangePointEnumeration.from_dict(enum.to_dict())
        assert restored.task_summary == enum.task_summary
        assert len(restored.change_points) == 1
        assert restored.change_points[0].file_path == "a.py"

    def test_from_dict_missing_key(self):
        with pytest.raises(ValueError, match="missing required key"):
            ChangePointEnumeration.from_dict({"task_summary": "t"})


class TestPartGroup:
    def test_basic(self):
        pg = PartGroup(id="p1", description="Part 1", change_point_indices=[0, 1])
        assert pg.id == "p1"
        assert pg.change_point_indices == [0, 1]
        assert pg.affected_files == []

    def test_with_affected_files(self):
        pg = PartGroup(
            id="p1", description="Part 1",
            change_point_indices=[0], affected_files=["a.py"],
        )
        assert pg.affected_files == ["a.py"]

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="id must be non-empty"):
            PartGroup(id="", description="d", change_point_indices=[0])

    def test_empty_indices_raises(self):
        with pytest.raises(ValueError, match="change_point_indices must be non-empty"):
            PartGroup(id="p1", description="d", change_point_indices=[])

    def test_round_trip(self):
        pg = PartGroup(id="p1", description="d", change_point_indices=[0, 2], affected_files=["a.py"])
        restored = PartGroup.from_dict(pg.to_dict())
        assert restored.id == pg.id
        assert restored.change_point_indices == [0, 2]
        assert restored.affected_files == ["a.py"]

    def test_from_dict_missing_key(self):
        with pytest.raises(ValueError, match="missing required key"):
            PartGroup.from_dict({"id": "p1"})

    def test_from_dict_validates_list_type(self):
        with pytest.raises(ValueError, match="must be a list"):
            PartGroup.from_dict({
                "id": "p1", "description": "d",
                "change_point_indices": "not a list",
            })


class TestPartGrouping:
    def test_basic(self):
        pg = PartGrouping(
            parts=[PartGroup(id="p1", description="d", change_point_indices=[0])],
        )
        assert len(pg.parts) == 1

    def test_empty_parts_raises(self):
        with pytest.raises(ValueError, match="parts must be non-empty"):
            PartGrouping(parts=[])

    def test_round_trip(self):
        pg = PartGrouping(
            parts=[
                PartGroup(id="p1", description="d1", change_point_indices=[0]),
                PartGroup(id="p2", description="d2", change_point_indices=[1]),
            ],
        )
        restored = PartGrouping.from_dict(pg.to_dict())
        assert len(restored.parts) == 2
        assert restored.parts[0].id == "p1"

    def test_from_dict_missing_key(self):
        with pytest.raises(ValueError, match="missing required key"):
            PartGrouping.from_dict({})


class TestSymbolTarget:
    def test_basic(self):
        st = SymbolTarget(file_path="a.py", symbol="foo", action="modify", rationale="Fix")
        assert st.file_path == "a.py"
        assert st.action == "modify"

    def test_empty_symbol_raises(self):
        with pytest.raises(ValueError, match="symbol must be non-empty"):
            SymbolTarget(file_path="a.py", symbol="", action="modify", rationale="r")

    def test_round_trip(self):
        st = SymbolTarget(file_path="a.py", symbol="foo", action="add", rationale="New")
        restored = SymbolTarget.from_dict(st.to_dict())
        assert restored.file_path == st.file_path
        assert restored.symbol == st.symbol
        assert restored.action == st.action

    def test_from_dict_missing_key(self):
        with pytest.raises(ValueError, match="missing required key"):
            SymbolTarget.from_dict({"file_path": "a.py"})


class TestSymbolTargetEnumeration:
    def _make_enum(self):
        return SymbolTargetEnumeration(
            part_id="p1",
            targets=[SymbolTarget(file_path="a.py", symbol="foo", action="modify", rationale="Fix")],
        )

    def test_basic(self):
        enum = self._make_enum()
        assert enum.part_id == "p1"
        assert len(enum.targets) == 1

    def test_empty_targets_raises(self):
        with pytest.raises(ValueError, match="targets must be non-empty"):
            SymbolTargetEnumeration(part_id="p1", targets=[])

    def test_empty_part_id_raises(self):
        with pytest.raises(ValueError, match="part_id must be non-empty"):
            SymbolTargetEnumeration(
                part_id="",
                targets=[SymbolTarget(file_path="a.py", symbol="f", action="modify", rationale="r")],
            )

    def test_round_trip(self):
        enum = self._make_enum()
        restored = SymbolTargetEnumeration.from_dict(enum.to_dict())
        assert restored.part_id == enum.part_id
        assert len(restored.targets) == 1
        assert restored.targets[0].symbol == "foo"

    def test_from_dict_missing_key(self):
        with pytest.raises(ValueError, match="missing required key"):
            SymbolTargetEnumeration.from_dict({"part_id": "p1"})
