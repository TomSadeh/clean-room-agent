"""Tests for decomposed planning (multi-stage meta-plan and part-plan)."""

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from clean_room_agent.execute.dataclasses import (
    ChangePoint,
    ChangePointEnumeration,
    MetaPlan,
    PartGroup,
    PartGrouping,
    PartPlan,
    PlanStep,
    SymbolTarget,
    SymbolTargetEnumeration,
)
from clean_room_agent.execute.decomposed_plan import (
    _assemble_meta_plan,
    _assemble_part_plan,
    _build_grouping_from_components,
    _generate_part_description,
    _run_change_point_enum,
    _run_part_dependencies,
    _run_part_grouping,
    _run_part_grouping_binary,
    _run_step_dependencies,
    _run_step_design,
    _run_symbol_targeting,
    _single_change_point_grouping,
    _union_find_groups,
    decomposed_meta_plan,
    decomposed_part_plan,
)
from clean_room_agent.llm.client import LLMResponse, ModelConfig
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


def _make_mock_llm(model_config, responses):
    """Create a mock LoggedLLMClient that returns responses in order."""
    llm = MagicMock()
    llm.config = model_config
    llm.flush.return_value = []

    response_iter = iter(responses)
    def complete_side_effect(prompt, system=None):
        text = next(response_iter)
        return LLMResponse(
            text=text, thinking=None,
            prompt_tokens=100, completion_tokens=50, latency_ms=100,
        )
    llm.complete.side_effect = complete_side_effect
    return llm


# -- Change point enumeration tests --


class TestRunChangePointEnum:
    def test_valid_response(self, context_package, model_config):
        response_json = json.dumps({
            "task_summary": "Add validation",
            "change_points": [
                {"file_path": "a.py", "symbol": "validate", "change_type": "modify", "rationale": "Fix"},
                {"file_path": "b.py", "symbol": "check", "change_type": "add", "rationale": "New"},
            ],
        })
        llm = _make_mock_llm(model_config, [response_json])
        result = _run_change_point_enum(context_package, "Add validation", llm)
        assert isinstance(result, ChangePointEnumeration)
        assert len(result.change_points) == 2
        assert result.task_summary == "Add validation"

    def test_invalid_json_raises(self, context_package, model_config):
        llm = _make_mock_llm(model_config, ["not json"])
        with pytest.raises(ValueError):
            _run_change_point_enum(context_package, "task", llm)


# -- Part grouping tests --


class TestRunPartGrouping:
    def test_valid_response(self, context_package, model_config):
        enum_result = ChangePointEnumeration(
            task_summary="t",
            change_points=[
                ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
                ChangePoint(file_path="b.py", symbol="bar", change_type="add", rationale="r"),
            ],
        )
        response_json = json.dumps({
            "parts": [
                {"id": "p1", "description": "Part 1", "change_point_indices": [0], "affected_files": ["a.py"]},
                {"id": "p2", "description": "Part 2", "change_point_indices": [1], "affected_files": ["b.py"]},
            ],
        })
        llm = _make_mock_llm(model_config, [response_json])
        result = _run_part_grouping(context_package, "task", enum_result, llm)
        assert isinstance(result, PartGrouping)
        assert len(result.parts) == 2

    def test_invalid_grouping_raises(self, context_package, model_config):
        """Missing index raises validation error."""
        enum_result = ChangePointEnumeration(
            task_summary="t",
            change_points=[
                ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
                ChangePoint(file_path="b.py", symbol="bar", change_type="add", rationale="r"),
            ],
        )
        # Only assigns index 0, missing index 1
        response_json = json.dumps({
            "parts": [
                {"id": "p1", "description": "Part 1", "change_point_indices": [0], "affected_files": ["a.py"]},
            ],
        })
        llm = _make_mock_llm(model_config, [response_json])
        with pytest.raises(ValueError, match="grouping validation failed"):
            _run_part_grouping(context_package, "task", enum_result, llm)


# -- Part dependency tests --


class TestRunPartDependencies:
    def test_single_part_no_deps(self, model_config):
        grouping = PartGrouping(parts=[
            PartGroup(id="p1", description="d1", change_point_indices=[0]),
        ])
        llm = _make_mock_llm(model_config, [])
        result = _run_part_dependencies(grouping, "task", llm)
        assert result == {"p1": []}

    def test_two_parts_with_dependency(self, model_config):
        grouping = PartGrouping(parts=[
            PartGroup(id="p1", description="Foundation", change_point_indices=[0]),
            PartGroup(id="p2", description="Depends on foundation", change_point_indices=[1]),
        ])
        # Pairs: (p1,p2) and (p2,p1). p2 depends on p1 → yes; p1 depends on p2 → no
        llm = _make_mock_llm(model_config, ["yes", "no"])
        result = _run_part_dependencies(grouping, "task", llm)
        assert result["p1"] == []
        assert result["p2"] == ["p1"]

    def test_two_parts_no_dependency(self, model_config):
        grouping = PartGrouping(parts=[
            PartGroup(id="p1", description="Independent A", change_point_indices=[0]),
            PartGroup(id="p2", description="Independent B", change_point_indices=[1]),
        ])
        llm = _make_mock_llm(model_config, ["no", "no"])
        result = _run_part_dependencies(grouping, "task", llm)
        assert result["p1"] == []
        assert result["p2"] == []

    def test_unparseable_response_raises(self, model_config):
        """A1: Unparseable binary response raises — incomplete judgments are not silent."""
        grouping = PartGrouping(parts=[
            PartGroup(id="p1", description="d1", change_point_indices=[0]),
            PartGroup(id="p2", description="d2", change_point_indices=[1]),
        ])
        llm = _make_mock_llm(model_config, ["garbage", "garbage"])
        with pytest.raises(ValueError, match="failed to parse"):
            _run_part_dependencies(grouping, "task", llm)


# -- Assembly tests --


class TestAssembleMetaPlan:
    def test_basic_assembly(self):
        enum_result = ChangePointEnumeration(
            task_summary="Fix bugs",
            change_points=[
                ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
            ],
        )
        grouping = PartGrouping(parts=[
            PartGroup(id="p1", description="Fix foo", change_point_indices=[0], affected_files=["a.py"]),
        ])
        dep_edges = {"p1": []}
        result = _assemble_meta_plan(enum_result, grouping, dep_edges)
        assert isinstance(result, MetaPlan)
        assert result.task_summary == "Fix bugs"
        assert len(result.parts) == 1
        assert result.parts[0].id == "p1"
        assert result.parts[0].affected_files == ["a.py"]
        assert result.parts[0].depends_on == []

    def test_with_dependencies(self):
        enum_result = ChangePointEnumeration(
            task_summary="t",
            change_points=[
                ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
                ChangePoint(file_path="b.py", symbol="bar", change_type="add", rationale="r"),
            ],
        )
        grouping = PartGrouping(parts=[
            PartGroup(id="p1", description="d1", change_point_indices=[0], affected_files=["a.py"]),
            PartGroup(id="p2", description="d2", change_point_indices=[1], affected_files=["b.py"]),
        ])
        dep_edges = {"p1": [], "p2": ["p1"]}
        result = _assemble_meta_plan(enum_result, grouping, dep_edges)
        assert result.parts[1].depends_on == ["p1"]

    def test_cycle_raises(self):
        enum_result = ChangePointEnumeration(
            task_summary="t",
            change_points=[
                ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
                ChangePoint(file_path="b.py", symbol="bar", change_type="modify", rationale="r"),
            ],
        )
        grouping = PartGrouping(parts=[
            PartGroup(id="p1", description="d1", change_point_indices=[0]),
            PartGroup(id="p2", description="d2", change_point_indices=[1]),
        ])
        dep_edges = {"p1": ["p2"], "p2": ["p1"]}
        with pytest.raises(ValueError, match="validation failed"):
            _assemble_meta_plan(enum_result, grouping, dep_edges)

    def test_missing_dep_edges_key_raises(self):
        """M1: Missing key in dep_edges is an invariant violation."""
        enum_result = ChangePointEnumeration(
            task_summary="t",
            change_points=[
                ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
            ],
        )
        grouping = PartGrouping(parts=[
            PartGroup(id="p1", description="d1", change_point_indices=[0], affected_files=["a.py"]),
        ])
        dep_edges = {}  # Missing p1 key
        with pytest.raises(KeyError):
            _assemble_meta_plan(enum_result, grouping, dep_edges)


# -- Symbol targeting tests --


class TestRunSymbolTargeting:
    def test_valid_response(self, context_package, model_config):
        response_json = json.dumps({
            "part_id": "p1",
            "targets": [
                {"file_path": "a.py", "symbol": "validate", "action": "modify", "rationale": "Fix"},
            ],
        })
        llm = _make_mock_llm(model_config, [response_json])
        result = _run_symbol_targeting(context_package, "Part 1 desc", "p1", llm, None)
        assert isinstance(result, SymbolTargetEnumeration)
        assert result.part_id == "p1"
        assert len(result.targets) == 1


# -- Step design tests --


class TestRunStepDesign:
    def test_valid_response(self, context_package, model_config):
        targets = SymbolTargetEnumeration(
            part_id="p1",
            targets=[SymbolTarget(file_path="a.py", symbol="foo", action="modify", rationale="Fix")],
        )
        response_json = json.dumps({
            "part_id": "p1",
            "task_summary": "Fix foo",
            "steps": [{"id": "s1", "description": "Modify foo", "target_files": ["a.py"], "target_symbols": ["foo"], "depends_on": []}],
            "rationale": "One step needed",
        })
        llm = _make_mock_llm(model_config, [response_json])
        result = _run_step_design(context_package, "Part 1 desc", targets, llm, None)
        assert isinstance(result, PartPlan)
        assert len(result.steps) == 1


# -- Step dependency tests --


class TestRunStepDependencies:
    def test_single_step_no_deps(self, model_config):
        plan = PartPlan(
            part_id="p1", task_summary="t",
            steps=[PlanStep(id="s1", description="d")],
            rationale="r",
        )
        llm = _make_mock_llm(model_config, [])
        result = _run_step_dependencies(plan, "desc", llm)
        assert result == {"s1": []}

    def test_two_steps_with_dependency(self, model_config):
        plan = PartPlan(
            part_id="p1", task_summary="t",
            steps=[
                PlanStep(id="s1", description="Create function"),
                PlanStep(id="s2", description="Use function"),
            ],
            rationale="r",
        )
        # (s1,s2): s2 depends on s1 → yes; (s2,s1): s1 depends on s2 → no
        llm = _make_mock_llm(model_config, ["yes", "no"])
        result = _run_step_dependencies(plan, "desc", llm)
        assert result["s1"] == []
        assert result["s2"] == ["s1"]


class TestAssemblePartPlan:
    def test_basic_assembly(self):
        step_plan = PartPlan(
            part_id="p1", task_summary="Fix foo",
            steps=[
                PlanStep(id="s1", description="Step 1", target_files=["a.py"]),
                PlanStep(id="s2", description="Step 2", target_files=["b.py"]),
            ],
            rationale="r",
        )
        dep_edges = {"s1": [], "s2": ["s1"]}
        result = _assemble_part_plan(step_plan, dep_edges)
        assert isinstance(result, PartPlan)
        assert result.steps[0].depends_on == []
        assert result.steps[1].depends_on == ["s1"]

    def test_cycle_raises(self):
        step_plan = PartPlan(
            part_id="p1", task_summary="t",
            steps=[
                PlanStep(id="s1", description="d1"),
                PlanStep(id="s2", description="d2"),
            ],
            rationale="r",
        )
        dep_edges = {"s1": ["s2"], "s2": ["s1"]}
        with pytest.raises(ValueError, match="validation failed"):
            _assemble_part_plan(step_plan, dep_edges)

    def test_missing_dep_edges_key_raises(self):
        """M1: Missing key in dep_edges is an invariant violation."""
        step_plan = PartPlan(
            part_id="p1", task_summary="t",
            steps=[PlanStep(id="s1", description="d1")],
            rationale="r",
        )
        dep_edges = {}  # Missing s1 key
        with pytest.raises(KeyError):
            _assemble_part_plan(step_plan, dep_edges)


# -- Full integration tests --


class TestDecomposedMetaPlan:
    def test_end_to_end(self, context_package, model_config):
        """Full decomposed meta-plan: enum → grouping → binary deps → MetaPlan."""
        # Stage 1: change point enum
        enum_json = json.dumps({
            "task_summary": "Add validation",
            "change_points": [
                {"file_path": "a.py", "symbol": "validate", "change_type": "modify", "rationale": "Fix"},
                {"file_path": "b.py", "symbol": "check", "change_type": "add", "rationale": "New"},
            ],
        })
        # Stage 2: part grouping
        grouping_json = json.dumps({
            "parts": [
                {"id": "p1", "description": "Fix validation", "change_point_indices": [0], "affected_files": ["a.py"]},
                {"id": "p2", "description": "Add check", "change_point_indices": [1], "affected_files": ["b.py"]},
            ],
        })
        # Stage 3: binary deps — 2 pairs: (p1→p2) and (p2→p1)
        # p2 depends on p1, p1 does not depend on p2
        responses = [enum_json, grouping_json, "yes", "no"]
        llm = _make_mock_llm(model_config, responses)

        result = decomposed_meta_plan(context_package, "Add validation", llm)

        assert isinstance(result, MetaPlan)
        assert result.task_summary == "Add validation"
        assert len(result.parts) == 2
        assert result.parts[0].id == "p1"
        assert result.parts[1].depends_on == ["p1"]
        # 4 LLM calls: enum + grouping + 2 binary
        assert llm.complete.call_count == 4


class TestDecomposedPartPlan:
    def test_end_to_end(self, context_package, model_config):
        """Full decomposed part-plan: targeting → design → binary deps → PartPlan."""
        # Stage 1: symbol targeting
        targeting_json = json.dumps({
            "part_id": "p1",
            "targets": [
                {"file_path": "a.py", "symbol": "validate", "action": "modify", "rationale": "Fix"},
                {"file_path": "a.py", "symbol": "check", "action": "add", "rationale": "New helper"},
            ],
        })
        # Stage 2: step design
        design_json = json.dumps({
            "part_id": "p1",
            "task_summary": "Fix validation",
            "steps": [
                {"id": "s1", "description": "Add check helper", "target_files": ["a.py"], "target_symbols": ["check"], "depends_on": []},
                {"id": "s2", "description": "Fix validate", "target_files": ["a.py"], "target_symbols": ["validate"], "depends_on": []},
            ],
            "rationale": "Two steps",
        })
        # Stage 3: binary deps — 2 pairs: s2 depends on s1, s1 not on s2
        responses = [targeting_json, design_json, "yes", "no"]
        llm = _make_mock_llm(model_config, responses)

        result = decomposed_part_plan(
            context_package, "Fix validation", "p1", llm,
        )

        assert isinstance(result, PartPlan)
        assert result.part_id == "p1"
        assert len(result.steps) == 2
        assert result.steps[1].depends_on == ["s1"]
        # 4 LLM calls: targeting + design + 2 binary
        assert llm.complete.call_count == 4

    def test_with_cumulative_diff(self, context_package, model_config):
        """cumulative_diff is passed through to prompts."""
        targeting_json = json.dumps({
            "part_id": "p1",
            "targets": [
                {"file_path": "a.py", "symbol": "foo", "action": "modify", "rationale": "Fix"},
            ],
        })
        design_json = json.dumps({
            "part_id": "p1",
            "task_summary": "Fix foo",
            "steps": [{"id": "s1", "description": "Fix", "target_files": ["a.py"], "target_symbols": ["foo"], "depends_on": []}],
            "rationale": "r",
        })
        responses = [targeting_json, design_json]
        llm = _make_mock_llm(model_config, responses)

        result = decomposed_part_plan(
            context_package, "Fix foo", "p1", llm,
            cumulative_diff="--- a/a.py\n+++ b/a.py",
        )
        assert isinstance(result, PartPlan)
        # Verify cumulative_diff was passed (appears in prompts)
        calls = llm.complete.call_args_list
        assert any("prior_changes" in str(c) for c in calls)


# -- Union-find tests --


class TestUnionFindGroups:
    def test_empty_pairs(self):
        """No pairs -> empty groups."""
        result = _union_find_groups([])
        assert result == {}

    def test_single_pair(self):
        """One pair -> one group of two."""
        result = _union_find_groups([(0, 1)])
        assert result == {0: [0, 1]}

    def test_transitive_closure(self):
        """(0,1) + (1,2) -> one group {0,1,2}."""
        result = _union_find_groups([(0, 1), (1, 2)])
        assert result == {0: [0, 1, 2]}

    def test_two_separate_groups(self):
        """(0,1) + (2,3) -> two groups."""
        result = _union_find_groups([(0, 1), (2, 3)])
        assert 0 in result and 2 in result
        assert result[0] == [0, 1]
        assert result[2] == [2, 3]

    def test_deterministic_root_selection(self):
        """Smaller index always becomes root."""
        result = _union_find_groups([(5, 3), (3, 1)])
        assert 1 in result
        assert sorted(result[1]) == [1, 3, 5]


# -- Description generation tests --


class TestGeneratePartDescription:
    def test_single_file(self):
        cps = [
            ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="Fix bug"),
            ChangePoint(file_path="a.py", symbol="bar", change_type="modify", rationale="Related fix"),
        ]
        desc = _generate_part_description([0, 1], cps)
        assert "a.py" in desc
        assert "foo" in desc
        assert "bar" in desc

    def test_multiple_files(self):
        cps = [
            ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="Fix"),
            ChangePoint(file_path="b.py", symbol="bar", change_type="add", rationale="New"),
        ]
        desc = _generate_part_description([0, 1], cps)
        assert "a.py" in desc
        assert "b.py" in desc

    def test_long_rationale_truncated(self):
        long_rationale = "x" * 200
        cps = [ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale=long_rationale)]
        desc = _generate_part_description([0], cps)
        assert len(desc) < 300  # reasonable bound
        assert "..." in desc


# -- Build grouping from components tests --


class TestBuildGroupingFromComponents:
    def test_single_component(self):
        cps = [
            ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
            ChangePoint(file_path="a.py", symbol="bar", change_type="modify", rationale="r"),
        ]
        groups = {0: [0, 1]}
        result = _build_grouping_from_components(groups, cps)
        assert len(result.parts) == 1
        assert result.parts[0].change_point_indices == [0, 1]
        assert result.parts[0].affected_files == ["a.py"]
        assert result.parts[0].id == "p1"

    def test_two_components(self):
        cps = [
            ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
            ChangePoint(file_path="b.py", symbol="bar", change_type="add", rationale="r"),
        ]
        groups = {0: [0], 1: [1]}
        result = _build_grouping_from_components(groups, cps)
        assert len(result.parts) == 2
        assert result.parts[0].id == "p1"
        assert result.parts[1].id == "p2"

    def test_affected_files_deduplication(self):
        cps = [
            ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
            ChangePoint(file_path="a.py", symbol="bar", change_type="modify", rationale="r"),
            ChangePoint(file_path="b.py", symbol="baz", change_type="add", rationale="r"),
        ]
        groups = {0: [0, 1, 2]}
        result = _build_grouping_from_components(groups, cps)
        assert result.parts[0].affected_files == ["a.py", "b.py"]


# -- Single change point grouping tests --


class TestSingleChangePointGrouping:
    def test_one_change_point(self):
        cps = [ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r")]
        result = _single_change_point_grouping(cps)
        assert len(result.parts) == 1
        assert result.parts[0].change_point_indices == [0]

    def test_zero_change_points_raises(self):
        with pytest.raises(ValueError, match="zero change points"):
            _single_change_point_grouping([])


# -- Binary part grouping tests --


class TestRunPartGroupingBinary:
    def test_two_change_points_same_part(self, model_config):
        """Two change points, classifier says yes -> one part."""
        enum_result = ChangePointEnumeration(
            task_summary="t",
            change_points=[
                ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
                ChangePoint(file_path="a.py", symbol="bar", change_type="modify", rationale="r"),
            ],
        )
        llm = _make_mock_llm(model_config, ["yes"])
        result = _run_part_grouping_binary("task", enum_result, llm)
        assert isinstance(result, PartGrouping)
        assert len(result.parts) == 1
        assert result.parts[0].change_point_indices == [0, 1]

    def test_two_change_points_different_parts(self, model_config):
        """Two change points, classifier says no -> two parts."""
        enum_result = ChangePointEnumeration(
            task_summary="t",
            change_points=[
                ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
                ChangePoint(file_path="b.py", symbol="bar", change_type="add", rationale="r"),
            ],
        )
        llm = _make_mock_llm(model_config, ["no"])
        result = _run_part_grouping_binary("task", enum_result, llm)
        assert isinstance(result, PartGrouping)
        assert len(result.parts) == 2

    def test_three_change_points_transitive(self, model_config):
        """Three CPs: (0,1)=yes, (0,2)=no, (1,2)=yes -> all in one group via transitivity."""
        enum_result = ChangePointEnumeration(
            task_summary="t",
            change_points=[
                ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
                ChangePoint(file_path="a.py", symbol="bar", change_type="modify", rationale="r"),
                ChangePoint(file_path="a.py", symbol="baz", change_type="modify", rationale="r"),
            ],
        )
        # Pairs: (0,1), (0,2), (1,2)
        llm = _make_mock_llm(model_config, ["yes", "no", "yes"])
        result = _run_part_grouping_binary("task", enum_result, llm)
        assert len(result.parts) == 1
        assert sorted(result.parts[0].change_point_indices) == [0, 1, 2]

    def test_single_change_point(self, model_config):
        """One change point -> one part, no LLM calls."""
        enum_result = ChangePointEnumeration(
            task_summary="t",
            change_points=[
                ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
            ],
        )
        llm = _make_mock_llm(model_config, [])
        result = _run_part_grouping_binary("task", enum_result, llm)
        assert len(result.parts) == 1
        assert result.parts[0].change_point_indices == [0]
        assert llm.complete.call_count == 0

    def test_parse_failure_defaults_to_separate(self, model_config):
        """R2: unparseable response defaults to 'no' (separate parts)."""
        enum_result = ChangePointEnumeration(
            task_summary="t",
            change_points=[
                ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
                ChangePoint(file_path="b.py", symbol="bar", change_type="add", rationale="r"),
            ],
        )
        llm = _make_mock_llm(model_config, ["garbage"])
        result = _run_part_grouping_binary("task", enum_result, llm)
        # Unparseable -> treated as no -> two separate parts
        assert len(result.parts) == 2

    def test_validation_passes(self, model_config):
        """Grouping passes validate_part_grouping (all indices covered)."""
        enum_result = ChangePointEnumeration(
            task_summary="t",
            change_points=[
                ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r"),
                ChangePoint(file_path="b.py", symbol="bar", change_type="add", rationale="r"),
                ChangePoint(file_path="c.py", symbol="baz", change_type="modify", rationale="r"),
            ],
        )
        # (0,1)=yes, (0,2)=no, (1,2)=no -> group {0,1} + isolated {2}
        llm = _make_mock_llm(model_config, ["yes", "no", "no"])
        result = _run_part_grouping_binary("task", enum_result, llm)
        assert len(result.parts) == 2
        # All indices are covered
        all_indices = []
        for part in result.parts:
            all_indices.extend(part.change_point_indices)
        assert sorted(all_indices) == [0, 1, 2]


# -- Integration test: full pipeline with binary grouping --


class TestDecomposedMetaPlanBinaryGrouping:
    def test_end_to_end_binary_grouping(self, context_package, model_config):
        """Full decomposed meta-plan with binary grouping."""
        # Stage 1: change point enum
        enum_json = json.dumps({
            "task_summary": "Add validation",
            "change_points": [
                {"file_path": "a.py", "symbol": "validate", "change_type": "modify", "rationale": "Fix"},
                {"file_path": "b.py", "symbol": "check", "change_type": "add", "rationale": "New"},
            ],
        })
        # Stage 2: binary grouping — 1 pair: (0,1) -> no (separate parts)
        # Stage 3: binary deps — 2 pairs: (p1→p2) and (p2→p1)
        # p2 depends on p1, p1 does not depend on p2
        responses = [enum_json, "no", "yes", "no"]
        llm = _make_mock_llm(model_config, responses)

        result = decomposed_meta_plan(
            context_package, "Add validation", llm,
            use_binary_grouping=True,
        )

        assert isinstance(result, MetaPlan)
        assert result.task_summary == "Add validation"
        assert len(result.parts) == 2
        assert result.parts[0].id == "p1"
        assert result.parts[1].depends_on == ["p1"]
        # 4 LLM calls: enum + 1 binary grouping + 2 binary deps
        assert llm.complete.call_count == 4
