"""Tests for Phase 3 response parsers."""

import pytest

from clean_room_agent.execute.dataclasses import (
    ChangePointEnumeration,
    MetaPlan,
    MetaPlanPart,
    PartGroup,
    PartGrouping,
    PartPlan,
    PlanAdjustment,
    PlanStep,
    SymbolTargetEnumeration,
)
from clean_room_agent.execute.parsers import (
    parse_implement_response,
    parse_plan_response,
    validate_part_grouping,
    validate_plan,
)


class TestParsePlanResponseMetaPlan:
    def test_valid_json(self):
        text = '''{
            "task_summary": "Add validation",
            "parts": [
                {"id": "p1", "description": "Add input checks", "affected_files": ["a.py"], "depends_on": []}
            ],
            "rationale": "Simple task"
        }'''
        result = parse_plan_response(text, "meta_plan")
        assert isinstance(result, MetaPlan)
        assert result.task_summary == "Add validation"
        assert len(result.parts) == 1
        assert result.parts[0].id == "p1"

    def test_with_markdown_fencing(self):
        text = '```json\n{"task_summary": "t", "parts": [{"id": "p1", "description": "d"}], "rationale": "r"}\n```'
        result = parse_plan_response(text, "meta_plan")
        assert isinstance(result, MetaPlan)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_plan_response("not json at all", "meta_plan")

    def test_missing_field_raises(self):
        text = '{"task_summary": "t", "rationale": "r"}'
        with pytest.raises(ValueError, match="missing required key"):
            parse_plan_response(text, "meta_plan")

    def test_array_instead_of_object_raises(self):
        with pytest.raises(ValueError, match="Expected JSON object"):
            parse_plan_response("[1, 2, 3]", "meta_plan")


class TestParsePlanResponsePartPlan:
    def test_valid(self):
        text = '''{
            "part_id": "p1",
            "task_summary": "Implement part",
            "steps": [{"id": "s1", "description": "Do thing"}],
            "rationale": "Because"
        }'''
        result = parse_plan_response(text, "part_plan")
        assert isinstance(result, PartPlan)
        assert result.part_id == "p1"
        assert len(result.steps) == 1

    def test_missing_field_raises(self):
        text = '{"part_id": "p1", "rationale": "r"}'
        with pytest.raises(ValueError, match="missing required key"):
            parse_plan_response(text, "part_plan")


class TestParsePlanResponseAdjustment:
    def test_valid(self):
        text = '''{
            "revised_steps": [{"id": "s2", "description": "New step"}],
            "rationale": "Test failed",
            "changes_made": ["Removed s1"]
        }'''
        result = parse_plan_response(text, "adjustment")
        assert isinstance(result, PlanAdjustment)
        assert result.rationale == "Test failed"
        assert result.changes_made == ["Removed s1"]

    def test_empty_revised_steps(self):
        text = '{"revised_steps": [], "rationale": "All done", "changes_made": ["Cleared all"]}'
        result = parse_plan_response(text, "adjustment")
        assert result.revised_steps == []


class TestParsePlanResponseTestPlan:
    def test_valid(self):
        text = '''{
            "part_id": "p1_tests",
            "task_summary": "Test coverage for part p1",
            "steps": [{"id": "t1", "description": "Test hello function", "target_files": ["tests/test_main.py"], "target_symbols": ["hello"]}],
            "rationale": "Cover all new functions"
        }'''
        result = parse_plan_response(text, "test_plan")
        assert isinstance(result, PartPlan)
        assert result.part_id == "p1_tests"
        assert len(result.steps) == 1
        assert result.steps[0].target_symbols == ["hello"]

    def test_missing_field_raises(self):
        text = '{"part_id": "p1_tests", "rationale": "r"}'
        with pytest.raises(ValueError, match="missing required key"):
            parse_plan_response(text, "test_plan")


class TestParsePlanResponseUnknownType:
    def test_unknown_pass_type_raises(self):
        with pytest.raises(ValueError, match="Unknown pass_type"):
            parse_plan_response("{}", "unknown_type")


class TestParseImplementResponse:
    def test_single_edit(self):
        text = '''<edit file="src/main.py">
<search>
def hello():
    pass
</search>
<replacement>
def hello():
    return "world"
</replacement>
</edit>'''
        edits = parse_implement_response(text)
        assert len(edits) == 1
        assert edits[0].file_path == "src/main.py"
        assert "def hello():" in edits[0].search
        assert "return \"world\"" in edits[0].replacement

    def test_multiple_edits(self):
        text = '''<edit file="a.py">
<search>old_a</search>
<replacement>new_a</replacement>
</edit>

<edit file="b.py">
<search>old_b</search>
<replacement>new_b</replacement>
</edit>'''
        edits = parse_implement_response(text)
        assert len(edits) == 2
        assert edits[0].file_path == "a.py"
        assert edits[1].file_path == "b.py"

    def test_empty_replacement(self):
        text = '''<edit file="a.py">
<search>remove_this</search>
<replacement></replacement>
</edit>'''
        edits = parse_implement_response(text)
        assert len(edits) == 1
        assert edits[0].replacement == ""

    def test_no_edits_raises(self):
        with pytest.raises(ValueError, match="No valid <edit> blocks"):
            parse_implement_response("Just some text without edit blocks")

    def test_preserves_order(self):
        text = '''<edit file="c.py">
<search>c</search>
<replacement>c2</replacement>
</edit>
<edit file="a.py">
<search>a</search>
<replacement>a2</replacement>
</edit>
<edit file="b.py">
<search>b</search>
<replacement>b2</replacement>
</edit>'''
        edits = parse_implement_response(text)
        assert [e.file_path for e in edits] == ["c.py", "a.py", "b.py"]

    def test_with_surrounding_text(self):
        text = '''Here are the changes:

<edit file="main.py">
<search>old</search>
<replacement>new</replacement>
</edit>

That should fix the issue.'''
        edits = parse_implement_response(text)
        assert len(edits) == 1
        assert edits[0].file_path == "main.py"

    def test_multiline_content(self):
        text = '''<edit file="test.py">
<search>
def test_foo():
    assert 1 == 2
    assert True
</search>
<replacement>
def test_foo():
    assert 1 == 1
    assert True
</replacement>
</edit>'''
        edits = parse_implement_response(text)
        assert "assert 1 == 2" in edits[0].search
        assert "assert 1 == 1" in edits[0].replacement

    def test_content_with_search_tag(self):
        """T49: Parser handles code containing literal </search> tags."""
        text = '''<edit file="parser.py">
<search>
pattern = re.compile(r'<search>(.*?)</search>')
</search>
<replacement>
pattern = re.compile(r'<search>(.*?)</search>', re.DOTALL)
</replacement>
</edit>'''
        edits = parse_implement_response(text)
        assert len(edits) == 1
        assert "</search>" in edits[0].search
        assert "re.DOTALL" in edits[0].replacement

    def test_content_with_replacement_tag(self):
        """T49: Parser handles code containing literal </replacement> tags."""
        text = '''<edit file="parser.py">
<search>
text = "use </replacement> carefully"
</search>
<replacement>
text = "use </replacement> safely"
</replacement>
</edit>'''
        edits = parse_implement_response(text)
        assert len(edits) == 1
        assert "</replacement>" in edits[0].search

    def test_strips_one_formatting_newline_only(self):
        """T50: Parser strips exactly one leading/trailing formatting newline."""
        text = '''<edit file="a.py">
<search>

blank_line_above_and_below

</search>
<replacement>

still_has_blank_lines

</replacement>
</edit>'''
        edits = parse_implement_response(text)
        # Should strip one formatting newline but preserve the content newlines
        assert edits[0].search.startswith("\nblank_line_above_and_below\n")
        assert edits[0].replacement.startswith("\nstill_has_blank_lines\n")

    def test_handles_crlf_line_endings(self):
        """T54: Parser handles \\r\\n line endings in formatting newlines."""
        text = '<edit file="a.py">\r\n<search>\r\nold code\r\n</search>\r\n<replacement>\r\nnew code\r\n</replacement>\r\n</edit>'
        edits = parse_implement_response(text)
        assert len(edits) == 1
        assert edits[0].search == "old code"
        assert edits[0].replacement == "new code"


class TestValidatePlan:
    def test_valid_meta_plan(self):
        plan = MetaPlan(
            task_summary="t",
            parts=[
                MetaPlanPart(id="p1", description="d1"),
                MetaPlanPart(id="p2", description="d2", depends_on=["p1"]),
            ],
            rationale="r",
        )
        warnings = validate_plan(plan)
        assert warnings == []

    def test_valid_part_plan(self):
        plan = PartPlan(
            part_id="p1", task_summary="t",
            steps=[
                PlanStep(id="s1", description="d1"),
                PlanStep(id="s2", description="d2", depends_on=["s1"]),
            ],
            rationale="r",
        )
        warnings = validate_plan(plan)
        assert warnings == []

    def test_duplicate_ids(self):
        plan = MetaPlan(
            task_summary="t",
            parts=[
                MetaPlanPart(id="p1", description="d1"),
                MetaPlanPart(id="p1", description="d2"),
            ],
            rationale="r",
        )
        warnings = validate_plan(plan)
        assert any("Duplicate" in w for w in warnings)

    def test_unknown_dependency(self):
        plan = MetaPlan(
            task_summary="t",
            parts=[
                MetaPlanPart(id="p1", description="d1", depends_on=["p99"]),
            ],
            rationale="r",
        )
        warnings = validate_plan(plan)
        assert any("unknown ID" in w for w in warnings)

    def test_circular_dependency(self):
        plan = PartPlan(
            part_id="p1", task_summary="t",
            steps=[
                PlanStep(id="s1", description="d1", depends_on=["s2"]),
                PlanStep(id="s2", description="d2", depends_on=["s1"]),
            ],
            rationale="r",
        )
        warnings = validate_plan(plan)
        assert any("Circular" in w for w in warnings)

    def test_self_dependency_cycle(self):
        plan = PartPlan(
            part_id="p1", task_summary="t",
            steps=[
                PlanStep(id="s1", description="d1", depends_on=["s1"]),
            ],
            rationale="r",
        )
        warnings = validate_plan(plan)
        assert any("Circular" in w for w in warnings)

    def test_no_dependencies_valid(self):
        plan = MetaPlan(
            task_summary="t",
            parts=[
                MetaPlanPart(id="p1", description="d1"),
                MetaPlanPart(id="p2", description="d2"),
            ],
            rationale="r",
        )
        warnings = validate_plan(plan)
        assert warnings == []


class TestParsePlanResponseChangePointEnum:
    def test_valid(self):
        text = '''{
            "task_summary": "Fix validation",
            "change_points": [
                {"file_path": "a.py", "symbol": "validate", "change_type": "modify", "rationale": "Fix bug"}
            ]
        }'''
        result = parse_plan_response(text, "change_point_enum")
        assert isinstance(result, ChangePointEnumeration)
        assert result.task_summary == "Fix validation"
        assert len(result.change_points) == 1
        assert result.change_points[0].symbol == "validate"

    def test_missing_field_raises(self):
        text = '{"task_summary": "t"}'
        with pytest.raises(ValueError, match="missing required key"):
            parse_plan_response(text, "change_point_enum")


class TestParsePlanResponsePartGrouping:
    def test_valid(self):
        text = '''{
            "parts": [
                {"id": "p1", "description": "Part 1", "change_point_indices": [0, 1], "affected_files": ["a.py"]}
            ]
        }'''
        result = parse_plan_response(text, "part_grouping")
        assert isinstance(result, PartGrouping)
        assert len(result.parts) == 1
        assert result.parts[0].change_point_indices == [0, 1]

    def test_missing_field_raises(self):
        text = '{}'
        with pytest.raises(ValueError, match="missing required key"):
            parse_plan_response(text, "part_grouping")


class TestParsePlanResponseSymbolTargeting:
    def test_valid(self):
        text = '''{
            "part_id": "p1",
            "targets": [
                {"file_path": "a.py", "symbol": "foo", "action": "modify", "rationale": "Fix"}
            ]
        }'''
        result = parse_plan_response(text, "symbol_targeting")
        assert isinstance(result, SymbolTargetEnumeration)
        assert result.part_id == "p1"
        assert len(result.targets) == 1

    def test_missing_field_raises(self):
        text = '{"part_id": "p1"}'
        with pytest.raises(ValueError, match="missing required key"):
            parse_plan_response(text, "symbol_targeting")


class TestParsePlanResponseStepDesign:
    def test_valid(self):
        text = '''{
            "part_id": "p1",
            "task_summary": "Implement part",
            "steps": [{"id": "s1", "description": "Do thing"}],
            "rationale": "Because"
        }'''
        result = parse_plan_response(text, "step_design")
        assert isinstance(result, PartPlan)
        assert result.part_id == "p1"


class TestValidatePartGrouping:
    def test_valid_full_coverage(self):
        grouping = PartGrouping(parts=[
            PartGroup(id="p1", description="d1", change_point_indices=[0, 1]),
            PartGroup(id="p2", description="d2", change_point_indices=[2]),
        ])
        warnings = validate_part_grouping(grouping, 3)
        assert warnings == []

    def test_missing_index(self):
        grouping = PartGrouping(parts=[
            PartGroup(id="p1", description="d1", change_point_indices=[0]),
        ])
        warnings = validate_part_grouping(grouping, 3)
        assert any("Unassigned" in w for w in warnings)

    def test_duplicate_index(self):
        grouping = PartGrouping(parts=[
            PartGroup(id="p1", description="d1", change_point_indices=[0]),
            PartGroup(id="p2", description="d2", change_point_indices=[0, 1]),
        ])
        warnings = validate_part_grouping(grouping, 2)
        assert any("duplicate" in w.lower() or "multiple" in w.lower() for w in warnings)

    def test_out_of_range_index(self):
        grouping = PartGrouping(parts=[
            PartGroup(id="p1", description="d1", change_point_indices=[0, 5]),
        ])
        warnings = validate_part_grouping(grouping, 2)
        assert any("out-of-range" in w for w in warnings)

    def test_duplicate_part_ids(self):
        grouping = PartGrouping(parts=[
            PartGroup(id="p1", description="d1", change_point_indices=[0]),
            PartGroup(id="p1", description="d2", change_point_indices=[1]),
        ])
        warnings = validate_part_grouping(grouping, 2)
        assert any("Duplicate part ID" in w for w in warnings)

    def test_single_part_all_indices(self):
        grouping = PartGrouping(parts=[
            PartGroup(id="p1", description="d1", change_point_indices=[0, 1, 2]),
        ])
        warnings = validate_part_grouping(grouping, 3)
        assert warnings == []
