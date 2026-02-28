"""Tests for decomposed adjustment (binary sub-task plan revision)."""

import json
from unittest.mock import MagicMock

import pytest

from clean_room_agent.execute.dataclasses import (
    AdjustmentVerdicts,
    FailureSignal,
    PlanAdjustment,
    PlanStep,
    StepResult,
)
from clean_room_agent.execute.decomposed_adjustment import (
    FAILURE_CATEGORY_COMPILE,
    FAILURE_CATEGORY_PATCH,
    FAILURE_CATEGORY_TEST,
    FAILURE_CATEGORY_UNKNOWN,
    _build_finalize_prompt,
    _finalize_adjustment,
    _format_failures_summary,
    _format_step_for_viability,
    _run_new_step_detection,
    _run_root_cause_attribution,
    _run_step_viability,
    decomposed_adjustment,
    extract_failure_signals,
)
from clean_room_agent.execute.prompts import SYSTEM_PROMPTS
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
        raw_task="Fix hash table",
        task_id="test-adj-001",
        mode="plan",
        repo_id=1,
    )
    return ContextPackage(
        task=task,
        files=[
            FileContent(
                file_id=1, path="src/hash_table.c", language="c",
                content="int hash_init() { return 0; }", token_estimate=10,
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


def _make_steps(*ids_and_descs):
    """Helper to create PlanStep objects from (id, description) pairs."""
    return [
        PlanStep(id=sid, description=desc, target_files=["hash_table.c"])
        for sid, desc in ids_and_descs
    ]


# ---------------------------------------------------------------------------
# Tests for extract_failure_signals()
# ---------------------------------------------------------------------------


class TestExtractFailureSignals:
    def test_extract_none_prior_results_returns_empty(self):
        assert extract_failure_signals(None, 32768, 4096) == []

    def test_extract_no_failures_returns_empty(self):
        results = [StepResult(success=True)]
        assert extract_failure_signals(results, 32768, 4096) == []

    def test_extract_success_result_returns_empty(self):
        results = [StepResult(success=True, error_info=None)]
        assert extract_failure_signals(results, 32768, 4096) == []

    def test_extract_compile_error_categorization(self):
        results = [StepResult(
            success=False,
            error_info="error: implicit declaration of function 'hash_init'",
        )]
        signals = extract_failure_signals(results, 32768, 4096)
        assert len(signals) == 1
        assert signals[0].category == FAILURE_CATEGORY_COMPILE
        assert signals[0].source == "error_info"

    def test_extract_compile_error_gcc(self):
        results = [StepResult(
            success=False,
            error_info="gcc: error: no input files",
        )]
        signals = extract_failure_signals(results, 32768, 4096)
        assert len(signals) == 1
        assert signals[0].category == FAILURE_CATEGORY_COMPILE

    def test_extract_test_failure_categorization(self):
        results = [StepResult(
            success=False,
            error_info="test_insert FAILED: assertion hash_size == 3 failed",
        )]
        signals = extract_failure_signals(results, 32768, 4096)
        assert len(signals) == 1
        assert signals[0].category == FAILURE_CATEGORY_TEST

    def test_extract_patch_failure_categorization(self):
        results = [StepResult(
            success=False,
            error_info="could not find search text in hash_table.c",
        )]
        signals = extract_failure_signals(results, 32768, 4096)
        assert len(signals) == 1
        assert signals[0].category == FAILURE_CATEGORY_PATCH

    def test_extract_unknown_failure_categorization(self):
        results = [StepResult(
            success=False,
            error_info=None,
        )]
        signals = extract_failure_signals(results, 32768, 4096)
        assert len(signals) == 1
        assert signals[0].category == FAILURE_CATEGORY_UNKNOWN
        assert signals[0].source == "step_failed"

    def test_extract_unknown_with_unrecognized_message(self):
        results = [StepResult(
            success=False,
            error_info="some completely novel error type",
        )]
        signals = extract_failure_signals(results, 32768, 4096)
        assert len(signals) == 1
        assert signals[0].category == FAILURE_CATEGORY_UNKNOWN

    def test_extract_multiple_failures(self):
        results = [
            StepResult(success=False, error_info="gcc: error: redefinition of 'foo'"),
            StepResult(success=True),
            StepResult(success=False, error_info="test_bar FAILED"),
        ]
        signals = extract_failure_signals(results, 32768, 4096)
        assert len(signals) == 2
        assert signals[0].category == FAILURE_CATEGORY_COMPILE
        assert signals[1].category == FAILURE_CATEGORY_TEST

    def test_extract_truncates_long_messages(self):
        """Budget-aware truncation: fraction=0.2, context=1000, max_tokens=100.
        Budget = (1000 - 100) * 3 * 0.2 = 540 chars. 1000-char msg gets truncated.
        """
        long_msg = "x" * 1000
        results = [StepResult(success=False, error_info=long_msg)]
        signals = extract_failure_signals(results, 1000, 100)
        expected_len = int((1000 - 100) * 3 * 0.2)
        assert len(signals[0].message) == expected_len
        assert expected_len < 1000

    def test_extract_no_truncation_when_budget_sufficient(self):
        """With large context window, messages pass through untouched."""
        msg = "error: foo bar" * 10
        results = [StepResult(success=False, error_info=msg)]
        signals = extract_failure_signals(results, 32768, 4096)
        assert signals[0].message == msg


# ---------------------------------------------------------------------------
# Tests for binary step viability (stage 2)
# ---------------------------------------------------------------------------


class TestStepViability:
    def test_step_viability_all_valid(self, model_config):
        steps = _make_steps(("s1", "Init hash"), ("s2", "Add resize"))
        failures = [FailureSignal("compile_error", "error: foo", "error_info")]
        # Both steps answered "yes"
        llm = _make_mock_llm(model_config, ["yes", "yes"])
        result = _run_step_viability(failures, steps, "fix hash", llm)
        assert result == {"s1": True, "s2": True}

    def test_step_viability_some_invalid(self, model_config):
        steps = _make_steps(("s1", "Init hash"), ("s2", "Add resize"))
        failures = [FailureSignal("compile_error", "error: foo", "error_info")]
        llm = _make_mock_llm(model_config, ["yes", "no"])
        result = _run_step_viability(failures, steps, "fix hash", llm)
        assert result == {"s1": True, "s2": False}

    def test_step_viability_all_invalid_raises(self, model_config):
        """All parse failures -> ValueError (fail-fast)."""
        steps = _make_steps(("s1", "Init hash"), ("s2", "Add resize"))
        failures = [FailureSignal("compile_error", "error: foo", "error_info")]
        # Both return unparseable garbage
        llm = _make_mock_llm(model_config, ["maybe", "dunno"])
        with pytest.raises(ValueError, match="All.*step viability judgments failed"):
            _run_step_viability(failures, steps, "fix hash", llm)

    def test_step_viability_empty_steps_returns_empty(self, model_config):
        failures = [FailureSignal("compile_error", "error", "error_info")]
        llm = _make_mock_llm(model_config, [])
        result = _run_step_viability(failures, [], "fix hash", llm)
        assert result == {}

    def test_step_viability_no_failures_all_valid(self, model_config):
        steps = _make_steps(("s1", "Init hash"), ("s2", "Add resize"))
        llm = _make_mock_llm(model_config, [])  # no calls expected
        result = _run_step_viability([], steps, "fix hash", llm)
        assert result == {"s1": True, "s2": True}
        llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for root cause attribution (stage 3)
# ---------------------------------------------------------------------------


class TestRootCauseAttribution:
    def test_root_cause_single_failure_single_step(self, model_config):
        failures = [FailureSignal("compile_error", "undefined reference to hash_init", "error_info")]
        steps = _make_steps(("s1", "Implement hash init"))
        viability = {"s1": True}
        llm = _make_mock_llm(model_config, ["yes"])
        result = _run_root_cause_attribution(
            failures, steps, viability, None, "fix hash", llm,
        )
        assert result == {"s1": [0]}

    def test_root_cause_multi_failure_multi_step(self, model_config):
        failures = [
            FailureSignal("compile_error", "error A", "error_info"),
            FailureSignal("test_failure", "test B FAILED", "error_info"),
        ]
        steps = _make_steps(("s1", "Step 1"), ("s2", "Step 2"))
        viability = {"s1": True, "s2": True}
        # 4 pairs: (f0,s1)=yes, (f0,s2)=no, (f1,s1)=no, (f1,s2)=yes
        llm = _make_mock_llm(model_config, ["yes", "no", "no", "yes"])
        result = _run_root_cause_attribution(
            failures, steps, viability, "diff content", "fix", llm,
        )
        assert result == {"s1": [0], "s2": [1]}

    def test_root_cause_no_attribution(self, model_config):
        failures = [FailureSignal("unknown", "mystery error", "error_info")]
        steps = _make_steps(("s1", "Step 1"))
        viability = {"s1": True}
        llm = _make_mock_llm(model_config, ["no"])
        result = _run_root_cause_attribution(
            failures, steps, viability, None, "fix", llm,
        )
        assert result == {}

    def test_root_cause_empty_failures_returns_empty(self, model_config):
        steps = _make_steps(("s1", "Step 1"))
        viability = {"s1": True}
        llm = _make_mock_llm(model_config, [])
        result = _run_root_cause_attribution(
            [], steps, viability, None, "fix", llm,
        )
        assert result == {}
        llm.complete.assert_not_called()

    def test_root_cause_skips_invalid_steps(self, model_config):
        failures = [FailureSignal("compile_error", "error", "error_info")]
        steps = _make_steps(("s1", "Step 1"), ("s2", "Step 2"))
        viability = {"s1": True, "s2": False}  # s2 invalid
        # Only 1 pair evaluated: (f0, s1)
        llm = _make_mock_llm(model_config, ["yes"])
        result = _run_root_cause_attribution(
            failures, steps, viability, None, "fix", llm,
        )
        assert result == {"s1": [0]}
        assert llm.complete.call_count == 1


# ---------------------------------------------------------------------------
# Tests for new step detection (stage 4)
# ---------------------------------------------------------------------------


class TestNewStepDetection:
    def test_new_step_needed(self, model_config):
        failures = [
            (0, FailureSignal("test_failure", "test_lookup FAILED", "error_info")),
        ]
        steps = _make_steps(("s3", "Add resize"), ("s4", "Add delete"))
        llm = _make_mock_llm(model_config, ["yes"])
        result = _run_new_step_detection(failures, steps, "fix hash", llm)
        assert result == [0]

    def test_new_step_not_needed(self, model_config):
        failures = [
            (0, FailureSignal("test_failure", "transient error", "error_info")),
        ]
        steps = _make_steps(("s3", "Add resize"))
        llm = _make_mock_llm(model_config, ["no"])
        result = _run_new_step_detection(failures, steps, "fix hash", llm)
        assert result == []

    def test_new_step_skipped_when_all_attributed(self, model_config):
        """Empty unattributed list -> no calls, empty result."""
        llm = _make_mock_llm(model_config, [])
        result = _run_new_step_detection([], _make_steps(), "fix hash", llm)
        assert result == []
        llm.complete.assert_not_called()

    def test_new_step_empty_unattributed_returns_empty(self, model_config):
        llm = _make_mock_llm(model_config, [])
        result = _run_new_step_detection([], _make_steps(("s1", "A")), "fix", llm)
        assert result == []


# ---------------------------------------------------------------------------
# Tests for _finalize_adjustment() (stage 5)
# ---------------------------------------------------------------------------


class TestFinalizeAdjustment:
    def _make_verdicts(self, steps, viability=None, root_causes=None, new_steps=None, signals=None):
        if viability is None:
            viability = {s.id: True for s in steps}
        return AdjustmentVerdicts(
            step_viability=viability,
            root_causes=root_causes or {},
            new_steps_needed=new_steps or [],
            failure_signals=signals or [],
        )

    def test_finalize_produces_valid_plan_adjustment(self, context_package, model_config):
        steps = _make_steps(("s1", "Init hash"), ("s2", "Add resize"))
        verdicts = self._make_verdicts(steps)
        response_json = json.dumps({
            "revised_steps": [
                {"id": "s1", "description": "Init hash", "target_files": [], "target_symbols": [], "depends_on": []},
                {"id": "s2", "description": "Add resize", "target_files": [], "target_symbols": [], "depends_on": ["s1"]},
            ],
            "rationale": "Steps unchanged",
            "changes_made": [],
        })
        llm = _make_mock_llm(model_config, [response_json])
        result = _finalize_adjustment(
            context_package, "fix hash", verdicts, steps, None, llm,
        )
        assert isinstance(result, PlanAdjustment)
        assert len(result.revised_steps) == 2

    def test_finalize_validates_revised_steps_no_cycles(self, context_package, model_config):
        steps = _make_steps(("s1", "A"), ("s2", "B"))
        verdicts = self._make_verdicts(steps)
        # Create cycle: s1->s2, s2->s1
        response_json = json.dumps({
            "revised_steps": [
                {"id": "s1", "description": "A", "target_files": [], "target_symbols": [], "depends_on": ["s2"]},
                {"id": "s2", "description": "B", "target_files": [], "target_symbols": [], "depends_on": ["s1"]},
            ],
            "rationale": "Bad plan",
            "changes_made": ["cycle"],
        })
        llm = _make_mock_llm(model_config, [response_json])
        with pytest.raises(ValueError, match="Circular dependency"):
            _finalize_adjustment(context_package, "fix", verdicts, steps, None, llm)

    def test_finalize_validates_revised_steps_no_duplicates(self, context_package, model_config):
        steps = _make_steps(("s1", "A"))
        verdicts = self._make_verdicts(steps)
        response_json = json.dumps({
            "revised_steps": [
                {"id": "s1", "description": "A", "target_files": [], "target_symbols": [], "depends_on": []},
                {"id": "s1", "description": "Duplicate", "target_files": [], "target_symbols": [], "depends_on": []},
            ],
            "rationale": "Bad plan",
            "changes_made": ["dup"],
        })
        llm = _make_mock_llm(model_config, [response_json])
        with pytest.raises(ValueError, match="Duplicate"):
            _finalize_adjustment(context_package, "fix", verdicts, steps, None, llm)

    def test_finalize_empty_revised_steps_ok(self, context_package, model_config):
        steps = _make_steps(("s1", "A"))
        verdicts = self._make_verdicts(steps, viability={"s1": False})
        response_json = json.dumps({
            "revised_steps": [],
            "rationale": "All steps dropped",
            "changes_made": ["dropped s1"],
        })
        llm = _make_mock_llm(model_config, [response_json])
        result = _finalize_adjustment(
            context_package, "fix", verdicts, steps, None, llm,
        )
        assert isinstance(result, PlanAdjustment)
        assert result.revised_steps == []

    def test_finalize_budget_validation(self, model_config):
        """Prompt that exceeds budget raises ValueError."""
        # Tiny context window to trigger R3
        tiny_config = ModelConfig(
            model="test-model",
            base_url="http://localhost:11434",
            context_window=100,  # impossibly small
            max_tokens=50,
        )
        task = TaskQuery(raw_task="Fix hash", task_id="t", mode="plan", repo_id=1)
        ctx = ContextPackage(
            task=task,
            files=[FileContent(
                file_id=1, path="a.c", language="c",
                content="x" * 500, token_estimate=200,
                detail_level="primary",
            )],
            total_token_estimate=200,
            budget=BudgetConfig(context_window=100, reserved_tokens=50),
        )
        steps = _make_steps(("s1", "A"))
        verdicts = AdjustmentVerdicts(
            step_viability={"s1": True}, root_causes={},
            new_steps_needed=[], failure_signals=[],
        )
        llm = _make_mock_llm(tiny_config, [])
        with pytest.raises(ValueError, match="R3"):
            _finalize_adjustment(ctx, "fix", verdicts, steps, None, llm)


# ---------------------------------------------------------------------------
# Tests for _build_finalize_prompt()
# ---------------------------------------------------------------------------


class TestBuildFinalizePrompt:
    def test_build_adjustment_finalize_prompt_includes_verdicts(self, context_package):
        steps = _make_steps(("s1", "Init hash"), ("s2", "Add resize"))
        signals = [FailureSignal("compile_error", "error: foo", "error_info")]
        verdicts = AdjustmentVerdicts(
            step_viability={"s1": True, "s2": False},
            root_causes={"s1": [0]},
            new_steps_needed=[],
            failure_signals=signals,
        )
        prompt = _build_finalize_prompt(
            context_package, "fix hash", verdicts, steps, "diff content",
        )
        assert "VALID" in prompt
        assert "INVALID" in prompt
        assert "s1 caused failure [0]" in prompt
        assert "<prior_changes>" in prompt
        assert "<remaining_steps>" in prompt

    def test_build_prompt_no_root_causes(self, context_package):
        steps = _make_steps(("s1", "Init"))
        verdicts = AdjustmentVerdicts(
            step_viability={"s1": True}, root_causes={},
            new_steps_needed=[], failure_signals=[],
        )
        prompt = _build_finalize_prompt(
            context_package, "fix", verdicts, steps, None,
        )
        assert "(none)" in prompt
        assert "<prior_changes>" not in prompt

    def test_build_prompt_with_new_steps_needed(self, context_package):
        steps = _make_steps(("s1", "Init"))
        signals = [FailureSignal("test_failure", "test FAILED", "error_info")]
        verdicts = AdjustmentVerdicts(
            step_viability={"s1": True}, root_causes={},
            new_steps_needed=[0], failure_signals=signals,
        )
        prompt = _build_finalize_prompt(
            context_package, "fix", verdicts, steps, None,
        )
        assert "needs new step" in prompt

    def test_build_prompt_invalid_failure_index_raises(self, context_package):
        """H4: Invalid failure index in root_causes raises IndexError."""
        steps = _make_steps(("s1", "Init"))
        signals = [FailureSignal("compile_error", "error: foo", "error_info")]
        verdicts = AdjustmentVerdicts(
            step_viability={"s1": True},
            root_causes={"s1": [999]},  # invalid index
            new_steps_needed=[], failure_signals=signals,
        )
        with pytest.raises(IndexError):
            _build_finalize_prompt(
                context_package, "fix", verdicts, steps, None,
            )

    def test_build_prompt_invalid_new_step_index_raises(self, context_package):
        """H4: Invalid failure index in new_steps_needed raises IndexError."""
        steps = _make_steps(("s1", "Init"))
        signals = [FailureSignal("compile_error", "error: foo", "error_info")]
        verdicts = AdjustmentVerdicts(
            step_viability={"s1": True}, root_causes={},
            new_steps_needed=[999], failure_signals=signals,
        )
        with pytest.raises(IndexError):
            _build_finalize_prompt(
                context_package, "fix", verdicts, steps, None,
            )


# ---------------------------------------------------------------------------
# Integration tests for decomposed_adjustment()
# ---------------------------------------------------------------------------


class TestDecomposedAdjustment:
    def test_decomposed_adjustment_no_failures_returns_unchanged(self, context_package, model_config):
        steps = _make_steps(("s1", "Init"), ("s2", "Resize"))
        # No failures -> no LLM calls needed
        llm = _make_mock_llm(model_config, [])
        result = decomposed_adjustment(
            context_package, "fix hash", llm,
            prior_results=[StepResult(success=True)],
            remaining_steps=steps,
        )
        assert isinstance(result, PlanAdjustment)
        assert len(result.revised_steps) == 2
        assert result.rationale == "No failures detected -- steps unchanged"
        llm.complete.assert_not_called()

    def test_decomposed_adjustment_no_remaining_steps(self, context_package, model_config):
        llm = _make_mock_llm(model_config, [])
        result = decomposed_adjustment(
            context_package, "fix hash", llm,
            prior_results=[StepResult(success=False, error_info="fail")],
            remaining_steps=[],
        )
        assert isinstance(result, PlanAdjustment)
        assert result.revised_steps == []
        assert result.rationale == "No remaining steps to adjust"

    def test_decomposed_adjustment_full_flow(self, context_package, model_config):
        """Full flow: 1 failure, 2 steps, one attributed, finalize."""
        steps = _make_steps(("s1", "Init hash"), ("s2", "Add resize"))
        prior = [StepResult(
            success=False,
            error_info="error: implicit declaration of 'hash_init'",
        )]
        # Stage 2: viability — s1=yes, s2=yes (2 calls)
        # Stage 3: root cause — (f0,s1)=yes, (f0,s2)=no (2 calls)
        # Stage 4: skipped (all failures attributed)
        # Stage 5: finalize (1 call)
        finalize_response = json.dumps({
            "revised_steps": [
                {"id": "s1", "description": "Fix hash_init declaration", "target_files": ["hash_table.c"],
                 "target_symbols": ["hash_init"], "depends_on": []},
                {"id": "s2", "description": "Add resize", "target_files": ["hash_table.c"],
                 "target_symbols": ["hash_resize"], "depends_on": ["s1"]},
            ],
            "rationale": "Revised s1 to fix declaration issue",
            "changes_made": ["Revised s1 description to address implicit declaration"],
        })
        llm = _make_mock_llm(model_config, [
            "yes", "yes",       # viability
            "yes", "no",        # root cause
            finalize_response,  # finalize
        ])
        result = decomposed_adjustment(
            context_package, "fix hash", llm,
            prior_results=prior,
            remaining_steps=steps,
        )
        assert isinstance(result, PlanAdjustment)
        assert len(result.revised_steps) == 2
        assert result.revised_steps[0].description == "Fix hash_init declaration"

    def test_decomposed_adjustment_all_steps_invalid(self, context_package, model_config):
        """All steps judged invalid -> finalize gets all INVALID verdicts."""
        steps = _make_steps(("s1", "Init"), ("s2", "Resize"))
        prior = [StepResult(success=False, error_info="gcc: error: total failure")]
        # viability: both "no" -> but not parse failures, so not a crash
        finalize_response = json.dumps({
            "revised_steps": [],
            "rationale": "All steps dropped due to fundamental errors",
            "changes_made": ["Dropped s1", "Dropped s2"],
        })
        llm = _make_mock_llm(model_config, [
            "no", "no",          # viability (both invalid)
            # root cause: no viable steps -> 0 calls
            # new step: 1 unattributed failure, 1 call
            "yes",               # needs new step
            finalize_response,   # finalize
        ])
        result = decomposed_adjustment(
            context_package, "fix", llm,
            prior_results=prior,
            remaining_steps=steps,
        )
        assert isinstance(result, PlanAdjustment)
        assert result.revised_steps == []

    def test_decomposed_adjustment_with_new_steps(self, context_package, model_config):
        """Unattributed failure triggers new step detection."""
        steps = _make_steps(("s1", "Init hash"))
        prior = [StepResult(success=False, error_info="test_lookup FAILED: key not found")]
        # viability: s1=yes
        # root cause: (f0,s1)=no (not attributed)
        # new step: f0=yes (needs new step)
        finalize_response = json.dumps({
            "revised_steps": [
                {"id": "s1", "description": "Init hash", "target_files": ["hash_table.c"],
                 "target_symbols": ["hash_init"], "depends_on": []},
                {"id": "s_new1", "description": "Add lookup function", "target_files": ["hash_table.c"],
                 "target_symbols": ["hash_lookup"], "depends_on": ["s1"]},
            ],
            "rationale": "Added new step for lookup",
            "changes_made": ["Added s_new1 for unattributed test failure"],
        })
        llm = _make_mock_llm(model_config, [
            "yes",               # viability s1
            "no",                # root cause (f0,s1)
            "yes",               # new step for f0
            finalize_response,   # finalize
        ])
        result = decomposed_adjustment(
            context_package, "fix", llm,
            prior_results=prior,
            remaining_steps=steps,
        )
        assert isinstance(result, PlanAdjustment)
        assert len(result.revised_steps) == 2
        assert result.revised_steps[1].id == "s_new1"

    def test_decomposed_adjustment_matches_plan_adjustment_type(self, context_package, model_config):
        steps = _make_steps(("s1", "Init"))
        llm = _make_mock_llm(model_config, [])
        result = decomposed_adjustment(
            context_package, "fix", llm,
            prior_results=None,
            remaining_steps=steps,
        )
        assert isinstance(result, PlanAdjustment)

    def test_decomposed_adjustment_none_prior_results(self, context_package, model_config):
        """None prior_results -> no failures -> steps unchanged."""
        steps = _make_steps(("s1", "Init"))
        llm = _make_mock_llm(model_config, [])
        result = decomposed_adjustment(
            context_package, "fix", llm,
            prior_results=None,
            remaining_steps=steps,
        )
        assert result.rationale == "No failures detected -- steps unchanged"
        assert len(result.revised_steps) == 1


# ---------------------------------------------------------------------------
# Tests for config toggle in runner
# ---------------------------------------------------------------------------


class TestConfigToggle:
    def test_use_decomposed_adjustment_default_false(self):
        from clean_room_agent.orchestrator.runner import _use_decomposed_adjustment
        assert _use_decomposed_adjustment({}) is False

    def test_use_decomposed_adjustment_enabled(self):
        from clean_room_agent.orchestrator.runner import _use_decomposed_adjustment
        config = {"orchestrator": {"decomposed_adjustment": True}}
        assert _use_decomposed_adjustment(config) is True

    def test_use_decomposed_adjustment_disabled_explicit(self):
        from clean_room_agent.orchestrator.runner import _use_decomposed_adjustment
        config = {"orchestrator": {"decomposed_adjustment": False}}
        assert _use_decomposed_adjustment(config) is False


# ---------------------------------------------------------------------------
# Tests for new system prompts
# ---------------------------------------------------------------------------


class TestAdjustmentPrompts:
    def test_adjustment_prompts_in_system_prompts_dict(self):
        expected_keys = [
            "adjustment_step_viability",
            "adjustment_root_cause",
            "adjustment_new_step",
            "adjustment_finalize",
        ]
        for key in expected_keys:
            assert key in SYSTEM_PROMPTS, f"Missing system prompt: {key}"
            assert len(SYSTEM_PROMPTS[key]) > 0

    def test_binary_prompts_end_with_yes_or_no(self):
        for key in ("adjustment_step_viability", "adjustment_root_cause", "adjustment_new_step"):
            prompt = SYSTEM_PROMPTS[key]
            assert "yes" in prompt.lower() and "no" in prompt.lower(), (
                f"Binary prompt {key} should mention yes/no"
            )

    def test_finalize_prompt_mentions_json(self):
        prompt = SYSTEM_PROMPTS["adjustment_finalize"]
        assert "JSON" in prompt


# ---------------------------------------------------------------------------
# Tests for parser alias
# ---------------------------------------------------------------------------


class TestParserAlias:
    def test_adjustment_finalize_alias(self):
        from clean_room_agent.execute.parsers import parse_plan_response
        data = json.dumps({
            "revised_steps": [
                {"id": "s1", "description": "A", "target_files": [], "target_symbols": [], "depends_on": []},
            ],
            "rationale": "reason",
            "changes_made": ["change"],
        })
        result = parse_plan_response(data, "adjustment_finalize")
        assert isinstance(result, PlanAdjustment)
        assert len(result.revised_steps) == 1


# ---------------------------------------------------------------------------
# Tests for dataclass serialization
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_failure_signal_roundtrip(self):
        fs = FailureSignal(category="compile_error", message="error: foo", source="error_info")
        d = fs.to_dict()
        fs2 = FailureSignal.from_dict(d)
        assert fs2.category == "compile_error"
        assert fs2.message == "error: foo"
        assert fs2.source == "error_info"

    def test_failure_signal_non_empty_validation(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            FailureSignal(category="", message="error", source="error_info")

    def test_adjustment_verdicts_roundtrip(self):
        fs = FailureSignal(category="compile_error", message="err", source="error_info")
        av = AdjustmentVerdicts(
            step_viability={"s1": True, "s2": False},
            root_causes={"s1": [0]},
            new_steps_needed=[],
            failure_signals=[fs],
        )
        d = av.to_dict()
        av2 = AdjustmentVerdicts.from_dict(d)
        assert av2.step_viability == {"s1": True, "s2": False}
        assert av2.root_causes == {"s1": [0]}
        assert len(av2.failure_signals) == 1
        assert av2.failure_signals[0].category == "compile_error"

    def test_adjustment_verdicts_from_dict_missing_required(self):
        with pytest.raises(ValueError, match="missing required key"):
            AdjustmentVerdicts.from_dict({"step_viability": {}})


# ---------------------------------------------------------------------------
# Tests for formatting helpers
# ---------------------------------------------------------------------------


class TestFormattingHelpers:
    def test_format_failures_summary(self):
        signals = [
            FailureSignal("compile_error", "gcc: error: undefined", "error_info"),
            FailureSignal("test_failure", "test_foo FAILED", "error_info"),
        ]
        summary = _format_failures_summary(signals)
        assert "[compile_error]" in summary
        assert "[test_failure]" in summary
        assert "Failures observed:" in summary

    def test_format_step_for_viability(self):
        step = PlanStep(
            id="s3", description="Add hash resize",
            target_files=["hash_table.c"], target_symbols=["hash_resize"],
        )
        text = _format_step_for_viability(step)
        assert "ID: s3" in text
        assert "Add hash resize" in text
        assert "hash_table.c" in text
        assert "hash_resize" in text
        assert "yes or no" in text
