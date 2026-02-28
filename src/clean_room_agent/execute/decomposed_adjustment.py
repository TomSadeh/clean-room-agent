"""Decomposed adjustment: multi-stage plan revision after step failures.

Replaces single-call execute_plan("adjustment") with smaller cognitive tasks:
1. Deterministic failure extraction (no LLM)
2. Binary step viability per remaining step
3. Binary root cause attribution per (failure, step) pair
4. Binary new step detection per unattributed failure
5. Focused 1.7B finalize with all verdicts

Output type (PlanAdjustment) is unchanged. The orchestrator sees
no difference. This is a decomposition of the internal adjustment process.
"""

from __future__ import annotations

import logging
import re

from clean_room_agent.execute.dataclasses import (
    AdjustmentVerdicts,
    FailureSignal,
    PlanAdjustment,
    PlanStep,
    StepResult,
)
from clean_room_agent.execute.parsers import parse_plan_response, validate_plan
from clean_room_agent.execute.prompts import SYSTEM_PROMPTS
from clean_room_agent.llm.client import LoggedLLMClient
from clean_room_agent.retrieval.batch_judgment import run_binary_judgment
from clean_room_agent.retrieval.dataclasses import ContextPackage
from clean_room_agent.token_estimation import budget_truncate, validate_prompt_budget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: Deterministic failure extraction
# ---------------------------------------------------------------------------

FAILURE_CATEGORY_COMPILE = "compile_error"
FAILURE_CATEGORY_TEST = "test_failure"
FAILURE_CATEGORY_PATCH = "patch_failure"
FAILURE_CATEGORY_RUNTIME = "runtime_error"
FAILURE_CATEGORY_UNKNOWN = "unknown"

_COMPILE_PATTERNS = [
    re.compile(r"error:.*expected|undefined reference|implicit declaration", re.I),
    re.compile(r"gcc|clang|cc1|ld:.*error", re.I),
    re.compile(r"syntax error|parse error|redefinition", re.I),
]
_TEST_PATTERNS = [
    re.compile(r"FAIL|FAILED|AssertionError|assert.*failed", re.I),
    re.compile(r"test_\w+.*FAILED|pytest.*failed", re.I),
]
_PATCH_PATTERNS = [
    re.compile(r"Patch.*failed|search.*not found|could not find", re.I),
]

def _classify_failure(text: str) -> str:
    """Classify failure text into a category using regex patterns.

    Priority: compile > test > patch > runtime > unknown.
    """
    for pat in _COMPILE_PATTERNS:
        if pat.search(text):
            return FAILURE_CATEGORY_COMPILE
    for pat in _TEST_PATTERNS:
        if pat.search(text):
            return FAILURE_CATEGORY_TEST
    for pat in _PATCH_PATTERNS:
        if pat.search(text):
            return FAILURE_CATEGORY_PATCH
    # R2: log when default fires — no pattern matched
    logger.warning(
        "Failure text classified as unknown (no regex matched): %.200s",
        text,
    )
    return FAILURE_CATEGORY_UNKNOWN


def extract_failure_signals(
    prior_results: list[StepResult] | None,
    context_window: int,
    max_tokens: int,
) -> list[FailureSignal]:
    """Deterministically extract and categorize failure messages from step results.

    No LLM calls. Uses regex patterns against error_info and raw_response
    to classify failures into categories. Truncates messages via budget_truncate
    so downstream prompts stay within budget.

    Returns empty list if prior_results is None or all steps succeeded.
    """
    if not prior_results:
        return []

    signals: list[FailureSignal] = []
    for sr in prior_results:
        if sr.success and sr.error_info is None:
            continue

        if not sr.success and sr.error_info:
            msg = budget_truncate(
                sr.error_info, context_window, max_tokens,
                max_content_fraction=0.2, stage_name="failure_signal_extraction",
            )
            category = _classify_failure(sr.error_info)
            signals.append(FailureSignal(
                category=category, message=msg, source="error_info",
            ))
        elif not sr.success and sr.error_info is None:
            signals.append(FailureSignal(
                category=FAILURE_CATEGORY_UNKNOWN,
                message="Step failed with no error info",
                source="step_failed",
            ))

    return signals


# ---------------------------------------------------------------------------
# Stage 2: Binary step viability
# ---------------------------------------------------------------------------

def _format_failures_summary(failure_signals: list[FailureSignal]) -> str:
    """Format failure signals as a summary for binary prompts."""
    lines = ["Failures observed:"]
    for fs in failure_signals:
        lines.append(f"- [{fs.category}] {fs.message}")
    return "\n".join(lines)


def _format_step_for_viability(step: PlanStep) -> str:
    """Format a single PlanStep for the viability binary judgment."""
    parts = [
        f"\nStep to evaluate:",
        f"ID: {step.id}",
        f"Description: {step.description}",
    ]
    if step.target_files:
        parts.append(f"Target files: {', '.join(step.target_files)}")
    if step.target_symbols:
        parts.append(f"Target symbols: {', '.join(step.target_symbols)}")
    parts.append("\nIs this step still valid given the failures above? Answer yes or no.")
    return "\n".join(parts)


def _run_step_viability(
    failure_signals: list[FailureSignal],
    remaining_steps: list[PlanStep],
    task_description: str,
    llm: LoggedLLMClient,
) -> dict[str, bool]:
    """Binary per-step: 'Is step S still valid given these failures? yes/no'

    Uses run_binary_judgment with default_action="excluded" (fail-fast).
    Steps judged invalid will be dropped from the revised plan.
    """
    if not remaining_steps:
        return {}

    # No failures -> all steps valid
    if not failure_signals:
        return {step.id: True for step in remaining_steps}

    system_prompt = SYSTEM_PROMPTS["adjustment_step_viability"]
    task_context = _format_failures_summary(failure_signals) + "\n"

    verdict_map, omitted = run_binary_judgment(
        remaining_steps,
        system_prompt=system_prompt,
        task_context=task_context,
        llm=llm,
        format_item=_format_step_for_viability,
        stage_name="adjustment_step_viability",
        item_key=lambda step: step.id,
        default_action="excluded",
    )

    # Fail-fast: if ALL steps omitted due to parse failure, crash
    if omitted and len(omitted) == len(remaining_steps):
        raise ValueError(
            f"All {len(remaining_steps)} step viability judgments failed to parse. "
            f"Omitted keys: {sorted(omitted)}. Cannot proceed with adjustment."
        )

    return verdict_map


# ---------------------------------------------------------------------------
# Stage 3: Binary root cause attribution
# ---------------------------------------------------------------------------

def _format_root_cause_pair(pair: tuple[tuple[int, FailureSignal], PlanStep]) -> str:
    """Format a (failure, step) pair for root cause binary judgment."""
    (fail_idx, fs), step = pair
    parts = [
        f"\nFailure:",
        f"[{fs.category}] {fs.message}",
        f"\nStep:",
        f"ID: {step.id}",
        f"Description: {step.description}",
    ]
    if step.target_files:
        parts.append(f"Target files: {', '.join(step.target_files)}")
    parts.append(
        "\nDid this step's implementation cause or contribute to this failure? "
        "Answer yes or no."
    )
    return "\n".join(parts)


def _run_root_cause_attribution(
    failure_signals: list[FailureSignal],
    remaining_steps: list[PlanStep],
    viable_steps: dict[str, bool],
    cumulative_diff: str | None,
    task_description: str,
    llm: LoggedLLMClient,
) -> dict[str, list[int]]:
    """Binary per (failure, step) pair: 'Did step S cause failure F? yes/no'

    Only evaluates pairs where the step was judged viable (still in the plan).
    Returns dict mapping step_id -> list of failure indices it caused.
    """
    if not failure_signals:
        return {}

    # Only viable steps
    viable_remaining = [s for s in remaining_steps if viable_steps.get(s.id, False)]
    if not viable_remaining:
        return {}

    # Build cross product: (indexed_failure, step) pairs
    pairs = []
    for fail_idx, fs in enumerate(failure_signals):
        for step in viable_remaining:
            pairs.append(((fail_idx, fs), step))

    if not pairs:
        return {}

    system_prompt = SYSTEM_PROMPTS["adjustment_root_cause"]

    # Budget-aware diff truncation for 0.6B binary prompts
    diff_context = ""
    if cumulative_diff:
        capped_diff = budget_truncate(
            cumulative_diff, llm.config.context_window, llm.config.max_tokens,
            max_content_fraction=0.3, stage_name="adjustment_root_cause_diff",
            keep="tail",
        )
        diff_context = f"Prior changes (most recent):\n<prior_changes>{capped_diff}</prior_changes>\n"

    verdict_map, omitted = run_binary_judgment(
        pairs,
        system_prompt=system_prompt,
        task_context=diff_context,
        llm=llm,
        format_item=_format_root_cause_pair,
        stage_name="adjustment_root_cause",
        item_key=lambda pair: f"{pair[0][0]}:{pair[1].id}",
        default_action="not_attributed",
    )

    # Fail-fast: if ALL judgments failed to parse, crash
    if omitted and len(omitted) == len(pairs):
        raise ValueError(
            f"All {len(pairs)} root cause attribution judgments failed to parse. "
            f"Omitted keys: {sorted(omitted)}. Cannot proceed with adjustment."
        )

    # Build result: step_id -> list of failure indices
    result: dict[str, list[int]] = {}
    for pair in pairs:
        (fail_idx, _fs), step = pair
        key = f"{fail_idx}:{step.id}"
        if verdict_map.get(key, False):
            result.setdefault(step.id, []).append(fail_idx)

    return result


# ---------------------------------------------------------------------------
# Stage 4: Binary new step detection
# ---------------------------------------------------------------------------

def _format_remaining_steps_summary(remaining_steps: list[PlanStep]) -> str:
    """Format remaining steps as a brief summary."""
    lines = ["Remaining plan steps:"]
    for step in remaining_steps:
        lines.append(f"- {step.id}: {step.description}")
    return "\n".join(lines)


def _format_unattributed_failure(pair: tuple[int, FailureSignal]) -> str:
    """Format a single unattributed failure for new step detection."""
    fail_idx, fs = pair
    return (
        f"\nUnattributed failure:\n"
        f"[{fs.category}] {fs.message}\n\n"
        f"No existing step was identified as causing this failure.\n"
        f"Does this failure require adding a new implementation step? Answer yes or no."
    )


def _run_new_step_detection(
    unattributed_failures: list[tuple[int, FailureSignal]],
    remaining_steps: list[PlanStep],
    task_description: str,
    llm: LoggedLLMClient,
) -> list[int]:
    """Binary per unattributed failure: 'Does this failure require a new step? yes/no'

    Only called for failures that no existing step was attributed as causing.
    Returns list of failure indices that need new steps.
    """
    if not unattributed_failures:
        return []

    system_prompt = SYSTEM_PROMPTS["adjustment_new_step"]
    task_context = _format_remaining_steps_summary(remaining_steps) + "\n"

    verdict_map, omitted = run_binary_judgment(
        unattributed_failures,
        system_prompt=system_prompt,
        task_context=task_context,
        llm=llm,
        format_item=_format_unattributed_failure,
        stage_name="adjustment_new_step",
        item_key=lambda pair: pair[0],
        default_action="no_new_step",
    )

    # Fail-fast: if ALL judgments failed to parse, crash
    if omitted and len(omitted) == len(unattributed_failures):
        raise ValueError(
            f"All {len(unattributed_failures)} new step detection judgments failed to parse. "
            f"Omitted keys: {sorted(omitted)}. Cannot proceed with adjustment."
        )

    return [fail_idx for fail_idx, _fs in unattributed_failures if verdict_map.get(fail_idx, False)]


# ---------------------------------------------------------------------------
# Stage 5: Focused 1.7B finalize
# ---------------------------------------------------------------------------

def _build_finalize_prompt(
    context: ContextPackage,
    task_description: str,
    verdicts: AdjustmentVerdicts,
    remaining_steps: list[PlanStep],
    cumulative_diff: str | None,
) -> str:
    """Build user prompt for the finalize adjustment call."""
    parts = [context.to_prompt_text()]

    if task_description != context.task.raw_task:
        parts.append(f"\n# Current Objective\n{task_description}\n")

    # Format verdicts
    verdict_lines = ["<adjustment_verdicts>", "Step viability:"]
    for step in remaining_steps:
        viable = verdicts.step_viability.get(step.id, False)
        status = "VALID" if viable else "INVALID -- will be dropped"
        verdict_lines.append(f"- {step.id} ({step.description}): {status}")

    verdict_lines.append("\nRoot cause attribution:")
    has_root_causes = False
    for step_id, fail_indices in verdicts.root_causes.items():
        for fi in fail_indices:
            fs = verdicts.failure_signals[fi]
            verdict_lines.append(
                f"- {step_id} caused failure [{fi}]: [{fs.category}] {fs.message}"
            )
            has_root_causes = True
    if not has_root_causes:
        verdict_lines.append("- (none)")

    verdict_lines.append("\nNew steps needed for:")
    if verdicts.new_steps_needed:
        for fi in verdicts.new_steps_needed:
            fs = verdicts.failure_signals[fi]
            verdict_lines.append(
                f"- Failure [{fi}]: [{fs.category}] {fs.message} -- unattributed, needs new step"
            )
    else:
        verdict_lines.append("- (none)")
    verdict_lines.append("</adjustment_verdicts>")

    parts.append("\n" + "\n".join(verdict_lines) + "\n")

    # Remaining steps as JSON
    import json
    step_dicts = [s.to_dict() for s in remaining_steps]
    parts.append(f"\n<remaining_steps>\n{json.dumps(step_dicts, indent=2)}\n</remaining_steps>\n")

    if cumulative_diff:
        parts.append(f"\n<prior_changes>\n{cumulative_diff}\n</prior_changes>\n")

    return "".join(parts)


def _finalize_adjustment(
    context: ContextPackage,
    task_description: str,
    verdicts: AdjustmentVerdicts,
    remaining_steps: list[PlanStep],
    cumulative_diff: str | None,
    llm: LoggedLLMClient,
) -> PlanAdjustment:
    """Focused 1.7B call: given binary verdicts, produce the revised step list.

    The hard judgment work is done. The model only needs to:
    1. Keep viable steps
    2. Revise steps that caused failures (based on root_causes)
    3. Add new steps for unattributed failures that need them
    4. Produce a coherent step sequence with dependencies

    Returns PlanAdjustment (same output as monolithic adjustment).
    """
    system = SYSTEM_PROMPTS["adjustment_finalize"]
    user = _build_finalize_prompt(
        context, task_description, verdicts, remaining_steps, cumulative_diff,
    )

    # R3: Budget validation
    validate_prompt_budget(
        user, system,
        llm.config.context_window, llm.config.max_tokens,
        "adjustment_finalize",
    )

    response = llm.complete(user, system=system)
    adjustment = parse_plan_response(response.text, "adjustment_finalize")
    if not isinstance(adjustment, PlanAdjustment):
        raise ValueError(
            f"Expected PlanAdjustment from adjustment_finalize parse, "
            f"got {type(adjustment).__name__}. Response: {response.text[:200]}"
        )

    # Validate revised steps (same as monolithic path)
    from clean_room_agent.execute.dataclasses import PartPlan
    if adjustment.revised_steps:
        synthetic = PartPlan(
            part_id="adjustment",
            task_summary="Revised steps",
            steps=adjustment.revised_steps,
            rationale=adjustment.rationale,
        )
        warnings = validate_plan(synthetic)
        if warnings:
            raise ValueError(
                f"Revised plan validation failed: {'; '.join(warnings)}"
            )

    return adjustment


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def decomposed_adjustment(
    context: ContextPackage,
    task_description: str,
    llm: LoggedLLMClient,
    *,
    prior_results: list[StepResult] | None = None,
    remaining_steps: list[PlanStep],
    cumulative_diff: str | None = None,
) -> PlanAdjustment:
    """Generate a PlanAdjustment via decomposed binary sub-tasks.

    1. Extract failure signals deterministically from prior_results
    2. Binary viability per remaining step
    3. Binary root cause per (failure, step) pair
    4. Binary new step detection per unattributed failure
    5. Focused 1.7B finalize with all verdicts

    Returns a PlanAdjustment identical in structure to execute_plan("adjustment").
    """
    # Edge case: no remaining steps
    if not remaining_steps:
        return PlanAdjustment(
            revised_steps=[],
            rationale="No remaining steps to adjust",
            changes_made=[],
        )

    # Stage 1: Deterministic failure extraction
    failure_signals = extract_failure_signals(
        prior_results, llm.config.context_window, llm.config.max_tokens,
    )

    # Edge case: no failures -> return steps unchanged
    if not failure_signals:
        return PlanAdjustment(
            revised_steps=list(remaining_steps),
            rationale="No failures detected -- steps unchanged",
            changes_made=[],
        )

    logger.info(
        "Decomposed adjustment: %d failures, %d remaining steps",
        len(failure_signals), len(remaining_steps),
    )

    # Stage 2: Binary step viability
    step_viability = _run_step_viability(
        failure_signals, remaining_steps, task_description, llm,
    )

    # Stage 3: Binary root cause attribution
    root_causes = _run_root_cause_attribution(
        failure_signals, remaining_steps, step_viability,
        cumulative_diff, task_description, llm,
    )

    # Stage 4: Binary new step detection (only for unattributed failures)
    attributed_failures: set[int] = set()
    for fail_indices in root_causes.values():
        attributed_failures.update(fail_indices)
    unattributed = [
        (i, fs) for i, fs in enumerate(failure_signals)
        if i not in attributed_failures
    ]

    new_steps_needed = _run_new_step_detection(
        unattributed, remaining_steps, task_description, llm,
    )

    # Assemble verdicts
    verdicts = AdjustmentVerdicts(
        step_viability=step_viability,
        root_causes=root_causes,
        new_steps_needed=new_steps_needed,
        failure_signals=failure_signals,
    )

    # Stage 5: Focused 1.7B finalize
    return _finalize_adjustment(
        context, task_description, verdicts, remaining_steps,
        cumulative_diff, llm,
    )
