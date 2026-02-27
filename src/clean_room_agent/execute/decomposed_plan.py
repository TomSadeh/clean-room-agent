"""Decomposed planning: multi-stage meta-plan and part-plan generation.

Replaces single-call meta_plan/part_plan with smaller cognitive tasks:
- Meta-plan: change_point_enum → part_grouping → binary part_dependency
- Part-plan: symbol_targeting → step_design → binary step_dependency

Output types (MetaPlan, PartPlan) are unchanged. The orchestrator sees
no difference. This is a decomposition of the internal planning process.
"""

from __future__ import annotations

import json
import logging

from clean_room_agent.execute.dataclasses import (
    ChangePointEnumeration,
    MetaPlan,
    MetaPlanPart,
    PartGrouping,
    PartPlan,
    PlanStep,
    SymbolTargetEnumeration,
)
from clean_room_agent.execute.parsers import (
    parse_plan_response,
    validate_part_grouping,
    validate_plan,
)
from clean_room_agent.execute.prompts import build_decomposed_plan_prompt
from clean_room_agent.llm.client import LoggedLLMClient
from clean_room_agent.retrieval.batch_judgment import run_binary_judgment
from clean_room_agent.retrieval.dataclasses import ContextPackage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decomposed meta-plan
# ---------------------------------------------------------------------------

def decomposed_meta_plan(
    context: ContextPackage,
    task_description: str,
    llm: LoggedLLMClient,
) -> MetaPlan:
    """Generate a MetaPlan via three decomposed stages.

    1. Change point enumeration — identify all files/symbols that need to change
    2. Part grouping — cluster change points into logical parts
    3. Binary part dependencies — pairwise "does B depend on A?" judgments

    Returns a MetaPlan structurally identical to what execute_plan("meta_plan") returns.
    """
    enum_result = _run_change_point_enum(context, task_description, llm)
    grouping = _run_part_grouping(context, task_description, enum_result, llm)
    dep_edges = _run_part_dependencies(grouping, task_description, llm)
    return _assemble_meta_plan(enum_result, grouping, dep_edges)


def _run_change_point_enum(
    context: ContextPackage,
    task_description: str,
    llm: LoggedLLMClient,
) -> ChangePointEnumeration:
    """Stage 1: Enumerate all change points (files + symbols)."""
    system, user = build_decomposed_plan_prompt(
        context, task_description,
        pass_type="change_point_enum",
        model_config=llm.config,
    )
    response = llm.complete(user, system=system)
    result = parse_plan_response(response.text, "change_point_enum")
    if not isinstance(result, ChangePointEnumeration):
        raise ValueError(
            f"Expected ChangePointEnumeration from change_point_enum parse, "
            f"got {type(result).__name__}. Response: {response.text[:200]}"
        )
    return result


def _run_part_grouping(
    context: ContextPackage,
    task_description: str,
    enum_result: ChangePointEnumeration,
    llm: LoggedLLMClient,
) -> PartGrouping:
    """Stage 2: Group change points into logical parts."""
    # Serialize change points as prior_stage_output
    cp_list = []
    for i, cp in enumerate(enum_result.change_points):
        cp_list.append({
            "index": i,
            "file_path": cp.file_path,
            "symbol": cp.symbol,
            "change_type": cp.change_type,
            "rationale": cp.rationale,
        })
    prior_output = json.dumps(cp_list, indent=2)

    system, user = build_decomposed_plan_prompt(
        context, task_description,
        pass_type="part_grouping",
        model_config=llm.config,
        prior_stage_output=prior_output,
    )
    response = llm.complete(user, system=system)
    result = parse_plan_response(response.text, "part_grouping")
    if not isinstance(result, PartGrouping):
        raise ValueError(
            f"Expected PartGrouping from part_grouping parse, "
            f"got {type(result).__name__}. Response: {response.text[:200]}"
        )

    # Validate grouping covers all change points
    warnings = validate_part_grouping(result, len(enum_result.change_points))
    if warnings:
        raise ValueError(
            f"Part grouping validation failed: {'; '.join(warnings)}"
        )

    return result


def _run_part_dependencies(
    grouping: PartGrouping,
    task_description: str,
    llm: LoggedLLMClient,
) -> dict[str, list[str]]:
    """Stage 3: Binary pairwise dependency judgment between parts."""
    parts = grouping.parts
    if len(parts) <= 1:
        return {p.id: [] for p in parts}

    # Generate all ordered pairs (A, B) — check "does B depend on A?"
    pairs = []
    for a in parts:
        for b in parts:
            if a.id != b.id:
                pairs.append((a, b))

    from clean_room_agent.execute.prompts import SYSTEM_PROMPTS
    system_prompt = SYSTEM_PROMPTS["part_dependency"]

    def format_pair(pair):
        a, b = pair
        return (
            f"\nPart A — {a.id}: {a.description}\n"
            f"Part B — {b.id}: {b.description}\n"
            f"Does Part B depend on Part A?"
        )

    def pair_key(pair):
        a, b = pair
        return f"{a.id}->{b.id}"

    verdict_map, omitted = run_binary_judgment(
        pairs,
        system_prompt=system_prompt,
        task_context=f"Task: {task_description}\n",
        llm=llm,
        format_item=format_pair,
        stage_name="part_dependency",
        item_key=pair_key,
        default_action="no_dependency",
    )
    if omitted:
        raise ValueError(
            f"Part dependency judgment failed to parse {len(omitted)} of "
            f"{len(pairs)} pairs: {sorted(omitted)}. "
            f"Cannot build dependency graph from incomplete judgments."
        )

    # Build dependency dict: {part_id: [depends_on_ids]}
    dep_edges: dict[str, list[str]] = {p.id: [] for p in parts}
    for pair in pairs:
        a, b = pair
        key = f"{a.id}->{b.id}"
        if verdict_map.get(key, False):
            dep_edges[b.id].append(a.id)

    return dep_edges


def _assemble_meta_plan(
    enum_result: ChangePointEnumeration,
    grouping: PartGrouping,
    dep_edges: dict[str, list[str]],
) -> MetaPlan:
    """Assemble a MetaPlan from decomposed stage outputs."""
    meta_parts = []
    for pg in grouping.parts:
        meta_parts.append(MetaPlanPart(
            id=pg.id,
            description=pg.description,
            affected_files=list(pg.affected_files),
            depends_on=list(dep_edges.get(pg.id, [])),
        ))

    meta_plan = MetaPlan(
        task_summary=enum_result.task_summary,
        parts=meta_parts,
        rationale=(
            f"Decomposed from {len(enum_result.change_points)} change points "
            f"into {len(grouping.parts)} parts"
        ),
    )

    warnings = validate_plan(meta_plan)
    if warnings:
        raise ValueError(
            f"Assembled meta-plan validation failed: {'; '.join(warnings)}"
        )

    return meta_plan


# ---------------------------------------------------------------------------
# Decomposed part-plan
# ---------------------------------------------------------------------------

def decomposed_part_plan(
    context: ContextPackage,
    part_description: str,
    part_id: str,
    llm: LoggedLLMClient,
    *,
    cumulative_diff: str | None = None,
) -> PartPlan:
    """Generate a PartPlan via three decomposed stages.

    1. Symbol targeting — identify specific symbols to modify
    2. Step design — design implementation steps (no dependency info)
    3. Binary step dependencies — pairwise "does B depend on A?" judgments

    Returns a PartPlan structurally identical to what execute_plan("part_plan") returns.
    """
    targets = _run_symbol_targeting(context, part_description, part_id, llm, cumulative_diff)
    step_plan = _run_step_design(context, part_description, targets, llm, cumulative_diff)
    dep_edges = _run_step_dependencies(step_plan, part_description, llm)
    return _assemble_part_plan(step_plan, dep_edges)


def _run_symbol_targeting(
    context: ContextPackage,
    part_description: str,
    part_id: str,
    llm: LoggedLLMClient,
    cumulative_diff: str | None,
) -> SymbolTargetEnumeration:
    """Stage 1: Identify specific symbols to modify for this part."""
    system, user = build_decomposed_plan_prompt(
        context, part_description,
        pass_type="symbol_targeting",
        model_config=llm.config,
        cumulative_diff=cumulative_diff,
    )
    response = llm.complete(user, system=system)
    result = parse_plan_response(response.text, "symbol_targeting")
    if not isinstance(result, SymbolTargetEnumeration):
        raise ValueError(
            f"Expected SymbolTargetEnumeration from symbol_targeting parse, "
            f"got {type(result).__name__}. Response: {response.text[:200]}"
        )
    return result


def _run_step_design(
    context: ContextPackage,
    part_description: str,
    targets: SymbolTargetEnumeration,
    llm: LoggedLLMClient,
    cumulative_diff: str | None,
) -> PartPlan:
    """Stage 2: Design implementation steps from symbol targets."""
    # Serialize targets as prior_stage_output
    target_list = []
    for t in targets.targets:
        target_list.append({
            "file_path": t.file_path,
            "symbol": t.symbol,
            "action": t.action,
            "rationale": t.rationale,
        })
    prior_output = json.dumps(target_list, indent=2)

    system, user = build_decomposed_plan_prompt(
        context, part_description,
        pass_type="step_design",
        model_config=llm.config,
        prior_stage_output=prior_output,
        cumulative_diff=cumulative_diff,
    )
    response = llm.complete(user, system=system)
    result = parse_plan_response(response.text, "step_design")
    if not isinstance(result, PartPlan):
        raise ValueError(
            f"Expected PartPlan from step_design parse, "
            f"got {type(result).__name__}. Response: {response.text[:200]}"
        )
    return result


def _run_step_dependencies(
    step_plan: PartPlan,
    part_description: str,
    llm: LoggedLLMClient,
) -> dict[str, list[str]]:
    """Stage 3: Binary pairwise dependency judgment between steps."""
    steps = step_plan.steps
    if len(steps) <= 1:
        return {s.id: [] for s in steps}

    # Generate all ordered pairs (A, B) — check "does B depend on A?"
    pairs = []
    for a in steps:
        for b in steps:
            if a.id != b.id:
                pairs.append((a, b))

    from clean_room_agent.execute.prompts import SYSTEM_PROMPTS
    system_prompt = SYSTEM_PROMPTS["step_dependency"]

    def format_pair(pair):
        a, b = pair
        return (
            f"\nStep A — {a.id}: {a.description}\n"
            f"Step B — {b.id}: {b.description}\n"
            f"Does Step B depend on Step A?"
        )

    def pair_key(pair):
        a, b = pair
        return f"{a.id}->{b.id}"

    verdict_map, omitted = run_binary_judgment(
        pairs,
        system_prompt=system_prompt,
        task_context=f"Part: {part_description}\n",
        llm=llm,
        format_item=format_pair,
        stage_name="step_dependency",
        item_key=pair_key,
        default_action="no_dependency",
    )
    if omitted:
        raise ValueError(
            f"Step dependency judgment failed to parse {len(omitted)} of "
            f"{len(pairs)} pairs: {sorted(omitted)}. "
            f"Cannot build dependency graph from incomplete judgments."
        )

    # Build dependency dict: {step_id: [depends_on_ids]}
    dep_edges: dict[str, list[str]] = {s.id: [] for s in steps}
    for pair in pairs:
        a, b = pair
        key = f"{a.id}->{b.id}"
        if verdict_map.get(key, False):
            dep_edges[b.id].append(a.id)

    return dep_edges


def _assemble_part_plan(
    step_plan: PartPlan,
    dep_edges: dict[str, list[str]],
) -> PartPlan:
    """Augment step plan with dependency edges from binary judgments."""
    augmented_steps = []
    for step in step_plan.steps:
        augmented_steps.append(PlanStep(
            id=step.id,
            description=step.description,
            target_files=list(step.target_files),
            target_symbols=list(step.target_symbols),
            depends_on=list(dep_edges.get(step.id, [])),
        ))

    part_plan = PartPlan(
        part_id=step_plan.part_id,
        task_summary=step_plan.task_summary,
        steps=augmented_steps,
        rationale=step_plan.rationale,
    )

    warnings = validate_plan(part_plan)
    if warnings:
        raise ValueError(
            f"Assembled part-plan validation failed: {'; '.join(warnings)}"
        )

    return part_plan
