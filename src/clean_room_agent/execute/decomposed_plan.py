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
    ChangePoint,
    ChangePointEnumeration,
    MetaPlan,
    MetaPlanPart,
    PartGroup,
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
from clean_room_agent.execute.prompts import SYSTEM_PROMPTS, build_decomposed_plan_prompt
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
    *,
    use_binary_grouping: bool = False,
) -> MetaPlan:
    """Generate a MetaPlan via three decomposed stages.

    1. Change point enumeration — identify all files/symbols that need to change
    2. Part grouping — cluster change points into logical parts
       (monolithic or pairwise binary depending on use_binary_grouping)
    3. Binary part dependencies — pairwise "does B depend on A?" judgments

    Returns a MetaPlan structurally identical to what execute_plan("meta_plan") returns.
    """
    enum_result = _run_change_point_enum(context, task_description, llm)
    if use_binary_grouping:
        grouping = _run_part_grouping_binary(task_description, enum_result, llm)
    else:
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
    response = llm.complete(user, system=system, sub_stage="change_point_enum")
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
    response = llm.complete(user, system=system, sub_stage="part_grouping")
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


# ---------------------------------------------------------------------------
# Binary part grouping (A11 decomposition)
# ---------------------------------------------------------------------------

def _format_change_point_pair(pair: tuple[tuple[int, ChangePoint], tuple[int, ChangePoint]]) -> str:
    """Format a pair of change points for binary judgment."""
    (idx_a, cp_a), (idx_b, cp_b) = pair
    return (
        f"\nChange Point A (index {idx_a}): [{cp_a.change_type}] "
        f"{cp_a.file_path} :: {cp_a.symbol}\n"
        f"  Rationale: {cp_a.rationale}\n"
        f"\nChange Point B (index {idx_b}): [{cp_b.change_type}] "
        f"{cp_b.file_path} :: {cp_b.symbol}\n"
        f"  Rationale: {cp_b.rationale}\n"
        f"\nShould these two change points be in the same implementation part?"
    )


def _union_find_groups(pairs: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Union-find grouping from pairwise yes verdicts.

    Args:
        pairs: list of (index_a, index_b) pairs that should be grouped.

    Returns:
        dict mapping root index -> list of member indices (sorted).
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])  # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            if rx > ry:
                rx, ry = ry, rx  # determinism: smaller root wins
            parent[ry] = rx

    for a, b in pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    groups: dict[int, list[int]] = {}
    for idx in parent:
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    # Sort members within each group for determinism
    for root in groups:
        groups[root].sort()

    return groups


def _single_change_point_grouping(cps: list[ChangePoint]) -> PartGrouping:
    """Handle trivial case: 0 or 1 change points."""
    if not cps:
        raise ValueError("Cannot group zero change points")
    return PartGrouping(parts=[
        PartGroup(
            id="p1",
            description=f"{cps[0].change_type.capitalize()} {cps[0].symbol} in {cps[0].file_path}",
            change_point_indices=[0],
            affected_files=[cps[0].file_path],
        ),
    ])


def _generate_part_description(
    member_indices: list[int],
    change_points: list[ChangePoint],
) -> str:
    """Generate a deterministic part description from its change points.

    Strategy:
    - If all change points share the same file: "{change_type} {symbols} in {file}"
    - If multiple files: "{change_types} across {files}: {symbols}"
    - Uses the change points' rationales to add context
    """
    cps = [change_points[i] for i in member_indices]

    files = list(dict.fromkeys(cp.file_path for cp in cps))  # deduplicated, ordered
    symbols = list(dict.fromkeys(cp.symbol for cp in cps))
    change_types = list(dict.fromkeys(cp.change_type for cp in cps))

    type_str = "/".join(change_types)
    sym_str = ", ".join(symbols)

    if len(files) == 1:
        desc = f"{type_str.capitalize()} {sym_str} in {files[0]}"
    else:
        file_str = ", ".join(files)
        desc = f"{type_str.capitalize()} across {file_str}: {sym_str}"

    # Append first rationale as context
    desc += f" -- {cps[0].rationale}"

    return desc


def _build_grouping_from_components(
    groups: dict[int, list[int]],
    change_points: list[ChangePoint],
) -> PartGrouping:
    """Build PartGrouping deterministically from union-find components.

    Description: derived from the change points' file paths and symbols.
    Affected files: unique file_paths from the component's change points.
    Part IDs: "p1", "p2", ... in order of smallest member index.
    """
    # Sort groups by smallest member index for deterministic part ordering
    sorted_roots = sorted(groups.keys())

    parts = []
    for part_num, root in enumerate(sorted_roots, start=1):
        members = groups[root]  # already sorted in _union_find_groups

        # Collect affected files (deduplicated, order-preserving)
        affected_files: list[str] = []
        seen_files: set[str] = set()
        for idx in members:
            fp = change_points[idx].file_path
            if fp not in seen_files:
                affected_files.append(fp)
                seen_files.add(fp)

        # Generate description from member change points
        description = _generate_part_description(members, change_points)

        parts.append(PartGroup(
            id=f"p{part_num}",
            description=description,
            change_point_indices=list(members),
            affected_files=affected_files,
        ))

    return PartGrouping(parts=parts)


def _run_part_grouping_binary(
    task_description: str,
    enum_result: ChangePointEnumeration,
    llm: LoggedLLMClient,
) -> PartGrouping:
    """Stage 2 (decomposed): Group change points via pairwise binary judgments.

    1. Enumerate all N*(N-1)/2 unique pairs
    2. Binary judgment per pair: "same part? yes/no"
    3. Union-find on yes verdicts
    4. Deterministic description + affected_files from components

    No context package needed -- only change points and task description.
    """
    cps = enum_result.change_points
    n = len(cps)

    # Edge case: single change point -> single part
    if n <= 1:
        return _single_change_point_grouping(cps)

    # Step 1: Enumerate all unique pairs (N*(N-1)/2)
    indexed_cps = list(enumerate(cps))
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((indexed_cps[i], indexed_cps[j]))

    # Step 2: Binary judgment per pair
    system_prompt = SYSTEM_PROMPTS["part_grouping_binary"]

    def pair_key(pair):
        (idx_a, _), (idx_b, _) = pair
        return f"{idx_a}-{idx_b}"

    verdict_map, omitted = run_binary_judgment(
        pairs,
        system_prompt=system_prompt,
        task_context=f"Task: {task_description}\n",
        llm=llm,
        format_item=_format_change_point_pair,
        stage_name="part_grouping_binary",
        item_key=pair_key,
        default_action="separate_parts",
    )
    # R2 default-deny: omitted pairs default to "no" (separate parts).
    # This is safe -- the worst case is too many parts, not too few.
    # Unlike part_dependency (which fail-fasts on omissions), grouping
    # tolerates parse failures because over-splitting is recoverable.

    # Step 3: Union-find on yes verdicts
    yes_pairs = []
    for pair in pairs:
        key = pair_key(pair)
        if verdict_map.get(key, False):
            (idx_a, _), (idx_b, _) = pair
            yes_pairs.append((idx_a, idx_b))

    # Ensure all indices appear in parent map (including isolated ones)
    groups = _union_find_groups(yes_pairs)
    # Add isolated indices (no yes verdict with any other)
    grouped_indices = set()
    for members in groups.values():
        grouped_indices.update(members)
    for i in range(n):
        if i not in grouped_indices:
            groups[i] = [i]

    # Step 4: Deterministic PartGrouping construction
    grouping = _build_grouping_from_components(groups, cps)

    # Validate: same check as monolithic path
    warnings = validate_part_grouping(grouping, len(cps))
    if warnings:
        raise ValueError(
            f"Binary part grouping validation failed: {'; '.join(warnings)}"
        )

    return grouping


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
    # omitted check above guarantees all keys present in verdict_map
    dep_edges: dict[str, list[str]] = {p.id: [] for p in parts}
    for pair in pairs:
        a, b = pair
        key = f"{a.id}->{b.id}"
        if verdict_map[key]:
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
            depends_on=list(dep_edges[pg.id]),
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
    response = llm.complete(user, system=system, sub_stage="symbol_targeting")
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
    response = llm.complete(user, system=system, sub_stage="step_design")
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
    # omitted check above guarantees all keys present in verdict_map
    dep_edges: dict[str, list[str]] = {s.id: [] for s in steps}
    for pair in pairs:
        a, b = pair
        key = f"{a.id}->{b.id}"
        if verdict_map[key]:
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
            depends_on=list(dep_edges[step.id]),
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
