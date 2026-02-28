# Implementation Plan: Decompose A11 Part Grouping

Date: 2026-02-28

## 1. Overview

**Current state**: `_run_part_grouping()` in `src/clean_room_agent/execute/decomposed_plan.py` (lines 82-122) sends all change points to a 1.7B model in a single call, asking it to produce clustered groups (a `PartGrouping` with `PartGroup` objects). This is task A11 in the adapter map.

**Target state**: Replace the single LLM call with:
1. Deterministic enumeration of all N*(N-1)/2 unique change point pairs
2. Per-pair binary judgment via `run_binary_judgment()`: "Should these two change points be in the same implementation part?"
3. Union-find on "yes" verdicts to build connected components
4. Deterministic construction of `PartGrouping` from the components (descriptions, affected_files, change_point_indices -- all derivable from the change points themselves)

**Output type unchanged**: `PartGrouping` with `PartGroup` elements. No downstream code changes needed.

## 2. Config Toggle

**File**: `src/clean_room_agent/config.py` (line ~128)

Add a new commented-out config line in the template:
```
'# decomposed_part_grouping = false  # Pairwise binary part grouping (A11 decomposition)\n'
```

**File**: `src/clean_room_agent/orchestrator/runner.py` (after line 85)

Add a new config check function, following the exact pattern of `_use_decomposed_planning` and `_use_decomposed_scaffold`:

```python
def _use_decomposed_part_grouping(config: dict) -> bool:
    """Check if decomposed part grouping is enabled. Supplementary, default False."""
    orch = config.get("orchestrator", {})
    if "decomposed_part_grouping" not in orch:
        logger.debug("decomposed_part_grouping not in config, defaulting to False")
    return bool(orch.get("decomposed_part_grouping", False))
```

Note: This toggle is independent of `decomposed_planning`. When `decomposed_planning = true` and `decomposed_part_grouping = true`, the decomposed meta-plan pipeline uses the pairwise binary grouping. When `decomposed_planning = true` and `decomposed_part_grouping = false` (or absent), the existing monolithic `_run_part_grouping()` is used. When `decomposed_planning = false`, neither is called (the monolithic M1 meta-plan handles everything).

## 3. New System Prompt

**File**: `src/clean_room_agent/execute/prompts.py`

Add a new entry to `SYSTEM_PROMPTS` dict (after the `"part_grouping"` entry at line 152):

```python
"part_grouping_binary": (
    "You are Jane, a task decomposition analyst. You will be given two change points "
    "(files/symbols that need modification) from the same task. Determine whether these "
    "two change points should be implemented together in the same part.\n\n"
    "Two change points belong in the same part if:\n"
    "- They modify the same logical component or subsystem\n"
    "- One change depends on or closely interacts with the other\n"
    "- They must be tested together to verify correctness\n"
    "- Splitting them into separate parts would create artificial entanglement\n\n"
    "Two change points belong in different parts if:\n"
    "- They affect independent subsystems\n"
    "- Either could be implemented and tested without the other\n"
    "- They share no data flow or control flow coupling\n\n"
    "Answer with exactly one word: \"yes\" or \"no\""
),
```

This prompt is designed for a 0.6B classifier. It is concise, the question is binary, and the criteria are concrete enough to answer without seeing the full change point list.

## 4. Binary Pair Format Function

Each pair presented to the 0.6B will see:

```
Task: <task_description>

Change Point A (index 0): [modify] src/main.py :: validate
  Rationale: Fix input validation for edge cases

Change Point B (index 3): [modify] src/main.py :: sanitize
  Rationale: Add sanitization before validation

Should these two change points be in the same implementation part?
```

The format function will be:

```python
def _format_change_point_pair(pair: tuple[tuple[int, ChangePoint], tuple[int, ChangePoint]]) -> str:
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
```

Each pair item is `tuple[tuple[int, ChangePoint], tuple[int, ChangePoint]]` -- a pair of (index, change_point) tuples. The index is the position in the original `enum_result.change_points` list. Showing the index provides traceability. Showing `change_type`, `file_path`, `symbol`, and `rationale` gives the classifier enough context without overwhelming it.

## 5. Union-Find Implementation

Inline in `decomposed_plan.py`, following the exact pattern from `similarity_stage.py` lines 205-228. Not extracted to a shared module -- the union-find is 15 lines and used in only two places with slightly different key types (int symbol_id vs. int change point index). The cost of a shared abstraction outweighs the cost of the duplication.

```python
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
```

## 6. Main Decomposed Function

**File**: `src/clean_room_agent/execute/decomposed_plan.py`

Add a new function `_run_part_grouping_binary()` alongside the existing `_run_part_grouping()`. The existing function remains untouched (it is the fallback).

```python
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
    from clean_room_agent.execute.prompts import SYSTEM_PROMPTS
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
```

## 7. Deterministic Description Generation

```python
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

    # Append first rationale as context (truncated)
    first_rationale = cps[0].rationale
    if len(first_rationale) > 80:
        first_rationale = first_rationale[:77] + "..."
    desc += f" -- {first_rationale}"

    return desc
```

## 8. Integration Point: Modify `decomposed_meta_plan()`

**File**: `src/clean_room_agent/execute/decomposed_plan.py`, function `decomposed_meta_plan()` (line 42)

Add a `use_binary_grouping` kwarg and dispatch to the binary version when enabled:

```python
def decomposed_meta_plan(
    context: ContextPackage,
    task_description: str,
    llm: LoggedLLMClient,
    *,
    use_binary_grouping: bool = False,
) -> MetaPlan:
    """Generate a MetaPlan via three decomposed stages.

    1. Change point enumeration
    2. Part grouping (monolithic or pairwise binary depending on use_binary_grouping)
    3. Binary part dependencies
    """
    enum_result = _run_change_point_enum(context, task_description, llm)
    if use_binary_grouping:
        grouping = _run_part_grouping_binary(task_description, enum_result, llm)
    else:
        grouping = _run_part_grouping(context, task_description, enum_result, llm)
    dep_edges = _run_part_dependencies(grouping, task_description, llm)
    return _assemble_meta_plan(enum_result, grouping, dep_edges)
```

Note that `_run_part_grouping_binary()` does NOT take a `ContextPackage` -- it only needs the change points and task description. The monolithic version needs context because it sends the full codebase context to the LLM; the binary version's per-pair prompts only need the two change points.

## 9. Caller Updates

**File**: `src/clean_room_agent/orchestrator/runner.py` (line 653)

```python
if _use_decomposed_planning(config):
    meta_plan = decomposed_meta_plan(
        context, task, llm,
        use_binary_grouping=_use_decomposed_part_grouping(config),
    )
```

**File**: `src/clean_room_agent/commands/plan.py` (line 73-77)

```python
if decomposed:
    from clean_room_agent.orchestrator.runner import _use_decomposed_part_grouping
    meta_plan = decomposed_meta_plan(
        package, task, llm,
        use_binary_grouping=_use_decomposed_part_grouping(config),
    )
```

## 10. Validation

The existing `validate_part_grouping()` in `parsers.py` (lines 235-280) is still called. The binary version validates at the end of `_run_part_grouping_binary()` before returning. This validation is a deterministic sanity check (all indices covered, no duplicates, no out-of-range). Since the binary path builds groups deterministically from the union-find output + isolated index handling, this should never fire -- but it catches implementation bugs in the union-find logic, which is the correct fail-fast behavior.

## 11. What Does NOT Change

- **`PartGrouping` / `PartGroup` dataclasses**: Unchanged. Same fields, same `from_dict()` / `to_dict()`.
- **`validate_part_grouping()`**: Unchanged. Both paths validate with the same function.
- **`_assemble_meta_plan()`**: Unchanged. It consumes `PartGrouping` regardless of how it was produced.
- **`_run_part_dependencies()`**: Unchanged. It receives `PartGrouping` and produces dependency edges.
- **`MetaPlan` / `MetaPlanPart`**: Unchanged.
- **Orchestrator, `run_pipeline()`, all downstream**: Unchanged.
- **`parse_plan_response()` for "part_grouping"**: Still exists for the monolithic path. Not used by the binary path (which doesn't parse JSON from the LLM).
- **`SYSTEM_PROMPTS["part_grouping"]`**: Retained for the monolithic fallback.

## 12. Scaling Analysis

For N change points:
- Pairs: N*(N-1)/2
- Typical N: 3-10 change points per task
  - N=3: 3 pairs
  - N=5: 10 pairs
  - N=10: 45 pairs
  - N=15: 105 pairs
  - N=20: 190 pairs

At N=20, 190 binary 0.6B calls is significant but manageable (each is tiny). For N > 20, consider a warning log. For the current project scope (most tasks have 3-15 change points), this is fine. If N gets large enough to be problematic, the solution is deterministic pre-clustering (e.g., group by file first), not falling back to monolithic -- but that's a future optimization.

## 13. Test Plan

**File**: `tests/execute/test_decomposed_plan.py`

Add a new test class `TestRunPartGroupingBinary` and expand integration tests.

### Unit Tests for `_union_find_groups`

```python
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
```

### Unit Tests for `_generate_part_description`

```python
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
```

### Unit Tests for `_build_grouping_from_components`

```python
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
```

### Unit Tests for `_single_change_point_grouping`

```python
class TestSingleChangePointGrouping:
    def test_one_change_point(self):
        cps = [ChangePoint(file_path="a.py", symbol="foo", change_type="modify", rationale="r")]
        result = _single_change_point_grouping(cps)
        assert len(result.parts) == 1
        assert result.parts[0].change_point_indices == [0]

    def test_zero_change_points_raises(self):
        with pytest.raises(ValueError, match="zero change points"):
            _single_change_point_grouping([])
```

### Unit Tests for `_run_part_grouping_binary`

```python
class TestRunPartGroupingBinary:
    def test_two_change_points_same_part(self, model_config):
        """Two change points, classifier says yes -> one part."""

    def test_two_change_points_different_parts(self, model_config):
        """Two change points, classifier says no -> two parts."""

    def test_three_change_points_transitive(self, model_config):
        """Three CPs: (0,1)=yes, (0,2)=no, (1,2)=yes -> all in one group via transitivity."""

    def test_single_change_point(self, model_config):
        """One change point -> one part, no LLM calls."""

    def test_parse_failure_defaults_to_separate(self, model_config):
        """R2: unparseable response defaults to 'no' (separate parts)."""

    def test_validation_passes(self, model_config):
        """Grouping passes validate_part_grouping (all indices covered)."""
```

### Integration Test: Full Pipeline with Binary Grouping

```python
class TestDecomposedMetaPlanBinaryGrouping:
    def test_end_to_end_binary_grouping(self, context_package, model_config):
        """Full decomposed meta-plan with binary grouping."""
```

## 14. Summary of File Changes

| File | Change |
|------|--------|
| `src/clean_room_agent/execute/decomposed_plan.py` | Add `_run_part_grouping_binary()`, `_union_find_groups()`, `_single_change_point_grouping()`, `_build_grouping_from_components()`, `_generate_part_description()`, `_format_change_point_pair()`. Modify `decomposed_meta_plan()` signature to accept `use_binary_grouping` kwarg. |
| `src/clean_room_agent/execute/prompts.py` | Add `"part_grouping_binary"` entry to `SYSTEM_PROMPTS` dict. |
| `src/clean_room_agent/orchestrator/runner.py` | Add `_use_decomposed_part_grouping()`. Modify two call sites to pass `use_binary_grouping` kwarg. |
| `src/clean_room_agent/commands/plan.py` | Pass `use_binary_grouping` to `decomposed_meta_plan()`. |
| `src/clean_room_agent/config.py` | Add commented template line for `decomposed_part_grouping`. |
| `tests/execute/test_decomposed_plan.py` | Add 6 new test classes (~20 test functions). |
| `protocols/design_records/per_task_adapter_map.md` | Update A11 decomposition status to "Done". |

## 15. Implementation Sequence

1. Add `"part_grouping_binary"` to `SYSTEM_PROMPTS` in `prompts.py`
2. Add `_union_find_groups()`, `_single_change_point_grouping()`, `_generate_part_description()`, `_build_grouping_from_components()`, `_format_change_point_pair()`, `_run_part_grouping_binary()` to `decomposed_plan.py`
3. Modify `decomposed_meta_plan()` to accept `use_binary_grouping` kwarg
4. Add `_use_decomposed_part_grouping()` to `orchestrator/runner.py`
5. Update call sites in `runner.py` and `commands/plan.py`
6. Add config template line in `config.py`
7. Write all tests
8. Update `per_task_adapter_map.md` decomposition status

## Critical Files for Implementation

- `src/clean_room_agent/execute/decomposed_plan.py` - Primary implementation file
- `src/clean_room_agent/execute/prompts.py` - Add the `part_grouping_binary` system prompt
- `src/clean_room_agent/orchestrator/runner.py` - Config toggle and call site update
- `src/clean_room_agent/retrieval/similarity_stage.py` - Reference pattern: existing union-find at lines 191-244
- `tests/execute/test_decomposed_plan.py` - All new tests
