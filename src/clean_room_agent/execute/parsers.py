"""Response parsers for plan JSON and implement XML."""

from __future__ import annotations

import re
from collections import deque

from clean_room_agent.execute.dataclasses import (
    ChangePointEnumeration,
    InterfaceEnumeration,
    MetaPlan,
    MetaPlanPart,
    PartGrouping,
    PartPlan,
    PatchEdit,
    PlanAdjustment,
    PlanStep,
    SymbolTargetEnumeration,
)
from clean_room_agent.retrieval.utils import parse_json_response


def parse_plan_response(text: str, pass_type: str) -> MetaPlan | PartPlan | PlanAdjustment:
    """Parse an LLM plan response into a typed dataclass.

    Args:
        text: Raw LLM response text (may include markdown fencing).
        pass_type: One of "meta_plan", "part_plan", "adjustment".

    Raises:
        ValueError: On malformed JSON or missing required fields.
    """
    data = parse_json_response(text, context=f"{pass_type} plan")
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object for {pass_type}, got {type(data).__name__}")

    if pass_type == "meta_plan":
        return MetaPlan.from_dict(data)
    elif pass_type == "part_plan":
        return PartPlan.from_dict(data)
    elif pass_type == "test_plan":
        return PartPlan.from_dict(data)
    elif pass_type == "adjustment":
        return PlanAdjustment.from_dict(data)
    elif pass_type == "change_point_enum":
        return ChangePointEnumeration.from_dict(data)
    elif pass_type == "part_grouping":
        return PartGrouping.from_dict(data)
    elif pass_type == "symbol_targeting":
        return SymbolTargetEnumeration.from_dict(data)
    elif pass_type == "step_design":
        return PartPlan.from_dict(data)
    else:
        raise ValueError(f"Unknown pass_type: {pass_type!r}")


def parse_scaffold_response(text: str, pass_type: str) -> InterfaceEnumeration:
    """Parse a decomposed scaffold LLM response into a typed dataclass.

    Args:
        text: Raw LLM response text (may include markdown fencing).
        pass_type: Must be "interface_enum".

    Raises:
        ValueError: On malformed JSON, missing required fields, or unknown pass_type.
    """
    if pass_type != "interface_enum":
        raise ValueError(f"Unknown scaffold pass_type: {pass_type!r}")

    data = parse_json_response(text, context=f"{pass_type} scaffold")
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object for {pass_type}, got {type(data).__name__}")

    result = InterfaceEnumeration.from_dict(data)
    if not result.functions:
        raise ValueError(
            "InterfaceEnumeration has no functions — scaffold requires at least "
            "one function to generate stubs"
        )
    return result


# Regex for finding <edit file="...">...</edit> blocks.
# Inner <search>/<replacement> tags are parsed sequentially within each block
# to avoid content injection when code contains literal </search> or
# </replacement> tags (R5).
_EDIT_BLOCK_PATTERN = re.compile(
    r'<edit\s+file="([^"]+)">(.*?)</edit>',
    re.DOTALL,
)


def _strip_one_newline(s: str) -> str:
    """Strip exactly one leading and one trailing newline (formatting only).

    Handles both \\n and \\r\\n line endings. Does not strip multiple
    newlines — those are part of the content.
    """
    if s.startswith("\r\n"):
        s = s[2:]
    elif s.startswith("\n"):
        s = s[1:]
    if s.endswith("\r\n"):
        s = s[:-2]
    elif s.endswith("\n"):
        s = s[:-1]
    return s


def _parse_edit_block(block: str, file_path: str) -> tuple[str, str]:
    """Parse <search> and <replacement> from within an <edit> block.

    Uses rfind for closing tags so that code containing literal </search>
    or </replacement> tags does not break parsing (R5).
    """
    search_open = block.find("<search>")
    search_close = block.rfind("</search>")
    repl_open = block.find("<replacement>")
    repl_close = block.rfind("</replacement>")

    if search_open == -1 or search_close == -1 or search_close <= search_open:
        raise ValueError(f"Missing or malformed <search> block in edit for {file_path!r}")
    if repl_open == -1 or repl_close == -1 or repl_close <= repl_open:
        raise ValueError(f"Missing or malformed <replacement> block in edit for {file_path!r}")

    search = block[search_open + len("<search>"):search_close]
    replacement = block[repl_open + len("<replacement>"):repl_close]

    search = _strip_one_newline(search)
    replacement = _strip_one_newline(replacement)

    return search, replacement


def parse_implement_response(text: str) -> list[PatchEdit]:
    """Parse implementation response XML into PatchEdit list.

    Expected format:
        <edit file="path/to/file.py">
        <search>old code</search>
        <replacement>new code</replacement>
        </edit>

    Uses sequential tag parsing within each block for robustness against
    content injection (R5: code containing literal XML-like tags).

    Raises:
        ValueError: On no valid edit blocks found or malformed structure.
    """
    matches = _EDIT_BLOCK_PATTERN.findall(text)
    if not matches:
        raise ValueError(
            f"No valid <edit> blocks found in implement response.\nRaw: {text}"
        )

    edits = []
    for file_path, block in matches:
        file_path = file_path.strip()
        if not file_path:
            raise ValueError("Empty file path in <edit> block")
        search, replacement = _parse_edit_block(block, file_path)
        if not search:
            raise ValueError(f"Empty <search> block for file {file_path!r}")
        edits.append(PatchEdit(file_path=file_path, search=search, replacement=replacement))

    return edits


def validate_plan(plan: MetaPlan | PartPlan) -> list[str]:
    """Validate plan structure. Returns list of warnings (empty = valid).

    Checks:
    - All IDs unique
    - Dependency references point to valid IDs
    - No circular dependencies (Kahn's algorithm)
    - At least one part/step
    """
    warnings = []

    if isinstance(plan, MetaPlan):
        items = plan.parts
        label = "part"
    elif isinstance(plan, PartPlan):
        items = plan.steps
        label = "step"
    else:
        warnings.append(f"validate_plan received unexpected type: {type(plan).__name__}")
        return warnings

    if not items:
        warnings.append(f"Plan has no {label}s")
        return warnings

    # Check ID uniqueness
    ids = [item.id for item in items]
    id_set = set(ids)
    if len(ids) != len(id_set):
        seen = set()
        for i in ids:
            if i in seen:
                warnings.append(f"Duplicate {label} ID: {i!r}")
            seen.add(i)

    # Check dependency references
    for item in items:
        for dep in item.depends_on:
            if dep not in id_set:
                warnings.append(f"{label} {item.id!r} depends on unknown ID {dep!r}")

    # Check for cycles using Kahn's algorithm
    in_degree = {item.id: 0 for item in items}
    adjacency: dict[str, list[str]] = {item.id: [] for item in items}
    for item in items:
        for dep in item.depends_on:
            if dep in id_set:
                adjacency[dep].append(item.id)
                in_degree[item.id] += 1

    queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
    sorted_count = 0
    while queue:
        node = queue.popleft()
        sorted_count += 1
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if sorted_count != len(id_set):
        warnings.append(f"Circular dependency detected among {label}s")

    return warnings


def validate_part_grouping(grouping: PartGrouping, total_change_points: int) -> list[str]:
    """Validate that a PartGrouping covers all change points exactly once.

    Checks:
    - Every index in [0, total_change_points) assigned to exactly one part
    - No duplicate indices across parts
    - No out-of-range indices
    - Part IDs unique

    Returns list of warnings (empty = valid).
    """
    warnings = []

    # Check part ID uniqueness
    part_ids = [p.id for p in grouping.parts]
    if len(part_ids) != len(set(part_ids)):
        seen: set[str] = set()
        for pid in part_ids:
            if pid in seen:
                warnings.append(f"Duplicate part ID: {pid!r}")
            seen.add(pid)

    # Collect all assigned indices and check for duplicates / out-of-range
    all_indices: list[int] = []
    seen_indices: set[int] = set()
    for part in grouping.parts:
        for idx in part.change_point_indices:
            if idx < 0 or idx >= total_change_points:
                warnings.append(
                    f"Part {part.id!r} references out-of-range index {idx} "
                    f"(valid range: 0-{total_change_points - 1})"
                )
            if idx in seen_indices:
                warnings.append(
                    f"Index {idx} assigned to multiple parts (duplicate in {part.id!r})"
                )
            seen_indices.add(idx)
            all_indices.append(idx)

    # Check coverage: every index must be assigned
    expected = set(range(total_change_points))
    missing = expected - seen_indices
    if missing:
        warnings.append(f"Unassigned change point indices: {sorted(missing)}")

    return warnings
