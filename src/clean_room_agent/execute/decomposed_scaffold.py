"""Decomposed scaffold: multi-stage scaffold generation for smaller models.

Replaces single-call execute_scaffold() with three stages:
1. Interface enumeration — identify types + function signatures (LLM)
2. Per-file header generation — one LLM call per .h file
3. Deterministic stub generation — .c files from parsed headers (no LLM)

Output type (ScaffoldResult) is unchanged. The orchestrator sees no difference.

Unlike execute_scaffold() which returns edits for apply_edits() to write,
decomposed_scaffold() writes files directly to disk via repo_path. This
is because header and stub creation produces new files (not edits to
existing files), and PatchEdit/apply_edits require files to already exist.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from clean_room_agent.execute.dataclasses import (
    InterfaceEnumeration,
    PartPlan,
    ScaffoldResult,
)
from clean_room_agent.execute.parsers import parse_scaffold_response
from clean_room_agent.execute.prompts import (
    SYSTEM_PROMPTS,
    build_decomposed_scaffold_prompt,
)
from clean_room_agent.execute.scaffold import (
    extract_function_name,
    get_c_parser,
)
from clean_room_agent.llm.client import LoggedLLMClient
from clean_room_agent.retrieval.batch_judgment import run_binary_judgment
from clean_room_agent.retrieval.dataclasses import ContextPackage

logger = logging.getLogger(__name__)

# Matches ```c ... ``` or ``` ... ``` markdown code fences.
_CODE_FENCE_PATTERN = re.compile(
    r"^```[a-zA-Z]*\s*\n(.*?)\n```\s*$",
    re.DOTALL,
)


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences if the model wraps output in them.

    Models sometimes emit ```c\\n...\\n``` despite "no markdown fencing"
    instructions. This strips exactly one layer of fencing. If no fences
    are present, returns the text unchanged.
    """
    stripped = text.strip()
    m = _CODE_FENCE_PATTERN.match(stripped)
    if m:
        return m.group(1)
    return stripped


# ---------------------------------------------------------------------------
# Decomposed scaffold pipeline
# ---------------------------------------------------------------------------

def decomposed_scaffold(
    context: ContextPackage,
    part_plan: PartPlan,
    llm: LoggedLLMClient,
    *,
    repo_path: Path,
    cumulative_diff: str | None = None,
) -> ScaffoldResult:
    """Generate compilable C scaffold via three decomposed stages.

    1. Interface enumeration — identify all types and function signatures
    2. Per-file header generation — one LLM call per .h file
    3. Deterministic stub generation — .c files from parsed headers (no LLM)

    Writes generated files directly to disk under repo_path.
    Returns a ScaffoldResult with the files_written flag set.
    """
    task_description = f"Part: {part_plan.part_id}\nGoal: {part_plan.task_summary}"

    enum_result = _run_interface_enum(context, task_description, part_plan, llm, cumulative_diff)
    header_contents = _run_header_generation(enum_result, context, task_description, llm, cumulative_diff)
    stub_contents = _generate_deterministic_stubs(enum_result, header_contents)

    # Write all files to disk
    all_contents = {**header_contents, **stub_contents}
    for rel_path, content in all_contents.items():
        full_path = repo_path / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    return _assemble_scaffold_result(header_contents, stub_contents, enum_result)


def _run_interface_enum(
    context: ContextPackage,
    task_description: str,
    part_plan: PartPlan,
    llm: LoggedLLMClient,
    cumulative_diff: str | None,
) -> InterfaceEnumeration:
    """Stage 1: Enumerate all types and function signatures."""
    # Include step info in the task description
    steps_info = "\nSteps to scaffold:\n"
    for step in part_plan.steps:
        steps_info += f"  - {step.id}: {step.description}\n"
        if step.target_files:
            steps_info += f"    Target files: {', '.join(step.target_files)}\n"
        if step.target_symbols:
            steps_info += f"    Target symbols: {', '.join(step.target_symbols)}\n"

    full_description = task_description + steps_info

    system, user = build_decomposed_scaffold_prompt(
        context, full_description,
        pass_type="interface_enum",
        model_config=llm.config,
        cumulative_diff=cumulative_diff,
    )
    response = llm.complete(user, system=system)
    result = parse_scaffold_response(response.text, "interface_enum")
    return result


def _run_header_generation(
    enum_result: InterfaceEnumeration,
    context: ContextPackage,
    task_description: str,
    llm: LoggedLLMClient,
    cumulative_diff: str | None,
) -> dict[str, str]:
    """Stage 2: Generate one .h file per unique header path.

    Returns dict mapping header_path -> file content.
    """
    # Group types and functions by header file
    header_files: dict[str, dict] = {}
    for t in enum_result.types:
        header_files.setdefault(t.file_path, {"types": [], "functions": []})
        header_files[t.file_path]["types"].append(t.to_dict())

    for f in enum_result.functions:
        header_files.setdefault(f.file_path, {"types": [], "functions": []})
        header_files[f.file_path]["functions"].append(f.to_dict())

    header_contents: dict[str, str] = {}

    for header_path in sorted(header_files):
        spec = header_files[header_path]
        prior_output = json.dumps({
            "file_path": header_path,
            "includes": enum_result.includes,
            "types": spec["types"],
            "functions": spec["functions"],
        }, indent=2)

        system, user = build_decomposed_scaffold_prompt(
            context, task_description,
            pass_type="header_gen",
            model_config=llm.config,
            prior_stage_output=prior_output,
            cumulative_diff=cumulative_diff,
        )
        response = llm.complete(user, system=system)

        # Raw output — the LLM produces the header file content directly.
        # Strip markdown code fences if the model wraps it despite instructions.
        content = _strip_code_fences(response.text)
        if not content.strip():
            raise ValueError(
                f"Header generation for {header_path} produced empty content"
            )
        header_contents[header_path] = content

    return header_contents


def _generate_deterministic_stubs(
    enum_result: InterfaceEnumeration,
    header_contents: dict[str, str],
) -> dict[str, str]:
    """Stage 3: Generate .c stub files deterministically from headers.

    Parses header content using tree-sitter-c to find function declarations,
    then generates .c files with #include + stub bodies.
    Zero LLM calls.

    Returns dict mapping source_path -> file content.
    """
    parser = get_c_parser()

    # Build map of function_name -> source_file from enum_result
    func_source_map: dict[str, str] = {}
    for func in enum_result.functions:
        func_source_map[func.name] = func.source_file

    # For each header, parse declarations and generate corresponding .c stubs
    # Group stubs by source file
    source_stubs: dict[str, list[str]] = {}  # source_path -> [stub_lines]
    source_includes: dict[str, set[str]] = {}  # source_path -> {header_includes}

    for header_path, content in header_contents.items():
        tree = parser.parse(content.encode("utf-8"))
        declarations = _extract_declarations(tree.root_node, content.encode("utf-8"))

        for decl_name, decl_signature, return_type in declarations:
            # Determine which .c file this function belongs in
            source_file = func_source_map.get(decl_name)
            if source_file is None:
                # Default: same basename as header
                source_file = header_path[:-2] + ".c" if header_path.endswith(".h") else header_path + ".c"

            source_stubs.setdefault(source_file, [])
            source_includes.setdefault(source_file, set())

            # Add include for this header
            header_basename = header_path.rsplit("/", 1)[-1] if "/" in header_path else header_path
            source_includes[source_file].add(header_basename)

            # Generate stub body with appropriate return value
            stub_return = _stub_return_for_type(return_type)
            stub_body = f"{decl_signature} {{\n    {stub_return}\n}}"
            source_stubs[source_file].append(stub_body)

    # Also add system includes from enum_result
    system_includes = [f"#include <{inc}>" for inc in sorted(enum_result.includes)]

    # Generate content for each .c file
    stub_contents: dict[str, str] = {}
    for source_path in sorted(source_stubs):
        parts: list[str] = []

        # System includes
        if system_includes:
            parts.extend(system_includes)

        # Local includes
        for inc in sorted(source_includes.get(source_path, set())):
            parts.append(f'#include "{inc}"')

        parts.append("")  # blank line

        # Function stubs
        parts.append("\n\n".join(source_stubs[source_path]))
        parts.append("")  # trailing newline

        stub_contents[source_path] = "\n".join(parts)

    return stub_contents


def _extract_declarations(
    root_node, source_bytes: bytes,
) -> list[tuple[str, str, str]]:
    """Extract function declarations from a tree-sitter-c parsed header.

    Recursively searches through preprocessor blocks (#ifndef/#endif)
    to find all function declarations.

    Returns list of (func_name, full_signature, return_type) for each
    function declaration (not function definition).
    """
    declarations: list[tuple[str, str, str]] = []
    _collect_declarations(root_node, source_bytes, declarations)
    return declarations


def _collect_declarations(
    node, source_bytes: bytes,
    declarations: list[tuple[str, str, str]],
) -> None:
    """Recursively collect function declarations from all tree nodes."""
    if node.type == "declaration":
        # Check if this declaration contains a function declarator
        declarator = node.child_by_field_name("declarator")
        if declarator is not None:
            func_declarator = _find_function_declarator(declarator)
            if func_declarator is not None:
                func_name = extract_function_name(func_declarator, source_bytes)
                if func_name is not None:
                    # Extract return type
                    type_node = node.child_by_field_name("type")
                    return_type = ""
                    if type_node is not None:
                        return_type = source_bytes[type_node.start_byte:type_node.end_byte].decode("utf-8")

                    # Check for pointer in declarator
                    if declarator.type == "pointer_declarator":
                        return_type += " *"

                    # Build full signature: everything except trailing semicolon
                    full_text = source_bytes[node.start_byte:node.end_byte].decode("utf-8").rstrip()
                    if full_text.endswith(";"):
                        full_text = full_text[:-1].rstrip()

                    declarations.append((func_name, full_text, return_type))
        return  # Don't recurse into declaration children

    # Recurse into children (handles preproc_ifdef, preproc_if, etc.)
    for child in node.children:
        _collect_declarations(child, source_bytes, declarations)


def _find_function_declarator(node):
    """Recursively find a function_declarator node."""
    if node.type == "function_declarator":
        return node
    for child in node.children:
        result = _find_function_declarator(child)
        if result is not None:
            return result
    return None


def _stub_return_for_type(return_type: str) -> str:
    """Generate appropriate stub return statement for a C return type."""
    rt = return_type.strip()
    if rt == "void":
        return "return;"
    if "*" in rt:
        return "return NULL;"
    if rt in ("float", "double"):
        return "return 0.0;"
    if rt == "bool" or rt == "_Bool":
        return "return false;"
    # Default for int, size_t, etc.
    return "return 0;"


def _assemble_scaffold_result(
    header_contents: dict[str, str],
    stub_contents: dict[str, str],
    enum_result: InterfaceEnumeration,
) -> ScaffoldResult:
    """Combine header and stub info into a ScaffoldResult.

    Files are already written to disk by decomposed_scaffold().
    The ScaffoldResult contains empty edits (files were written directly)
    but populated header_files/source_files for downstream use.
    """
    header_files = sorted(header_contents.keys())
    source_files = sorted(stub_contents.keys())

    # Build raw_response summary from enum_result for logging
    raw_response = json.dumps({
        "decomposed": True,
        "interface_enum": enum_result.to_dict(),
        "header_count": len(header_files),
        "source_count": len(source_files),
    })

    return ScaffoldResult(
        success=True,
        edits=[],  # Files written directly, not via apply_edits
        header_files=header_files,
        source_files=source_files,
        raw_response=raw_response,
    )


# ---------------------------------------------------------------------------
# KB pattern selection
# ---------------------------------------------------------------------------

def select_kb_patterns_for_function(
    stub,
    context: ContextPackage,
    llm: LoggedLLMClient,
) -> list[str]:
    """Select relevant KB patterns for implementing a specific function.

    Filters ContextPackage files for KB paths (prefix 'kb/'), then uses
    binary judgment to select relevant sections.

    Returns content strings for sections judged relevant. Returns [] if
    no KB files are in the context.
    """
    # Filter KB files from context
    kb_files = [f for f in context.files if f.path.startswith("kb/")]
    if not kb_files:
        return []

    system_prompt = SYSTEM_PROMPTS["kb_pattern_relevance"]
    task_context = (
        f"Function to implement:\n"
        f"Name: {stub.name}\n"
        f"Signature: {stub.signature}\n"
    )
    if stub.docstring:
        task_context += f"Contract: {stub.docstring}\n"

    def format_kb_file(kb_file):
        return (
            f"\nKB Section: {kb_file.path}\n"
            f"Content (first 500 chars):\n{kb_file.content[:500]}\n"
            f"Is this section relevant to implementing the function above?"
        )

    def kb_key(kb_file):
        return kb_file.path

    verdict_map, _omitted = run_binary_judgment(
        kb_files,
        system_prompt=system_prompt,
        task_context=task_context,
        llm=llm,
        format_item=format_kb_file,
        stage_name="kb_pattern_relevance",
        item_key=kb_key,
        default_action="exclude_pattern",
    )

    # Return full content for sections judged relevant
    selected: list[str] = []
    for kb_file in kb_files:
        if verdict_map.get(kb_file.path, False):
            selected.append(kb_file.content)

    return selected
