"""Scaffold-then-implement for C projects.

Generates compilable C scaffolds (headers + stubs), validates compilation,
then implements each function independently against the compact scaffold context.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from clean_room_agent.execute.dataclasses import (
    FunctionStub,
    PartPlan,
    PatchEdit,
    ScaffoldResult,
    StepResult,
)
from clean_room_agent.execute.parsers import parse_implement_response
from clean_room_agent.execute.prompts import (
    build_function_implement_prompt,
    build_scaffold_prompt,
)
from clean_room_agent.llm.client import LoggedLLMClient
from clean_room_agent.retrieval.dataclasses import ContextPackage

logger = logging.getLogger(__name__)

# Stub body patterns recognized by extract_function_stubs().
# These match the minimal valid return statements produced by scaffold generation.
_STUB_BODY_PATTERNS = re.compile(
    r"^\s*\{\s*"
    r"(?:/\*.*?\*/\s*)?"  # optional /* comment */
    r"return\s+(?:0|NULL|false|-1|0\.0f?)\s*;"
    r"\s*\}$",
    re.DOTALL,
)


def execute_scaffold(
    context: ContextPackage,
    part_plan: PartPlan,
    llm: LoggedLLMClient,
    *,
    cumulative_diff: str | None = None,
) -> ScaffoldResult:
    """Generate compilable C scaffold from part plan.

    Builds prompt from ContextPackage + all steps in PartPlan, sends to LLM,
    parses XML edit blocks. Does NOT apply edits or validate compilation —
    the caller handles that.

    Raises ValueError on parse failure — caller handles logging to raw DB.
    """
    system, user = build_scaffold_prompt(
        context, part_plan,
        model_config=llm.config,
        cumulative_diff=cumulative_diff,
    )

    response = llm.complete(user, system=system)

    edits = parse_implement_response(response.text)

    header_files = sorted({e.file_path for e in edits if e.file_path.endswith(".h")})
    source_files = sorted({e.file_path for e in edits if e.file_path.endswith(".c")})

    return ScaffoldResult(
        success=True,
        edits=edits,
        header_files=header_files,
        source_files=source_files,
        raw_response=response.text,
    )


def validate_scaffold_compilation(
    files: list[str],
    repo_path: Path,
    *,
    compiler: str = "gcc",
    flags: str = "-c -fsyntax-only -Wall",
) -> tuple[bool, str]:
    """Run compiler on scaffolded .c files. Returns (success, output).

    Only compiles .c files (headers are included transitively).
    Uses syntax-only check — no object files produced.
    """
    c_files = [f for f in files if f.endswith(".c")]
    if not c_files:
        return True, ""

    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise RuntimeError(
            f"Scaffold compiler {compiler!r} not found on PATH. "
            f"Install it or set scaffold_enabled = false in config."
        )

    all_output: list[str] = []
    all_success = True

    for c_file in c_files:
        full_path = repo_path / c_file
        if not full_path.exists():
            all_output.append(f"{c_file}: file not found after scaffold edits")
            all_success = False
            continue

        cmd = [compiler_path] + flags.split() + [str(full_path)]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(repo_path),
            )
            if result.returncode != 0:
                all_success = False
                all_output.append(f"{c_file}:\n{result.stderr}")
            else:
                if result.stderr.strip():
                    all_output.append(f"{c_file} (warnings):\n{result.stderr}")
        except subprocess.TimeoutExpired:
            all_success = False
            all_output.append(f"{c_file}: compilation timed out")

    return all_success, "\n".join(all_output)


def extract_function_stubs(
    source_files: list[str],
    repo_path: Path,
) -> list[FunctionStub]:
    """Extract function stubs from scaffolded .c files using tree-sitter-c.

    Finds all function definitions with stub bodies (minimal return statements).
    Returns FunctionStub objects with line ranges for search/replace targeting.

    Requires tree-sitter-c grammar. Raises ImportError if not installed.
    """
    try:
        import tree_sitter_c as tsc
        from tree_sitter import Language, Parser
    except ImportError as e:
        raise ImportError(
            "tree-sitter-c is required for scaffold stub extraction. "
            "Install with: pip install tree-sitter-c"
        ) from e

    parser = Parser(Language(tsc.language()))
    stubs: list[FunctionStub] = []

    for source_file in source_files:
        full_path = repo_path / source_file
        if not full_path.exists():
            logger.warning("Scaffold source file not found: %s", source_file)
            continue

        source_bytes = full_path.read_bytes()
        tree = parser.parse(source_bytes)

        # Find the corresponding header file (same basename, .h extension)
        header_file = _find_header_for_source(source_file)

        for node in _iter_function_definitions(tree.root_node):
            body_node = node.child_by_field_name("body")
            if body_node is None:
                continue

            body_text = source_bytes[body_node.start_byte:body_node.end_byte].decode("utf-8", errors="replace")
            if not _STUB_BODY_PATTERNS.match(body_text):
                continue

            # Extract function name from declarator
            declarator = node.child_by_field_name("declarator")
            func_name = _extract_function_name(declarator, source_bytes)
            if not func_name:
                continue

            # Extract full signature (everything before the body)
            signature_bytes = source_bytes[node.start_byte:body_node.start_byte].rstrip()
            signature = signature_bytes.decode("utf-8", errors="replace").strip()

            # Extract docstring (comment immediately before the function)
            docstring = _extract_preceding_comment(node, source_bytes)

            # Extract dependencies from body comments (/* calls: foo, bar */)
            dependencies = _extract_stub_dependencies(body_text)

            stubs.append(FunctionStub(
                name=func_name,
                file_path=source_file,
                signature=signature,
                docstring=docstring,
                start_line=node.start_point[0] + 1,  # 1-indexed
                end_line=node.end_point[0] + 1,
                dependencies=dependencies,
                header_file=header_file,
            ))

    return stubs


def execute_function_implement(
    stub: FunctionStub,
    scaffold_content: dict[str, str],
    llm: LoggedLLMClient,
) -> StepResult:
    """Implement one function body from its scaffold stub.

    Context per call: header file(s) + source file containing the stub +
    the stub's docstring. Much smaller context than full ContextPackage.

    Raises ValueError on parse failure — caller handles logging to raw DB.
    """
    system, user = build_function_implement_prompt(
        stub, scaffold_content,
        model_config=llm.config,
    )

    response = llm.complete(user, system=system)

    edits = parse_implement_response(response.text)
    return StepResult(
        success=True,
        edits=edits,
        raw_response=response.text,
    )


def is_c_part(part_plan: PartPlan) -> bool:
    """Check if a part plan targets C files (per-part language detection).

    Returns True if all target files across all steps have .c or .h extensions.
    Returns False if no target files are specified.
    """
    all_targets: list[str] = []
    for step in part_plan.steps:
        all_targets.extend(step.target_files)

    if not all_targets:
        return False

    return all(
        f.endswith(".c") or f.endswith(".h")
        for f in all_targets
    )


# -- Internal helpers --


def _find_header_for_source(source_file: str) -> str | None:
    """Find header file corresponding to a .c source file."""
    if source_file.endswith(".c"):
        header = source_file[:-2] + ".h"
        return header
    return None


def _iter_function_definitions(node):
    """Yield all function_definition nodes in the tree."""
    if node.type == "function_definition":
        yield node
    for child in node.children:
        yield from _iter_function_definitions(child)


def _extract_function_name(declarator, source_bytes: bytes) -> str | None:
    """Extract the function name from a declarator node."""
    if declarator is None:
        return None
    # Walk down to find the identifier within the declarator
    # (handles pointer declarators, nested declarators, etc.)
    node = declarator
    while node is not None:
        if node.type == "identifier":
            return source_bytes[node.start_byte:node.end_byte].decode("utf-8")
        # For function declarators, the name is in the declarator child
        child = node.child_by_field_name("declarator")
        if child is not None:
            node = child
        else:
            # Fallback: find first identifier child
            for c in node.children:
                if c.type == "identifier":
                    return source_bytes[c.start_byte:c.end_byte].decode("utf-8")
            break
    return None


def _extract_preceding_comment(func_node, source_bytes: bytes) -> str:
    """Extract comment block immediately preceding a function definition."""
    # Look at the previous sibling(s) for comment nodes
    prev = func_node.prev_sibling
    comments: list[str] = []
    while prev is not None and prev.type == "comment":
        text = source_bytes[prev.start_byte:prev.end_byte].decode("utf-8", errors="replace")
        comments.insert(0, text)
        prev = prev.prev_sibling

    return "\n".join(comments)


def _extract_stub_dependencies(body_text: str) -> list[str]:
    """Extract dependency hints from stub body comments.

    Looks for patterns like: /* calls: foo, bar, baz */
    """
    match = re.search(r"/\*\s*calls?:\s*([^*]+)\*/", body_text)
    if match:
        names = [n.strip() for n in match.group(1).split(",")]
        return [n for n in names if n]
    return []
