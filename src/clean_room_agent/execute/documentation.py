"""Documentation pass: LLM-driven docstring and comment improvement with AST verification."""

from __future__ import annotations

import logging
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from clean_room_agent.execute.dataclasses import PatchResult, StepResult
from clean_room_agent.execute.parsers import parse_implement_response
from clean_room_agent.execute.patch import apply_edits
from clean_room_agent.execute.prompts import build_documentation_prompt
from clean_room_agent.llm.client import LoggedLLMClient, ModelConfig

logger = logging.getLogger(__name__)

PY_LANGUAGE = Language(tspython.language())

# Extension to language mapping for supported languages
_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
}


def _get_doc_byte_ranges(
    tree, source_bytes: bytes, language: str,
) -> list[tuple[int, int]]:
    """Return sorted, non-overlapping byte ranges of all documentation content.

    For Python: comment nodes + docstring nodes (expression_statement > string
    as first statement in function/class/module body).
    """
    if language != "python":
        raise ValueError(f"Unsupported language for doc range extraction: {language!r}")

    ranges: list[tuple[int, int]] = []

    def _walk(node):
        for child in node.children:
            # Comment nodes
            if child.type == "comment":
                ranges.append((child.start_byte, child.end_byte))

            # Module-level docstring: first expression_statement with string
            # Function/class docstring: first expression_statement in body
            elif child.type == "expression_statement":
                if _is_docstring(child, node):
                    ranges.append((child.start_byte, child.end_byte))

            # Recurse into compound structures
            if child.type in (
                "module", "block", "function_definition", "class_definition",
                "decorated_definition", "if_statement", "try_statement",
                "else_clause", "except_clause", "finally_clause",
                "for_statement", "while_statement", "with_statement",
            ):
                _walk(child)

    _walk(tree.root_node)

    # Sort and merge overlapping ranges
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


def _is_docstring(expr_stmt_node, parent_node) -> bool:
    """Check if an expression_statement is a docstring.

    A docstring is a string literal as the first non-comment statement in:
    - module body (parent is module)
    - function/class body (parent is block, grandparent is function/class)
    """
    # Must contain a string or concatenated_string child
    has_string = False
    for child in expr_stmt_node.children:
        if child.type in ("string", "concatenated_string"):
            has_string = True
            break
    if not has_string:
        return False

    # Check it's the first non-comment statement in its parent
    for sibling in parent_node.children:
        if sibling.type == "comment":
            continue
        if sibling.type == "newline":
            continue
        # Skip non-statement nodes (e.g. "def", "class", keywords, etc.)
        if not sibling.is_named or sibling.type in (
            "identifier", "parameters", "type", "return_type",
            "decorator", ":", "block",
        ):
            continue
        return sibling is expr_stmt_node

    return False


def _mask_doc_ranges(
    source_bytes: bytes, ranges: list[tuple[int, int]],
) -> str:
    """Replace doc byte ranges with spaces (preserving newlines), then normalize."""
    result = bytearray(source_bytes)
    for start, end in ranges:
        for i in range(start, end):
            if result[i] != ord(b"\n") and result[i] != ord(b"\r"):
                result[i] = ord(b" ")

    text = result.decode("utf-8", errors="replace")
    # Normalize: strip trailing whitespace per line, drop blank-only lines
    lines = [line.rstrip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def verify_doc_only_edits(original: str, edited: str, language: str) -> bool:
    """Verify that edits only changed documentation (docstrings + comments).

    Parses both versions with tree-sitter, masks out doc byte ranges,
    and compares the remaining code-only content. Returns True if the
    code-only content is identical.

    Only Python is supported. For unsupported languages, returns False
    (default-deny per R2).
    """
    if language not in _EXTENSION_TO_LANGUAGE.values():
        return False

    parser = Parser(PY_LANGUAGE)

    original_bytes = original.encode("utf-8")
    edited_bytes = edited.encode("utf-8")

    original_tree = parser.parse(original_bytes)
    edited_tree = parser.parse(edited_bytes)

    original_ranges = _get_doc_byte_ranges(original_tree, original_bytes, language)
    edited_ranges = _get_doc_byte_ranges(edited_tree, edited_bytes, language)

    original_code = _mask_doc_ranges(original_bytes, original_ranges)
    edited_code = _mask_doc_ranges(edited_bytes, edited_ranges)

    return original_code == edited_code


def _language_from_extension(file_path: str) -> str | None:
    """Map file extension to supported language, or None if unsupported."""
    for ext, lang in _EXTENSION_TO_LANGUAGE.items():
        if file_path.endswith(ext):
            return lang
    return None


def _apply_edits_to_string(original: str, edits) -> str | None:
    """Apply PatchEdit search/replace operations to a string. Returns None on failure."""
    content = original
    for edit in edits:
        count = content.count(edit.search)
        if count != 1:
            return None
        content = content.replace(edit.search, edit.replacement, 1)
    return content


def execute_documentation_file(
    file_path: str,
    repo_path: Path,
    task_description: str,
    part_description: str,
    llm: LoggedLLMClient,
    model_config: ModelConfig,
    environment_brief: str | None = None,
) -> StepResult:
    """Run the documentation pass on a single file.

    Returns StepResult with success=True and edits if documentation was
    improved, or success=True with empty edits for unsupported languages.
    Returns success=False if parsing or verification fails.
    """
    language = _language_from_extension(file_path)
    if language is None:
        logger.warning("Documentation pass: unsupported language for %s, skipping", file_path)
        return StepResult(success=True, edits=[])

    full_path = repo_path / file_path
    file_content = full_path.read_text(encoding="utf-8")

    system, user = build_documentation_prompt(
        file_content, file_path, task_description, part_description,
        model_config, environment_brief,
    )

    response = llm.complete(user, system=system)

    # Parse edit blocks from response
    try:
        edits = parse_implement_response(response.text)
    except ValueError:
        # No edit blocks means LLM decided no changes needed — that's fine
        if "<edit" not in response.text:
            return StepResult(success=True, edits=[], raw_response=response.text)
        logger.warning("Documentation pass: failed to parse response for %s", file_path)
        return StepResult(
            success=False, edits=[],
            error_info=f"Failed to parse documentation edits for {file_path}",
            raw_response=response.text,
        )

    # Apply edits hypothetically to check AST constraint
    edited_content = _apply_edits_to_string(file_content, edits)
    if edited_content is None:
        logger.warning("Documentation pass: edit application failed for %s", file_path)
        return StepResult(
            success=False, edits=[],
            error_info=f"Edit search strings did not match file content for {file_path}",
            raw_response=response.text,
        )

    if not verify_doc_only_edits(file_content, edited_content, language):
        logger.warning("Documentation pass: edits modify code logic for %s", file_path)
        return StepResult(
            success=False, edits=[],
            error_info="Edits modify code logic",
            raw_response=response.text,
        )

    return StepResult(success=True, edits=edits, raw_response=response.text)


def run_documentation_pass(
    modified_files: list[str],
    repo_path: Path,
    task_description: str,
    part_description: str,
    llm: LoggedLLMClient,
    model_config: ModelConfig,
    environment_brief: str | None = None,
) -> list[PatchResult]:
    """Run the documentation pass on all modified files.

    For each unique file, calls execute_documentation_file and applies
    validated edits. Files that fail verification or parsing are skipped
    (logged as warnings).

    Returns list of PatchResults for files that had edits applied.
    """
    doc_patch_results: list[PatchResult] = []

    for file_path in modified_files:
        step_result = execute_documentation_file(
            file_path, repo_path, task_description, part_description,
            llm, model_config, environment_brief,
        )

        if not step_result.success:
            logger.warning(
                "Documentation pass failed for %s: %s",
                file_path, step_result.error_info,
            )
            continue

        if not step_result.edits:
            continue

        patch_result = apply_edits(step_result.edits, repo_path)
        if patch_result.success:
            doc_patch_results.append(patch_result)
        else:
            logger.warning(
                "Documentation pass: failed to apply edits for %s: %s",
                file_path, patch_result.error_info,
            )

    return doc_patch_results
