"""Context assembly: build ContextPackage from stage outputs + disk reads."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from clean_room_agent.retrieval.budget import (
    BudgetTracker,
    estimate_framing_tokens,
    estimate_tokens,
)
from clean_room_agent.token_estimation import check_prompt_budget
from clean_room_agent.retrieval.dataclasses import (
    BudgetConfig,
    ContextPackage,
    FileContent,
)
from clean_room_agent.retrieval.stage import StageContext
from clean_room_agent.retrieval.utils import parse_json_response

if TYPE_CHECKING:
    from clean_room_agent.llm.client import LLMClient
    from clean_room_agent.query.api import KnowledgeBase

logger = logging.getLogger(__name__)

# Detail level priority: lower = higher priority
_DETAIL_PRIORITY = {"primary": 0, "supporting": 1, "type_context": 2}

REFILTER_SYSTEM = (
    "You are Jane, a context budget optimizer. Given a list of files with their sizes and "
    "detail levels, select the subset that fits within the given token budget. "
    "Prioritize primary files, then supporting, then type_context. "
    "Respond with a JSON array of file paths to KEEP: [\"path1.py\", \"path2.py\"]. "
    "Respond with ONLY the JSON array, no markdown fencing or extra text."
)


def assemble_context(
    context: StageContext,
    budget: BudgetConfig,
    repo_path: Path,
    llm: LLMClient | None = None,
    kb: "KnowledgeBase | None" = None,
) -> ContextPackage:
    """Build a budget-compliant ContextPackage from stage outputs.

    Reads file content from disk. Renders at classified detail levels.

    R1 compliance: if assembled content exceeds budget:
    - With LLM: call _refilter_files() to let LLM choose subset to keep.
    - Without LLM: drop entire files by priority (type_context first, then
      supporting, then primary). No within-level downgrades.
    """
    tracker = BudgetTracker(budget)
    assembly_decisions: list[dict] = []

    # R5: account for task/intent header overhead before file assembly
    task_header = f"# Task\n{context.task.raw_task}\n"
    intent_header = f"# Intent\n{context.task.intent_summary}\n" if context.task.intent_summary else ""
    tracker.consume(estimate_tokens(task_header + intent_header))

    # Phase 1: classify files by detail level
    file_detail, file_symbols, file_info, sorted_fids, classify_decisions = (
        _classify_file_inclusions(context)
    )
    assembly_decisions.extend(classify_decisions)

    # Phase 2: read from disk and render at classified levels
    rendered_files, read_decisions = _read_and_render_files(
        sorted_fids, file_info, file_detail, repo_path, context, kb,
    )
    assembly_decisions.extend(read_decisions)

    # Phase 3: trim to budget (refilter, priority drop, group integrity)
    rendered_files, trim_decisions = _trim_to_budget(
        rendered_files, tracker, context, llm,
    )
    assembly_decisions.extend(trim_decisions)

    # Phase 4: build final file contents, consuming budget
    file_contents: list[FileContent] = []
    for rf in rendered_files:
        if not tracker.can_fit(rf["tokens"]):
            logger.warning(
                "R1: dropping '%s' — cannot fit %d tokens in remaining %d budget",
                rf["path"], rf["tokens"], tracker.remaining,
            )
            assembly_decisions.append({
                "file_id": rf["file_id"], "included": False,
                "reason": f"budget_overflow: {rf['tokens']} tokens exceeds remaining {tracker.remaining}",
            })
            continue

        tracker.consume(rf["tokens"])
        file_contents.append(FileContent(
            file_id=rf["file_id"],
            path=rf["path"],
            language=rf["language"],
            content=rf["rendered"],
            token_estimate=rf["tokens"],
            detail_level=rf["detail"],
            included_symbols=file_symbols[rf["file_id"]],
            metadata_summary=rf["metadata_summary"],
        ))

    for fc in file_contents:
        assembly_decisions.append({
            "file_id": fc.file_id, "included": True,
            "reason": f"included: detail={fc.detail_level}, tokens={fc.token_estimate}",
        })

    # T2-11: debug validation of framing token estimates vs actual rendering
    if os.environ.get("CRA_DEBUG_BUDGET"):
        _validate_framing_estimates(file_contents)

    return ContextPackage(
        task=context.task,
        files=file_contents,
        total_token_estimate=tracker.used,
        budget=budget,
        metadata={
            "files_considered": len(file_info),
            "files_included": len(file_contents),
            "budget_remaining": tracker.remaining,
            "assembly_decisions": assembly_decisions,
        },
    )


def _validate_framing_estimates(file_contents: list[FileContent]) -> None:
    """Debug-only: compare estimated framing tokens vs actual rendered framing.

    Enabled by CRA_DEBUG_BUDGET env var. Logs warnings for mismatches.
    T2-11: ensures R5 (framing overhead is part of the budget) holds in practice.
    """
    for fc in file_contents:
        estimated = estimate_framing_tokens(
            fc.path, fc.language, fc.detail_level,
            has_metadata=bool(fc.metadata_summary),
        )
        # Render just the framing portion (header + code fence markers)
        actual_framing = _count_actual_framing_tokens(fc)
        if abs(estimated - actual_framing) > 5:
            logger.warning(
                "T2-11 framing mismatch for %s: estimated=%d actual=%d (delta=%d)",
                fc.path, estimated, actual_framing, estimated - actual_framing,
            )


def _count_actual_framing_tokens(fc: FileContent) -> int:
    """Count tokens consumed by framing (headers, code fences) for a rendered file.

    Subtracts the raw content tokens from the total rendered tokens to isolate
    the framing overhead.
    """
    # Build the framing text without content
    framing_parts = [f"## {fc.path} [{fc.detail_level}]"]
    if fc.metadata_summary:
        framing_parts.append(fc.metadata_summary)
    framing_parts.append(f"```{fc.language}")
    framing_parts.append("```")
    framing_text = "\n".join(framing_parts) + "\n"
    return estimate_tokens(framing_text)


def _classify_file_inclusions(
    context: StageContext,
) -> tuple[dict[int, str], dict[int, list[str]], dict[int, dict], list[int], list[dict]]:
    """Classify files by detail level from classified symbols and scoped files.

    Returns:
        (file_detail, file_symbols, file_info, sorted_fids, assembly_decisions)
    """
    file_detail: dict[int, str] = {}
    file_symbols: dict[int, list[str]] = {}
    decisions: list[dict] = []

    for cs in context.classified_symbols:
        if cs.detail_level == "excluded":
            continue
        current = file_detail.get(cs.file_id)
        if current is None or _DETAIL_PRIORITY[cs.detail_level] < _DETAIL_PRIORITY[current]:
            file_detail[cs.file_id] = cs.detail_level
        file_symbols.setdefault(cs.file_id, []).append(cs.name)

    # R2: files with no classified symbols → default exclude
    for fid in context.included_file_ids:
        if fid not in file_detail:
            logger.warning("R2: file_id=%d has no classified symbols — excluding from context", fid)
            decisions.append({
                "file_id": fid, "included": False,
                "reason": "R2: no classified symbols — default exclude",
            })

    file_info: dict[int, dict] = {}
    for sf in context.scoped_files:
        if sf.file_id in file_detail and sf.relevance == "relevant":
            file_info[sf.file_id] = {
                "path": sf.path,
                "language": sf.language,
                "tier": sf.tier,
            }

    sorted_fids = sorted(
        file_info.keys(),
        key=lambda fid: (
            _DETAIL_PRIORITY[file_detail[fid]],
            file_info[fid]["tier"],
        ),
    )

    return file_detail, file_symbols, file_info, sorted_fids, decisions


def _read_and_render_files(
    sorted_fids: list[int],
    file_info: dict[int, dict],
    file_detail: dict[int, str],
    repo_path: Path,
    context: StageContext,
    kb: "KnowledgeBase | None",
) -> tuple[list[dict], list[dict]]:
    """Read files from disk and render at classified detail levels.

    Raises RuntimeError for unreadable primary files (R1).
    Returns (rendered_files, decisions).
    """
    # Batch-fetch metadata for all files
    metadata_map = {}
    if kb is not None:
        metadata_map = kb.get_file_metadata_batch(list(sorted_fids))

    rendered_files: list[dict] = []
    decisions: list[dict] = []
    for fid in sorted_fids:
        info = file_info[fid]
        if fid not in file_detail:
            logger.warning("R2: file %d has no detail classification — excluding", fid)
            decisions.append({
                "file_id": fid, "included": False,
                "reason": "R2: no detail classification — default exclude",
            })
            continue
        detail = file_detail[fid]

        # Knowledge base files: read content from ref_sections table, not disk
        source = _read_file_source(fid, info, detail, repo_path, kb)
        rendered = _render_at_level(source, detail, fid, context, kb=kb)
        content_tokens = estimate_tokens(rendered)
        # Build metadata summary from enrichment data (before framing estimate)
        metadata_summary = ""
        meta = metadata_map.get(fid)
        if meta:
            parts = []
            if meta.purpose:
                parts.append(f"Purpose: {meta.purpose}")
            if meta.domain:
                parts.append(f"Domain: {meta.domain}")
            if parts:
                metadata_summary = " | ".join(parts)

        # R5: account for metadata_summary tokens and framing overhead in budget
        meta_tokens = estimate_tokens(metadata_summary) if metadata_summary else 0
        framing_tokens = estimate_framing_tokens(
            info["path"], info["language"], detail,
            has_metadata=bool(metadata_summary),
        )

        rendered_files.append({
            "file_id": fid,
            "path": info["path"],
            "language": info["language"],
            "detail": detail,
            "rendered": rendered,
            "tokens": content_tokens + framing_tokens + meta_tokens,
            "metadata_summary": metadata_summary,
        })

    return rendered_files, decisions


def _trim_to_budget(
    rendered_files: list[dict],
    tracker: BudgetTracker,
    context: StageContext,
    llm: LLMClient | None,
) -> tuple[list[dict], list[dict]]:
    """Trim rendered files to fit within the budget tracker's remaining capacity.

    Applies refilter (LLM) or priority drop, then group integrity enforcement.
    Returns (trimmed_files, assembly_decisions).
    """
    decisions: list[dict] = []
    total_tokens = sum(rf["tokens"] for rf in rendered_files)

    if total_tokens > tracker.remaining:
        files_before = list(rendered_files)
        if llm is not None:
            keep_paths = _refilter_files(rendered_files, tracker.remaining, context, llm)
            rendered_files = [rf for rf in rendered_files if rf["path"] in keep_paths]
            for rf in files_before:
                if rf["path"] not in keep_paths:
                    decisions.append({
                        "file_id": rf["file_id"], "included": False,
                        "reason": f"refilter: LLM excluded (detail={rf['detail']}, tokens={rf['tokens']})",
                    })
        else:
            logger.warning("R1: budget exceeded (%d > %d) without LLM — dropping files by priority",
                           total_tokens, tracker.remaining)
            rendered_files = _drop_by_priority(rendered_files, tracker.remaining)
            kept_ids = {rf["file_id"] for rf in rendered_files}
            for rf in files_before:
                if rf["file_id"] not in kept_ids:
                    decisions.append({
                        "file_id": rf["file_id"], "included": False,
                        "reason": f"priority_drop: budget exceeded (detail={rf['detail']}, tokens={rf['tokens']})",
                    })

    # Group integrity: partial groups → drop entire group (R2 default-deny)
    rendered_files, group_drops = _enforce_group_integrity(rendered_files, context.classified_symbols)
    for drop in group_drops:
        decisions.append({
            "file_id": drop["file_id"], "included": False,
            "reason": drop["reason"],
        })

    return rendered_files, decisions


def _enforce_group_integrity(
    rendered_files: list[dict],
    classified_symbols: list,
) -> tuple[list[dict], list[dict]]:
    """Enforce group integrity: if a group is partially included, drop the entire group.

    R2 default-deny: partial groups cannot provide the full context needed for
    dedup/extract-pattern work.

    Returns (filtered_files, drop_decisions).
    """
    from clean_room_agent.retrieval.dataclasses import ClassifiedSymbol

    # Build group → required file_ids mapping
    group_files: dict[str, set[int]] = {}
    for cs in classified_symbols:
        if isinstance(cs, ClassifiedSymbol) and cs.group_id is not None:
            group_files.setdefault(cs.group_id, set()).add(cs.file_id)

    if not group_files:
        return rendered_files, []

    included_file_ids = {rf["file_id"] for rf in rendered_files}

    # Find groups that are partially included
    drop_file_ids: set[int] = set()
    for group_id, required_fids in group_files.items():
        present = required_fids & included_file_ids
        if present and present != required_fids:
            # Partially included → drop entire group
            logger.warning(
                "R2: group %s partially included (%d/%d files) — dropping all group files",
                group_id, len(present), len(required_fids),
            )
            drop_file_ids |= present

    if not drop_file_ids:
        return rendered_files, []

    drops: list[dict] = []
    for rf in rendered_files:
        if rf["file_id"] in drop_file_ids:
            drops.append({
                "file_id": rf["file_id"],
                "reason": "group_integrity: partial group dropped (R2 default-deny)",
            })

    filtered = [rf for rf in rendered_files if rf["file_id"] not in drop_file_ids]
    return filtered, drops


def _require_logged_client(llm, caller: str) -> None:
    """Enforce LoggedLLMClient at runtime so LLM I/O logging cannot be forgotten."""
    if not hasattr(llm, "flush"):
        raise TypeError(
            f"{caller} requires a logging-capable LLM client (with flush()), "
            f"got {type(llm).__name__}"
        )


def _refilter_files(
    rendered_files: list[dict],
    budget_limit: int,
    context: StageContext,
    llm: LLMClient,
) -> set[str]:
    """Ask LLM which files to keep when budget is exceeded.

    Returns set of paths to keep. Files keep their original detail levels.
    """
    _require_logged_client(llm, "_refilter_files")

    file_lines = []
    for rf in rendered_files:
        file_lines.append(
            f"- {rf['path']} [{rf['detail']}] ~{rf['tokens']} tokens"
        )

    prompt = (
        f"Task: {context.task.raw_task}\n"
        f"Budget: {budget_limit} tokens available\n\n"
        f"Files (total exceeds budget):\n" + "\n".join(file_lines) + "\n\n"
        f"Select the subset of files to keep within budget."
    )

    # R3: validate prompt size before sending (conservative to match LLMClient gate)
    estimated, available, fits = check_prompt_budget(
        prompt, REFILTER_SYSTEM, llm.config.context_window, llm.config.max_tokens,
    )
    if not fits:
        logger.warning(
            "R3: refilter prompt too large (%d tokens, available %d) — falling back to priority drop",
            estimated, available,
        )
        return {rf["path"] for rf in _drop_by_priority(rendered_files, budget_limit)}

    response = llm.complete(prompt, system=REFILTER_SYSTEM)
    keep_list = parse_json_response(response.text, "refilter")

    if not isinstance(keep_list, list):
        raise ValueError(
            f"R18: refilter LLM returned {type(keep_list).__name__}, expected list. "
            f"Cannot curate context with invalid LLM response."
        )

    valid_paths = {rf["path"] for rf in rendered_files}
    keep_paths = {p for p in keep_list if isinstance(p, str) and p in valid_paths}
    dropped = [p for p in keep_list if isinstance(p, str) and p not in valid_paths]
    if dropped:
        logger.warning("R2: refilter LLM suggested %d unknown paths — dropped: %s", len(dropped), dropped)
    if not keep_paths and keep_list:
        raise ValueError(
            f"R18: refilter LLM returned {len(keep_list)} paths, none valid. "
            f"Expected paths from: {sorted(valid_paths)}"
        )
    return keep_paths


def _drop_by_priority(rendered_files: list[dict], budget_limit: int) -> list[dict]:
    """Drop entire files in reverse priority order to fit budget.

    Drops type_context first, then supporting, then primary. No downgrades.
    """
    result = list(rendered_files)

    for level in ("type_context", "supporting", "primary"):
        total = sum(rf["tokens"] for rf in result)
        if total <= budget_limit:
            break
        result = [rf for rf in result if rf["detail"] != level]

    return result


def _read_file_source(
    fid: int,
    info: dict,
    detail: str,
    repo_path: Path,
    kb: "KnowledgeBase | None",
) -> str:
    """Read file content from disk or knowledge base DB.

    Knowledge base files (file_source='knowledge_base', path starts with 'kb/')
    are read from ref_sections via the bridge. All other files are read from disk.
    Raises RuntimeError if the file cannot be read.
    """
    path = info["path"]

    # Knowledge base bridge files: read from ref_sections table
    if path.startswith("kb/") and kb is not None:
        content = kb.get_ref_section_content(fid)
        if content is not None:
            return content
        raise RuntimeError(
            f"R1: cannot read KB file '{path}' (detail={detail}): "
            f"section not found in ref_sections."
        )

    # Regular files: read from disk
    abs_path = repo_path / path
    try:
        return abs_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, IOError) as e:
        raise RuntimeError(
            f"R1: cannot read file '{path}' (detail={detail}): {e}. "
            f"File was classified by precision stage but cannot be read."
        ) from e


def _render_at_level(
    source: str,
    detail_level: str,
    file_id: int,
    context: StageContext,
    kb: "KnowledgeBase | None" = None,
) -> str:
    """Render file content at the specified detail level.

    R4: Uses parsed AST data (stored signatures and symbol ranges) from
    classified_symbols rather than string heuristics.
    """
    if detail_level == "primary":
        return source

    lines = source.split("\n")

    if detail_level == "type_context":
        # R4: render stored signatures from classified symbols
        return _extract_signatures(lines, file_id, context)

    # supporting: signatures + context around symbol ranges, separated by ...
    return _extract_supporting(lines, file_id, context, kb=kb)


def _extract_signatures(
    lines: list[str],
    file_id: int | None = None,
    context: StageContext | None = None,
) -> str:
    """Extract signatures using parsed AST data from classified symbols.

    R4: Uses stored symbol signatures rather than keyword matching.
    Falls back to line-based extraction only when no classified symbol data
    is available (legacy/test path).
    """
    # R4: prefer parsed signatures from classified symbols
    if context is not None and file_id is not None:
        sigs = []
        for cs in context.classified_symbols:
            if cs.file_id == file_id and cs.detail_level != "excluded":
                if cs.signature:
                    sigs.append(cs.signature)
                else:
                    # Fallback: use the source line at start_line
                    idx = cs.start_line - 1
                    if 0 <= idx < len(lines):
                        sigs.append(lines[idx])
        if sigs:
            return "\n".join(sigs)

    return "# (no signatures found)"


def _extract_supporting(
    lines: list[str],
    file_id: int,
    context: StageContext,
    kb: "KnowledgeBase | None" = None,
) -> str:
    """Extract signatures + context around symbol ranges.

    R4: Uses stored symbol line ranges from classified_symbols
    (parsed AST boundaries) and docstring data rather than heuristic line scanning.
    """
    # Gather symbol data for this file
    file_symbols = []
    for cs in context.classified_symbols:
        if cs.file_id == file_id and cs.detail_level != "excluded":
            file_symbols.append(cs)

    if not file_symbols:
        return _extract_signatures(lines, file_id, context)

    # R4: look up docstring line counts per symbol from curated DB
    docstring_lines_by_symbol: dict[int, int] = {}
    if kb is not None:
        docstrings = kb.get_docstrings_for_file(file_id)
        for doc in docstrings:
            if doc.symbol_id is not None:
                doc_line_count = doc.content.count("\n") + 1
                docstring_lines_by_symbol[doc.symbol_id] = doc_line_count

    parts = []
    prev_end = 0
    for cs in sorted(file_symbols, key=lambda s: s.start_line):
        if prev_end > 0 and cs.start_line > prev_end + 1:
            parts.append("...")

        # R4: use docstring end if available, otherwise fall back to full symbol
        doc_lines = docstring_lines_by_symbol.get(cs.symbol_id, 0)  # 0 = no docstring found
        if doc_lines > 0:
            # signature line(s) + docstring
            sig_end = min(cs.start_line + 1 + doc_lines, cs.end_line, len(lines))
        else:
            # No docstring — include full symbol
            sig_end = min(cs.end_line, len(lines))

        chunk = "\n".join(lines[max(0, cs.start_line - 1):sig_end])
        parts.append(chunk)
        prev_end = cs.end_line

    return "\n".join(parts) if parts else _extract_signatures(lines, file_id, context)
