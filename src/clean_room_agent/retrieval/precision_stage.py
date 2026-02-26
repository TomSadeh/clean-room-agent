"""Precision stage: symbol extraction + LLM classification."""

import logging

from clean_room_agent.llm.client import LLMClient
from clean_room_agent.query.api import KnowledgeBase
from clean_room_agent.retrieval.batch_judgment import (
    log_r2_omission,
    run_batched_judgment,
)
from clean_room_agent.retrieval.dataclasses import ClassifiedSymbol, TaskQuery
from clean_room_agent.retrieval.stage import StageContext, register_stage

logger = logging.getLogger(__name__)

MAX_CALLEES = 5
MAX_CALLERS = 5

PRECISION_SYSTEM = (
    "You are Jane, a code precision analyst. Given a task description and a list of symbols, "
    "classify each symbol's relevance to the task. "
    "Detail levels:\n"
    "- primary: directly involved in the change\n"
    "- supporting: provides context needed to understand primary symbols\n"
    "- type_context: type definitions, interfaces, or constants referenced by primary/supporting\n"
    "- excluded: not relevant to this task\n\n"
    "Respond with a JSON array: [{\"name\": \"...\", \"file_path\": \"...\", \"start_line\": N, \"detail_level\": \"...\", \"reason\": \"...\"}]. "
    "Include start_line to disambiguate symbols with the same name in the same file. "
    "Respond with ONLY the JSON array, no markdown fencing or extra text."
)


def extract_precision_symbols(
    file_ids: set[int],
    task: TaskQuery,
    kb: KnowledgeBase,
    *,
    max_callees: int = MAX_CALLEES,
    max_callers: int = MAX_CALLERS,
) -> list[dict]:
    """Extract candidate symbols from included files with edge info.

    For Python files: symbol_references edges (callees/callers).
    For TS/JS files: name matching against task keywords/symbols.
    """
    candidates = []
    file_cache: dict[int, tuple[str, str]] = {}  # file_id -> (path, language)

    # Cache file paths, languages, and source
    file_source_cache: dict[int, str] = {}  # file_id -> file_source
    for fid in file_ids:
        f = kb.get_file_by_id(fid)
        if f:
            file_cache[fid] = (f.path, f.language)
            file_source_cache[fid] = f.file_source

    for fid in file_ids:
        if fid not in file_cache:
            continue
        file_path, language = file_cache[fid]
        symbols = kb.get_symbols_for_file(fid)

        for sym in symbols:
            entry = {
                "symbol_id": sym.id,
                "file_id": fid,
                "file_path": file_path,
                "name": sym.name,
                "kind": sym.kind,
                "start_line": sym.start_line,
                "end_line": sym.end_line,
                "signature": sym.signature or "",
                "connections": [],
                "file_source": file_source_cache.get(fid, "project"),
            }

            if language == "python":
                # Edge traversal for Python
                # R6c: order neighbors by whether their file is in included set, then cap
                callees = kb.get_symbol_neighbors(sym.id, "callees")
                callees.sort(key=lambda c: (c.file_id not in file_ids, c.name))
                for c in callees[:max_callees]:
                    entry["connections"].append(f"calls {c.name}")
                callers = kb.get_symbol_neighbors(sym.id, "callers")
                callers.sort(key=lambda c: (c.file_id not in file_ids, c.name))
                for c in callers[:max_callers]:
                    entry["connections"].append(f"called by {c.name}")
            else:
                # TS/JS: name matching against task keywords/symbols
                task_terms = set(task.keywords + task.mentioned_symbols)
                name_lower = sym.name.lower()
                for term in task_terms:
                    if term.lower() in name_lower or name_lower in term.lower():
                        entry["connections"].append(f"name matches task term '{term}'")

            candidates.append(entry)

    return candidates


_TOKENS_PER_SYMBOL = 70  # ~70 tokens per symbol line (name + path + sig + connections + docstring)


def classify_symbols(
    candidates: list[dict],
    task: TaskQuery,
    llm: LLMClient,
    kb: "KnowledgeBase | None" = None,
) -> list[ClassifiedSymbol]:
    """LLM classification of symbol detail levels.

    Batches symbols to fit within the model's context window.
    """
    if not candidates:
        return []

    # Batch-fetch docstrings for all involved files
    docstring_summaries: dict[tuple[int, int], str] = {}  # (file_id, symbol_id) -> summary
    if kb is not None:
        file_ids_seen: set[int] = set()
        for c in candidates:
            fid = c["file_id"]
            if fid not in file_ids_seen:
                file_ids_seen.add(fid)
                docstrings = kb.get_docstrings_for_file(fid)
                for doc in docstrings:
                    if doc.symbol_id is not None:
                        first_line = doc.content.split("\n", 1)[0].strip()
                        docstring_summaries[(fid, doc.symbol_id)] = first_line

    def _format(c: dict) -> str:
        conn_info = ", ".join(c["connections"]) if c["connections"] else "no connections"
        line = (
            f"- {c['name']} ({c['kind']}) in {c['file_path']}:{c['start_line']}-{c['end_line']} "
            f"[{conn_info}]"
            + (f" sig: {c['signature']}" if c['signature'] else "")
        )
        doc_key = (c["file_id"], c["symbol_id"])
        doc_summary = docstring_summaries.get(doc_key)
        if doc_summary:
            line += f" doc: {doc_summary}"
        return line

    task_header = f"Task: {task.raw_task}\nIntent: {task.intent_summary}\n\nSymbols:\n"

    class_map = run_batched_judgment(
        candidates,
        system_prompt=PRECISION_SYSTEM,
        task_header=task_header,
        llm=llm,
        tokens_per_item=_TOKENS_PER_SYMBOL,
        format_item=_format,
        extract_key=lambda cl: (cl["name"], cl.get("file_path", ""), cl.get("start_line", 0)) if "name" in cl else None,
        stage_name="precision",
    )

    results = []
    for c in candidates:
        key = (c["name"], c["file_path"], c["start_line"])
        cl = class_map.get(key)
        if cl is None:
            log_r2_omission(
                f"{c['name']} in {c['file_path']}:{c['start_line']}",
                "precision", "excluded",
            )
            cl = {}
        detail_level = cl.get("detail_level", "excluded")
        if detail_level not in ("primary", "supporting", "type_context", "excluded"):
            logger.warning("R2: invalid detail_level %r for %s — defaulting to excluded", detail_level, c["name"])
            detail_level = "excluded"
        reason = cl.get("reason", "")

        file_source = c["file_source"]

        # Library symbols cannot be "primary" — downgrade to "type_context"
        if file_source == "library" and detail_level == "primary":
            logger.warning(
                "Library symbol %s in %s classified as primary — downgrading to type_context",
                c["name"], c["file_path"],
            )
            detail_level = "type_context"
            reason = f"downgraded from primary (library file): {reason}"

        results.append(ClassifiedSymbol(
            symbol_id=c["symbol_id"],
            file_id=c["file_id"],
            name=c["name"],
            kind=c["kind"],
            start_line=c["start_line"],
            end_line=c["end_line"],
            detail_level=detail_level,
            reason=reason,
            signature=c.get("signature", ""),
            file_source=file_source,
        ))

    return results


@register_stage("precision", description=(
    "Classifies individual symbols (functions, classes) within scoped files "
    "by relevance level (primary, supporting, type_context) to control how "
    "much of each file enters the context window."
))
class PrecisionStage:
    """Precision retrieval stage: symbol extraction + LLM classification."""

    @property
    def name(self) -> str:
        return "precision"

    def run(
        self,
        context: StageContext,
        kb: KnowledgeBase,
        task: TaskQuery,
        llm: LLMClient,
    ) -> StageContext:
        rp = context.retrieval_params
        candidates = extract_precision_symbols(
            context.included_file_ids, task, kb,
            max_callees=rp.get("max_callees", MAX_CALLEES),
            max_callers=rp.get("max_callers", MAX_CALLERS),
        )
        classified = classify_symbols(candidates, task, llm, kb=kb)

        context.classified_symbols = classified
        return context
