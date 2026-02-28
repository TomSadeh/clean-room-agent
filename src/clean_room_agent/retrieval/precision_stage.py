"""Precision stage: symbol extraction + 3-pass binary classification cascade.

Binary decomposition: the 4-way classification (primary/supporting/type_context/excluded)
decomposes into 3 sequential binary passes, each filtering the input for the next:
  Pass 1: "Is this symbol relevant?" → yes=proceed, no=excluded
  Pass 2: "Is this symbol directly involved in the change?" → yes=primary, no=proceed
  Pass 3: "Does this symbol need full source or just signatures?" → yes=supporting, no=type_context

Each pass is a trivially parseable yes/no. The cascade is volume-reducing: pass 1
eliminates most symbols, passes 2-3 run on progressively smaller sets.
"""

import logging

from clean_room_agent.llm.client import LLMClient
from clean_room_agent.query.api import KnowledgeBase
from clean_room_agent.retrieval.batch_judgment import run_binary_judgment
from clean_room_agent.retrieval.dataclasses import ClassifiedSymbol, TaskQuery
from clean_room_agent.retrieval.stage import StageContext, register_stage, resolve_retrieval_param

logger = logging.getLogger(__name__)

MAX_CALLEES = 5
MAX_CALLERS = 5

PRECISION_PASS1_SYSTEM = (
    "You are a code relevance judge. Given a task and one code symbol, "
    "determine if this symbol is relevant to the task. "
    "Respond with ONLY \"yes\" or \"no\"."
)

PRECISION_PASS2_SYSTEM = (
    "You are a code relevance judge. Given a task and one code symbol "
    "that is relevant to the task, determine if this symbol is directly "
    "involved in the change (will be modified or is central to the logic). "
    "Respond with ONLY \"yes\" or \"no\"."
)

PRECISION_PASS3_SYSTEM = (
    "You are a code relevance judge. Given a task and one code symbol "
    "that provides context for the change, determine if the full source "
    "code is needed (yes) or if just the signature/type definition is "
    "sufficient (no). "
    "Respond with ONLY \"yes\" or \"no\"."
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
            sig = sym.signature or ""
            if not sym.signature:
                logger.debug(
                    "Symbol %s (%s) has no signature — using empty for classification",
                    sym.name, sym.kind,
                )
            entry = {
                "symbol_id": sym.id,
                "file_id": fid,
                "file_path": file_path,
                "name": sym.name,
                "kind": sym.kind,
                "start_line": sym.start_line,
                "end_line": sym.end_line,
                "signature": sig,
                "connections": [],
                "file_source": file_source_cache[fid],
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


def classify_symbols(
    candidates: list[dict],
    task: TaskQuery,
    llm: LLMClient,
    kb: "KnowledgeBase | None" = None,
) -> list[ClassifiedSymbol]:
    """3-pass binary cascade for symbol classification.

    Library symbols are pre-classified as type_context (R17: they cannot be
    primary, and sending them to the LLM wastes tokens). Only project symbols
    enter the binary cascade.

    Pass 1: relevant or excluded? (all project symbols)
    Pass 2: primary or not? (relevant subset only)
    Pass 3: supporting or type_context? (non-primary relevant subset only)
    """
    if not candidates:
        return []

    # R17: partition — library symbols skip LLM, auto-classified as type_context
    project_candidates = [c for c in candidates if c["file_source"] != "library"]
    library_candidates = [c for c in candidates if c["file_source"] == "library"]

    results: list[ClassifiedSymbol] = []

    # Auto-classify library symbols
    for c in library_candidates:
        results.append(ClassifiedSymbol(
            symbol_id=c["symbol_id"],
            file_id=c["file_id"],
            name=c["name"],
            kind=c["kind"],
            start_line=c["start_line"],
            end_line=c["end_line"],
            detail_level="type_context",
            reason="library symbol (auto-classified)",
            signature=c["signature"],
            file_source="library",
        ))

    if not project_candidates:
        return results

    # Batch-fetch docstrings for all involved files
    docstring_summaries: dict[tuple[int, int], str] = {}  # (file_id, symbol_id) -> summary
    if kb is not None:
        file_ids_seen: set[int] = set()
        for c in project_candidates:
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
            f"Symbol: {c['name']} ({c['kind']}) in {c['file_path']}:{c['start_line']}-{c['end_line']}\n"
            f"[{conn_info}]"
            + (f"\nsig: {c['signature']}" if c['signature'] else "")
        )
        doc_key = (c["file_id"], c["symbol_id"])
        doc_summary = docstring_summaries.get(doc_key)
        if doc_summary:
            line += f"\ndoc: {doc_summary}"
        return line

    def _symbol_key(c: dict) -> tuple:
        return (c["name"], c["file_path"], c["start_line"])

    task_header = f"Task: {task.raw_task}\nIntent: {task.intent_summary}\n\n"

    # === Pass 1: relevant or excluded? ===
    pass1_map, _omitted = run_binary_judgment(
        project_candidates,
        system_prompt=PRECISION_PASS1_SYSTEM,
        task_context=task_header,
        llm=llm,
        format_item=_format,
        stage_name="precision_pass1",
        item_key=_symbol_key,
        default_action="excluded",
    )

    relevant = [c for c in project_candidates if pass1_map.get(_symbol_key(c), False)]
    for c in project_candidates:
        if not pass1_map.get(_symbol_key(c), False):
            results.append(_make_classified(c, "excluded", "pass1: not relevant"))

    if not relevant:
        logger.info(
            "R2: precision pass1 excluded all %d project symbols — none judged relevant",
            len(project_candidates),
        )
        return results

    # === Pass 2: primary or not? (only relevant symbols) ===
    pass2_map, _omitted = run_binary_judgment(
        relevant,
        system_prompt=PRECISION_PASS2_SYSTEM,
        task_context=task_header,
        llm=llm,
        format_item=_format,
        stage_name="precision_pass2",
        item_key=_symbol_key,
        default_action="non-primary",
    )

    primary = [c for c in relevant if pass2_map.get(_symbol_key(c), False)]
    non_primary = [c for c in relevant if not pass2_map.get(_symbol_key(c), False)]

    for c in primary:
        results.append(_make_classified(c, "primary", "pass2: directly involved"))

    if not non_primary:
        logger.info(
            "Precision pass2: all %d relevant symbols classified as primary — skipping pass3",
            len(primary),
        )
        return results

    # === Pass 3: supporting or type_context? (only non-primary relevant symbols) ===
    pass3_map, _omitted = run_binary_judgment(
        non_primary,
        system_prompt=PRECISION_PASS3_SYSTEM,
        task_context=task_header,
        llm=llm,
        format_item=_format,
        stage_name="precision_pass3",
        item_key=_symbol_key,
        default_action="type_context",
    )

    for c in non_primary:
        if pass3_map.get(_symbol_key(c), False):
            results.append(_make_classified(c, "supporting", "pass3: full source needed"))
        else:
            results.append(_make_classified(c, "type_context", "pass3: signature sufficient"))

    return results


def _make_classified(c: dict, detail_level: str, reason: str) -> ClassifiedSymbol:
    """Build a ClassifiedSymbol from a candidate dict and cascade result."""
    return ClassifiedSymbol(
        symbol_id=c["symbol_id"],
        file_id=c["file_id"],
        name=c["name"],
        kind=c["kind"],
        start_line=c["start_line"],
        end_line=c["end_line"],
        detail_level=detail_level,
        reason=reason,
        signature=c["signature"],
        file_source=c["file_source"],
    )


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

    @property
    def preferred_role(self) -> str:
        return "reasoning"

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
            max_callees=resolve_retrieval_param(rp, "max_callees"),
            max_callers=resolve_retrieval_param(rp, "max_callers"),
        )
        classified = classify_symbols(candidates, task, llm, kb=kb)

        context.classified_symbols = classified
        return context
