"""Precision stage: symbol extraction + LLM classification."""

import logging

from clean_room_agent.llm.client import LLMClient
from clean_room_agent.query.api import KnowledgeBase
from clean_room_agent.retrieval.budget import estimate_tokens_conservative
from clean_room_agent.retrieval.dataclasses import ClassifiedSymbol, TaskQuery
from clean_room_agent.retrieval.stage import StageContext, register_stage
from clean_room_agent.retrieval.utils import parse_json_response

logger = logging.getLogger(__name__)

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
) -> list[dict]:
    """Extract candidate symbols from included files with edge info.

    For Python files: symbol_references edges (callees/callers).
    For TS/JS files: name matching against task keywords/symbols.
    """
    candidates = []
    file_cache: dict[int, tuple[str, str]] = {}  # file_id -> (path, language)

    # Cache file paths and languages
    for fid in file_ids:
        f = kb.get_file_by_id(fid)
        if f:
            file_cache[fid] = (f.path, f.language)

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
            }

            if language == "python":
                # Edge traversal for Python
                # R6c: order neighbors by whether their file is in included set, then cap
                callees = kb.get_symbol_neighbors(sym.id, "callees")
                callees.sort(key=lambda c: (c.file_id not in file_ids, c.name))
                for c in callees[:5]:
                    entry["connections"].append(f"calls {c.name}")
                callers = kb.get_symbol_neighbors(sym.id, "callers")
                callers.sort(key=lambda c: (c.file_id not in file_ids, c.name))
                for c in callers[:5]:
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


_TOKENS_PER_SYMBOL = 50  # ~50 tokens per symbol line (name + path + sig + connections)


def classify_symbols(
    candidates: list[dict],
    task: TaskQuery,
    llm: LLMClient,
) -> list[ClassifiedSymbol]:
    """LLM classification of symbol detail levels.

    Batches symbols to fit within the model's context window.
    """
    if not candidates:
        return []

    # Calculate batch size from available context (conservative to match LLMClient gate)
    task_header = f"Task: {task.raw_task}\nIntent: {task.intent_summary}\n\nSymbols:\n"
    system_overhead = estimate_tokens_conservative(PRECISION_SYSTEM)
    header_overhead = estimate_tokens_conservative(task_header)
    available = llm.config.context_window - llm.config.max_tokens - system_overhead - header_overhead
    batch_size = max(1, available // _TOKENS_PER_SYMBOL)

    # Classify in batches, merge results
    class_map: dict[tuple[str, str, int], dict] = {}
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]

        symbol_lines = []
        for c in batch:
            conn_info = ", ".join(c["connections"]) if c["connections"] else "no connections"
            symbol_lines.append(
                f"- {c['name']} ({c['kind']}) in {c['file_path']}:{c['start_line']}-{c['end_line']} "
                f"[{conn_info}]"
                + (f" sig: {c['signature']}" if c['signature'] else "")
            )

        prompt = task_header + "\n".join(symbol_lines)
        response = llm.complete(prompt, system=PRECISION_SYSTEM)
        classifications = parse_json_response(response.text, "precision")

        if isinstance(classifications, list):
            for cl in classifications:
                if isinstance(cl, dict) and "name" in cl:
                    key = (cl["name"], cl.get("file_path", ""), cl.get("start_line", 0))
                    class_map[key] = cl

    results = []
    for c in candidates:
        key = (c["name"], c["file_path"], c["start_line"])
        cl = class_map.get(key)
        if cl is None:
            logger.warning(
                "R2: LLM omitted %s in %s:%d from precision classification — defaulting to excluded",
                c["name"], c["file_path"], c["start_line"],
            )
            cl = {}
        detail_level = cl.get("detail_level", "excluded")
        if detail_level not in ("primary", "supporting", "type_context", "excluded"):
            logger.warning("R2: invalid detail_level %r for %s — defaulting to excluded", detail_level, c["name"])
            detail_level = "excluded"
        reason = cl.get("reason", "")

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
        ))

    return results


@register_stage("precision")
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
        candidates = extract_precision_symbols(
            context.included_file_ids, task, kb,
        )
        classified = classify_symbols(candidates, task, llm)

        context.classified_symbols = classified
        return context
