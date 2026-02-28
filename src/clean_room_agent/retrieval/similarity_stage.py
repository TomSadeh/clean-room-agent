"""Similarity stage: find structurally similar symbols for dedup/extract-pattern tasks.

Binary decomposition: each candidate pair gets an independent yes/no LLM call.
The group_label (used only for logging, not by union-find) is dropped — the
input + verdict is the audit trail.
"""

import logging

from clean_room_agent.llm.client import LLMClient
from clean_room_agent.query.api import KnowledgeBase
from clean_room_agent.retrieval.batch_judgment import run_binary_judgment
from clean_room_agent.retrieval.dataclasses import ClassifiedSymbol, TaskQuery
from clean_room_agent.retrieval.stage import StageContext, register_stage, resolve_retrieval_param

logger = logging.getLogger(__name__)

# Module constants — configurable via [retrieval] config
MAX_CANDIDATE_PAIRS = 50
MIN_COMPOSITE_SCORE = 0.3
MAX_GROUP_SIZE = 8

SIMILARITY_BINARY_SYSTEM = (
    "You are a code similarity classifier. Given a task and two functions/methods "
    "with structural signals, answer: are these functions similar enough to group "
    "for deduplication or pattern extraction? Respond with ONLY \"yes\" or \"no\"."
)


def _longest_common_subsequence_ratio(a: str, b: str) -> float:
    """LCS ratio between two strings, normalized to [0, 1]."""
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    # Space-optimized LCS
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs_len = prev[n]
    return lcs_len / max(m, n)


def find_similar_pairs(
    symbols: list[ClassifiedSymbol],
    kb: KnowledgeBase,
    *,
    max_pairs: int = MAX_CANDIDATE_PAIRS,
    min_score: float = MIN_COMPOSITE_SCORE,
) -> list[dict]:
    """Deterministic pre-filter: score all same-kind function/method pairs.

    Scoring (weights sum to 1.0):
    - Line count ratio: 0.3
    - Shared callee Jaccard: 0.35
    - Name LCS ratio: 0.2
    - Same parent: 0.15

    Returns list of {pair_id, sym_a, sym_b, score, signals} sorted by score desc.
    """
    # Filter to functions/methods only (classes too coarse)
    candidates = [s for s in symbols if s.kind in ("function", "method")]
    if len(candidates) < 2:
        return []

    # Pre-compute callee sets and parent_symbol_id for all candidates
    callee_sets: dict[int, set[str]] = {}
    parent_ids: dict[int, int | None] = {}
    for sym in candidates:
        neighbors = kb.get_symbol_neighbors(sym.symbol_id, "callees")
        callee_sets[sym.symbol_id] = {n.name for n in neighbors}
        sym_record = kb.get_symbol_by_id(sym.symbol_id)
        parent_ids[sym.symbol_id] = sym_record.parent_symbol_id if sym_record else None

    pairs: list[dict] = []
    pair_id = 0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            # Must be same kind
            if a.kind != b.kind:
                continue

            # Line count ratio (1.0 = same size, 0.0 = very different)
            lines_a = a.end_line - a.start_line + 1
            lines_b = b.end_line - b.start_line + 1
            line_ratio = min(lines_a, lines_b) / max(lines_a, lines_b) if max(lines_a, lines_b) > 0 else 0.0

            # Shared callee Jaccard
            callees_a = callee_sets.get(a.symbol_id, set())
            callees_b = callee_sets.get(b.symbol_id, set())
            if callees_a or callees_b:
                jaccard = len(callees_a & callees_b) / len(callees_a | callees_b)
            else:
                jaccard = 0.0

            # Name LCS ratio
            name_lcs = _longest_common_subsequence_ratio(a.name, b.name)

            # Same parent
            parent_a = parent_ids.get(a.symbol_id)
            parent_b = parent_ids.get(b.symbol_id)
            same_parent = 1.0 if (parent_a is not None and parent_a == parent_b) else 0.0

            # Composite score
            score = (
                0.3 * line_ratio
                + 0.35 * jaccard
                + 0.2 * name_lcs
                + 0.15 * same_parent
            )

            if score >= min_score:
                pairs.append({
                    "pair_id": pair_id,
                    "sym_a": a,
                    "sym_b": b,
                    "score": score,
                    "signals": {
                        "line_ratio": round(line_ratio, 3),
                        "callee_jaccard": round(jaccard, 3),
                        "name_lcs": round(name_lcs, 3),
                        "same_parent": bool(same_parent),
                    },
                })
                pair_id += 1

    # R6: sort by score descending, then cap
    pairs.sort(key=lambda p: p["score"], reverse=True)
    return pairs[:max_pairs]


def judge_similarity(
    pairs: list[dict],
    task: TaskQuery,
    llm: LLMClient,
) -> list[dict]:
    """Binary LLM judgment on similarity pairs.

    Each pair gets an independent yes/no LLM call.
    Binary decomposition is architectural — not conditional on model size.

    R2: only confirmed pairs (yes) are returned. Omitted pairs default to denied.
    """
    if not pairs:
        return []

    def _format(p: dict) -> str:
        a, b = p["sym_a"], p["sym_b"]
        sigs = p["signals"]
        return (
            f"Function A: {a.name} ({a.kind}, {a.end_line - a.start_line + 1} lines)\n"
            f"Function B: {b.name} ({b.kind}, {b.end_line - b.start_line + 1} lines)\n"
            f"Signals: line_ratio={sigs['line_ratio']}, callee_jaccard={sigs['callee_jaccard']}, "
            f"name_lcs={sigs['name_lcs']}, same_parent={sigs['same_parent']}"
        )

    task_header = f"Task: {task.raw_task}\nIntent: {task.intent_summary}\n\n"

    verdict_map, omitted = run_binary_judgment(
        pairs,
        system_prompt=SIMILARITY_BINARY_SYSTEM,
        task_context=task_header,
        llm=llm,
        format_item=_format,
        stage_name="similarity",
        item_key=lambda p: p["pair_id"],
        default_action="denied",
    )

    confirmed: list[dict] = []
    for p in pairs:
        pid = p["pair_id"]
        if verdict_map.get(pid, False):
            confirmed.append({
                "pair_id": pid,
                "sym_a": p["sym_a"],
                "sym_b": p["sym_b"],
                "group_label": "",
                "reason": "binary classifier confirmed",
            })

    return confirmed


def assign_groups(
    confirmed_pairs: list[dict],
    *,
    max_group_size: int = MAX_GROUP_SIZE,
) -> dict[int, str]:
    """Union-find grouping from confirmed pairs.

    Returns {symbol_id: group_id} mapping.
    Group ID = "sim_group_{smallest_symbol_id_in_group}".
    Groups exceeding max_group_size are capped.
    """
    if not confirmed_pairs:
        logger.info("assign_groups: no confirmed similar pairs — returning empty groups")
        return {}

    # Union-find
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            # Always make the smaller root the parent for determinism
            if rx > ry:
                rx, ry = ry, rx
            parent[ry] = rx

    for pair in confirmed_pairs:
        a_id = pair["sym_a"].symbol_id
        b_id = pair["sym_b"].symbol_id
        parent.setdefault(a_id, a_id)
        parent.setdefault(b_id, b_id)
        union(a_id, b_id)

    # Build groups
    groups: dict[int, list[int]] = {}  # root -> members
    for sym_id in parent:
        root = find(sym_id)
        groups.setdefault(root, []).append(sym_id)

    # Assign group IDs, cap size
    result: dict[int, str] = {}
    for root, members in groups.items():
        group_id = f"sim_group_{min(members)}"
        # R6: all group members have equal relevance (confirmed similar by LLM);
        # sorting by symbol_id provides deterministic selection.
        for sym_id in sorted(members)[:max_group_size]:
            result[sym_id] = group_id

    return result


@register_stage("similarity", description=(
    "Identifies structurally similar functions/methods across scoped files "
    "using line count, callee overlap, name similarity, and parent context. "
    "Groups similar symbols so the execute stage can see them together for "
    "deduplication or pattern extraction tasks."
))
class SimilarityStage:
    """Similarity retrieval stage: find and group similar symbols."""

    @property
    def name(self) -> str:
        return "similarity"

    @property
    def preferred_role(self) -> str:
        return "classifier"

    def run(
        self,
        context: StageContext,
        kb: KnowledgeBase,
        task: TaskQuery,
        llm: LLMClient,
    ) -> StageContext:
        if not context.classified_symbols:
            return context

        rp = context.retrieval_params
        pairs = find_similar_pairs(
            context.classified_symbols, kb,
            max_pairs=resolve_retrieval_param(rp, "max_candidate_pairs"),
            min_score=resolve_retrieval_param(rp, "min_composite_score"),
        )

        if not pairs:
            return context

        confirmed = judge_similarity(pairs, task, llm)

        if not confirmed:
            return context

        group_map = assign_groups(
            confirmed,
            max_group_size=resolve_retrieval_param(rp, "max_group_size"),
        )

        # Apply group_ids to existing symbols
        for cs in context.classified_symbols:
            gid = group_map.get(cs.symbol_id)
            if gid is not None:
                cs.group_id = gid

        return context
