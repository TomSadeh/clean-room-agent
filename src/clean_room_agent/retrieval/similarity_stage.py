"""Similarity stage: find structurally similar symbols for dedup/extract-pattern tasks."""

import logging

from clean_room_agent.llm.client import LLMClient
from clean_room_agent.query.api import KnowledgeBase
from clean_room_agent.retrieval.batch_judgment import (
    calculate_judgment_batch_size,
    validate_judgment_batch,
)
from clean_room_agent.retrieval.dataclasses import ClassifiedSymbol, TaskQuery
from clean_room_agent.retrieval.stage import StageContext, register_stage
from clean_room_agent.retrieval.utils import parse_json_response

logger = logging.getLogger(__name__)

# Module constants — configurable via [retrieval] config
MAX_CANDIDATE_PAIRS = 50
MIN_COMPOSITE_SCORE = 0.3
MAX_GROUP_SIZE = 8

SIMILARITY_JUDGMENT_SYSTEM = (
    "You are Jane, a code similarity analyst. Given pairs of functions/methods "
    "with structural signals, determine which pairs are truly similar enough to "
    "group for deduplication or pattern extraction.\n"
    "Respond with a JSON array: "
    '[{"pair_id": N, "keep": true/false, "group_label": "...", "reason": "..."}]. '
    "Only confirm pairs that share genuine structural or behavioral similarity. "
    "Respond with ONLY the JSON array, no markdown fencing or extra text."
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


_TOKENS_PER_PAIR = 40  # ~40 tokens per pair line in judgment prompt


def judge_similarity(
    pairs: list[dict],
    task: TaskQuery,
    llm: LLMClient,
) -> list[dict]:
    """LLM judgment on similarity pairs.

    R2: only confirmed pairs (keep=True) are returned.
    Omitted pairs default to denied.
    """
    if not pairs:
        return []

    # Calculate batch size from available context
    task_header = f"Task: {task.raw_task}\nIntent: {task.intent_summary}\n\nSymbol pairs:\n"
    batch_size = calculate_judgment_batch_size(
        SIMILARITY_JUDGMENT_SYSTEM, task_header,
        llm.config.context_window, llm.config.max_tokens,
        _TOKENS_PER_PAIR,
    )

    confirmed: list[dict] = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        pair_lines = []
        for p in batch:
            a, b = p["sym_a"], p["sym_b"]
            sigs = p["signals"]
            pair_lines.append(
                f"- pair_id={p['pair_id']}: {a.name} ({a.kind}, {a.end_line - a.start_line + 1} lines) "
                f"vs {b.name} ({b.kind}, {b.end_line - b.start_line + 1} lines) "
                f"[line_ratio={sigs['line_ratio']}, callee_jaccard={sigs['callee_jaccard']}, "
                f"name_lcs={sigs['name_lcs']}, same_parent={sigs['same_parent']}]"
            )

        prompt = task_header + "\n".join(pair_lines)

        validate_judgment_batch(
            prompt, SIMILARITY_JUDGMENT_SYSTEM, "similarity",
            llm.config.context_window, llm.config.max_tokens,
        )

        response = llm.complete(prompt, system=SIMILARITY_JUDGMENT_SYSTEM)
        try:
            judgments = parse_json_response(response.text, "similarity judgment")
        except ValueError:
            logger.warning("R2: similarity judgment parse failed — denying all pairs in batch")
            continue

        if not isinstance(judgments, list):
            logger.warning("R2: similarity judgment returned non-list — denying all pairs in batch")
            continue

        judgment_map: dict[int, dict] = {}
        for j in judgments:
            if isinstance(j, dict) and "pair_id" in j:
                judgment_map[j["pair_id"]] = j

        for p in batch:
            j = judgment_map.get(p["pair_id"])
            if j is None:
                logger.warning("R2: LLM omitted pair_id=%d — defaulting to denied", p["pair_id"])
                continue
            if j.get("keep", False):
                confirmed.append({
                    "pair_id": p["pair_id"],
                    "sym_a": p["sym_a"],
                    "sym_b": p["sym_b"],
                    "group_label": j.get("group_label", ""),
                    "reason": j.get("reason", ""),
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
        # Sort for determinism, take first max_group_size
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
            max_pairs=rp.get("max_candidate_pairs", MAX_CANDIDATE_PAIRS),
            min_score=rp.get("min_composite_score", MIN_COMPOSITE_SCORE),
        )

        if not pairs:
            return context

        confirmed = judge_similarity(pairs, task, llm)

        if not confirmed:
            return context

        group_map = assign_groups(
            confirmed,
            max_group_size=rp.get("max_group_size", MAX_GROUP_SIZE),
        )

        # Apply group_ids to existing symbols
        for cs in context.classified_symbols:
            gid = group_map.get(cs.symbol_id)
            if gid is not None:
                cs.group_id = gid

        return context
