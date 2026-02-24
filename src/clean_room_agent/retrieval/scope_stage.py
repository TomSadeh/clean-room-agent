"""Scope stage: tiered expansion + LLM judgment."""

import logging

from clean_room_agent.llm.client import LLMClient
from clean_room_agent.query.api import KnowledgeBase
from clean_room_agent.retrieval.budget import estimate_tokens_conservative
from clean_room_agent.retrieval.dataclasses import ScopedFile, TaskQuery
from clean_room_agent.retrieval.stage import StageContext, register_stage
from clean_room_agent.retrieval.utils import parse_json_response

logger = logging.getLogger(__name__)

# Per-tier caps to avoid explosion
MAX_DEPS = 30
MAX_CO_CHANGES = 20
MAX_METADATA = 20

SCOPE_JUDGMENT_SYSTEM = (
    "You are Jane, a code retrieval judge. Given a task description and a list of candidate files, "
    "determine which files are relevant to the task. "
    "Respond with a JSON array of objects: [{\"path\": \"...\", \"verdict\": \"relevant\" or \"irrelevant\", \"reason\": \"...\"}]. "
    "Respond with ONLY the JSON array, no markdown fencing or extra text."
)


def expand_scope(
    task: TaskQuery,
    kb: KnowledgeBase,
    repo_id: int,
    plan_file_ids: list[int] | None = None,
    *,
    max_deps: int = MAX_DEPS,
    max_co_changes: int = MAX_CO_CHANGES,
    max_metadata: int = MAX_METADATA,
) -> list[ScopedFile]:
    """Deterministic tiered expansion from seeds.

    Tiers: 0=plan, 1=seed, 2=dep, 3=co-change, 4=metadata.
    Lower tier wins on dedup.
    """
    seen: dict[int, ScopedFile] = {}

    def _add(file_id: int, path: str, language: str, tier: int, reason: str) -> None:
        if file_id not in seen:
            seen[file_id] = ScopedFile(
                file_id=file_id, path=path, language=language,
                tier=tier, relevance="pending", reason=reason,
            )

    def _file_lookup(file_id: int):
        f = kb.get_file_by_id(file_id)
        if f:
            return {"id": f.id, "path": f.path, "language": f.language}
        return None

    # Tier 0: plan files
    for fid in (plan_file_ids or []):
        info = _file_lookup(fid)
        if info:
            _add(info["id"], info["path"], info["language"], 0, "plan artifact")

    # Tier 1: seed files from task
    for fid in task.seed_file_ids:
        info = _file_lookup(fid)
        if info:
            _add(info["id"], info["path"], info["language"], 1, "seed file")

    # Tier 1: files containing seed symbols
    for sid in task.seed_symbol_ids:
        sym = kb.get_symbol_by_id(sid)
        if sym:
            info = _file_lookup(sym.file_id)
            if info:
                _add(info["id"], info["path"], info["language"], 1, "contains seed symbol")

    # Expansion base: tier 0 and 1 files
    base_fids = [fid for fid, sf in seen.items() if sf.tier <= 1]
    base_fids_set = set(base_fids)

    # Tier 2: dependencies (imports + imported_by)
    # R6a: collect all dep candidates, order by connections to seed files, then cap
    # Cache all seed dependencies in a single pass to avoid O(N*M) re-queries
    seed_imports: dict[int, set[int]] = {}  # seed_fid -> set of target fids
    seed_imported_by: dict[int, set[int]] = {}  # seed_fid -> set of source fids
    for fid in base_fids:
        seed_imports[fid] = {d.target_file_id for d in kb.get_dependencies(fid, "imports")}
        seed_imported_by[fid] = {d.source_file_id for d in kb.get_dependencies(fid, "imported_by")}

    dep_candidates: list[tuple[int, str, str, str, int]] = []  # (fid, path, lang, reason, seed_connections)
    for fid in base_fids:
        for target_fid in seed_imports[fid]:
            info = _file_lookup(target_fid)
            if info and info["id"] not in seen:
                seed_conns = sum(1 for sf in base_fids if target_fid in seed_imports.get(sf, set()))
                dep_candidates.append((info["id"], info["path"], info["language"],
                                       f"imported by {fid}", seed_conns))
        for source_fid in seed_imported_by[fid]:
            info = _file_lookup(source_fid)
            if info and info["id"] not in seen:
                seed_conns = sum(1 for sf in base_fids if source_fid in seed_imported_by.get(sf, set()))
                dep_candidates.append((info["id"], info["path"], info["language"],
                                       f"imports {fid}", seed_conns))

    # Dedup by file_id, keeping highest seed_connections
    dep_dedup: dict[int, tuple[int, str, str, str, int]] = {}
    for fid, path, lang, reason, sc in dep_candidates:
        if fid not in dep_dedup or sc > dep_dedup[fid][4]:
            dep_dedup[fid] = (fid, path, lang, reason, sc)

    # R6: sort by seed connections descending, then apply cap
    sorted_deps = sorted(dep_dedup.values(), key=lambda x: x[4], reverse=True)
    for fid, path, lang, reason, _ in sorted_deps[:max_deps]:
        _add(fid, path, lang, 2, reason)

    # Tier 3: co-change neighbors — R6: collect globally, sort by count, then cap
    co_candidates: list[tuple[int, str, str, str, int]] = []
    for fid in base_fids:
        neighbors = kb.get_co_change_neighbors(fid, min_count=2)
        for cc in neighbors:
            other_fid = cc.file_b_id if cc.file_a_id == fid else cc.file_a_id
            info = _file_lookup(other_fid)
            if info and info["id"] not in seen:
                co_candidates.append((
                    info["id"], info["path"], info["language"],
                    f"co-changed {cc.count}x with {fid}", cc.count,
                ))

    # Dedup by file_id, keeping highest count
    co_dedup: dict[int, tuple[int, str, str, str, int]] = {}
    for fid_c, path_c, lang_c, reason_c, count_c in co_candidates:
        if fid_c not in co_dedup or count_c > co_dedup[fid_c][4]:
            co_dedup[fid_c] = (fid_c, path_c, lang_c, reason_c, count_c)

    sorted_co = sorted(co_dedup.values(), key=lambda x: x[4], reverse=True)
    for fid_c, path_c, lang_c, reason_c, _ in sorted_co[:max_co_changes]:
        _add(fid_c, path_c, lang_c, 3, reason_c)

    # Tier 4: metadata search (enrichment)
    if task.keywords:
        # R6b: order keywords by length (longer = more specific) before capping
        ordered_keywords = sorted(task.keywords, key=len, reverse=True)[:5]
        meta_candidates: list[tuple[int, str, str, str, int]] = []  # (fid, path, lang, reason, kw_specificity)
        for kw_idx, kw in enumerate(ordered_keywords):
            matches = kb.search_files_by_metadata(repo_id, concepts=kw)
            if not matches:
                continue
            for f in matches:
                if f.id not in seen:
                    # R6b: specificity = keyword length + position priority (earlier = more specific)
                    specificity = len(kw) * 10 + (len(ordered_keywords) - kw_idx)
                    meta_candidates.append((f.id, f.path, f.language, f"metadata match: {kw}", specificity))

        # Dedup by file_id, keeping highest specificity
        meta_dedup: dict[int, tuple[int, str, str, str, int]] = {}
        for fid, path, lang, reason, spec in meta_candidates:
            if fid not in meta_dedup or spec > meta_dedup[fid][4]:
                meta_dedup[fid] = (fid, path, lang, reason, spec)

        # R6: sort by specificity descending, then apply cap
        sorted_meta = sorted(meta_dedup.values(), key=lambda x: x[4], reverse=True)
        for fid, path, lang, reason, _ in sorted_meta[:max_metadata]:
            _add(fid, path, lang, 4, reason)

        if not meta_candidates:
            logger.info("No enrichment data matched keywords; Tier 4 skipped.")

    return list(seen.values())


_TOKENS_PER_SCOPE_CANDIDATE = 20  # ~20 tokens per candidate line (path + tier info)


def judge_scope(
    candidates: list[ScopedFile],
    task: TaskQuery,
    llm: LLMClient,
) -> list[ScopedFile]:
    """LLM judgment on scope candidates. Seeds (tier 0/1) always relevant.

    Batches non-seed candidates to fit within the model's context window.
    """
    if not candidates:
        return candidates

    # Seeds skip LLM judgment entirely
    seeds = [sf for sf in candidates if sf.tier <= 1]
    non_seeds = [sf for sf in candidates if sf.tier > 1]

    for sf in seeds:
        sf.relevance = "relevant"
        if not sf.reason:
            sf.reason = "seed (always included)"

    if not non_seeds:
        return candidates

    # Calculate batch size from available context (conservative to match LLMClient gate)
    task_header = f"Task: {task.raw_task}\nIntent: {task.intent_summary}\n\nCandidate files:\n"
    system_overhead = estimate_tokens_conservative(SCOPE_JUDGMENT_SYSTEM)
    header_overhead = estimate_tokens_conservative(task_header)
    available = llm.config.context_window - llm.config.max_tokens - system_overhead - header_overhead
    batch_size = max(1, available // _TOKENS_PER_SCOPE_CANDIDATE)

    # Judge in batches
    verdict_map: dict[str, dict] = {}
    for i in range(0, len(non_seeds), batch_size):
        batch = non_seeds[i:i + batch_size]
        candidate_lines = [
            f"- {sf.path} (tier={sf.tier}, language={sf.language}, reason={sf.reason})"
            for sf in batch
        ]
        prompt = task_header + "\n".join(candidate_lines)

        response = llm.complete(prompt, system=SCOPE_JUDGMENT_SYSTEM)
        judgments = parse_json_response(response.text, "scope judgment")

        if isinstance(judgments, list):
            for j in judgments:
                if isinstance(j, dict) and "path" in j:
                    verdict_map[j["path"]] = j

    # Apply judgments to non-seeds
    valid_verdicts = ("relevant", "irrelevant")
    for sf in non_seeds:
        if sf.path in verdict_map:
            v = verdict_map[sf.path]
            verdict = v.get("verdict", "irrelevant")
            if verdict not in valid_verdicts:
                logger.warning("R2: invalid verdict %r for %s — defaulting to irrelevant", verdict, sf.path)
                verdict = "irrelevant"
            sf.relevance = verdict
            sf.reason = v.get("reason", sf.reason)
        else:
            logger.warning("R2: LLM omitted %s from scope judgment — defaulting to irrelevant", sf.path)
            sf.relevance = "irrelevant"

    return candidates


@register_stage("scope", description=(
    "Finds related files beyond those explicitly mentioned, using import "
    "dependencies, co-change history, and metadata search, then judges "
    "each file's relevance."
))
class ScopeStage:
    """Scope retrieval stage: tiered expansion + LLM judgment."""

    @property
    def name(self) -> str:
        return "scope"

    def run(
        self,
        context: StageContext,
        kb: KnowledgeBase,
        task: TaskQuery,
        llm: LLMClient,
    ) -> StageContext:
        plan_file_ids = [
            sf.file_id for sf in context.scoped_files if sf.tier == 0
        ] or None

        candidates = expand_scope(task, kb, context.repo_id, plan_file_ids)
        judged = judge_scope(candidates, task, llm)

        context.scoped_files = judged
        context.included_file_ids = {
            sf.file_id for sf in judged if sf.relevance == "relevant"
        }

        return context
