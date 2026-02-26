"""Scope stage: tiered expansion + LLM judgment."""

import logging
from typing import NamedTuple

from clean_room_agent.llm.client import LLMClient
from clean_room_agent.query.api import KnowledgeBase
from clean_room_agent.retrieval.batch_judgment import run_batched_judgment
from clean_room_agent.retrieval.dataclasses import ScopedFile, TaskQuery
from clean_room_agent.retrieval.stage import StageContext, register_stage, resolve_retrieval_param

logger = logging.getLogger(__name__)

# Per-tier caps to avoid explosion
MAX_DEPS = 30
MAX_CO_CHANGES = 20
MAX_METADATA = 20
MAX_KEYWORDS = 5

class ScopeCandidate(NamedTuple):
    """A candidate file for scope expansion with its sort score."""
    file_id: int
    path: str
    language: str
    reason: str
    score: int


def _dedup_by_score(candidates: list[ScopeCandidate]) -> list[ScopeCandidate]:
    """Deduplicate candidates by file_id, keeping the highest score."""
    best: dict[int, ScopeCandidate] = {}
    for c in candidates:
        if c.file_id not in best or c.score > best[c.file_id].score:
            best[c.file_id] = c
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


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
    max_keywords: int = MAX_KEYWORDS,
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

    dep_candidates: list[ScopeCandidate] = []
    for fid in base_fids:
        for target_fid in seed_imports[fid]:
            info = _file_lookup(target_fid)
            if info and info["id"] not in seen:
                seed_conns = sum(1 for sf in base_fids if target_fid in seed_imports.get(sf, set()))
                dep_candidates.append(ScopeCandidate(
                    info["id"], info["path"], info["language"],
                    f"imported by {fid}", seed_conns))
        for source_fid in seed_imported_by[fid]:
            info = _file_lookup(source_fid)
            if info and info["id"] not in seen:
                seed_conns = sum(1 for sf in base_fids if source_fid in seed_imported_by.get(sf, set()))
                dep_candidates.append(ScopeCandidate(
                    info["id"], info["path"], info["language"],
                    f"imports {fid}", seed_conns))

    # R6: dedup by file_id (highest score wins), sort descending, then cap
    for c in _dedup_by_score(dep_candidates)[:max_deps]:
        _add(c.file_id, c.path, c.language, 2, c.reason)

    # Tier 3: co-change neighbors — R6: collect globally, sort by count, then cap
    co_candidates: list[ScopeCandidate] = []
    for fid in base_fids:
        neighbors = kb.get_co_change_neighbors(fid, min_count=2)
        for cc in neighbors:
            other_fid = cc.file_b_id if cc.file_a_id == fid else cc.file_a_id
            info = _file_lookup(other_fid)
            if info and info["id"] not in seen:
                co_candidates.append(ScopeCandidate(
                    info["id"], info["path"], info["language"],
                    f"co-changed {cc.count}x with {fid}", cc.count,
                ))

    # R6: dedup by file_id (highest score wins), sort descending, then cap
    for c in _dedup_by_score(co_candidates)[:max_co_changes]:
        _add(c.file_id, c.path, c.language, 3, c.reason)

    # Tier 4: metadata search (enrichment)
    if task.keywords:
        # R6b: order keywords by length (longer = more specific) before capping
        ordered_keywords = sorted(task.keywords, key=len, reverse=True)[:max_keywords]
        meta_candidates: list[ScopeCandidate] = []
        for kw_idx, kw in enumerate(ordered_keywords):
            matches = kb.search_files_by_metadata(repo_id, concepts=kw)
            if not matches:
                continue
            for f in matches:
                if f.id not in seen:
                    # R6b: specificity = keyword length + position priority (earlier = more specific)
                    specificity = len(kw) * 10 + (len(ordered_keywords) - kw_idx)
                    meta_candidates.append(ScopeCandidate(
                        f.id, f.path, f.language, f"metadata match: {kw}", specificity))

        # R6: dedup by file_id (highest score wins), sort descending, then cap
        for c in _dedup_by_score(meta_candidates)[:max_metadata]:
            _add(c.file_id, c.path, c.language, 4, c.reason)

        if not meta_candidates:
            logger.info("No enrichment data matched keywords; Tier 4 skipped.")

    return list(seen.values())


_TOKENS_PER_SCOPE_CANDIDATE = 100  # ~100 tokens per candidate line (path + tier + metadata suffix)


def judge_scope(
    candidates: list[ScopedFile],
    task: TaskQuery,
    llm: LLMClient,
    kb: KnowledgeBase | None = None,
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

    # Batch-fetch metadata for all non-seed candidates
    metadata_map: dict = {}
    if kb is not None:
        all_fids = [sf.file_id for sf in non_seeds]
        metadata_map = kb.get_file_metadata_batch(all_fids)

    def _format(sf: ScopedFile) -> str:
        line = f"- {sf.path} (tier={sf.tier}, language={sf.language}, reason={sf.reason})"
        meta = metadata_map.get(sf.file_id)
        if meta:
            parts = []
            if meta.purpose:
                parts.append(f"purpose={meta.purpose}")
            if meta.domain:
                parts.append(f"domain={meta.domain}")
            if meta.concepts:
                parts.append(f"concepts={meta.concepts}")
            if parts:
                line += f" [{', '.join(parts)}]"
        return line

    task_header = f"Task: {task.raw_task}\nIntent: {task.intent_summary}\n\nCandidate files:\n"

    verdict_map, omitted = run_batched_judgment(
        non_seeds,
        system_prompt=SCOPE_JUDGMENT_SYSTEM,
        task_header=task_header,
        llm=llm,
        tokens_per_item=_TOKENS_PER_SCOPE_CANDIDATE,
        format_item=_format,
        extract_key=lambda j: j.get("path"),
        stage_name="scope",
        item_key=lambda sf: sf.path,
        default_action="irrelevant",
    )

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

        rp = context.retrieval_params
        candidates = expand_scope(
            task, kb, context.repo_id, plan_file_ids,
            max_deps=resolve_retrieval_param(rp, "max_deps"),
            max_co_changes=resolve_retrieval_param(rp, "max_co_changes"),
            max_metadata=resolve_retrieval_param(rp, "max_metadata"),
            max_keywords=resolve_retrieval_param(rp, "max_keywords"),
        )
        judged = judge_scope(candidates, task, llm, kb=kb)

        context.scoped_files = judged
        context.included_file_ids = {
            sf.file_id for sf in judged if sf.relevance == "relevant"
        }

        return context
