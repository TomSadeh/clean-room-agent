#!/usr/bin/env python3
"""Interactive retrieval pipeline debugger.

Runs the retrieval pipeline stage by stage, saving JSON snapshots between
stages and printing human-readable summaries. Imports and calls real
production code — not a mock or simulation.

Subcommands:
    run <task>                        Full step-through, snapshot after each stage
    rerun <stage> --from <dir>        Re-run one stage from saved snapshot
    inspect <dir>                     Print summaries from saved snapshots
    diff <dir_a> <dir_b>              Diff two runs

DB strategy: curated DB read-only, NO writes to raw DB or session DB.
Everything goes to filesystem JSON snapshots.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure the src directory is importable when running as a standalone script
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from clean_room_agent.config import load_config, require_models_config
from clean_room_agent.db.connection import get_connection
from clean_room_agent.environment import build_environment_brief, build_repo_file_tree
from clean_room_agent.llm.client import EnvironmentLLMClient, LoggedLLMClient
from clean_room_agent.llm.router import ModelRouter
from clean_room_agent.query.api import KnowledgeBase
from clean_room_agent.retrieval.context_assembly import (
    assemble_context,
    _classify_file_inclusions,
    _read_and_render_files,
)
from clean_room_agent.retrieval.dataclasses import BudgetConfig, TaskQuery
from clean_room_agent.retrieval.precision_stage import (
    classify_symbols,
    extract_precision_symbols,
)
from clean_room_agent.retrieval.routing import route_stages
from clean_room_agent.retrieval.scope_stage import expand_scope, judge_scope
from clean_room_agent.retrieval.similarity_stage import (
    assign_groups,
    find_similar_pairs,
    judge_similarity,
)
from clean_room_agent.retrieval.stage import (
    StageContext,
    get_stage_descriptions,
    resolve_retrieval_param,
)
from clean_room_agent.retrieval.task_analysis import (
    analyze_task,
    enrich_task_intent,
    extract_task_signals,
    resolve_seeds,
)
from clean_room_agent.trace import TraceLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """Handle non-serializable types for JSON dumps."""
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _dump_json(data, path: Path) -> None:
    """Write data to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def _load_json(path: Path):
    """Load JSON from a file. Raises FileNotFoundError if missing."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary(text: str, path: Path) -> None:
    """Write summary text to file AND print to stdout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(text)


# ---------------------------------------------------------------------------
# Serialization helpers for intermediate data
# ---------------------------------------------------------------------------

def _scoped_file_to_dict(sf) -> dict:
    return {
        "file_id": sf.file_id, "path": sf.path, "language": sf.language,
        "tier": sf.tier, "relevance": sf.relevance, "reason": sf.reason,
    }


def _classified_symbol_to_dict(cs) -> dict:
    return {
        "symbol_id": cs.symbol_id, "file_id": cs.file_id, "name": cs.name,
        "kind": cs.kind, "start_line": cs.start_line, "end_line": cs.end_line,
        "detail_level": cs.detail_level, "reason": cs.reason,
        "signature": cs.signature, "group_id": cs.group_id,
        "file_source": cs.file_source,
    }


def _pair_to_dict(p: dict) -> dict:
    """Serialize a similarity pair (sym_a/sym_b are ClassifiedSymbol objects)."""
    return {
        "pair_id": p["pair_id"],
        "sym_a": _classified_symbol_to_dict(p["sym_a"]),
        "sym_b": _classified_symbol_to_dict(p["sym_b"]),
        "score": p["score"],
        "signals": p["signals"],
    }


def _confirmed_pair_to_dict(p: dict) -> dict:
    return {
        "pair_id": p["pair_id"],
        "sym_a": _classified_symbol_to_dict(p["sym_a"]),
        "sym_b": _classified_symbol_to_dict(p["sym_b"]),
        "group_label": p.get("group_label", ""),
        "reason": p.get("reason", ""),
    }


# ---------------------------------------------------------------------------
# DebugSession
# ---------------------------------------------------------------------------

class DebugSession:
    """Holds shared state for a debug retrieval run."""

    def __init__(self, repo_path: Path, config: dict, budget: BudgetConfig):
        self.repo_path = repo_path
        self.config = config
        self.budget = budget

        # Open curated DB read-only
        self.curated_conn = get_connection("curated", repo_path=repo_path, read_only=True)
        self.kb = KnowledgeBase(self.curated_conn)

        # Resolve repo_id
        row = self.curated_conn.execute(
            "SELECT id FROM repos WHERE path = ?", (str(repo_path),)
        ).fetchone()
        if not row:
            raise RuntimeError(f"No indexed repo at {repo_path}. Run 'cra index' first.")
        self.repo_id = row["id"]

        # Model router
        models_config = require_models_config(config)
        self.router = ModelRouter(models_config)

        # Environment brief + file tree
        self.env_brief = build_environment_brief(config, self.kb, self.repo_id)
        self.repo_file_tree = build_repo_file_tree(self.kb, self.repo_id)
        self.brief_text = self.env_brief.to_prompt_text()

        # Retrieval params
        self.retrieval_params = config.get("retrieval", {})

    def close(self):
        self.curated_conn.close()

    # -- Stage runners (each returns artifacts dict) --

    def run_task_analysis(self, raw_task: str, task_id: str, mode: str) -> dict:
        """Run task analysis: extract signals, resolve seeds, enrich intent."""
        # Step 1: deterministic extraction
        signals = extract_task_signals(raw_task)

        # Step 2: resolve seeds
        max_sym = resolve_retrieval_param(self.retrieval_params, "max_symbol_matches")
        file_ids, symbol_ids = resolve_seeds(
            signals, self.kb, self.repo_id, max_symbol_matches=max_sym,
        )

        # Step 3: LLM intent enrichment
        reasoning_config = self.router.resolve("reasoning", "task_analysis")
        with LoggedLLMClient(reasoning_config) as llm:
            start = time.monotonic()
            intent = enrich_task_intent(
                raw_task, signals, llm,
                repo_file_tree=self.repo_file_tree,
                environment_brief=self.brief_text,
            )
            elapsed = int((time.monotonic() - start) * 1000)
            llm_calls = llm.flush()

        task_query = TaskQuery(
            raw_task=raw_task, task_id=task_id, mode=mode, repo_id=self.repo_id,
            mentioned_files=signals["files"], mentioned_symbols=signals["symbols"],
            keywords=signals["keywords"], error_patterns=signals["error_patterns"],
            task_type=signals["task_type"], intent_summary=intent,
            seed_file_ids=file_ids, seed_symbol_ids=symbol_ids,
        )

        # Build seeds info with paths
        seeds_info = []
        for fid in file_ids:
            f = self.kb.get_file_by_id(fid)
            seeds_info.append({"file_id": fid, "path": f.path if f else "?"})
        for sid in symbol_ids:
            s = self.kb.get_symbol_by_id(sid)
            if s:
                f = self.kb.get_file_by_id(s.file_id)
                seeds_info.append({
                    "symbol_id": sid, "name": s.name,
                    "file_path": f.path if f else "?",
                })

        return {
            "task_query": task_query,
            "signals": signals,
            "seeds": seeds_info,
            "llm_calls": llm_calls,
            "elapsed_ms": elapsed,
        }

    def run_stage_routing(self, task_query: TaskQuery, stage_names: list[str]) -> dict:
        """Run stage routing: LLM selects which stages to run."""
        available = {
            name: desc for name, desc in get_stage_descriptions().items()
            if name in stage_names
        }
        routing_config = self.router.resolve("reasoning", "stage_routing")
        with LoggedLLMClient(routing_config) as base_llm:
            llm = EnvironmentLLMClient(base_llm, self.brief_text)
            start = time.monotonic()
            selected, reasoning = route_stages(task_query, available, llm)
            elapsed = int((time.monotonic() - start) * 1000)
            llm_calls = llm.flush()

        return {
            "selected_stages": selected,
            "reasoning": reasoning,
            "available_stages": list(available.keys()),
            "llm_calls": llm_calls,
            "elapsed_ms": elapsed,
        }

    def run_scope(self, task_query: TaskQuery, context: StageContext) -> dict:
        """Run scope stage: deterministic expansion + LLM judgment, separately."""
        rp = self.retrieval_params
        plan_file_ids = [
            sf.file_id for sf in context.scoped_files if sf.tier == 0
        ] or None

        # Step 1: deterministic expansion
        start = time.monotonic()
        candidates = expand_scope(
            task_query, self.kb, self.repo_id, plan_file_ids,
            max_deps=resolve_retrieval_param(rp, "max_deps"),
            max_co_changes=resolve_retrieval_param(rp, "max_co_changes"),
            max_metadata=resolve_retrieval_param(rp, "max_metadata"),
            max_keywords=resolve_retrieval_param(rp, "max_keywords"),
        )
        expand_elapsed = int((time.monotonic() - start) * 1000)

        # Snapshot candidates BEFORE LLM judgment
        candidates_snapshot = [_scoped_file_to_dict(sf) for sf in candidates]

        # Step 2: LLM judgment
        scope_config = self.router.resolve(
            "classifier" if self.router.has_role("classifier") else "reasoning",
            "scope",
        )
        binary = self.router.has_role("classifier")
        with LoggedLLMClient(scope_config) as base_llm:
            llm = EnvironmentLLMClient(base_llm, self.brief_text)
            start = time.monotonic()
            judged = judge_scope(candidates, task_query, llm, kb=self.kb, binary=binary)
            judge_elapsed = int((time.monotonic() - start) * 1000)
            llm_calls = llm.flush()

        # Update context
        context.scoped_files = judged
        context.included_file_ids = {
            sf.file_id for sf in judged if sf.relevance == "relevant"
        }

        return {
            "candidates": candidates_snapshot,
            "judged": [_scoped_file_to_dict(sf) for sf in judged],
            "input_context": context.to_dict(),
            "llm_calls": llm_calls,
            "expand_elapsed_ms": expand_elapsed,
            "judge_elapsed_ms": judge_elapsed,
            "context": context,
        }

    def run_precision(self, task_query: TaskQuery, context: StageContext) -> dict:
        """Run precision stage: symbol extraction + LLM classification, separately."""
        rp = self.retrieval_params

        # Step 1: deterministic extraction
        start = time.monotonic()
        candidates = extract_precision_symbols(
            context.included_file_ids, task_query, self.kb,
            max_callees=resolve_retrieval_param(rp, "max_callees"),
            max_callers=resolve_retrieval_param(rp, "max_callers"),
        )
        extract_elapsed = int((time.monotonic() - start) * 1000)

        # Snapshot candidates BEFORE LLM classification
        candidates_snapshot = list(candidates)  # list of dicts already

        # Step 2: LLM classification
        precision_config = self.router.resolve("reasoning", "precision")
        with LoggedLLMClient(precision_config) as base_llm:
            llm = EnvironmentLLMClient(base_llm, self.brief_text)
            start = time.monotonic()
            classified = classify_symbols(candidates, task_query, llm, kb=self.kb)
            classify_elapsed = int((time.monotonic() - start) * 1000)
            llm_calls = llm.flush()

        context.classified_symbols = classified

        return {
            "candidates": candidates_snapshot,
            "classified": [_classified_symbol_to_dict(cs) for cs in classified],
            "input_context": context.to_dict(),
            "llm_calls": llm_calls,
            "extract_elapsed_ms": extract_elapsed,
            "classify_elapsed_ms": classify_elapsed,
            "context": context,
        }

    def run_similarity(self, task_query: TaskQuery, context: StageContext) -> dict:
        """Run similarity stage: find pairs + LLM judgment + group assignment."""
        rp = self.retrieval_params

        if not context.classified_symbols:
            return {
                "pairs": [], "confirmed": [], "groups": {},
                "input_context": context.to_dict(),
                "llm_calls": [], "elapsed_ms": 0, "context": context,
            }

        # Step 1: deterministic pair finding
        start = time.monotonic()
        pairs = find_similar_pairs(
            context.classified_symbols, self.kb,
            max_pairs=resolve_retrieval_param(rp, "max_candidate_pairs"),
            min_score=resolve_retrieval_param(rp, "min_composite_score"),
        )
        find_elapsed = int((time.monotonic() - start) * 1000)

        if not pairs:
            return {
                "pairs": [], "confirmed": [], "groups": {},
                "input_context": context.to_dict(),
                "llm_calls": [], "find_elapsed_ms": find_elapsed,
                "context": context,
            }

        # Snapshot pairs BEFORE LLM judgment
        pairs_snapshot = [_pair_to_dict(p) for p in pairs]

        # Step 2: LLM judgment
        sim_config = self.router.resolve(
            "classifier" if self.router.has_role("classifier") else "reasoning",
            "similarity",
        )
        binary = self.router.has_role("classifier")
        with LoggedLLMClient(sim_config) as base_llm:
            llm = EnvironmentLLMClient(base_llm, self.brief_text)
            start = time.monotonic()
            confirmed = judge_similarity(pairs, task_query, llm, binary=binary)
            judge_elapsed = int((time.monotonic() - start) * 1000)
            llm_calls = llm.flush()

        # Step 3: group assignment
        groups = {}
        if confirmed:
            groups = assign_groups(
                confirmed,
                max_group_size=resolve_retrieval_param(rp, "max_group_size"),
            )
            for cs in context.classified_symbols:
                gid = groups.get(cs.symbol_id)
                if gid is not None:
                    cs.group_id = gid

        return {
            "pairs": pairs_snapshot,
            "confirmed": [_confirmed_pair_to_dict(p) for p in confirmed],
            "groups": groups,
            "input_context": context.to_dict(),
            "llm_calls": llm_calls,
            "find_elapsed_ms": find_elapsed,
            "judge_elapsed_ms": judge_elapsed,
            "context": context,
        }

    def run_assembly(self, task_query: TaskQuery, context: StageContext) -> dict:
        """Run context assembly with optional LLM refilter."""
        assembly_config = self.router.resolve("reasoning")
        with LoggedLLMClient(assembly_config) as base_llm:
            llm = EnvironmentLLMClient(base_llm, self.brief_text)
            start = time.monotonic()
            package = assemble_context(
                context, self.budget, self.repo_path, llm=llm, kb=self.kb,
            )
            elapsed = int((time.monotonic() - start) * 1000)
            llm_calls = llm.flush()

        # Build output package metadata
        package_meta = {
            "files": [
                {
                    "file_id": fc.file_id, "path": fc.path,
                    "language": fc.language, "detail_level": fc.detail_level,
                    "token_estimate": fc.token_estimate,
                    "included_symbols": fc.included_symbols,
                }
                for fc in package.files
            ],
            "total_token_estimate": package.total_token_estimate,
            "budget_effective": self.budget.effective_budget,
            "budget_remaining": package.metadata["budget_remaining"],
            "files_considered": package.metadata["files_considered"],
            "files_included": package.metadata["files_included"],
            "assembly_decisions": package.metadata["assembly_decisions"],
        }

        return {
            "input_context": context.to_dict(),
            "output_package": package_meta,
            "llm_calls": llm_calls,
            "elapsed_ms": elapsed,
            "package": package,
        }


# ---------------------------------------------------------------------------
# Summary formatters
# ---------------------------------------------------------------------------

def _llm_call_stats(calls: list[dict]) -> str:
    """Format LLM call statistics."""
    if not calls:
        return "  LLM calls: 0"
    total_prompt = sum(c.get("prompt_tokens") or 0 for c in calls)
    total_completion = sum(c.get("completion_tokens") or 0 for c in calls)
    total_ms = sum(c.get("elapsed_ms") or 0 for c in calls)
    errors = sum(1 for c in calls if c.get("error"))
    lines = [
        f"  LLM calls: {len(calls)}",
        f"  Prompt tokens: {total_prompt}",
        f"  Completion tokens: {total_completion}",
        f"  Total latency: {total_ms}ms",
    ]
    if errors:
        lines.append(f"  Errors: {errors}")
    return "\n".join(lines)


def format_task_analysis_summary(artifacts: dict) -> str:
    signals = artifacts["signals"]
    seeds = artifacts["seeds"]
    tq = artifacts["task_query"]
    lines = [
        "=" * 60,
        "TASK ANALYSIS",
        "=" * 60,
        f"  Task type: {tq.task_type}",
        f"  Intent: {tq.intent_summary}",
        f"  Mentioned files: {signals['files']}",
        f"  Mentioned symbols: {signals['symbols']}",
        f"  Keywords: {signals['keywords']}",
        f"  Error patterns: {signals['error_patterns']}",
        f"  Resolved seeds: {len(seeds)}",
    ]
    for s in seeds:
        if "name" in s:
            lines.append(f"    symbol {s['symbol_id']}: {s['name']} in {s['file_path']}")
        else:
            lines.append(f"    file {s['file_id']}: {s['path']}")
    lines.append(_llm_call_stats(artifacts["llm_calls"]))
    return "\n".join(lines)


def format_routing_summary(artifacts: dict) -> str:
    lines = [
        "=" * 60,
        "STAGE ROUTING",
        "=" * 60,
        f"  Available: {artifacts['available_stages']}",
        f"  Selected: {artifacts['selected_stages']}",
        f"  Reasoning: {artifacts['reasoning']}",
        _llm_call_stats(artifacts["llm_calls"]),
    ]
    return "\n".join(lines)


def format_scope_summary(artifacts: dict) -> str:
    candidates = artifacts["candidates"]
    judged = artifacts["judged"]
    relevant = [f for f in judged if f["relevance"] == "relevant"]
    irrelevant = [f for f in judged if f["relevance"] == "irrelevant"]

    # Count by tier
    tier_counts = {}
    for c in candidates:
        t = c["tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1

    lines = [
        "=" * 60,
        "SCOPE",
        "=" * 60,
        f"  Total candidates: {len(candidates)}",
        f"  By tier: {dict(sorted(tier_counts.items()))}",
        f"  Relevant: {len(relevant)}, Irrelevant: {len(irrelevant)}",
        "",
        "  Top relevant files:",
    ]
    for f in relevant[:10]:
        lines.append(f"    + {f['path']} (tier={f['tier']}) {f['reason']}")
    if len(relevant) > 10:
        lines.append(f"    ... and {len(relevant) - 10} more")
    lines.append("")
    lines.append("  Top irrelevant files:")
    for f in irrelevant[:5]:
        lines.append(f"    - {f['path']} (tier={f['tier']}) {f['reason']}")
    if len(irrelevant) > 5:
        lines.append(f"    ... and {len(irrelevant) - 5} more")
    lines.append(_llm_call_stats(artifacts["llm_calls"]))
    return "\n".join(lines)


def format_precision_summary(artifacts: dict) -> str:
    classified = artifacts["classified"]
    by_level = {}
    for cs in classified:
        lvl = cs["detail_level"]
        by_level.setdefault(lvl, []).append(cs)

    lines = [
        "=" * 60,
        "PRECISION",
        "=" * 60,
        f"  Total candidates: {len(artifacts['candidates'])}",
        f"  Classified: {len(classified)}",
    ]
    for lvl in ("primary", "supporting", "type_context", "excluded"):
        syms = by_level.get(lvl, [])
        lines.append(f"  {lvl}: {len(syms)}")

    # Show primary symbols
    primary = by_level.get("primary", [])
    if primary:
        lines.append("")
        lines.append("  Primary symbols:")
        for cs in primary[:15]:
            lines.append(f"    * {cs['name']} ({cs['kind']}) in file_id={cs['file_id']} — {cs['reason']}")
        if len(primary) > 15:
            lines.append(f"    ... and {len(primary) - 15} more")

    # Derive file detail levels
    file_levels: dict[int, str] = {}
    priority = {"primary": 0, "supporting": 1, "type_context": 2}
    for cs in classified:
        if cs["detail_level"] == "excluded":
            continue
        fid = cs["file_id"]
        current = file_levels.get(fid)
        if current is None or priority[cs["detail_level"]] < priority[current]:
            file_levels[fid] = cs["detail_level"]
    lines.append(f"\n  File detail levels derived: {len(file_levels)}")
    for fid, lvl in sorted(file_levels.items(), key=lambda x: priority[x[1]]):
        lines.append(f"    file_id={fid}: {lvl}")

    lines.append(_llm_call_stats(artifacts["llm_calls"]))
    return "\n".join(lines)


def format_similarity_summary(artifacts: dict) -> str:
    pairs = artifacts["pairs"]
    confirmed = artifacts["confirmed"]
    groups = artifacts["groups"]

    lines = [
        "=" * 60,
        "SIMILARITY",
        "=" * 60,
        f"  Candidate pairs: {len(pairs)}",
        f"  Confirmed pairs: {len(confirmed)}",
        f"  Groups formed: {len(set(groups.values())) if groups else 0}",
    ]
    for p in confirmed[:10]:
        lines.append(f"    {p['sym_a']['name']} <-> {p['sym_b']['name']} — {p['reason']}")
    if len(confirmed) > 10:
        lines.append(f"    ... and {len(confirmed) - 10} more")
    lines.append(_llm_call_stats(artifacts["llm_calls"]))
    return "\n".join(lines)


def format_assembly_summary(artifacts: dict) -> str:
    pkg = artifacts["output_package"]
    files = pkg["files"]
    total = pkg["total_token_estimate"]
    effective = pkg["budget_effective"]
    utilization = (total / effective * 100) if effective > 0 else 0

    lines = [
        "=" * 60,
        "ASSEMBLY",
        "=" * 60,
        f"  Files considered: {pkg['files_considered']}",
        f"  Files included: {pkg['files_included']}",
        f"  Total tokens: {total}",
        f"  Budget: {total}/{effective} ({utilization:.1f}%)",
        "",
        "  Included files:",
    ]
    for f in files:
        lines.append(
            f"    {f['path']} [{f['detail_level']}] ~{f['token_estimate']} tokens"
        )

    # Show dropped files from assembly decisions
    dropped = [d for d in pkg["assembly_decisions"] if not d["included"]]
    if dropped:
        lines.append("")
        lines.append("  Dropped files:")
        for d in dropped:
            lines.append(f"    file_id={d['file_id']}: {d['reason']}")

    lines.append(_llm_call_stats(artifacts["llm_calls"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

STAGE_ORDER = ["scope", "precision", "similarity"]
STAGE_INDEX = {name: i for i, name in enumerate(STAGE_ORDER)}


def cmd_run(args):
    """Full step-through: run all stages, snapshot after each."""
    repo_path = Path(args.repo).resolve()
    config = load_config(repo_path)
    if config is None:
        print(f"ERROR: No config at {repo_path / '.clean_room' / 'config.toml'}", file=sys.stderr)
        sys.exit(1)

    # Budget
    budget_section = config.get("budget", {})
    models_section = require_models_config(config)
    cw = budget_section.get("context_window") or models_section["context_window"]
    if isinstance(cw, dict):
        cw = max(cw.values())
    reserved = budget_section.get("reserved_tokens", 4096)
    budget = BudgetConfig(context_window=cw, reserved_tokens=reserved)

    # Available stages from config
    stages_section = config.get("stages", {})
    stage_names = [s.strip() for s in stages_section.get("default", "scope,precision").split(",")]

    task_id = f"debug_{uuid.uuid4().hex[:8]}"
    mode = args.mode
    raw_task = args.task

    # Create output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = repo_path / ".clean_room" / "debug" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    session = DebugSession(repo_path, config, budget)
    trace_logger = TraceLogger(out_dir / "trace.md", task_id, raw_task)

    try:
        # --- Run meta ---
        print(f"\nDebug run: {out_dir}")
        print(f"Task: {raw_task}")
        print(f"Mode: {mode}, Budget: {budget.effective_budget} tokens")
        print(f"Configured stages: {stage_names}\n")

        # --- 00: Task Analysis ---
        print("Running task analysis...")
        ta = session.run_task_analysis(raw_task, task_id, mode)
        task_query = ta["task_query"]

        stage_dir = out_dir / "00_task_analysis"
        _dump_json(task_query.to_dict(), stage_dir / "output.json")
        _dump_json(ta["signals"], stage_dir / "signals.json")
        _dump_json(ta["seeds"], stage_dir / "seeds.json")
        _dump_json(ta["llm_calls"], stage_dir / "llm_calls.json")
        trace_logger.log_calls("task_analysis", "task_analysis", ta["llm_calls"],
                               session.router.resolve("reasoning", "task_analysis").model)

        summary = format_task_analysis_summary(ta)
        _write_summary(summary, stage_dir / "summary.txt")
        print()

        # --- 01: Stage Routing ---
        print("Running stage routing...")
        routing = session.run_stage_routing(task_query, stage_names)
        selected_stages = routing["selected_stages"]

        stage_dir = out_dir / "01_stage_routing"
        _dump_json(routing, stage_dir / "output.json")
        _dump_json(routing["llm_calls"], stage_dir / "llm_calls.json")
        trace_logger.log_calls("stage_routing", "stage_routing", routing["llm_calls"],
                               session.router.resolve("reasoning", "stage_routing").model)

        summary = format_routing_summary(routing)
        _write_summary(summary, stage_dir / "summary.txt")
        print()

        # --- Initialize context ---
        context = StageContext(
            task=task_query, repo_id=session.repo_id, repo_path=str(repo_path),
        )
        context.retrieval_params = session.retrieval_params

        # --- Run selected stages ---
        stage_num = 2
        for stage_name in selected_stages:
            stage_prefix = f"{stage_num:02d}_{stage_name}"
            stage_dir = out_dir / stage_prefix
            input_ctx_snapshot = context.to_dict()

            print(f"Running {stage_name}...")

            if stage_name == "scope":
                result = session.run_scope(task_query, context)
                context = result["context"]
                _dump_json(input_ctx_snapshot, stage_dir / "input_context.json")
                _dump_json(context.to_dict(), stage_dir / "output_context.json")
                _dump_json(result["candidates"], stage_dir / "candidates.json")
                _dump_json(result["llm_calls"], stage_dir / "llm_calls.json")
                summary = format_scope_summary(result)

            elif stage_name == "precision":
                result = session.run_precision(task_query, context)
                context = result["context"]
                _dump_json(input_ctx_snapshot, stage_dir / "input_context.json")
                _dump_json(context.to_dict(), stage_dir / "output_context.json")
                _dump_json(result["candidates"], stage_dir / "candidates.json")
                _dump_json(result["llm_calls"], stage_dir / "llm_calls.json")
                summary = format_precision_summary(result)

            elif stage_name == "similarity":
                result = session.run_similarity(task_query, context)
                context = result["context"]
                _dump_json(input_ctx_snapshot, stage_dir / "input_context.json")
                _dump_json(context.to_dict(), stage_dir / "output_context.json")
                _dump_json(result["pairs"], stage_dir / "pairs.json")
                _dump_json(result["confirmed"], stage_dir / "confirmed.json")
                _dump_json(result["llm_calls"], stage_dir / "llm_calls.json")
                summary = format_similarity_summary(result)

            else:
                print(f"  Unknown stage {stage_name}, skipping")
                stage_num += 1
                continue

            trace_logger.log_calls(stage_name, stage_name, result["llm_calls"],
                                   session.router.resolve(
                                       "classifier" if stage_name in ("scope", "similarity")
                                       and session.router.has_role("classifier")
                                       else "reasoning", stage_name).model)
            _write_summary(summary, stage_dir / "summary.txt")
            print()
            stage_num += 1

        # --- Assembly ---
        print("Running assembly...")
        stage_dir = out_dir / f"{stage_num:02d}_assembly"
        asm = session.run_assembly(task_query, context)
        _dump_json(asm["input_context"], stage_dir / "input_context.json")
        _dump_json(asm["output_package"], stage_dir / "output_package.json")
        _dump_json(asm["llm_calls"], stage_dir / "llm_calls.json")
        trace_logger.log_calls("assembly", "assembly_refilter", asm["llm_calls"],
                               session.router.resolve("reasoning").model)

        summary = format_assembly_summary(asm)
        _write_summary(summary, stage_dir / "summary.txt")
        print()

        # --- Save run meta ---
        _dump_json({
            "task": raw_task, "task_id": task_id, "mode": mode,
            "repo_path": str(repo_path),
            "budget": {"context_window": budget.context_window, "reserved_tokens": budget.reserved_tokens},
            "configured_stages": stage_names,
            "selected_stages": selected_stages,
            "timestamp": timestamp,
        }, out_dir / "run_meta.json")

        # --- Finalize trace ---
        trace_path = trace_logger.finalize()
        print(f"Trace written to: {trace_path}")
        print(f"Snapshots saved to: {out_dir}")

    finally:
        session.close()


# ---------------------------------------------------------------------------
# Subcommand: rerun
# ---------------------------------------------------------------------------

def cmd_rerun(args):
    """Re-run a single stage from a saved snapshot directory."""
    from_dir = Path(args.from_dir).resolve()
    stage_name = args.stage

    # Load run meta
    meta = _load_json(from_dir / "run_meta.json")
    repo_path = Path(meta["repo_path"])
    config = load_config(repo_path)
    if config is None:
        print("ERROR: No config found", file=sys.stderr)
        sys.exit(1)

    budget_data = meta["budget"]
    budget = BudgetConfig(
        context_window=budget_data["context_window"],
        reserved_tokens=budget_data["reserved_tokens"],
    )
    task_query_data = _load_json(from_dir / "00_task_analysis" / "output.json")
    task_query = TaskQuery.from_dict(task_query_data)

    session = DebugSession(repo_path, config, budget)

    # Create new output directory for the rerun
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = repo_path / ".clean_room" / "debug" / f"{timestamp}_rerun_{stage_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy task analysis from source
    ta_src = from_dir / "00_task_analysis"
    ta_dst = out_dir / "00_task_analysis"
    ta_dst.mkdir(parents=True, exist_ok=True)
    for f in ta_src.iterdir():
        if f.is_file():
            (ta_dst / f.name).write_bytes(f.read_bytes())

    try:
        # Find the stage's input_context from the previous run
        # Look for a directory matching the stage name
        stage_dirs = sorted(from_dir.iterdir())
        prev_stage_dir = None
        for d in stage_dirs:
            if d.is_dir() and stage_name in d.name:
                prev_stage_dir = d
                break

        if prev_stage_dir is None:
            print(f"ERROR: Stage '{stage_name}' not found in {from_dir}", file=sys.stderr)
            sys.exit(1)

        # Load input context from the previous run's stage directory
        input_ctx_path = prev_stage_dir / "input_context.json"
        if not input_ctx_path.exists():
            print(f"ERROR: No input_context.json in {prev_stage_dir}", file=sys.stderr)
            sys.exit(1)

        input_ctx_data = _load_json(input_ctx_path)
        context = StageContext.from_dict(input_ctx_data, task_query)
        context.retrieval_params = session.retrieval_params

        stage_dir = out_dir / prev_stage_dir.name
        print(f"Re-running {stage_name} from {from_dir.name}...")

        if stage_name == "scope":
            result = session.run_scope(task_query, context)
            _dump_json(input_ctx_data, stage_dir / "input_context.json")
            _dump_json(result["context"].to_dict(), stage_dir / "output_context.json")
            _dump_json(result["candidates"], stage_dir / "candidates.json")
            _dump_json(result["llm_calls"], stage_dir / "llm_calls.json")
            summary = format_scope_summary(result)

        elif stage_name == "precision":
            result = session.run_precision(task_query, context)
            _dump_json(input_ctx_data, stage_dir / "input_context.json")
            _dump_json(result["context"].to_dict(), stage_dir / "output_context.json")
            _dump_json(result["candidates"], stage_dir / "candidates.json")
            _dump_json(result["llm_calls"], stage_dir / "llm_calls.json")
            summary = format_precision_summary(result)

        elif stage_name == "similarity":
            result = session.run_similarity(task_query, context)
            _dump_json(input_ctx_data, stage_dir / "input_context.json")
            _dump_json(result["context"].to_dict(), stage_dir / "output_context.json")
            _dump_json(result["pairs"], stage_dir / "pairs.json")
            _dump_json(result["confirmed"], stage_dir / "confirmed.json")
            _dump_json(result["llm_calls"], stage_dir / "llm_calls.json")
            summary = format_similarity_summary(result)

        elif stage_name == "assembly":
            result = session.run_assembly(task_query, context)
            _dump_json(input_ctx_data, stage_dir / "input_context.json")
            _dump_json(result["output_package"], stage_dir / "output_package.json")
            _dump_json(result["llm_calls"], stage_dir / "llm_calls.json")
            summary = format_assembly_summary(result)

        else:
            print(f"ERROR: Unknown stage '{stage_name}'", file=sys.stderr)
            sys.exit(1)

        _write_summary(summary, stage_dir / "summary.txt")

        # Save rerun meta
        _dump_json({
            "source_run": str(from_dir),
            "rerun_stage": stage_name,
            "task": meta["task"], "task_id": meta["task_id"], "mode": meta["mode"],
            "repo_path": str(repo_path),
            "budget": budget_data,
            "timestamp": timestamp,
        }, out_dir / "run_meta.json")

        print(f"\nRerun snapshots saved to: {out_dir}")

    finally:
        session.close()


# ---------------------------------------------------------------------------
# Subcommand: inspect
# ---------------------------------------------------------------------------

def cmd_inspect(args):
    """Print summaries from saved snapshots."""
    run_dir = Path(args.dir).resolve()
    if not run_dir.exists():
        print(f"ERROR: Directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        meta = _load_json(meta_path)
        print(f"Run: {run_dir.name}")
        print(f"Task: {meta.get('task', '?')}")
        print(f"Mode: {meta.get('mode', '?')}")
        print(f"Stages: {meta.get('selected_stages', meta.get('configured_stages', '?'))}")
        print()

    # Find and print all summary.txt files in stage directories
    stage_dirs = sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    for stage_dir in stage_dirs:
        summary_path = stage_dir / "summary.txt"
        if summary_path.exists():
            print(summary_path.read_text(encoding="utf-8"))
            print()


# ---------------------------------------------------------------------------
# Subcommand: diff
# ---------------------------------------------------------------------------

def cmd_diff(args):
    """Compare two snapshot directories."""
    dir_a = Path(args.dir_a).resolve()
    dir_b = Path(args.dir_b).resolve()

    for d in (dir_a, dir_b):
        if not d.exists():
            print(f"ERROR: Directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    print(f"Comparing:")
    print(f"  A: {dir_a.name}")
    print(f"  B: {dir_b.name}")
    print()

    # Find common stage directories
    dirs_a = {d.name: d for d in sorted(dir_a.iterdir()) if d.is_dir()}
    dirs_b = {d.name: d for d in sorted(dir_b.iterdir()) if d.is_dir()}
    common = sorted(set(dirs_a) & set(dirs_b))
    only_a = sorted(set(dirs_a) - set(dirs_b))
    only_b = sorted(set(dirs_b) - set(dirs_a))

    if only_a:
        print(f"Only in A: {only_a}")
    if only_b:
        print(f"Only in B: {only_b}")
    if only_a or only_b:
        print()

    for stage_name in common:
        sa = dirs_a[stage_name]
        sb = dirs_b[stage_name]
        print(f"{'=' * 60}")
        print(f"DIFF: {stage_name}")
        print(f"{'=' * 60}")

        # Compare specific files based on stage type
        if "scope" in stage_name:
            _diff_scope(sa, sb)
        elif "precision" in stage_name:
            _diff_precision(sa, sb)
        elif "similarity" in stage_name:
            _diff_similarity(sa, sb)
        elif "assembly" in stage_name:
            _diff_assembly(sa, sb)
        elif "task_analysis" in stage_name:
            _diff_task_analysis(sa, sb)
        elif "routing" in stage_name:
            _diff_routing(sa, sb)
        else:
            print("  (no diff logic for this stage)")
        print()


def _diff_task_analysis(sa: Path, sb: Path):
    a = _load_json(sa / "output.json")
    b = _load_json(sb / "output.json")
    for key in ("task_type", "intent_summary", "mentioned_files", "keywords",
                "seed_file_ids", "seed_symbol_ids"):
        va, vb = a.get(key), b.get(key)
        if va != vb:
            print(f"  {key}:")
            print(f"    A: {va}")
            print(f"    B: {vb}")


def _diff_routing(sa: Path, sb: Path):
    a = _load_json(sa / "output.json")
    b = _load_json(sb / "output.json")
    sel_a = a.get("selected_stages", [])
    sel_b = b.get("selected_stages", [])
    if sel_a != sel_b:
        print(f"  Selected stages:")
        print(f"    A: {sel_a}")
        print(f"    B: {sel_b}")
    else:
        print(f"  Selected stages: same ({sel_a})")
    if a.get("reasoning") != b.get("reasoning"):
        print(f"  Reasoning differs")


def _diff_scope(sa: Path, sb: Path):
    oa = _load_json(sa / "output_context.json") if (sa / "output_context.json").exists() else {}
    ob = _load_json(sb / "output_context.json") if (sb / "output_context.json").exists() else {}

    files_a = {f["path"]: f for f in oa.get("scoped_files", [])}
    files_b = {f["path"]: f for f in ob.get("scoped_files", [])}

    rel_a = {p for p, f in files_a.items() if f["relevance"] == "relevant"}
    rel_b = {p for p, f in files_b.items() if f["relevance"] == "relevant"}

    added = rel_b - rel_a
    removed = rel_a - rel_b
    print(f"  Relevant files: A={len(rel_a)}, B={len(rel_b)}")
    if added:
        print(f"  Added in B ({len(added)}):")
        for p in sorted(added):
            print(f"    + {p}")
    if removed:
        print(f"  Removed in B ({len(removed)}):")
        for p in sorted(removed):
            print(f"    - {p}")
    if not added and not removed:
        print("  Relevant files: identical")


def _diff_precision(sa: Path, sb: Path):
    oa = _load_json(sa / "output_context.json") if (sa / "output_context.json").exists() else {}
    ob = _load_json(sb / "output_context.json") if (sb / "output_context.json").exists() else {}

    def _sym_key(cs):
        return (cs["name"], cs["file_id"], cs["start_line"])

    syms_a = {_sym_key(cs): cs for cs in oa.get("classified_symbols", [])}
    syms_b = {_sym_key(cs): cs for cs in ob.get("classified_symbols", [])}

    # Compare detail levels
    changed = []
    for key in sorted(set(syms_a) | set(syms_b)):
        a_lvl = syms_a[key]["detail_level"] if key in syms_a else "(missing)"
        b_lvl = syms_b[key]["detail_level"] if key in syms_b else "(missing)"
        if a_lvl != b_lvl:
            changed.append((key, a_lvl, b_lvl))

    print(f"  Symbols: A={len(syms_a)}, B={len(syms_b)}")
    if changed:
        print(f"  Changed classifications ({len(changed)}):")
        for (name, fid, line), a_lvl, b_lvl in changed[:20]:
            print(f"    {name} (file={fid}:{line}): {a_lvl} -> {b_lvl}")
        if len(changed) > 20:
            print(f"    ... and {len(changed) - 20} more")
    else:
        print("  Classifications: identical")


def _diff_similarity(sa: Path, sb: Path):
    ca = _load_json(sa / "confirmed.json") if (sa / "confirmed.json").exists() else []
    cb = _load_json(sb / "confirmed.json") if (sb / "confirmed.json").exists() else []
    print(f"  Confirmed pairs: A={len(ca)}, B={len(cb)}")

    def _pair_key(p):
        return (p["sym_a"]["name"], p["sym_b"]["name"])

    keys_a = {_pair_key(p) for p in ca}
    keys_b = {_pair_key(p) for p in cb}
    added = keys_b - keys_a
    removed = keys_a - keys_b
    if added:
        print(f"  New in B: {sorted(added)}")
    if removed:
        print(f"  Removed in B: {sorted(removed)}")
    if not added and not removed:
        print("  Confirmed pairs: identical")


def _diff_assembly(sa: Path, sb: Path):
    pa = _load_json(sa / "output_package.json") if (sa / "output_package.json").exists() else {}
    pb = _load_json(sb / "output_package.json") if (sb / "output_package.json").exists() else {}

    paths_a = {f["path"] for f in pa.get("files", [])}
    paths_b = {f["path"] for f in pb.get("files", [])}

    print(f"  Files: A={len(paths_a)}, B={len(paths_b)}")
    print(f"  Tokens: A={pa.get('total_token_estimate', '?')}, B={pb.get('total_token_estimate', '?')}")
    added = paths_b - paths_a
    removed = paths_a - paths_b
    if added:
        print(f"  Added in B: {sorted(added)}")
    if removed:
        print(f"  Removed in B: {sorted(removed)}")
    if not added and not removed:
        print("  Included files: identical")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive retrieval pipeline debugger",
    )
    parser.add_argument(
        "--repo", default=".",
        help="Repository path (default: current directory)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Full step-through with snapshots")
    p_run.add_argument("task", help="Task description string")
    p_run.add_argument("--mode", default="plan", choices=["plan", "implement"],
                       help="Pipeline mode (default: plan)")

    # rerun
    p_rerun = sub.add_parser("rerun", help="Re-run one stage from snapshot")
    p_rerun.add_argument("stage", help="Stage name to re-run")
    p_rerun.add_argument("--from", dest="from_dir", required=True,
                         help="Source snapshot directory")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Print summaries from snapshots")
    p_inspect.add_argument("dir", help="Snapshot directory to inspect")

    # diff
    p_diff = sub.add_parser("diff", help="Compare two snapshot directories")
    p_diff.add_argument("dir_a", help="First snapshot directory")
    p_diff.add_argument("dir_b", help="Second snapshot directory")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "run":
        cmd_run(args)
    elif args.command == "rerun":
        cmd_rerun(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "diff":
        cmd_diff(args)


if __name__ == "__main__":
    main()
