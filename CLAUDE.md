# Clean Room Agent

## What This Project Is

A custom coding agent harness built around the thesis: **the primary bottleneck in LLM application performance is not model capability but context curation.** The model's job is reasoning, not filtering. Clean the room before the model enters.

This is a standalone Python coding agent. No external platform dependency - the harness and retrieval pipeline are all ours.

## Core Architecture: The N-Prompt Pipeline

Instead of stuffing a 200K context window and hoping the model finds what matters, use a multi-stage pipeline where each prompt starts clean with curated context:

1. **Three-Database Architecture** - raw (append-only log of all activity), curated (verified signals the model reads from), and session (ephemeral per-task working memory). Cold-startable from git history.
2. **Deterministic Retrieval** - metadata extraction first, AI-assisted only for ambiguous items. Not embedding similarity.
3. **N-Stage Prompt Pipeline** - early stages filter and ground, later stages reason and execute. No conversation accumulation, no compaction.
4. **Per-Stage LoRA Adapters** (long-term) - one per pipeline stage, fine-tuned for that stage's job.

Target: a 32K window at ~100% signal relevance, beating a 200K window at 10-15% utilization.

## Data Architecture: Three-Database Model

Three separate SQLite files, not three schemas in one file. Independent WAL journals, backups, and lifecycles.

1. **Raw DB** (`raw.sqlite`) - append-only log of everything: indexing runs, retrieval decisions, solve attempts, LLM outputs, validation results. Training corpus and source of truth for analysis. Writers: runtime components from Phases 1-3. Readers: analysis and future fine-tuning pipelines.

2. **Curated DB** (`curated.sqlite`) - derived from raw, contains only verified/promoted signals. This is the "clean room" the model reads from. Phase 1 indexing populates it directly (AST, deps, git metadata). Phase 2 and Phase 3 read from it but never write to it. Cold-startable from `cra index` alone.

3. **Session DB** (`session_<task_id>.sqlite`) - ephemeral per-task working memory. Created per solve run, discarded after (optionally archived to raw). Intentionally minimal: key-value retrieval state, staged working context, scratch notes. Phase 2 creates it, Phase 3 inherits and closes it.

**Connection factory**: `get_connection(role, task_id=None)` where role is `"curated"`, `"raw"`, or `"session"`. Single point of DB management.

**Cold start**: `cra index` populates curated DB. Raw DB gets first real data from indexing run metadata. Session DB gets first real data in Phase 2/3.

**Raw->curated derivation**: Starts manual/scripted. No premature automation.

## MVP: Three-Prompt Strategy

The minimum viable experiment:

- **Stage 1 (Scope)**: Full repo + task -> 50-100 relevant files. Can be deterministic (AST + metadata heuristics).
- **Stage 2 (Precision)**: Scoped files + task -> exact context needed. Extracts signatures, types, tests, docs.
- **Stage 3 (Execute)**: Curated context + task -> code generation. Pure reasoning, zero exploration.

Designed to remain model-agnostic in implementation so model/provider selection can change without altering core architecture.

## Repository Contents

```
planning/
  meta-plan.md                   - Top-level phase boundaries and gates
  phase1-knowledge-base.md       - Knowledge Base + Indexer (8 steps)
  phase2-retrieval-pipeline.md   - Retrieval pipeline build (7 steps)
  phase3-agent-harness.md        - Agent harness build (8 steps)
archive/
  (archived notes and superseded research/context documents)
```

### Runtime Data Layout

```
.clean_room/
  curated.sqlite                 - indexed knowledge base (Phase 1 writes, Phase 2/3 read)
  raw.sqlite                     - append-only activity log (written by Phase 1-3 runtime components)
  sessions/
    session_<task_id>.sqlite     - ephemeral per-task working memory (created/destroyed per run)
```

## Development Principles

- **The room, not the model** - performance comes from what's in the context window
- **Results and capabilities, never mechanisms** - show what the system does, never explain how (competitive advantage)
- **Deterministic first, AI second** - metadata extraction before semantic search
- **Each token earns its place** - default-deny context architecture, not additive stuffing
- **Log everything, curate deliberately** - raw DB captures all activity; promotion to curated is an explicit, reviewed act

### Coding Style (Development Mode)

- We are developing, not maintaining a product: optimize for fast debugging, not graceful degradation.
- No fallbacks and no hardcoded defaults in core logic.
- Keep `try/except` blocks minimal and intentional.
- When catching errors, add context and re-raise; do not silently recover.
- Prefer fail-fast behavior so incorrect assumptions break hard and early.

## Validation Note

Formal benchmarking and thesis validation are intentionally outside the active Phase 1-3 plan and can be redesigned later without constraining implementation.

## Status

Research and design phase. Next steps:
1. Build the knowledge base and indexer (Phase 1)
2. Build the retrieval pipeline (Phase 2)
3. Build the standalone agent harness (Phase 3)
4. Revisit validation and benchmark planning after Phases 1-3 are stable







