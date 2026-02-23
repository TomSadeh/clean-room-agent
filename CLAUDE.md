# Clean Room Agent

## What This Project Is

A custom coding agent harness built around the thesis: **the primary bottleneck in LLM application performance is not model capability but context curation.** The model's job is reasoning, not filtering. Clean the room before the model enters.

This is a standalone Python coding agent. No external platform dependency - the harness and retrieval pipeline are all ours.

## Core Architecture: The N-Prompt Pipeline

Instead of stuffing a 200K context window and hoping the model finds what matters, use a multi-stage pipeline where each prompt starts clean with curated context:

1. **Three-Database Architecture** - raw (append-only log of all activity), curated (verified signals the model reads from), and session (ephemeral per-task working memory). Cold-startable from git history.
2. **Deterministic Pre-filtering + LLM Judgment** - deterministic methods (AST, deps, git, metadata queries) narrow candidates, then an LLM call per stage evaluates relevance. Not embedding similarity.
3. **N-Stage Prompt Pipeline** - variable-length sequence of retrieval stages followed by a terminal execute stage. Each stage is a genuine LLM prompt with a clean context window containing only what that stage needs. Pipeline length adapts to task/repo complexity. No conversation accumulation, no compaction.
4. **Per-Stage LoRA Adapters** (long-term) - one per pipeline stage, fine-tuned for that stage's job.

Target: a 32K window at ~100% signal relevance, beating a 200K window at 10-15% utilization.

## Data Architecture: Three-Database Model

Three separate SQLite files, not three schemas in one file. Independent WAL journals, backups, and lifecycles.

1. **Raw DB** (`raw.sqlite`) - append-only log of everything: indexing runs, LLM enrichment outputs, retrieval decisions, stage LLM call results, solve attempts, validation results. Training corpus and source of truth for analysis. Writers: runtime components from Phases 1-3. Readers: analysis and future fine-tuning pipelines.

2. **Curated DB** (`curated.sqlite`) - deterministic indexing output (AST, deps, git metadata) plus explicitly promoted LLM enrichment data. This is the "clean room" the model reads from. Phase 1 indexing populates it directly. Phase 2 and Phase 3 read from it but never write to it. Cold-startable from `cra index` alone.

3. **Session DB** (`session_<task_id>.sqlite`) - ephemeral per-task working memory. Created per solve run, discarded after (optionally archived to raw). Intentionally minimal: key-value retrieval state, staged working context, scratch notes. Phase 2 creates it, Phase 3 inherits and closes it.

**Connection factory**: `get_connection(role, task_id=None, read_only=False)` where role is `"curated"`, `"raw"`, or `"session"`. Single point of DB management.

**Cold start**: `cra index` populates curated DB. Raw DB gets first real data from indexing run metadata. Session DB gets first real data in Phase 2/3.

**Enrichment data flow**: `cra enrich` writes LLM-generated metadata to raw DB (`enrichment_outputs` table). The `--promote` flag copies enrichment to curated DB (`file_metadata`). Raw entry is the permanent audit trail. Deterministic indexing data (AST, deps, git) writes directly to curated (verified by construction).

## N-Prompt Pipeline Design

The pipeline is a variable-length sequence of retrieval stages followed by a terminal execute stage. The number of retrieval stages adapts to the task and repository — a simple rename in a small project may need only [Scope → Execute], while planning a complex feature across a large codebase may need [Scope → Dependency Analysis → Impact Assessment → Precision → Execute].

**Retrieval stages** each implement a common protocol: deterministic pre-filtering → LLM judgment call. The deterministic part narrows candidates using structured queries (deps, co-change, AST). The LLM evaluates the candidates against the task, making the judgment call that code alone can't make. Each LLM call gets a clean context window with only what that stage needs. Stages read from the curated DB, write working state to session DB, and log decisions (including LLM call inputs/outputs) to raw DB. The pipeline runner sequences them, threading budget and session state through.

**The execute stage** is always terminal: curated context + task → code generation. Pure reasoning, zero exploration.

**MVP configuration**: [Task Analysis, Scope, Precision] → Execute (4 prompts total). Task Analysis parses intent and identifies targets. Scope expands from seeds and filters by relevance. Precision extracts symbols and classifies detail levels. This is the initial configuration, not the architecture.

The pipeline architecture (stages, context curation, budget management) is model-agnostic — nothing in the retrieval or orchestration layers depends on a specific model or provider. The LLM transport layer (`llm/client.py`) is Ollama-specific for MVP. Swapping to a different provider means reimplementing `llm/client.py` internals; no other module changes.

## Repository Contents

```
planning/
  meta-plan.md                   - Top-level phase boundaries and gates
  phase1-knowledge-base.md       - Knowledge Base + Indexer (8 steps)
  phase2-retrieval-pipeline.md   - Retrieval pipeline build (6 steps)
  phase3-agent-harness.md        - Agent harness build (8 steps)
research_reviews/
  (research reviews and analysis documents)
archive/
  (archived notes and superseded research/context documents)
```

### Runtime Data Layout

```
.clean_room/
  config.toml                    - project-level settings (created by `cra init`)
  curated.sqlite                 - indexed knowledge base (Phase 1 writes, Phase 2/3 read)
  raw.sqlite                     - append-only activity log (written by Phase 1-3 runtime components)
  sessions/
    session_<task_id>.sqlite     - ephemeral per-task working memory (created/destroyed per run)
```

**Config file** (`.clean_room/config.toml`): Optional project-level settings created by `cra init`. In active development mode, required runtime inputs stay explicit at call sites (no fallback loading for missing required values).

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
