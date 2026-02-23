# Clean Room Agent

## What This Project Is

A self-improving coding agent built around the thesis: **the primary bottleneck in LLM application performance is not model capability but context curation.** The model's job is reasoning, not filtering. Clean the room before the model enters.

This is a standalone Python coding agent. No external platform dependency - the harness, retrieval pipeline, and training loop are all ours. The same N-prompt pipeline architecture serves multiple modes (planning, coding, training planning, data curation), creating a closed loop where the agent improves itself from its own logged activity. Bootstrapping from external repo commit histories breaks the chicken-and-egg dependency, allowing LoRA training before the agent produces any real runs.

## Core Architecture: The N-Prompt Pipeline

Instead of stuffing a 200K context window and hoping the model finds what matters, use a multi-stage pipeline where each prompt starts clean with curated context:

1. **Three-Database Architecture** - raw (append-only log of all activity and training corpus), curated (verified signals the model reads from), and session (ephemeral per-task working memory). Cold-startable from git history.
2. **Deterministic Pre-filtering + LLM Judgment** - deterministic methods (AST, deps, git, metadata queries) narrow candidates, then an LLM call per stage evaluates relevance. Not embedding similarity.
3. **Mode-Parameterized Pipeline** - variable-length sequence of retrieval stages followed by a terminal execute stage. The pipeline structure is constant; what changes per mode is the data source, retrieval stages, and execute output. Modes: Plan (structured plans), Implement (code edits), Train-Plan (training plans), Curate-Data (training datasets).
4. **Multi-Model Architecture** - two base models: Qwen2.5-Coder-3B for coding, Qwen3-4B for reasoning/planning. Config-only routing, no CLI model flags.
5. **Per-Stage LoRA Adapters** (Phase 4) - one per pipeline stage, fine-tuned from logged activity and synthetic data bootstrapped from external repo commit histories. The self-improvement loop.

Target: a 32K window at ~100% signal relevance, beating a 200K window at 10-15% utilization.

## Data Architecture: Three-Database Model

Three separate SQLite files, not three schemas in one file. Independent WAL journals, backups, and lifecycles.

1. **Raw DB** (`raw.sqlite`) - append-only log of everything: indexing runs, LLM enrichment outputs, retrieval decisions, stage LLM call results, solve attempts, validation results, training plans, curated datasets, adapter registry. This IS the training corpus. Every LLM call with full I/O is traceable to a final task outcome via `task_id`. Writers: runtime components from all phases. Readers: analysis and Phase 4 training pipeline.

2. **Curated DB** (`curated.sqlite`) - deterministic indexing output (AST, deps, git metadata) plus explicitly promoted LLM enrichment data plus active adapter metadata. This is the "clean room" the model reads from. Phase 1 indexing populates it directly. Phase 2 reads from it (read-only). Phase 3 has no direct curated DB access -- all context arrives via Phase 2's `ContextPackage`. Phase 4 writes only adapter metadata. Cold-startable from `cra index` alone.

3. **Session DB** (`session_<task_id>.sqlite`) - ephemeral per-task working memory. Created per task run, discarded after (optionally archived to raw). Intentionally minimal: key-value store. Phase 2 creates it, Phase 3 inherits and closes it.

**Connection factory**: `get_connection(role, task_id=None, read_only=False)` where role is `"curated"`, `"raw"`, or `"session"`. `read_only=True` is required for Phase 2 curated reads. Single point of DB management.

**Cold start**: `cra index` populates curated DB. Raw DB gets first real data from indexing run metadata. Session DB gets first real data in Phase 2/3.

**Enrichment data flow**: `cra enrich` writes LLM-generated metadata to raw DB (`enrichment_outputs` table). The `--promote` flag copies enrichment to curated DB (`file_metadata`). Raw entry is the permanent audit trail. Deterministic indexing data (AST, deps, git) writes directly to curated (verified by construction).

## N-Prompt Pipeline Design

The pipeline is mode-parameterized: `[Task Analysis] -> [Retrieval Stage 1, ..., N] -> [Execute]`. The core structure is constant across all modes. What changes per mode is the data source, retrieval stages, and execute output.

**Retrieval stages** each implement a common protocol: deterministic pre-filtering -> LLM judgment call. The deterministic part narrows candidates using structured queries (deps, co-change, AST for code modes; log queries for training modes). The LLM evaluates the candidates against the task, making the judgment call that code alone can't make. Each LLM call gets a clean context window with only what that stage needs. The pipeline runner sequences them, threading budget and session state through.

**The execute stage** is always terminal. Its output depends on mode: Plan mode produces structured plan artifacts, Implement mode produces code edits (search/replace), Train-Plan mode produces training plans, Curate-Data mode produces training datasets.

**Modes**: Plan (`cra plan`), Implement (`cra solve`), Train-Plan (`cra train-plan`), Curate-Data (`cra curate-data`). Plan and Implement are Phase 3 (MVP). Train-Plan and Curate-Data are Phase 4 (post-MVP).

**MVP configuration**: [Task Analysis] -> [Scope, Precision] -> Execute (4 prompts total). Task Analysis (always-run preamble) parses intent and identifies targets. Scope expands from seeds and filters by relevance. Precision extracts symbols and classifies detail levels. This is the initial configuration, not the architecture.

The pipeline architecture (stages, context curation, budget management) is model-agnostic -- nothing in the retrieval or orchestration layers depends on a specific model or provider. The LLM transport layer (`llm/client.py`) is Ollama-specific for MVP. Above it, a `ModelRouter` resolves which model to call based on (role, stage_name), reading from config. Swapping to a different provider means reimplementing `llm/client.py` internals; no other module changes.

## Repository Contents

```
planning/
  meta-plan.md                   - Single source of truth: phases, schemas, contracts, conventions
research_reviews/
  (research reviews and analysis documents)
archive/
  planning-v1/                   - Superseded v1 planning documents
  (archived notes and superseded research/context documents)
```

### Runtime Data Layout

```
.clean_room/
  config.toml                    - project-level settings including model routing (created by `cra init`)
  curated.sqlite                 - indexed knowledge base + adapter metadata (Phase 1 writes, Phase 2/3 read, Phase 4 writes adapters)
  raw.sqlite                     - append-only activity log + training corpus (written by all phases)
  sessions/
    session_<task_id>.sqlite     - ephemeral per-task working memory (created/destroyed per run)
```

**Config file** (`.clean_room/config.toml`): Project-level settings created by `cra init`. All model configuration (coding model, reasoning model, per-stage LoRA overrides) lives here -- no CLI model flags. Missing `[models]` section when an LLM-using command runs is a hard error.

## Development Principles

- **The room, not the model** - performance comes from what's in the context window
- **Results and capabilities, never mechanisms** - show what the system does, never explain how (competitive advantage)
- **Deterministic first, AI second** - metadata extraction before semantic search
- **Each token earns its place** - default-deny context architecture, not additive stuffing
- **Log everything, curate deliberately** - raw DB captures all activity; promotion to curated is an explicit, reviewed act
- **The loop closes** - logged activity becomes training data; the agent uses itself to plan its own improvement

### Coding Style (Development Mode)

- We are developing, not maintaining a product: optimize for fast debugging, not graceful degradation.
- No fallbacks and no hardcoded defaults in core logic.
- Keep `try/except` blocks minimal and intentional.
- When catching errors, add context and re-raise; do not silently recover.
- Prefer fail-fast behavior so incorrect assumptions break hard and early.

## Validation Note

Formal benchmarking and thesis validation are intentionally outside the active Phase 1-4 plan and can be redesigned later without constraining implementation.

## Status

Research and design phase. Next steps:
1. Build the knowledge base and indexer (Phase 1)
2. Build the retrieval pipeline (Phase 2)
3. Build the code agent with plan + implement modes (Phase 3) -- MVP boundary
4. Build the self-improvement loop (Phase 4) -- post-MVP

Phase 4's bootstrapping path (synthetic training data from external repos) depends only on Phase 1 -- LoRA training can begin before Phases 2-3 are built. The self-improvement path within Phase 4 requires Phase 3 logged data.
