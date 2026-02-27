# Clean Room Agent

## What This Project Is

A self-improving coding agent built around the thesis: **the primary bottleneck in LLM application performance is not model capability but context curation.** The model's job is reasoning, not filtering. Clean the room before the model enters.

This is a standalone Python coding agent. No external platform dependency - the harness, retrieval pipeline, and training loop are all ours. The same N-prompt pipeline architecture serves multiple modes (planning, coding, training planning, data curation), creating a closed loop where the agent improves itself from its own logged activity. Bootstrapping from external repo commit histories breaks the chicken-and-egg dependency, allowing LoRA training before the agent produces any real runs.

## Core Principle: Transparency Is Load-Bearing

Transparency is not a safety feature bolted on top. It is the mechanism by which the system produces results. Remove it and the system stops working.

Every LLM call is a fresh instance — if context isn't explicit, the call fails. If reasoning isn't logged, there's no training data for the self-improvement loop. If retrieval isn't deterministic, the next stage gets garbage. This is the same relationship as a methodology section in a paper: remove it and what remains isn't a less transparent paper — it's not a paper. It doesn't replicate, it doesn't build, nobody can extend it.

**The traceability test:** a human must be able to trace any output back through every decision that produced it using only the logs. If they can't, something is broken — fix the transparency, don't patch around it. The Three-Database Architecture, Context Curation Rules, and Coding Style below are all load-bearing implementations of this principle.

## Core Architecture: The N-Prompt Pipeline

Instead of stuffing a 200K context window and hoping the model finds what matters, use a multi-stage pipeline where each prompt starts clean with curated context:

1. **Three-Database Architecture** - raw (append-only log of all activity and training corpus), curated (verified signals the model reads from), and session (ephemeral per-task working memory). Cold-startable from git history.
2. **Deterministic Pre-filtering + LLM Judgment** - deterministic methods (AST, deps, git, metadata queries) narrow candidates, then an LLM call per stage evaluates relevance. Not embedding similarity.
3. **Mode-Parameterized Pipeline** - variable-length sequence of retrieval stages followed by a terminal execute stage. The pipeline structure is constant; what changes per mode is the data source, retrieval stages, and execute output. Modes: Plan (structured plans), Implement (code edits), Train-Plan (training plans), Curate-Data (training datasets).
4. **Multi-Model Architecture** - two base models: Qwen2.5-Coder-3B for coding, Qwen3-4B for reasoning/planning. Config-only routing, no CLI model flags.
5. **Per-Stage LoRA Adapters** (Phase 4) - one per pipeline stage, fine-tuned via teacher-student distillation (Qwen3.5-397B as primary teacher) and from logged activity. Planning is the critical path (hardest stage, no ground truth, requires CoT-SFT + DPO). Self-improvement guardrails: plateaus after ~2 iterations at sub-7B scale; sub-6B fails to bootstrap without external teacher signal.

Target: a 32K window at ~100% signal relevance, beating a 200K window at 10-15% utilization.

## Data Architecture: Three-Database Model

Three separate SQLite files, not three schemas in one file. Independent WAL journals, backups, and lifecycles.

1. **Raw DB** (`raw.sqlite`) - append-only log of everything: indexing runs, LLM enrichment outputs, retrieval decisions, stage LLM call results, solve attempts, validation results, training plans, curated datasets, adapter registry. This IS the training corpus. Every LLM call with full I/O is traceable to a final task outcome via `task_id`. Writers: runtime components from all phases. Readers: analysis and Phase 4 training pipeline.

2. **Curated DB** (`curated.sqlite`) - deterministic indexing output (AST, deps, git metadata) plus explicitly promoted LLM enrichment data plus active adapter metadata. This is the "clean room" the model reads from. Phase 1 indexing populates it directly. Phase 2 reads from it (read-only). Phase 3 has no direct curated DB access -- all context arrives via Phase 2's `ContextPackage`. Phase 4 writes only adapter metadata. Cold-startable from `cra index` alone.

3. **Session DB** (`session_<task_id>.sqlite`) - ephemeral per-task working memory. Created per task run, archived to raw DB (`session_archives`) and deleted after pipeline completion. Intentionally minimal: key-value store. Phase 2 creates it, Phase 3 inherits and closes it.

**Connection factory**: `get_connection(role, *, repo_path, task_id=None, read_only=False)` where role is `"curated"`, `"raw"`, or `"session"`. `repo_path` is keyword-only, points to the repository root. `read_only=True` is required for Phase 2 curated reads. Single point of DB management.

**Cold start**: `cra index` populates curated DB. Raw DB gets first real data from indexing run metadata. Session DB gets first real data in Phase 2/3.

**Enrichment data flow**: `cra enrich` writes LLM-generated metadata to raw DB (`enrichment_outputs` table). The `--promote` flag copies enrichment to curated DB (`file_metadata`). Raw entry is the permanent audit trail. Deterministic indexing data (AST, deps, git) writes directly to curated (verified by construction).

## N-Prompt Pipeline Design

The pipeline is mode-parameterized: `[Task Analysis] -> [Retrieval Stage 1, ..., N] -> [Execute]`. The core structure is constant across all modes. What changes per mode is the data source, retrieval stages, and execute output.

**Retrieval stages** each implement a common protocol: deterministic pre-filtering -> LLM judgment call. The deterministic part narrows candidates using structured queries (deps, co-change, AST for code modes; log queries for training modes). The LLM evaluates the candidates against the task, making the judgment call that code alone can't make. Each LLM call gets a clean context window with only what that stage needs. The pipeline runner sequences them, threading budget and session state through.

**The execute stage** is always terminal. Its output depends on mode: Plan mode produces structured plan artifacts, Implement mode produces code edits (search/replace), Train-Plan mode produces training plans, Curate-Data mode produces training datasets.

**Modes**: Plan (`cra plan`), Implement (`cra solve`), Train-Plan (`cra train-plan`), Curate-Data (`cra curate-data`). Plan and Implement are Phase 3 (MVP). Train-Plan and Curate-Data are Phase 4 (post-MVP).

**MVP configuration**: [Task Analysis] -> [Scope, Precision] -> Execute (4 prompts total). Task Analysis (always-run preamble) parses intent and identifies targets. Scope expands from seeds and filters by relevance. Precision extracts symbols and classifies detail levels. This is the initial configuration, not the architecture.

The pipeline architecture (stages, context curation, budget management) is model-agnostic -- nothing in the retrieval or orchestration layers depends on a specific model or provider. The LLM transport layer (`llm/client.py`) is Ollama-specific for MVP (Phases 1-3). Above it, a `ModelRouter` resolves which model to call based on (role, stage_name), reading from config. Phase 4 shifts to vLLM (or llama-server) for per-request LoRA adapter routing, and adds remote API backends for teacher model distillation. The external `complete()` interface does not change.

## Repository Contents

```
planning/
  meta-plan.md                   - Single source of truth: phases, architecture, contracts (index to topic files)
  schemas.md                     - Full DDL for all three databases + session key contracts
  cli-and-config.md              - CLI commands, arguments, config file format, resolution order
  pipeline-and-modes.md          - Retrieval stages, LLM client, budget, pipeline modes, orchestrator
  training-strategy.md           - LoRA training targets, bootstrapping, distillation, guardrails
  phase1-implementation.md       - Phase 1 work items (completed)
research/
  (literature reviews, technical reports, feasibility studies)
references/
  (curated catalogs: benchmarks, bibliographies, repo corpuses)
infrastructure/
  (hardware specs, environment setup, air-gap design)
protocols/
  design_records/                - Architectural decision records
  audits/                        - Transparency and design audits
  enforcement/                   - Enforcement protocols
  governance/                    - Design philosophy and theoretical frameworks
  retrieval_audit/               - Retrieval audit protocol and reference tasks
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

All of these serve the core principle: every decision must be explicit, logged, and traceable.

- **The room, not the model** - performance comes from what's in the context window
- **Results and capabilities, never mechanisms** - show what the system does, never explain how (competitive advantage)
- **Deterministic first, AI second** - metadata extraction before semantic search
- **Each token earns its place** - default-deny context architecture, not additive stuffing (see Context Curation Rules below)
- **Log everything, curate deliberately** - raw DB captures all activity; promotion to curated is an explicit, reviewed act
- **The loop closes** - logged activity becomes training data; the agent uses itself to plan its own improvement

### Context Curation Rules

These rules operationalize the core transparency principle for context curation. Every component that touches content destined for an LLM context window — retrieval stages, context assembly, prompt construction, enrichment — must follow them.

**1. No degradation. Fix decisions, not content.** If the precision stage classified a file as "primary", the execute stage needs the full source — downgrading to signatures means the model can't edit code it can't see. If assembled content exceeds the budget, the upstream stages made wrong decisions. The correct response is to re-filter: add an LLM call to re-prioritize and drop entire files or symbols. Never silently degrade, truncate, or partially render content that was classified at a specific detail level.

**2. Default-deny, not default-include.** When a curation decision cannot be made — LLM omits a file from its response, returns an invalid classification, or a symbol has no classification — the default is to **exclude**. Including content without a positive curation signal violates the architecture. Log a warning when the default fires, but do not promote unclassified content into the context window.

**3. Every LLM prompt must be budget-validated.** Not just the final execute-stage context, but every intermediate prompt (scope judgment, precision classification, task analysis, enrichment) must be validated against the target model's input capacity before sending. If the prompt exceeds the model's context window, batch or pre-filter — never send an oversized prompt and let the provider silently truncate it, because silent input truncation can discard the system prompt and task description.

**4. Use parsed structure, not string heuristics.** The indexer already parses AST, extracts signatures, identifies docstring boundaries, and stores symbol line ranges. Rendering and extraction must use this parsed data. Do not re-derive structure from raw source with keyword matching (e.g., `line.startswith("def ")`) — it breaks on multi-line signatures, decorators, and language edge cases.

**5. Framing overhead is part of the budget.** Headers, code fences, section markers, and other structural text consume tokens. Budget tracking must account for framing, not just content. If the framing format can be broken by content (e.g., triple backticks inside code fences), use a format that is immune to content injection.

**6. Arbitrary caps must be ordered and justified.** When a numeric limit is necessary (e.g., max candidates per tier, max connections per symbol), the items must be ordered by a defined relevance criterion before the cap is applied. Slicing an unordered list (`results[:5]`) is random selection, not curation. If no ordering criterion exists, send everything and let an LLM call decide what to cut.

### Coding Style (Development Mode)

- We are developing, not maintaining a product: optimize for fast debugging, not graceful degradation.
- No fallbacks and no hardcoded defaults in core logic.
- Keep `try/except` blocks minimal and intentional.
- When catching errors, add context and re-raise; do not silently recover.
- Prefer fail-fast behavior so incorrect assumptions break hard and early.
- Every config field must be classified as Required (fail-fast when missing), Optional (documented default in a named constant or derivation), or Supplementary (non-core, safe fallback). Required fields are uncommented in the config template. Optional fields use named constants or derivation functions, not magic numbers in `.get()`. No field may exist in an unclassified state. See `planning/cli-and-config.md` Section 4.4.

## Validation Note

Formal benchmarking and thesis validation are intentionally outside the active Phase 1-4 plan and can be redesigned later without constraining implementation.

## Status

Phase 2 complete (retrieval pipeline with scope, precision, assembly, budget management). Next steps:
1. ~~Build the knowledge base and indexer (Phase 1)~~ -- DONE
2. ~~Build the retrieval pipeline (Phase 2)~~ -- DONE
3. Build the code agent with plan + implement modes (Phase 3) -- MVP boundary
4. Build the self-improvement loop (Phase 4) -- post-MVP

Phase 4's bootstrapping path (synthetic training data from external repos plus open cold-start datasets like OpenCodeInstruct and CommitPackFT) depends only on Phase 1 -- LoRA training can begin before Phases 2-3 are built. The self-improvement path within Phase 4 requires Phase 3 logged data.
