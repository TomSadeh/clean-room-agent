# Clean Room Agent

**Status**: Research and design phase
**Thesis**: The primary bottleneck in LLM application performance is not model capability but context curation.

## Core Idea

LLM agent performance is bottlenecked by context management, not model capability. Researchers are trying to make models better at finding relevant items in a noisy room. Our approach is to clean the room before the model enters. The model's job is reasoning, not filtering. Separate the responsibilities.

## The N-Prompt Architecture

A custom coding agent harness built around a multi-stage context curation pipeline:

1. **Three-Database Architecture** - Raw DB (append-only log of all activity), Curated DB (verified signals the model reads from), and Session DB (ephemeral per-task working memory). Three separate SQLite files with independent lifecycles. Cold-startable from git history.
2. **Deterministic Pre-filtering + LLM Judgment** - Not embedding similarity hoping to capture relevance. Deterministic methods (AST, deps, git, metadata queries) narrow candidates, then an LLM call per stage evaluates relevance against the task.
3. **N-Stage Prompt Pipeline** - Early stages filter and ground. Later stages reason and execute. Each prompt starts clean with curated context. No conversation accumulation, no compaction, no degradation.
4. **Model Architecture** - Qwen3-1.7B as the primary model for code generation and structured classification, with an optional Qwen3-0.6B for high-volume binary classification. Planning decomposition reduces per-call cognitive load, making the 1.7B sufficient for tasks previously assigned to larger models. Routing is config-only via three roles (`coding`, `reasoning`, `classifier`).
5. **Per-Stage LoRA Adapters** (Phase 4) - One per pipeline stage, fine-tuned from logged activity and synthetic data bootstrapped from external repo commit histories.

**Result**: A 32K context window at nearly 100% signal relevance, versus a 200K window at 10-15% effective utilization.

## The Self-Improving Loop

The agent is a self-improving system. The same pipeline architecture serves four modes -- planning, coding, training planning, data curation -- creating a closed loop:

1. **Plan + Implement** (`cra plan`, `cra solve`) -- work on real coding tasks, logging every LLM call, decision, and outcome to raw DB.
2. **Train-Plan** (`cra train-plan`) -- analyze logged runs to identify weak pipeline stages and produce training plans.
3. **Curate-Data** (`cra curate-data`) -- curate training datasets from logged activity, filtered by quality signals.
4. **LoRA Training + Deploy** -- train per-stage adapters from curated data, deploy as model overrides.

**Bootstrapping**: The loop doesn't require the agent's own output to start. Mature, well-tested repos cloned from GitHub provide commit histories with natural mistake-to-solution pairs. Synthetic training data is generated from these *before* the agent produces any real runs, breaking the chicken-and-egg dependency. This path depends only on Phase 1 (indexing), so LoRA training can begin before the coding agent (Phase 3) is built.

## N-Prompt Pipeline Design

The pipeline is a variable-length sequence of retrieval stages followed by a terminal execute stage. The number of retrieval stages adapts to the task and repository:

- **Simple task, small repo** -- [Task Analysis -> Scope -> Execute] (3 prompts)
- **Typical bug fix** -- [Task Analysis -> Scope -> Precision -> Execute] (4 prompts, MVP configuration)
- **Complex feature, large codebase** -- [Task Analysis -> Scope -> Dependency Analysis -> Impact Assessment -> Precision -> Execute] (6+ prompts)

Each retrieval stage is a genuine LLM prompt: deterministic pre-filtering narrows candidates, then the LLM evaluates relevance. Each LLM call gets a clean context window with only what that stage needs. The execute stage is always terminal. Its output depends on mode: Plan produces structured plan artifacts, Implement produces code edits (search/replace), Train-Plan produces training plans, Curate-Data produces training datasets.

The pipeline architecture is model-agnostic -- stages, context curation, and budget management have no provider dependency. The LLM transport layer (`llm/client.py`) is Ollama-specific for MVP; swapping providers means reimplementing that module's internals, not changing the pipeline.

## Development-Mode Runtime Contract

- Required runtime inputs are explicit. Missing required inputs are hard errors.
- Required inputs resolve as: CLI flag -> `.clean_room/config.toml` value -> hard error.
- No `--model` or `--base-url` CLI flags. All model configuration lives in `.clean_room/config.toml`.
- `--stages` is required for all pipeline commands: `cra retrieve`, `cra plan`, `cra solve`, `cra train-plan`, `cra curate-data`.
- Budget input is required for all pipeline commands: either `--context-window` + `--reserved-tokens`, or `--budget-config <path>`.
- `cra enrich` is optional. It writes to raw DB; `--promote` copies to curated DB. Retrieval works without enrichment (Tier 4 skipped, stages use deterministic signals + stage LLM judgment).

## Prior Art

Validated in [Auto-GM](https://github.com/TomSadeh/Auto-GM)'s knowledge system, where curated retrieval can outperform much larger models using naive full-context approaches.

## Repository Contents

- `planning/meta-plan.md` - Single source of truth: phases, schemas, contracts, conventions
- `research/` - Literature reviews, technical reports, feasibility studies
- `references/` - Curated catalogs: benchmarks, bibliographies, repo corpuses
- `infrastructure/` - Hardware specs, environment setup, air-gap design
- `protocols/` - Decision records, audits, enforcement, governance
- `archive/planning-v1/` - Superseded v1 planning documents
- `archive/` - Archived notes and superseded research/context documents
