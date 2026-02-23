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
4. **Per-Stage LoRA Adapters** (long-term) - One per pipeline stage, fine-tuned for that stage's specific job. Same base model, tiny adapter swap between stages.

**Result**: A 32K context window at nearly 100% signal relevance, versus a 200K window at 10-15% effective utilization.

## N-Prompt Pipeline Design

The pipeline is a variable-length sequence of retrieval stages followed by a terminal execute stage. The number of retrieval stages adapts to the task and repository:

- **Simple task, small repo** — [Task Analysis → Scope → Execute] (3 prompts)
- **Typical bug fix** — [Task Analysis → Scope → Precision → Execute] (4 prompts, MVP configuration)
- **Complex feature, large codebase** — [Task Analysis → Scope → Dependency Analysis → Impact Assessment → Precision → Execute] (6+ prompts)

Each retrieval stage is a genuine LLM prompt: deterministic pre-filtering narrows candidates, then the LLM evaluates relevance. Each LLM call gets a clean context window with only what that stage needs. The execute stage is always terminal: curated context + task → code generation.

The pipeline architecture is model-agnostic — stages, context curation, and budget management have no provider dependency. The LLM transport layer (`llm/client.py`) is Ollama-specific for MVP; swapping providers means reimplementing that module's internals, not changing the pipeline.

## Development-Mode Runtime Contract

- Required runtime inputs are explicit (fail-fast on missing required values).
- No fallback loading of required values from `.clean_room/config.toml` during active development.
- `cra retrieve` requires `--model` and `--base-url` (every retrieval stage uses LLM calls).
- `cra retrieve` and `cra solve` both require explicit stage selection (for example: `--stages scope,precision`).
- Budget input is explicit: either `--context-window` + `--reserved-tokens`, or `--budget-config <path>`.
- `cra enrich` is optional. Writes to raw DB; `--promote` copies to curated. Retrieval works without enrichment (Tier 4 skipped, stages use own LLM judgment).

## Prior Art

Validated in [Auto-GM](https://github.com/TomSadeh/Auto-GM)'s knowledge system, where curated retrieval can outperform much larger models using naive full-context approaches.

## Repository Contents

- `planning/` - Active plans and phase documents (source of truth for current work)
- `research_reviews/` - Research reviews and analysis documents
- `archive/` - Archived notes and superseded research/context documents




