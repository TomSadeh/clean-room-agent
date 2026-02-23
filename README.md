# Clean Room Agent

**Status**: Research and design phase
**Thesis**: The primary bottleneck in LLM application performance is not model capability but context curation.

## Core Idea

LLM agent performance is bottlenecked by context management, not model capability. Researchers are trying to make models better at finding relevant items in a noisy room. Our approach is to clean the room before the model enters. The model's job is reasoning, not filtering. Separate the responsibilities.

## The N-Prompt Architecture

A custom coding agent harness built around a multi-stage context curation pipeline:

1. **Three-Database Architecture** - Raw DB (append-only log of all activity), Curated DB (verified signals the model reads from), and Session DB (ephemeral per-task working memory). Three separate SQLite files with independent lifecycles. Cold-startable from git history.
2. **Deterministic Retrieval** - Not embedding similarity hoping to capture relevance. Deterministic metadata extraction first, AI-assisted only for ambiguous items, grounded with confirmed metadata.
3. **N-Stage Prompt Pipeline** - Early stages filter and ground. Later stages reason and execute. Each prompt starts clean with curated context. No conversation accumulation, no compaction, no degradation.
4. **Per-Stage LoRA Adapters** (long-term) - One per pipeline stage, fine-tuned for that stage's specific job. Same base model, tiny adapter swap between stages.

**Result**: A 32K context window at nearly 100% signal relevance, versus a 200K window at 10-15% effective utilization.

## N-Prompt Pipeline Design

The pipeline is a variable-length sequence of retrieval stages followed by a terminal execute stage. The number of retrieval stages adapts to the task and repository:

- **Simple task, small repo** — [Scope → Execute] (2 prompts)
- **Typical bug fix** — [Scope → Precision → Execute] (3 prompts, MVP configuration)
- **Complex feature, large codebase** — [Scope → Dependency Analysis → Impact Assessment → Precision → Execute] (5+ prompts)

Each retrieval stage implements a common protocol (context in → refined context out). The execute stage is always terminal: curated context + task → code generation.

The pipeline architecture is model-agnostic — stages, context curation, and budget management have no provider dependency. The LLM transport layer (`llm/client.py`) is Ollama-specific for MVP; swapping providers means reimplementing that module's internals, not changing the pipeline.

## Development-Mode Runtime Contract

- Required runtime inputs are explicit (fail-fast on missing required values).
- No fallback loading of required values from `.clean_room/config.toml` during active development.
- `cra retrieve` and `cra solve` both require explicit stage selection (for example: `--stages scope,precision`).
- Budget input is explicit: either `--context-window` + `--reserved-tokens`, or `--budget-config <path>`.
- `cra enrich` must be run before `cra retrieve` and `cra solve` (retrieval preflight enforces `file_metadata` presence).

## Prior Art

Validated in [Auto-GM](https://github.com/TomSadeh/Auto-GM)'s knowledge system, where curated retrieval can outperform much larger models using naive full-context approaches.

## Repository Contents

- `planning/` - Active plans and phase documents (source of truth for current work)
- `archive/` - Archived notes and superseded research/context documents




