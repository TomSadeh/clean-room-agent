# Clean Room Agent

**Status**: Research and design phase (post-alpha)
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

## MVP: Three-Prompt Strategy

The initial experiment uses a three-stage pipeline as the minimum viable version:

1. **Scope & Relevance Filter** - Full repository + task description -> manageable scope for a small context window.
2. **Precision Filter** - Scoped material -> exactly the context needed for the coding task.
3. **Full Prompt** - Curated context + task description -> code generation.

Stress-tested with a deliberately small local model (default benchmark profile: Qwen 3B class) to prove context curation, not model scale, is the primary driver of agent quality.

## Prior Art

Validated in [Auto-GM](https://github.com/TomSadeh/Auto-GM)'s knowledge system, where a 4B local model with curated retrieval outperforms much larger models with full context.

## Repository Contents

- `planning/` - Active plans and phase documents (source of truth for current work)
- `archive/` - Archived notes and superseded research/context documents


