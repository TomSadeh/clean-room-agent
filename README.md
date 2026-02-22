# OpenClaw Context Curation Experiment

**Status**: Reconnaissance complete, implementation deferred (post-alpha)
**Thesis**: Strategic pre-filtering of what the model sees produces better outputs than larger models with naive context.
**Approach**: Three-prompt retrieval strategy, stress-tested with a deliberately small model.

## Core Idea

LLM agent performance is bottlenecked by context management, not model capability. Researchers are trying to make models better at finding relevant items in a noisy room. Our approach is to clean the room before the model enters. The model's job is reasoning, not filtering. Separate the responsibilities.

## Three-Prompt Strategy

The experiment uses a three-stage retrieval pipeline to progressively narrow what the coding agent sees:

1. **Scope & Relevance Filter** — Takes the full repository and the task description, filters down to a scope that a small context window can handle. This is the coarse pass: which files, modules, and directories are even in the ballpark?

2. **Precision Filter** — Takes the scoped material from stage 1 and extracts exactly what is relevant to the coding task. Function signatures, type definitions, dependencies, test expectations, relevant documentation. No noise.

3. **Full Prompt** — The actual coding prompt, grounded with the curated context from stage 2. The model receives only what it needs to reason about the task.

The pipeline is designed to be stress-tested with a very small model (7B class or smaller) to prove that context curation, not model scale, is the primary driver of agent quality.

## Prior Art

This approach is validated in [Auto-GM](https://github.com/TomSadeh/Auto-GM)'s knowledge system, where a 4B local model with curated retrieval outperforms much larger models with full context. The experiment extends this thesis from game narration to coding agents.

## Target Platform: OpenClaw

[OpenClaw](https://github.com/openclaw/openclaw) is the test harness. See [findings/architecture-sweep.md](findings/architecture-sweep.md) for the full reconnaissance report.

## Strategic Notes

The thesis that context curation beats model scale is a competitive advantage. If the experiment validates in the coding domain, results will not be published immediately. This is a moat, not a paper.
