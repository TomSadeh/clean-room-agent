# Three-Prompt Retrieval Strategy

**Date**: 2026-02-22
**Status**: Design concept - not implemented. Superseded by the repository's N-prompt architecture (see `CLAUDE.md` and `planning/`).

---

## Overview

A three-stage pipeline that progressively narrows what the coding agent sees, designed to be stress-tested with a deliberately small model to prove that context curation — not model scale — drives agent quality.

---

## Stage 1: Scope & Relevance Filter

**Input**: Full repository + task description
**Output**: A manageable scope for a small context window

This is the coarse pass. Given a coding task (e.g., "fix the auth bug in the login flow"), this stage determines which files, modules, and directories are even in the ballpark.

Signals to use:
- File paths and directory structure
- Import/dependency graphs
- File-level metadata (size, recency, type)
- Keyword/semantic match against file names and top-level declarations
- Git blame/history for recently changed files

**Goal**: Reduce a 10,000-file repo to the 50-100 files that could plausibly matter.

**Model requirement**: Can be a very small model or even partially deterministic (AST parsing + metadata heuristics).

---

## Stage 2: Precision Filter

**Input**: Scoped material from Stage 1 + task description
**Output**: Exactly the context needed for the coding task

This is the fine pass. From the scoped files, extract:
- Relevant function signatures and type definitions
- Dependency relationships between the scoped files
- Test expectations and assertions
- Documentation excerpts
- Error messages and log patterns

**Goal**: Produce a focused context package — the minimum information a competent developer would need to understand and complete the task, with zero noise.

**Model requirement**: Small model with good extraction/summarization capability.

---

## Stage 3: Full Prompt

**Input**: Curated context from Stage 2 + task description
**Output**: The coding agent's response

The actual coding prompt. The model receives only what it needs:
- The task description
- Curated, relevant code context
- Type information and constraints
- Test expectations

The model's job is now purely reasoning and code generation. It doesn't need to explore, discover, or filter — all of that happened in Stages 1 and 2.

**Model requirement**: This is where model quality matters most, but the hypothesis is that even a small model will perform well here because the context is clean.

---

## Stress Test Design

Use a deliberately small local model profile (default: Qwen 3B class) for all three stages. The point is to demonstrate that the pipeline compensates for limited model capability through better context management.

### Comparison Matrix

| Configuration | Model | Context Strategy |
|---------------|-------|-----------------|
| A (baseline) | Small local (3B class) | Naive � full context, no curation |
| B (experiment) | Small local (3B class) | Three-prompt pipeline |
| C (reference) | Large (70B+) | Naive — full context, no curation |
| D (combined) | Large (70B+) | Three-prompt pipeline |

**Hypothesis**: B > A (curation helps small models dramatically), B ≈ C (curated small model matches naive large model), D > C (curation helps large models too, but less dramatically).

### Metrics
- First-attempt correctness (does the code work on first try?)
- Iteration count to working code
- Total tokens consumed
- Wall-clock time to completion
- Tool call count (fewer = more efficient)

---

## Relationship to Auto-GM

This strategy mirrors the knowledge curation approach in Auto-GM:
- Auto-GM: deterministic metadata extraction → relevance filtering → grounded LLM narration
- This experiment: repo-level scoping → precision context extraction → grounded code generation

The principle is the same: the model's job is reasoning, not filtering. Separate the responsibilities.


