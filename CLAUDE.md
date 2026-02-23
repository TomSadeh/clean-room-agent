# Clean Room Agent

## What This Project Is

A custom coding agent harness built around the thesis: **the primary bottleneck in LLM application performance is not model capability but context curation.** The model's job is reasoning, not filtering. Clean the room before the model enters.

This is a standalone Python coding agent. No external platform dependency — the harness and retrieval pipeline are all ours; benchmark validation is a separate phase.

## Core Architecture: The N-Prompt Pipeline

Instead of stuffing a 200K context window and hoping the model finds what matters, use a multi-stage pipeline where each prompt starts clean with curated context:

1. **Three-Database Architecture** — raw (append-only log of all activity), curated (verified signals the model reads from), and session (ephemeral per-task working memory). Cold-startable from git history.
2. **Deterministic Retrieval** — metadata extraction first, AI-assisted only for ambiguous items. Not embedding similarity.
3. **N-Stage Prompt Pipeline** — early stages filter and ground, later stages reason and execute. No conversation accumulation, no compaction.
4. **Per-Stage LoRA Adapters** (long-term) — one per pipeline stage, fine-tuned for that stage's job.

Target: a 32K window at ~100% signal relevance, beating a 200K window at 10-15% utilization.

## Data Architecture: Three-Database Model

Three separate SQLite files, not three schemas in one file. Independent WAL journals, backups, and lifecycles.

1. **Raw DB** (`raw.sqlite`) — append-only log of everything: indexing runs, retrieval decisions, solve attempts, LLM outputs, validation results. Training corpus and source of truth for analysis. Writers: all phases. Readers: Phase 4 analysis, future fine-tuning pipelines.

2. **Curated DB** (`curated.sqlite`) — derived from raw, contains only verified/promoted signals. This is the "clean room" the model reads from. Phase 1 indexing populates it directly (AST, deps, git metadata). Phase 2 and Phase 3 read from it but never write to it. Cold-startable from `cra index` alone.

3. **Session DB** (`session_<task_id>.sqlite`) — ephemeral per-task working memory. Created per solve run, discarded after (optionally archived to raw). Intentionally minimal: key-value retrieval state, staged working context, scratch notes. Phase 2 creates it, Phase 3 inherits and closes it.

**Connection factory**: `get_connection(role, task_id=None)` where role is `"curated"`, `"raw"`, or `"session"`. Single point of DB management.

**Cold start**: `cra index` populates curated DB. Raw DB gets first real data from indexing run metadata. Session DB gets first real data in Phase 2/3.

**Raw→curated derivation**: Starts manual/scripted. Phase 4 analysis informs what to automate. No premature automation.

## MVP: Three-Prompt Strategy

The minimum viable experiment:

- **Stage 1 (Scope)**: Full repo + task → 50-100 relevant files. Can be deterministic (AST + metadata heuristics).
- **Stage 2 (Precision)**: Scoped files + task → exact context needed. Extracts signatures, types, tests, docs.
- **Stage 3 (Execute)**: Curated context + task → code generation. Pure reasoning, zero exploration.

Stress-tested with a deliberately small local model (default benchmark profile: Qwen 3B class) to prove context curation > model scale.

## Repository Contents

```
planning/
  meta-plan.md                   - Top-level phase boundaries and gates
  phase1-knowledge-base.md       - Knowledge Base + Indexer (8 steps)
  phase2-retrieval-pipeline.md   - Retrieval pipeline build (7 steps)
  phase3-agent-harness.md        - Agent harness build (8 steps)
  phase4-validation-benchmark.md - Validation + benchmark plan (4 steps)
archive/
  (archived notes and superseded research/context documents)
```

### Runtime Data Layout

```
.clean_room/
  curated.sqlite                 — indexed knowledge base (Phase 1 writes, Phase 2/3 read)
  raw.sqlite                     — append-only activity log (all phases write)
  sessions/
    session_<task_id>.sqlite     — ephemeral per-task working memory (created/destroyed per run)
```

## Prior Art

Validated in [Auto-GM](https://github.com/TomSadeh/Auto-GM)'s knowledge system: a 4B local model with curated retrieval outperforms much larger models with full context. Also supported by research: GPT-4 SWE-bench scores vary 2.7%-28.3% from scaffolding alone (>10x from same model).

## Stress Test Matrix

| Config | Model | Context Strategy | Expected |
|--------|-------|-----------------|----------|
| A (baseline) | Small local (3B class) | Naive full context | Low |
| B (experiment) | Small local (3B class) | Three-prompt pipeline | B > A |
| C (reference) | 70B+ | Naive full context | B ~ C |
| D (combined) | 70B+ | Three-prompt pipeline | D > C |

## Development Principles

- **The room, not the model** — performance comes from what's in the context window
- **Results and capabilities, never mechanisms** — show what the system does, never explain how (competitive advantage)
- **Deterministic first, AI second** — metadata extraction before semantic search
- **Each token earns its place** — default-deny context architecture, not additive stuffing
- **Log everything, curate deliberately** — raw DB captures all activity; promotion to curated is an explicit, reviewed act

## Benchmarking Strategy

### Priority Benchmarks (must-have)

1. **SWE-bench Verified** (500 instances, human-validated) — the lingua franca. Everyone reports scores here. Start with this.
2. **SWE-ContextBench** (Feb 2026) — directly measures context retrieval quality, isolated from patch generation. This is the benchmark that proves our thesis.
3. **Aider's polyglot benchmark** — tests the edit-apply loop across multiple languages. Harness-engineering diagnostic, not just model capability.

### Secondary Benchmarks (track, run when ready)

4. **SWE-bench Pro** (Scale AI) — harder, enterprise-grade, longer-horizon tasks. Current agents top out ~30-40%. This is where differentiation happens in 2026 as Verified approaches saturation (mini-SWE-agent already hits 65%+ with 100 lines of bash).
5. **ContextBench** (arXiv:2602.05892) — retrieval precision/recall for coding agents. Newer, less established, but directly measures what we care about.
6. **SWE-EVO** (arXiv:2512.18470) — evolving codebases, sequences of changes over time. Stress-tests memory and cross-session systems.
7. **LiveCodeBench** — continuously updated competitive programming problems, post-cutoff. No contamination risk.

### Sanity Checks & Diagnostics

8. **HumanEval / HumanEval+ / MBPP** — too easy for agent evaluation (frontier 95%+), but useful to verify the harness doesn't break basic generation, especially with small models.
9. **DevBench / RepoBench** — full dev lifecycle and cross-file completion. More realistic than function-level benchmarks.

### Worth Watching

- **Live-SWE-agent leaderboard** — tracks self-evolving agents on real GitHub issues. Relevant if we build toward self-improving tooling.

### Benchmark Selection Rationale

SWE-bench Verified is comparability. SWE-ContextBench is thesis validation. Aider's benchmark is harness diagnostics. The rest layer on as the system matures and we need to show where the ceiling is (Pro, EVO) or prove robustness (LiveCodeBench, DevBench).

## Baseline Model

The stress test model is a **small local Qwen profile** with fixed configuration across baseline and pipeline runs (default: `qwen2.5-coder:3b`). Using one fixed small model for both arms isolates the impact of context curation. If the three-prompt pipeline makes the small-model profile competitive, that is the thesis signal.

## Status

Research and design phase. Next steps:
1. Build the knowledge base and indexer (Phase 1)
2. Build the retrieval pipeline (Phase 2)
3. Build the standalone agent harness (Phase 3)
4. Validate with Phase 4 benchmark plan (A/B comparison: pipeline on/off, same model, same tasks)
5. Measure against SWE-bench Verified first, then SWE-ContextBench for thesis validation
6. Graduate to SWE-bench Pro and SWE-EVO once Verified baseline is established






