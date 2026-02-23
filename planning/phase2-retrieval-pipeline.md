# Phase 2: Retrieval Pipeline Build

## Context

This is the retrieval build phase of the Clean Room Agent. Given a task description and the curated DB from Phase 1, it produces a curated context package with minimal noise for code generation.

Phase 1 indexes (writes curated DB). Phase 2 retrieves (reads curated, logs to raw, uses session for working state). Phase 3 executes.

The gate for Phase 2: `cra retrieve` works end-to-end, reliably produces budget-compliant task-relevant context packages, and logs all retrieval decisions to raw DB.

---

## Scope Boundary

- `In scope`: Task analysis, file scoring/ranking, scope expansion, precision extraction, budget enforcement, context assembly, retrieval CLI.
- `Out of scope`: Benchmark-grade evaluation/reporting workflows (moved to Phase 4).

---

## Database Interaction Model

| DB | Access | Purpose |
|----|--------|---------|
| Curated | **Read-only** | All scoring, scoping, and precision queries go through the Phase 1 Query API |
| Raw | **Append-only** | Log every retrieval decision: which files scored what, which were included/excluded, and why |
| Session | **Read/write** | Per-task working state: retrieval stage progress, intermediate scores, staged context fragments |

Phase 2 **creates** the session DB for each task run. Phase 3 inherits it.

---

## Project Structure (additions to Phase 1)

```
src/
  clean_room/
    retrieval/
      __init__.py
      pipeline.py
      task_analysis.py
      scope.py
      precision.py
      budget.py
      context_assembly.py
      dataclasses.py
      raw_logger.py                # Logs retrieval decisions to raw DB
      session_state.py             # Manages per-task session DB state
      scoring/
        __init__.py
        signals.py
        combiner.py
tests/
  test_task_analysis.py
  test_scoring_signals.py
  test_scope.py
  test_precision.py
  test_budget.py
  test_context_assembly.py
  test_pipeline.py
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scoring | Weighted deterministic signals | Interpretable and tunable without model training. |
| Task parsing | Heuristics first, optional local LLM fallback | Deterministic default behavior with graceful fallback. |
| Budget control | Priority-based eviction and demotion | Preserve highest-value context under hard token limits. |
| Context output | Typed dataclasses + separate renderers | Easier testing and format portability. |

---

## Core Data Structures

- `TaskQuery`: parsed task intent (`task_type`, keywords, file/symbol hints, error patterns, concepts).
- `FileScore`: per-file relevance score with signal breakdown.
- `ScopeResult`: ranked scoped files plus expansion provenance.
- `SymbolContext`: selected symbol details and relevance tier.
- `FileContext`: selected context per file.
- `ContextPackage`: final retrieval output with token/budget metadata.

---

## Implementation Steps

### Step 1: Data Structures + Task Analysis

**Delivers**:
- Retrieval dataclasses.
- `analyze_task()` deterministic extraction for files/symbols/errors/keywords/task type.
- Optional `--llm-assist` for vague tasks.

---

### Step 2: Scoring Signals

**Delivers**:
- Signal functions for path match, symbol match, dependency proximity, co-change affinity, recency, metadata concept/domain overlap, and structural centrality.

**Data source**: All signal functions read from the **curated DB** via the Phase 1 Query API.

---

### Step 3: Signal Combination + Scope (Stage 1)

**Delivers**:
- Weighted score combination and ranking.
- Seed-file handling.
- Dependency and co-change expansion.

**Defaults**:
- Scope size target around 75 files (configurable).
- Task-type weight tweaks (`bug_fix`, `refactor`, `test`).

**Session state**: Writes scope results (ranked file list, scores) to **session DB** for downstream stages. Logs all scoring decisions (file scores, inclusion/exclusion reasons) to **raw DB**.

---

### Step 4: Precision Extraction (Stage 2)

**Delivers**:
- Symbol-tier classification: `primary`, `supporting`, `type_context`, `excluded`.
- Python symbol-edge traversal via `symbol_references`.
- TS/JS MVP fallback heuristics.
- Test assertion extraction and rationale-comment inclusion.

**Data source**: Reads scoped file list from **session DB**, queries symbol details from **curated DB**. Logs precision decisions to **raw DB**.

---

### Step 5: Token Budget Management

**Delivers**:
- Token counting and hard budget enforcement.
- Tiered eviction and primary-body demotion to signatures when needed.

**Constraint**:
- Output must never exceed requested budget.

---

### Step 6: Context Assembly

**Delivers**:
- Ordered, deduplicated `ContextPackage` output.
- Dependency edges between included files only.
- Renderers for markdown/prompt/json output paths.

---

### Step 7: Pipeline Orchestrator + CLI

**Delivers**:
- `cra retrieve` command.
- End-to-end retrieval execution with verbose diagnostics.

**Database lifecycle**:
1. Open curated DB connection (read-only) via connection factory.
2. Open raw DB connection (append) via connection factory.
3. Create session DB for this task via `get_connection('session', task_id=task_id)`.
4. Run pipeline stages (each reads curated, writes session state, logs decisions to raw).
5. Leave session DB open for Phase 3 handoff (or close if running retrieval standalone).

---

## Step Dependency Graph

```
Step 1 (Task Analysis + Data Structures)
  |
  +--> Step 2 (Scoring Signals)
         |
         +--> Step 3 (Scope)
                |
                +--> Step 4 (Precision)
                       |
                       +--> Step 5 (Budget)
                              |
                              +--> Step 6 (Context Assembly)
                                     |
                                     +--> Step 7 (Pipeline + CLI)

[All steps depend on Phase 1]
```

---

## Verification (Phase 2 Gate)

```bash
# Prerequisite: repo is indexed
cra index /path/to/repo -v

# Retrieval
cra retrieve "fix the login validation bug" --repo /path/to/repo -v

# JSON output
cra retrieve "fix the login validation bug" --repo /path/to/repo --format json > context.json
```

**Gate criteria**:
1. Retrieval runs end-to-end without crashes.
2. Seed files always appear in scope.
3. Dependency expansion includes direct neighbors of seed files.
4. Budget is always respected.
5. Output is coherent and actionable for Phase 3 solve prompts.
6. Raw DB contains retrieval decision records for every scored file.
7. Session DB contains retrieval state and working context for the task run.

---

## Handoff to Phase 4

Phase 4 owns formal retrieval quality evaluation and benchmark reporting.
