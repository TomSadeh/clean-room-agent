# [ARCHIVED] Phase 2: Retrieval Pipeline Build

## Context

This is the retrieval build phase of the Clean Room Agent. Given a task description and the curated DB from Phase 1, it produces a curated context package with minimal noise for code generation.

Phase 1 indexes (writes curated DB). Phase 2 retrieves (reads curated, logs to raw, uses session for working state). Phase 3 executes.

The gate for Phase 2: `cra retrieve` works end-to-end, reliably produces budget-compliant task-relevant context packages, and logs all retrieval decisions to raw DB.

---

## Scope Boundary

- `In scope`: Retrieval stage protocol, pipeline runner, task analysis, MVP stage implementations (Scope, Precision), budget enforcement, context assembly, retrieval CLI.
- `Out of scope`: Benchmark-grade evaluation/reporting workflows (outside the active plan).

---

## Database Interaction Model

| DB | Access | Purpose |
|----|--------|---------|
| Curated | **Read-only** | All scoping and precision queries go through the Phase 1 Query API |
| Raw | **Append-only** | Log every retrieval decision: which files were included/excluded per tier, and why |
| Session | **Read/write** | Per-task working state: retrieval stage progress, staged context fragments |

Phase 2 **creates** the session DB for each task run. Phase 3 inherits it.

---

## Project Structure (additions to Phase 1)

```
src/
  clean_room/
    retrieval/
      __init__.py
      stage.py                     # RetrievalStage protocol, StageContext dataclass
      pipeline.py                  # Pipeline runner: sequences stages, threads budget/session
      task_analysis.py
      budget.py
      context_assembly.py
      dataclasses.py               # BudgetConfig, TaskQuery, RefinementRequest, ContextPackage, etc.
      raw_logger.py                # Convenience wrapper around db/raw_queries.py for retrieval-specific logging. All raw DB inserts go through raw_queries.py.
      session_state.py             # Manages per-task session DB state
      stages/
        __init__.py
        scope.py                   # Scope stage: tiered expansion from seeds
        precision.py               # Precision stage: symbol-level extraction
tests/
  test_task_analysis.py
  test_scope_stage.py
  test_precision_stage.py
  test_budget.py
  test_context_assembly.py
  test_pipeline.py
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pipeline | Variable-length stage sequence with common protocol | Pipeline length adapts to task/repo complexity. Need more nuance? Add a stage, don't add knobs. |
| MVP stages | Scope (tiered expansion) + Precision (symbol extraction) | Minimal viable pipeline. Additional stages (dependency analysis, impact assessment, etc.) are future stage implementations, not structural changes. |
| Task parsing | Deterministic extraction + LLM enrichment (always) | Deterministic extraction runs first (file/symbol mentions, error patterns), LLM enriches with intent analysis. Every stage uses an LLM call. |
| Budget control | Priority-based eviction and demotion | Preserve highest-value context under hard token limits. |
| Context output | Typed dataclasses + separate renderers | Easier testing and format portability. |

---

## Core Data Structures

**Stage protocol** (defined in `retrieval/stage.py`):
- `StageContext`: the data threaded through the pipeline. Contains the current file set, symbol selections, token estimates, and provenance records. Each stage receives it from the previous stage (or empty for the first stage) and returns a refined version.
- `RetrievalStage`: protocol with `name: str` and `run(context: StageContext, kb: KnowledgeBase, task: TaskQuery, llm: LLMClient) -> StageContext`. Every retrieval stage implements this. Each stage receives the LLM client and is responsible for constructing its own focused prompt from the StageContext. Stage LLM call inputs/outputs are logged to raw DB. The pipeline runner calls `run()` on each stage in sequence.

**Pipeline data** (defined in `retrieval/dataclasses.py`):
- `BudgetConfig`: shared between Phase 2 and Phase 3. Contains model context window size, reserved tokens for system prompt/retry overhead, and the effective retrieval budget `(window - reserved)`. Imported by Phase 3.
- `TaskQuery`: parsed task intent (`task_type`, keywords, file/symbol hints, error patterns, concepts).
- `RefinementRequest`: structured request from Phase 3 when context is insufficient (`reason`, `missing_files[]`, `missing_symbols[]`, `missing_tests[]`, `error_signatures[]`).
- `SymbolContext`: selected symbol details and relevance tier.
- `FileContext`: selected context per file.
- `ContextPackage`: final retrieval output with token/budget metadata. Assembled from the last stage's `StageContext` by the context assembly step.

---

## Implementation Steps

### Step 1: Stage Protocol + Data Structures + Task Analysis

**Delivers**:
- `RetrievalStage` protocol and `StageContext` dataclass (in `retrieval/stage.py`).
- All retrieval dataclasses (`BudgetConfig`, `TaskQuery`, `RefinementRequest`, `ContextPackage`, etc.).
- `analyze_task()` deterministic extraction runs first (file/symbol mentions, error patterns, keyword extraction, task type classification), then LLM enriches with intent analysis ("What does this task need? Which files/symbols/domains?").
- Task analysis ALWAYS uses LLM. `--model` and `--base-url` are required for `cra retrieve`.

**Stage protocol contract**:
- `RetrievalStage.run(context: StageContext, kb: KnowledgeBase, task: TaskQuery, llm: LLMClient) -> StageContext`
- Each stage receives the accumulated context from prior stages, the LLM client, and returns a refined version.
- Each stage performs deterministic pre-filtering followed by an LLM judgment call. The LLM evaluates candidates against the task, making the judgment call that code alone can't make.
- Stages are stateless — all per-task state lives in `StageContext` and session DB.
- The pipeline runner is responsible for sequencing, budget threading, session writes, and raw DB logging (including stage LLM call inputs, outputs, latency, and token counts) between stages.

---

### Step 2: Scope Stage

**Delivers**:
- First `RetrievalStage` implementation: `ScopeStage` (in `retrieval/stages/scope.py`).
- Seed file identification from `TaskQuery` (explicit file/symbol hints, path matches, symbol-name matches).
- Tiered expansion outward from seeds, filling budget in priority order.
- Provenance tracking: every included file records which tier and which relationship caused its inclusion.

**Expansion tiers** (processed in order, each tier adds files until budget is exhausted):
1. **Seeds** — files directly mentioned in the task, or containing symbols mentioned in the task. Always included.
2. **Direct dependencies** — imports/imported-by for seed files via the dependency graph.
3. **Co-change neighbors** — files that historically change with seed files (above a minimum co-change count).
4. **Metadata matches** — files sharing domain/module/concepts with the task query, discovered via `file_metadata`. Skipped when `file_metadata` is absent (enrichment not promoted); stages rely on LLM evaluation from tiers 1-3.

Within each tier, files are ordered by relevance to seeds (e.g., dependency depth, co-change count, number of matching concepts). This is a tie-breaker within the tier, not a cross-tier score.

**LLM evaluation step**: After deterministic tiered expansion, the scope stage makes an LLM call: given the task and candidate files (with summaries), classify each as relevant/irrelevant. The scope prompt: task + candidate files with summaries → relevance classification. LLM output is logged to raw DB.

**Data source**: All expansion queries read from the **curated DB** via the Phase 1 Query API.

---

### Step 3: Precision Stage

**Delivers**:
- Second `RetrievalStage` implementation: `PrecisionStage` (in `retrieval/stages/precision.py`).
- Deterministic symbol extraction: Python symbol-edge traversal via `symbol_references`, TS/JS MVP heuristics path (file-level dependencies + symbol-name signals), test assertion extraction, rationale-comment inclusion.
- LLM classification step: after deterministic symbol extraction, the precision stage makes an LLM call: given the task and extracted symbols, classify each into tiers (`primary`, `supporting`, `type_context`, `excluded`). The precision prompt: task + symbols → tier classification. LLM output is logged to raw DB.

**Data source**: Reads scoped file list from incoming `StageContext`, queries symbol details from **curated DB**.

---

### Step 4: Token Budget Management

**Delivers**:
- Token counting and hard budget enforcement.
- Tiered eviction and primary-body demotion to signatures when needed.

**Token counting**: Uses a character-based approximation (chars / 4) for budget checks. Exact counts come from Ollama response metadata post-call. Budget enforcement uses the approximation with a configurable safety margin. No tiktoken dependency.

**Budget source**: Phase 2 receives a `BudgetConfig` from its caller. In `cra solve`, Phase 3 constructs and passes it. In standalone `cra retrieve`, the CLI requires explicit budget inputs and constructs it (no implicit defaults). This config contains the target model's context window size and reserved tokens for system prompt/retry overhead. Phase 2 targets `(window - reserved)` tokens for the context package. The `BudgetConfig` is a shared data structure defined in a common module, consumed by both Phase 2 and Phase 3.

**Constraint**:
- Output must never exceed the budget allocated by `BudgetConfig`.

---

### Step 5: Context Assembly

**Delivers**:
- Ordered, deduplicated `ContextPackage` output.
- Dependency edges between included files only.
- Renderers for markdown/prompt/json output paths.

---

### Step 6: Pipeline Runner + CLI

**Delivers**:
- `RetrievalPipeline` class: takes a list of `RetrievalStage` implementations and runs them in sequence, threading `StageContext` through. Handles budget threading, session state persistence between stages, and raw DB logging of per-stage decisions.
- `cra retrieve` standalone command (for inspection/debugging; outputs context package as JSON/markdown to stdout).
- End-to-end retrieval execution with verbose diagnostics.

**Pipeline configuration**: The caller provides the stage list explicitly. No implicit default stage list in active development mode. `cra solve` and `cra retrieve` must both receive a stage list (for example: `--stages scope,precision`). Future configurations add stage names to this list — the runner does not change.

**Standalone budget contract**: `cra retrieve` requires explicit budget input: either `--context-window` plus `--reserved-tokens`, or `--budget-config <path>`. Missing budget inputs are a hard error.

**CLI rules (`cra retrieve`)**:
- `--model <model-id>` and `--base-url <ollama-url>` are required (every retrieval stage uses LLM calls).
- `--stages <csv>` is required (example: `--stages scope,precision`). Empty stage lists are invalid.
- Provide either:
  - `--context-window <int>` and `--reserved-tokens <int>`, or
  - `--budget-config <path>`
- `--budget-config` is mutually exclusive with `--context-window`/`--reserved-tokens`.
- If neither pair nor `--budget-config` is provided, fail fast with a hard error.
- Validation is strict and fail-fast:
  - `context_window > 0`
  - `reserved_tokens >= 0`
  - `reserved_tokens < context_window`
- No model-name inference in standalone retrieval.
- Effective retrieval budget is always computed as: `retrieval_budget = context_window - reserved_tokens`.
- Persist the effective budget config for the run to raw DB alongside retrieval decisions for reproducibility.

**Handoff model**: In production, `cra solve` calls `RetrievalPipeline` internally. The `ContextPackage` stays in-memory (no serialization required). `cra retrieve` exists as a standalone CLI command for inspection/debugging, not the normal solve path.

**Refinement model**: `RetrievalPipeline` supports re-entry from Phase 3 using `RefinementRequest`. The same `task_id` and session handle are reused. Refinement adds/adjusts context and returns a new `ContextPackage` without giving Phase 3 direct curated access.

**Session state contract for refinement re-entry**:

Phase 2 writes these session DB keys during initial retrieval (via `set_retrieval_state`):
- `"stage_outputs"` — map of `stage_name -> serialized stage output`
- `"stage_progress"` — ordered list of completed stage names and their status
- `"final_context"` — serialized final `StageContext` after all stages complete

Phase 2 reads on re-entry:
- `"final_context"` — to know what was previously selected (base for refinement)
- `"refinement_request"` — written by Phase 3 before calling re-entry (see Phase 3 Step 6)
- `"stage_progress"` — to determine which stages ran and can be re-entered

Phase 2 re-entry behavior:
1. Load `"final_context"` and `"refinement_request"` from session DB.
2. Apply refinement constraints (missing files/symbols/tests from the request) to expand the existing context.
3. Re-run stages as needed with the refinement constraints, logging new decisions to raw DB.
4. Write updated `"final_context"` and `"stage_progress"` back to session DB.
5. Return a new `ContextPackage`.

**Enrichment preflight**: `RetrievalPipeline` checks whether `file_metadata` is populated before running stage expansion. This check runs in both production (`cra solve`) and standalone (`cra retrieve`) paths. Missing enrichment data is an **info log** ("enrichment not found, Tier 4 skipped"), not a hard error. When `file_metadata` is absent, Scope Tier 4 (metadata matches) is skipped and stages rely on their own LLM judgment from deterministic signals (AST, deps, git, file content samples).

**Database lifecycle**:
1. Open curated DB connection in read-only mode via connection factory (`get_connection('curated', read_only=True)`).
2. Open raw DB connection (append) via connection factory.
3. Create session DB for this task via `get_connection('session', task_id=task_id)`. Task ID is generated by the caller (`cra solve` generates it at startup; `cra retrieve` standalone generates its own).
4. Run each stage in sequence. Between stages, the runner persists `StageContext` to session DB and logs stage decisions to raw DB (including stage LLM call inputs, outputs, latency, and token counts). Each stage reads curated via the Query API.
5. On refinement re-entry, load prior stage context from session DB, apply refinement constraints, rerun stages deterministically, and log refinement decisions to raw DB.
6. Return `ContextPackage` plus an explicit session handle to the caller. In `cra solve`, Phase 3 takes ownership and closes it. In standalone `cra retrieve`, close it before process exit.

---

## Step Dependency Graph

```
Step 1 (Stage Protocol + Data Structures + Task Analysis)
  |
  +--> Step 2 (Scope Stage)         --\
  +--> Step 3 (Precision Stage)     ---+--> Step 6 (Pipeline Runner + CLI)
  +--> Step 4 (Budget)              --/
  +--> Step 5 (Context Assembly)    -/

[All steps depend on Phase 1]
```

Steps 2, 3, 4, 5 are independent of each other (all depend on Step 1) and can be built in parallel. Step 6 integrates them all.

---

## Verification (Phase 2 Gate)

```bash
# [ARCHIVED] Prerequisite: repo is indexed
cra index /path/to/repo -v

# [ARCHIVED] Optional: enrich with LLM metadata (improves Tier 4 signals)
cra enrich /path/to/repo --model <your-loaded-model> --promote

# Retrieval (--model and --base-url required for stage LLM calls)
cra retrieve "fix the login validation bug" --repo /path/to/repo --model <model-id> --base-url <ollama-url> --stages scope,precision --context-window 32768 --reserved-tokens 4096 -v

# [ARCHIVED] JSON output
cra retrieve "fix the login validation bug" --repo /path/to/repo --model <model-id> --base-url <ollama-url> --stages scope,precision --context-window 32768 --reserved-tokens 4096 --format json > context.json
```

**Gate criteria**:
1. Retrieval runs end-to-end without crashes.
2. Seed files always appear in scope.
3. Dependency expansion includes direct neighbors of seed files.
4. Budget is always respected.
5. Output is coherent and actionable for Phase 3 solve prompts.
6. Raw DB contains retrieval decision records for every file considered per expansion tier.
7. Session DB contains retrieval state and working context for the task run.
8. Refinement re-entry works for the same `task_id` and produces an updated, budget-compliant `ContextPackage`.

---

## Future Handoff

Formal retrieval quality evaluation and benchmark reporting are handled outside the active plan.

