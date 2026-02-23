# Phase 3: Agent Harness Build

## Context

This is the execution build phase of the Clean Room Agent. It takes the curated context package from Phase 2 and uses an LLM to generate and apply code changes with a retry loop and local validation.

Phase 1 indexes (writes curated DB). Phase 2 retrieves (reads curated). Phase 3 executes without direct curated DB access. Phase 3 logs all attempts, results, and LLM outputs to raw DB and uses session DB for retry working memory.

The gate for Phase 3: `cra solve` works end-to-end, produces valid patches, logs all activity to raw DB, and is operationally stable.

---

## Scope Boundary

- `In scope`: LLM client, prompt builder, response parser, patch application, validation, retry loop, solve orchestration, and explicit context-refinement handoff to Phase 2.
- `Out of scope`: Baseline context mode, benchmark loading/running/reporting, thesis validation.

---

## Database Interaction Model

| DB | Access | Purpose |
|----|--------|---------|
| Curated | **No direct access** | Phase 3 does not open curated DB connections; preflight checks run inside the Phase 2 retrieval entrypoint and context arrives as a `ContextPackage` |
| Raw | **Append-only** | Log all attempts (`AttemptRecord`->`run_attempts`), task results (`TaskResult`->`task_runs`), validation results (`ValidationResult`->`validation_results`), and raw LLM outputs |
| Session | **Read/write** | Inherits session DB from Phase 2, writes retry context (error classifications, attempt history), closes session on task completion |

Phase 3 **inherits** the session DB created by Phase 2 and **closes** it after the task run completes. Optionally archives session content to raw DB.

---

## Project Structure (additions to Phases 1-2)

```
src/
  clean_room/
    agent/
      __init__.py
      harness.py                  # Top-level agent orchestrator
      prompt_builder.py           # Context + task -> model-ready prompt, prompt-format helpers
      response_parser.py          # LLM output -> structured edits
      patch.py                    # Apply edits to files
      validation.py               # Run tests, lint, type check
      retry.py                    # Retry logic with error feedback
      dataclasses.py              # Phase 3 data structures
tests/
  fixtures/
    sample_llm_responses.py       # Canned LLM outputs for parser tests
    sample_patches/               # Known-good patches for application tests
  test_prompt_builder.py
  test_response_parser.py
  test_patch.py
  test_validation.py
  test_harness.py
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM client | Shared Ollama transport from `llm/client.py` (Phase 1), imported directly by Phase 3 | Local path must work with zero API keys; shared transport avoids duplicating httpx/retry logic. `llm/client.py` is the provider boundary — Ollama-specific for MVP, swappable without changing callers. Prompt-format helpers live in `prompt_builder.py`. |
| Output format | Search-and-replace blocks (single required format) | Single strict format keeps parser behavior deterministic and fail-fast. |
| Patch isolation | `git worktree` per attempt | Clean rollback on failure and isolated retries. Worktree creation must succeed - if it fails (e.g., Windows long-path issues), error out with diagnostic context. Test worktree lifecycle early on Windows. |
| Retry strategy | Fresh prompt with structured error context; max attempts provided explicitly by caller config | Bounded retries without hardcoded core defaults. |

---

## Data Structures

```python
@dataclass
class OllamaConfig:
    model: str
    temperature: float
    max_tokens: int
    base_url: str                     # Required, no default — user specifies their Ollama URL

@dataclass
class EditBlock:
    file_path: str
    search: str
    replace: str
    diff_hunk: str | None = None

@dataclass
class GenerationResult:
    edits: list[EditBlock]
    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    attempt: int

@dataclass
class ValidationResult:
    success: bool
    test_output: str | None
    lint_output: str | None
    type_check_output: str | None
    failing_tests: list[str]

@dataclass
class AttemptRecord:
    attempt: int
    generation: GenerationResult
    patch_applied: bool
    validation: ValidationResult | None
    unified_diff: str

# RefinementRequest is defined in retrieval/dataclasses.py (Phase 2 owns the contract).
# Phase 3 imports it: from clean_room.retrieval.dataclasses import RefinementRequest

@dataclass
class TaskResult:
    task_id: str
    success: bool
    attempts: list[AttemptRecord]
    total_tokens: int
    total_latency_ms: int
    final_diff: str | None
```

**Database mapping**: These dataclasses map directly to raw DB tables - `AttemptRecord`->`run_attempts`, `TaskResult`->`task_runs`, `ValidationResult`->`validation_results`. All instances are persisted to raw DB as they are created.

---

## Implementation Steps

### Step 0: Platform Prerequisites

**Delivers**: Verified `git worktree add/remove` lifecycle on the target platform.

**Requirements**:
- Verify `git worktree add` and `git worktree remove` work correctly, including cleanup. On Windows, test long paths and cleanup behavior. This is a blocking prerequisite for Steps 4-7 (patch isolation depends on worktrees).

---

### Step 1: Dataclasses + Prompt-Format Helpers

**Delivers**: Phase 3 data structures (`OllamaConfig`, `EditBlock`, `GenerationResult`, `ValidationResult`, `AttemptRecord`, `TaskResult`) and prompt-format helpers in `prompt_builder.py`. Phase 3 imports `llm/client.py` directly for LLM transport — no wrapper layer.

**Files**:
- `src/clean_room/agent/dataclasses.py`
- `src/clean_room/agent/prompt_builder.py` (prompt-format helpers)

**Requirements**:
- `OllamaConfig` with required `base_url` (no default — user specifies their Ollama URL). This is the MVP provider config; the fields (model, temperature, max_tokens, base_url) are specific to the Ollama transport in `llm/client.py`.
- Prompt-format helpers convert `ContextPackage` + task into model-ready prompt strings.
- Phase 3 uses `llm/client.py` (shared Ollama transport from Phase 1) directly for LLM calls.

---

### Step 2: Prompt Builder

**Delivers**: Prompt construction for pipeline and retry attempts.

**Files**:
- `src/clean_room/agent/prompt_builder.py`
- `tests/test_prompt_builder.py`

**Requirements**:
- Shared system prompt with strict edit output contract.
- Token-aware retry prompts with bounded error output.
- Hard prompt token gate before LLM call: verifies the assembled prompt fits within the `BudgetConfig` context window. Phase 2 does the heavy lifting (context package is already budget-compliant), but this gate catches overflow from system prompt + retry context additions. Note that `reserved_tokens` must cover: system prompt + task description + retry context + generation overhead. The example `--reserved-tokens 4096` may be tight for multi-attempt runs. Overflow is trimmed deterministically by dropping lowest-priority retry details.
- Retry wording must assume fresh worktree application per attempt.

---

### Step 3: Response Parser

**Delivers**: Parse LLM output into `EditBlock`s.

**Files**:
- `src/clean_room/agent/response_parser.py`
- `tests/fixtures/sample_llm_responses.py`
- `tests/test_response_parser.py`

**Requirements**:
- Parse format: search-and-replace blocks only.
- Malformed blocks raise with full context (raw block content included) so you see exactly what the model produced.

---

### Step 4: Patch Application

**Delivers**: Apply edits in an isolated worktree and emit unified diff.

**Files**:
- `src/clean_room/agent/patch.py`
- `tests/test_patch.py`

**Requirements**:
- Exact matching first; normalized-whitespace matching only when uniquely resolvable.
- No fuzzy patching by default.
- Full apply report with failed edit reasons.

---

### Step 5: Validation

**Delivers**: Local test/lint/type-check validation.

**Files**:
- `src/clean_room/agent/validation.py`
- `tests/test_validation.py`

**Requirements**:
- Detect test runner reliably.
- Parse failing tests into structured output.
- Optional lint/type-check flags.

---

### Step 6: Retry Loop

**Delivers**: Generate -> apply -> validate retry orchestration.

**Files**:
- `src/clean_room/agent/retry.py`
- Tested in `tests/test_harness.py`

**Requirements**:
- Fresh worktree each attempt.
- Structured error classification for retry prompts.
- Clear terminal states for no-edits, apply-failure, validation-failure.
- `insufficient_context` is a first-class terminal for an attempt cycle: emit `RefinementRequest`, call Phase 2 retrieval re-entry, then continue retries on the returned context package.
- Refinement is bounded by explicit caller config (`max_refinement_loops`) and requires concrete evidence (missing symbol/file/test/error signature).

**Pre-refinement session writes** (Phase 3 writes before calling Phase 2 re-entry):
- `"refinement_request"` — serialized `RefinementRequest` with `reason`, `missing_files[]`, `missing_symbols[]`, `missing_tests[]`, `error_signatures[]`. This is the key Phase 2 reads on re-entry.
- `"attempt_summary"` — summary of the attempt(s) that led to the `insufficient_context` classification (attempt number, error type, what was missing). For Phase 2's logging and decision-making.

Phase 2's re-entry contract: it reads `"refinement_request"` and `"final_context"` from session DB, expands context to cover the missing items, and returns an updated `ContextPackage`. See Phase 2 Step 6 for the full re-entry protocol.

**Database writes**:
- At solve startup, create a `task_runs` row first and keep its `task_run_id` for this run.
- Each attempt: write `AttemptRecord` (including raw LLM response) to **raw DB** immediately after generation, linked by `task_run_id`.
- Each validation: write `ValidationResult` to **raw DB** immediately after validation completes.
- Retry context (error classifications, attempt summaries): write to **session DB** for prompt builder to consume on next attempt.
- Refinement requests and outcomes: write to **session DB** and **raw DB** for traceability.

---

### Step 7: Agent Harness + CLI

**Delivers**: `cra solve` end-to-end command.

**Files**:
- `src/clean_room/agent/harness.py`
- Update `src/clean_room/cli.py`
- `tests/test_harness.py`

**CLI interface**:
```bash
cra solve "fix the login validation bug" --repo /path/to/repo --model <model-id> --base-url <ollama-url> --stages scope,precision --context-window <int> --reserved-tokens <int> --max-attempts <int> --max-refinement-loops <int>
cra solve "fix the login validation bug" --repo /path/to/repo --model <model-id> --base-url <ollama-url> --stages scope,precision --budget-config <path> --dry-run
```

**CLI rules (`cra solve`)**:
- Provide either:
  - `--context-window <int>` and `--reserved-tokens <int>`, or
  - `--budget-config <path>`
- `--budget-config` is mutually exclusive with `--context-window`/`--reserved-tokens`.
- Required runtime inputs are explicit in active development mode: `--model`, `--base-url`, and `--stages` are mandatory. Budget values are also mandatory via explicit pair or `--budget-config`.
- `--max-attempts <int>` — maximum number of generate/apply/validate attempts per solve run. Required (no hardcoded default).
- `--max-refinement-loops <int>` — maximum number of Phase 2 re-entry loops for context expansion. Required (no hardcoded default).
- If required values are missing, fail fast with a hard error.
- Validation is strict and fail-fast:
  - `context_window > 0`
  - `reserved_tokens >= 0`
  - `reserved_tokens < context_window`
- No inference from `--model`.
- Effective retrieval budget is computed as: `retrieval_budget = context_window - reserved_tokens`.

**Task ID and handoff**: `cra solve` generates a task ID (UUID4) at startup. This ID is passed to the retrieval pipeline (Phase 2) which creates the session DB, and then used throughout the solve loop. The user never needs to manage task IDs.

**Startup sequence**:
1. Generate task ID.
2. Build `BudgetConfig` and retrieval config from explicit CLI inputs (or explicit `--budget-config` file), validate constraints, and fail fast on invalid/missing values. No required-value fallback loading.
3. Open raw DB connection (append) and create a `task_runs` row immediately to obtain `task_run_id` before any attempts are generated. Persist the effective budget config for run reproducibility.
4. Call `RetrievalPipeline` in-process, passing task ID and `BudgetConfig`. Retrieval performs curated preflight checks (including `file_metadata` presence), creates session DB, and returns `ContextPackage` plus a session handle in-memory.
5. Run retry loop (each attempt writes to raw DB and session DB). When context is insufficient, emit `RefinementRequest` and call retrieval re-entry for updated context.
6. Finalize the existing `task_runs` row with success/totals/final diff.
7. Close session handle. Optionally archive session content to raw DB: read the closed session SQLite file as raw bytes (`open(path, 'rb').read()`) and insert into `session_archives` table as a BLOB. To restore an archived session for debugging, write the blob to a temp `.sqlite` file and open it.
8. Exit with task status and diagnostics.

---

## Step Dependency Graph

```
Step 0 (Platform Prerequisites — blocking)
  |
Step 1 (Dataclasses + Prompt-Format Helpers)
  |
  +--> Step 2 (Prompt Builder)    --|
  +--> Step 3 (Response Parser)   --|--> Step 6 (Retry Loop)
  +--> Step 4 (Patch Application) --|        |
  +--> Step 5 (Validation)        --|        +--> Step 7 (Agent Harness + CLI)

[All steps depend on Phases 1 + 2]
```

Step 0 is a blocking prerequisite (worktree lifecycle verification). Steps 2, 3, 4, 5 are independent of each other - they are all consumed by Step 6 (Retry Loop) and can be built in parallel.

---

## Verification (Phase 3 Gate)

```bash
# Prerequisite: repo is indexed
cra index /path/to/repo -v

# Optional: enrich with LLM metadata (improves retrieval signals)
cra enrich /path/to/repo --model <your-loaded-model> --promote

# Solve
cra solve "fix the broken test in test_auth.py" --repo /path/to/repo --model <model-id> --base-url <ollama-url> --stages scope,precision --context-window 32768 --reserved-tokens 4096 --max-attempts 3 --max-refinement-loops 2 -v
```

**Gate criteria**:
1. `cra solve` runs end-to-end without crashes.
2. At least one retry path succeeds on a real task and emits a valid unified diff.
3. Retry loop surfaces parse failures and patch failures with actionable context.
4. Local validation output is structured and actionable.
5. Logs are sufficient to debug failed attempts.
6. Raw DB contains complete attempt records (including raw LLM responses) for every solve run.
7. Session DB lifecycle is correct: created at retrieval start, inherited by solve, closed at end, optionally archived.
8. Final model-bound prompt never exceeds `BudgetConfig` context limits.
9. `cra solve` errors clearly if `cra index` hasn't been run. Missing enrichment logs info and skips Tier 4 (not an error).
10. Refinement handoff works end-to-end without Phase 3 opening curated DB connections.

---

## Future Handoff

External benchmark validation and reporting consume Phase 3 outputs (`TaskResult`, unified diffs, logs) outside the active plan.
