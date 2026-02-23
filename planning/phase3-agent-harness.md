# Phase 3: Agent Harness Build

## Context

This is the execution build phase of the Clean Room Agent. It takes the curated context package from Phase 2 and uses an LLM to generate and apply code changes with a retry loop and local validation.

Phase 1 indexes (writes curated DB). Phase 2 retrieves (reads curated). Phase 3 executes (never touches curated DB). Phase 3 logs all attempts, results, and LLM outputs to raw DB and uses session DB for retry working memory.

The gate for Phase 3: `cra solve` works end-to-end in both pipeline and baseline context modes, produces valid patches, logs all activity to raw DB, and is operationally stable.

---

## Scope Boundary

- `In scope`: LLM client, prompt builder, response parser, patch application, validation, retry loop, solve orchestration.
- `In scope`: Baseline context construction as an alternate solve mode.
- `Out of scope`: SWE-bench loading, benchmark runner, pass-rate reporting, config matrix comparison.
- `Out of scope`: Thesis validation (moved to Phase 4).

---

## Database Interaction Model

| DB | Access | Purpose |
|----|--------|---------|
| Curated | **Never touches** | Phase 3 has no curated DB access — all context comes via the context package from Phase 2 |
| Raw | **Append-only** | Log all attempts (`AttemptRecord`→`run_attempts`), task results (`TaskResult`→`task_runs`), validation results (`ValidationResult`→`validation_results`), and raw LLM outputs |
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
      prompt_builder.py           # Context + task -> model-ready prompt
      llm_client.py               # Unified LLM interface
      response_parser.py          # LLM output -> structured edits
      patch.py                    # Apply edits to files
      validation.py               # Run tests, lint, type check
      baseline.py                 # Naive full-context construction
      retry.py                    # Retry logic with error feedback
      dataclasses.py              # Phase 3 data structures
tests/
  fixtures/
    sample_llm_responses.py       # Canned LLM outputs for parser tests
    sample_patches/               # Known-good patches for application tests
  test_prompt_builder.py
  test_llm_client.py
  test_response_parser.py
  test_patch.py
  test_validation.py
  test_harness.py
  test_baseline.py
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM client | `httpx` for Ollama, optional SDK adapters for API providers | Local path must work with zero API keys; adapters keep interface extensible. |
| Output format | Search-and-replace blocks (primary), unified diff (fallback) | S&R blocks are easier for small models and easier to validate. |
| Patch isolation | `git worktree` per attempt | Clean rollback on failure and isolated retries. |
| Retry strategy | Fresh prompt with structured error context, max 3 attempts | Bounded retries and deterministic behavior. |
| Baseline approach | Structured naive context mode | Provides a build-time alternate solve path without retrieval dependency. |

---

## Data Structures

```python
@dataclass
class LLMConfig:
    provider: str                     # "ollama", "anthropic", "openai"
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    base_url: str | None = None

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

@dataclass
class TaskResult:
    task_id: str
    mode: str                         # "pipeline" or "baseline"
    success: bool
    attempts: list[AttemptRecord]
    total_tokens: int
    total_latency_ms: int
    final_diff: str | None
```

**Database mapping**: These dataclasses map directly to raw DB tables — `AttemptRecord`→`run_attempts`, `TaskResult`→`task_runs`, `ValidationResult`→`validation_results`. All instances are persisted to raw DB as they are created.

---

## Implementation Steps

### Step 1: LLM Client

**Delivers**: Unified interface for sending prompts and receiving completions.

**Files**:
- `src/clean_room/agent/llm_client.py`
- `src/clean_room/agent/dataclasses.py`
- `tests/test_llm_client.py`

**Requirements**:
- Ollama path is required (`/api/chat` and `/api/generate` support).
- Clear token/latency extraction and error mapping.
- Retries for transient transport errors.

---

### Step 2: Prompt Builder

**Delivers**: Prompt construction for pipeline, baseline, and retry attempts.

**Files**:
- `src/clean_room/agent/prompt_builder.py`
- `tests/test_prompt_builder.py`

**Requirements**:
- Shared system prompt with strict edit output contract.
- Token-aware retry prompts with bounded error output.
- Retry wording must assume fresh worktree application per attempt.

---

### Step 3: Response Parser

**Delivers**: Parse LLM output into `EditBlock`s.

**Files**:
- `src/clean_room/agent/response_parser.py`
- `tests/fixtures/sample_llm_responses.py`
- `tests/test_response_parser.py`

**Requirements**:
- Primary parse format: search-and-replace blocks.
- Fallback formats: unified diff, then fenced replacements.
- Robust malformed-block handling with non-crashing behavior.

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

**Database writes**:
- Each attempt: write `AttemptRecord` (including raw LLM response) to **raw DB** immediately after generation.
- Each validation: write `ValidationResult` to **raw DB** immediately after validation completes.
- Retry context (error classifications, attempt summaries): write to **session DB** for prompt builder to consume on next attempt.

---

### Step 7: Baseline Context Mode

**Delivers**: Naive context builder for `solve --mode baseline`.

**Files**:
- `src/clean_room/agent/baseline.py`
- `tests/test_baseline.py`

**Requirements**:
- Budget-aware packing of raw files.
- Mentioned files and nearby files prioritized.
- Safe handling of binary/non-UTF-8 files.

---

### Step 8: Agent Harness + CLI

**Delivers**: `cra solve` end-to-end command.

**Files**:
- `src/clean_room/agent/harness.py`
- Update `src/clean_room/cli.py`
- `tests/test_harness.py`

**CLI interface**:
```bash
cra solve "fix the login validation bug" --repo /path/to/repo --model qwen2.5-coder:3b --mode pipeline
cra solve "fix the login validation bug" --repo /path/to/repo --model qwen2.5-coder:3b --mode baseline
cra solve "fix the login validation bug" --repo /path/to/repo --dry-run
```

**Database lifecycle**:
1. In pipeline mode: inherit session DB from Phase 2 retrieval run.
2. In baseline mode: create a new session DB for this task.
3. Open raw DB connection (append) via connection factory.
4. Run retry loop (each attempt writes to raw DB and session DB).
5. Write final `TaskResult` to raw DB.
6. Close session DB. Optionally archive session content to raw DB (`session_archives` table).

---

## Step Dependency Graph

```
Step 1 (LLM Client)
  |
  +--> Step 2 (Prompt Builder)
  |      |
  |      +--> Step 3 (Response Parser)
  |             |
  |             +--> Step 4 (Patch Application)
  |                    |
  |                    +--> Step 5 (Validation)
  |                           |
  |                           +--> Step 6 (Retry Loop)
  |
  +--> Step 7 (Baseline Context Mode)
          |
          +--> Step 8 (Agent Harness + CLI) <-- Step 6

[All steps depend on Phases 1 + 2]
```

---

## Verification (Phase 3 Gate)

```bash
# Prerequisite: repo is indexed for pipeline mode
cra index /path/to/repo -v

# Pipeline solve
cra solve "fix the broken test in test_auth.py" --repo /path/to/repo --model qwen2.5-coder:3b --mode pipeline -v

# Baseline solve
cra solve "fix the broken test in test_auth.py" --repo /path/to/repo --model qwen2.5-coder:3b --mode baseline -v
```

**Gate criteria**:
1. `cra solve` runs end-to-end in both modes without crashes.
2. At least one retry path succeeds on a real task and emits a valid unified diff.
3. Retry loop handles parse failures and patch failures gracefully.
4. Local validation output is structured and actionable.
5. Logs are sufficient to debug failed attempts.
6. Raw DB contains complete attempt records (including raw LLM responses) for every solve run.
7. Session DB lifecycle is correct: created/inherited at start, closed at end, optionally archived.

---

## Handoff to Phase 4

Phase 4 consumes Phase 3 outputs (`TaskResult`, unified diffs, logs) to run external benchmark validation and reporting.
