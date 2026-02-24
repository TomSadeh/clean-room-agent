# Phase 3 Implementation Plan: Code Agent (Plan + Implement Modes)

## Context

Phase 3 builds the code agent — the MVP that produces working code from task descriptions. It composes Phase 2's retrieval pipeline with new execute functions (plan + implement), patch application, mandatory test validation, and a deterministic orchestrator for multi-step tasks.

**Core thesis** (from `pipeline-and-modes.md`): The primary bottleneck is planning quality, not coding capability. Qwen2.5-Coder-3B is excellent at small, well-scoped tasks — the planner's job (Qwen3-4B) is to decompose tasks into pieces the coder can't fail on.

**Architecture**: Each "pass" in the agent is a fresh N-prompt pipeline invocation (Phase 2) followed by an execute function call (Phase 3). The pipeline handles retrieval and context curation. The execute function consumes the `ContextPackage`, calls the LLM, and produces a structured result (plan or code edits). The orchestrator sequences passes deterministically.

**Status**: Phase 1 (knowledge base) and Phase 2 (retrieval pipeline) complete. 474 tests passing. Contracts defined in `planning/meta-plan.md` (Section 5), `planning/pipeline-and-modes.md` (Sections 3-5), `planning/schemas.md` (Sections 2, 4).

---

## Package Structure

```
src/clean_room_agent/
  execute/                           # NEW — terminal execute stage
    __init__.py
    dataclasses.py                   # MetaPlan, PartPlan, StepResult, PatchEdit, etc.
    prompts.py                       # Prompt construction for all five pass types
    parsers.py                       # Response parsing: plan JSON + search/replace XML
    plan.py                          # execute_plan(): plan/adjustment LLM calls
    implement.py                     # execute_implement(): code generation LLM calls
    patch.py                         # Search/replace validation + atomic file application
  orchestrator/                      # NEW — deterministic sequencer
    __init__.py
    runner.py                        # Orchestrator loop + single-pass entry point
    validator.py                     # Test runner: subprocess execute, capture, log
  db/
    schema.py                        # MODIFIED — add 4 Phase 3 tables to raw schema
    raw_queries.py                   # MODIFIED — add insert/update helpers for new tables
  cli.py                             # MODIFIED — add `cra plan` and `cra solve` commands
  config.py                          # MODIFIED — add helpers for [testing], [orchestrator] sections
tests/
  execute/
    __init__.py
    test_dataclasses.py
    test_prompts.py
    test_parsers.py
    test_plan.py
    test_implement.py
    test_patch.py
  orchestrator/
    __init__.py
    test_runner.py
    test_validator.py
```

**Existing code unchanged**: `retrieval/*`, `llm/*`, `query/*`, `indexer/*`, `parsers/*`, `extractors/*`, `db/connection.py`, `db/session_helpers.py`, `db/queries.py`, `token_estimation.py`.

## External Dependencies

No new runtime dependencies. Phase 3 uses only:
- `subprocess` (stdlib) — test runner execution
- `tempfile` (stdlib) — temporary plan files for Tier 0 seeding
- Existing deps: `click` (CLI), `httpx` (via LLMClient), `sqlite3`, `json`, `uuid`, `dataclasses`, `pathlib`, `logging`

---

## Work Items

### WI-0: Save Implementation Plan
Save this plan to `planning/phase3-implementation.md`.

---

### WI-1: Raw DB Schema Extensions + Query Helpers
**Deps**: None

Add four Phase 3 tables to raw DB schema (per `schemas.md` Section 2):

- `db/schema.py` — add to `create_raw_schema()`:
  - `run_attempts` — per-attempt results within a step implementation pass:
    `(id, task_run_id REFERENCES task_runs, attempt INTEGER, prompt_tokens, completion_tokens, latency_ms, raw_response TEXT, patch_applied INTEGER, timestamp TEXT)`
  - `validation_results` — test/lint/type-check output per attempt:
    `(id, attempt_id REFERENCES run_attempts, success INTEGER, test_output TEXT, lint_output TEXT, type_check_output TEXT, failing_tests TEXT)`
  - `orchestrator_runs` — one row per `cra solve` invocation (without `--plan`):
    `(id, task_id TEXT, repo_path TEXT, task_description TEXT, total_parts INTEGER, total_steps INTEGER, parts_completed INTEGER DEFAULT 0, steps_completed INTEGER DEFAULT 0, status TEXT, timestamp TEXT, completed_at TEXT)`
  - `orchestrator_passes` — links orchestrator runs to constituent pipeline passes:
    `(id, orchestrator_run_id REFERENCES orchestrator_runs, task_run_id REFERENCES task_runs, pass_type TEXT, part_id TEXT, step_id TEXT, sequence_order INTEGER, timestamp TEXT)`

- `db/raw_queries.py` — add helpers:
  - `insert_run_attempt(conn, task_run_id, attempt, prompt_tokens, completion_tokens, latency_ms, raw_response, patch_applied) -> int`
  - `insert_validation_result(conn, attempt_id, success, test_output=None, lint_output=None, type_check_output=None, failing_tests=None) -> int`
  - `insert_orchestrator_run(conn, task_id, repo_path, task_description) -> int`
  - `update_orchestrator_run(conn, run_id, *, total_parts=None, total_steps=None, parts_completed=None, steps_completed=None, status=None, completed_at=None) -> None`
  - `insert_orchestrator_pass(conn, orchestrator_run_id, task_run_id, pass_type, sequence_order, *, part_id=None, step_id=None) -> int`

- Tests: schema creation (all 4 tables exist), insert/update round-trips, FK constraints, nullable fields

---

### WI-2: Phase 3 Dataclasses
**Deps**: None

- `execute/dataclasses.py`:

  **Plan data structures** (orchestrator-internal):
  - `PlanStep` — id: str, description: str, target_files: list[str], target_symbols: list[str], depends_on: list[str]
  - `MetaPlanPart` — id: str, description: str, affected_files: list[str], depends_on: list[str]
  - `MetaPlan` — task_summary: str, parts: list[MetaPlanPart], rationale: str
  - `PartPlan` — part_id: str, task_summary: str, steps: list[PlanStep], rationale: str
  - `PlanAdjustment` — revised_steps: list[PlanStep], rationale: str, changes_made: list[str]

  **User-facing plan format** (per `pipeline-and-modes.md` Section 3.3):
  - `PlanArtifact` — task_summary: str, affected_files: list[dict] (path, role, changes), execution_order: list[str], rationale: str

  **Implementation data structures**:
  - `PatchEdit` — file_path: str, search: str, replacement: str
  - `StepResult` — success: bool, edits: list[PatchEdit], error_info: str | None, raw_response: str
  - `PatchResult` — success: bool, files_modified: list[str], error_info: str | None, original_contents: dict[str, str] (for rollback)

  **Validation data structures**:
  - `ValidationResult` — success: bool, test_output: str | None, lint_output: str | None, type_check_output: str | None, failing_tests: list[str]

  **Orchestrator data structures**:
  - `PassResult` — pass_type: str, task_run_id: int, success: bool, artifact: MetaPlan | PartPlan | StepResult | PlanAdjustment | None
  - `OrchestratorResult` — task_id: str, status: str (`complete`/`partial`/`failed`), parts_completed: int, steps_completed: int, cumulative_diff: str, pass_results: list[PassResult]

  All dataclasses with `to_dict()` / `from_dict()` class methods for JSON serialization to session DB.

- Tests: construction, validation, serialization round-trips, `from_dict` with missing/extra fields

---

### WI-3: Prompt Templates + Response Parsers
**Deps**: WI-2

- `execute/prompts.py` — prompt construction:
  - `build_plan_prompt(context: ContextPackage, task_description: str, *, pass_type: str, cumulative_diff: str | None = None, prior_results: list[StepResult] | None = None, test_results: list[ValidationResult] | None = None) -> tuple[str, str]`
    - Returns `(system_prompt, user_prompt)`
    - pass_type selects system prompt: `META_PLAN_SYSTEM`, `PART_PLAN_SYSTEM`, or `ADJUSTMENT_SYSTEM`
    - User prompt assembles: `ContextPackage.to_prompt_text()` + task description + optional `<prior_changes>` section (cumulative diff) + optional `<completed_steps>` + `<test_results>` sections
    - R3: validates total prompt size against model context window before returning
    - R5: framing overhead counted in budget

  - `build_implement_prompt(context: ContextPackage, step: PlanStep, *, plan: PartPlan | None = None, cumulative_diff: str | None = None, failure_context: ValidationResult | None = None) -> tuple[str, str]`
    - Returns `(system_prompt, user_prompt)`
    - System prompt: `IMPLEMENT_SYSTEM` — specifies `<edit>/<search>/<replacement>` output format
    - User prompt assembles: context + step description + optional `<plan_constraints>` + `<prior_changes>` + `<test_failures>` sections
    - R3: validates prompt size

  **System prompt contracts** (content specified at implementation time, structure specified here):
  - `META_PLAN_SYSTEM` — role: task decomposition planner. Output: JSON with `task_summary`, `parts[]` (id, description, affected_files, depends_on), `rationale`. Constraint: no code in plans, explicit dependency edges mandatory.
  - `PART_PLAN_SYSTEM` — role: detailed step planner. Output: JSON with `part_id`, `task_summary`, `steps[]` (id, description, target_files, target_symbols, depends_on), `rationale`. Constraint: steps must be small enough for reliable single-pass code generation.
  - `ADJUSTMENT_SYSTEM` — role: plan reviewer. Output: JSON with `revised_steps[]`, `rationale`, `changes_made[]`. Constraint: cannot undo completed steps, only revise remaining.
  - `IMPLEMENT_SYSTEM` — role: code editor. Output: `<edit file="..."><search>...</search><replacement>...</replacement></edit>` blocks. Constraint: search text must match exactly, minimal changes, all edits in one response.

- `execute/parsers.py` — response parsing:
  - `parse_plan_response(text: str, pass_type: str) -> MetaPlan | PartPlan | PlanAdjustment`:
    - Strips markdown fencing (reuses `retrieval/utils.py:parse_json_response` internally)
    - Validates required fields per pass_type
    - Constructs typed dataclass from parsed JSON
    - Raises `ValueError` on malformed response with raw text in message

  - `parse_implement_response(text: str) -> list[PatchEdit]`:
    - Parses `<edit file="..."><search>...</search><replacement>...</replacement></edit>` blocks
    - Validates: well-formed tags, file paths present, search strings non-empty
    - Returns ordered list of `PatchEdit` (order preserved from response)
    - Raises `ValueError` on malformed response

  - `validate_plan(plan: MetaPlan | PartPlan) -> list[str]`:
    - Checks: all IDs unique, dependency references valid, no circular dependencies, at least one part/step
    - Returns list of warnings (empty = valid)

- Tests: valid/invalid JSON parsing, valid/invalid XML parsing, edge cases (empty response, extra whitespace, markdown fencing variations, nested code blocks), plan validation (circular deps, missing refs)

---

### WI-4: Execute Functions (Plan + Implement)
**Deps**: WI-2, WI-3

- `execute/plan.py`:
  - `execute_plan(context: ContextPackage, task_description: str, llm: LoggedLLMClient, *, pass_type: str, cumulative_diff: str | None = None, prior_results: list[StepResult] | None = None, test_results: list[ValidationResult] | None = None) -> MetaPlan | PartPlan | PlanAdjustment`
  - Flow: `build_plan_prompt` -> R3 budget validation -> `llm.complete` -> `parse_plan_response` -> `validate_plan` -> return
  - Hard error on budget overflow (R3). Hard error on parse/validation failure — caller decides how to handle.

- `execute/implement.py`:
  - `execute_implement(context: ContextPackage, step: PlanStep, llm: LoggedLLMClient, *, plan: PartPlan | None = None, cumulative_diff: str | None = None, failure_context: ValidationResult | None = None) -> StepResult`
  - Flow: `build_implement_prompt` -> R3 budget validation -> `llm.complete` -> `parse_implement_response` -> wrap in `StepResult` -> return
  - On parse failure: returns `StepResult(success=False, edits=[], error_info=..., raw_response=...)`

- Tests: mock LLM responses, verify prompt arguments, verify parsing delegation, verify error handling paths

---

### WI-5: Patch Application
**Deps**: WI-2

- `execute/patch.py`:
  - `validate_edits(edits: list[PatchEdit], repo_path: Path) -> list[str]`:
    - For each edit: file exists, search string found exactly once in file content
    - Returns list of error strings (empty = all valid)
    - Does NOT modify any files

  - `apply_edits(edits: list[PatchEdit], repo_path: Path) -> PatchResult`:
    - Calls `validate_edits` first — returns failure immediately if any invalid
    - Saves original file contents to `PatchResult.original_contents` (for rollback)
    - Groups edits by file, applies in response order within each file
    - Each edit: `content = content.replace(search, replacement, 1)`
    - Writes modified files atomically (write to temp file, then rename)
    - On any failure: calls `rollback_edits`, returns failure `PatchResult`
    - Returns success `PatchResult` with `files_modified` list

  - `rollback_edits(patch_result: PatchResult, repo_path: Path) -> None`:
    - Restores all files from `original_contents`

  **Key decisions**:
  - No git worktree isolation for MVP. Direct application with pre-validation ensures atomicity (validate-then-apply). Worktree isolation deferred to post-MVP hardening.
  - Edits to same file applied sequentially in response order. Each subsequent edit sees the result of prior edits to that file.
  - `str.replace(search, replacement, 1)` — exactly one replacement per edit. If search appears multiple times in the file, `validate_edits` catches this as an error.

- Tests: successful apply + verify file contents, rollback on failure, non-existent file, search string not found, duplicate search string in file, multiple edits to same file in sequence, empty replacement (deletion)

---

### WI-6: Test Runner + Validator
**Deps**: WI-1, WI-2

- `orchestrator/validator.py`:
  - `run_validation(repo_path: Path, config: dict, raw_conn: Connection, attempt_id: int) -> ValidationResult`:
    - Reads `[testing]` section from config:
      - `test_command` (required, str): e.g. `"pytest tests/"`, `"npm test"`
      - `lint_command` (optional, str): e.g. `"ruff check src/"`
      - `type_check_command` (optional, str): e.g. `"mypy src/"`
      - `timeout` (optional, int, default 120): seconds per command
    - Runs each configured command via `subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=True, cwd=repo_path)`
    - Parses test output for failing test names (best-effort regex for pytest `FAILED` lines and common frameworks)
    - Logs to raw DB via `insert_validation_result(raw_conn, attempt_id, success, ...)`
    - Returns `ValidationResult`

  - `require_testing_config(config: dict) -> dict`:
    - Extracts `[testing]` section, hard error if missing or no `test_command` (fail-fast: `cra solve` cannot run without a test command)

  **Key decisions**:
  - `shell=True` required for compound commands (e.g. `"cd src && pytest"`)
  - Timeout kills subprocess and returns failure with timeout info in `test_output`
  - Test output captured in full (not truncated) for raw DB logging and retry context
  - lint/type_check failures are informational — only test_command determines success/failure

- `config.py` — add:
  - `require_testing_config(config: dict | None) -> dict` (extracts `[testing]`, hard error if missing)
  - Update `create_default_config` to include `[testing]` and `[orchestrator]` section templates

- Tests: mock subprocess for pass/fail/timeout, config validation, failing test name extraction

---

### WI-7: Orchestrator
**Deps**: WI-1, WI-4, WI-5, WI-6

- `orchestrator/runner.py`:

  **Main entry — full orchestrator:**
  - `run_orchestrator(task: str, repo_path: Path, config: dict) -> OrchestratorResult`:
    - Generates orchestrator `task_id` (UUID4)
    - Opens raw DB, inserts `orchestrator_run` record
    - Creates orchestrator session DB (`session_{task_id}.sqlite`) for cross-pass state
    - Resolves stages from `config["stages"]["default"]` (or hard-coded `"scope,precision"`)
    - Reads `[orchestrator]` config: `max_retries_per_step` (default 1), `max_adjustment_rounds` (default 1)

    **Pass sequencing:**
    ```
    1. META-PLAN PASS
       sub_task_id = "{task_id}:meta_plan"
       budget = BudgetConfig(reasoning_model.context_window, reserved_tokens)
       context = run_pipeline(task, repo_path, stages, budget, mode="plan",
                              task_id=sub_task_id, config)
       llm = LoggedLLMClient(reasoning_model_config)
       meta_plan = execute_plan(context, task, llm, pass_type="meta_plan")
       flush llm records to raw DB
       store meta_plan in orchestrator session
       insert orchestrator_pass record

    2. FOR EACH PART (topological order by depends_on):
       a. PART-PLAN PASS
          Write temporary plan JSON with part.affected_files → temp_plan_path
          sub_task_id = "{task_id}:part_plan:{part.id}"
          context = run_pipeline(part.description, repo_path, stages, budget,
                                 mode="plan", task_id=sub_task_id, config,
                                 plan_artifact_path=temp_plan_path)
          part_plan = execute_plan(context, part.description, llm,
                                   pass_type="part_plan", cumulative_diff=cumulative_diff)
          store in session, insert orchestrator_pass

       b. FOR EACH STEP (dependency order within part):
          i.  IMPLEMENT + APPLY
              Write temp plan JSON with step.target_files
              sub_task_id = "{task_id}:impl:{part.id}:{step.id}"
              impl_budget = BudgetConfig(coding_model.context_window, reserved_tokens)
              context = run_pipeline(step.description, repo_path, stages,
                                     impl_budget, mode="implement",
                                     task_id=sub_task_id, config,
                                     plan_artifact_path=temp_plan_path)
              llm_impl = LoggedLLMClient(coding_model_config)
              step_result = execute_implement(context, step, llm_impl,
                                              plan=part_plan,
                                              cumulative_diff=cumulative_diff)
              flush llm records to raw DB
              insert run_attempt record
              if step_result.success:
                  patch_result = apply_edits(step_result.edits, repo_path)

          ii. TEST (mandatory, after every step)
              validation = run_validation(repo_path, config, raw_conn, attempt_id)

          iii. RETRY (if test failed, up to max_retries_per_step)
              rollback_edits(patch_result, repo_path)
              sub_task_id = "{task_id}:impl:{part.id}:{step.id}:retry_{n}"
              context = run_pipeline(..., mode="implement", ...)
              step_result = execute_implement(context, step, llm_impl,
                                              plan=part_plan,
                                              cumulative_diff=cumulative_diff,
                                              failure_context=validation)
              re-apply, re-test, insert attempt + validation records

          iv. ADJUSTMENT (unconditional, after every step)
              sub_task_id = "{task_id}:adjust:{part.id}:after_{step.id}"
              context = run_pipeline(remaining_steps_desc, repo_path, stages,
                                     budget, mode="plan", task_id=sub_task_id,
                                     config, plan_artifact_path=temp_plan_path)
              adjustment = execute_plan(context, remaining_steps_desc, llm,
                                         pass_type="adjustment",
                                         prior_results=[step_result],
                                         test_results=[validation],
                                         cumulative_diff=cumulative_diff)
              part_plan.steps = adjustment.revised_steps (for remaining steps)

          v.  UPDATE STATE
              cumulative_diff += step's applied diff
              steps_completed += 1
              update orchestrator_run in raw DB

       c. parts_completed += 1

    3. FINALIZE
       Update orchestrator_run status (complete/partial/failed)
       Archive orchestrator session to raw DB
       Clean up temp files
       Return OrchestratorResult
    ```

  **Single-pass entry — with pre-computed plan:**
  - `run_single_pass(task: str, repo_path: Path, config: dict, *, plan_path: Path) -> OrchestratorResult`:
    - Loads `PlanArtifact` from plan_path
    - Generates task_id
    - Runs one pipeline call (mode="implement", plan_artifact_path=plan_path) -> ContextPackage
    - Constructs a synthetic `PlanStep` from the plan artifact
    - Calls `execute_implement` -> `apply_edits` -> `run_validation`
    - Retries once on failure
    - No orchestrator_run/passes records (single atomic pass, only task_run + run_attempt + validation_result)
    - Returns `OrchestratorResult` with single pass

  **Session state management** (keys per `schemas.md` Section 4):
  - `meta_plan` — serialized MetaPlan
  - `part_plan:{part_id}` — serialized PartPlan
  - `step_result:{part_id}:{step_id}` — serialized StepResult
  - `adjustment:{part_id}:after_{step_id}` — serialized PlanAdjustment
  - `orchestrator_progress` — `{current_part_index, current_step_index, status}`
  - `cumulative_diff` — accumulated search/replace text from all completed steps

  **Key decisions**:
  - Each pipeline call gets a unique sub_task_id derived from orchestrator task_id (colon-separated hierarchy). `task_runs.task_id UNIQUE` constraint satisfied.
  - Temporary plan JSON files for Tier 0 seeding (written to `.clean_room/tmp/`, cleaned up in `finally` block). Minimal plan format: just `{"affected_files": [{"path": "..."}]}` to provide seed file paths.
  - Partial success is explicit: failed steps marked failed, but orchestrator continues to next step/part (doesn't abort). `OrchestratorResult.status = "partial"` if any steps failed.
  - Budget per pass derived from model config: plan passes use `reasoning` model's context_window, implement passes use `coding` model's context_window. `reserved_tokens` from `config["budget"]["reserved_tokens"]`.
  - Orchestrator creates its own session DB (separate from per-pass sessions managed by `run_pipeline`).

- Tests: mock `run_pipeline` + execute functions + `apply_edits` + `run_validation`. Verify pass sequencing, session state persistence, DB logging for all record types, partial success handling, retry logic, adjustment step revision.

---

### WI-8: CLI Commands
**Deps**: WI-4 (for `cra plan`), WI-7 (for `cra solve`)

- `cli.py` — add two commands:

  **`cra plan <task>`**:
  ```python
  @cli.command()
  @click.argument("task")
  @click.option("--repo", "repo_path", type=click.Path(exists=True), default=".")
  @click.option("--stages", default=None, help="Retrieval stages (comma-separated)")
  @click.option("--output", type=click.Path(), default=None, help="Save plan to file")
  def plan(task, repo_path, stages, output): ...
  ```
  - Loads config via `load_config` + `require_models_config`
  - Resolves budget: `context_window` from `ModelRouter.resolve("reasoning").context_window`, `reserved_tokens` from `config["budget"]["reserved_tokens"]`
  - Resolves stages from `--stages` flag or `config["stages"]["default"]` or hard-coded `"scope,precision"`
  - Generates task_id (UUID4)
  - Calls `run_pipeline(task, ..., mode="plan", ...)` -> ContextPackage
  - Creates `LoggedLLMClient` with reasoning model config
  - Calls `execute_plan(context, task, llm, pass_type="meta_plan")` -> MetaPlan
  - Flushes LLM records to raw DB
  - Converts MetaPlan to PlanArtifact JSON (for user consumption)
  - Writes to `--output` file or prints to stdout

  **`cra solve <task>`**:
  ```python
  @cli.command()
  @click.argument("task")
  @click.option("--repo", "repo_path", type=click.Path(exists=True), default=".")
  @click.option("--stages", default=None, help="Retrieval stages (comma-separated)")
  @click.option("--plan", "plan_path", type=click.Path(exists=True), default=None,
                help="Pre-computed plan file (single atomic pass, skips orchestrator)")
  def solve(task, repo_path, stages, plan_path): ...
  ```
  - Loads config, validates `[models]` and `[testing]` sections exist (hard error if missing)
  - If `--plan`: calls `run_single_pass(task, repo_path, config, plan_path=plan_path)`
  - If no `--plan`: calls `run_orchestrator(task, repo_path, config)`
  - Prints summary: status, files modified, tests passed/failed, parts/steps completed

  **Config template updates** (`config.py`):
  - `create_default_config` updated to include:
    ```toml
    [testing]
    test_command = "pytest tests/"
    # lint_command = "ruff check src/"
    # type_check_command = "mypy src/"
    # timeout = 120

    [orchestrator]
    # max_retries_per_step = 1
    # max_adjustment_rounds = 1
    ```

- Tests: CliRunner invocations for `plan` and `solve`, config validation errors, `--output` file writing

---

## Dependency Graph

```
WI-0 (save plan)
  |
WI-1 (DB schema) --------+
WI-2 (dataclasses) -------+---> WI-3 (prompts + parsers)
  |                        |            |
  +---> WI-5 (patch) -----+     WI-4 (execute functions)
  |                        |            |
  +---> WI-6 (validator) -+            |
                           |            |
                           +--- WI-7 (orchestrator)
                                        |
                                 WI-8 (CLI commands)
```

**Parallelization**: WI-1 and WI-2 have no mutual dependency. WI-5 depends only on WI-2 (dataclasses) and can proceed in parallel with WI-3. WI-6 depends on WI-1 + WI-2 and can proceed in parallel with WI-3/WI-4.

---

## Risks

1. **Search/replace ambiguity** (MEDIUM): Model may produce search strings that match multiple locations or don't match at all. Mitigated by: pre-validation before application, exact-match-once requirement, failure context passed to retry prompt.

2. **LLM response format compliance** (MEDIUM): Models may produce malformed JSON or XML despite prompting. Mitigated by: robust parsers (strip markdown fencing, handle common deviations), clear error messages for retry context, and the adjustment pass can work around format issues.

3. **Pipeline call overhead** (LOW): Full N-prompt pipeline per pass (3-4 LLM calls for retrieval + 1 for execute = 4-5 calls per pass). For a 3-part, 3-steps-each task with retries and adjustments: up to ~60 LLM calls. Acceptable with local Ollama inference (no network latency).

4. **Windows git worktree** (DEFERRED): Worktree isolation for patch application deferred to post-MVP. MVP uses direct file modification with pre-validation and rollback. If `validate_edits` passes, application should always succeed.

5. **Test runner portability** (LOW): `shell=True` subprocess behavior differs slightly between Windows and Unix. Mitigated by: configurable test command in config.toml, timeout handling.

---

## Phase 3 Gate (from meta-plan)

- `cra plan <task>` produces valid PlanArtifact JSON for a real codebase
- `cra solve <task> --plan <path>` applies edits and runs tests for a single plan
- `cra solve <task>` runs full orchestrator (meta-plan -> parts -> steps -> tests -> adjustments)
- All LLM calls logged to raw DB with full I/O (traceability chain: `orchestrator_run` -> `orchestrator_passes` -> `task_runs` -> `retrieval_llm_calls`)
- Test validation is mandatory — no code path skips tests
- Coder retry works: failed test -> retry with failure context -> re-test
- Adjustment pass runs after every step (even if step succeeded)
- Partial success tracked: failed steps don't abort remaining work
- Orchestrator state serialized to session DB (keys per schemas.md Section 4)
- Raw DB populated: `task_runs`, `run_attempts`, `validation_results`, `orchestrator_runs`, `orchestrator_passes`

---

## Verification

1. `pip install -e ".[dev]"` succeeds (no new deps)
2. `pytest` passes all tests (existing 474 + new Phase 3 tests)
3. Manual: `cra plan "Add a hello world endpoint"` on a sample repo -> valid JSON plan output
4. Manual: `cra solve "Fix the failing test" --plan plan.json` -> edits applied, tests run
5. Manual: `cra solve "Add input validation to the login function"` -> full orchestrator run with meta-plan, part-plan, step implementation, testing, and adjustment
6. Verify raw DB: `retrieval_llm_calls` has execute-stage entries (`call_type` includes execute pass types), `run_attempts` populated, `validation_results` populated, `orchestrator_runs` + `orchestrator_passes` populated
7. Verify traceability: for any `cra solve` run, trace from `orchestrator_runs` -> `orchestrator_passes` -> `task_runs` -> `retrieval_llm_calls` -> full prompt/response for every LLM call
