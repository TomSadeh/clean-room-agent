# Refactoring Implementation Plan (A1-A14, R1-R18, R22)

## Context

The repo review identified 22 refactoring opportunities. R19-R21 (LLM enhancements) are deferred to a separate feature pass. A subsequent transparency doctrine audit (`research_reviews/transparency_audit.md`, 2026-02-26) identified 14 findings (1 P0, 4 P1, 8 P2, 1 supplementary P2) that must be addressed.

P0/P1 audit fixes run first (Batch 0) because they are correctness/schema bugs — decomposing runner.py (Batch 4) while it has schema violations would bake bugs into new helpers. P2 audit fixes are integrated into the batches that already touch the same files.

This plan covers all 14 audit findings + 19 refactoring items, organized into 7 batches. Each batch must leave all ~1050 tests green.

---

## Batch 0: Audit Fixes — P0/P1 (A1-A5) — DONE

All 5 fixes implemented and verified. 12 new tests in `tests/test_batch0_audit_fixes.py`. 1062 tests passing.

- **A1 (P0):** `orchestrator_passes.task_run_id` → nullable in DDL + `insert_orchestrator_pass` accepts `int | None`
- **A2 (P1):** `thinking TEXT` column added to `retrieval_llm_calls` (migration + insert function + all 6 flush sites in pipeline.py, runner.py, cli.py)
- **A3 (P1):** `error_message TEXT` column added to `orchestrator_runs` (migration + `update_orchestrator_run`)
- **A4 (P1):** `_finalize_orchestrator_run` accepts `error_message`; both `run_orchestrator` and `run_single_pass` except blocks capture `str(e)` and pass it
- **A5 (P1):** Documentation pass creates synthetic `task_run` via `insert_task_run` before `insert_orchestrator_pass`, linking doc LLM calls to `task_runs`

---

## Batch 1: Foundations (R8, R5, R12, A10) — DONE

All 4 items implemented and verified. 1062 tests passing.

- **R8:** Created `db/helpers.py` with `_now()`, `_insert_row()`, `_build_update_clause()`. Both `db/raw_queries.py` and `db/queries.py` import from helpers. `update_orchestrator_run` refactored to use `_build_update_clause()`.
- **R5:** Collapsed 7 individual prompt string constants into single `SYSTEM_PROMPTS` dict in `execute/prompts.py`. Updated `tests/execute/test_prompts.py` to use `SYSTEM_PROMPTS["key"]` instead of individual names.
- **R12:** Added `_NON_EMPTY` class var + default `__post_init__` to `_SerializableMixin`. Replaced manual `__post_init__` in 8 dataclasses (PlanStep, MetaPlanPart, MetaPlan, PartPlan, PlanAdjustment, PlanArtifact, PatchEdit, PassResult). `OrchestratorResult` calls `super().__post_init__()` then custom status-enum check.
- **A10 (P2):** Added `task_id` non-empty validation to `TaskQuery.__post_init__`.

---

## Batch 2: Isolated Low-Risk (R9, R14, R15, R16, A8, A9, A14) — DONE

All 7 items implemented and verified. 1062 tests passing.

- **R9+R14:** Added `ScopeCandidate(NamedTuple)` and `_dedup_by_score()` helper to `scope_stage.py`. Replaced 3 identical dedup+sort blocks and all raw tuple indexing with named fields.
- **R15:** Replaced explicit `config`/`flush`/`close` delegations in `EnvironmentLLMClient` with `__getattr__` to `_inner`. Kept `complete()` (custom) and `__enter__`/`__exit__` (dunder dispatch).
- **R16:** Narrowed `except Exception:` to specific types in `library_scanner.py` (`OSError|ValueError|SyntaxError` for parse, `ImportError|ValueError|AttributeError|ModuleNotFoundError` for find_spec). Added `logger.debug()`.
- **A8 (P2):** R3 skip now calls `insert_enrichment_output()` with `R3_SKIP` response — raw DB record for every skipped file.
- **A9 (P2):** Already-enriched skips now log via `logger.debug()`.
- **A14 (P2):** Enrichment captures `thinking`, `prompt_tokens`, `completion_tokens`, `latency_ms` from LLM response. Added 4 new columns to `enrichment_outputs` (schema migration + `insert_enrichment_output` signature).

---

## Batch 3: Code Simplification (R7, R10, R11, R13)

Cross-file but low risk. Each item is self-contained.

### R7: Patch validation dedup

**Modify** `src/clean_room_agent/execute/patch.py`:
- Extract `_validate_and_simulate(edits, repo_path, *, track_originals=False) -> (simulated, originals, errors)`
- `validate_edits()` calls it and returns errors
- `apply_edits()` calls it with `track_originals=True` then proceeds to write

### R10 + R11: TS/JS parser cleanup

**Modify** `src/clean_room_agent/parsers/ts_js_parser.py`:
- R10: Extract `_extract_import_clause_names(clause_node) -> list[str]` from `_parse_es_import` (flatten 9-level nesting to 4)
- R10: Extract `_extract_require_names(name_node) -> list[str]` from `_parse_commonjs_require` (flatten 8-level nesting)
- R11: Merge duplicate blocks in `_extract_variable_symbols` — single block with `kind = "function" if value_node else "variable"`

### R13: Row converter factory

**Modify** `src/clean_room_agent/query/api.py`:
- Add `_make_row_converter(dc_type)` using `dataclasses.fields()` introspection
- Replace 9 of 10 `_row_to_*` static methods (keep `_row_to_comment` which has `bool()` cast)

**Test**: Run full suite.

---

## Batch 4: Runner Decomposition (R1, R2, A6, A7, A11) — Highest Risk

Incremental extraction from `runner.py`. Audit fixes (A6, A7, A11) are integrated into the decomposition — they fix bugs in the code being extracted, so doing them together avoids double-touching the same functions.

Test after each step.

### Step 1: `_OrchestratorContext` dataclass

**Add** to `runner.py`:
```python
@dataclass
class _OrchestratorContext:
    task_id: str
    repo_path: Path
    config: dict
    raw_conn: sqlite3.Connection
    session_conn: sqlite3.Connection
    git: GitWorkflow | None
    trace_logger: TraceLogger | None
    orch_run_id: int
    sequence_order: int
    cumulative_diff: str
    max_diff_chars: int
    pass_results: list[PassResult]
    # Resolved from config once at init:
    stage_names: list[str]
    budget: BudgetConfig
    coding_config: ModelConfig
    reasoning_config: ModelConfig | None  # None for single_pass
```

### Step 2: `_init_orchestrator()` (R2 part 1)

Extract the duplicated init blocks (lines 244-311 and 954-1014):
```python
def _init_orchestrator(task, repo_path, config, trace_logger, *, needs_reasoning=True) -> _OrchestratorContext
```
Handles: task_id generation, router setup, config resolution, DB connections, git setup, orch_run record.

### Step 3: `_cleanup_orchestrator()` (R2 part 2, incorporates A11)

Extract the duplicated finally blocks (lines 900-929 and 1114-1137):
```python
def _cleanup_orchestrator(ctx: _OrchestratorContext, status: str, error_message: str | None = None) -> None
```
Handles: finalize orch run (with error_message from A4), session state, git cleanup, archive session, close connections.

**A11 integration (P2-6 hardcoded defaults):** During init extraction, replace hardcoded defaults with fail-fast validation:
- `max_adjustment_rounds`: require in config (remove `.get("max_adjustment_rounds", 3)`)
- `max_cumulative_diff_chars`: require in config (remove `.get("max_cumulative_diff_chars", _MAX_CUMULATIVE_DIFF_CHARS)`)
- `documentation_pass`: keep as optional with default `True` but add code comment documenting intentional default (cosmetic feature, not behavioral)

**Modify** `src/clean_room_agent/config.py`:
- Uncomment `max_adjustment_rounds` and `max_cumulative_diff_chars` in the default config template (make them always present)

### Step 4: `_update_cumulative_diff()` (R1 part 1, incorporates A7)

Extract the 4 repeated diff-update blocks:
```python
def _update_cumulative_diff(ctx, edits, commit_msg) -> str  # returns new SHA from commit_checkpoint
```

**A7 integration (P2-5 git commit SHAs):** Store the return value of `git.commit_checkpoint()`. The extracted helper returns the commit SHA, which the caller passes to the orchestrator_pass record.

**Modify** `src/clean_room_agent/db/schema.py`:
- Add migration: `ALTER TABLE orchestrator_passes ADD COLUMN commit_sha TEXT`

**Modify** `src/clean_room_agent/db/raw_queries.py`:
- Add `commit_sha: str | None = None` parameter to `insert_orchestrator_pass()`

### Step 5: `_run_plan_stage()` (R1 part 2)

Extract the pipeline+LLM+flush+record pattern (7 occurrences):
```python
def _run_plan_stage(ctx, *, task_desc, pass_type, plan_artifact_path=None, ...) -> plan_result
```
Handles: `run_pipeline()` -> `LoggedLLMClient` context -> `execute_plan()` -> `_flush_llm_calls()` -> DB records -> state update. Increments `ctx.sequence_order`.

### Step 6: `_execute_code_step()` (R1 part 3)

Extract the per-step retry loop (lines 424-542):
```python
def _execute_code_step(ctx, step, part_plan, part_id, max_retries) -> tuple[bool, StepResult | None, list[PatchResult]]
```

### Step 7: `_execute_test_loop()` (R1 part 4)

Extract the test step loop (lines 729-836):
```python
def _execute_test_loop(ctx, test_plan, part_id, max_retries) -> tuple[bool, list[PatchResult]]
```

### Step 8: `_rollback_part()` (incorporates A6)

**Audit finding A6:** P2-4 (W4-C7). Rollback events are not logged to raw DB. The DB shows `run_attempt` records with `patch_applied=True` but no record that patches were reverted.

Extract the rollback logic (lines 853-875) into a dedicated helper:
```python
def _rollback_part(ctx, *, part_id: str, code_patches, test_patches, doc_patches, validation_result_id: int) -> None
```

This helper:
- Performs LIFO rollback (test → doc → code) or git rollback
- **Logs rollback events to raw DB**: update affected `run_attempts` to `patch_applied=False`
- Records the trigger (validation failure) via a retrieval_decision with `stage="rollback"` and reason linking to the validation_result_id

### Step 9: Rewrite `run_orchestrator()` and `run_single_pass()`

With all helpers extracted, both functions become ~80-120 lines each: init -> orchestration logic -> cleanup. `run_orchestrator` is the full loop; `run_single_pass` is the single-step shortcut.

**Test**: Run `tests/orchestrator/test_runner.py` after each step. Run full suite after steps 3, 5, and 9.

---

## Batch 5: Cross-Module (R6, R22, R3, A6b, A13)

### R6 + R22 + A6b + A13: Shared batched judgment runner + similarity logging

**Modify** `src/clean_room_agent/retrieval/batch_judgment.py`:
- Add `run_batched_judgment(candidates, *, system_prompt, task_header, llm, tokens_per_candidate, format_candidate, extract_key, stage_name) -> dict[key, judgment_dict]`
- Add `log_r2_omission(label, stage_name, default_action)` (R22)

**Modify** scope_stage.py, precision_stage.py, similarity_stage.py:
- Replace the ~40-line batch loop in each with a call to `run_batched_judgment()`
- Each stage provides its own `format_candidate` and `extract_key` callbacks
- R2 default-deny application stays in each stage (different defaults) but uses the shared `log_r2_omission`

**A6b (P2-1, W2-C9):** When refactoring `similarity_stage.py` to use `run_batched_judgment()`, add decision logging for denied pairs. The shared runner returns all judgments (including denials); the stage calls `insert_retrieval_decision()` for denied pairs with `included=False`, `reason="similarity_denied: {reason}"`.

**A13 (P2-8, W3-C6):** Add R6 justification comment at the group cap slice (`sorted(members)[:max_group_size]`): "R6: all group members have equal relevance (confirmed similar by LLM); sorting by symbol_id provides deterministic selection."

### R3: CLI `commands/` extraction

**Create** `src/clean_room_agent/commands/__init__.py`, `retrieve.py`, `plan.py`, `solve.py`

**Modify** `src/clean_room_agent/cli.py`:
- Move `_resolve_budget()` and `_resolve_stages()` to `commands/__init__.py`
- Move `retrieve` logic (lines 142-224) to `commands/retrieve.py: run_retrieve()`
- Move `plan` logic (lines 289-386) to `commands/plan.py: run_plan()`
- Move `solve` logic (lines 396-443) to `commands/solve.py: run_solve()`
- CLI layer retains only Click decorators, arg parsing, config loading, output formatting

**Test**: Run full suite.

---

## Batch 6: Indexer + Fail-Fast (R4, R17, R18)

### R4: Indexer decomposition

**Modify** `src/clean_room_agent/indexer/orchestrator.py`:
- Extract `_insert_file_symbols(conn, file_id, parsed_result) -> list[dict]` (shared by `_do_index` and `index_libraries`)
- Extract `_attach_docstrings_comments(conn, file_id, parsed_result, symbol_records)`
- Extract `_resolve_deps_and_refs(conn, file_id_map, parse_results, ...)`
- Extract `_index_git_history(conn, repo_id, repo_path, file_id_map, ...)`

### R17: Precision stage pre-filter (R1 violation fix)

**Modify** `src/clean_room_agent/retrieval/precision_stage.py`:
- Before LLM batching, partition: library candidates auto-classified as `type_context`, only project candidates enter LLM judgment
- Remove post-hoc downgrade block (lines 194-200)
- Update tests that expected the downgrade behavior

### R18: Assembly refilter fail-fast

**Modify** `src/clean_room_agent/retrieval/context_assembly.py`:
- When LLM refilter returns only invalid paths, raise `ValueError` instead of falling back to priority drop
- Budget-exceeded priority drop (non-LLM path) remains unchanged
- Update tests expecting fallback behavior

**Test**: Run full suite.

---

## Verification Plan

After each batch:
1. `python -m pytest tests/ -x -q` — all ~1050 tests must pass
2. Spot-check the refactored module's test file specifically
3. Verify no new imports of removed symbols (e.g., individual prompt constants)

After Batch 0 specifically:
1. Verify schema migrations are idempotent (run `create_raw_schema` twice)
2. Verify doc pass orchestrator_pass insert succeeds with `task_run_id=None` on new DBs
3. Verify thinking content round-trips: insert with thinking → SELECT → matches
4. Verify error_message round-trips: update with error → SELECT → matches

After all batches:
1. Full test suite green
2. `python -m pytest tests/ --tb=short` for a clean summary
3. Line count check: `wc -l` on runner.py (target: <750), cli.py (target: <250), scope_stage.py, patch.py
4. Verify TODO.md items can be marked complete
5. Re-run transparency audit checkpoints W1-C6, W1-C10, W4-C7, W4-C8, W4-C10, W5-C3, W5-C7 to confirm PASS

---

## Audit Finding Cross-Reference

All 14 findings from `research_reviews/transparency_audit.md` mapped to batches:

| Finding | Sev | Batch | Item |
|---|---|---|---|
| P0-1: Doc pass NULL task_run_id | P0 | 0 | A1 |
| P1-1: Thinking not in raw DB | P1 | 0 | A2 |
| P1-3: No error_message column | P1 | 0 | A3 |
| P1-2: Error message swallowed | P1 | 0 | A4 |
| P1-4: Doc pass LLM calls unlinkable | P1 | 0 | A5 |
| P2-7: No task_id validation in TaskQuery | P2 | 1 | A10 |
| P2-2: Enrichment R3 skip reasons | P2 | 2 | A8 |
| P2-3: Enrichment already-enriched skips | P2 | 2 | A9 |
| S-1: Enrichment raw LLMClient metadata | P2 | 2 | A14 |
| P2-4: Rollback events not logged | P2 | 4 | A6 (step 8) |
| P2-5: Git commit SHAs not stored | P2 | 4 | A7 (step 4) |
| P2-6: Hardcoded config defaults | P2 | 4 | A11 (step 3) |
| P2-1: Similarity denied pairs | P2 | 5 | A6b |
| P2-8: R6 minor group cap ordering | P2 | 5 | A13 |

---

## Items Deferred

- **R19**: Task-aware budget tie-breaking — new LLM call, separate feature pass
- **R20**: LLM-ranked dependency ordering — new LLM call, separate feature pass
- **R21**: Docstring-based pre-filter — new LLM call, separate feature pass
