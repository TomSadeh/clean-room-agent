# Transparency Doctrine Audit Report

**Date:** 2026-02-26
**Scope:** Full codebase (~10.5K LOC, 57 source files, 13 LLM call sites, 3 databases)
**Auditors:** 5 parallel workstream agents + consolidation review

## Executive Summary

| Workstream | PASS | FAIL | GAP | Total |
|---|---|---|---|---|
| W1: LLM Traceability Chain | 7 | 1 | 2 | 10 |
| W2: Decision Logging Completeness | 8 | 0 | 3 | 11 |
| W3: Context Curation Rules R1-R6 | 5 | 0 | 1 | 6 |
| W4: Orchestrator + Session Traceability | 5 | 2 | 3 | 10 |
| W5: Cross-Cutting Concerns | 2 | 3 | 2 | 7 |
| **Total** | **27** | **6** | **11** | **44** |

**By severity:**
- P0 Critical: 1 (schema constraint violation crashes documentation pass logging)
- P1 High: 4 (thinking not in raw DB, error message swallowed, no error_message column, doc pass linkage broken)
- P2 Medium: 8 (similarity denied pairs, enrichment skips, rollback events, commit SHAs, hardcoded defaults, task_id validation, task_id threading, R6 minor)

---

## P0 Findings

### P0-1: Documentation pass orchestrator_pass crashes on NOT NULL constraint

**Checkpoints:** W4-C10, W4-C2
**Files:** `db/schema.py:303`, `orchestrator/runner.py:645-648`

The `orchestrator_passes.task_run_id` column is declared `INTEGER NOT NULL REFERENCES task_runs(id)` (schema.py:303). The documentation pass inserts `None` for this column (runner.py:646):

```python
insert_orchestrator_pass(
    raw_conn, orch_run_id, None, "documentation", sequence_order,
    part_id=part.id,
)
```

This will raise `sqlite3.IntegrityError` at runtime, crashing the documentation pass's DB logging. The documentation pass does not run a retrieval pipeline, so there is no `task_run` record to reference.

**Impact:** Every documentation pass fails to record its orchestrator_pass. The exception propagates to the except handler at line 661, masking the real cause. The documentation work (file edits, commit) may succeed, but its traceability record is lost.

**Fix:** Make `task_run_id` nullable in the schema (drop NOT NULL) and add a migration:
```sql
-- In create_raw_schema, after existing migrations:
-- Cannot ALTER to drop NOT NULL in SQLite; must handle at insert level.
```
Since SQLite cannot ALTER a column to drop NOT NULL, the practical fix is:
1. For new DBs: change DDL to `task_run_id INTEGER REFERENCES task_runs(id)` (without NOT NULL).
2. For existing DBs: the constraint already exists but only fires on INSERT, so no migration needed — just ensure the application handles nullable `task_run_id`.
3. Update `insert_orchestrator_pass` signature: `task_run_id: int | None`.

---

## P1 Findings

### P1-1: Thinking content not persisted to raw DB

**Checkpoint:** W1-C6
**Files:** `db/schema.py:172-184`, `db/raw_queries.py:42-61`, `llm/client.py:214-215`

The `retrieval_llm_calls` table has no `thinking` column. `LoggedLLMClient.complete()` captures thinking content in the call record (client.py:214-215), but `insert_retrieval_llm_call` has no `thinking` parameter and the column does not exist. The `enrichment_outputs` table also lacks a `thinking` column.

Thinking content is only preserved in the ephemeral markdown trace file (`trace.py:46`).

**Impact:** The raw DB is the training corpus (CLAUDE.md: "This IS the training corpus"). CoT-SFT training requires thinking chains. Losing them means the most valuable signal for distillation is discarded.

**Fix:**
1. Add `thinking TEXT` column to `retrieval_llm_calls` (migration: `ALTER TABLE retrieval_llm_calls ADD COLUMN thinking TEXT`).
2. Add `thinking` parameter to `insert_retrieval_llm_call()`.
3. Pass `call.get("thinking")` at all flush sites (pipeline.py lines 181-186, runner.py `_flush_llm_calls`).
4. Optionally add `thinking TEXT` to `enrichment_outputs`.

### P1-2: Orchestrator error message swallowed (not in raw DB)

**Checkpoint:** W5-C3
**Files:** `orchestrator/runner.py:896-898`, `orchestrator/runner.py:1110-1112`

When the orchestrator catches a top-level exception:
```python
except (ValueError, RuntimeError, OSError) as e:
    logger.error("Orchestrator failed: %s", e)
    status = "failed"
```

The error message `str(e)` is logged to Python's ephemeral logger but not stored in the raw DB. A human tracing a failed run through the DB sees `status="failed"` with no explanation.

Inner exception handlers (lines 405-410, 511-516, 661-665, 722-727, 813-818) follow the same pattern: log via logger, no DB record of the error.

**Impact:** Directly violates the traceability principle. Failed runs cannot be diagnosed from the raw DB alone.

**Fix:** See P1-3 (add error_message column) and store `str(e)` when finalizing failed runs.

### P1-3: No error_message column in orchestrator_runs

**Checkpoint:** W5-C7
**Files:** `db/schema.py:286-298`, `db/raw_queries.py:200-240`

The `orchestrator_runs` table has no `error_message` or `error` column. `update_orchestrator_run` has no error_message parameter.

**Fix:**
1. Add migration: `ALTER TABLE orchestrator_runs ADD COLUMN error_message TEXT`.
2. Add `error_message: str | None = None` parameter to `update_orchestrator_run`.
3. Pass `str(e)` from both `run_orchestrator` and `run_single_pass` exception handlers via `_finalize_orchestrator_run`.

### P1-4: Documentation pass LLM calls not linkable to task_runs

**Checkpoint:** W1-C10
**Files:** `orchestrator/runner.py:624-635`

Documentation LLM calls are logged to `retrieval_llm_calls` with `task_id=f"{task_id}:doc:{part.id}"`, but no `task_runs` entry exists for that sub-task-id. The LLM calls exist in the raw DB but cannot be joined to a `task_runs` record.

**Impact:** Training data extraction queries that join `retrieval_llm_calls` to `task_runs` will miss all documentation pass LLM calls.

**Fix:** Either:
- Insert a synthetic `task_run` for the doc pass sub-task-id before the orchestrator_pass insert, or
- Accept that documentation LLM calls are linked only via `task_id` string prefix matching (weaker traceability).

---

## P2 Findings

### P2-1: Similarity denied pairs not in raw DB

**Checkpoint:** W2-C9
**Files:** `retrieval/similarity_stage.py:144-216`

Denied similarity pairs are only logged via `logger.warning()`. The LLM call I/O is preserved (so decisions are reconstructable from raw response text), but no structured per-pair record exists in `retrieval_decisions`.

**Fix:** In `judge_similarity()`, call `insert_retrieval_decision()` for each denied pair with `included=False`, `reason="similarity_denied: {reason}"`.

### P2-2: Enrichment R3 skip reasons only in logger.warning

**Checkpoint:** W2-C10
**Files:** `llm/enrichment.py:89-98`

Files skipped due to R3 (prompt too large) are recorded only via `logger.warning()`. No raw DB record of the skip.

**Fix:** Call `insert_enrichment_output()` with `raw_response="R3_SKIP"` for skipped files, or add a dedicated `enrichment_skips` table.

### P2-3: Enrichment "already enriched" skips are silent

**Checkpoint:** W2-C11
**Files:** `llm/enrichment.py:77-84`

Already-enriched files are skipped with no log and no DB record. The prior enrichment record exists as implicit evidence, so this is lowest severity.

**Fix:** Add `logger.debug("Enrichment skip: %s already enriched (id=%d)", file_path, existing["id"])`.

### P2-4: LIFO rollback events not logged to raw DB

**Checkpoint:** W4-C7
**Files:** `orchestrator/runner.py:853-875`

Rollback events (both git and LIFO paths) are only logged via `logger.info/debug`. The raw DB shows `run_attempt` records with `patch_applied=True` but no record that patches were subsequently reverted. Validation failure is logged (via `insert_validation_result`), but the rollback action itself is invisible.

**Fix:** After rollback, either:
- Mark affected `run_attempts` as `patch_applied=False`, or
- Insert a dedicated rollback record with trigger reason and reverted patches/commits.

### P2-5: Git commit SHAs not persisted to raw DB

**Checkpoint:** W4-C8
**Files:** `orchestrator/runner.py:526-528,640,825-827,1086`

`git.commit_checkpoint()` returns the commit SHA, but the return value is discarded at every call site. `part_start_sha` (line 361) is captured for rollback but also never stored. When the task branch is deleted (always, via `_git_cleanup`), the commit history is unrecoverable.

**Fix:** Add a `commit_sha` column to `orchestrator_passes` and update each pass record with the SHA after committing, or add a `git_commits` table.

### P2-6: Hardcoded config defaults

**Checkpoint:** W5-C4
**Files:** `orchestrator/runner.py:260,611,52`, `llm/router.py:48,55`

Several config values have hardcoded fallbacks instead of fail-fast required config:
- `max_adjustment_rounds` defaults to `3` (runner.py:260)
- `documentation_pass` defaults to `True` (runner.py:611)
- `max_cumulative_diff_chars` defaults to `50_000` (runner.py:52,261)
- `max_tokens` defaults to `{"coding": 4096, "reasoning": 4096}` (router.py:55)
- `temperature` defaults to `0.0` (router.py:48)

Contrast with correctly fail-fast patterns: `max_retries_per_step`, `git_workflow`, `reserved_tokens`.

**Fix:** Make `max_adjustment_rounds` and `max_cumulative_diff_chars` required in the config template (uncomment). For `documentation_pass`, either make it required or add a code comment documenting the intentional default.

### P2-7: No task_id validation in TaskQuery

**Checkpoint:** W5-C2
**Files:** `retrieval/dataclasses.py:52-56`

`TaskQuery.__post_init__` validates `task_type` but not that `task_id` is non-empty. All current callers generate UUIDs (making empty values impossible), but an empty `task_id` would silently propagate and break traceability. `OrchestratorResult` already has this validation.

**Fix:** Add `if not self.task_id: raise ValueError("task_id must be non-empty")` to `TaskQuery.__post_init__`.

### P2-8: R6 minor — similarity group cap ordering

**Checkpoint:** W3-C6
**Files:** `retrieval/similarity_stage.py:268`

`sorted(members)[:max_group_size]` uses symbol ID ordering rather than a relevance criterion. All group members are LLM-confirmed similar (equal relevance), so the practical impact is negligible, but R6 technically requires justification.

**Fix:** Add a comment explaining why symbol ID ordering is appropriate for equal-relevance members.

---

## Supplementary Finding

### S-1: Enrichment uses raw LLMClient (no thinking/latency/token logging)

**Checkpoint:** W1-C4 (supplementary)
**Severity:** P2
**Files:** `llm/enrichment.py:54`

Enrichment uses raw `LLMClient(model_config)` instead of `LoggedLLMClient`. While `insert_enrichment_output` captures prompt/response/system_prompt, it does not capture: thinking content, latency_ms, token counts, or error calls. Enrichment is offline and not task-driven, so this is lower severity.

---

## Summary of Required Fixes (sorted by severity)

| ID | Sev | Issue | Files to Change |
|---|---|---|---|
| P0-1 | P0 | Doc pass NULL task_run_id vs NOT NULL | `db/schema.py`, `db/raw_queries.py` |
| P1-1 | P1 | Thinking not in raw DB | `db/schema.py`, `db/raw_queries.py`, `retrieval/pipeline.py`, `orchestrator/runner.py` |
| P1-2 | P1 | Error message swallowed | `orchestrator/runner.py` |
| P1-3 | P1 | No error_message column | `db/schema.py`, `db/raw_queries.py`, `orchestrator/runner.py` |
| P1-4 | P1 | Doc pass LLM calls not linkable | `orchestrator/runner.py` |
| P2-1 | P2 | Similarity denied pairs | `retrieval/similarity_stage.py` |
| P2-2 | P2 | Enrichment R3 skip reasons | `llm/enrichment.py` |
| P2-3 | P2 | Enrichment already-enriched skips | `llm/enrichment.py` |
| P2-4 | P2 | Rollback events not logged | `orchestrator/runner.py` |
| P2-5 | P2 | Git commit SHAs not stored | `orchestrator/runner.py`, `db/schema.py` |
| P2-6 | P2 | Hardcoded config defaults | `orchestrator/runner.py`, `llm/router.py` |
| P2-7 | P2 | No task_id validation in TaskQuery | `retrieval/dataclasses.py` |
| P2-8 | P2 | R6 minor group cap ordering | `retrieval/similarity_stage.py` |
| S-1 | P2 | Enrichment raw LLMClient | `llm/enrichment.py` |

---

## Passing Checkpoints (27/44)

For completeness, the following checkpoints passed with full compliance:

**W1:** C1 (LoggedLLMClient full I/O), C2 (EnvironmentLLMClient brief embedded), C3 (all retrieval uses LoggedLLMClient), C4 (enrichment logged to raw DB), C5 (flush writes system_prompt), C7 (error calls logged with [ERROR]), C8 (TraceLogger captures all metadata), C9 (orchestrator execute-phase uses LoggedLLMClient)

**W2:** C1 (per-file scope decisions logged with dedup), C2 (per-symbol precision decisions logged), C3 (assembly decisions flow to raw DB), C4 (R2 default-deny logged), C5 (refilter/priority/group drops logged), C6 (budget overflow drops logged), C7 (scope LLM omission defaults to irrelevant), C8 (precision LLM omission defaults to excluded)

**W3:** C1 (R1 no degradation), C2 (R2 default-deny comprehensive), C3 (R3 all 13 call sites validated), C4 (R4 AST-based rendering), C5 (R5 framing overhead tracked)

**W4:** C1 (orchestrator_run lifecycle), C3 (run_attempt records complete), C4 (validation full output to raw DB), C5 (session state comprehensive), C6 (session archived and deleted in finally)

**W5:** C5 (test coverage adequate), C6 (no silent exception swallowing)

---

## Architecture Assessment

The codebase demonstrates strong transparency compliance in its core pipeline. The retrieval system (Phases 1-2) is particularly well-built: all 6 Context Curation Rules are enforced, decision logging is comprehensive with dedup tracking, and the default-deny principle is consistently applied across scope, precision, and similarity stages.

The gaps are concentrated in two areas:
1. **Schema gaps** (P0-1, P1-1, P1-3) — missing columns prevent logging data that the application layer already captures.
2. **Orchestrator error/event recording** (P1-2, P2-4, P2-5) — the orchestrator logs milestones well but does not persist failure details or rollback/git events to the raw DB.

These are fixable with targeted schema additions and ~50 lines of application code changes. No architectural rework is needed.
