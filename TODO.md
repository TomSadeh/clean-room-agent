# TODO

Consolidated findings from code reviews. Items reference the core transparency
principle and Context Curation Rules (R1-R6) from CLAUDE.md.

---

## ~~P0 — Critical~~ — ALL FIXED

### ~~T25. Adjustment pass is dead code — revised steps never applied~~ — FIXED

Converted `for` loop to index-based `while` loop. Adjustment `revised_steps` now spliced
into `sorted_steps[step_idx + 1:]`. Tests: `test_adjustment_revises_remaining_steps`.

### ~~T26. `steps_completed` counts all steps, not just successful ones~~ — FIXED

Moved `steps_completed += 1` inside `if step_success` block.
Tests: `test_steps_completed_counts_only_successes`.

### ~~T27. `patch_applied` column logged before patch is actually applied~~ — FIXED

`insert_run_attempt` now always passes `patch_applied=False`. New
`update_run_attempt_patch()` called after `apply_edits` succeeds. Same fix in
`run_single_pass`.

### ~~T28. LLM calls lost when `execute_plan` raises in `cra plan`~~ — FIXED

Wrapped `execute_plan` in try/finally; LLM flush runs in `finally` block with its own
connection management.

### ~~T29. `failing_tests` stored as comma-separated string~~ — FIXED

Changed to `json.dumps(failing_tests)`. Test: `test_test_fails` verifies JSON roundtrip.

### ~~T30. CLI `_resolve_budget` will crash with per-role dict~~ — FIXED

Rewritten to use `ModelRouter.resolve(role)` — same approach as orchestrator's
`_resolve_budget`.

### ~~T31. `_atomic_write` is not atomic on Windows~~ — FIXED

Added retry loop (5 retries on Windows, 0 on POSIX) with temp file cleanup on failure.

---

## ~~P1 — High~~ — ALL FIXED

### ~~T32. `_topological_sort` silently falls back on cycles~~ — FIXED

Now raises `RuntimeError` instead of silently returning original order.

### ~~T33. Broad `except Exception` swallows programming bugs~~ — FIXED

Narrowed all 4 catches to `(ValueError, RuntimeError, OSError)`. Programming bugs
(`TypeError`, `KeyError`, `AttributeError`) now propagate.

### ~~T34. `rollback_edits` silently swallows errors~~ — FIXED

Now collects all rollback errors and raises `RuntimeError` after attempting all files.

### ~~T35. `PlanAdjustment` is never validated~~ — FIXED

`execute_plan` now wraps adjustment `revised_steps` in a synthetic `PartPlan` and runs
`validate_plan()` (cycle/duplicate detection).

### ~~T36. Hardcoded defaults in orchestrator~~ — FIXED

Removed fallback defaults from `_resolve_budget`, `_resolve_stages`, `max_retries_per_step`.
All raise `RuntimeError` if missing from config. Config template updated with
`max_retries_per_step = 1` uncommented.

### ~~T37. `cra solve --stages` flag is a dead parameter~~ — FIXED

Removed the `--stages` flag from the `solve` CLI command.

### ~~T38. Part/step failure creates `task_run_id=0`~~ — FIXED

`PassResult.task_run_id` is now `int | None` (default `None`). Failure cases use `None`.

### ~~T39. Adjustment failure silently swallowed~~ — FIXED

Adjustment exceptions now produce a `PassResult(pass_type="adjustment", success=False)`
in the orchestrator result.

### ~~T40. Token counts always None in run_attempts~~ — FIXED

`_flush_llm_calls` now returns `(prompt_tokens, completion_tokens, latency_ms)` totals.
Used in `insert_run_attempt` for both `run_orchestrator` and `run_single_pass`.

### ~~T41. Session DB never archived or deleted~~ — FIXED

Added archive-to-raw + file deletion in the `finally` block of `run_orchestrator`.

### ~~T42. TOCTOU race between validate and apply~~ — FIXED

`apply_edits` now reads each file exactly once and does validation + application from
the same in-memory content. `validate_edits` remains for standalone validation.

### ~~T43. No path traversal validation~~ — FIXED

Added `_check_path_traversal()` — checks `resolved.relative_to(repo_resolved)`. Called
in both `validate_edits` and `apply_edits`.

### ~~T44. Task description duplicated in prompts~~ — FIXED

`build_plan_prompt` now only adds a `# Current Objective` section when `task_description`
differs from `context.task.raw_task`.

### ~~T45. `all_steps_count in dir()` fragile~~ — FIXED

Initialized at function top alongside other counters. Also removed dead
`total_steps = sum(0 for ...)`.

### ~~T46. Config section/parameter name mismatches~~ — FIXED

Removed dead `[solve]` section from config template. `max_retries_per_step` uncommented
in `[orchestrator]`.

---

## ~~P2 — Medium (dead code, edge cases, spec drift)~~ — ALL FIXED

### ~~T47. `VALID_PASS_TYPES` defined but never used~~ — FIXED

Deleted the dead constant.

### ~~T48. `OrchestratorResult.from_dict` drops `pass_results`~~ — FIXED

Added `PassResult.from_dict()`. `OrchestratorResult.from_dict()` now deserializes
`pass_results` on round-trip. Tests: `test_from_dict`, `test_round_trip_with_pass_results`.

### ~~T49. Edit parser regex vulnerable to XML content injection~~ — FIXED

Rewrote parser: outer regex finds `<edit>...</edit>` blocks, inner `<search>` and
`<replacement>` tags parsed sequentially using `rfind` for closing tags. Code containing
literal `</search>` or `</replacement>` no longer breaks parsing (R5).
Tests: `test_content_with_search_tag`, `test_content_with_replacement_tag`.

### ~~T50. `search.strip("\n")` too aggressive in parser~~ — FIXED

New `_strip_one_newline()` strips exactly one leading and one trailing newline (the
formatting ones between tag and content). Preserves content newlines.
Test: `test_strips_one_formatting_newline_only`.

### ~~T51. `total_steps = sum(0 for _ in meta_plan.parts)` is always 0~~ — FIXED (P1, T45)

Already removed during P1 fixes.

### ~~T52. Cumulative diff unbounded~~ — FIXED

Added `_cap_cumulative_diff()` with `_MAX_CUMULATIVE_DIFF_CHARS = 50_000` (~12,500 tokens).
Truncates oldest entries first, aligns to block boundary. Applied in both
`run_orchestrator` and `run_single_pass`.
Tests: `test_short_diff_unchanged`, `test_long_diff_truncated`,
`test_truncation_aligns_to_block_boundary`.

### ~~T53. Vacuous assertion in test~~ — FIXED (P1)

Already fixed during P1 — changed to `assert result.exception is not None`.

### ~~T54. Line ending issues on Windows~~ — FIXED

Parser: `_strip_one_newline()` handles both `\r\n` and `\n`.
Patch: `_atomic_write` now uses binary mode (`mode="wb"`) to prevent `\n` → `\r\n`
conversion on Windows.
Tests: `test_handles_crlf_line_endings`, `test_preserves_lf_endings`.

### ~~T55. Empty `file_path` check unreachable in parser~~ — FIXED

Removed unreachable dead code. The regex `[^"]+` prevents empty file paths from matching.
`PatchEdit.__post_init__` validates non-empty as a second line of defense.

### ~~T56. `model_config` passed separately from `LoggedLLMClient` that already contains it~~ — FIXED

Removed `model_config` parameter from `execute_plan` and `execute_implement`. Both now
derive it from `llm.config`. Updated all callers in `runner.py` and `cli.py`.

---

## ~~P3 — Low Priority~~ — ALL FIXED

### ~~T22. `__del__` with `except Exception: pass`~~ — FIXED

Narrowed except to `(OSError, AttributeError, TypeError)` — the specific failures that
can occur during destructor cleanup (network errors, partially initialized objects,
interpreter shutdown). Added comment explaining why pass is acceptable here.

### ~~T23. Orphan commits in curated DB~~ — FIXED

Commits now filtered before insertion: only commits touching at least one tracked file
(in `file_id_map`) are inserted. Commits touching only excluded files are skipped.

### ~~T24. TOCTOU: hash and parse may read different file versions~~ — FIXED

After reading file bytes for parsing, SHA-256 hash is verified against the hash computed
during scanning. `RuntimeError` raised on mismatch (fail-fast).

---

## Completed

### ~~T20. `_LoggingLLMClient` is pipeline-internal, not system-wide~~ — FIXED

`LoggedLLMClient` promoted to `llm/client.py` as the standard production LLM client.
Wraps `LLMClient` and records all calls with full I/O for the traceability chain.
`flush()` returns and clears accumulated records. Pipeline uses it exclusively.

### ~~T21. Context window is global, not per-model override~~ — FIXED

`ModelRouter` now supports `context_window` as int (global) or dict (per-role), matching
the `max_tokens` pattern. Stage overrides can be a string (model tag, inherits role's
context_window) or a dict with `model` and optional `context_window`.

---

## ~~Test Gaps — Phase 3 (from code review)~~ — COVERED (29 tests added)

**Orchestrator:**
- ~~Adjustment pass actually revising remaining steps~~ — added
- ~~Multi-part plans with dependencies / partial success~~ — added
- ~~Patch application failure within orchestrator (retry path)~~ — added
- ~~Meta-plan failure (outer except + finally cleanup)~~ — added
- ~~`steps_completed` accuracy~~ — added
- DB record correctness (orchestrator tests use MagicMock) — deferred (functional, not a gap)
- ~~Missing config raises (budget, stages, max_retries)~~ — added
- ~~`_flush_llm_calls` helper~~ — added
- ~~`run_single_pass` with multiple retries and mixed failure modes~~ — added

**Execute:**
- ~~`execute_plan` with adjustment pass receiving `prior_results` and `test_results`~~ — added
- ~~Budget overflow propagation through `execute_plan` / `execute_implement`~~ — added
- ~~`execute_implement` with `plan` context (orchestrator always passes `plan=part_plan`)~~ — added
- ~~`validate_plan` on `PlanAdjustment.revised_steps`~~ — added
- ~~Prompt construction with empty `ContextPackage.files`~~ — added
- ~~`parse_implement_response` with content containing `</search>` tags (R5)~~ — added

**Patch:**
- ~~`_atomic_write` directly (platform-sensitive, zero direct tests)~~ — added
- ~~Rollback failure path~~ — added
- ~~`apply_edits` exception path (I/O failure during application)~~ — added
- ~~Empty edits list~~ — added
- ~~Path traversal~~ — added
- ~~Non-UTF-8 files~~ — added
- ~~Unicode content in search/replace~~ — added
- ~~TOCTOU: file modified between validate and apply~~ — added

**CLI:**
- ~~`_resolve_budget` and `_resolve_stages` helpers directly~~ — added
- `cra plan` and `cra solve` success paths (only error paths tested) — deferred (requires live LLM)

---

## ~~Test Gaps — Phase 1/2 (from prior review)~~ — COVERED (40 tests added)

**Critical — tests that miss code paths:**
- ~~`enrich_repository()` has zero test coverage~~ — added (2 tests: no-indexed-repo raises, skips-already-enriched)

**Missing public API tests:**
- ~~`get_file_by_id`, `get_symbol_by_id` — no direct test~~ — added (4 tests)
- ~~`search_symbols_by_name` with LIKE-injection characters~~ — added (4 tests: %, _, \)
- ~~`search_files_by_metadata` with `module` parameter / multiple filters~~ — added (2 tests)
- ~~`get_symbol_neighbors` with `kinds` filter~~ — added (2 tests)
- `get_adapter_for_stage` with an active adapter — deferred (Phase 4)
- ~~`get_repo_overview` — `domain_counts` and `most_connected_files` not asserted~~ — added (2 tests)

**Missing error path tests:**
- ~~`ModelRouter.resolve("reasoning")` when reasoning not configured~~ — added
- ~~`enrich_repository()` with no indexed repo~~ — added (mock-based, no live LLM needed)
- ~~`_check_git_available` when git is not on PATH~~ — added (2 tests)
- ~~`extract_git_history` with non-zero returncode~~ — added
- ~~`LLMClient.complete()` with HTTP error / large system prompt triggering R3 rejection~~ — added (6 tests: HTTP 500, connect error, timeout, large system prompt, ModelConfig validation)

**Missing integration tests:**
- ~~CLI `index`, `enrich` help~~ — added (2 tests)
- ~~CLI `init` with existing `.gitignore` containing `.clean_room/`~~ — added (2 tests)
- ~~`ScopeStage.run()` and `PrecisionStage.run()` wiring~~ — covered via R2 default-deny tests

**Missing edge case tests:**
- ~~`_resolve_ts_js_baseurl` entirely untested~~ — added (3 tests: no tsconfig, baseUrl+paths, baseUrl only)
- ~~`expand_scope` with `seed_symbol_ids`~~ — added (3 tests: tier 1 placement, dedup with seed_files, nonexistent symbol)
- ~~`judge_scope` / `classify_symbols` with LLM returning non-list JSON~~ — covered (invalid verdict test)
- ~~`_refilter_files` when LLM returns non-list — fallback~~ — added
- ~~`Assembly with all symbols excluded / multiple detail levels`~~ — added (3 tests: all excluded, multiple files excluded, mixed)

---

## Other (Carried Forward)

### Large-file refactoring exceeds context budget

A single 4000+ line file classified as "primary" consumes the entire 32K context window.
R1 prohibits downgrading to signatures, so assembly can't fit the file plus system prompt,
task description, and supporting context. The agent won't create such files itself, but it
may encounter them in external repos.

**Chunked strategy by refactor type** (task analysis already identifies the type):

- **Structural** (split, move, rename, reorganize): signatures + docstrings + return
  statements are sufficient for planning. Edits execute per-symbol with only the relevant
  slice in context. Fits current architecture.
- **Simplify**: function-by-function, method-by-method, class-by-class. Each pass gets
  only the body it's simplifying. Fits current architecture.
- **Deduplication / extract common pattern**: requires seeing multiple bodies simultaneously
  to recognize shared logic. Function-by-function fails because the point is *comparing*
  them. Needs a different retrieval strategy: deterministic candidate pairing from the symbol
  graph (similar signatures, similar callees/imports, similar line counts, similar AST
  structure) to pre-filter likely pairs, then show their bodies side by side within budget.
  This is a new retrieval stage pattern, not a configuration of existing stages.

### Git workflow integration

The agent needs a git workflow for its own operations — committing changes, creating
branches, handling failed attempts. Not GitHub (no PRs, no issues) — local git only.
Needs: auto-commit after successful validation, branch-per-task isolation, rollback via
git reset on failure (cleaner than file-level rollback), cumulative diff from git diff
instead of string accumulation. The orchestrator currently does file-level patch/rollback
which is fragile; git gives atomic rollback for free.

### ~~Magic Numbers (from prior review)~~ — DONE

Extracted to named constants with config overrides via `[retrieval]` and `[indexer]`
sections. Algorithmic caps (max_deps, max_callees, etc.) remain as defaults but are
overridable per-project. Safety constants (token estimation, batch sizing) stay hardcoded.

### Scope inheritance across orchestrator checkpoints

Currently each sub-task in the orchestrator runs a fully isolated retrieval pipeline.
The meta_plan retrieval discovers relevant files and symbols, but that knowledge is
discarded — each subsequent part_plan, code step, adjustment, and test step re-discovers
from scratch using only its narrow task description and plan_artifact Tier 0 seeds.

**Problem:** Isolated retrieval is wasteful (re-discovers the same files N times) and
occasionally blind (a step's narrow description misses files the broad meta_plan found).
But hard-constraining all steps to the meta_plan's scope prevents discovery of files that
only become relevant after code changes or validation failures.

**Design direction:** Soft inheritance — meta_plan scope provides strong seeds (not
boundaries) for subsequent retrievals. Steps can narrow freely and expand when there's
evidence (mentioned in step description, revealed by cumulative_diff, pointed to by
validation errors). Specific checkpoints to consider:

1. **Meta_plan → part/step retrievals**: inherited scope as Tier 1 seeds
2. **After validation failure**: error patterns + failing file paths as additional seeds
3. **After cumulative_diff growth**: newly-touched files as seeds for subsequent steps
4. **Adjustment pass**: adjustment may identify new files needed for revised steps

Needs live testing before design — run the orchestrator on real tasks and observe where
isolated retrieval makes wrong decisions vs where inheritance would have helped.

### Phase 1 Legacy

**8.1** Parser comment classification differs: Python `^#\s*TODO\b` (anchored) vs TS/JS
`\bTODO\b` (anywhere). Intentional per language design.

**~~8.2~~ FIXED:** Enrichment skip logic now uses `file_path` (stable across curated DB
rebuilds). `enrichment_outputs` schema has `file_path TEXT` column with migration for
existing DBs.
