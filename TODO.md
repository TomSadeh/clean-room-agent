# TODO

Consolidated findings from transparency principle review (2026-02-24). Items reference
the core transparency principle and Context Curation Rules (R1-R6) from CLAUDE.md.

---

## P2 — Medium Priority

### T20. `_LoggingLLMClient` is pipeline-internal, not system-wide

The wrapper is defined inside `pipeline.py` and only used in `run_pipeline()`. Any code
calling `LLMClient.complete()` outside the pipeline bypasses logging. As Phase 3/4 add
more LLM call sites, unlogged calls become more likely.

**Elevated from P3:** Fragmented logging violates traceability completeness. All LLM call
sites currently log correctly, but through different paths — consolidation needed before
Phase 3 adds more call sites.

**Consider:** Making logging intrinsic to `LLMClient.complete()` via callback or
making `LoggedLLMClient` the only way to obtain an `LLMClient` during pipeline execution.

### ~~T21. Context window is global, not per-model override~~ — FIXED

`ModelRouter` now supports `context_window` as int (global) or dict (per-role), matching
the `max_tokens` pattern. Stage overrides can be a string (model tag, inherits role's
context_window) or a dict with `model` and optional `context_window`.

---

## P3 — Low Priority

### T22. `__del__` with `except Exception: pass`

**Location:** `llm/client.py:58-59`

Acceptable for destructors but technically violates "do not silently recover."

### T23. Orphan commits in curated DB

Re-indexing inserts all commits (up to 500) but only creates `file_commits` for files in
`file_id_map`. Commits touching only excluded files have no associations.

**Location:** `indexer/orchestrator.py:286-298`

### T24. TOCTOU: hash and parse may read different file versions

File hash computed during scanning, content read again during parsing. No verification
that bytes match.

**Location:** `indexer/file_scanner.py:145-158`, `indexer/orchestrator.py:153-165`

---

## Existing Items (Carried Forward)

### Test Gaps (from prior review)

These are test coverage improvements — not bugs. The codebase has 464 passing tests.

**Critical — tests that miss code paths:**
- `enrich_repository()` has zero test coverage

**Missing public API tests:**
- `get_file_by_id`, `get_symbol_by_id` — no direct test
- `search_symbols_by_name` with LIKE-injection characters
- `search_files_by_metadata` with `module` parameter / multiple filters
- `get_symbol_neighbors` with `kinds` filter
- `get_adapter_for_stage` with an active adapter
- `get_repo_overview` — `domain_counts` and `most_connected_files` not asserted

**Missing error path tests:**
- `ModelRouter.resolve("reasoning")` when reasoning not configured
- `_resume_task_from_session` with missing `task_query` in session
- `enrich_repository()` with no indexed repo
- `_check_git_available` when git is not on PATH
- `extract_git_history` with non-zero returncode
- `LLMClient.complete()` with HTTP error / large system prompt triggering R3 rejection

**Missing integration tests:**
- CLI `index`, `enrich`, `retrieve` commands
- CLI `init` with existing `.gitignore` containing `.clean_room/`
- `ScopeStage.run()` and `PrecisionStage.run()` wiring
- `run_pipeline` with `refinement_request` set

**Missing edge case tests:**
- `_resolve_ts_js_baseurl` entirely untested
- `expand_scope` with `seed_symbol_ids`
- `judge_scope` / `classify_symbols` with LLM returning non-list JSON
- `_refilter_files` when LLM returns non-list — fallback
- Assembly with all symbols `excluded` / multiple detail levels

### Magic Numbers (from prior review)

Budget-derived or LLM-decided values that are hardcoded. Functional but could be
improved for configurability. See git history for the full table (commit `a803eb2`).

### Phase 1 Legacy

**8.1** Parser comment classification differs: Python `^#\s*TODO\b` (anchored) vs TS/JS
`\bTODO\b` (anywhere). Intentional per language design.

**8.2** Enrichment skip logic checks `file_id` which isn't stable across curated DB
rebuilds. Proper fix requires adding `file_path` to `enrichment_outputs` schema.
