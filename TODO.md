# TODO

Consolidated findings from full repository review (2026-02-24). Items tagged with a
**Context Curation Rule** reference the numbered rules in CLAUDE.md § Context Curation Rules.

---

## Open Items

### Test Gaps (Section 5)

These are test coverage improvements — not bugs or design issues. The codebase is functionally
correct with 388 passing tests.

**5.1 Critical — tests that can't fail or miss entire code paths**

- `enrich_repository()` has zero test coverage
- `test_tier_3_co_changes` has zero assertions
- `test_keywords_ordered_by_length` is vacuously true

**5.2 Missing public API tests**

- `get_file_by_id`, `get_symbol_by_id` — no direct test
- `search_symbols_by_name` with LIKE-injection characters
- `search_files_by_metadata` with `module` parameter
- `search_files_by_metadata` with multiple simultaneous filters
- `get_symbol_neighbors` with `kinds` filter
- `get_adapter_for_stage` with an active adapter
- `get_repo_overview` — `domain_counts` and `most_connected_files` not asserted

**5.3 Missing error path tests**

- `ModelRouter.resolve("reasoning")` when reasoning not configured
- `_resume_task_from_session` with missing `task_query` in session
- `enrich_repository()` with no indexed repo
- `_check_git_available` when git is not on PATH
- `extract_git_history` with non-zero returncode
- `LLMClient.complete()` with HTTP error response
- `LLMClient.complete()` with large system prompt triggering R3 rejection

**5.4 Missing integration tests**

- CLI `index` command, CLI `enrich` command
- CLI `retrieve` success path
- CLI `init` with existing `.gitignore` containing `.clean_room/`
- `ScopeStage.run()` and `PrecisionStage.run()` wiring
- `run_pipeline` with `refinement_request` set

**5.5 Missing edge case tests**

- `_resolve_ts_js_baseurl` entirely untested
- `expand_scope` with `seed_symbol_ids`
- `judge_scope` with LLM returning non-list JSON
- `classify_symbols` with LLM returning non-list JSON
- `_refilter_files` when LLM returns non-list — fallback
- Assembly with all symbols `excluded`
- File with multiple symbols at different detail levels
- Various parser edge cases (decorators, nested classes, empty source, etc.)

**5.6 Weak assertions** (assertions that could be tightened)

- `pytest.raises(Exception)` should be `sqlite3.IntegrityError`
- Various `>= 1` / `>= 2` assertions that could be exact
- Session state existence-only checks (`is not None`)

**5.7 Test style issues**

- Mock LLM routing via system prompt substring is fragile
- Stage registration imports repeated per test
- `parse_json_response` tests duplicated across modules

### Magic Numbers (Section 6)

Budget-derived or LLM-decided values that are currently hardcoded. Functional but could
be improved for configurability.

| Location | Value | Status |
|---|---|---|
| `scope_stage.py:14-16` | `MAX_DEPS=30, MAX_CO_CHANGES=20, MAX_METADATA=20` | Justified caps; configurable via function params |
| `scope_stage.py:141` | `keywords[:5]` | Ordered by length (R6) |
| `scope_stage.py:170` | `_TOKENS_PER_SCOPE_CANDIDATE=20` | Batch sizing estimate |
| `precision_stage.py:70,74` | `callees[:5], callers[:5]` | Ordered by file inclusion (R6) |
| `precision_stage.py:89` | `_TOKENS_PER_SYMBOL=50` | Batch sizing estimate |
| `budget.py:5` | `SAFETY_MARGIN=0.9` | Documented |
| `enrichment.py:160` | `LIMIT 3 ORDER BY id` | Fixed: now ordered |
| `enrichment.py:165` | `[:200]` char truncation | Enrichment prompt sizing |
| `enrichment.py:173` | `[:100]` line truncation | Source preview |
| `query/api.py:215` | `limit=10` for recent commits | Caller-overridable param |
| `query/api.py:238` | `min_count=2` for co-changes | Caller-overridable param |
| `query/api.py:269` | `LIMIT 10` for connected files | Overview summary |
| `file_scanner.py:12` | `MAX_FILE_SIZE=1_048_576` | 1MB cap |
| `git_extractor.py:13,16` | `CO_CHANGE_MAX_FILES=50, MIN_COUNT=2` | Documented |
| `git_extractor.py:147` | `max_commits=500` | Caller-overridable param |
| `config.py:56` | `reserved_tokens=4096` | Config default |

### Phase 1 Legacy (Section 8)

**8.1** Parser comment classification differs: Python `^#\s*TODO\b` (anchored) vs TS/JS `\bTODO\b`
(anywhere). Both correct for their languages. `_find_enclosing_symbol` signatures also differ
(AST node vs line number) — intentional per language parser design.

**8.2** Enrichment skip logic checks `file_id` which isn't stable across curated DB rebuilds.
Proper fix requires adding `file_path` column to `enrichment_outputs` schema.

---

## Completed (Reference)

<details>
<summary>All fixed items (click to expand)</summary>

### Section 1: Critical Bugs — All Fixed

**1.1** `StageContext.to_dict()` now includes `signature` field.
**1.2** Execute model resolves `"reasoning"` for plan mode, `"coding"` for implement.
**1.3** `error_patterns` stored to session DB.
**1.4** Plan artifact `affected_files` handles both dict and string entries.
**1.5** `ModelRouter` raises on missing `context_window` (no silent default).
**1.6** Git history errors propagate (no silent swallowing).
**1.7** Connections initialized to `None` before try blocks (no UnboundLocalError).
**1.8** Symbol key changed to `(name, start_line)` for collision safety.

### Section 2: Curation Rule Violations — All Fixed

**2.1** `_LoggingLLMClient` records all stage/task analysis LLM calls for raw DB logging.
**2.2** `_extract_supporting` uses docstring data from curated DB (no +10 heuristic).
**2.3** Co-change candidates collected globally, sorted by count, then capped.
**2.4** `resolve_seeds` prefers exact matches, caps at 10 per pattern.
**2.5** Refilter prompt budget-validated; hallucinated path fallback.
**2.6** Task/intent header overhead consumed from budget.
**2.7** Warning logged when post-refilter `can_fit` drops files.
**2.8** `enrich_task_intent` catches `ValueError` and re-raises with context.

### Section 3: Design Issues — All Fixed

**3.1** Removed duplicate `context_window` from `[budget]` config.
**3.2** CLI checks `config is None` before pipeline.
**3.3** `ModelRouter` resolves `max_tokens` per role from config.
**3.4** `ModelConfig.__post_init__` validates `max_tokens < context_window`.
**3.5** `search_symbols_by_name` LIKE pattern escaped.
**3.6** Scope verdict validated against `("relevant", "irrelevant")`.
**3.7** `ScopedFile.__post_init__` validates relevance.
**3.8** Seed dependencies cached (no quadratic re-query).
**3.9** Added `get_symbol_by_id()` to KB; replaced `kb._conn` access.
**3.10** Preflight uses `get_connection` (no raw sqlite3).
**3.11** Task type rule ordering documented.
**3.12** `get_state` returns deserialized Python objects.
**3.13** Token rejection uses chars/3 (more conservative).
**3.14** Refilter validates LLM-returned paths exist.
**3.15** `delete_state` warns on nonexistent key.
**3.16** Windows path URI uses RFC 8089 `file:///C:/...`.

### Section 4: Documentation Drift — All Fixed

**4.1** CLAUDE.md status: Phase 2 complete.
**4.2** `phase2-implementation.md` filled with all WI details.
**4.3** `ModelConfig` spec: `context_window` field added.
**4.4** Session key contracts: `task_query`, `stage_output_{name}`.
**4.5** Curated table count: 12 (not 13).
**4.6** CLI spec gaps documented (Phase 4 items left for Phase 4).
**4.7** Query API spec: `get_file_by_id`, `get_symbol_by_id` added.
**4.8** Renamed `error_signatures` to `error_patterns` everywhere.
**4.9** "Fix before Phase 3" header no longer misleading.

### Section 7: Low Priority — Key Items Fixed

**7.3** `LLMClient.__del__` added.
**7.6** Indexer logs `"partial"` when parse_errors > 0.
**7.7** Git commit parsing logs debug warning on skipped entries.
**7.8** `FileContent.__post_init__` validates detail_level.
**7.9** Dead code `validate_budget()` removed.
**7.10** Dead code `ensure_schema()` removed.
**7.11** `ContextPackage.metadata` typed as `dict[str, Any]`.

### Pre-Phase 3 Fixes (WU1-WU8) — All Fixed

See prior "Completed" items from original review.

</details>
