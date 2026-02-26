# TODO

Remaining findings from code reviews and audits. Completed items removed — see git history.

---

## Audit — Fail-Fast Violations (2026-02-26)

### ~~A1. `except OSError: pass` in patch.py temp cleanup~~ — DONE
### ~~A2. Schema migrations catch `OperationalError` with bare `pass`~~ — DONE
### ~~A3. Library scanner `except OSError: return` with no logging~~ — DONE

### A4. Pipeline `except Exception` wraps ~250 lines
- File: `retrieval/pipeline.py:372-380`
- Single catch-all wraps the entire pipeline body (task analysis through assembly).
  Recoverable stage errors and fatal DB corruption handled identically.
- Fix: catch `(ValueError, RuntimeError)` separately from unexpected exceptions.

### A5. Archive session `except Exception` swallows all errors
- File: `orchestrator/runner.py:192-199`
- 5-line try block (read, insert, commit, unlink) with one `except Exception`.
  DB insert failure means file is never deleted. Code logs warning and continues.
- Fix: separate catches for `OSError` (file read/delete) vs `sqlite3.Error` (DB insert).

### A6. `_git_cleanup` single `except Exception` leaves repo in unknown state
- File: `orchestrator/runner.py:222-235`
- Entire rollback+merge+branch-delete wrapped in one catch. If rollback fails, branch
  delete is skipped. Caller can't tell whether cleanup succeeded.
- Fix: per-operation catches. Rollback/return-to-original failures should raise; branch
  delete failure is best-effort warning.

### ~~A7. Detail level falls back to `"type_context"` (violates R2)~~ — DONE
### ~~A8. Hardcoded defaults for coding_style~~ — BY DESIGN (supplementary, not core logic)
### ~~A9. Validator timeout defaults to 120 seconds~~ — DONE

### A10. 3-level config fallback chain for context_window
- File: `commands/retrieve.py:40-47`
- Falls through CLI → `[budget].context_window` → `[models].context_window`.
  The `[models]` fallback is undocumented. Comment says "single source of truth"
  but code treats it as fallback.
- Fix: explicit required config with clear error message listing all valid sources.

### ~~A11. Task type falls back to `"unknown"`~~ — DONE
### ~~A12. Precision JSON fields fall back to empty strings~~ — DONE
### ~~A13. Library scanner config falls back to empty name/path~~ — DONE

---

## Audit — Transparency (2026-02-26)

### ~~A14. File read failures not recorded in assembly_decisions~~ — DONE

---

## Audit — Deduplication (2026-02-26)

### A15. R3 budget validation repeated in 3 places
- `execute/prompts.py:155-166` — `_validate_prompt_budget()`
- `retrieval/routing.py:85-90` — manual estimate+check
- `retrieval/batch_judgment.py:38-55` — `validate_judgment_batch()`
- All implement: estimate tokens → check against available → raise ValueError.
- Fix: single `validate_prompt_budget(prompt, system, stage_name, model_config)` in `budget.py`.

### A16. JSON structure validation repeated in 4 places
- `execute/dataclasses.py:52-79` — `_SerializableMixin.from_dict()`
- `retrieval/routing.py:47-56` — validates dict, list, string items
- `retrieval/batch_judgment.py:128-135` — `isinstance` checks
- `execute/parsers.py:30-31` — `isinstance(data, dict)` check
- Fix: shared `validate_json_structure()` helper.

### A17. File extension constants scattered across 7-8 locations
- `retrieval/task_analysis.py:15` — `KNOWN_EXTENSIONS`
- `execute/prompts.py:357-363` — `_lang_from_path()` with `.endswith()` checks
- `execute/documentation.py:158` — `if file_path.endswith(ext)`
- `parsers/ts_js_parser.py:42` — `if file_path.endswith(".tsx")`
- `extractors/dependencies.py:148,182,198` — multiple extension checks
- Fix: centralize in `constants.py` with `LANGUAGE_EXTENSIONS` dict + `get_language_from_path()`.

---

## P3 — Low

### T72. Python/TS/JS signature extraction falls back to string split (R4)

Three places use `split("\n")[0]` instead of AST structure for signature extraction:

1. `parsers/python_parser.py:230` — module-level variable assignments.
2. `parsers/python_parser.py:254-256` — fallback when body node not found.
3. `parsers/ts_js_parser.py:157,166` — variable/arrow function symbols.

These violate R4 ("use parsed structure, not string heuristics"). Multi-line assignments or
arrow functions get truncated to their first line. Low impact since these are signatures
(not source bodies) and the truncation only loses parameter continuation lines.

### T73. Grammatical error in TEST_IMPLEMENT_SYSTEM prompt

`execute/prompts.py:121-122` — "For new test files, use an empty <search></search> is not
allowed —" is grammatically broken. Should be: "For new test files, empty <search></search>
is not allowed; instead, use a search string that matches the insertion point context."

### T74. `_topological_sort()` gives unhelpful errors on malformed items

`orchestrator/runner.py:127-155` — If items lack expected attributes, the lambda accessors
raise `AttributeError` with no context about which item was malformed.

---

## Refactor Review (2026-02-26)

### File Size Reduction / Decomposition

#### R1. Decompose `run_orchestrator()` (705 lines) in runner.py

The pipeline+LLM+flush+record pattern repeats 7 times (meta_plan, part_plan, code step,
adjustment, test_plan, test_step, documentation). Extract `_run_plan_stage()` helper that
bundles the 30-40 line sequence: `run_pipeline()` -> LoggedLLMClient context -> `execute_X()`
-> `_flush_llm_calls()` -> `_get_task_run_id()` -> `insert_orchestrator_pass()` -> `set_state()`
-> `raw_conn.commit()`.

Also extract:
- `_execute_code_step()` — step execution with retries (lines 424-542, 119 lines)
- `_execute_test_loop()` — test step loop (lines 729-836)
- `_update_cumulative_diff()` — git-vs-LIFO diff update (repeated 4 times)

Estimated savings: 200-300 lines from main function, flatten 7-level nesting to 3-4.

#### R2. Consolidate `run_single_pass()` with `run_orchestrator()`

51% code duplication between the two entry points. Init (config/router/DB setup), finally
block (archive/cleanup), and git init are near-identical. Extract:
- `_init_orchestrator()` -> returns (router, config, connections, git)
- `_cleanup_orchestrator()` -> finally block logic

Alternative: make `run_single_pass()` call `run_orchestrator()` with a synthetic 1-part plan.

Estimated savings: ~150 lines.

#### R3. Extract CLI command logic into `commands/` package

`cli.py` (443 lines) has 50-100 line implementations inline for `retrieve`, `plan`, `solve`.
Extract to `commands/retrieve.py`, `commands/plan.py`, `commands/solve.py`. CLI layer becomes
UI glue only.

Estimated savings: ~243 lines from cli.py.

#### R4. Decompose `_do_index()` (269 lines) in indexer/orchestrator.py

Handles 10 concerns with 6 nesting levels. Extract:
- `_register_repo_and_scan()` — repo lookup, file scan, diff
- `_parse_and_insert_symbols()` — parse + symbol/docstring/comment insertion (lines 163-252)
- `_resolve_and_insert_dependencies()` — deps + co-changes + references (lines 256-305)
- `_extract_and_index_git_history()` — git extraction + insertion (lines 307-339)

Symbol insertion logic (lines 195-232) is duplicated with `index_libraries()` (lines 478-503).
Extract shared `_insert_file_symbols()`.

Estimated savings: ~80-100 lines.

#### R5. Collapse system prompt constants in prompts.py

Seven system prompt strings defined as individual variables (lines 18-124) then re-collected
into `SYSTEM_PROMPTS` dict (lines 148-156). Define directly in the dict — individual constants
are never referenced elsewhere.

Estimated savings: ~79 lines.

### Code Simplification / Deduplication

#### R6. Shared batched judgment runner for retrieval stages

Scope, precision, and similarity stages all implement identical protocol: batch by context
window -> call LLM -> parse JSON -> apply R2 default-deny. Extract `run_batched_judgment()`
utility that handles batching + JSON parse + R2 validation. Stages provide only a candidate
formatter and result mapper callback.

- `scope_stage.py:179-267` (judge_scope)
- `precision_stage.py:105-215` (classify_symbols)
- `similarity_stage.py:144-216` (judge_similarity)

Estimated savings: ~80-100 lines across three files.

#### R7. Deduplicate validation logic in patch.py

`validate_edits()` (lines 31-68) and `apply_edits()` (lines 71-130) contain near-identical
validation loops (path traversal check, file existence, search string matching, simulation).
Extract `_validate_and_simulate_edits()` called by both.

Estimated savings: ~47 lines.

#### R8. Extract `db/helpers.py` for shared DB utilities

`_now()` and `_insert_row()` are identical in both `raw_queries.py` (lines 7-24) and
`queries.py` (lines 7-24). Extract to `db/helpers.py`. Also extract `_build_update_clause()`
for the verbose UPDATE builder in `raw_queries.py:200-241`.

Estimated savings: ~60 lines.

#### R9. Use NamedTuple for scope stage candidates

`scope_stage.py` uses `tuple[int, str, str, str, int]` with magic index `[4]` throughout
(lines 101-167). Replace with `NamedTuple` (`CandidateWithScore`) for readability.

#### R10. Flatten deep nesting in TS/JS parser

`_parse_es_import()` (lines 262-298) has 9-level nesting. `_parse_commonjs_require()`
(lines 300-358) has 8-level nesting. Extract `_extract_import_names()` and
`_extract_commonjs_names()` helpers to flatten to 4 levels.

#### R11. Simplify TS/JS variable symbol extraction

`ts_js_parser.py:131-154` — two near-identical `ExtractedSymbol(...)` blocks for
function vs. variable. Merge to single block with `kind = "function" if value_node else
"variable"` and early continue.

#### R12. Add `_NON_EMPTY` validation to `_SerializableMixin`

`execute/dataclasses.py` — `__post_init__` non-empty validation is copy-pasted across
5+ dataclasses. Add `_NON_EMPTY: tuple[str, ...] = ()` class var to `_SerializableMixin`
with auto-validation in `__post_init__`.

Estimated savings: ~43 lines.

#### R13. Row converter boilerplate in query/api.py

10 static `_row_to_*` methods (lines 362-431) follow identical patterns. Replace with a
factory `_make_row_converter(dataclass_type)` that introspects the dataclass.

Estimated savings: ~61 lines.

#### R14. Scope stage dedup pattern repeated 3 times

`scope_stage.py` — three identical dedup+sort blocks (lines 111-118, 135-142, 160-167)
for deps, co-changes, and metadata candidates. Extract `dedup_by_score()` helper.

Estimated savings: ~30 lines.

#### R15. Use `__getattr__` delegation in EnvironmentLLMClient

`client.py:235-264` — re-implements `__enter__`, `__exit__`, `close()`, `config` property
that already exist in `LoggedLLMClient`. Use `__getattr__` for automatic delegation, keep
only `complete()` override.

Estimated savings: ~24 lines.

### Fail-Fast / Transparency Violations

#### R16. Silent exception swallowing in library_scanner.py

Two bare `except Exception: continue` blocks with **no logging**:
- `library_scanner.py:87-88` — parse failure in `_auto_resolve()`
- `library_scanner.py:101-102` — `find_spec()` failure in `_auto_resolve()`

Fix: catch specific exceptions (`OSError`, `ValueError`, `AttributeError`, `ImportError`),
add `logger.debug()` before continue.

#### R17. Library symbol detail downgrade violates R1 (no degradation)

`precision_stage.py:194-200` — downgrades library symbols from `primary` to `type_context`
after LLM classification. This is a post-hoc band-aid, not a re-filter. Violates R1:
"Fix decisions, not content."

Fix: add deterministic pre-filter **before** LLM judgment to exclude library symbols from
primary classification candidates entirely.

#### R18. Assembly refilter silent fallback

`context_assembly.py:389-390` — when LLM refilter returns only invalid paths, falls back to
priority-based drop with a warning. Should raise `ValueError` if LLM returns garbage, not
silently switch strategies.

### LLM Enhancement Opportunities

#### R19. Task-aware budget tie-breaking in context assembly

`context_assembly.py` — when budget exceeded, currently drops files by detail level tier
(type_context first, supporting second, primary last). An LLM call could instead decide
which files matter most for *this specific task*, providing task-aware dropping instead of
blanket tier-based dropping.

#### R20. LLM-ranked dependency ordering in scope stage

`scope_stage.py:100-106` — dependencies sorted by "connections to seed files" (simple count).
A lightweight LLM call could rank deps by relevance to the task's intent summary — cheaper
than full scope judgment but richer than counting.

#### R21. Docstring-based pre-filter in precision stage

`precision_stage.py:147-157` — symbols enter classification with signatures but no docstring
context. A pre-filter LLM call on docstring summaries could reduce batch sizes by excluding
clearly irrelevant symbols before the full classification prompt.

#### R22. Consistent R2 default-deny handler across stages

Each stage reimplements R2 omission handling slightly differently (scope -> "irrelevant",
precision -> "excluded", similarity -> skip). Extract unified `handle_omitted_candidate()`
to ensure consistent logging and policy.

---

## Deferred Test Gaps

- `cra plan` and `cra solve` success paths — requires live LLM
- `get_adapter_for_stage` with an active adapter — Phase 4
- DB record correctness in orchestrator — tests use MagicMock (functional, not a gap)

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

1. **Meta_plan -> part/step retrievals**: inherited scope as Tier 1 seeds
2. **After validation failure**: error patterns + failing file paths as additional seeds
3. **After cumulative_diff growth**: newly-touched files as seeds for subsequent steps
4. **Adjustment pass**: adjustment may identify new files needed for revised steps

Needs live testing before design — run the orchestrator on real tasks and observe where
isolated retrieval makes wrong decisions vs where inheritance would have helped.

### Surface curated DB enrichment metadata in LLM prompts

The enrichment pipeline (`cra enrich`) produces per-file metadata (purpose, module, domain,
concepts, public_api_surface, complexity_notes) and stores it in `file_metadata`. Docstrings
are also indexed. None of this reaches any LLM prompt today:

- **Scope stage** queries `file_metadata.concepts` to *find* files, but discards the metadata
  content — only file paths enter the judgment prompt. The model doesn't know *why* a Tier 4
  file was included.
- **Precision stage** shows symbol names, signatures, and call graph edges, but no docstrings
  or purpose summaries.
- **Execute stage** (`to_prompt_text()`) renders source code only. No file-level metadata.
- **Docstrings** are read in context assembly for line-count boundary calculations only — the
  text is never rendered.
- **Inline comments**, **commit history** are indexed but completely unused.

This is a large item — blocked until the DB is actually populated on real repos. Once
`cra enrich` has run, consider:
1. Adding purpose/domain/concepts to scope judgment prompts (helps relevance decisions).
2. Adding docstring summaries to precision classification (helps detail-level decisions).
3. Adding file-level purpose to execute prompts (helps the model understand unfamiliar code).
4. Surfacing inline TODOs/FIXMEs when task_type is bug_fix.

### Phase 1 Legacy

**8.1** Parser comment classification differs: Python `^#\s*TODO\b` (anchored) vs TS/JS
`\bTODO\b` (anywhere). Intentional per language design.

---

## Commit f26e1c6 Code Review (Remaining)

### Feature 1: Surface Enrichment Metadata — P2

- [ ] **1-P2-1: Scope stage metadata only surfaced for non-seeds**
  - File: `src/clean_room_agent/retrieval/scope_stage.py:193-206`
  - Seeds (tier 0/1) skip LLM judgment and never have metadata appended. Not a bug, just a note.

- [ ] **1-P2-2: `get_file_metadata_batch` IN clause limited by SQLITE_MAX_VARIABLE_NUMBER**
  - File: `src/clean_room_agent/query/api.py:108-128`
  - SQLite default limit is 999 placeholders. Pipeline caps make this unlikely (~100 files max).

- [ ] **1-P2-6: No test for token-per-candidate accuracy after metadata additions**

### Feature 2: Library/Dependency Indexing — P2

- [ ] **2-P2-3: `_SKIP_DIRS` not configurable**
  - File: `src/clean_room_agent/indexer/library_scanner.py:13`

- [ ] **2-P2-4: `index_libraries` hardcodes `language="python"`**
  - File: `src/clean_room_agent/indexer/orchestrator.py:449`

- [ ] **2-P2-6: No CLI test for `cra index-libraries` command**

### Feature 3: Git Workflow Integration

- [ ] **3-P1-3: LIFO rollback error abandons remaining parts**
  - File: `src/clean_room_agent/orchestrator/runner.py:771-784`

- [ ] **3-P2-8: No integration test for orchestrator git/LIFO fallback path**

### Feature 4: Pipeline Trace Log — P2

- [ ] **4-P2-1: `model` parameter defaults to empty string**
  - File: `src/clean_room_agent/trace.py:18`

- [ ] **4-P2-3: Large traces could consume significant memory**
  - File: `src/clean_room_agent/trace.py:15,22-34`

- [ ] **4-P2-4: `_make_trace_logger` imports Path redundantly**
  - File: `src/clean_room_agent/cli.py:230`

---

## Commit 46b2b5f Code Review (Remaining)

- [ ] **5-P2-6: No integration test for `run_single_pass` git lifecycle**
