# TODO

Consolidated findings from code reviews. Items reference the core transparency
principle and Context Curation Rules (R1-R6) from CLAUDE.md.

---

## P0 — Critical

### T57. R3 violations: 5 LLM call sites lack budget validation before send

Per R3 ("every LLM prompt must be budget-validated"), these call `llm.complete()` without
pre-validating the assembled prompt against the model's context window:

1. `retrieval/task_analysis.py:171` — `enrich_task_intent()`. Prompt includes
   `repo_file_tree` which can be arbitrarily large. No R3 check.
2. `retrieval/routing.py:71` — `route_stages()`. Prompt is small but no explicit validation.
3. `retrieval/scope_stage.py:217` — `judge_scope()` per-batch. Batch sizing estimates
   candidate count but never validates the actual assembled prompt.
4. `retrieval/precision_stage.py:133` — `classify_symbols()` per-batch. Same issue.
5. `retrieval/similarity_stage.py:176` — `judge_similarity()` per-batch. Same issue.

The transport layer (`LLMClient.complete()`) has a gate, but R3 requires explicit validation
and logging at the call site. Batching calculates batch size from a per-candidate token
estimate, but if that estimate is wrong (e.g., a symbol with an unusually long signature),
the assembled batch exceeds budget and gets silently truncated by Ollama.

Fix: add `estimate_tokens_conservative(prompt) + estimate_tokens_conservative(system)` check
before each `llm.complete()` call. For the batch callers, validate the actual assembled prompt
(not just the estimate) before sending.

### T58. Pipeline symbol decisions logged with wrong stage name

`retrieval/pipeline.py:273-282` — The symbol decision logging loop runs inside the stage
loop for **every** stage, attributing all `context.classified_symbols` to whichever stage
just ran. When routing selects `["precision", "similarity"]`, precision's symbols are logged
twice: once correctly with `stage_name="precision"`, then again with
`stage_name="similarity"`. This creates duplicate entries in `retrieval_decisions` with false
attribution — a traceability violation.

Fix: track logged symbol IDs in a `logged_symbol_ids` set (like `logged_file_ids` for files),
or only log symbols for the specific stage that populated them.

---

## P1 — High

### T59. Orchestrator rollback can crash mid-LIFO, leaving files corrupt

`orchestrator/runner.py:673-678` — `rollback_edits()` raises `RuntimeError` on any file
write failure. If the first rollback in the LIFO sequence fails, the exception propagates
and remaining rollbacks are skipped. Repo is left in a partially-rolled-back state.

Fix: wrap each rollback call in try/except, collect errors, continue rolling back all
remaining patches, then raise a combined error after all attempts.

### T60. Adjustment cycle has no depth limit

`orchestrator/runner.py:433-486` — The adjustment pass splices `revised_steps` into the step
list after every code step. If the LLM keeps generating more steps than it consumes, the list
grows without bound. The config template has `max_adjustment_rounds` (commented out) but the
orchestrator never reads it.

Fix: read `max_adjustment_rounds` from `[orchestrator]` config. Track how many times
adjustment has produced revised_steps. Stop adjustment after the limit and continue with
remaining steps as-is.

### T61. Parser missing type validation on nested list fields from LLM JSON

`execute/parsers.py:30,114,153` — `data["parts"]` and `data["steps"]` are passed directly
to list comprehensions without checking they're actually lists. If the LLM returns
`"parts": "some string"`, the error is `TypeError: 'str' object is not iterable` with no
useful context.

Fix: add `if not isinstance(data.get("parts"), list)` checks before list comprehension,
raising `ValueError` with a descriptive message including the actual type received.

### T75. Validation output flows unbounded into LLM prompts

`orchestrator/validator.py:59-76` → `execute/prompts.py:250-261` — Test, lint, and type-check
subprocess output can be arbitrarily large (test frameworks routinely produce 10KB+ verbose
output). This output flows unfiltered into `<test_failures>` sections of implement prompts
when retrying failed steps.

The R3 `_validate_prompt_budget` at the end of prompt construction catches the overall size,
but at that point it raises `ValueError` — there's no refilter path. A 50KB test output makes
the prompt exceed budget, and the step fails with a budget error instead of the model seeing a
curated summary of what actually broke.

Fix: when validation output exceeds a budget-derived threshold, use an LLM call to summarize
the failures (extract the relevant error messages, stack traces, and failing test names). The
full output stays in raw DB for traceability; only the curated summary enters the prompt.

### T76. Enrichment prompt uses arbitrary caps instead of R3 gate

`llm/enrichment.py:181-197` — Three arbitrary caps in `_build_prompt()`:
1. Docstrings: `LIMIT 3` by insertion order (not relevance), each truncated to `[:200]` chars.
2. Source preview: first `[:100]` lines (imports and boilerplate, not necessarily the important
   code).

The R3 pre-validation gate (line 89-98) already skips files whose enrichment prompt exceeds
the model's context window. The arbitrary caps are redundant for most files and
counterproductive for the few where they bite — the model gets a non-representative slice
of the file for its metadata generation.

Fix: remove the caps. Let `_build_prompt()` include all docstrings (full text) and full source.
The R3 gate handles the rare case where a file is genuinely too large — skip with a warning,
same as now. Most files fit easily, and the enrichment model gets complete information.

---

## P2 — Medium

T62-T71 fixed. Tests in `tests/test_p2_fixes.py`.

### T77. Silent exception swallowing in orchestrator temp file cleanup

`orchestrator/runner.py:~788` — `except Exception: pass` on temp file cleanup. Violates
fail-hard coding style. Cleanup failures are invisible — orphaned temp files accumulate
with no diagnostic trail.

Fix: log the exception with `logger.warning()` before continuing. Cleanup is best-effort,
but failures must be visible.

### T78. Enrichment JSON silently stores None for missing required fields

`llm/enrichment.py:116-140` — Parsed enrichment JSON uses `.get()` for fields like
`purpose`, `module`, `domain`, `concepts`. If the LLM returns incomplete JSON, `None` is
silently stored to the database. The enrichment appears successful but is degraded.

Fix: validate required fields are present before storing. Raise on missing required keys
so the failure is visible and the file can be retried.

### T79. route_stages() accepts bare LLMClient — logging not enforced

`retrieval/routing.py` — `route_stages()` parameter is typed as `LLMClient`. Pipeline
passes `LoggedLLMClient` (so logging works today), but the function signature doesn't
enforce it. A direct caller with a bare client silently loses the audit trail.

Fix: type the parameter as `LoggedLLMClient` or add a runtime isinstance check.

---

## Refactoring (All Complete)

T80-T86 fixed. 929 tests.

### T80. Decompose run_orchestrator() — 590-line monolithic function

`orchestrator/runner.py` — `run_orchestrator()` is ~590 lines with 4+ nesting levels,
mixing 7 distinct phases: meta_plan, part iteration, code steps, testing, validation,
adjustment, and cleanup. The `_get_task_run_id()` pattern (fetchone + None check) repeats
7 times.

Fix: extract phase functions: `_run_meta_plan_phase()`, `_run_part_code_steps()`,
`_run_testing_phase()`, `_run_adjustment_pass()`. Extract `_get_task_run_id()` helper
for the repeated DB lookup pattern.

### T81. Extract dataclass serialization boilerplate

`execute/dataclasses.py` (464 lines) — 37 boilerplate `to_dict()` / `from_dict()` methods
across 10+ dataclasses following identical patterns. Each class manually maps fields to/from
dicts with hand-written validation.

Fix: create a `SerializableMixin` that auto-generates `to_dict()` and `from_dict()` from
`__dataclass_fields__`, with a `_REQUIRED_FIELDS` class variable for validation.

### T82. Extract generic DB insert helper

`db/raw_queries.py` (294 lines) + `db/queries.py` (271 lines) — 25+ insert functions
following identical pattern: generate timestamp, build INSERT SQL, execute, return lastrowid.
`datetime.now(timezone.utc).isoformat()` repeated 15+ times.

Fix: create `_insert_row(conn, table, columns, values, include_timestamp=True)` helper.
Each `insert_*()` becomes a thin wrapper calling the helper.

### T83. Unify retrieval stage LLM batching logic

`retrieval/scope_stage.py`, `precision_stage.py`, `similarity_stage.py` — all three stages
duplicate: token estimation → batch sizing → LLM judgment call → JSON parse → R2
default-deny merge. ~100 lines of duplicated logic across 3 files.

Fix: extract a shared batching utility (function or class) that handles token estimation,
batch splitting, LLM calling, and R2 default-deny. Each stage defines its candidate format,
prompt template, and parse function.

### T84. Extract BaseLanguageParser for shared parser logic

`parsers/python_parser.py` (595 lines) + `parsers/ts_js_parser.py` (410 lines) — duplicated
comment classification regexes, node text extraction, and symbol extraction patterns.
Both classify TODO/FIXME/HACK/NOTE/BUG comments with similar regex sets.

Fix: create `BaseLanguageParser` with shared `_classify_comment()`, `_node_text()`, and
common regex patterns. Language-specific parsers inherit and define their AST walk logic.

### T85. Extract prompt builder for execute prompts

`execute/prompts.py` (326 lines) — 6 hardcoded system prompt constants with repetitive
structure (all start "You are Jane, ..."). `build_plan_prompt()` and
`build_implement_prompt()` share ~50% of logic (task header + context + optional sections +
budget validation).

Fix: data-driven prompt structure + shared `_build_user_prompt()` helper. System prompts
defined as structured data, assembled by a common builder.

### T86. Decompose assemble_context()

`retrieval/context_assembly.py` (439 lines) — `assemble_context()` is 177 lines with 3+
nesting levels mixing file classification, rendering, and budget enforcement.

Fix: extract phases: `_classify_files()`, `_render_files()`, `_enforce_budget()`,
`_build_package()`.

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

### Git workflow integration

The agent needs a git workflow for its own operations — committing changes, creating
branches, handling failed attempts. Not GitHub (no PRs, no issues) — local git only.
Needs: auto-commit after successful validation, branch-per-task isolation, rollback via
git reset on failure (cleaner than file-level rollback), cumulative diff from git diff
instead of string accumulation. The orchestrator currently does file-level patch/rollback
which is fragile; git gives atomic rollback for free.

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

## Commit f26e1c6 Code Review (58 findings)

### Feature 1: Surface Enrichment Metadata in LLM Prompts

#### P1

- [x] **1-P1-1: `_TOKENS_PER_SCOPE_CANDIDATE` too low after metadata additions**
  - File: `src/clean_room_agent/retrieval/scope_stage.py:176`
  - Still set to 50, but metadata suffix (`[purpose=..., domain=..., concepts=...]`) pushes candidates to ~100+ tokens. Batch sizing will over-fill, triggering more `ValueError`s from `validate_judgment_batch`. CLAUDE.md R3 violation (budget validation).
  - Fix: Increase to ~100.

- [x] **1-P1-2: `estimate_framing_tokens` does not account for metadata line**
  - File: `src/clean_room_agent/retrieval/budget.py:27-36`
  - `to_prompt_text` adds `metadata_summary\n` between header and `<code>` tag. Assembly tracks `meta_tokens` separately but the trailing `\n` is not counted. Minor CLAUDE.md R5 violation (framing overhead in budget).
  - Fix: Include metadata line in framing estimate, or count the `\n`.

- [x] **1-P1-3: `search_files_by_metadata` with no filters returns all enriched files**
  - File: `src/clean_room_agent/query/api.py:53-81`
  - If `domain`, `module`, and `concepts` are all None, query returns all files with any metadata — unbounded result set. Current callers always pass `concepts=kw` but the API is public.
  - Fix: Raise `ValueError` if no filter parameters provided, or document the behavior.

- [x] **1-P1-4: `_row_to_file` silent fallback for `file_source`**
  - File: `src/clean_room_agent/query/api.py:360-364`
  - `except (IndexError, KeyError): file_source = "project"` — violates CLAUDE.md fail-fast coding style. If schema migration hasn't run, this hides the problem.
  - Fix: Remove fallback, rely on migration, fail-fast on missing column.

#### P2

- [ ] **1-P2-1: Scope stage metadata only surfaced for non-seeds**
  - File: `src/clean_room_agent/retrieval/scope_stage.py:193-206`
  - Seeds (tier 0/1) skip LLM judgment and never have metadata appended. Metadata is available and used in assembly, but not during scope. Not a bug, just a note.

- [ ] **1-P2-2: `get_file_metadata_batch` IN clause limited by SQLITE_MAX_VARIABLE_NUMBER**
  - File: `src/clean_room_agent/query/api.py:108-128`
  - SQLite default limit is 999 placeholders. Pipeline caps make this unlikely to hit (~100 files max), but worth noting if caps change.

- [x] **1-P2-3: `classify_symbols` passes `file_source` through dict with `.get()` fallback**
  - File: `src/clean_room_agent/retrieval/precision_stage.py:65-76,192`
  - `c.get("file_source", "project")` is a hardcoded default. Since `extract_precision_symbols` always sets it, the fallback is unreachable. Should use `c["file_source"]` for fail-fast.

- [x] **1-P2-4: Docstring summary truncation uses magic number**
  - File: `src/clean_room_agent/retrieval/precision_stage.py:129-131`
  - `first_line[:100]` — hardcoded 100-char truncation. Should be a named constant.

- [x] **1-P2-5: No test for `search_files_by_metadata` with `module` parameter**
  - File: `tests/query/test_file_metadata.py`
  - No direct unit tests for `search_files_by_metadata` with `domain`, `module`, or `concepts` filters. The LIKE escaping path (api.py:71-73) has tricky edge cases.

- [ ] **1-P2-6: No test for token-per-candidate accuracy after metadata additions**
  - No test validating `_TOKENS_PER_SCOPE_CANDIDATE` / `_TOKENS_PER_SYMBOL` are still accurate. A regression test would catch issues like 1-P1-1.

### Feature 2: Library/Dependency Indexing

#### P0

- [x] **2-P0-1: `_auto_resolve` uses string heuristics to parse imports**
  - File: `src/clean_room_agent/indexer/library_scanner.py:82-94`
  - Uses `line.startswith("import ")` — violates CLAUDE.md Rule 4 ("Use parsed structure, not string heuristics"). Breaks on multi-line imports, comments, conditional imports, semicolons.
  - Fix: Use `get_parser("python").parse()` to extract `ExtractedImport` objects.

- [x] **2-P0-2: Single-file module bug walks entire `site-packages/`**
  - File: `src/clean_room_agent/indexer/library_scanner.py:108-112`
  - For single-file modules like `six.py`, sets `pkg_path` to `site-packages/` itself.
  - Fix: Use `Path(spec.origin)` directly as a single file for non-`__init__.py` origins.

- [x] **2-P0-3: `index_libraries` silently swallows OSError on file read**
  - File: `src/clean_room_agent/indexer/orchestrator.py:433-436`
  - `except (OSError, IOError): continue` — silent skip, no logging, no counter.
  - Fix: Log warning with file path and exception, increment error counter.

- [x] **2-P0-4: `library_file_index` parameter is dead code**
  - File: `src/clean_room_agent/extractors/dependencies.py:23`
  - Never passed by any caller. Library dependency fallback path unreachable.
  - Fix: Remove dead parameter and unreachable code path.

#### P1

- [x] **2-P1-1: `_auto_resolve` doesn't exclude `venv/` directories**
  - File: `src/clean_room_agent/indexer/library_scanner.py:73-76`
  - Skip list catches `.venv` (starts with `.`) but not plain `venv/` or `env/`.
  - Fix: Add `"venv"`, `"env"`, `".tox"` to skip set.

- [x] **2-P1-2: No stale library file cleanup**
  - File: `src/clean_room_agent/indexer/orchestrator.py:420-504`
  - Removed library files persist in DB forever.
  - Fix: Compute set difference, delete stale entries.

- [x] **2-P1-3: `find_spec` can execute arbitrary code and has narrow exception catch**
  - File: `src/clean_room_agent/indexer/library_scanner.py:100-103`
  - Exception list misses `ImportError`, `AttributeError`, and custom finder exceptions.
  - Fix: Broaden to `except Exception`.

- [x] **2-P1-4: Changed library files counted as "new"**
  - File: `src/clean_room_agent/indexer/orchestrator.py:440-462`
  - No `files_changed` field in `LibraryIndexResult`.
  - Fix: Add `files_changed` field, track separately.

- [x] **2-P1-5: `_row_to_file` silent fallback for `file_source`** (same as 1-P1-4)
  - File: `src/clean_room_agent/query/api.py:360-364`

- [x] **2-P1-6: Silent `OSError` on `py_file.stat()` in `scan_library`**
  - File: `src/clean_room_agent/indexer/library_scanner.py:149-152`
  - Fix: Log warning with file path.

#### P2

- [x] **2-P2-1: No tests for `library_file_index` fallback in `resolve_dependencies`**
  - File: `tests/extractors/test_dependencies.py`
  - N/A: Dead code removed in 2-P0-4.

- [x] **2-P2-2: `test_resolve_auto_mocked` assertion is weak**
  - File: `tests/indexer/test_library_scanner.py:107-126`

- [ ] **2-P2-3: `_SKIP_DIRS` not configurable**
  - File: `src/clean_room_agent/indexer/library_scanner.py:13`

- [ ] **2-P2-4: `index_libraries` hardcodes `language="python"`**
  - File: `src/clean_room_agent/indexer/orchestrator.py:449`

- [x] **2-P2-5: `_DEFAULT_LIBRARY_MAX_FILE_SIZE` duplicated**
  - File: `src/clean_room_agent/indexer/library_scanner.py:16` and `orchestrator.py:413`

- [ ] **2-P2-6: No CLI test for `cra index-libraries` command**

### Feature 3: Git Workflow Integration

#### P0

- [x] **3-P0-1: Detached HEAD makes `return_to_original_branch` a no-op**
  - File: `src/clean_room_agent/orchestrator/git_ops.py:47-48`
  - `rev-parse --abbrev-ref HEAD` returns `"HEAD"` in detached state.
  - Fix: Detect and raise `RuntimeError`.

- [x] **3-P0-2: Branch name collision on leftover task branches**
  - File: `src/clean_room_agent/orchestrator/git_ops.py:18,55`
  - Uses `task_id[:12]` truncation.
  - Fix: Use full task_id in branch name.

- [x] **3-P0-3: `git_workflow` defaults to `True` via hardcoded fallback**
  - File: `src/clean_room_agent/orchestrator/runner.py:265`
  - Fix: Remove default, validate explicitly.

- [x] **3-P0-4: `create_task_branch()` failure leaves orchestrator inconsistent**
  - File: `src/clean_room_agent/orchestrator/runner.py:267-270`
  - Called outside try block.
  - Fix: Move inside try, guard finalize with `if orch_run_id is not None`.

#### P1

- [ ] **3-P1-1: `rollback_to_checkpoint` + `return_to_original_branch` may fail on untracked files**
  - File: `src/clean_room_agent/orchestrator/runner.py:833-840`

- [x] **3-P1-2: `--allow-empty` commits on no-change checkpoints**
  - File: `src/clean_room_agent/orchestrator/git_ops.py:73`

- [ ] **3-P1-3: LIFO rollback error abandons remaining parts**
  - File: `src/clean_room_agent/orchestrator/runner.py:771-784`

- [x] **3-P1-4: `get_cumulative_diff` returns `""` when `_base_ref is None`**
  - File: `src/clean_room_agent/orchestrator/git_ops.py:87-88`

- [x] **3-P1-5: Diff truncation uses `"\n--- "` string heuristic**
  - File: `src/clean_room_agent/orchestrator/git_ops.py:96-98` and `runner.py:59-61`

- [x] **3-P1-6: Duplicate diff truncation logic**
  - File: `src/clean_room_agent/orchestrator/git_ops.py:93-99` and `runner.py:53-62`

#### P2

- [x] **3-P2-1: `task_id` not validated before use in branch name**
  - File: `src/clean_room_agent/orchestrator/git_ops.py:18`

- [x] **3-P2-2: `rollback_part` runs `clean -fd` which could delete `.clean_room/` data**
  - File: `src/clean_room_agent/orchestrator/git_ops.py:107`

- [x] **3-P2-3: `return_to_original_branch` logs warning instead of raising when `_original_branch is None`**
  - File: `src/clean_room_agent/orchestrator/git_ops.py:120-122`

- [x] **3-P2-4: No test for `return_to_original_branch` when `_original_branch is None`**
  - File: `tests/orchestrator/test_git_ops.py`

- [x] **3-P2-5: No test for `get_cumulative_diff` when `_base_ref is None`**
  - File: `tests/orchestrator/test_git_ops.py`

- [x] **3-P2-6: No test for `rollback_to_checkpoint` with explicit `commit_sha` argument**
  - File: `tests/orchestrator/test_git_ops.py:167-190`

- [x] **3-P2-7: No test for branch name truncation edge case (task_id < 12 chars)**
  - File: `tests/orchestrator/test_git_ops.py`

- [ ] **3-P2-8: No integration test for orchestrator git/LIFO fallback path**

### Feature 4: Pipeline Trace Log

#### P0

- [x] **4-P0-1: Markdown injection breaks `<details>` sections**
  - File: `src/clean_room_agent/trace.py:100-128`
  - Content containing `</details>` prematurely closes collapsible sections.
  - Fix: Wrap content in dynamically-lengthed fenced code blocks within `<details>`.

- [x] **4-P0-2: Response content breaks code fences**
  - File: `src/clean_room_agent/trace.py:131-135`
  - LLM responses containing triple backticks break the ``` fence.
  - Fix: Use dynamically-lengthed fences.

- [x] **4-P0-3: Trace task_id mismatch with orchestrator task_id**
  - File: `src/clean_room_agent/cli.py:420-423`
  - CLI generates a separate UUID; orchestrator generates its own.
  - Fix: Add `update_task_id()` method to TraceLogger; call from orchestrator.

#### P1

- [x] **4-P1-1: `system` field None vs "" inconsistency**
  - File: `src/clean_room_agent/trace.py:28`

- [x] **4-P1-2: `elapsed_ms` defaults to 0 masking missing timing data**
  - File: `src/clean_room_agent/trace.py:32`

- [x] **4-P1-3: Refinement merge path skips trace logging**
  - File: `src/clean_room_agent/retrieval/pipeline.py:127-152`

- [x] **4-P1-4: `finalize()` can be called multiple times**
  - File: `src/clean_room_agent/trace.py:36-45`

- [x] **4-P1-5: No test for markdown content injection**
  - File: `tests/test_trace.py`

- [x] **4-P1-6: No test for None system prompt**
  - File: `tests/test_trace.py`

#### P2

- [ ] **4-P2-1: `model` parameter defaults to empty string**
  - File: `src/clean_room_agent/trace.py:18`

- [x] **4-P2-2: No timestamp in trace header**
  - File: `src/clean_room_agent/trace.py:72-80`

- [ ] **4-P2-3: Large traces could consume significant memory**
  - File: `src/clean_room_agent/trace.py:15,22-34`

- [ ] **4-P2-4: `_make_trace_logger` imports Path redundantly**
  - File: `src/clean_room_agent/cli.py:230`

- [x] **4-P2-5: Ordering assertion in test is fragile**
  - File: `tests/test_trace.py:179-182`
