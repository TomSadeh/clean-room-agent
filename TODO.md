# TODO

Remaining findings from code reviews. Completed items removed — see git history for audit.

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

#### P1

- [ ] **3-P1-1: `rollback_to_checkpoint` + `return_to_original_branch` may fail on untracked files**
  - File: `src/clean_room_agent/orchestrator/runner.py:833-840`

- [ ] **3-P1-3: LIFO rollback error abandons remaining parts**
  - File: `src/clean_room_agent/orchestrator/runner.py:771-784`

#### P2

- [ ] **3-P2-8: No integration test for orchestrator git/LIFO fallback path**

### Feature 4: Pipeline Trace Log — P2

- [ ] **4-P2-1: `model` parameter defaults to empty string**
  - File: `src/clean_room_agent/trace.py:18`

- [ ] **4-P2-3: Large traces could consume significant memory**
  - File: `src/clean_room_agent/trace.py:15,22-34`

- [ ] **4-P2-4: `_make_trace_logger` imports Path redundantly**
  - File: `src/clean_room_agent/cli.py:230`
