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
