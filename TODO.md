# TODO

Remaining findings from code reviews and audits. Completed items removed — see git history.

---

## Audit — Fail-Fast Violations (2026-02-26)

### ~~A1. `except OSError: pass` in patch.py temp cleanup~~ — DONE
### ~~A2. Schema migrations catch `OperationalError` with bare `pass`~~ — DONE
### ~~A3. Library scanner `except OSError: return` with no logging~~ — DONE

### ~~A4. Pipeline `except Exception` wraps ~250 lines~~ — DONE
### ~~A5. Archive session `except Exception` swallows all errors~~ — DONE
### ~~A6. `_git_cleanup` single `except Exception` leaves repo in unknown state~~ — DONE

### ~~A7. Detail level falls back to `"type_context"` (violates R2)~~ — DONE
### ~~A8. Hardcoded defaults for coding_style~~ — BY DESIGN (supplementary, not core logic)
### ~~A9. Validator timeout defaults to 120 seconds~~ — DONE

### ~~A10. 3-level config fallback chain for context_window~~ — DONE

### ~~A11. Task type falls back to `"unknown"`~~ — DONE
### ~~A12. Precision JSON fields fall back to empty strings~~ — DONE
### ~~A13. Library scanner config falls back to empty name/path~~ — DONE

---

## Audit — Transparency (2026-02-26)

### ~~A14. File read failures not recorded in assembly_decisions~~ — DONE

---

## Audit — Deduplication (2026-02-26)

### ~~A15. R3 budget validation repeated in 6 places~~ — DONE
### ~~A16. JSON structure validation repeated in 4 places~~ — BY DESIGN (fundamentally different operations)
### ~~A17. File extension constants scattered across 7-8 locations~~ — DONE

---

## P3 — Low

### ~~T72. Python/TS/JS signature extraction falls back to string split (R4)~~ — DONE
### ~~T73. Grammatical error in TEST_IMPLEMENT_SYSTEM prompt~~ — DONE
### ~~T74. `_topological_sort()` gives unhelpful errors on malformed items~~ — DONE

---

## Refactor Review (2026-02-26)

### File Size Reduction / Decomposition

#### R1. Decompose `run_orchestrator()` — PARTIAL

Steps 1-4, 8 done (context dataclass, init/cleanup, diff update, rollback). Steps 5-7
deferred (`_run_plan_stage()`, `_execute_code_step()`, `_execute_test_loop()`). Step 9
partial (init/cleanup done, body reduction blocked on steps 5-7).

#### ~~R2. Consolidate `run_single_pass()` with `run_orchestrator()`~~ — DONE
#### ~~R3. Extract CLI command logic into `commands/` package~~ — DONE
#### ~~R4. Decompose `_do_index()` in indexer/orchestrator.py~~ — DONE
#### ~~R5. Collapse system prompt constants in prompts.py~~ — DONE

### Code Simplification / Deduplication

#### ~~R6. Shared batched judgment runner for retrieval stages~~ — DONE
#### ~~R7. Deduplicate validation logic in patch.py~~ — DONE
#### ~~R8. Extract `db/helpers.py` for shared DB utilities~~ — DONE
#### ~~R9. Use NamedTuple for scope stage candidates~~ — DONE
#### ~~R10. Flatten deep nesting in TS/JS parser~~ — DONE
#### ~~R11. Simplify TS/JS variable symbol extraction~~ — DONE
#### ~~R12. Add `_NON_EMPTY` validation to `_SerializableMixin`~~ — DONE
#### ~~R13. Row converter boilerplate in query/api.py~~ — DONE
#### ~~R14. Scope stage dedup pattern repeated 3 times~~ — DONE
#### ~~R15. Use `__getattr__` delegation in EnvironmentLLMClient~~ — DONE

### Fail-Fast / Transparency Violations

#### ~~R16. Silent exception swallowing in library_scanner.py~~ — DONE
#### ~~R17. Library symbol detail downgrade violates R1~~ — DONE
#### ~~R18. Assembly refilter silent fallback~~ — DONE

### LLM Enhancement Opportunities (Deferred — new LLM calls, separate feature pass)

#### R19. Task-aware budget tie-breaking in context assembly
#### R20. LLM-ranked dependency ordering in scope stage
#### R21. Docstring-based pre-filter in precision stage

### ~~R22. Consistent R2 default-deny handler across stages~~ — DONE

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

- [ ] ~~**2-P2-4: `index_libraries` hardcodes `language="python"`**~~ — DONE

- [ ] **2-P2-6: No CLI test for `cra index-libraries` command**

### Feature 3: Git Workflow Integration

- [ ] ~~**3-P1-3: LIFO rollback error abandons remaining parts**~~ — DONE (DB cleanup before raise)

- [ ] **3-P2-8: No integration test for orchestrator git/LIFO fallback path**

### Feature 4: Pipeline Trace Log — P2

- [ ] ~~**4-P2-1: `model` parameter defaults to empty string**~~ — DONE

- [ ] **4-P2-3: Large traces could consume significant memory**
  - File: `src/clean_room_agent/trace.py:15,22-34`

- [ ] ~~**4-P2-4: `_make_trace_logger` imports Path redundantly**~~ — NOT A BUG — Path is only imported inside the function (deferred conditional), not at module level. No redundancy.

---

## Commit 46b2b5f Code Review (Remaining)

- [ ] **5-P2-6: No integration test for `run_single_pass` git lifecycle**

---

## Commit 4f6a8df Code Review — KB Indexer (2026-02-27)

### ~~S1 (P0): Silent catch-all in indexer loop — tracebacks lost~~ — DONE
### ~~S2 (P1): Raw SQL in indexer bypasses query layer~~ — DONE
### ~~S3 (P1): `_read_file_source` accesses `kb._conn` directly~~ — DONE

### Deferred (P2-P3)
- ~~S4: C standard detection false positives (`restrict`, `inline`)~~ — DONE (word-boundary regex)
- S5: Domain tie-breaking is arbitrary (dict order)
- S6: No minimum threshold for domain assignment
- ~~S7: Short keyword substring matches are noisy (`&`, `or`)~~ — DONE (removed from domain keywords)
- ~~S10: Missing UNIQUE constraint on `ref_sections(source_id, section_path)`~~ — DONE
- S8: Section heading regex may miss OCR-garbled case
- S9: PDF timeout is a magic number
- ~~S11: `_title_to_path` duplicated in markdown_parser.py and html_parser.py~~ — DONE (shared in models.py)
