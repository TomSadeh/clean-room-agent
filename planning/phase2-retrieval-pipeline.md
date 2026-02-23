# Phase 2: Retrieval Pipeline

## Context

This is the intelligence layer of the Clean Room Agent. Given a task description and a populated knowledge base (Phase 1), it produces a curated context package — the minimum information needed to complete the task, with zero noise. This implements Stages 1 and 2 of the three-prompt strategy.

Phase 1 gives us a structured, queryable knowledge base. Phase 2 decides **what to pull from it** for a specific task. Phase 3 (agent harness) will consume the context package to actually generate code.

The gate for Phase 2: given a task and a repo, does the pipeline produce a context package that contains all relevant files/symbols and excludes irrelevant ones? Measured by precision and recall against ground truth.

---

## Architecture

```
Task Description
       │
       ▼
┌─────────────────┐
│  Task Analysis   │   Extract keywords, detect task type, find entry points
└────────┬────────┘
         │ TaskQuery
         ▼
┌─────────────────┐
│  Stage 1: Scope  │   Score every file against the task, rank, select top N
│  (Deterministic)  │   Expand via dependency graph + co-change affinity
└────────┬────────┘
         │ ScopeResult (50-100 files, ranked)
         ▼
┌─────────────────┐
│ Stage 2: Precision│  For each scoped file, extract only what's relevant
│  (Deterministic   │  Symbol-level filtering, signature vs full body decisions
│   + optional LLM) │  Token budget enforcement
└────────┬────────┘
         │ PrecisionResult (symbols, types, tests, docs)
         ▼
┌─────────────────┐
│ Context Assembly  │  Format into structured prompt sections
│                   │  Provenance metadata, token counts
└────────┬────────┘
         │ ContextPackage
         ▼
    Phase 3 consumes
```

The entire pipeline is **deterministic by default**. LLM assist is optional and isolated — used only when keyword extraction from the task is ambiguous (e.g., "fix the auth thing"). The pipeline must produce useful results with no LLM calls at all.

---

## Project Structure (additions to Phase 1)

```
src/
  clean_room/
    retrieval/
      __init__.py
      pipeline.py               # Top-level pipeline orchestrator
      task_analysis.py           # Parse task description into structured query
      scoring/
        __init__.py
        signals.py              # Individual scoring functions
        combiner.py             # Weighted combination + ranking
      scope.py                  # Stage 1: file-level scoping
      precision.py              # Stage 2: symbol-level extraction
      context_assembly.py       # Build formatted context package
      budget.py                 # Token budget allocation + enforcement
      dataclasses.py            # All Phase 2 data structures
tests/
  fixtures/
    sample_task_descriptions.py  # Test task descriptions with expected results
  test_task_analysis.py
  test_scoring_signals.py
  test_scope.py
  test_precision.py
  test_budget.py
  test_context_assembly.py
  test_pipeline.py
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scoring approach | Weighted signal combination | Interpretable, tunable, no training data needed. Each signal is independently testable. |
| Token counting | `tiktoken` with cl100k_base | Fast, accurate enough for budget enforcement. Model-agnostic fallback: 4 chars = 1 token. |
| Keyword extraction | Deterministic heuristics first | Regex for identifiers, file paths, error patterns. No NLP library dependency. |
| LLM assist | Optional, isolated, local-only | Same Ollama setup as Phase 1 enrichment. Only for task analysis when heuristics fail. Never in the scoring/ranking hot path. |
| Context format | Structured dataclass → renderable | Internal representation is typed Python; rendering to markdown/prompt text is a separate step. Keeps the pipeline testable without string matching. |
| Budget strategy | Priority-based eviction | Most relevant items survive; least relevant get trimmed first. Graceful degradation, not hard cutoff. |

---

## Data Structures

```python
@dataclass
class TaskQuery:
    """Structured representation of a task for retrieval."""
    raw_description: str
    task_type: str                    # "bug_fix", "feature", "refactor", "test", "investigation"
    keywords: list[str]              # Extracted identifiers, terms
    file_hints: list[str]            # Explicitly mentioned file paths/patterns
    symbol_hints: list[str]          # Explicitly mentioned function/class names
    error_patterns: list[str]        # Stack traces, error messages
    concepts: list[str]              # Higher-level concepts ("authentication", "caching")

@dataclass
class FileScore:
    """A file's relevance score with breakdown."""
    file_id: int
    path: str
    total_score: float
    signal_breakdown: dict[str, float]   # signal_name -> contribution
    seed: bool                           # True if directly mentioned in task

@dataclass
class ScopeResult:
    """Output of Stage 1."""
    task_query: TaskQuery
    scored_files: list[FileScore]        # Ranked by total_score, descending
    seed_files: list[int]                # File IDs of direct mentions
    expanded_via_deps: list[int]         # File IDs added by dependency expansion
    expanded_via_cochange: list[int]     # File IDs added by co-change affinity

@dataclass
class SymbolContext:
    """A symbol selected for inclusion in the context package."""
    symbol_id: int
    file_id: int
    name: str
    kind: str
    signature: str
    body: str | None                     # Full source — included only for high-relevance symbols
    docstring: str | None
    relevance: str                       # "primary", "supporting", "type_context"

@dataclass
class FileContext:
    """Extracted context for one file."""
    file_id: int
    path: str
    language: str
    relevance_rank: int
    symbols: list[SymbolContext]
    rationale_comments: list[str]
    test_assertions: list[str] | None    # Only for test files

@dataclass
class ContextPackage:
    """The final curated context for Stage 3."""
    task_query: TaskQuery
    files: list[FileContext]             # Ordered by relevance
    dependency_edges: list[tuple[str, str]]  # (source_path, target_path) among included files
    token_count: int
    budget: int
    provenance: dict                     # Scoring metadata for debugging/evaluation
```

---

## Implementation Steps

### Step 1: Data Structures + Task Analysis

**Delivers**: Parse a task description into a structured `TaskQuery`. Deterministic keyword extraction with optional LLM fallback.

**Files**:
- `src/clean_room/retrieval/dataclasses.py` — All dataclasses above
- `src/clean_room/retrieval/task_analysis.py` — `analyze_task(description, ...) -> TaskQuery`
- `tests/fixtures/sample_task_descriptions.py`
- `tests/test_task_analysis.py`

**Deterministic extraction**:
1. **File hints**: Regex for path-like patterns (`src/foo/bar.py`, `components/Auth.tsx`, `*.test.js`). Also detect bare filenames (`auth.py`, `index.ts`).
2. **Symbol hints**: CamelCase and snake_case identifiers that look like function/class names. Heuristic: multi-word identifiers not in common English vocabulary.
3. **Error patterns**: Lines that look like stack traces (` File "...", line N`), error class names (`ValueError`, `TypeError`, `ENOENT`), quoted error messages.
4. **Keywords**: All remaining significant terms after stripping stop words. Preserve technical terms (split CamelCase into constituent words too for matching).
5. **Task type**: Keyword classification — "fix"/"bug"/"broken"/"error" → `bug_fix`, "add"/"implement"/"create"/"new" → `feature`, "refactor"/"clean"/"reorganize" → `refactor`, "test"/"spec"/"coverage" → `test`. Default: `investigation`.
6. **Concepts**: Higher-level terms extracted by matching keywords against the concept vocabulary already in `file_metadata.concepts` across the KB. This grounds concept extraction to terms the KB actually knows about, not arbitrary NLP.

**Optional LLM fallback**: If the heuristic extractor produces fewer than 3 keywords and no file/symbol hints (i.e., the task is vague like "make the app faster"), offer an LLM-assisted extraction via Ollama. Same local-only pattern as Phase 1 enrichment. The LLM receives the task description and returns structured JSON with the same fields. Disabled by default — flag `--llm-assist` on CLI.

**Verify**: Test against a suite of 10+ task descriptions covering all types. Assert extracted fields match expected values.

---

### Step 2: Scoring Signals

**Delivers**: Individual scoring functions that each produce a `float` score for a file given a `TaskQuery`. Pure functions, independently testable.

**Files**:
- `src/clean_room/retrieval/scoring/signals.py`
- `tests/test_scoring_signals.py`

**Signals** (each is a function `(kb: KnowledgeBase, repo_id: int, file_id: int, query: TaskQuery) -> float`):

1. **`score_path_match()`** — Score 0-1. Fuzzy match of task keywords against file path components. Exact directory/filename match scores highest. Partial component match (e.g., keyword "auth" matches path `src/auth/handler.py`) scores lower.

2. **`score_symbol_match()`** — Score 0-1. Match task keywords and symbol_hints against symbol names in the file. Exact name match = 1.0, substring match = 0.5, CamelCase constituent match = 0.3 (e.g., keyword "auth" matches `AuthHandler`).

3. **`score_dependency_proximity()`** — Score 0-1. How close is the file to seed files in the import graph? Direct import of/by a seed file = 1.0, two hops = 0.5, three hops = 0.25. Uses `KnowledgeBase.get_dependency_subgraph()` from Phase 1. Zero if no seed files identified.

4. **`score_cochange_affinity()`** — Score 0-1. Normalized co-change count between this file and any seed file. `co_change_count / max_co_change_count` across all candidates. Zero if no seed files or no co-change data.

5. **`score_recency()`** — Score 0-1. How recently was the file modified? Most recent commit timestamp normalized against the repo's commit range. Weighted higher for `bug_fix` task type (bugs are usually in recently changed code).

6. **`score_domain_match()`** — Score 0-1. Match task concepts against the file's `domain` and `module` from `file_metadata`. Requires Phase 1 LLM enrichment to be populated. Graceful zero if no metadata.

7. **`score_concept_overlap()`** — Score 0-1. Jaccard-like overlap between task concepts and the file's `concepts` array from `file_metadata`. Same enrichment dependency as above.

8. **`score_structural_centrality()`** — Score 0-1. Number of files that depend on this file (in-degree in the dependency graph), normalized. High-centrality files (e.g., `types.ts`, `utils.py`, `config.ts`) are more likely to provide needed type context, less likely to be the primary edit target. Used as a supporting signal, not a primary one.

Each signal returns 0.0 if its data source is unavailable (no enrichment, no git history, etc.). Signals degrade gracefully — the pipeline works with a partially populated KB.

**Verify**: Unit tests with a small in-memory KB. Each signal tested in isolation with known inputs/outputs.

---

### Step 3: Signal Combination + File Ranking (Stage 1: Scope)

**Delivers**: Combine signals into a single score per file. Rank. Select top-N. Expand via dependency graph.

**Files**:
- `src/clean_room/retrieval/scoring/combiner.py` — `combine_scores()`, `rank_files()`
- `src/clean_room/retrieval/scope.py` — `scope_files(kb, repo_id, query, ...) -> ScopeResult`
- `tests/test_scope.py`

**Default signal weights** (tunable via config):

| Signal | Default Weight | Rationale |
|--------|---------------|-----------|
| `path_match` | 0.20 | Strong signal — file paths are highly informative |
| `symbol_match` | 0.25 | Strongest — if the task names a symbol, its file is relevant |
| `dependency_proximity` | 0.15 | Good for finding supporting context |
| `cochange_affinity` | 0.10 | Empirical signal, sometimes noisy |
| `recency` | 0.05 | Weak default, boosted to 0.15 for bug_fix tasks |
| `domain_match` | 0.10 | Useful when enrichment data exists |
| `concept_overlap` | 0.10 | Useful when enrichment data exists |
| `structural_centrality` | 0.05 | Tie-breaker, not a primary signal |

Weights are normalized to sum to 1.0 after adjustment. If a signal returns 0.0 for all files (data source unavailable), its weight is redistributed proportionally to the remaining signals.

**Scoring procedure**:
1. Identify **seed files**: files whose paths or symbols are directly mentioned in the task. These get a score of 1.0 regardless of signal calculation — they are always included.
2. For every non-seed file in the KB, compute all 8 signal scores.
3. Combine via weighted sum → `total_score`.
4. Rank descending. Take top-N (default 75, configurable `--scope-size`).
5. **Dependency expansion**: For each seed file, include its direct import targets and sources (1-hop) even if they didn't make the top-N. These are marked `expanded_via_deps`.
6. **Co-change expansion**: For each seed file, include files with co-change count ≥ 3 even if they didn't make the top-N. Marked `expanded_via_cochange`.
7. Deduplicate. Final scope typically 50-120 files depending on expansion.

**Task-type weight adjustments**:
- `bug_fix`: Boost `recency` to 0.15, reduce `structural_centrality` to 0.0 (bugs are in leaf code, not shared utilities)
- `refactor`: Boost `structural_centrality` to 0.15 (refactors often touch central files)
- `test`: Boost `symbol_match` to 0.30 (test tasks usually name the thing being tested)

**Verify**: Integration test with a populated KB. Given a task like "fix the login validation in auth.py", assert that `auth.py` is in seed files, its imports are in scope, and unrelated files (e.g., `README.md`, `scripts/deploy.sh`) are excluded.

---

### Step 4: Precision Extraction (Stage 2)

**Delivers**: For each scoped file, determine which symbols to include and at what level of detail (full body vs. signature only).

**Files**:
- `src/clean_room/retrieval/precision.py` — `extract_precision(kb, scope_result, query, ...) -> list[FileContext]`
- `tests/test_precision.py`

**Symbol relevance tiers**:

| Tier | Criteria | Inclusion |
|------|----------|-----------|
| **Primary** | Symbol name matches task keywords/hints, or symbol is in a seed file and its docstring/comments reference task keywords | Full source body + docstring + rationale comments |
| **Supporting** | Symbol is in a scoped file, used by a primary symbol (called, imported, inherited), or is a type/interface used in a primary symbol's signature | Signature only + docstring summary (first line) |
| **Type context** | Interface, type alias, enum, or class used transitively by primary/supporting symbols | Signature/definition only |
| **Excluded** | Everything else in the scoped file | Not included |

**Classification algorithm**:
1. Start with seed files. Mark all symbols whose names match `symbol_hints` or whose docstrings/comments contain task keywords as **primary**.
2. If no primary symbols found in seed files, promote all symbols in seed files with matching kind (e.g., for `bug_fix`, promote functions over classes; for `feature`, promote classes).
3. Walk outward: for each primary symbol, find symbols it references (via dependency graph, same-file calls). Mark these **supporting**.
4. For each primary and supporting symbol, find type definitions they reference. Mark these **type context**.
5. Everything else: **excluded**.

**Test file handling**: Files matching `*test*`, `*spec*`, `__tests__/*` patterns get special treatment:
- Test functions whose names contain task keywords → **primary** (include full body — tests are executable documentation)
- Assertion lines extracted as `test_assertions` on the `FileContext`
- Test files are never included just because they import a scoped file — they must independently match the task

**Rationale comment extraction**: For each included symbol, pull rationale comments (from Phase 1's `is_rationale` flag) and include them. These are high-signal context that explains *why* code exists, not *what* it does.

**Verify**: Given a scoped result with 50 files, assert that primary symbols come from seed files, supporting symbols are reachable from primaries, and excluded symbols don't leak into the output.

---

### Step 5: Token Budget Management

**Delivers**: Enforce a token budget on the context package. Priority-based eviction ensures the most relevant content survives.

**Files**:
- `src/clean_room/retrieval/budget.py` — `enforce_budget(files, budget, ...) -> list[FileContext]`
- `tests/test_budget.py`

**Token counting**:
- Primary: `tiktoken` with `cl100k_base` encoding (works for most models).
- Fallback: `len(text) / 4` when tiktoken unavailable or for non-standard models.
- Count is computed per `SymbolContext` (signature + body + docstring) and per `FileContext` (sum of symbols + comments + assertions).

**Budget allocation** (default 32,768 tokens, configurable `--budget`):

| Section | Default Allocation | Priority |
|---------|-------------------|----------|
| Task description | 512 | 0 (always included) |
| Primary symbols | 60% of remaining | 1 |
| Test assertions | 10% of remaining | 2 |
| Supporting symbols (signatures) | 15% of remaining | 3 |
| Type context | 10% of remaining | 4 |
| Rationale comments | 5% of remaining | 5 |

**Eviction procedure**:
1. Compute token count for every item.
2. If total ≤ budget, include everything.
3. If over budget, evict items bottom-up by priority tier (rationale comments first, then type context, then supporting symbols, etc.).
4. Within a tier, evict lowest-scored items first (using `FileScore.total_score` from Stage 1).
5. If still over budget after evicting all of a tier, demote remaining items: primary symbol bodies → signature-only (saves significant tokens).
6. Never evict the task description or seed file primary symbols — these are the floor.

**Body-to-signature demotion**: The biggest token saver. A 50-line function body might be 200 tokens; its signature is 15. When budget is tight, demote the lowest-ranked primary symbols to signature-only. This preserves breadth (more files represented) at the cost of depth (less source code).

**Verify**: Test with a context package that exceeds budget. Assert eviction removes the right items in the right order. Assert the floor items survive even at very small budgets.

---

### Step 6: Context Assembly

**Delivers**: Format the precision-extracted, budget-enforced content into a structured context package ready for Stage 3 consumption.

**Files**:
- `src/clean_room/retrieval/context_assembly.py` — `assemble_context(files, query, ...) -> ContextPackage`
- `tests/test_context_assembly.py`

**Assembly rules**:
1. **Ordering**: Files ordered by relevance rank (from Stage 1 scoring). Within a file, symbols ordered: primary first, then supporting, then type context.
2. **Dependency edges**: Only include edges between files that are both in the final context. This gives Stage 3 the import graph without referencing files it can't see.
3. **Deduplication**: If the same type/interface is referenced by multiple files, include it once in the file where it's defined.

**Rendering** (the `ContextPackage` is a data structure — rendering is a separate concern):
- `render_markdown(package) -> str` — Human-readable, useful for debugging and evaluation.
- `render_prompt(package, template) -> str` — Model-ready prompt text. Template is a string with `{task}`, `{context}`, `{types}`, `{tests}` placeholders. Default template provided, user can override.

**Markdown render format**:
```markdown
## Task
{task_description}

## Primary Context
### src/auth/handler.py (rank #1)
```python
# Authentication request handler
def validate_login(request: LoginRequest) -> AuthResult:
    """Validate credentials and return auth token.

    Args:
        request: Login request with username and password
    Returns:
        AuthResult with token on success, error on failure
    """
    # Workaround for legacy LDAP integration — remove after Q2 migration
    if request.source == "ldap":
        return _validate_ldap(request)
    ...full body...
```
**Depends on**: `src/auth/types.py`, `src/auth/ldap.py`

### src/auth/types.py (rank #3, type context)
```python
@dataclass
class LoginRequest:
    username: str
    password: str
    source: str = "default"

@dataclass
class AuthResult:
    success: bool
    token: str | None
    error: str | None
```

## Test Expectations
### tests/test_auth.py (rank #5)
- `test_valid_login_returns_token`: asserts `result.success is True` and `result.token is not None`
- `test_invalid_password_returns_error`: asserts `result.success is False` and `"invalid" in result.error`

## Dependency Map
auth/handler.py → auth/types.py
auth/handler.py → auth/ldap.py
tests/test_auth.py → auth/handler.py
```

**Provenance metadata** (included in `ContextPackage.provenance`, not rendered to prompt):
- Per-file: scoring breakdown, which expansion added it
- Per-symbol: relevance tier, why included
- Budget: original total, post-eviction total, what was evicted
- Useful for evaluation and debugging retrieval quality

**Verify**: Given a `PrecisionResult`, assert assembly produces valid markdown, respects ordering, deduplicates types, and includes only edges between present files.

---

### Step 7: Pipeline Orchestrator + CLI

**Delivers**: `cra retrieve "task description" --repo /path` runs the full retrieval pipeline and outputs the context package.

**Files**:
- `src/clean_room/retrieval/pipeline.py` — `retrieve(kb, repo_id, task_description, ...) -> ContextPackage`
- Update `src/clean_room/cli.py` — Add `retrieve` command
- `tests/test_pipeline.py`

**CLI interface**:
```bash
# Basic usage — outputs rendered markdown to stdout
cra retrieve "fix the login validation bug" --repo /path/to/repo

# With options
cra retrieve "fix the login validation bug" \
  --repo /path/to/repo \
  --budget 16384 \
  --scope-size 50 \
  --format prompt \
  --template ./my-template.txt \
  --llm-assist \
  --model qwen2.5-coder:7b \
  --verbose \
  --output context.md

# JSON output for programmatic consumption
cra retrieve "fix the login validation bug" \
  --repo /path/to/repo \
  --format json
```

**Pipeline sequence**:
1. Open KB, verify repo is indexed (error if not — point user to `cra index`)
2. Analyze task → `TaskQuery`
3. Scope files → `ScopeResult` (with verbose logging of signal scores)
4. Extract precision context → `list[FileContext]`
5. Enforce token budget → trimmed `list[FileContext]`
6. Assemble → `ContextPackage`
7. Render in requested format
8. Output to stdout or file

**Verbose mode** (`-v`): Prints Stage 1 scoring table (top 20 files with signal breakdowns), Stage 2 symbol counts per tier, budget summary (included/evicted token counts). Critical for tuning signal weights.

**Error handling**:
- Repo not indexed → clear error message with `cra index` suggestion
- KB partially populated (no enrichment) → proceed with available signals, warn about missing metadata
- Zero files scored above threshold → fall back to returning top-10 most central files with a warning

**Depends on**: Steps 1-6 (everything in Phase 2), Phase 1 complete

---

### Step 8: Evaluation Harness

**Delivers**: Tooling to measure retrieval quality. Prepares for SWE-ContextBench evaluation.

**Files**:
- `src/clean_room/retrieval/eval.py` — Evaluation metrics and runner
- `tests/test_pipeline.py` (extended with evaluation cases)

**Metrics**:
- **File-level recall**: Of the files a human would need to complete this task, what fraction did Stage 1 scope include?
- **File-level precision**: Of the files Stage 1 scoped, what fraction are actually relevant?
- **Symbol-level recall**: Of the symbols actually needed, what fraction did Stage 2 include?
- **Symbol-level precision**: Of the symbols Stage 2 included, what fraction are actually needed?
- **Token efficiency**: `relevant_tokens / total_tokens` in the final context package. The thesis metric — we want this near 1.0.
- **Budget utilization**: `total_tokens / budget`. Ideally 0.8-1.0 — using most of the budget without waste.

**Evaluation cases**: Hand-crafted test cases for repos we control:
- Task description + expected files + expected symbols → run pipeline → measure precision/recall
- Start with 5-10 cases against this repo and a small open-source Python project
- Later: automate against SWE-ContextBench ground truth (separate effort)

**`cra evaluate` CLI command**:
```bash
# Run evaluation suite against a test case file
cra evaluate --cases eval_cases.json --repo /path/to/repo

# Output: precision/recall/token-efficiency per case + aggregate
```

**Evaluation case format** (JSON):
```json
{
  "task": "fix the login validation bug when LDAP source is used",
  "expected_files": ["src/auth/handler.py", "src/auth/ldap.py", "tests/test_auth.py"],
  "expected_symbols": ["validate_login", "_validate_ldap", "test_ldap_login"],
  "repo_path": "/path/to/repo"
}
```

**Depends on**: Step 7 (pipeline must be working end-to-end)

---

## Step Dependency Graph

```
Step 1 (Data Structures + Task Analysis)
  ├──► Step 2 (Scoring Signals)
  │      │
  │      └──► Step 3 (Signal Combination + Scope)
  │                 │
  │                 └──► Step 4 (Precision Extraction)
  │                        │
  │                        └──► Step 5 (Token Budget)
  │                               │
  │                               └──► Step 6 (Context Assembly)
  │                                      │
  │                                      └──► Step 7 (Pipeline + CLI)
  │                                             │
  │                                             └──► Step 8 (Evaluation)
  │
  [All steps depend on Phase 1 being complete]
```

Phase 2 is a linear pipeline — each step feeds the next. Unlike Phase 1, there are no parallel tracks. Build sequentially, test each stage independently.

---

## Verification (Phase 2 Gate)

After all 8 steps, run this end-to-end test:

```bash
# Prerequisite: repo is indexed
cra index /path/to/repo -v

# Basic retrieval
cra retrieve "fix the login validation bug" --repo /path/to/repo -v

# Expected output:
# - Task analysis shows extracted keywords, file hints, task type
# - Scope shows 50-100 ranked files with signal breakdowns
# - Precision shows symbol counts per tier (primary/supporting/type/excluded)
# - Budget shows token allocation and any evictions
# - Final context is coherent markdown with relevant code

# JSON output for inspection
cra retrieve "fix the login validation bug" \
  --repo /path/to/repo --format json > context.json

python -c "
import json
ctx = json.load(open('context.json'))
print(f'Files: {len(ctx[\"files\"])}')
print(f'Tokens: {ctx[\"token_count\"]} / {ctx[\"budget\"]}')
print(f'Primary symbols: {sum(1 for f in ctx[\"files\"] for s in f[\"symbols\"] if s[\"relevance\"] == \"primary\")}')
"

# Evaluation
cra evaluate --cases eval_cases.json --repo /path/to/repo
# Expected: file-level recall > 0.8, precision > 0.5, token efficiency > 0.6
```

**Gate criteria**:
1. Pipeline produces context for any task description without crashing
2. Seed files (directly mentioned) always appear in scope
3. Dependency expansion includes direct imports of seed files
4. Primary symbols are from seed files and match task keywords
5. Token budget is respected — output never exceeds `--budget`
6. Graceful degradation: works with no LLM enrichment (just AST + git data)
7. Evaluation harness runs and reports meaningful metrics
8. On at least one real repo, file-level recall > 0.8 for hand-crafted test cases

---

## Tuning Notes (for after implementation)

The signal weights in Step 3 are starting points. After the pipeline is working end-to-end:
1. Run evaluation cases, examine precision/recall per signal
2. Use verbose mode to identify which signals help vs. add noise
3. Adjust weights empirically — this is fast because it's just config, no code change
4. Consider per-task-type weight profiles if one set doesn't fit all

The key signals will likely be `symbol_match` and `dependency_proximity`. Path matching is a cheap approximation. Co-change and enrichment metadata are bonus signals that improve results when available but shouldn't be required.
