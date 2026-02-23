# Phase 1: Knowledge Base + Indexer

## Context

This is the foundation layer of the Clean Room Agent. It creates all three database schemas (curated, raw, session) and indexes a codebase into the curated DB with structured metadata - AST-extracted symbols, dependency graphs, docstrings, comments, git history, and LLM-generated file summaries. No embeddings, no vector search. Everything is structured and queryable with deterministic SQL.

Phase 1 also establishes the connection factory and raw DB logging infrastructure. Indexing run metadata is logged to raw DB. Session DB schema is created but not populated until Phase 2/3.

Phase 2 (retrieval pipeline) reads from the curated DB. The gate for Phase 1: can we index a real repo, query it meaningfully by metadata, and verify all three DB schemas and the connection factory work correctly?

---

## Project Structure

```
clean-room-agent/
  pyproject.toml
  src/
    clean_room/
      __init__.py
      __main__.py
      cli.py                        # Click CLI
      db/
        __init__.py
        schema.py                   # DDL for all three DBs: create_curated_schema(), create_raw_schema(), create_session_schema()
        connection.py               # Connection factory: get_connection(role, task_id=None), WAL mode
        queries.py                  # Parameterized insert/query helpers for curated DB
        raw_queries.py              # Insert helpers for raw DB (index runs, retrieval decisions, attempts)
        session_queries.py          # Insert/query helpers for session DB (retrieval state, working context)
      indexer/
        __init__.py
        orchestrator.py             # Top-level indexing coordinator
        file_scanner.py             # Walk repo, hash files, detect language
        incremental.py              # Content-hash diffing
      parsers/
        __init__.py
        base.py                     # LanguageParser protocol + dataclasses
        python_parser.py
        typescript_parser.py
        javascript_parser.py
        _js_common.py               # Shared TS/JS utilities (JSDoc, etc.)
        registry.py                 # Language -> parser dispatch
      extractors/
        __init__.py
        dependencies.py             # Import resolution to file paths
        docstrings.py               # Format detection + structured parsing
        comments.py                 # Classification (rationale, todo, etc.)
      git/
        __init__.py
        history.py                  # Git log parsing
        cochange.py                 # Co-change analysis
      llm/
        __init__.py
        metadata.py                 # Ollama-based file metadata generation
      query/
        __init__.py
        api.py                      # Public query interface for Phase 2
  tests/
    conftest.py
    fixtures/
      sample_python.py
      sample_typescript.ts
      sample_javascript.js
    test_db.py
    test_file_scanner.py
    test_python_parser.py
    test_typescript_parser.py
    test_javascript_parser.py
    test_dependencies.py
    test_git.py
    test_cochange.py
    test_llm_metadata.py
    test_query_api.py
    test_orchestrator.py
    test_cli.py
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Build backend | Hatchling | Modern, PyPA-endorsed, zero-config for pure Python |
| Layout | `src/` layout | Prevents import shadowing, industry consensus |
| Python floor | 3.11+ | tree-sitter requires >=3.10, 3.11 adds tomllib |
| CLI | Click | Mature, battle-tested |
| Tree-sitter | Individual packages (`tree-sitter-python`, `-typescript`, `-javascript`) | Minimal deps, latest grammar versions, only 3 languages needed |
| DB | Raw SQL + helper functions | No ORM overhead, full control over queries |
| LLM client | `httpx` to local Ollama | 100% local inference, no data leaves the machine. Model is configurable - use whatever's loaded in Ollama |
| Linter | Ruff | Replaces flake8+isort+black in one tool |

---

## MVP Boundary Checklist

- `Must`: Store Python symbol-level reference edges (`symbol_references`) during indexing.
- `Must`: Expose Python symbol neighbor traversal in Query API (`get_symbol_neighbors`).
- `Must`: Keep file-level dependency graph for Python/TS/JS.
- `Must`: Keep indexing functional without any LLM dependency (`cra index` remains deterministic).
- `Must Not`: Require TS/JS symbol-level call/reference edges for MVP completion.
- `Post-MVP`: Add TS/JS symbol-edge extraction and extend precision traversal to consume it.

---

## Implementation Steps

### Step 1: Project Skeleton + Three-Database Layer

**Delivers**: Working Python package, `cra` CLI command (help only), all three SQLite schemas (curated, raw, session), connection factory, basic CRUD helpers for each DB.

**Files**:
- `pyproject.toml` - hatchling build, dependencies, `[project.scripts] cra = "clean_room.cli:main"`
- `src/clean_room/__init__.py`, `__main__.py`
- `src/clean_room/cli.py` - Click group with stub `index` command
- `src/clean_room/db/connection.py` - Connection factory: `get_connection(role, task_id=None)` where role is `"curated"`, `"raw"`, or `"session"`. WAL mode, foreign keys, `sqlite3.Row` factory. Manages DB file paths under `.clean_room/` (curated.sqlite, raw.sqlite, sessions/session_<task_id>.sqlite).
- `src/clean_room/db/schema.py` - Full DDL for all three DBs: `create_curated_schema()`, `create_raw_schema()`, `create_session_schema()`
- `src/clean_room/db/queries.py` - Curated DB helpers: `upsert_repo`, `upsert_file`, `get_file_hash`, `insert_symbol`, `insert_docstring`, `insert_inline_comment`, `insert_dependency`, `insert_symbol_reference`, `insert_commit`, `insert_file_commit`, `upsert_co_change`, `upsert_file_metadata`, `delete_file_data` (cascade)
- `src/clean_room/db/raw_queries.py` - Raw DB helpers: `insert_index_run`, `insert_retrieval_decision`, `insert_task_run`, `insert_run_attempt`, `insert_validation_result`, `insert_session_archive`
- `src/clean_room/db/session_queries.py` - Session DB helpers: `set_retrieval_state`, `get_retrieval_state`, `set_working_context`, `get_working_context`, `set_scratch_note`, `get_scratch_notes`
- `tests/conftest.py`, `tests/test_db.py`

**Curated DB schema highlights** (existing tables, unchanged):
- `INTEGER PRIMARY KEY` for all IDs (SQLite rowid alias)
- `content_hash` as TEXT (hex SHA-256) on `files`
- `co_changes` composite PK `(file_a_id, file_b_id)` with CHECK `file_a_id < file_b_id`
- `symbol_references` for symbol-level edges: `caller_symbol_id`, `callee_symbol_id`, `reference_kind`, `confidence`
- JSON columns (`concepts`, `parsed_fields`, `files_involved`) as TEXT
- Indexes on: `files(repo_id, path)`, `symbols(file_id)`, `symbols(name)`, `symbol_references(caller_symbol_id)`, `symbol_references(callee_symbol_id)`, `dependencies(source_file_id)`, `dependencies(target_file_id)`, `commits(repo_id, hash)`, `file_metadata(domain)`, `file_metadata(module)`

**Raw DB schema highlights** (new):
- `index_runs` - timestamp, repo_path, files_scanned, files_changed, duration_ms, status
- `retrieval_decisions` - task_id, stage, file_id, score, included, reason, timestamp
- `task_runs` - task_id, mode, repo_path, model, success, total_tokens, total_latency_ms, final_diff, timestamp
- `run_attempts` - task_run_id, attempt, prompt_tokens, completion_tokens, latency_ms, raw_response, patch_applied, timestamp
- `validation_results` - attempt_id, success, test_output, lint_output, type_check_output, failing_tests (JSON)
- `session_archives` - task_id, session_blob (full session DB content), archived_at

**Session DB schema highlights** (new):
- `retrieval_state` - key-value store for retrieval pipeline state (stage, scores, decisions)
- `working_context` - staged context fragments being assembled during retrieval/solve
- `scratch_notes` - freeform per-task notes (error classifications, retry context)

**Verify**: `pip install -e ".[dev]" && cra --help && pytest tests/test_db.py`

---

### Step 2: File Scanner + Incremental Diffing

**Delivers**: Walk a repo, detect languages, compute content hashes, diff against previous index run.

**Files**:
- `src/clean_room/indexer/file_scanner.py` - `scan_repo(repo_path) -> Iterator[FileInfo]`
- `src/clean_room/indexer/incremental.py` - `diff_files(conn, repo_id, files) -> IncrementalDiff`
- `tests/test_file_scanner.py`

**Key details**:
- Language detection by extension: `.py` -> python, `.ts`/`.tsx` -> typescript, `.js`/`.jsx`/`.mjs`/`.cjs` -> javascript
- Skip: `.git/`, `node_modules/`, `__pycache__/`, `.venv/`, `dist/`, `build/`, `*.pyc`, `*.min.js`, `*.map`, `*.lock`
- Max file size: 1MB (configurable), skip with warning
- `IncrementalDiff` has: `new_files`, `changed_files`, `unchanged_files`, `deleted_paths`

**Depends on**: Step 1

---

### Step 3: Parser Abstraction + Python Parser

**Delivers**: The `LanguageParser` protocol and complete Python implementation. This establishes the pattern for all parsers.

**Files**:
- `src/clean_room/parsers/base.py` - Protocol + dataclasses: `ExtractedSymbol`, `ExtractedDocstring`, `ExtractedComment`, `ExtractedImport`, `ParseResult`
- `src/clean_room/parsers/python_parser.py` - Full Python parser
- `src/clean_room/extractors/docstrings.py` - Format detection (Google/NumPy/Sphinx/plain) + structured field parsing
- `src/clean_room/extractors/comments.py` - Comment classification heuristics
- `src/clean_room/parsers/registry.py` - Language -> parser dispatch
- `tests/fixtures/sample_python.py`, `tests/test_python_parser.py`

**Python parser extracts**:
- **Symbols**: `function_definition`, `class_definition`, `decorated_definition`, type-annotated module-level assignments. Signature = source text of the def/class line.
- **Docstrings**: First `expression_statement > string` in function/class/module body (anchored with `.` in tree-sitter query). Format detected, structured fields parsed for recognized formats.
- **Comments**: Every `comment` node. Classified by keyword: TODO, FIXME, HACK, NOTE, bug refs (`#123`), rationale patterns ("because", "workaround for", "so that"). `is_rationale` flag. Associated with innermost enclosing symbol.
- **Imports**: `import_statement`, `import_from_statement`. Preserves relative import dots.
- **MVP symbol references (Python-only)**: intra-file call/reference edges (caller symbol -> candidate callee symbol) captured for Phase 2 precision traversal.

**Key tree-sitter queries** (py-tree-sitter 0.25.x API):
```python
# Symbols
"(function_definition name: (identifier) @name) @def"
"(class_definition name: (identifier) @name) @def"

# Docstrings (anchored to first statement in body)
"(function_definition body: (block . (expression_statement (string) @doc)))"
"(class_definition body: (block . (expression_statement (string) @doc)))"
"(module . (expression_statement (string) @doc))"

# Comments
"(comment) @comment"

# Imports
"(import_statement) @import"
"(import_from_statement) @import"
```

**Depends on**: Step 1 (dataclass definitions), no DB interaction yet

---

### Step 4: TypeScript + JavaScript Parsers

**Delivers**: Complete TS and JS parsers following the same protocol.

**Files**:
- `src/clean_room/parsers/typescript_parser.py`
- `src/clean_room/parsers/javascript_parser.py`
- `src/clean_room/parsers/_js_common.py` - Shared JSDoc parsing, JS/TS comment handling
- `tests/fixtures/sample_typescript.ts`, `tests/fixtures/sample_javascript.js`
- `tests/test_typescript_parser.py`, `tests/test_javascript_parser.py`

**TypeScript additions beyond Python**:
- Symbol kinds: `interface_declaration`, `type_alias_declaration`, `enum_declaration`, `method_definition`, `arrow_function` (when const-assigned)
- `export_statement` wrapping declarations
- JSDoc (`/** */`) as docstrings - associated with the next sibling declaration node
- TSX handled by loading `language_tsx()` for `.tsx` files

**JavaScript differences**:
- No interfaces, type aliases, enums
- CommonJS: `const foo = require('./bar')` detected via call_expression query
- Both ES module imports and `require()` calls extracted

**MVP scope note**: TS/JS do not produce symbol-level call/reference edges in Phase 1. They rely on file-level dependency + symbol-name signals in Phase 2. Symbol-edge extraction for TS/JS is a post-MVP enhancement.

**Depends on**: Step 3 (protocol, shared extractors)

---

### Step 5: Import Resolution + Dependency Graph

**Delivers**: Resolve raw import strings to file paths within the repo. Populate `dependencies` table.

**Files**:
- `src/clean_room/extractors/dependencies.py`
- `tests/test_dependencies.py`

**Python resolution**:
1. Detect package roots: directories with `__init__.py`, `src/` layout detection
2. Absolute imports: convert dots to path separators, search for `.py` or `/__init__.py`
3. Relative imports: count leading dots, resolve relative to importing file's directory
4. External packages (stdlib, third-party): discarded - intra-repo only

**TypeScript/JavaScript resolution**:
1. Relative imports (`./foo`, `../bar`): try extensions in order `.ts`, `.tsx`, `.js`, `.jsx`, `.mjs`, `.cjs`, then `index.*`
2. Simple `tsconfig.json` `baseUrl` + `paths` mapping (best-effort, no `extends` chains)
3. Non-relative, non-mapped imports: assumed external, discarded

**Dependency kinds**: `"import"` (standard), `"type_ref"` (TypeScript `import type`). `"call"` and `"inheritance"` defined in schema but not populated in Phase 1.

Uses a pre-built file index (`dict[str, int]` mapping relative paths to file IDs) to avoid filesystem lookups during resolution.

**Depends on**: Steps 3-4 (for `ExtractedImport` data)

---

### Step 6: Git Metadata + Co-change Analysis

**Delivers**: Commit history, file-commit associations, and co-change pairs.

**Files**:
- `src/clean_room/git/history.py` - `extract_git_history(repo_path, since=None) -> GitHistory`
- `src/clean_room/git/cochange.py` - `compute_co_changes(commits, file_index) -> list[CoChange]`
- `tests/test_git.py`, `tests/test_cochange.py`

**Git history**: Uses `git log --pretty=format:... --name-status --numstat` via subprocess. Bounded by `--max-commits` (default 5000). `--since` flag for incremental extraction.

**Co-change algorithm**: For each commit, take file pairs changed together, increment co-change count. Skip commits touching >50 files (bulk operations pollute the signal).

**Depends on**: Step 1 (DB). Independent of Steps 3-5 (can be built in parallel).

---

### Step 7: Indexing Orchestrator + CLI Wiring

**Delivers**: `cra index /path/to/repo` produces a fully populated SQLite database.

**Files**:
- `src/clean_room/indexer/orchestrator.py` - `index_repository(repo_path, ...) -> IndexingResult`
- Update `src/clean_room/cli.py` - Wire `index` command
- `tests/test_orchestrator.py`, `tests/test_cli.py`

**Pipeline sequence** (annotated with DB targets):
1. Open/create curated + raw DBs, apply schemas for all three DB types (session schema defined, no per-task session DB created during indexing)
2. Register repo (detect git remote URL) -> **curated**
3. Scan files -> (in-memory)
4. Compute incremental diff -> (in-memory, reads **curated**)
5. Delete data for removed/changed files -> **curated**
6. Upsert file records -> **curated**
7. Parse each new/changed file -> insert symbols, docstrings, comments, and Python symbol-reference edges -> **curated**
8. Resolve all imports -> insert dependencies -> **curated**
9. Extract git history -> insert commits, file-commits -> **curated**
10. Compute co-changes -> upsert pairs -> **curated**
11. Log indexing run metadata (files scanned, changed, duration, status) -> **raw**
12. Report results

**Error handling**: Individual file parse failures caught and logged, indexing continues. Each file's data committed atomically.

**Depends on**: Steps 1-6 (everything)

---

### Step 8: LLM Metadata + Query Interface

**Delivers**: `cra enrich` command for LLM metadata generation (separate from indexing, optional), and the `KnowledgeBase` query class that Phase 2 will consume.

**Files**:
- `src/clean_room/llm/metadata.py` - Ollama `/api/generate` via httpx, structured prompt, JSON output parsing
- `src/clean_room/query/api.py` - `KnowledgeBase` class with query methods
- Update `src/clean_room/cli.py` - Add `enrich` command
- `tests/test_llm_metadata.py`, `tests/test_query_api.py`

**`cra enrich`** is separate from `cra index` because:
- Indexing must work without a running LLM
- LLM generation is seconds/file vs milliseconds/file for parsing
- Users may re-enrich with different models without re-indexing

**All LLM calls are local** - Ollama on localhost, no data leaves the machine. Model is a CLI flag (`--model`), no default hardcoded - user specifies whatever they have loaded.

**LLM prompt** (~2000 tokens input): file path, symbol list, docstrings, first 200 lines of source. Asks for JSON: `purpose`, `module`, `domain`, `concepts[]`, `public_api_surface[]`, `complexity_notes`. Graceful fallback on JSON parse failure.

**Query API** (`KnowledgeBase` class) - reads exclusively from the **curated DB**. This is the contract Phase 2 depends on:
- `get_files(repo_id, language?)`, `get_file_by_path(repo_id, path)`
- `search_files_by_metadata(repo_id, domain?, module?, concepts?)`
- `get_symbols_for_file(file_id, kind?)`, `search_symbols_by_name(repo_id, pattern)`
- `get_symbol_neighbors(symbol_id, direction, kinds?)` (Python in MVP)
- `get_dependencies(file_id, direction)`, `get_dependency_subgraph(file_ids, depth)`
- `get_co_change_neighbors(file_id, min_count)`
- `get_docstrings_for_file(file_id)`, `get_rationale_comments(file_id)`
- `get_recent_commits_for_file(file_id, limit)`
- `get_file_context(file_id)` - composite: symbols + docstrings + rationale comments + deps + co-changes + commits
- `get_repo_overview(repo_id)` - file counts, domain distribution, most-connected files

**Depends on**: Step 7 (orchestrator populates the data this queries)

---

## Step Dependency Graph

```
Step 1 (DB + Skeleton)
  -> Step 2 (File Scanner)
     -> Step 3 (Python Parser)
        -> Step 4 (TS + JS Parsers)
        -> Step 5 (Import Resolution) <- Step 4
  -> Step 6 (Git) [parallel track]
  -> Step 7 (Orchestrator) <- Steps 5, 6
  -> Step 8 (LLM + Query API)
```

Steps 3-5 and Step 6 are independent tracks - can be built in parallel.

---

## Verification (Phase 1 Gate)

After all 8 steps, run this end-to-end test:

```bash
# Index a real repo (this repo, or any Python/TS/JS project)
cra index /path/to/repo -v

# Verify curated DB
sqlite3 .clean_room/curated.sqlite <<'SQL'
SELECT COUNT(*) as files FROM files;
SELECT COUNT(*) as symbols FROM symbols;
SELECT COUNT(*) as deps FROM dependencies;
SELECT COUNT(*) as commits FROM commits;
SELECT kind, COUNT(*) FROM symbols GROUP BY kind;
SELECT language, COUNT(*) FROM files GROUP BY language;
SQL

# Verify raw DB has indexing run metadata
sqlite3 .clean_room/raw.sqlite <<'SQL'
SELECT * FROM index_runs ORDER BY timestamp DESC LIMIT 5;
SQL

# Verify connection factory
python -c "
from clean_room.db.connection import get_connection
# All three roles should work
curated = get_connection('curated')
raw = get_connection('raw')
session = get_connection('session', task_id='test_verify')
print('All three DB connections created successfully')
curated.close()
raw.close()
session.close()
"

# Verify queries from Python (reads curated DB only)
python -c "
from clean_room.query.api import KnowledgeBase
kb = KnowledgeBase()
overview = kb.get_repo_overview(1)
print(overview)
# Pick a file and get its full context
ctx = kb.get_file_context(1)
print(f'{ctx.file.path}: {len(ctx.symbols)} symbols, {len(ctx.outgoing_deps)} deps')
# Traverse dependency graph
graph = kb.get_dependency_subgraph([1], depth=2)
print(f'Subgraph: {len(graph.files)} files, {len(graph.edges)} edges')
"

# Optional: enrich with LLM metadata (local Ollama, no data leaves machine)
cra enrich /path/to/repo --model <your-loaded-model>
```

**Gate criteria**: All curated DB tables populated, queries return meaningful results, incremental re-index works (modify a file, re-run, only changed file re-parsed), raw DB logs indexing runs, connection factory creates all three DB types correctly.

