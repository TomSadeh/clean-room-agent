# Phase 1 Implementation Plan: Knowledge Base + Indexer

## Context

Phase 1 is the foundation layer everything else builds on. It creates the three-database architecture, deterministic indexing pipeline, query API, LLM client/routing, and enrichment pipeline. Built from the contracts in `planning/meta-plan.md`.

**Status**: Complete. All work items (WI-0 through WI-9) implemented and tested.

---

## Package Structure

```
pyproject.toml
src/
  clean_room_agent/
    __init__.py
    __main__.py                       # python -m clean_room_agent
    cli.py                            # Click CLI: cra init, cra index, cra enrich
    config.py                         # TOML loader + validation
    db/
      __init__.py
      connection.py                   # get_connection(role, task_id, read_only)
      schema.py                       # DDL for all 3 DBs + index creation
      queries.py                      # Curated DB insert/upsert/delete helpers
      raw_queries.py                  # Raw DB insert helpers
    indexer/
      __init__.py
      orchestrator.py                 # index_repository(): coordinates extractors, incremental logic
      file_scanner.py                 # scan_repo() -> list[FileInfo], gitignore, hashing
    parsers/
      __init__.py
      base.py                         # LanguageParser protocol + shared dataclasses
      python_parser.py                # tree-sitter: symbols, docstrings, comments, symbol references
      ts_js_parser.py                 # tree-sitter: file-level symbols, JSDoc, imports (no symbol refs)
      registry.py                     # language -> parser dispatch
    extractors/
      __init__.py
      dependencies.py                 # Import resolution -> file-level dependency edges
      git_extractor.py                # Git history, file-commit associations, co-change pairs
    llm/
      __init__.py
      client.py                       # ModelConfig, LLMResponse, LLMClient (Ollama httpx transport)
      router.py                       # ModelRouter: resolve(role, stage_name) -> ModelConfig
      enrichment.py                   # Per-file LLM enrichment, prompt building, JSON parsing
    query/
      __init__.py
      api.py                          # KnowledgeBase class: ~15 query methods
      models.py                       # Dataclasses: File, Symbol, Docstring, Comment, Commit, etc.
tests/
  conftest.py                         # Shared fixtures: tmp DBs, sample repos
  fixtures/
    sample_python/                    # Multi-file Python package
    sample_typescript/                # Small TS project
    sample_javascript/                # Small JS project
  db/
    test_connection.py
    test_schema.py
    test_queries.py
  indexer/
    test_file_scanner.py
    test_orchestrator.py
  parsers/
    test_python_parser.py
    test_ts_js_parser.py
  extractors/
    test_dependencies.py
    test_git_extractor.py
  llm/
    test_client.py
    test_router.py
    test_enrichment.py
  query/
    test_api.py
  test_cli.py
```

## External Dependencies

**Runtime** (6 packages):
- `click` — CLI framework (subcommand ergonomics over argparse)
- `tree-sitter>=0.25,<0.26` — AST parsing for all 3 languages
- `tree-sitter-python>=0.23` — Python grammar
- `tree-sitter-javascript>=0.25` — JS grammar
- `tree-sitter-typescript>=0.23` — TS/TSX grammar
- `httpx` — Ollama HTTP transport
- `pathspec` — gitignore pattern matching

**Dev**: `pytest`, `pytest-cov`, `ruff`

**Stdlib** (no extra deps): `tomllib` (config), `sqlite3` (DB), `hashlib` (SHA-256), `subprocess` (git), `pathlib`, `json`, `uuid`, `dataclasses`, `logging`

**Key decision — tree-sitter for Python too**: Python's `ast` module cannot extract comments (they're discarded by the tokenizer). Since comment classification (rationale, TODO, etc.) is a first-class feature, tree-sitter is required even for Python. One parsing API for all 3 languages.

## Work Items

### WI-0: Save Implementation Plan
Save this plan to `planning/phase1-implementation.md` for audit trail.

### WI-1: Project Skeleton + DB Layer
**Deps**: None

- `pyproject.toml` — hatchling build, deps, `cra` entry point
- `db/connection.py` — `get_connection(role, *, repo_path, task_id=None, read_only=False)`:
  - Validates role is `"curated"` / `"raw"` / `"session"`
  - `"session"` requires `task_id`
  - Paths: `.clean_room/{curated,raw}.sqlite`, `.clean_room/sessions/session_{task_id}.sqlite`
  - Sets WAL mode, FK enabled, `sqlite3.Row` factory
  - `read_only=True` uses `?mode=ro` URI
  - Applies schema on first creation (idempotent)
- `db/schema.py` — `create_curated_schema(conn)` (13 tables + indexes per meta-plan 4.5), `create_raw_schema(conn)` (13 tables per meta-plan 4.5-4.7), `create_session_schema(conn)` (1 KV table per meta-plan 4.7)
- `db/queries.py` — Curated helpers: `upsert_repo`, `upsert_file`, `insert_symbol`, `insert_docstring`, `insert_inline_comment`, `insert_dependency`, `insert_symbol_reference`, `insert_commit`, `insert_file_commit`, `upsert_co_change`, `upsert_file_metadata`, `delete_file_data` (cascade)
- `db/raw_queries.py` — Raw helpers: `insert_index_run`, `insert_enrichment_output`
- `cli.py` — Click group with stub commands
- Tests for connection, schema, queries

### WI-2: Config Loader
**Deps**: WI-1

- `config.py`:
  - `load_config(repo_path) -> dict | None` — reads `.clean_room/config.toml` with `tomllib`, returns `None` if file missing
  - `require_models_config(config) -> dict` — extracts `[models]` or raises hard error directing to `cra init`
  - No hardcoded defaults for required values
  - Resolution: CLI flag -> config.toml -> hard error

### WI-3: File Scanner
**Deps**: WI-1

- `indexer/file_scanner.py`:
  - `FileInfo` dataclass: path (relative), abs_path, language, content_hash (SHA-256 hex), size_bytes
  - `scan_repo(repo_path) -> list[FileInfo]`
  - Language detection: `.py` -> python, `.ts`/`.tsx` -> typescript, `.js`/`.jsx`/`.mjs`/`.cjs` -> javascript
  - Skips: `.git/`, `node_modules/`, `__pycache__/`, `.venv/`, `.clean_room/`, `*.pyc`, `*.min.js`, etc.
  - Uses `pathspec` for `.gitignore` parsing
  - Max file size 1MB (skip with warning log)

### WI-4: Python AST Parser
**Deps**: WI-1

- `parsers/base.py` — `LanguageParser` protocol + shared dataclasses: `ExtractedSymbol`, `ExtractedDocstring`, `ExtractedComment`, `ExtractedImport`, `ExtractedReference`, `ParseResult`
- `parsers/python_parser.py`:
  - Symbol extraction via tree-sitter: functions, classes, methods, nested, module-level variables
  - Parent-child relationships via tree node containment
  - Docstring extraction with format detection (Google, NumPy, Sphinx, plain)
  - Comment classification: TODO, FIXME, HACK, NOTE, bug_ref, rationale, general
  - Symbol reference edges (Python MVP only): scan function bodies for identifier/attribute nodes matching known symbols (same-file only; cross-file relevance is Phase 2 LLM judgment)
- `parsers/registry.py` — `get_parser(language) -> LanguageParser`

### WI-5: TS/JS Parser
**Deps**: WI-4 (shares base.py)

- `parsers/ts_js_parser.py`:
  - `TSJSParser` with language parameter ("typescript" or "javascript")
  - tree-sitter-typescript for .ts/.tsx, tree-sitter-javascript for .js/.jsx
  - Symbols: function/class declarations, interface/type_alias/enum (TS), arrow functions, exports
  - JSDoc extraction with tag parsing (@param, @returns, etc.)
  - Comment classification (same heuristics as Python)
  - Import extraction: ES modules + CommonJS (JS only)
  - No symbol reference edges (explicit MVP boundary)
- **First step**: verify tree-sitter-typescript 0.23.2 ABI compatibility with tree-sitter 0.25.x. If incompatible, fall back to `tree-sitter-language-pack`

### WI-6: Dependency Extractor
**Deps**: WI-4, WI-5

- `extractors/dependencies.py`:
  - `resolve_dependencies(imports, file_path, language, file_index, repo_path) -> list[ResolvedDep]`
  - Python: absolute imports (dots to path), relative imports (level counting), `__init__.py` resolution
  - TS/JS: relative imports (try extensions .ts/.tsx/.js/.jsx + index.*), basic `tsconfig.json` `baseUrl`+`paths`
  - External packages discarded (intra-repo only)
  - `kind`: TS `import type` -> `"type_ref"`, everything else -> `"import"`

### WI-7: Git Extractor
**Deps**: WI-1 (independent of parsers — can parallelize with WI-4/5/6)

- `extractors/git_extractor.py`:
  - `extract_git_history(repo_path, file_index, max_commits) -> GitHistory`
  - Uses `git log --pretty=format:... --numstat` via subprocess
  - Commit info: hash, author, message, timestamp, files_changed, insertions, deletions
  - File-commit associations (only files in `file_index`)
  - Co-change pairs: all pairs from each commit (skip commits with >50 files), accumulate counts, filter count < 2
  - Remote URL detection: `git config --get remote.origin.url`
  - Git not found -> clear error with instructions

### WI-8: Indexing Orchestrator + CLI Wiring
**Deps**: WI-1 through WI-7

- `indexer/orchestrator.py`:
  - `index_repository(repo_path, continue_on_error=False) -> IndexResult`
  - Pipeline: open DBs -> apply schemas -> register repo -> scan files -> incremental diff (content_hash comparison) -> delete changed/removed file data -> upsert files -> parse new/changed (try/except per file if continue_on_error) -> resolve all dependencies -> extract git history -> log to raw DB -> return results
  - `IndexResult`: files_scanned, files_new, files_changed, files_deleted, files_unchanged, parse_errors, duration_ms
- `cli.py` expanded:
  - `cra init` — create `.clean_room/config.toml`, add `.clean_room/` to `.gitignore`
  - `cra index <repo-path> [--continue-on-error] [-v]` — run indexer, print summary
- Integration tests: create temp repo, index, verify DB, modify file, re-index, verify incremental

### WI-9: LLM Client + Enrichment + Query API
**Deps**: WI-8

- `llm/client.py` — `ModelConfig`, `LLMResponse`, `LLMClient` (httpx POST to Ollama `/api/generate`, fail-fast, no retry)
- `llm/router.py` — `ModelRouter.resolve(role, stage_name)`: override -> role default -> hard error. Temperature 0.0 for both roles
- `llm/enrichment.py` — per-file enrichment: build prompt (file path + symbols + docstrings + source preview), call reasoning model, parse JSON (purpose/module/domain/concepts/public_api_surface/complexity_notes), write to raw DB, optionally promote to curated
- `query/models.py` — dataclasses: `File`, `Symbol`, `Docstring`, `Comment`, `Commit`, `FileContext`, `RepoOverview`
- `query/api.py` — `KnowledgeBase(conn)` with all methods per meta-plan section 12:
  - `get_files`, `get_file_by_path`, `search_files_by_metadata`
  - `get_symbols_for_file`, `search_symbols_by_name`, `get_symbol_neighbors`
  - `get_dependencies`, `get_dependency_subgraph`
  - `get_co_change_neighbors`
  - `get_docstrings_for_file`, `get_rationale_comments`
  - `get_recent_commits_for_file`
  - `get_file_context` (composite), `get_repo_overview`
  - `get_adapter_for_stage`
- `cli.py` — add `cra enrich <repo-path> [--promote]`

## Dependency Graph

```
WI-0 (save plan)
  |
WI-1 (DB + skeleton)
  |
  +---> WI-2 (config) --------+
  +---> WI-3 (file scanner) --+
  +---> WI-4 (Python parser)  |
  |       |                    |
  |       +-> WI-5 (TS/JS) ---+
  |            |               |
  |            +-> WI-6 (deps) +
  +---> WI-7 (git extractor) -+
                               |
                        WI-8 (orchestrator + CLI)
                               |
                        WI-9 (LLM + enrichment + query API)
```

## Risks

1. **tree-sitter-typescript ABI** (MEDIUM): 0.23.2 grammar may not load in 0.25.x runtime. Mitigated by verifying in WI-5 first step; fallback is `tree-sitter-language-pack`.
2. **Python symbol reference accuracy** (LOW): Name-based matching produces false positives. Acceptable — Phase 2 LLM judgment filters.
3. **Git subprocess on Windows** (LOW): Requires git on PATH. Mitigated by clear error message.

## Phase 1 Gate (from meta-plan)

- `cra index` populates curated DB for a real repo
- Query API returns meaningful results for all methods
- Incremental re-index works (modify file, re-run, only changed file re-parsed)
- Raw DB logs indexing runs
- Connection factory manages all 3 DB types with WAL/FK/Row
- `cra index` works without LLM (deterministic only)
- `cra enrich` runs end-to-end, `--promote` copies to curated

## Verification

1. `pip install -e ".[dev]"` succeeds
2. `cra --help` shows all commands
3. `pytest` passes all tests
4. Manual: `cra init` on this repo, `cra index .`, inspect `.clean_room/curated.sqlite` and `.clean_room/raw.sqlite`
5. Manual: modify a file, re-run `cra index .`, verify incremental behavior
6. Manual: `cra enrich . --promote` (requires Ollama running with configured models)
