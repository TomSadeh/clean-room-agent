# Phase 2 Implementation Plan: Pipeline Infrastructure + Code Retrieval

Status: **Complete**

## Implemented Components

### WI-0: Session Helpers (`db/session_helpers.py`)
- `set_state(conn, key, value)` — JSON-serialize and UPSERT
- `get_state(conn, key)` — returns deserialized Python object or None
- `delete_state(conn, key)` — delete with rowcount warning
- `list_keys(conn, prefix?)` — LIKE-based prefix search with escaping

### WI-1: Raw DB Additions (`db/raw_queries.py`)
- `insert_retrieval_llm_call` — log all stage/task analysis LLM calls
- `insert_retrieval_decision` — log per-file inclusion decisions
- `insert_task_run` / `update_task_run` — task lifecycle tracking

### WI-2: Retrieval Dataclasses (`retrieval/dataclasses.py`)
- `BudgetConfig` — context_window, reserved_tokens with validation
- `TaskQuery` — parsed task with seeds, keywords, error_patterns
- `ScopedFile` — tier + relevance with `__post_init__` validation
- `ClassifiedSymbol` — detail level + signature with `__post_init__` validation
- `FileContent` — rendered file entry for context package
- `ContextPackage` — final curated context with `to_prompt_text()` (R5: XML tags)
- `RefinementRequest` — re-entry with missing files/symbols/error_signatures

### WI-3: Budget Module (`retrieval/budget.py`)
- `estimate_tokens(text)` — chars/4 estimate
- `estimate_framing_tokens(path, language, detail_level)` — R5 overhead
- `BudgetTracker` — 0.9 safety margin, consume/can_fit tracking

### WI-4: Stage Protocol (`retrieval/stage.py`)
- `StageContext` — mutable context threaded through stages, with `to_dict()`/`from_dict()` serialization (includes `signature` field)
- `RetrievalStage` — runtime-checkable Protocol
- `@register_stage("name")` — decorator-based registry

### WI-5: Task Analysis (`retrieval/task_analysis.py`)
- `extract_task_signals(raw_task)` — regex extraction of files, symbols, keywords, error patterns, task type
- `resolve_seeds(signals, kb, repo_id)` — resolve to DB IDs with exact-match preference, per-pattern cap (R6)
- `enrich_task_intent(raw_task, signals, llm)` — LLM intent summary

### WI-6: Scope Stage (`retrieval/scope_stage.py`)
- `expand_scope(task, kb, repo_id)` — 5-tier deterministic expansion (plan/seed/dep/co-change/metadata)
- `judge_scope(candidates, task, llm)` — LLM judgment with batching (R3), verdict validation, default-deny (R2)
- Cached dependency lookups (no quadratic re-queries), globally sorted co-changes (R6)

### WI-7: Precision Stage (`retrieval/precision_stage.py`)
- `classify_symbols(symbols, task, llm)` — LLM detail level classification with batching
- Key: `(name, file_path, start_line)` for collision safety

### WI-8: Context Assembly (`retrieval/context_assembly.py`)
- `assemble_context(context, budget, repo_path, llm?, kb?)` — budget-compliant rendering
- R1: `_refilter_files()` LLM call + `_drop_by_priority()` fallback, no downgrades
- R4: `_extract_supporting()` uses docstring data from curated DB
- R5: task/intent header overhead consumed before file assembly
- R3: refilter prompt budget-validated; hallucinated path fallback

### WI-9: Pipeline Runner (`retrieval/pipeline.py`)
- `run_pipeline(...)` — full orchestration with DB logging, session persistence
- `_LoggingLLMClient` — wraps LLMClient to record all calls for raw DB logging
- Mode-aware execute model resolution (plan→reasoning, implement→coding)
- Refinement re-entry via `_resume_task_from_session()`

### WI-10: Preflight (`retrieval/preflight.py`)
- DB existence and population checks using `get_connection`
- Enrichment status informational logging

### WI-11: Shared Utilities (`retrieval/utils.py`)
- `parse_json_response(text, context)` — robust JSON extraction from LLM output
