# Database Schemas

Full schema definitions for all three databases plus session DB key contracts.

---

## 1. Curated DB Schema

```sql
-- Repository registration
repos (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL,
  remote_url TEXT,
  indexed_at TEXT NOT NULL
)

-- Indexed files
files (
  id INTEGER PRIMARY KEY,
  repo_id INTEGER NOT NULL REFERENCES repos(id),
  path TEXT NOT NULL,
  language TEXT NOT NULL,
  content_hash TEXT NOT NULL,     -- hex SHA-256
  size_bytes INTEGER NOT NULL
)

-- AST-extracted symbols
symbols (
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id),
  name TEXT NOT NULL,
  kind TEXT NOT NULL,             -- function, class, method, interface, type_alias, enum, variable
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  signature TEXT,                 -- source text of the def/class line
  parent_symbol_id INTEGER REFERENCES symbols(id)
)

-- Docstrings
docstrings (
  id INTEGER PRIMARY KEY,
  symbol_id INTEGER REFERENCES symbols(id),  -- NULL for module-level
  file_id INTEGER NOT NULL REFERENCES files(id),
  content TEXT NOT NULL,
  format TEXT,                   -- google, numpy, sphinx, jsdoc, plain
  parsed_fields TEXT             -- JSON: structured field extraction
)

-- Inline comments
inline_comments (
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id),
  symbol_id INTEGER REFERENCES symbols(id),  -- innermost enclosing, NULL if module-level
  line INTEGER NOT NULL,
  content TEXT NOT NULL,
  kind TEXT,                     -- todo, fixme, hack, note, bug_ref, rationale, general
  is_rationale INTEGER NOT NULL DEFAULT 0
)

-- File-level dependency edges
dependencies (
  id INTEGER PRIMARY KEY,
  source_file_id INTEGER NOT NULL REFERENCES files(id),
  target_file_id INTEGER NOT NULL REFERENCES files(id),
  kind TEXT NOT NULL              -- "import" or "type_ref"
)

-- Symbol-level reference edges (Python MVP only)
symbol_references (
  id INTEGER PRIMARY KEY,
  caller_symbol_id INTEGER NOT NULL REFERENCES symbols(id),
  callee_symbol_id INTEGER NOT NULL REFERENCES symbols(id),
  reference_kind TEXT NOT NULL
)

-- Git commits
commits (
  id INTEGER PRIMARY KEY,
  repo_id INTEGER NOT NULL REFERENCES repos(id),
  hash TEXT NOT NULL,
  author TEXT,
  message TEXT,
  timestamp TEXT NOT NULL,
  files_changed INTEGER,
  insertions INTEGER,
  deletions INTEGER
)

-- File-commit associations
file_commits (
  file_id INTEGER NOT NULL REFERENCES files(id),
  commit_id INTEGER NOT NULL REFERENCES commits(id),
  PRIMARY KEY (file_id, commit_id)
)

-- Co-change pairs
co_changes (
  file_a_id INTEGER NOT NULL REFERENCES files(id),
  file_b_id INTEGER NOT NULL REFERENCES files(id),
  count INTEGER NOT NULL DEFAULT 1,
  last_commit_hash TEXT,
  PRIMARY KEY (file_a_id, file_b_id),
  CHECK (file_a_id < file_b_id)
)

-- LLM-enriched file metadata (populated by cra enrich --promote)
file_metadata (
  file_id INTEGER PRIMARY KEY REFERENCES files(id),
  purpose TEXT,
  module TEXT,
  domain TEXT,
  concepts TEXT,                 -- JSON array
  public_api_surface TEXT,       -- JSON array
  complexity_notes TEXT
)

-- Active LoRA adapter metadata (Phase 4, read by pipeline at runtime)
adapter_metadata (
  id INTEGER PRIMARY KEY,
  stage_name TEXT NOT NULL,          -- pipeline stage this adapter serves
  base_model TEXT NOT NULL,          -- "coding" or "reasoning"
  model_tag TEXT NOT NULL,           -- model/adapter identifier for the inference server (e.g. "qwen3:4b-scope-v1")
  version INTEGER NOT NULL DEFAULT 1,
  active INTEGER NOT NULL DEFAULT 1, -- 1 = active, 0 = inactive
  performance_notes TEXT,            -- free-form metrics/observations
  deployed_at TEXT NOT NULL
)
```

**Indexes**: `UNIQUE repos(path)`, `UNIQUE files(repo_id, path)`, `symbols(file_id)`, `symbols(name)`, `symbol_references(caller_symbol_id)`, `symbol_references(callee_symbol_id)`, `dependencies(source_file_id)`, `dependencies(target_file_id)`, `UNIQUE commits(repo_id, hash)`, `file_metadata(domain)`, `file_metadata(module)`, `adapter_metadata(stage_name, active)`.

---

## 2. Raw DB Schema

```sql
-- Phase 1: indexing run metadata
index_runs (
  id INTEGER PRIMARY KEY,
  repo_path TEXT NOT NULL,
  files_scanned INTEGER NOT NULL,
  files_changed INTEGER NOT NULL,
  duration_ms INTEGER NOT NULL,
  status TEXT NOT NULL,
  timestamp TEXT NOT NULL
)

-- Phase 1: LLM enrichment outputs (full record including prompt/response)
enrichment_outputs (
  id INTEGER PRIMARY KEY,
  task_id TEXT,                   -- nullable; NULL for standalone cra enrich, set when pipeline-driven
  file_id INTEGER NOT NULL,
  model TEXT NOT NULL,
  purpose TEXT,
  module TEXT,
  domain TEXT,
  concepts TEXT,                  -- JSON
  public_api_surface TEXT,        -- JSON
  complexity_notes TEXT,
  system_prompt TEXT,             -- the system prompt sent with this call
  raw_prompt TEXT NOT NULL,
  raw_response TEXT NOT NULL,
  promoted INTEGER NOT NULL DEFAULT 0,
  timestamp TEXT NOT NULL
)

-- Phase 2: LLM calls during retrieval (task analysis + stage judgment + assembly refilter)
retrieval_llm_calls (
  id INTEGER PRIMARY KEY,
  task_id TEXT NOT NULL,
  call_type TEXT NOT NULL,       -- "task_analysis", "scope", "precision", "assembly_refilter", etc.
  stage_name TEXT,               -- NULL for task_analysis, stage name for stage/assembly calls
  model TEXT NOT NULL,
  system_prompt TEXT,            -- the system prompt sent with this call
  prompt TEXT NOT NULL,
  response TEXT NOT NULL,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  latency_ms INTEGER NOT NULL,
  timestamp TEXT NOT NULL
)

-- Phase 2: per-file and per-symbol retrieval decisions
retrieval_decisions (
  id INTEGER PRIMARY KEY,
  task_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  file_id INTEGER NOT NULL,
  symbol_id INTEGER,              -- NULL for file-level decisions, set for symbol-level
  detail_level TEXT,              -- NULL for file-level; primary/supporting/type_context/excluded for symbols
  tier TEXT,
  included INTEGER NOT NULL,
  reason TEXT,
  timestamp TEXT NOT NULL
)

-- Pipeline run metadata (one row per pipeline command invocation)
task_runs (
  id INTEGER PRIMARY KEY,
  task_id TEXT NOT NULL UNIQUE,
  repo_path TEXT NOT NULL,
  mode TEXT NOT NULL,             -- "plan", "implement", "train_plan", "curate_data" (underscores, not hyphens)
  execute_model TEXT NOT NULL,    -- model used for execute stage
  context_window INTEGER NOT NULL,
  reserved_tokens INTEGER NOT NULL,
  stages TEXT NOT NULL,           -- CSV of stage names
  plan_artifact TEXT,             -- path to plan file, if mode=implement and --plan provided
  success INTEGER,               -- NULL until finalized
  total_tokens INTEGER,
  total_latency_ms INTEGER,
  final_diff TEXT,                -- for implement mode; NULL for plan mode
  final_plan TEXT,                -- for plan mode; NULL for implement mode
  timestamp TEXT NOT NULL
)

-- Phase 3: per-attempt results
run_attempts (
  id INTEGER PRIMARY KEY,
  task_run_id INTEGER NOT NULL REFERENCES task_runs(id),
  attempt INTEGER NOT NULL,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  latency_ms INTEGER NOT NULL,
  raw_response TEXT NOT NULL,
  patch_applied INTEGER NOT NULL,
  timestamp TEXT NOT NULL
)

-- Phase 3: validation results
validation_results (
  id INTEGER PRIMARY KEY,
  attempt_id INTEGER NOT NULL REFERENCES run_attempts(id),
  success INTEGER NOT NULL,
  test_output TEXT,
  lint_output TEXT,
  type_check_output TEXT,
  failing_tests TEXT             -- JSON array
)

-- Archived session DB files
session_archives (
  id INTEGER PRIMARY KEY,
  task_id TEXT NOT NULL,
  session_blob BLOB NOT NULL,    -- raw bytes of session SQLite file
  archived_at TEXT NOT NULL
)

-- Phase 4: training plans (output of cra train-plan)
training_plans (
  id INTEGER PRIMARY KEY,
  task_id TEXT NOT NULL,          -- task_id of the train-plan run
  target_stage TEXT NOT NULL,     -- which stage to train
  base_model TEXT NOT NULL,       -- "coding" or "reasoning"
  source TEXT NOT NULL DEFAULT 'logged',  -- "logged" or "synthetic"
  data_criteria TEXT NOT NULL,    -- JSON: selection criteria for training data
  hyperparameters TEXT,           -- JSON: suggested hyperparameters
  improvement_targets TEXT,       -- JSON: expected improvement metrics
  raw_plan TEXT NOT NULL,         -- full plan text from LLM
  timestamp TEXT NOT NULL
)

-- Phase 4: curated training datasets (output of cra curate-data)
training_datasets (
  id INTEGER PRIMARY KEY,
  training_plan_id INTEGER NOT NULL REFERENCES training_plans(id),
  stage_name TEXT NOT NULL,
  base_model TEXT NOT NULL,       -- "coding" or "reasoning"
  source TEXT NOT NULL DEFAULT 'logged',  -- "logged" (from agent runs) or "synthetic" (from external repo history)
  dataset_path TEXT NOT NULL,     -- path to JSONL file
  example_count INTEGER NOT NULL,
  positive_count INTEGER NOT NULL,
  negative_count INTEGER NOT NULL,
  format TEXT NOT NULL DEFAULT 'jsonl',
  timestamp TEXT NOT NULL
)

-- Phase 4: adapter training and deployment history
adapter_registry (
  id INTEGER PRIMARY KEY,
  dataset_id INTEGER NOT NULL REFERENCES training_datasets(id),
  stage_name TEXT NOT NULL,
  base_model TEXT NOT NULL,
  model_tag TEXT NOT NULL,        -- model/adapter identifier for the inference server
  training_loss REAL,
  eval_metrics TEXT,              -- JSON: evaluation metrics
  deployed INTEGER NOT NULL DEFAULT 0,
  trained_at TEXT NOT NULL
)

-- Phase 3: orchestrator run metadata (one row per cra solve without --plan)
orchestrator_runs (
  id INTEGER PRIMARY KEY,
  task_id TEXT NOT NULL,           -- top-level task_id for the orchestrator invocation
  repo_path TEXT NOT NULL,
  task_description TEXT NOT NULL,
  total_parts INTEGER,
  total_steps INTEGER,
  parts_completed INTEGER NOT NULL DEFAULT 0,
  steps_completed INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,            -- "running", "complete", "partial", "failed"
  timestamp TEXT NOT NULL,
  completed_at TEXT
)

-- Phase 3: links orchestrator runs to constituent pipeline passes
orchestrator_passes (
  id INTEGER PRIMARY KEY,
  orchestrator_run_id INTEGER NOT NULL REFERENCES orchestrator_runs(id),
  task_run_id INTEGER NOT NULL REFERENCES task_runs(id),
  pass_type TEXT NOT NULL,         -- "meta_plan", "part_plan", "step_implement", "adjustment"
  part_id TEXT,                    -- NULL for meta_plan pass
  step_id TEXT,                    -- NULL for meta_plan and part_plan passes
  sequence_order INTEGER NOT NULL,
  timestamp TEXT NOT NULL
)
```

---

## 3. Session DB Schema

A single key-value table. The session is ephemeral, never queried externally, and doesn't need relational structure.

```sql
kv (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,            -- JSON-serialized
  updated_at TEXT NOT NULL
)
```

**Helpers**: `set_state(key, value)`, `get_state(key) -> str | None`, `delete_state(key)`, `list_keys(prefix?) -> list[str]`.

---

## 4. Session DB Key Contracts

| Key | Writer | Reader | Content |
|-----|--------|--------|---------|
| `task_query` | Phase 2 | Phase 2 (re-entry) | Serialized `TaskQuery` fields including error_patterns |
| `stage_output_{stage_name}` | Phase 2 | Phase 2 (re-entry) | Serialized `StageContext.to_dict()` per stage |
| `stage_progress` | Phase 2 | Phase 2 (re-entry) | `{completed: [...], remaining: [...]}` |
| `final_context` | Phase 2 | Phase 2 (re-entry), Phase 3 | Serialized final `StageContext` |
| `refinement_request` | Phase 3 | Phase 2 (re-entry) | Serialized `RefinementRequest` |
| `attempt_summary` | Phase 3 | Phase 2 (logging) | Summary of attempts leading to refinement |
| `retry_context` | Phase 3 | Phase 3 (prompt builder) | Error classifications, attempt history for retry prompts |
| `plan_artifact` | Phase 3 (plan mode) | Phase 3 (implement mode via file) | Serialized plan output |
| `meta_plan` | Orchestrator | Orchestrator | Serialized `MetaPlan` decomposition |
| `part_plan:<part_id>` | Orchestrator | Orchestrator | Serialized `PartPlan` for one part |
| `step_result:<part_id>:<step_id>` | Orchestrator | Orchestrator | Serialized `StepResult` |
| `adjustment:<part_id>:<after_step_id>` | Orchestrator | Orchestrator | Serialized `PlanAdjustment` |
| `orchestrator_progress` | Orchestrator | Orchestrator | Current part/step index, status |
| `cumulative_diff` | Orchestrator | Orchestrator | Accumulated diff from all completed steps |
