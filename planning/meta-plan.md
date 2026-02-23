# Clean Room Agent Meta-Plan

Single source of truth for phase boundaries, cross-phase contracts, database schemas, pipeline protocol, CLI conventions, and all architectural decisions. Phase-specific implementation plans derive from this document and must not contradict it.

---

## 1. Vision: The Self-Improving Loop

The agent is a self-improving system. The same N-prompt pipeline architecture (analyze task, curate context, execute) serves multiple modes — planning, coding, training planning, data curation — creating a closed loop where the agent improves itself from its own logged activity.

```
                    +--------------------------------------+
                    |         Real Coding Tasks            |
                    +------------------+-------------------+
                                       |
                                       v
                    +----------------------+
                    |   Plan Mode          |  analyze -> retrieve code -> produce plan
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Implement Mode     |  analyze -> retrieve code+plan -> produce code
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Raw DB             |  every LLM call, decision, attempt, outcome
                    |   (training corpus)  |  logged with full I/O and quality signals
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Train-Plan Mode    |  analyze -> retrieve logs/metrics -> training plan
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Curate-Data Mode   |  analyze -> retrieve logs -> curated dataset
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   LoRA Training      |  per-stage adapters from curated data
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Deploy Adapters    |  stage-specific model overrides
                    +----------+-----------+
                               |
                               +--------> back to top (better at real tasks)
```

The loop: work on real tasks -> log everything -> analyze what's weak -> curate training data -> train LoRAs -> deploy -> work better on real tasks.

**Bootstrapping**: The loop does not require the agent's own output to start. Mature, well-tested repos cloned from GitHub provide commit histories with natural mistake -> solution pairs. These are used to generate synthetic training data *before* the agent produces any real runs, breaking the chicken-and-egg dependency between Phase 3 output quality and Phase 4 training data quality.

---

## 2. Phase Boundaries

### Phase 1: Knowledge Base + Indexer

**Builds**: All three DB schemas, connection factory, deterministic indexing (file scanning, AST parsing for Python/TS/JS, dependency graphs, git metadata, co-change analysis), query API, LLM enrichment (optional), project config loader.

**CLI commands**: `cra init`, `cra index`, `cra enrich`

**Gate**: `cra index` populates curated DB for a real repo. Query API returns meaningful results. Incremental re-index works (modify a file, re-run, only changed file re-parsed). Raw DB logs indexing runs. Connection factory manages all three DB types. All three schemas are correct.

### Phase 2: Pipeline Infrastructure + Code Retrieval

**Builds**: Stage protocol, pipeline runner, task analysis, MVP retrieval stages (Scope, Precision), budget enforcement, context assembly, retrieval CLI.

**CLI commands**: `cra retrieve`

**Gate**: Retrieval runs end-to-end, produces budget-compliant context packages, logs all decisions and LLM calls to raw DB. Refinement re-entry works.

### Phase 3: Code Agent

**Builds**: Plan mode execute stage + implement mode execute stage. Prompt builder, response parser, patch application (git worktree isolation), validation, retry loop, refinement handoff to Phase 2, solve orchestrator.

**CLI commands**: `cra plan`, `cra solve`

**Gate**: `cra plan` produces actionable plans. `cra solve` produces valid patches. All activity logged to raw DB with full I/O and outcome linkage.

### Phase 4: Self-Improvement Loop

**Builds**: Log analysis query API (raw DB equivalent of KnowledgeBase), training-specific retrieval stages, train-plan mode execute stage, curate-data mode execute stage, LoRA adapter registry and stage-level model overrides, **synthetic training data pipeline** (commit-history extraction, diff filtering, synthetic retrieval pair generation from external repos).

**CLI commands**: `cra train-plan`, `cra curate-data`

**Two data sources for training**:
- **Bootstrapping (external repos)**: Clone mature, well-tested repos from GitHub. Extract commit histories as mistake -> solution pairs. Generate synthetic pipeline run data (what scope/precision *should have* retrieved for a given commit's changes). Does not require prior agent runs.
- **Self-improvement (logged runs)**: Once Phase 3 has generated enough logged data, the original self-improvement loop applies -- analyze real runs, curate from actual agent behavior, train on the agent's own successes and failures.

**Gate**: Can generate synthetic training pairs from external repo commit histories. Can analyze logged runs (when available), produce training plans, curate valid datasets, deploy adapters that improve stage performance.

### Dependency Order

Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 (self-improvement path). No phase pulls validation gates from a later phase.

**Exception**: Phase 4's bootstrapping path (synthetic training data from external repos) depends only on Phase 1 (indexing). Once a repo is indexed, commit-history extraction and synthetic retrieval pair generation can proceed without Phases 2-3. This means LoRA training can begin as soon as Phase 1 is complete and Phase 4's synthetic pipeline is built, potentially improving base model quality before the code agent (Phase 3) is first used.

The self-improvement path within Phase 4 still requires Phase 3 to have generated enough logged data to be useful.

### MVP Boundary

Phase 3 (coding agent with plan + implement modes) is the MVP. Phase 4 (training loop) is post-MVP but architecturally planned for throughout Phases 1-3.

### Out of Scope

Formal validation, benchmark claims, baseline comparison mode.

---

## 3. Multi-Model Architecture

Two base models, each serving a different role:

- **Qwen2.5-Coder-3B-Instruct** -- coding tasks (code generation, code editing)
- **Qwen3-4B-Instruct-2507** -- reasoning/planning tasks (task analysis, retrieval judgments, plan generation, enrichment, training planning, data curation)

### 3.1 Model Assignment by Stage

| Stage | Role | Base Model |
|-------|------|-----------|
| Task Analysis | reasoning | Qwen3-4B |
| Scope judgment | reasoning | Qwen3-4B |
| Precision judgment | reasoning | Qwen3-4B |
| Execute -- Plan | reasoning | Qwen3-4B |
| Execute -- Code | coding | Qwen2.5-Coder-3B |
| Enrichment | reasoning | Qwen3-4B |
| Train-Plan stages | reasoning | Qwen3-4B |
| Curate-Data stages | reasoning | Qwen3-4B |

### 3.2 Model Routing is Config-Only

No `--model` or `--base-url` CLI flags. All model configuration lives in `.clean_room/config.toml`, set up by `cra init`:

```toml
[models]
coding = "qwen2.5-coder:3b-instruct"
reasoning = "qwen3:4b-instruct-2507"
base_url = "http://localhost:11434"

[models.overrides]    # per-stage LoRA overrides (populated by training loop)
# scope = "qwen3:4b-scope-v1"
# execute_code = "qwen2.5-coder:3b-code-v1"
```

Missing `[models]` section when an LLM-using command runs -> hard error directing user to `cra init`.

### 3.3 Budget Depends on Execute Model

Budget is calculated against the execute model's context window for the current mode:
- Plan mode budgets against Qwen3-4B's context window.
- Implement mode budgets against Qwen2.5-Coder-3B's context window.
- Retrieval stage prompts are bounded by the reasoning model's window (separate, smaller concern).

### 3.4 LoRA Training Targets Two Base Models

**Qwen3-4B LoRAs (reasoning):**

| Stage | Training Pair | Quality Signal |
|-------|--------------|----------------|
| Task Analysis | task description -> structured TaskQuery | did retrieval find the right files? |
| Scope | task + candidates -> relevance classification | were the included files actually needed? |
| Precision | task + symbols -> tier classification | did the selected symbols lead to correct code? |
| Execute (Plan) | context + task -> plan | did the plan lead to successful implementation? |

**Qwen2.5-Coder-3B LoRAs (coding):**

| Stage | Training Pair | Quality Signal |
|-------|--------------|----------------|
| Execute (Code) | context + task -> code edits | did the code pass validation? |

Training data curation must separate examples by base model -- a Qwen3-4B reasoning LoRA cannot be applied to Qwen2.5-Coder-3B and vice versa.

---

## 4. Three-Database Architecture

Three separate SQLite files. Independent WAL journals, backups, and lifecycles.

### 4.1 File Layout

```
.clean_room/
  config.toml                    - project-level settings (created by cra init)
  curated.sqlite                 - indexed knowledge base
  raw.sqlite                     - append-only activity log + training corpus
  sessions/
    session_<task_id>.sqlite     - ephemeral per-task working memory
```

`.clean_room/` must be in the target repo's `.gitignore`.

### 4.2 Connection Factory

```python
get_connection(role, task_id=None, read_only=False)
```

- `role`: `"curated"`, `"raw"`, or `"session"`
- `task_id`: required when role is `"session"`, ignored otherwise
- `read_only`: when True, opens in read-only mode. Phase 2 curated reads must use `read_only=True`. Phase 3 does not access curated DB directly.
- All connections: WAL mode, foreign keys enabled, `sqlite3.Row` row factory.
- File paths managed automatically under `.clean_room/`.

### 4.3 Ownership Table

| Phase | Curated DB | Raw DB | Session DB |
|-------|-----------|--------|------------|
| Phase 1 | **Creates schema + populates** (indexing) | **Creates schema** + logs indexing run metadata + stores enrichment outputs | **Defines schema only** (no per-task files created) |
| Phase 2 | Read-only (`read_only=True`) | Appends retrieval LLM calls + per-file decisions | **Creates** per-task file, writes pipeline state |
| Phase 3 | No direct access | Appends task runs, attempts, validation results | Inherits from Phase 2, writes retry state, **closes** |
| Phase 4 | Writes adapter_metadata | Appends training plans, datasets, adapter registry | Uses session DB for training pipeline state |

### 4.4 Key Constraints

- Phases 2 and 3 never write to curated DB.
- Phase 4 writes only to `adapter_metadata` in curated DB (verified configuration the pipeline reads at runtime).
- One session DB per task run. Phase 2 creates it and returns an explicit session handle. Phase 3 takes ownership and closes it.
- Task IDs are UUID4. Generated by `cra solve`/`cra plan` at startup, or by `cra retrieve` when run standalone. Passed to Phase 2 which creates the session DB.
- `cra enrich` writes to raw DB (`enrichment_outputs` table). The `--promote` flag copies to curated DB (`file_metadata`). Retrieval works without enrichment -- Scope Tier 4 is skipped and stages rely on their own LLM judgment from deterministic signals.

### 4.5 Curated DB Schema

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
  reference_kind TEXT NOT NULL,
  confidence REAL NOT NULL
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
  model_tag TEXT NOT NULL,           -- Ollama model tag (e.g. "qwen3:4b-scope-v1")
  version INTEGER NOT NULL DEFAULT 1,
  active INTEGER NOT NULL DEFAULT 1, -- 1 = active, 0 = inactive
  performance_notes TEXT,            -- free-form metrics/observations
  deployed_at TEXT NOT NULL
)
```

**Indexes**: `files(repo_id, path)`, `symbols(file_id)`, `symbols(name)`, `symbol_references(caller_symbol_id)`, `symbol_references(callee_symbol_id)`, `dependencies(source_file_id)`, `dependencies(target_file_id)`, `commits(repo_id, hash)`, `file_metadata(domain)`, `file_metadata(module)`, `adapter_metadata(stage_name, active)`.

### 4.6 Raw DB Schema

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
  file_id INTEGER NOT NULL,
  model TEXT NOT NULL,
  purpose TEXT,
  module TEXT,
  domain TEXT,
  concepts TEXT,                  -- JSON
  public_api_surface TEXT,        -- JSON
  complexity_notes TEXT,
  raw_prompt TEXT NOT NULL,
  raw_response TEXT NOT NULL,
  promoted INTEGER NOT NULL DEFAULT 0,
  timestamp TEXT NOT NULL
)

-- Phase 2: LLM calls during retrieval (task analysis + stage judgment calls)
retrieval_llm_calls (
  id INTEGER PRIMARY KEY,
  task_id TEXT NOT NULL,
  call_type TEXT NOT NULL,       -- "task_analysis", "scope_judgment", "precision_judgment", etc.
  stage_name TEXT,               -- NULL for task_analysis, stage name for stage calls
  model TEXT NOT NULL,
  prompt TEXT NOT NULL,
  response TEXT NOT NULL,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  latency_ms INTEGER NOT NULL,
  timestamp TEXT NOT NULL
)

-- Phase 2: per-file retrieval decisions
retrieval_decisions (
  id INTEGER PRIMARY KEY,
  task_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  file_id INTEGER NOT NULL,
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
  mode TEXT NOT NULL,             -- "plan", "implement", "train_plan", "curate_data"
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
  model_tag TEXT NOT NULL,        -- Ollama model tag after training
  training_loss REAL,
  eval_metrics TEXT,              -- JSON: evaluation metrics
  deployed INTEGER NOT NULL DEFAULT 0,
  trained_at TEXT NOT NULL
)
```

### 4.7 Session DB Schema

A single key-value table. The session is ephemeral, never queried externally, and doesn't need relational structure.

```sql
kv (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,            -- JSON-serialized
  updated_at TEXT NOT NULL
)
```

**Helpers**: `set_state(key, value)`, `get_state(key) -> str | None`, `delete_state(key)`, `list_keys(prefix?) -> list[str]`.

**Key contracts** -- see [Section 11: Session Lifecycle](#11-session-lifecycle) for the full key table.

---

## 5. N-Prompt Pipeline

### 5.1 Architecture

The pipeline is mode-parameterized. The core structure is constant:

```
[Task Analysis] -> [Retrieval Stage 1, ..., N] -> [Execute]
```

What stays constant across all modes:
- Pipeline runner (sequences stages, threads budget/session)
- Stage protocol (deterministic pre-filter -> LLM judgment)
- Budget management
- Session lifecycle
- Raw DB logging of every LLM call

What changes per mode:

| Mode | Data Source | Retrieval Stages | Execute Output |
|------|-----------|-----------------|----------------|
| Plan | curated DB (code) | code stages (scope, precision) | structured plan artifact |
| Implement | curated DB + plan | code stages (plan-aware seeding) | code edits (search/replace) |
| Train-Plan | raw DB (logs) | log analysis stages | training plan artifact |
| Curate-Data | raw DB (logs) | data selection stages | training dataset |

### 5.2 Task Analysis

Task analysis is a pipeline preamble, not a `RetrievalStage`. It always runs. Performs deterministic extraction (file/symbol mentions, error patterns, keywords, task type) then an LLM call for intent enrichment. Uses the reasoning model.

### 5.3 Stage Protocol

```python
class RetrievalStage(Protocol):
    name: str

    def run(
        self,
        context: StageContext,
        kb: KnowledgeBase,
        task: TaskQuery,
        llm: LLMClient,
    ) -> StageContext: ...
```

Each stage performs **deterministic pre-filtering** (structured queries against the data source) followed by an **LLM judgment call** (the model evaluates candidates against the task). The deterministic part narrows candidates; the LLM makes the relevance judgment that code alone can't make.

Stages are stateless -- all per-task state lives in `StageContext` (threaded through the pipeline) and session DB (persisted between stages). The pipeline runner handles budget threading, session writes, and raw DB logging between stage calls.

### 5.4 Stage Registry

Stage names (e.g. `"scope"`, `"precision"`) map to `RetrievalStage` implementations via a registry. The pipeline runner validates all names in `--stages` before execution. Unknown names are hard errors. Stage ordering is caller-specified -- the runner does not enforce ordering constraints.

The registry also resolves model overrides: stage name -> override (if exists in `[models.overrides]`) -> role-based default -> base model. See [Section 6.3](#63-model-routing).

### 5.5 MVP Configuration (Phases 2-3)

**Stages**: Scope -> Precision (2 `RetrievalStage` implementations).
**CLI**: `--stages scope,precision`
**Total LLM calls**: 1 (task analysis) + 2 (scope + precision) + 1 (execute) = **4**.

The `--stages` flag controls `RetrievalStage` implementations only. Task analysis runs implicitly before stages.

### 5.6 Scope Stage -- Tiered Expansion

Scope expands outward from seed files in priority tiers:

1. **Seeds** -- files directly mentioned in the task, or containing mentioned symbols. Always included.
2. **Direct dependencies** -- imports/imported-by for seed files via the dependency graph.
3. **Co-change neighbors** -- files that historically change with seeds (above a minimum count).
4. **Metadata matches** -- files sharing domain/module/concepts with the task, via `file_metadata`. Skipped when enrichment hasn't been promoted; stages use LLM judgment from tiers 1-3.

Within each tier, files are ordered by relevance to seeds (dependency depth, co-change count, concept overlap). This is a tie-breaker within the tier, not a cross-tier score.

After deterministic expansion, the scope stage LLM call classifies each candidate as relevant or irrelevant.

**Plan-aware seeding** (implement mode with `--plan`): Plan-identified files become Tier 0 seeds, before the normal tiered expansion. This gives the plan's file list the highest priority.

### 5.7 Precision Stage -- Symbol Extraction

Operates on the scoped file set from the previous stage:

- **Python**: symbol-edge traversal via `symbol_references`, test assertions, rationale comments.
- **TS/JS MVP**: file-level dependencies + symbol-name signals (no symbol-edge data in Phase 1 MVP).

After deterministic extraction, the precision stage LLM call classifies each symbol into detail tiers (`primary`, `supporting`, `type_context`, `excluded`).

---

## 6. LLM Client Architecture

### 6.1 Provider Boundary

`llm/client.py` encapsulates all Ollama-specific HTTP transport (httpx, retry, error handling). Used by all four phases:

- Phase 1: enrichment (`cra enrich`)
- Phase 2: task analysis + stage judgment calls
- Phase 3: code generation + plan generation
- Phase 4: training analysis + data curation

Swapping to a different LLM provider means reimplementing `llm/client.py` internals. No other module changes.

### 6.2 LLMClient Interface

```python
@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int

class LLMClient:
    def __init__(self, config: ModelConfig): ...
    def complete(self, prompt: str, system: str | None = None) -> LLMResponse: ...
```

The `LLMClient` is instantiated per-call with the resolved model config. It is a thin transport layer.

### 6.3 Model Routing

Above the transport layer, a routing layer resolves which model to call based on (role, stage_name):

```python
@dataclass
class ModelConfig:
    model: str            # resolved model tag
    base_url: str         # Ollama URL
    temperature: float    # tuning parameter, reasonable default OK (e.g. 0.0 for code)
    max_tokens: int       # tuning parameter, reasonable default OK

class ModelRouter:
    """Resolves which model to use for a given role and stage."""

    def resolve(self, role: str, stage_name: str | None = None) -> ModelConfig: ...
```

Resolution order:
1. `[models.overrides]` has a key matching `stage_name` -> use that model tag.
2. No stage override -> use `[models].<role>` (coding or reasoning).
3. `[models]` section missing -> hard error directing user to `cra init`.

`ModelRouter` reads from the loaded config. It is constructed once per CLI invocation and passed to the pipeline runner, which passes it to each stage.

### 6.4 Temperature Defaults by Role

- **Coding** (code generation): `temperature = 0.0` (deterministic)
- **Reasoning** (task analysis, scope, precision, plan, enrichment): `temperature = 0.0`

Both default to deterministic. Users can override via config.

---

## 7. Budget Model

### 7.1 BudgetConfig

```python
@dataclass
class BudgetConfig:
    context_window: int       # execute model's total context window size (tokens)
    reserved_tokens: int      # everything except the context package (see below)
```

**Defined once** in a common module. Imported by Phase 2 and Phase 3.

**Effective retrieval budget** = `context_window - reserved_tokens`. Phase 2 fills up to this limit with context.

**What `reserved_tokens` covers**: system prompt + task description + retry context (grows with attempts) + `max_tokens` (completion space). Phase 2's context package plus `reserved_tokens` must not exceed `context_window`.

### 7.2 Mode-Dependent Budget

Budget is calculated against the execute model's context window for the current mode:

| Mode | Execute Model | Budget Window |
|------|--------------|---------------|
| Plan | Qwen3-4B (reasoning) | Qwen3-4B context window |
| Implement | Qwen2.5-Coder-3B (coding) | Qwen2.5-Coder-3B context window |
| Train-Plan | Qwen3-4B (reasoning) | Qwen3-4B context window |
| Curate-Data | Qwen3-4B (reasoning) | Qwen3-4B context window |

Retrieval stage prompts are bounded by the reasoning model's window (separate, smaller concern -- each stage prompt is small).

### 7.3 Token Counting

Character-based approximation (`chars / 4`) for budget checks during context assembly. Exact counts come from Ollama response metadata post-call. The approximation uses a configurable safety margin to account for variance.

### 7.4 Budget Enforcement

- Phase 2 fills the context package up to `context_window - reserved_tokens`.
- Phase 3 verifies the assembled prompt (system + task + context + retry) fits within `context_window` before calling the LLM. This is a hard gate, not advisory.
- Overflow: deterministic trimming of lowest-priority content (retry details first, then supporting context).

### 7.5 Validation Rules

Enforced once in shared code, called by all CLI commands that use budgets:

- `context_window > 0`
- `reserved_tokens >= 0`
- `reserved_tokens < context_window`

---

## 8. Pipeline Modes in Detail

### 8.1 Plan Mode (`cra plan <task>`)

Same retrieval as the coding pipeline (scope, precision). Execute stage produces a structured plan: which files to change, what changes to make, in what order, with rationale. The plan is a persistent artifact (file) that implement mode can consume. Human can review/edit the plan before implementation.

Execute stage uses the reasoning model (Qwen3-4B).

### 8.2 Implement Mode (`cra solve <task>`)

Accepts optional `--plan <artifact>` -- if provided, the plan seeds the retrieval (plan-identified files become Tier 0 seeds) and constrains the execute stage. Without `--plan`, operates as one-shot retrieve -> generate. With `--plan`, the execute stage follows the plan's structure rather than improvising.

Execute stage uses the coding model (Qwen2.5-Coder-3B).

### 8.3 Train-Plan Mode (`cra train-plan`) -- Phase 4

Retrieves from raw DB instead of curated DB -- different query API, different stages. "Log analysis" retrieval stages: identify weak pipeline stages, failure patterns, outcome correlations. Execute stage produces a training plan: which stage to train, what data to select, hyperparameter suggestions, expected improvement targets.

Quality signal: links task outcomes (success/fail, validation results) back to individual stage decisions via `task_id` in `retrieval_llm_calls` joined with `task_runs.success`.

### 8.4 Curate-Data Mode (`cra curate-data --plan <training-plan>`) -- Phase 4

Retrieves from raw DB guided by the training plan. "Data selection" stages: filter logged LLM calls by stage, quality, diversity. Execute stage produces a curated dataset in the format needed for fine-tuning (JSONL of prompt/completion pairs per stage).

Handles: filtering bad examples, balancing positive/negative, deduplication, format conversion.

### 8.5 Writing Training Code

"Write training code" is not a separate mode -- it's Plan + Implement applied to the training infrastructure codebase. The coding pipeline is recursive.

### 8.6 Bootstrapping Mode (External Repos)

A preprocessing pipeline that generates synthetic training data from cloned GitHub repos without requiring any prior agent runs. Workflow:

1. `cra index <cloned-repo>` -- index the external repo normally (Phase 1).
2. Extract and filter commits using criteria from [Section 16.4.2](#1642-commit-filtering-criteria).
3. For each qualifying commit, generate synthetic pipeline run pairs per [Section 16.4.3](#1643-synthetic-pipeline-run-generation): reconstruct what correct scope, precision, and execute outputs should look like from the diff and dependency graph.
4. Store as training datasets with `source = "synthetic"`.

This may be implemented as a dedicated `cra bootstrap <repo-path>` command or as a flag on `cra curate-data`. The key constraint: it depends only on Phase 1 (indexing) and can run before Phases 2-3 exist.

---

## 9. CLI Interface

### 9.1 Commands

| Command | Primary argument | Phase | Requires LLM |
|---------|-----------------|-------|---------------|
| `cra init` | (interactive/flags) | Setup | No |
| `cra index <repo-path>` | repo path (positional) | 1 | No |
| `cra enrich <repo-path>` | repo path (positional) | 1 | Yes |
| `cra retrieve <task>` | task description (positional) | 2 | Yes |
| `cra plan <task>` | task description (positional) | 3 | Yes |
| `cra solve <task>` | task description (positional) | 3 | Yes |
| `cra train-plan` | (no positional) | 4 | Yes |
| `cra curate-data` | (no positional) | 4 | Yes |

### 9.2 Argument Conventions

- **Repo-focused commands** (`index`, `enrich`): repo path is the positional argument.
- **Task-focused commands** (`retrieve`, `plan`, `solve`): task description is the positional argument, `--repo <path>` is a named flag.
- **Training commands** (`train-plan`, `curate-data`): `--repo <path>` is a named flag. `curate-data` requires `--plan <training-plan>`.
- **No `--model` or `--base-url` flags.** All LLM-using commands read model config from `.clean_room/config.toml`.

### 9.3 Required Inputs by Command

| Flag | `index` | `enrich` | `retrieve` | `plan` | `solve` | `train-plan` | `curate-data` |
|------|---------|----------|------------|--------|---------|--------------|---------------|
| `--stages` | -- | -- | required | required | required | required | required |
| budget (see below) | -- | -- | required | required | required | required | required |
| `--plan` | -- | -- | -- | -- | optional | -- | required |
| `--max-attempts` | -- | -- | -- | -- | required | -- | -- |
| `--max-refinement-loops` | -- | -- | -- | -- | required | -- | -- |

**Budget input**: either `--context-window <int>` + `--reserved-tokens <int>`, or `--budget-config <path>`. Mutually exclusive. Missing budget is a hard error.

### 9.4 `--budget-config` File Format

TOML file:

```toml
context_window = 32768
reserved_tokens = 4096
```

No section header required. Just the two keys.

---

## 10. Config File

### 10.1 Location and Format

`.clean_room/config.toml`, created by `cra init`.

```toml
[models]
coding = "qwen2.5-coder:3b-instruct"
reasoning = "qwen3:4b-instruct-2507"
base_url = "http://localhost:11434"

[models.overrides]
# scope = "qwen3:4b-scope-v1"

[budget]
context_window = 32768
reserved_tokens = 4096

[solve]
max_attempts = 3
max_refinement_loops = 2

[stages]
default = "scope,precision"
```

`cra init` creates this interactively or from explicit flags. It does not populate values the user does not provide.

### 10.2 Resolution Order

For every CLI input:

1. CLI flag present -> use it.
2. CLI flag absent, config.toml has the value -> use config value.
3. Neither -> **hard error** for required values.

**Code never has hardcoded defaults for required values.** Config is a convenience to avoid retyping the same flags. It is not a silent fallback -- it is an explicit value source checked at the CLI layer.

**Model config**: there are no CLI flags for model selection. Models are always read from config.toml. Missing `[models]` section -> hard error.

### 10.3 Config Loader

`config.py`: reads TOML, returns a flat dict. Missing file returns `None` (not an error -- config is optional for `cra index`). Missing `[models]` section when an LLM command runs is a hard error at the CLI layer.

---

## 11. Session Lifecycle

### 11.1 Flow

1. **Create**: Phase 2 pipeline runner creates session DB via `get_connection('session', task_id=...)`.
2. **Phase 2 writes**: pipeline state (stage outputs, progress, final context).
3. **Handoff**: Phase 2 returns `ContextPackage` + session handle to caller.
4. **Phase 3 inherits**: writes retry context, refinement requests.
5. **Close**: Phase 3 closes the DB connection after the task run completes.
6. **Archive** (optional): read closed session file as raw bytes, insert into raw DB `session_archives` as BLOB.
7. **Delete**: remove the session SQLite file.

### 11.2 Session DB Key Contracts

| Key | Writer | Reader | Content |
|-----|--------|--------|---------|
| `stage_outputs` | Phase 2 | Phase 2 (re-entry) | Map of stage_name -> serialized stage output |
| `stage_progress` | Phase 2 | Phase 2 (re-entry) | Ordered list of completed stages + status |
| `final_context` | Phase 2 | Phase 2 (re-entry), Phase 3 | Serialized final `StageContext` |
| `refinement_request` | Phase 3 | Phase 2 (re-entry) | Serialized `RefinementRequest` |
| `attempt_summary` | Phase 3 | Phase 2 (logging) | Summary of attempts leading to refinement |
| `retry_context` | Phase 3 | Phase 3 (prompt builder) | Error classifications, attempt history for retry prompts |
| `plan_artifact` | Phase 3 (plan mode) | Phase 3 (implement mode via file) | Serialized plan output |

---

## 12. Query API

`KnowledgeBase` class in `query/api.py`. Reads exclusively from the curated DB. Constructor takes an existing curated DB connection (caller obtains it via `get_connection('curated', read_only=True)`).

**Methods** (the contract Phase 2 depends on):

- `get_files(repo_id, language?)`, `get_file_by_path(repo_id, path)`
- `search_files_by_metadata(repo_id, domain?, module?, concepts?)` -- returns empty when `file_metadata` is unpopulated (not an error)
- `get_symbols_for_file(file_id, kind?)`, `search_symbols_by_name(repo_id, pattern)`
- `get_symbol_neighbors(symbol_id, direction, kinds?)` -- Python MVP only
- `get_dependencies(file_id, direction)`, `get_dependency_subgraph(file_ids, depth)`
- `get_co_change_neighbors(file_id, min_count)`
- `get_docstrings_for_file(file_id)`, `get_rationale_comments(file_id)`
- `get_recent_commits_for_file(file_id, limit)`
- `get_file_context(file_id)` -- composite: symbols + docstrings + rationale comments + deps + co-changes + commits
- `get_repo_overview(repo_id)` -- file counts, domain distribution, most-connected files
- `get_adapter_for_stage(stage_name)` -- returns active adapter metadata for a stage, or None

**Phase 4 addition**: `LogAnalysisAPI` in a separate module. Reads from raw DB. Provides methods for querying logged LLM calls, retrieval decisions, task outcomes, and cross-referencing them. This is the raw DB equivalent of `KnowledgeBase`.

---

## 13. Refinement Protocol

When Phase 3 determines context is insufficient during the retry loop:

1. Phase 3 classifies the attempt failure as `insufficient_context` with concrete evidence (missing files, symbols, tests, error signatures).
2. Phase 3 writes `refinement_request` (serialized `RefinementRequest`) and `attempt_summary` to session DB.
3. Phase 3 calls Phase 2 re-entry with the same `task_id` and session handle.
4. Phase 2 reads `final_context` and `refinement_request` from session DB.
5. Phase 2 expands context to cover missing items, re-runs stages as needed, logs new decisions to raw DB.
6. Phase 2 writes updated `final_context` and `stage_progress` back to session DB.
7. Phase 2 returns a new `ContextPackage`.
8. Phase 3 continues retries with the updated context.

### Bounds

- `max_refinement_loops` from caller config (required, no default in code).
- Each `RefinementRequest` requires concrete evidence -- not just "needs more context."
- Phase 3 never opens curated DB connections. All context expansion goes through Phase 2.

### RefinementRequest

```python
@dataclass
class RefinementRequest:
    reason: str
    missing_files: list[str]
    missing_symbols: list[str]
    missing_tests: list[str]
    error_signatures: list[str]
```

Defined in a common retrieval module (Phase 2 owns the contract). Phase 3 imports it.

---

## 14. Preflight Checks

The retrieval pipeline runner performs these checks before executing stages:

1. **Model config exists**: checks that `[models]` section is present in config. Missing -> hard error with message directing user to `cra init`.
2. **Curated DB exists and is populated** (code modes): checks that `curated.sqlite` exists and the `files` table has rows. Missing or empty -> hard error with message directing user to run `cra index`.
3. **Raw DB exists and is populated** (training modes): checks that `raw.sqlite` exists and has logged data. Missing or empty -> hard error with message explaining that training modes require prior coding runs.
4. **Enrichment status**: checks whether `file_metadata` has rows. Absent -> info log ("enrichment not found, Tier 4 skipped"), not an error. Stages proceed with deterministic signals and their own LLM judgment.

These checks run in all `cra` command paths that use the pipeline.

---

## 15. Error Handling

- **Missing required inputs**: hard error with message listing what's missing and how to provide it.
- **Missing model config**: hard error directing user to `cra init`.
- **Parse failures** (AST, response parsing): raise with full context (raw content included). During indexing, `--continue-on-error` flag logs failures and continues; default is fail-fast.
- **LLM call failures**: `llm/client.py` handles transport-level retries (connection errors, timeouts). Application-level failures (malformed JSON from enrichment, incoherent stage response) propagate up with raw response included.
- **Budget overflow**: deterministic trimming at assembly time. Hard gate before LLM call in Phase 3.
- **Missing enrichment**: info log, Tier 4 skipped. Not an error.
- **Missing curated DB**: hard error with clear message.
- **Wrong mode for data state**: e.g. `cra train-plan` with no logged runs -> hard error explaining dependency.

---

## 16. LoRA Integration Strategy (Phase 4)

### 16.1 Per-Stage Model Overrides

Each pipeline stage can specify a model name override via `[models.overrides]` in config. With Ollama, each LoRA is a separate model tag (e.g., `qwen3:4b-scope-v1`, `qwen2.5-coder:3b-code-v2`). The stage registry resolves: stage name -> override (if exists) -> role-based default -> base model. No special LoRA-loading code needed -- Ollama handles adapter management natively.

### 16.2 Raw DB as Per-Stage Training Corpus

Each stage's training data can come from two sources:

**Logged runs (self-improvement)**: Extracted from `retrieval_llm_calls` filtered by:
- `call_type` (which stage produced it)
- linked `task_runs.success` (was the overall task successful?)
- optionally `retrieval_decisions` (was this specific decision part of a successful run?)

**Synthetic pairs (bootstrapping)**: Generated from external repo commit histories via the bootstrapping pipeline ([Section 16.4](#164-bootstrapping-from-external-repo-history)). These provide training data before the agent has produced any real runs.

Both sources produce datasets in the same format and are stored with a `source` field to allow mixing and weighting during training.

### 16.3 Training Artifact Storage

- Training plans and curated datasets go in raw DB (they're generated outputs, same as enrichment outputs).
- Adapter metadata (which stage, performance metrics, active/inactive) goes in curated DB (it's verified configuration the pipeline reads at runtime).

### 16.4 Bootstrapping from External Repo History

The self-improvement loop has a cold-start problem: Phase 4 training needs quality-signal-rich data, but Phase 3's output quality depends on models that haven't been trained yet. External repos break this dependency.

#### 16.4.1 Data Source

Clone mature, well-tested repos from GitHub. Ideal repos have:
- Clear commit messages (natural task descriptions).
- Small, focused bug-fix and feature commits (clean before/after pairs).
- Stable dependency graphs and well-established symbol structure (high-quality indexer output).
- Passing CI (implicit quality signal: the fix stuck, tests passed, it got merged).

#### 16.4.2 Commit Filtering Criteria

Not every commit is a useful training example. Filter for:
- **Diff size bounds**: single-file or small-cluster changes (e.g. 1-5 files, < 200 lines changed). Massive refactors are noise.
- **Descriptive messages**: commits with meaningful messages serve as synthetic task descriptions. Skip "fix", "wip", "misc" commits.
- **Non-merge commits**: merge commits conflate multiple logical changes.
- **Language match**: filter to Python/TS/JS files matching the indexer's current capability.

#### 16.4.3 Synthetic Pipeline Run Generation

From a filtered commit, reverse-engineer what a correct pipeline run *should have* looked like:

| Stage | Synthetic Training Pair | Derivation |
|-------|------------------------|------------|
| Task Analysis | commit message -> structured TaskQuery | The commit message is the task; the diff tells you what files/symbols were relevant. |
| Scope | task + repo files -> file relevance classification | Files changed in the diff are "relevant"; their direct dependencies are "supporting"; unrelated files are "irrelevant". |
| Precision | task + file symbols -> symbol tier classification | Symbols modified in the diff are "primary"; symbols referenced by modified code are "supporting"; others are "excluded". |
| Execute (Code) | context + task -> code edits | The diff itself is the target output. Context is the pre-commit state of relevant files. |
| Execute (Plan) | context + task -> plan | Derive a plan from the diff: which files to change, what changes, in what order. |

This is a distinct data curation step from `cra curate-data` (which curates from the agent's own logged runs). It may be implemented as a preprocessing pipeline or as an additional mode.

#### 16.4.4 Quality Signal Mapping

External repo commits provide quality signals analogous to what the self-improvement loop gets from logged runs:

| Self-Improvement Signal | External Repo Equivalent |
|------------------------|--------------------------|
| `task_runs.success` | commit was merged / tests passed |
| retrieval decision correctness | did the diff touch the files scope would have found? |
| symbol selection correctness | did the diff modify the symbols precision would have selected? |
| code generation correctness | does the generated diff match the actual commit diff? |

#### 16.4.5 Storage

Synthetic training pairs from external repos are stored in the same raw DB tables as self-improvement training data (`training_datasets`, `training_plans`). A `source` field distinguishes them:

- `source = "synthetic"`: generated from external repo commit history.
- `source = "logged"`: curated from the agent's own logged runs.

This allows training to mix both data sources as the agent matures.

---

## 17. Platform Notes

- **Development platform**: Windows 11, Python 3.11+.
- **Git worktree**: Phase 3 uses `git worktree add/remove` for patch isolation. Platform verification (especially Windows long paths and cleanup) is a blocking prerequisite before patch application code.
- **No silent fallback**: if worktrees fail, error with diagnostics. Do not silently degrade to in-place patching.
- **All LLM calls are local**: Ollama on localhost, no data leaves the machine.

---

## 18. Data Structures Summary

Core data structures defined in shared modules, used across phases:

| Structure | Defined by | Used by | Location |
|-----------|-----------|---------|----------|
| `ModelConfig` | Phase 1 | All phases | `llm/client.py` |
| `ModelRouter` | Phase 1 | All phases | `llm/router.py` |
| `LLMClient` | Phase 1 | All phases | `llm/client.py` |
| `LLMResponse` | Phase 1 | All phases | `llm/client.py` |
| `BudgetConfig` | Phase 2 | Phase 2, Phase 3 | `retrieval/dataclasses.py` |
| `TaskQuery` | Phase 2 | Phase 2 | `retrieval/dataclasses.py` |
| `StageContext` | Phase 2 | Phase 2 | `retrieval/stage.py` |
| `ContextPackage` | Phase 2 | Phase 2, Phase 3 | `retrieval/dataclasses.py` |
| `RefinementRequest` | Phase 2 | Phase 2, Phase 3 | `retrieval/dataclasses.py` |
| `EditBlock` | Phase 3 | Phase 3 | `agent/dataclasses.py` |
| `PlanArtifact` | Phase 3 | Phase 3 | `agent/dataclasses.py` |
| `GenerationResult` | Phase 3 | Phase 3 | `agent/dataclasses.py` |
| `ValidationResult` | Phase 3 | Phase 3 | `agent/dataclasses.py` |
| `AttemptRecord` | Phase 3 | Phase 3 | `agent/dataclasses.py` |
| `TaskResult` | Phase 3 | Phase 3 | `agent/dataclasses.py` |
| `TrainingPlan` | Phase 4 | Phase 4 | `training/dataclasses.py` |
| `TrainingDataset` | Phase 4 | Phase 4 | `training/dataclasses.py` |
| `AdapterRecord` | Phase 4 | Phase 4 | `training/dataclasses.py` |

---

## 19. MVP Boundary Checklist

- **Must**: Python symbol-level reference edges (`symbol_references`) in Phase 1 indexing.
- **Must**: Python symbol neighbor traversal in Query API.
- **Must**: File-level dependency graph for Python/TS/JS.
- **Must**: Indexing works without any LLM dependency (`cra index` is deterministic).
- **Must**: Search-and-replace as the single output format for code generation.
- **Must**: Plan mode produces structured plan artifacts consumable by implement mode.
- **Must**: Multi-model routing via config (coding + reasoning models).
- **Must Not**: TS/JS symbol-level call/reference edges in Phase 1 MVP.
- **Post-MVP**: TS/JS symbol-edge extraction. Additional retrieval stages. Phase 4 (self-improvement loop + bootstrapping from external repos). Baseline comparison mode.
