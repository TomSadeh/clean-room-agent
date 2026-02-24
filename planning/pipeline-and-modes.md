# Pipeline Architecture and Modes

Retrieval stage details, LLM client architecture, budget model, all pipeline modes (including the solve orchestrator), session lifecycle, query API, refinement protocol, and preflight checks.

---

## 1. Scope Stage — Tiered Expansion

Scope expands outward from seed files in priority tiers:

1. **Seeds** -- files directly mentioned in the task, or containing mentioned symbols. Always included.
2. **Direct dependencies** -- imports/imported-by for seed files via the dependency graph.
3. **Co-change neighbors** -- files that historically change with seeds (above a minimum count).
4. **Metadata matches** -- files sharing domain/module/concepts with the task, via `file_metadata`. Skipped when enrichment hasn't been promoted; stages use LLM judgment from tiers 1-3.

Within each tier, files are ordered by relevance to seeds (dependency depth, co-change count, concept overlap). This is a tie-breaker within the tier, not a cross-tier score.

After deterministic expansion, the scope stage LLM call classifies each candidate as relevant or irrelevant.

**Plan-aware seeding** (implement mode with `--plan`): Plan-identified files become Tier 0 seeds, before the normal tiered expansion. This gives the plan's file list the highest priority.

---

## 2. Precision Stage — Symbol Extraction

Operates on the scoped file set from the previous stage:

- **Python**: symbol-edge traversal via `symbol_references`, test assertions, rationale comments.
- **TS/JS MVP**: file-level dependencies + symbol-name signals (no symbol-edge data in Phase 1 MVP).

After deterministic extraction, the precision stage LLM call classifies each symbol into detail tiers (`primary`, `supporting`, `type_context`, `excluded`).

---

## 3. LLM Client Architecture

### 3.1 Provider Boundary

`llm/client.py` encapsulates all provider-specific HTTP transport (httpx, retry, error handling). Three provider backends are needed across the project lifecycle:

1. **Ollama** (Phases 1-3 base): Local inference for single-model operation. The MVP provider.
2. **OpenAI-compatible** (Phase 4 multi-adapter): vLLM or llama-server for per-request LoRA adapter selection. Same `complete()` interface, different transport internals.
3. **Remote API** (Phase 4 teacher inference): External API calls for distillation data generation (Qwen3.5-Plus, DeepSeek-V3.2). Used only during training data creation, not at runtime.

`ModelConfig` will gain a `provider` field (default `"ollama"`) to select the backend. The external `complete()` interface does not change — provider selection is internal to the transport layer. All phases call through the same `LLMClient.complete()` regardless of backend.

Used by all four phases:

- Phase 1: enrichment (`cra enrich`)
- Phase 2: task analysis + stage judgment calls
- Phase 3: code generation + plan generation
- Phase 4: training analysis + data curation + distillation data generation (remote API)

### 3.2 LLMClient Interface

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

### 3.3 Model Routing

Above the transport layer, a routing layer resolves which model to call based on (role, stage_name):

```python
@dataclass
class ModelConfig:
    model: str            # resolved model/adapter identifier
    base_url: str         # provider URL
    provider: str         # "ollama", "openai_compat", or "remote_api"
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

### 3.4 Temperature Defaults by Role

- **Coding** (code generation): `temperature = 0.0` (deterministic)
- **Reasoning** (task analysis, scope, precision, plan, enrichment): `temperature = 0.0`

Both default to deterministic. Users can override via config.

---

## 4. Budget Model

### 4.1 BudgetConfig

```python
@dataclass
class BudgetConfig:
    context_window: int       # execute model's total context window size (tokens)
    reserved_tokens: int      # everything except the context package (see below)
```

**Defined once** in a common module. Imported by Phase 2 and Phase 3.

**Effective retrieval budget** = `context_window - reserved_tokens`. Phase 2 fills up to this limit with context.

**What `reserved_tokens` covers**: system prompt + task description + retry context (grows with attempts) + `max_tokens` (completion space). Phase 2's context package plus `reserved_tokens` must not exceed `context_window`.

### 4.2 Mode-Dependent Budget

Budget is calculated against the execute model's context window for the current mode:

| Mode | Execute Model | Budget Window |
|------|--------------|---------------|
| Plan | Qwen3-4B (reasoning) | Qwen3-4B context window |
| Implement | Qwen2.5-Coder-3B (coding) | Qwen2.5-Coder-3B context window |
| Train-Plan | Qwen3-4B (reasoning) | Qwen3-4B context window |
| Curate-Data | Qwen3-4B (reasoning) | Qwen3-4B context window |

Retrieval stage prompts are bounded by the reasoning model's window (separate, smaller concern -- each stage prompt is small).

### 4.3 Token Counting

Character-based approximation (`chars / 4`) for budget checks during context assembly. Exact counts come from the provider's response metadata post-call (Ollama: `prompt_eval_count`/`eval_count`; OpenAI-compatible: `usage.prompt_tokens`/`usage.completion_tokens`). The approximation uses a configurable safety margin to account for variance.

### 4.4 Budget Enforcement

- Phase 2 fills the context package up to `context_window - reserved_tokens`.
- Phase 3 verifies the assembled prompt (system + task + context + retry) fits within `context_window` before calling the LLM. This is a hard gate, not advisory.
- Overflow: deterministic trimming of lowest-priority content (retry details first, then supporting context).

### 4.5 Validation Rules

Enforced once in shared code, called by all CLI commands that use budgets:

- `context_window > 0`
- `reserved_tokens >= 0`
- `reserved_tokens < context_window`

---

## 5. Pipeline Modes in Detail

### 5.1 Plan Mode (`cra plan <task>`)

Same retrieval as the coding pipeline (scope, precision). Execute stage produces a structured plan: which files to change, what changes to make, in what order, with rationale. The plan is a persistent artifact (file) that implement mode can consume. Human can review/edit the plan before implementation.

Execute stage uses the reasoning model (Qwen3-4B).

**Atomic mode**: Plan mode is a single pipeline pass (task analysis → retrieval → execute). The solve orchestrator ([Section 5.7](#57-solve-orchestrator--recursive-planning-pipeline)) composes multiple atomic plan and implement passes to handle multi-part tasks.

#### 5.1.1 Plan Schema

Plans are structured JSON artifacts, not free-form text. Machine-parseable for validation and implement mode consumption. No actual code in the plan — that is the execute stage's job in implement mode.

```json
{
  "task_summary": "string — one-sentence description of intent",
  "affected_files": [
    {
      "path": "src/module/file.py",
      "role": "modify | create | delete",
      "changes": [
        {
          "symbol": "ClassName.method_name",
          "action": "modify | add | delete | rename",
          "description": "what changes and why",
          "depends_on": ["other_file.py:OtherClass.method"],
          "depended_by": []
        }
      ]
    }
  ],
  "execution_order": [
    "src/module/types.py",
    "src/module/file.py",
    "tests/test_file.py"
  ],
  "rationale": "string — high-level reasoning for the approach"
}
```

**Design principles**:
- **Explicit dependency edges are mandatory** (informed by CodePlan, FSE 2024). Each change declares what it depends on and what depends on it, enabling correct ordering and incremental re-planning.
- **No code in plans**. Plans specify *what* to change and *why*, not *how* at the code level. Code generation is the implement mode execute stage's responsibility.
- **Must be machine-parseable**. Implement mode consumes plan JSON programmatically for Tier 0 seeding and execute stage constraint.

The `PlanArtifact` data structure wraps this schema with metadata (task_id, timestamp, generating model).

### 5.2 Implement Mode (`cra solve <task>`)

Accepts optional `--plan <artifact>` -- if provided, the plan seeds the retrieval (plan-identified files become Tier 0 seeds) and constrains the execute stage, running a single atomic implement pass. With `--plan`, the execute stage follows the plan's structure rather than improvising.

Without `--plan`, `cra solve` invokes the solve orchestrator ([Section 5.7](#57-solve-orchestrator--recursive-planning-pipeline)), which decomposes the task and composes multiple atomic plan and implement passes.

Execute stage uses the coding model (Qwen2.5-Coder-3B).

**Atomic mode**: Implement mode is a single pipeline pass (task analysis → retrieval → execute). When invoked directly via `--plan`, it bypasses the orchestrator entirely.

### 5.3 Train-Plan Mode (`cra train-plan`) — Phase 4

Retrieves from raw DB instead of curated DB -- different query API, different stages. "Log analysis" retrieval stages: identify weak pipeline stages, failure patterns, outcome correlations. Execute stage produces a training plan: which stage to train, what data to select, hyperparameter suggestions, expected improvement targets.

Quality signal: links task outcomes (success/fail, validation results) back to individual stage decisions via `task_id` in `retrieval_llm_calls` joined with `task_runs.success`.

### 5.4 Curate-Data Mode (`cra curate-data --training-plan <artifact>`) — Phase 4

Retrieves from raw DB guided by the training plan. "Data selection" stages: filter logged LLM calls by stage, quality, diversity. Execute stage produces a curated dataset in the format needed for fine-tuning (JSONL of prompt/completion pairs per stage).

Handles: filtering bad examples, balancing positive/negative, deduplication, format conversion.

### 5.5 Writing Training Code

"Write training code" is not a separate mode -- it's Plan + Implement applied to the training infrastructure codebase. The coding pipeline is recursive.

### 5.6 Bootstrapping Mode (External Repos)

A preprocessing pipeline that generates synthetic training data from cloned GitHub repos without requiring any prior agent runs. Workflow:

1. `cra index <cloned-repo>` -- index the external repo normally (Phase 1).
2. Extract and filter commits using criteria from the commit filtering spec (see [training-strategy.md](training-strategy.md)).
3. For each qualifying commit, generate synthetic pipeline run pairs: reconstruct what correct scope, precision, and execute outputs should look like from the diff and dependency graph.
4. Store as training datasets with `source = "synthetic"`.

This may be implemented as a dedicated `cra bootstrap <repo-path>` command or as a flag on `cra curate-data`. The key constraint: it depends only on Phase 1 (indexing) and can run before Phases 2-3 exist.

### 5.7 Solve Orchestrator — Recursive Planning Pipeline

`cra solve` (without `--plan`) invokes a solve orchestrator that composes multiple atomic plan and implement mode passes into a multi-part task execution. The orchestrator is deterministic logic — no LLM calls, no model autonomy. It sequences passes, routes outputs between them, and manages progress state.

#### 5.7.1 Pass Hierarchy

The orchestrator decomposes work through four types of atomic pipeline passes, each of which is a complete N-prompt pipeline invocation (task analysis → retrieval → execute):

1. **Meta-plan pass** (plan mode): Decomposes the full task into logical parts with dependency ordering. Input: original task description. Output: `MetaPlan` with ordered `MetaPlanPart` entries.

2. **Part-plan pass** (plan mode, per part): Produces a step-by-step plan for one part, with context from completed parts. Input: part description + cumulative diff from prior parts. Output: `PartPlan` with ordered `PlanStep` entries.

3. **Step implementation pass** (implement mode, per step): Executes one step with its plan as context. Input: step description + part plan + cumulative diff. Output: `StepResult` with the applied diff.

4. **Adjustment pass** (plan mode, after each step): Revises remaining steps given implementation results. Input: original part plan + step results so far + cumulative diff. Output: `PlanAdjustment` with revised remaining steps.

#### 5.7.2 Execution Flow

```
for each part in meta_plan.parts (dependency order):
    part_plan = run_plan_pass(part, cumulative_diff)
    for each step in part_plan.steps:
        step_result = run_implement_pass(step, part_plan, cumulative_diff)
        cumulative_diff += step_result.diff
        adjustment = run_plan_pass(remaining_steps, step_result, cumulative_diff)
        part_plan.steps = adjustment.revised_steps
```

The orchestrator makes all control flow decisions. The model sees only one pass at a time — it has no awareness of the orchestrator, no ability to request more passes, and no memory between passes beyond what the orchestrator explicitly provides as context.

#### 5.7.3 Design Principles

- **No model autonomy**: The orchestrator controls all sequencing, routing, and termination. Models produce outputs; the orchestrator iterates over them.
- **Each pass is stateless**: From the model's perspective, every pass is a fresh pipeline invocation with curated context. Cross-pass continuity comes from the orchestrator feeding previous outputs as context, not from model memory.
- **Plans are data, not instructions**: The orchestrator iterates over `MetaPlan` and `PartPlan` as data structures. The model never "follows a plan" — it receives a plan as context for generating output.
- **Adjustment always runs**: After every step implementation, the adjustment pass runs unconditionally. If nothing changed, the model outputs the same remaining steps. This is a function call, not a decision.
- **Partial success is an explicit outcome**: If a step fails and retries are exhausted, the orchestrator records what completed and what didn't. It does not retry the entire task or silently discard progress.
- **Natural degradation**: Single-part or single-step tasks flow through the same orchestrator — a meta-plan with one part, a part-plan with one step. No separate "simple mode."

#### 5.7.4 Data Structures

Defined in `agent/dataclasses.py`:

| Structure | Content |
|-----------|---------|
| `MetaPlan` | Task decomposition: ordered list of `MetaPlanPart` entries with dependency edges |
| `MetaPlanPart` | One logical part: description, target files, dependencies on other parts |
| `PartPlan` | Step-by-step plan for one part: ordered list of `PlanStep` entries |
| `PlanStep` | One atomic implementation step: description, target symbol/file, expected outcome |
| `StepResult` | Outcome of one step implementation pass: success/failure, applied diff, error info |
| `PlanAdjustment` | Revised remaining steps after an implementation: rationale, added/removed/modified steps |

#### 5.7.5 Session DB Keys

See [schemas.md](schemas.md) Section 4 for the full session key contracts table including orchestrator keys (`meta_plan`, `part_plan:<part_id>`, `step_result:<part_id>:<step_id>`, `adjustment:<part_id>:<after_step_id>`, `orchestrator_progress`, `cumulative_diff`).

#### 5.7.6 Raw DB Tables

Two tables link orchestrator runs to the underlying atomic pipeline passes:

- **`orchestrator_runs`**: One row per `cra solve` invocation (without `--plan`). Tracks the task, total parts/steps, final status (complete, partial, failed), and timestamps.
- **`orchestrator_passes`**: Links an orchestrator run to its constituent `task_runs` entries. Each row records the pass type (`meta_plan`, `part_plan`, `step_implement`, `adjustment`), sequence order, and the `task_run_id` of the underlying pipeline pass.

See [schemas.md](schemas.md) Section 2 for full DDL.

#### 5.7.7 Training Data

Each pass type produces distinct training pairs with different quality signals:

| Pass Type | Training Pair | Quality Signal |
|-----------|---------------|----------------|
| Meta-plan | task → decomposition | did all parts succeed? was the decomposition the right granularity? |
| Part-plan | part + context → steps | did the steps lead to successful implementation? |
| Step implementation | step + plan → code | did the code pass validation? |
| Adjustment | plan + results → revised plan | did revised steps succeed better than the original would have? |

Adjustment data is particularly valuable — it captures what plans look like after encountering implementation reality, providing natural preference pairs for DPO training (original plan vs. adjusted plan, with step outcome as the quality signal).

#### 5.7.8 CLI Integration

- `cra solve <task>`: Runs the full orchestrator (meta-plan → part plans → step implementations → adjustments).
- `cra solve <task> --plan <artifact>`: Bypasses the orchestrator. Runs a single atomic implement pass with the provided plan.
- `cra solve <task> --meta-plan <artifact>`: Skips the meta-plan pass, uses the provided decomposition. The orchestrator continues from the part-plan stage. Mutually exclusive with `--plan`.

#### 5.7.9 Configuration

```toml
[orchestrator]
max_parts = 10                    # maximum parts in a meta-plan decomposition
max_steps_per_part = 15           # maximum steps in a part-plan
max_adjustment_rounds = 3         # adjustment passes per step (usually 1, but allows retry)
```

Defaults are enforced in config — no hardcoded values in orchestrator logic.

---

## 6. Session Lifecycle

### 6.1 Flow

1. **Create**: Phase 2 pipeline runner creates session DB via `get_connection('session', task_id=...)`.
2. **Phase 2 writes**: pipeline state (stage outputs, progress, final context).
3. **Handoff**: Phase 2 returns `ContextPackage` + session handle to caller.
4. **Phase 3 inherits**: writes retry context, refinement requests.
5. **Close**: Phase 3 closes the DB connection after the task run completes.
6. **Archive** (optional): read closed session file as raw bytes, insert into raw DB `session_archives` as BLOB.
7. **Delete**: remove the session SQLite file.

---

## 7. Query API

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

## 8. Refinement Protocol

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

## 9. Preflight Checks

The retrieval pipeline runner performs these checks before executing stages:

1. **Model config exists**: checks that `[models]` section is present in config. Missing -> hard error with message directing user to `cra init`.
2. **Curated DB exists and is populated** (code modes): checks that `curated.sqlite` exists and the `files` table has rows. Missing or empty -> hard error with message directing user to run `cra index`.
3. **Raw DB exists and is populated** (training modes): checks that `raw.sqlite` exists and has logged data. Missing or empty -> hard error with message explaining that training modes require prior coding runs.
4. **Enrichment status**: checks whether `file_metadata` has rows. Absent -> info log ("enrichment not found, Tier 4 skipped"), not an error. Stages proceed with deterministic signals and their own LLM judgment.

These checks run in all `cra` command paths that use the pipeline.
