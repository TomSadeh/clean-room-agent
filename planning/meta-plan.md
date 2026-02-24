# Clean Room Agent Meta-Plan

Single source of truth for phase boundaries, cross-phase contracts, pipeline protocol, and all architectural decisions. Detailed specifications are split into topic files:

- [schemas.md](schemas.md) -- Full DDL for all three databases + session key contracts
- [cli-and-config.md](cli-and-config.md) -- CLI commands, arguments, config file format, resolution order
- [pipeline-and-modes.md](pipeline-and-modes.md) -- Retrieval stages, LLM client, budget model, all pipeline modes, session lifecycle, query API, refinement, preflight checks
- [training-strategy.md](training-strategy.md) -- LoRA training targets, bootstrapping, distillation, self-improvement guardrails

Phase-specific implementation plans derive from this document and must not contradict it.

---

## 1. Vision: The Self-Improving Loop

The agent is a self-improving system. The same N-prompt pipeline architecture (analyze task, curate context, execute) serves multiple modes -- planning, coding, training planning, data curation -- creating a closed loop where the agent improves itself from its own logged activity.

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
- **Bootstrapping (external repos + open datasets)**: Clone mature, well-tested repos from GitHub. Extract commit histories as mistake -> solution pairs. Generate synthetic pipeline run data (what scope/precision *should have* retrieved for a given commit's changes). Complement with open cold-start datasets (OpenCodeInstruct, CommitPackFT, SWE-bench, SRI Tuning -- see [training-strategy.md](training-strategy.md) Section 5.6). Both sources depend only on Phase 1 and do not require prior agent runs.
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

**Escape hatch**: If Qwen3-4B proves insufficient for planning (Stage 4), the per-stage architecture allows upgrading that single stage to Qwen3-8B. Only one adapter retrains; all other stages are unaffected.

### 3.2 Model Routing is Config-Only

No `--model` or `--base-url` CLI flags. All model configuration lives in `.clean_room/config.toml`, set up by `cra init`. Missing `[models]` section when an LLM-using command runs -> hard error directing user to `cra init`.

See [cli-and-config.md](cli-and-config.md) Section 5 for the full config file format, resolution order, and TOML example.

### 3.3 Budget Depends on Execute Model

Budget is calculated against the execute model's context window for the current mode. Plan mode budgets against Qwen3-4B; implement mode budgets against Qwen2.5-Coder-3B.

See [pipeline-and-modes.md](pipeline-and-modes.md) Section 4 for the full budget model.

### 3.4 LoRA Training Targets

See [training-strategy.md](training-strategy.md) Section 1 for per-stage training targets, techniques, and rank recommendations for both base models.

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
get_connection(role, *, repo_path, task_id=None, read_only=False)
```

- `role`: `"curated"`, `"raw"`, or `"session"`
- `repo_path`: keyword-only, path to the repository root (locates `.clean_room/` directory)
- `task_id`: required when role is `"session"`, ignored otherwise
- `read_only`: when True, opens in read-only mode. Phase 2 curated reads must use `read_only=True`. Phase 3 does not access curated DB directly.
- All connections: WAL mode, foreign keys enabled, `sqlite3.Row` row factory.

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

### 4.5-4.7 Database Schemas

See [schemas.md](schemas.md) for full DDL of all three databases (curated: 13 tables, raw: 13 tables, session: 1 KV table) plus indexes and session DB key contracts.

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

Stages are stateless -- all per-task state lives in `StageContext` (threaded through the pipeline) and session DB (persisted between stages). The pipeline runner handles model resolution (using `ModelRouter` to resolve `stage.name` -> `LLMClient`), budget threading, session writes, and raw DB logging between stage calls. Stages receive a pre-resolved `LLMClient` and do not make routing decisions.

### 5.4 Stage Registry

Stage names (e.g. `"scope"`, `"precision"`) map to `RetrievalStage` implementations via a registry. The pipeline runner validates all names in `--stages` before execution. Unknown names are hard errors. Stage ordering is caller-specified -- the runner does not enforce ordering constraints.

The registry also resolves model overrides: stage name -> override (if exists in `[models.overrides]`) -> role-based default -> base model. See [pipeline-and-modes.md](pipeline-and-modes.md) Section 3.3 for model routing details.

### 5.5 MVP Configuration (Phases 2-3)

**Stages**: Scope -> Precision (2 `RetrievalStage` implementations).
**CLI**: `--stages scope,precision`
**Total LLM calls**: 1 (task analysis) + 2 (scope + precision) + 1 (execute) = **4**.

The `--stages` flag controls `RetrievalStage` implementations only. Task analysis runs implicitly before stages.

### 5.6-5.7 Retrieval Stage Details

See [pipeline-and-modes.md](pipeline-and-modes.md) Sections 1-2 for the Scope stage (tiered expansion, plan-aware seeding) and Precision stage (symbol extraction, detail tiers).

---

## 6-8. LLM Client, Budget Model, Pipeline Modes

See [pipeline-and-modes.md](pipeline-and-modes.md) for:
- LLM client architecture: provider boundary, interface, model routing, temperature defaults (Section 3)
- Budget model: BudgetConfig, mode-dependent budgets, token counting, enforcement, validation (Section 4)
- All pipeline modes in detail: plan, implement, train-plan, curate-data, bootstrapping (Section 5)
- Solve orchestrator: pass hierarchy, execution flow, design principles, data structures, session keys, raw DB tables, training data, CLI integration, configuration (Section 5.7)

---

## 9-10. CLI Interface and Configuration

See [cli-and-config.md](cli-and-config.md) for:
- Command table with phases and LLM requirements (Section 1)
- Argument conventions (Section 2)
- Required inputs by command (Section 3)
- Budget config file format (Section 4)
- Config file: location, full TOML example, resolution order, config loader (Section 5)

---

## 11. Session Lifecycle

See [pipeline-and-modes.md](pipeline-and-modes.md) Section 6 for the session flow (create -> write -> handoff -> inherit -> close -> archive -> delete).

See [schemas.md](schemas.md) Section 4 for the full session DB key contracts table.

---

## 12. Query API

See [pipeline-and-modes.md](pipeline-and-modes.md) Section 7 for the `KnowledgeBase` class methods and the Phase 4 `LogAnalysisAPI` addition.

---

## 13. Refinement Protocol

See [pipeline-and-modes.md](pipeline-and-modes.md) Section 8 for the refinement flow, bounds, and `RefinementRequest` dataclass.

---

## 14. Preflight Checks

See [pipeline-and-modes.md](pipeline-and-modes.md) Section 9 for the four preflight checks (model config, curated DB, raw DB, enrichment status).

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

See [training-strategy.md](training-strategy.md) for the complete training strategy:
- Per-stage training targets and techniques (Section 1)
- Deployment architecture: vLLM per-request adapter selection (Section 2)
- Raw DB as per-stage training corpus (Section 3)
- Training artifact storage (Section 4)
- Bootstrapping from external repo history: data source, commit filtering, synthetic pair generation, quality signals, storage, cold-start datasets, fail-fast corpus (Section 5)
- Distillation strategy: teacher models, per-stage configurations, training infrastructure (Section 6)
- Self-improvement guardrails: hard limits, mandatory practices, timeline, thresholds (Section 7)
- Evidence base and advanced techniques (Section 8)

---

## 17. Platform Notes

- **Development platform**: Windows 11, Python 3.11+.
- **Git worktree**: Phase 3 uses `git worktree add/remove` for patch isolation. Platform verification (especially Windows long paths and cleanup) is a blocking prerequisite before patch application code.
- **No silent fallback**: if worktrees fail, error with diagnostics. Do not silently degrade to in-place patching.
- **Phases 1-3 inference is local**: Ollama (or vLLM/llama-server) on localhost, no data leaves the machine during normal operation.
- **Phase 4 teacher inference uses remote APIs**: Distillation data generation calls external teacher model APIs (Qwen3.5-Plus, DeepSeek-V3.2). Training itself is always local (GPU fine-tuning on local hardware).

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
| `MetaPlan` | Phase 3 | Phase 3 | `agent/dataclasses.py` |
| `MetaPlanPart` | Phase 3 | Phase 3 | `agent/dataclasses.py` |
| `PartPlan` | Phase 3 | Phase 3 | `agent/dataclasses.py` |
| `PlanStep` | Phase 3 | Phase 3 | `agent/dataclasses.py` |
| `StepResult` | Phase 3 | Phase 3 | `agent/dataclasses.py` |
| `PlanAdjustment` | Phase 3 | Phase 3 | `agent/dataclasses.py` |

---

## 19. MVP Boundary Checklist

- **Must**: Python symbol-level reference edges (`symbol_references`) in Phase 1 indexing.
- **Must**: Python symbol neighbor traversal in Query API.
- **Must**: File-level dependency graph for Python/TS/JS.
- **Must**: Indexing works without any LLM dependency (`cra index` is deterministic).
- **Must**: Search-and-replace as the single output format for code generation.
- **Must**: Plan mode produces structured plan artifacts consumable by implement mode.
- **Must**: Multi-model routing via config (coding + reasoning models).
- **Must**: Solve orchestrator composes atomic plan/implement passes (meta-plan -> part-plan -> step implementation -> adjustment).
- **Must**: `--plan` bypasses orchestrator for single atomic implement pass. `--meta-plan` skips meta-plan pass only.
- **Must**: Orchestrator logs to `orchestrator_runs` and `orchestrator_passes` in raw DB.
- **Must**: Partial success is an explicit outcome -- orchestrator records completed and incomplete parts/steps.
- **Must Not**: TS/JS symbol-level call/reference edges in Phase 1 MVP.
- **Post-MVP**: TS/JS symbol-edge extraction. Additional retrieval stages. Phase 4 (self-improvement loop + bootstrapping from external repos). Baseline comparison mode.
