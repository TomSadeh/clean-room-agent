# Per-Task Adapter Map

Date: 2026-02-27
Revised: 2026-02-28

## Principle

Every distinct cognitive task in the pipeline gets its own LoRA adapter trained on its specific input-output distribution. No adapter trains on mixed distributions. The model tier (0.6B / 1.7B) is chosen by the minimum capacity needed for the task.

> **Revision (Feb 2026):** Planning decomposition reduced per-call cognitive complexity enough that the 4B tier is likely redundant. The cascade is effectively two tiers: 0.6B (binary classification) → 1.7B (everything else). Tier 2 tasks below (originally assigned to 4B) are reassigned to 1.7B with decomposed sub-tasks.

This is the authoritative adapter inventory. `planning/training-strategy.md` defines the training strategy for these adapters (how to produce, curate, and deploy them) and references this document rather than maintaining a parallel task list.

## Current Task Inventory

This inventory reflects the **implemented code** as of 2026-02-28. Tasks are organized by judgment pattern (how the LLM is called), not by pipeline phase.

### Pattern 1 — Binary Judgment (one call per item, yes/no)

All judgment tasks use `run_binary_judgment()` — one LLM call per candidate, producing a single yes/no verdict. Maximum parallelizability, minimum cognitive complexity per call. Natural 0.6B candidates. Default-deny on parse failure (R2). Planning stages (A6, A7) fail-fast on incomplete judgments instead.

| # | Task | System Prompt | Input | Output | Training signal | Role | Failure mode |
|---|------|---------------|-------|--------|-----------------|------|--------------|
| A1 | **Scope judgment** | `SCOPE_BINARY_SYSTEM` | Task summary + single file description (path, language, tier, metadata) | yes/no | Was the file actually needed in execute? | classifier | R2 default-deny |
| A2 | **Precision pass 1** (relevance) | `PRECISION_PASS1_SYSTEM` | Task summary + single symbol (name, kind, path, signature, doc) | yes/no | Was the symbol needed? | reasoning | R2 default-deny |
| A3 | **Precision pass 2** (primary) | `PRECISION_PASS2_SYSTEM` | Task summary + single pass-1 survivor | yes/no | Was primary detail level justified? | reasoning | R2 default-deny |
| A4 | **Precision pass 3** (supporting) | `PRECISION_PASS3_SYSTEM` | Task summary + single non-primary survivor | yes/no | Was supporting detail level justified? | reasoning | R2 default-deny |
| A5 | **Similarity judgment** | `SIMILARITY_BINARY_SYSTEM` | Task summary + function pair (sizes, callee jaccard, name LCS, same parent) | yes/no | Did grouping improve context assembly? | classifier | R2 default-deny |
| A6 | **Part dependency** | `part_dependency` | Part A description + Part B description | yes/no | Was the dependency edge correct? | reasoning | Fail-fast (ValueError) |
| A7 | **Step dependency** | `step_dependency` | Step A description + Step B description | yes/no | Was the dependency edge correct? | reasoning | Fail-fast (ValueError) |
| A8 | **KB pattern relevance** | `kb_pattern_relevance` | Function stub + KB section (first 500 chars) | yes/no | Did selected KB patterns improve implementation? | coding | R2 default-deny |
| A9 | **Routing** | `ROUTING_BINARY_SYSTEM` | Task summary + stage name + description | yes/no | Did selected stages produce useful output? | (orchestrator) | R2 default-deny |

### Pattern 3 — Structured Enumeration (single call, structured output)

These tasks produce a structured list or decomposition from a single LLM call. Moderate complexity — the model must enumerate or group items. 1.7B tasks; 0.6B is unlikely to handle these reliably.

| # | Task | System Prompt | Input | Output | Training signal |
|---|------|---------------|-------|--------|-----------------|
| A10 | **Change point enumeration** | `change_point_enum` | Task description + context | List of files/symbols needing change | Did enumeration cover all actual changes? |
| A11 | **Part grouping** | `part_grouping` | Change points (from A10) | Logical groups with file assignments | Were parts well-scoped? No cross-part entanglement? |
| A12 | **Symbol targeting** | `symbol_targeting` | Part description + context | Specific symbols to modify/create | Were the right symbols targeted? |
| A13 | **Interface enumeration** | `interface_enum` | Plan + context | Types, function signatures, system includes | Did scaffold compile? Did implementations fit the interfaces? |
| A14 | **Task analysis** | (task analysis) | Raw task + repo file tree | Structured task query (type, seeds, keywords, intent) | Did retrieval find the right files? |

### Pattern 4 — Design / Generation (single call, complex output)

These tasks produce multi-line structured or code output. Highest per-call complexity. 1.7B minimum; likely the tasks that benefit most from LoRA specialization.

| # | Task | System Prompt | Input | Output | Training signal |
|---|------|---------------|-------|--------|-----------------|
| A15 | **Step design** | `step_design` | Symbol targets (from A12) + context | Implementation step sequence (PartPlan) | Did steps execute without adjustment loops? |
| A16 | **Header generation** | `header_gen` | Interface spec (from A13) + context | Complete .h file content (per header) | Did header compile? Did implementations compile against it? |
| A17 | **Implement (per-step)** | `implement` | Curated context + step description | Search/replace code edits | Did edits apply cleanly? Did validation pass? |
| A18 | **Function implement** | `function_implement` | Scaffold + function stub + KB patterns + compiler error (if retry) | Function body | Compiles? Passes function-level tests? |
| A19 | **Test implement** | `test_implement` | Context + test step description | Test code edits | Tests compile? Tests catch known-bad implementations? |
| A20 | **Adjustment** | `adjustment` | Test failures + prior changes + remaining steps | Revised step sequence | Did revised steps succeed? |
| A21 | **Documentation** | `documentation` | Source file + task context | Docstring/comment edits | AST-based doc verification pass |
| A22 | **Refilter** | (refilter) | File list with sizes + budget | Subset of files to keep | Did execute succeed with the kept subset? |

### Monolithic Alternatives (configurable, same output types)

When decomposed planning/scaffold is disabled (`orchestrator.decomposed_planning = false` / `orchestrator.decomposed_scaffold = false`), these monolithic tasks replace Patterns 2-3 above. They exist as fallbacks and baselines.

| # | Task | Replaces | Output |
|---|------|----------|--------|
| M1 | **Meta-plan** (monolithic) | A10 + A11 + A6 | MetaPlan (parts with dependency edges) |
| M2 | **Part-plan** (monolithic) | A12 + A15 + A7 | PartPlan (steps with dependency edges) |
| M3 | **Scaffold** (monolithic) | A13 + A16 + stubs | ScaffoldResult (headers + stubs as PatchEdits) |

These are not separate adapter targets — they're the pre-decomposition versions of the same tasks. If decomposition is active (the default path), they are unused.

## Decomposition Status

The following decompositions from the original design record are **now implemented**:

| Decomposition | Status | Implementation |
|---|---|---|
| Meta-plan → enum + grouping + binary deps | **Done** | `execute/decomposed_plan.py`: `decomposed_meta_plan()` |
| Part-plan → targeting + design + binary deps | **Done** | `execute/decomposed_plan.py`: `decomposed_part_plan()` |
| Scaffold → interface enum + header gen + deterministic stubs | **Done** | `execute/decomposed_scaffold.py`: `decomposed_scaffold()` |
| Precision → 3-pass binary cascade | **Done** | `retrieval/precision.py`: pass1 (relevance) → pass2 (primary) → pass3 (supporting) |
| Per-function implementation (scaffold path) | **Done** | `orchestrator/runner.py`: per-stub loop with `execute_function_implement()` |

## Further Decomposition

The 22-task inventory above reflects the current code. Further decomposition is not planned — it is triggered by measured per-task error rates from the audit protocol. If a specific task's error rate is disproportionately high, that task is a decomposition candidate. No speculative decomposition roadmap.

## Tier Assignment Summary

| Tier | Model | Pattern | Tasks | Count |
|---|---|---|---|---|
| 0 (candidate) | 0.6B | Binary judgment | A1, A2, A3, A4, A5, A6, A7, A8, A9 | 9 (all currently on 1.7B via reasoning/classifier roles) |
| 1 | 1.7B | Structured enum | A10, A11, A12, A13, A14 | 5 |
| 1 | 1.7B | Design/generation | A15, A16, A17, A18, A19, A20, A21, A22 | 8 |

All 9 binary tasks (A1-A9) are candidates for 0.6B. Each produces a single yes/no verdict. If 0.6B proves reliable, 9 of 22 adapters move to the smaller model — reducing total 1.7B inference load by ~40%.

## Training Data Sources (per adapter)

Every adapter's training data comes from the raw DB. Each LLM call is logged with:
- `task_id` → traces to final task outcome (success/failure)
- `stage_name` → identifies which adapter produced it
- Full prompt and response → the training pair
- Downstream outcome → the quality signal (did the pipeline succeed?)

The audit protocol's reference tasks provide the initial ground truth. As real tasks are run, the raw DB accumulates organic training pairs. The `cra curate-data` pipeline (Phase 4) extracts, labels, and filters these into per-adapter training sets.

### Data volume per task pattern

| Pattern | Examples per task run | Volume concern |
|---|---|---|
| Binary judgment (A1-A9) | 10-100+ per stage (one per candidate/pair) | Highest volume. Naturally balanced (yes/no). May need subsampling. |
| Structured enum (A10-A14) | 1 per stage per part | Low volume. Teacher distillation is the volume multiplier. |
| Design/generation (A15-A22) | 1 per step/function | Moderate volume. Scales with task complexity. |

## Relationship to vLLM Adapter Routing

Phase 4 uses vLLM (or llama-server) for per-request LoRA routing. The adapter map is the routing table:

```
(base_model, stage_name) → adapter_path
```

Config-only. No code changes needed to add or remove adapters. The ModelRouter already resolves per-stage, just needs adapter_path added to ModelConfig.

With 22 adapters across 2 base models, concurrent adapter loading becomes a practical concern. vLLM's `--lora-modules` allows multiple adapters loaded simultaneously. Adapter weights are small (~20-50 MB each), so 13 adapters on 1.7B adds ~260-650 MB — manageable on 24 GB VRAM.

## Open Questions

1. **0.6B reliability threshold.** All 9 binary tasks are natural 0.6B candidates, but 0.6B reliability on yes/no needs empirical validation. All use the same `run_binary_judgment()` call pattern, so migration is mechanical — change the LLM role from `reasoning`/`classifier` to `classifier` (pointing at 0.6B).
2. **Minimum training examples per adapter.** With 22 adapters, some tasks (A10 change_point_enum, A11 part_grouping) produce only 1 example per task run. Teacher distillation is the volume multiplier for scarce tasks.
3. **Adapter interference during serving.** With 13 adapters on the 1.7B and 9 on the 0.6B, memory overhead on both models must be measured. Grouping adapters by pipeline phase (retrieval adapters loaded during retrieval, execute adapters loaded during execution) could halve the concurrent adapter count.
4. **Shared vs. separate adapters for precision passes.** Precision passes 1-3 (A2-A4) share the same base structure (task + symbol → binary) but ask different questions. Empirical question: does a single "precision" adapter with the pass number in the prompt work as well as 3 separate adapters? Fewer adapters = more training data per adapter.