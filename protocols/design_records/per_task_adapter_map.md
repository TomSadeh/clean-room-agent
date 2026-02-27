# Per-Task Adapter Map

Date: 2026-02-27

## Principle

Every distinct cognitive task in the pipeline gets its own LoRA adapter trained on its specific input-output distribution. No adapter trains on mixed distributions. The model tier (0.6B / 1.7B) is chosen by the minimum capacity needed for the task.

> **Revision (Feb 2026):** Planning decomposition reduced per-call cognitive complexity enough that the 4B tier is likely redundant. The cascade is effectively two tiers: 0.6B (binary classification) → 1.7B (everything else). Tier 2 tasks below (originally assigned to 4B) are reassigned to 1.7B with decomposed sub-tasks.

This supersedes the adapter layout in `planning/training-strategy.md` Section 1, which was organized around the old two-model architecture. The two-tier model cascade and scaffold-then-implement pipeline create a finer-grained adapter map.

## Current Task Inventory

### Tier 2 → Tier 1 — 1.7B (Reasoning / Architecture, via decomposed sub-tasks)

> **Revision:** Originally assigned to 4B. Planning decomposition breaks these into atomic sub-tasks (enumeration → grouping → binary deps) that 1.7B handles. These tasks are now Tier 1 (1.7B) with decomposed prompts rather than monolithic ones.

| # | Task | Input | Output | Training signal |
|---|------|-------|--------|-----------------|
| A1 | **Meta-plan** | Task description + context | Part decomposition with dependency edges | Did downstream parts succeed independently? |
| A2 | **Part-plan** | Part description + context | Implementation step sequence | Did steps execute without adjustment loops? |
| A3 | **Scaffold** | Plan + context | Header files, signatures, struct layouts, docstrings | Did implementations compile? Did tests pass? |
| A4 | **Adjustment** | Test failures + prior changes + remaining steps | Revised step sequence | Did revised steps succeed? |
| A5 | **Task analysis** | Raw task + repo file tree | Structured task query (type, seeds, keywords, intent) | Did retrieval find the right files? |

### Tier 1 — 1.7B (Code Generation / Structured Classification)

These tasks require language fluency and the ability to follow a contract, but not architectural judgment.

| # | Task | Input | Output | Training signal |
|---|------|-------|--------|-----------------|
| A6 | **Implement (per-function)** | Scaffold + target function docstring + types | Function body | Compiles? Passes function-level tests? |
| A7 | **Test generation (per-function)** | Scaffold + target function docstring | Test function(s) | Tests compile? Tests catch known-bad implementations? |
| A8 | **Precision classification** | Task summary + symbol list with signatures | Detail level per symbol (primary/supporting/type_context/excluded) | Were classified symbols actually needed at that detail level? |
| A9 | **Documentation** | Source file + task context | Docstring/comment edits | AST-based doc verification pass |
| A10 | **Refilter** | File list with sizes + budget | Subset of files to keep | Did execute succeed with the kept subset? |

### Tier 0 — 0.6B (Binary Classification)

These tasks produce a single binary decision per call. Maximum volume, minimum complexity.

| # | Task | Input | Output | Training signal |
|---|------|-------|--------|-----------------|
| A11 | **Scope judgment** | Task summary + single file description | Relevant / irrelevant + reason | Was the file actually needed? |
| A12 | **Routing** | Task summary + available stage descriptions | Include stage / exclude stage (per stage) | Did selected stages produce useful output? |

## Further Decomposition Opportunities

The inventory above is the current pipeline's task set. Several tasks can be decomposed further into simpler decisions:

### Precision → binary cascade

Current: one call classifies a symbol into 4 categories.
Decomposed: 3 binary decisions in sequence.

```
Is this symbol relevant at all?  → no → excluded
Is this symbol primary?          → yes → primary
Is this symbol supporting?       → yes → supporting
                                 → no → type_context
```

Each binary decision is simpler and can potentially go to 0.6B. The cascade terminates early (most symbols are excluded in the first check), so the average number of calls per symbol is closer to 1.2 than 3.

### Scope judgment → per-tier binary

Current: one binary call per file.
Decomposed: separate adapters per expansion tier.

- **Dependency relevance** (Tier 2 files): "This file is imported by the seed. Is it relevant?"
- **Co-change relevance** (Tier 3 files): "This file historically changes with the seed. Is it relevant?"
- **Metadata relevance** (Tier 4 files): "This file matches task keywords. Is it relevant?"

Each tier has different base rates and different features that matter. A dependency-tier adapter sees import relationships; a co-change adapter sees commit history patterns. Separate adapters could specialize on each signal type.

### Meta-plan → sub-tasks

Current: one call produces full part decomposition.
Decomposed:

1. **Module identification**: Which modules does this task touch? (Could be binary per module.)
2. **Part boundary design**: Given the affected modules, where do the part boundaries go?
3. **Dependency edge assignment**: Given parts, which depends on which? (Binary per pair.)

### Scaffold → sub-tasks

Current: one call produces full header structure.
Decomposed:

1. **Type design**: Struct/enum/typedef definitions.
2. **Function signature design**: Names, parameters, return types.
3. **Contract writing**: Docstrings describing behavior, preconditions, error conditions.

Each has a different character — type design is about data modeling, signature design is about API boundaries, contract writing is about specification. Different training distributions.

### Test generation → sub-tasks

Current: one call per function generates all test cases.
Decomposed:

1. **Happy path test**: Does the function work for normal input?
2. **Edge case test**: Boundary values, empty inputs, max sizes.
3. **Error path test**: Invalid input, resource exhaustion, null pointers.

Each is a binary question: "for this input category, what should happen?" Different distributions of creativity required.

## Adapter Count Projection

| Granularity | Adapters | Base models |
|---|---|---|
| Current (coarse) | 12 (A1-A12) | 2 (0.6B + 1.7B) |
| With precision cascade | 14 | 2 |
| With per-tier scope | 15 | 2 |
| With plan/scaffold decomposition | ~20 | 2 |
| Full decomposition | ~25-30 | 2 |

The right granularity is an empirical question. Start with the 12-adapter map, measure per-task error rates via the audit protocol, and decompose tasks whose error rates are too high. If scope judgment error rate is 15% but co-change relevance is 30% while dependency relevance is 5%, that's a signal to split scope judgment into per-tier adapters and focus training on the co-change adapter.

## Training Data Sources (per adapter)

Every adapter's training data comes from the raw DB. Each LLM call is logged with:
- `task_id` → traces to final task outcome (success/failure)
- `stage_name` → identifies which adapter produced it
- Full prompt and response → the training pair
- Downstream outcome → the quality signal (did the pipeline succeed?)

The audit protocol's reference tasks provide the initial ground truth. As real tasks are run, the raw DB accumulates organic training pairs. The `cra curate-data` pipeline (Phase 4) extracts, labels, and filters these into per-adapter training sets.

## Relationship to vLLM Adapter Routing

Phase 4 uses vLLM (or llama-server) for per-request LoRA routing. The adapter map is the routing table:

```
(base_model, stage_name) → adapter_path
```

Config-only. No code changes needed to add or remove adapters. The ModelRouter already resolves per-stage, just needs adapter_path added to ModelConfig.

## Open Questions

1. **Minimum training examples per adapter.** With 25-30 adapters, each needs enough examples to train. Some tasks (meta-plan) produce few examples per real task. Teacher distillation from 4B → 1.7B and from 1.7B → 0.6B is the volume multiplier for scarce tasks.
2. **Adapter interference during serving.** vLLM supports concurrent LoRA adapters on one base model, but memory scales with adapter count. With 8-10 adapters on the 1.7B, memory overhead must be measured.
3. **When to stop decomposing.** A binary decision on a 0.6B model takes ~10ms. But 30 sequential binary decisions per symbol across 100 symbols is 3 seconds of classification latency. Parallelization helps but there are practical limits. The decomposition should stop when the latency cost exceeds the accuracy gain.
