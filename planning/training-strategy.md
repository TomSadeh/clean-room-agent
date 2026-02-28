# LoRA Training Strategy

Per-adapter LoRA training targets for 22 pipeline tasks, base model full fine-tuning (coding style + context window reduction + positional robustness), deployment architecture for 22 adapters across 2 base models, training data sources (bootstrapping + self-improvement + cold-start datasets + plan artifacts + plan validation harness), distillation strategy, execution priority, evaluation methodology, training infrastructure, and self-improvement guardrails.

**Adapter inventory source of truth**: `protocols/design_records/per_task_adapter_map.md` defines the authoritative list of 22 tasks (A1-A22), their judgment patterns, tier assignments, training signals, and data volume characteristics. This document defines the *training strategy* for those adapters — how to produce, curate, and deploy them. It does not maintain a parallel task list.

---

## 1. Per-Task LoRA Training Targets

The pipeline has 22 distinct LLM tasks organized into 3 judgment patterns. Each task gets its own LoRA adapter trained on its specific input-output distribution. No adapter trains on mixed distributions. See `per_task_adapter_map.md` for the complete task inventory.

### 1.1 Pattern 1 — Binary Judgment Adapters (A1-A9)

All 9 binary tasks use `run_binary_judgment()` — one LLM call per candidate, producing a single yes/no verdict. Each call produces exactly one clean training pair (prompt → yes/no). These are natural 0.6B candidates; all currently run on 1.7B and can be mechanically migrated by changing the LLM role in config.

| # | Task | Training Pair | Quality Signal | Technique | Rank |
|---|------|---------------|----------------|-----------|------|
| A1 | Scope judgment | task + file desc → yes/no | was file needed in execute? | SFT | 8-16 |
| A2 | Precision pass 1 | task + symbol → yes/no | was symbol needed? | SFT | 8-16 |
| A3 | Precision pass 2 | task + pass-1 survivor → yes/no | was primary justified? | SFT | 8-16 |
| A4 | Precision pass 3 | task + non-primary → yes/no | was supporting justified? | SFT | 8-16 |
| A5 | Similarity judgment | task + function pair → yes/no | did grouping improve assembly? | SFT | 8-16 |
| A6 | Part dependency | part A + part B → yes/no | was dependency edge correct? | SFT | 8-16 |
| A7 | Step dependency | step A + step B → yes/no | was dependency edge correct? | SFT | 8-16 |
| A8 | KB pattern relevance | function stub + KB section → yes/no | did KB patterns improve impl? | SFT | 8-16 |
| A9 | Routing | task + stage desc → yes/no | did selected stages help? | SFT | 8-16 |

**Data volume**: 10-100+ examples per pipeline run (one per candidate/pair). Highest volume of all patterns. Naturally balanced (yes/no). May need subsampling for balance.

**LoRA rank**: 8-16 is sufficient. Binary judgment is 1 bit of information — the adapter needs to learn item relevance patterns, not complex generation. Higher ranks waste parameters on a trivially structured output space.

**Shared adapter question (A2-A4)**: Precision passes 1-3 share the same structure (task + symbol → binary) but ask different questions (relevance, primary, supporting). Options: (a) 3 separate adapters with maximal specialization, or (b) 1 shared "precision" adapter with the pass number in the prompt. Empirical question — try shared first (3x more training data per adapter), split if accuracy suffers. See `per_task_adapter_map.md` open question 4.

### 1.2 Pattern 2 — Structured Enumeration Adapters (A10-A14)

5 tasks producing structured lists or decompositions from a single LLM call. 1.7B minimum — 0.6B is unlikely to enumerate reliably.

| # | Task | Training Pair | Quality Signal | Technique | Rank |
|---|------|---------------|----------------|-----------|------|
| A10 | Change point enum | task + context → file/symbol list | did enum cover all actual changes? | SFT | 16-32 |
| A11 | Part grouping | change points → grouped parts | were parts well-scoped? | SFT | 16-32 |
| A12 | Symbol targeting | part + context → specific symbols | were right symbols targeted? | SFT | 16-32 |
| A13 | Interface enum | plan + context → types, signatures | did scaffold compile? | SFT | 16-32 |
| A14 | Task analysis | raw task + file tree → structured query | did retrieval find right files? | SFT | 16-32 |

**Data volume**: ~1 per stage per part per pipeline run. Lowest volume of all patterns. Teacher distillation ([Section 7](#7-distillation-strategy)) is the volume multiplier.

**LoRA rank**: 16-32. These produce structured multi-item output — more capacity than binary, less than full code generation.

### 1.3 Pattern 3 — Design/Generation Adapters (A15-A22)

8 tasks producing multi-line structured or code output. Highest per-call complexity. 1.7B minimum; likely benefit most from LoRA specialization.

| # | Task | Training Pair | Quality Signal | Technique | Rank |
|---|------|---------------|----------------|-----------|------|
| A15 | Step design | symbol targets + context → PartPlan | did steps execute without adjustment? | SFT + DPO | 32-64 |
| A16 | Header generation | interface spec + context → .h file | did header compile? | SFT | 16-32 |
| A17 | Implement (per-step) | curated context + step → code edits | did edits apply? validation pass? | SFT + DPO | 32-64 |
| A18 | Function implement | scaffold + stub + KB + error → body | compiles? passes tests? | SFT + DPO | 32-64 |
| A19 | Test implement | context + test step → test code | tests compile? catch known-bad? | SFT + DPO | 32-64 |
| A20 | Adjustment | failures + changes + steps → revised | did revised steps succeed? | SFT | 32-64 |
| A21 | Documentation | source + context → doc edits | AST-based doc verification | SFT | 16-32 |
| A22 | Refilter | file list + sizes + budget → subset | did execute succeed with kept subset? | SFT | 16-32 |

**Data volume**: ~1 per step/function per pipeline run. Moderate volume. Scales with task complexity.

**Techniques**: SFT for all. DPO added for A15, A17, A18, A19 — the core planning and code generation tasks where preference alignment between "plan/code that works" and "plan/code that doesn't" provides the strongest training signal. DPO pairs come from the plan validation harness ([Section 6.9](#69-plan-validation-harness)) and from temperature-varied generation on the same task.

### 1.4 Training Pair Format

NL-to-Code instruction format (natural language description → code) significantly outperforms Code-to-Code format across all benchmarks (OpenCodeInstruct, 2025). All training pairs — including synthetic pairs from commit histories — should use natural language task descriptions as input, not code-in → code-out.

Training data curation must separate examples by base model — a 1.7B LoRA cannot be applied to 0.6B and vice versa. With 22 adapters across 2 base models, this means: 9 adapters (A1-A9) share the 0.6B base (once validated), 13 adapters (A10-A22) share the 1.7B base.

---

## 2. Base Model Full Fine-Tuning

Before LoRA adapters are trained, each base model undergoes a full fine-tune. This is not optional — it prepares the base for its specific role in the pipeline. The full fine-tune combines three objectives in a single training run:

1. **Coding style** — fail-fast error handling, no defensive patterns, narrow exception clauses, rich contextual error messages. Trained from the fail-fast corpus ([Section 6.7](#67-fail-fast-training-corpus)).
2. **Context window reduction** — reduce `max_position_embeddings` and retrain RoPE frequencies to match each base's actual operating range. This is a free rider on the training run that produces compounding gains when LoRAs are stacked on top. See the rationale and detailed analysis below.
3. **Positional robustness (IN2 training)** — train the model to attend to critical information at any position within the reduced window, not just the beginning and end. IN2 (INformation-INtensive) training synthesizes QA data where answers require attending to specific ~128-token segments scattered throughout the context. Applied to Mistral-7B, this produced FILM-7B with positional robustness comparable to GPT-4-Turbo while maintaining short-context performance. The root cause of lost-in-the-middle degradation is insufficient training supervision — IN2 directly addresses it. For the pipeline, curated context has no natural "important stuff first" ordering; symbols from different files appear in whatever order assembly produces. The model must attend uniformly across the window.

### 2.1 What Changed: Planning Moves from Base to Per-Task LoRAs

The previous strategy put planning capability in the base model fine-tune. The rationale was sound for monolithic planning — data scarcity favored all-parameter updates, and adapter swapping during the orchestrator loop was costly.

Planning decomposition changes both premises:

1. **No monolithic planning tasks.** The orchestrator no longer calls "produce a full MetaPlan" — it calls A10 (enumerate change points), A11 (group into parts), A6 (binary dependency per pair). Each is a distinct cognitive task with its own input-output distribution. Training them as separate LoRAs is no longer "splitting scarce data across adapters" — it's training each adapter on its specific distribution.

2. **Adapter swapping aligns with pipeline phases.** The orchestrator runs in phases: retrieval (A1-A5, A9, A14), planning (A6, A7, A10-A13, A15), execution (A8, A16-A22). Adapters within each phase can be loaded concurrently ([Section 3](#3-deployment-architecture)). Adapter swaps happen at phase boundaries, not within the hot loop.

3. **Data characteristics differ by task.** A10 (change point enumeration) and A15 (step design) have fundamentally different input-output distributions. Forcing them to share a base fine-tune's all-parameter updates means the base learns a mixture that may not serve either well. Separate LoRAs let each task specialize fully.

**What stays in the base**: coding style (objective 1) and model-level optimizations (objectives 2-3). These are genuinely cross-cutting — every adapter benefits from fail-fast patterns, reduced context windows, and positional robustness. Planning capability is no longer cross-cutting — it's decomposed into per-task LoRAs.

### 2.2 Context Window Targets by Task Pattern

With 22 tasks across 3 patterns, the context window requirements are more varied than the old 4-stage model assumed:

| Pattern | Tasks | Actual content range | Target `max_position_embeddings` |
|---------|-------|---------------------|----------------------------------|
| Binary judgment (A1-A9) | 9 | ~0.5-4K (task summary + one item) | 4K (0.6B base) |
| Structured enum (A10-A14) | 5 | ~4-16K (task + context + candidates) | 16K (1.7B base) |
| Design/generation (A15-A22) | 8 | ~8-32K (curated context + task + prompt) | 32K (1.7B base) |

**Key insight**: binary tasks use dramatically less context than the old strategy assumed. The 0.6B base only needs a 4K window — an aggressive 8x reduction from Qwen3-0.6B's stock 32K. This is ideal: the model's attention patterns are concentrated in the tiny range where all content lives.

**Two base models, two context targets**:
- **0.6B base (4K)** — serves A1-A9. Binary judgment on single items. RoPE retrained for 4K.
- **1.7B base (32K)** — serves A10-A22. Enum, design, and generation tasks. 32K covers worst-case design/generation; enum tasks operate within this window with headroom.

This means two full fine-tune runs total (4K 0.6B + 32K 1.7B), not five. The 1.7B base no longer needs an 8K variant — task analysis (A14) fits within 16K, well within the 32K base.

### 2.3 Why Context Window Reduction Matters

**Shorter context windows produce better output.** This is not an efficiency claim — it is a quality claim backed by converging empirical evidence (see `research/context_window_size_research.md` for the full literature review):

- Context length alone degrades performance **13.9–85%** even when retrieval is perfect and irrelevant tokens are completely masked (Du et al., 2025). The model isn't distracted by noise — the sheer computational burden of longer sequences interferes with reasoning.
- Most LLMs effectively utilize only **10–20%** of their context on reasoning tasks (BABILong, NeurIPS 2024). Open-source models demonstrate effective context of **less than 50%** of training length (RULER, COLM 2024).
- At 32K tokens, **11 of 12 tested models dropped below 50%** of their short-context performance. The effective context length maintaining ≥85% of baseline was **≤2K tokens** for most models (NoLiMa, 2025).
- RAG + Llama-3-8B (8K context) achieved perfect retrieval accuracy up to 2M tokens, while GPT-4o failed beyond 128K. OP-RAG achieved **38% better F1 with 60% fewer tokens** than full-context Llama3.1-70B.

The industry's race to extend context windows — 4K to 128K to 256K — has outpaced models' ability to use that context. For this project, context reduction is not a cost optimization. It is the mechanism by which the pipeline produces better results.

**This is the same thesis as the N-prompt pipeline itself.** The pipeline's core claim is that a 32K window at ~100% signal relevance beats a 200K window at 10-15% utilization. The research above validates this claim empirically: shorter, focused contexts paired with intelligent retrieval outperform naive long-context approaches on most practical tasks. Context reduction at the model level and context curation at the pipeline level are the same strategy applied at different layers.

**Why compression techniques are architecturally incompatible:** The research literature offers a rich toolkit for context compression — hard prompt compression (LLMLingua), soft token methods (gisting), KV-cache surgery (SnapKV). These are impressive engineering, but they are incompatible with this project's transparency architecture. Every compression technique inserts an opaque decision layer between the retrieval pipeline and the model: a BERT model silently dropping tokens, learned vectors replacing readable text, attention selectively masking KV pairs. The traceability chain breaks — logged context no longer equals received context, which means the self-improvement loop trains on corrupted pairs. Context reduction must happen at the model level (RoPE retraining) and the pipeline level (curated retrieval with logged decisions), not at an intermediate compression layer.

**What unused context capacity costs:**

1. **Wasted attention computation.** Self-attention is O(n²) in sequence length. A 4K model requires 64x less computation than a 32K model. In practice, 16x context increase produces 50x latency increase (Adnan et al., 2024). For binary tasks (A1-A9) that use ~0.5-4K of content, running on a 32K base wastes >90% of attention computation.

2. **Diluted positional resolution.** RoPE positional encodings are frequency-distributed across the full `max_position_embeddings` range. A model trained for 32K positions spreads its positional resolution thinly over a range binary tasks will never use.

3. **Wasted KV cache.** KV-cache memory scales linearly with sequence length. With 22 adapters across 2 models, KV cache savings from tight context windows directly enable more concurrent adapters in VRAM.

4. **Looser LoRA fit.** A LoRA adapter fine-tuned on a base trained for 32K but that only ever sees 1-4K during LoRA training is compensating for a positional distribution mismatch. A LoRA on a context-optimized base starts from a better foundation.

### 2.4 Why Context Reduction Compounds with LoRAs

- The base model's attention patterns are retrained for the actual operating range. Every subsequent LoRA inherits these tighter patterns for free.
- Smaller KV cache per layer — the model physically cannot attend beyond the new limit, freeing VRAM permanently. The freed VRAM goes to larger batch sizes, more concurrent LoRA adapters in vLLM, or reduced swapping.
- A LoRA fine-tuned on a context-optimized base learns task-specific behavior without also needing to compensate for positional distribution mismatch. This is a strictly better starting point than a LoRA on a generic 128-256K base.

### 2.5 Practical Notes

- **Irreversible per base.** The model loses long-context capability. Retrain from original weights if the target range needs to change. This means base checkpoints must be versioned.
- **Shrink ratio for 0.6B.** 32K → 4K (8x reduction) shifts the RoPE frequency distribution significantly — validate on a held-out set before committing. This is the most aggressive reduction in the system and the highest risk. Ablate 4K vs 8K before choosing.
- **Shrink ratio for 1.7B.** Qwen3-1.7B has a 32K stock window. If the 1.7B already trains at 32K, no reduction is needed — the base fine-tune applies coding style and IN2 only. If the stock window is larger (e.g., 128K), reduce to 32K (4x, safe range).
- **Unsloth supports full fine-tuning** on 24GB VRAM for 1.7B models. Training time is hours — one-time cost per base model release, amortized across all LoRA adapters trained on top of it.

> **Revision note (Feb 2026):** The specific `max_position_embeddings` targets above are based on current Qwen3 model specifications. When Qwen 3.5 small models release (expected with 64K+ base context), the reduction ratios and RoPE retraining parameters will need recalculation. The architecture and rationale are stable; the specific numbers are not.

### 2.6 Full-Window Generalist

One 1.7B base model is kept at the original `max_position_embeddings` without context reduction. This model receives the coding style fine-tuning but retains the full context window. It serves three purposes that context-reduced specialists cannot:

1. **Audit** — feed an entire pipeline run's logs into one context window to verify the traceability chain end-to-end. Specialists at 4-32K cannot hold a full run's worth of decisions.
2. **A/B testing** — compare specialist output against the unrestricted base to validate that context reduction actually improves quality. If context-reduced models don't outperform the generalist on their stage's task, the reduction is not justified.
3. **Fallback** — if a task exceeds a specialist's reduced window, route to the generalist rather than failing or retraining.

The generalist does not replace specialists in the pipeline — it runs alongside them as infrastructure.

**Emergent codebase discipline:** Tight context windows don't just constrain the model — they constrain the code the model can operate on. If a design/generation task has a 32K budget and must fit multiple files plus framing plus the task description, any single file that consumes 20K of that budget is a pipeline-breaking problem. The system naturally pressures toward small, focused files because that's what fits through the pipeline. This creates a reinforcing loop: small files → better retrieval precision → tighter context assembly → which in turn rewards small files.

---

## 3. Deployment Architecture

### 3.1 Phases 1-3 (MVP)

Ollama or any configured provider. Single model at a time, no adapter switching needed. All 22 tasks run on stock Qwen3-1.7B (or 0.6B for binary tasks once validated).

### 3.2 Phase 4 Deployment Target

vLLM with per-request LoRA adapter selection. Ollama cannot hot-swap LoRA adapters per request (each adapter is a separate model tag requiring full unload/reload). vLLM's `--lora-modules` flag enables multiple adapters loaded simultaneously with per-request selection via the `model` field in the OpenAI-compatible API.

**Two vLLM instances** serve the two base models:

| Instance | Base Model | Context Window | Adapters | VRAM Estimate |
|----------|-----------|----------------|----------|---------------|
| 1 | Qwen3-0.6B (4K) | 4K | A1-A9 (9 adapters, ~20 MB each) | ~1-2 GB base + ~180 MB adapters + KV cache |
| 2 | Qwen3-1.7B (32K) | 32K | A10-A22 (13 adapters, ~30-50 MB each) | ~2-3 GB base + ~400-650 MB adapters + KV cache |

Total VRAM: ~5-8 GB for both instances. Fits comfortably on a single 24 GB GPU with room for batch processing.

### 3.3 Phase-Based Adapter Grouping

With 22 adapters, concurrent loading of all adapters is possible but unnecessary. The pipeline runs in phases, and adapters within a phase are active concurrently while cross-phase adapters never coexist:

| Pipeline Phase | Active Adapters | Count | Base Model |
|---------------|-----------------|-------|------------|
| Task analysis | A14 | 1 | 1.7B |
| Retrieval | A1, A2, A3, A4, A5, A9 | 6 | 0.6B (A1-A5, A9) |
| Planning | A6, A7, A10, A11, A12, A13, A15 | 7 | 0.6B (A6, A7) + 1.7B (A10-A13, A15) |
| Execution | A8, A16, A17, A18, A19, A20, A21 | 7 | 0.6B (A8) + 1.7B (A16-A21) |
| Assembly | A22 | 1 | 1.7B |

Maximum concurrent adapters on either instance: ~7 on 1.7B, ~6 on 0.6B. With adapter weights at ~20-50 MB each, peak VRAM for adapters is ~350 MB on 1.7B — trivial on 24 GB.

**Adapter loading strategy**: load all adapters for the current phase at phase start. Unload at phase end. vLLM's `--lora-modules` supports this natively. Alternative: load all 22 at startup (total ~800 MB adapter VRAM) if the phase-switching overhead is measurable. Empirical choice.

### 3.4 Routing

The `[models.overrides]` stage→model mapping resolves which adapter to use. Override values are adapter identifiers (e.g., `scope-v1`) that the provider translates to its native adapter selection mechanism. The `ModelRouter` already supports per-stage resolution via config.toml:

```toml
[models.stages.scope_judgment]
model = "qwen3:0.6b"
adapter = "scope-v1"

[models.stages.precision_pass1]
model = "qwen3:0.6b"
adapter = "precision-v1"

[models.stages.change_point_enum]
model = "qwen3:1.7b"
adapter = "change-point-enum-v1"

[models.stages.implement]
model = "qwen3:1.7b"
adapter = "implement-v1"
```

---

## 4. Raw DB as Per-Task Training Corpus

Each task's training data comes from two sources:

**Logged runs (self-improvement)**: Extracted from `retrieval_llm_calls` filtered by:
- `call_type` / `stage_name` (which task produced it)
- linked `task_runs.success` (was the overall task successful?)
- optionally `retrieval_decisions` (was this specific decision part of a successful run?)

Every LLM call is logged with full I/O, `task_id` for traceability, and `stage_name` for adapter identification. Binary tasks (A1-A9) produce clean (prompt, yes/no, correct_label) triples. Design/generation tasks (A15-A22) produce (context, output, validation_result) triples.

**Synthetic pairs (bootstrapping)**: Generated from external repo commit histories via the bootstrapping pipeline ([Section 6](#6-bootstrapping-from-external-repo-history)). These provide training data before the agent has produced any real runs.

Both sources produce datasets in the same format and are stored with a `source` field to allow mixing and weighting during training.

---

## 5. Training Artifact Storage

- Training plans and curated datasets go in raw DB (they're generated outputs, same as enrichment outputs).
- Adapter metadata (which task, performance metrics, active/inactive, version) goes in curated DB (it's verified configuration the pipeline reads at runtime).

---

## 6. Bootstrapping from External Repo History

The self-improvement loop has a cold-start problem: Phase 4 training needs quality-signal-rich data, but Phase 3's output quality depends on models that haven't been trained yet. External repos break this dependency.

### 6.1 Data Source

Clone mature, well-tested repos from GitHub. Ideal repos have:
- Clear commit messages (natural task descriptions).
- Small, focused bug-fix and feature commits (clean before/after pairs).
- Stable dependency graphs and well-established symbol structure (high-quality indexer output).
- Passing CI (implicit quality signal: the fix stuck, tests passed, it got merged).

**Domain distribution**: The training corpus should span **6+ distinct domains** with no more than 20% from any single domain. This prevents the model from conflating domain idioms with the target coding style. Include repos from authors across at least 5 different organizations, vary codebase sizes from 1K to 50K LOC, and mix library code with tool code.

**Anti-patterns**: Web frameworks (Django, Flask, FastAPI), HTTP clients (requests, httpx), task queues (Celery), and message brokers embed deeply defensive patterns (top-level catch-all handlers, silent fallbacks, `.get()` with defaults everywhere). Any project whose primary job is keeping a long-running process alive will be defensive at its boundaries. Exclude these from the training corpus.

A curated fail-fast repository corpus with specific repo recommendations and an AST-based heuristic scorer for programmatic identification is documented in `references/fail_fast_research.md`. See also [Section 6.7](#67-fail-fast-training-corpus).

### 6.2 Commit Filtering Criteria

**Extraction tool**: PyDriller for commit traversal and diff extraction. **CommitChronicle** (JetBrains Research) provides a reproducible collection pipeline built on PyDriller with deduplication and outlier filtering. The **D3 paper** is the best architectural reference for LLM-powered instruction labeling of code edit sequences at the 1-3B parameter range. Note: no end-to-end commit→training-pair tool exists — expect 1-2 weeks of custom engineering for the full pipeline.

Not every commit is a useful training example. Filter for:
- **Diff size bounds**: single-file or small-cluster changes (e.g. 1-5 files, < 200 lines changed). Massive refactors are noise. CommitPackFT (NeurIPS 2023 Workshop) found that single-file commits produce the highest-quality instruction-completion pairs.
- **Descriptive messages**: commits with imperative-mood messages starting with action verbs ("Fix," "Add," "Verify") serve as synthetic task descriptions. Apply Verb-Direct-Object pattern filtering via NLTK/spaCy POS tagging to retain only actionable messages. Skip sub-3-word messages and generic tokens ("wip", "misc", "update").
- **Non-merge commits**: merge commits conflate multiple logical changes.
- **Language match**: filter to Python/TS/JS/C files matching the indexer's current capability.
- **Exclude automated commits**: filter out dependabot updates, version bumps, CI config changes, and documentation-only commits.

**Message regeneration**: For commits with poor messages but good diffs, use **OpenCommit** or a local Ollama model to regenerate descriptions from diffs. The **OMG paper** (ACM 2024) shows that ReAct prompting with broader software context dramatically improves generated descriptions over diff-only approaches. Apply LLM-as-judge scoring to filter generated descriptions, keeping only high-scoring ones.

### 6.3 Synthetic Pipeline Run Generation

From a filtered commit, reverse-engineer what a correct pipeline run *should have* looked like. With 22 tasks, the synthetic pair derivation table expands significantly:

**Pattern 1 — Binary judgment pairs (A1-A9):**

| # | Task | Synthetic Pair | Derivation |
|---|------|---------------|------------|
| A1 | Scope judgment | task + file desc → yes/no | Files changed in diff → yes. Direct dependencies → yes. Unrelated files → no. One pair per file. |
| A2 | Precision pass 1 | task + symbol → yes/no | Symbols modified or referenced in diff → yes. Others → no. |
| A3 | Precision pass 2 | task + pass-1 survivor → yes/no | Modified symbols → yes (primary). Referenced-only → no. |
| A4 | Precision pass 3 | task + non-primary → yes/no | Referenced-only symbols → yes (supporting). Others → no. |
| A5 | Similarity judgment | task + function pair → yes/no | Functions co-modified in same diff hunk → yes. Functions in unrelated files → no. |
| A6 | Part dependency | part A + part B → yes/no | Derive from commit ordering — if file group A was changed before group B and B depends on A's types/interfaces → yes. |
| A7 | Step dependency | step A + step B → yes/no | Derive from diff ordering within a file — if symbol A must exist before symbol B references it → yes. |
| A8 | KB pattern relevance | function stub + KB section → yes/no | Requires KB content matching — if the function's domain (e.g., string handling, memory management) matches the KB section's topic → yes. Teacher-assisted labeling recommended. |
| A9 | Routing | task + stage desc → yes/no | Derive from task type — bug fixes need scope+precision, features need planning stages, etc. Heuristic rules can generate initial labels. |

**Pattern 2 — Structured enumeration pairs (A10-A14):**

| # | Task | Synthetic Pair | Derivation |
|---|------|---------------|------------|
| A10 | Change point enum | task + context → file/symbol list | Extract the list of changed files and modified symbols directly from the diff. |
| A11 | Part grouping | change points → grouped parts | Group diff changes into independent logical units — files that changed together in the same functional area form a part. Multi-file commits with independent change clusters provide the best examples. |
| A12 | Symbol targeting | part + context → symbols | Within each part, extract the specific symbols (functions, classes, methods) modified by the diff. |
| A13 | Interface enum | plan + context → types, signatures | Extract from commits that add new .h files or new function declarations. The committed header is the target output. Requires C repos or repos with explicit interface files. |
| A14 | Task analysis | commit message + file tree → structured query | The commit message is the task; the diff tells you which files/symbols were relevant, which seed files should have been identified, and what keywords would have helped. |

**Pattern 3 — Design/generation pairs (A15-A22):**

| # | Task | Synthetic Pair | Derivation |
|---|------|---------------|------------|
| A15 | Step design | symbol targets + context → steps | Derive step sequence from the diff: which symbols to change, in what order, at what granularity. Order inferred from dependency analysis within the diff. |
| A16 | Header generation | interface spec + context → .h file | The committed .h file is the target output. Input is the interface specification derived from the corresponding .c file changes. C repos only. |
| A17 | Implement (per-step) | context + step → code edits | The diff itself is the target output, segmented per step. Context is the pre-commit state of relevant files. |
| A18 | Function implement | scaffold + stub + KB → body | Extract individual function implementations from the diff. The pre-commit stub (or newly added function signature) is the input; the committed body is the target. |
| A19 | Test implement | context + test step → test code | If the commit includes test changes: test file changes are the target, code changes are the input context. |
| A20 | Adjustment | failures + changes → revised steps | Requires two related commits: an initial implementation and a subsequent fix. The fix diff is the adjustment target; the initial failure provides the context. Relatively rare in clean commit histories. |
| A21 | Documentation | source + context → doc edits | Documentation-focused commits or doc strings added alongside implementation. |
| A22 | Refilter | file list + sizes + budget → subset | Derive from the diff: the files actually needed are those that were changed. Generate oversized candidate lists from the repo, with the diff-touched files as the correct subset. |

Note: synthetic planning pairs from diffs are post-hoc reconstructions with weaker signal than the plan validation harness ([Section 6.9](#69-plan-validation-harness)), which generates and validates plans forward. Use synthetic pairs for SFT volume; use harness pairs for high-quality SFT and DPO.

**LintSeq** (ICLR 2025) generates synthetic edit sequences from existing code using only a linter — no LLM needed for data generation. Apply LintSeq to transform any existing code dataset into edit-sequence format aligned with our search-and-replace output (A17, A18).

**Instruction format**: All synthetic training pairs should use **NL-to-Code format** (natural language task description → code output) rather than Code-to-Code format. This means commit messages (or regenerated descriptions) should serve as the instruction.

### 6.4 Quality Signal Mapping

External repo commits provide quality signals analogous to what the self-improvement loop gets from logged runs:

| Self-Improvement Signal | External Repo Equivalent |
|------------------------|--------------------------|
| `task_runs.success` | commit was merged / tests passed |
| retrieval decision correctness | did the diff touch the files scope would have found? |
| symbol selection correctness | did the diff modify the symbols precision would have selected? |
| code generation correctness | does the generated diff match the actual commit diff? |
| planning decision correctness | is the inferred decomposition consistent with the diff structure? |

### 6.5 Storage

Synthetic training pairs from external repos are stored in the same raw DB tables as self-improvement training data (`training_datasets`, `training_plans`). A `source` field distinguishes them:

- `source = "synthetic"`: generated from external repo commit history.
- `source = "logged"`: curated from the agent's own logged runs.
- `source = "harness"`: generated by the plan validation harness.

This allows training to mix all data sources as the agent matures.

### 6.6 Cold-Start Datasets

Open datasets that provide immediate training data without commit-history mining:

| Dataset | Size | License | Use |
|---------|------|---------|-----|
| **OpenCodeInstruct** | 5M examples | CC BY 4.0 | Subsample 100-500K for code generation SFT (A17, A18). NL-to-Code format. Safest license. |
| **CommitPackFT** | 702K pairs | Apache 2.0 | Real-world code change pairs (commit message + diff). Directly usable for A17 training. |
| **CodeFeedback-Filtered** | 156K examples | — | Pre-filtered by Qwen-72B for complexity ≥4/5. High-complexity, excellent size for LoRA. |
| **Magicoder OSS-Instruct** | 75K examples | — | Generated from real OSS code snippets as seeds. Orthogonal to Evol-Instruct — combine both. |
| **Evol-Instruct-Code** | 80-110K examples | — | Progressively evolved from Code Alpaca. Proven results on WizardCoder (ICLR 2024). |
| **OpenCoder SFT Stage 2** | 375K examples | — | Quality-filtered with test cases for RL. Well-suited for LoRA. |
| **D3** | 3.6M examples | — | LintSeq + LLM instruction labeling from The Stack. Tested on Llama 3.2 1B and 3B. |
| **SWE-bench** | 2,294 instances | MIT | Validated patches with task descriptions. High-quality but small — use for evaluation and DPO preference pairs. |
| **SRI Tuning** | 20K examples | Apache 2.0 | Search-and-replace instruction examples. Directly aligned with our code edit format (A17). |

These cold-start datasets primarily serve the design/generation adapters (A15-A22). Binary judgment adapters (A1-A9) and structured enumeration adapters (A10-A14) have no direct cold-start datasets — they depend on synthetic pipeline run generation ([Section 6.3](#63-synthetic-pipeline-run-generation)) and teacher distillation ([Section 7](#7-distillation-strategy)).

Both open datasets and commit-history mining complement each other and both depend only on Phase 1. Open datasets provide immediate volume; commit-history mining provides project-specific examples with richer dependency context.

### 6.7 Fail-Fast Training Corpus

Style-targeted training data is a distinct concern from functional-capability training. The code generation LoRAs (A17, A18 especially) should be biased toward fail-fast error handling (custom exception hierarchies, rich contextual error messages, narrow `except` clauses, no silent failure) to align with the project's coding style principles.

A curated 25-repo corpus spanning 6 domains (parsers/compilers, validation/type checking, structured data, developer tooling, small focused libraries, frameworks with good error design) is documented in `references/fail_fast_research.md`. Top-tier repos include strictyaml, attrs, cattrs, LibCST, structlog, Black, typeguard, and Zod (TS). The corpus targets 5,000-10,000 high-quality commit pairs filtered for error-handling improvements.

**Programmatic identification**: An AST-based heuristic scorer using 8 weighted metrics (bare-except ratio, blind-except ratio, try-except-pass count, specific exception ratio, dict["key"] vs .get() ratio, assert density, raise density, exception chaining ratio) can rank candidate repos before manual review. Repos scoring 75-100 are strong candidates.

**DPO preference pairs**: The CodeFavor approach (pre-commit code as rejected, post-commit code as accepted) is directly applicable. Repos like cattrs have commits showing progression from generic `raise Exception(...)` to custom exception classes — ideal before/after training pairs for preference optimization on A17 and A18.

### 6.8 Existing Plan Artifacts from Collaborative Repos

A distinct and immediately available data source: repositories where an AI assistant (Claude, Codex, etc.) generated plans that were committed to the repo as files, and a human then followed those plans to produce implementations.

**Why this is uniquely valuable for planning training:**

Unlike synthetic plan generation from commit diffs ([Section 6.3](#63-synthetic-pipeline-run-generation)), these are *real plans that were actually used*. They capture:
- The plan structure and reasoning as it was actually produced, not reverse-engineered.
- Step ordering, file targets, and rationale — information that is lost when working backwards from a diff.
- Implicit human approval signal: the plan was good enough that the developer chose to follow it.

**Natural quality labeling from git history:**

| Git history pattern | Quality label | Training use |
|--------------------|--------------| -------------|
| Plan → clean implementation, no rework | Good | SFT positive example. DPO accept. |
| Plan → implementation → small bug-fix commits | Fine | SFT positive example (plan was structurally sound). |
| Plan → implementation → significant rewrite or new replacement plan | Bad | DPO reject. |
| Plan → partial implementation → abandoned | Bad | DPO reject (only if an alternative exists). |

**Decomposition alignment**: With decomposed planning, plan artifacts need to be segmented into the sub-tasks they contain: change point enumeration (A10), part grouping (A11), symbol targeting (A12), step design (A15). A single plan artifact produces training examples for multiple adapters, not one monolithic "planning" adapter.

**Scale expectation:** Tens to low hundreds of plan→outcome trajectories per repo. Small volume compared to commit-history mining, but high quality per example. 50-200 high-quality plan trajectories with natural outcome labels may be more valuable than 5K synthetic pairs derived from diffs.

### 6.9 Plan Validation Harness

The highest-quality planning data source: forward-generate plans through our actual pipeline, execute them against real repos, and validate with tests. Produces execution-validated training pairs with zero distribution mismatch between training and inference.

**Core loop:**

1. **Select commit.** From a filtered external repo ([Section 6.2](#62-commit-filtering-criteria)), choose a feature/enhancement commit with passing tests.
2. **Snapshot pre-commit state.** Check out the parent commit.
3. **Index and retrieve.** Run our Phase 1 indexer and Phase 2 retrieval pipeline against the pre-commit repo state. This produces a `ContextPackage`.
4. **Generate plan.** Feed the `ContextPackage` + task description to the planner (teacher model during bootstrapping, student model during self-improvement). With decomposed planning, this produces training data for A10, A11, A6, A12, A15, A7 — each sub-task is a separate training pair.
5. **Execute plan.** Run the plan through our Phase 3 implementation pipeline — code generation for each step, patch application, the full orchestrator loop. This produces training data for A16, A17, A18, A19.
6. **Validate.** Run the repo's existing test suite against the modified code.
7. **Record outcome.** Store per-task `(input, output, outcome)` triples. Success → SFT positive, DPO accept. Failure → DPO reject.

**Why this works:**

- **Distribution match.** The training pair input is our curated context. Zero distribution mismatch.
- **Ground truth exists.** The commit was merged and tests passed — the task is known to be solvable.
- **Per-task quality signal.** With decomposed planning, each sub-task's output can be evaluated independently. A bad change point enumeration (A10) is distinguishable from a bad step design (A15), unlike monolithic planning where the error source is ambiguous.
- **Automated DPO at scale.** Run the planner N times per commit with temperature variation. Some succeed, some fail. Same task, different quality → natural DPO preference pairs per sub-task.

**Isolating quality per task:**

When a plan fails, the failure can be attributed to a specific decomposed sub-task more precisely than with monolithic planning:
- Change point enumeration (A10) missed a file → retrieval-level failure, different from planning failure.
- Part grouping (A11) created cross-cutting parts → structural planning failure.
- Step design (A15) ordered steps wrong → sequencing failure.
- Code generation (A17) botched the edit → execution failure.

Two mitigation strategies for ambiguous cases:
1. **Strong code model for execution.** During bootstrapping, use the teacher for code generation. If the teacher can't implement a plan, the plan is more likely flawed.
2. **Plan-diff structural alignment.** Compare the generated plan against the actual commit diff: file target overlap, symbol target overlap, dependency ordering consistency.

**Infrastructure requirements:**

- **Auto-created isolated environments.** The harness reads the repo's dependency spec and auto-creates a Python venv per repo. On the connected machine, dependencies install from PyPI. Post-migration on the air-gapped machine, dependencies install from a **pre-cached package directory** (`pip install --no-index --find-links ./packages/`).
- **Repo package format.** Each curated repo is packaged as a self-contained directory:
  ```
  repo_package/
    repo/                    # full git clone
    packages/                # pre-cached .whl files for all dependencies
    manifest.json            # filtered commit list, test command, Python version, metadata
  ```
- **Parallelizable.** Each commit is independent. Run N commits across M repos concurrently.
- **Storage.** Each per-task `(input, output, outcome)` triple stored in raw DB with `source = "harness"`.

**Scale expectation:**

From the existing fail-fast repo corpus, conservatively 100-300 qualifying commits per repo across 14 Python repos:
- ~1,400-4,200 total qualifying commits
- ~30-50% success rate with teacher → ~500-2,100 successful plans
- Multiple runs per commit (3 temperature variations) → 4,200-12,600 per-task triples
- With decomposed planning, each harness run produces training data for ~6-10 tasks simultaneously (A10, A11, A6, A12, A15, A7, A17, A18, etc.)

**Self-improvement loop integration:**

The harness architecture is model-agnostic in the planner slot. During bootstrapping, the teacher generates plans. Once student models have per-task LoRAs, the same harness runs with student models — producing on-policy training data where each task's adapter generates its output, executes it, and learns from the outcome.

1. **Teacher bootstrapping** — teacher generates all sub-task outputs, builds initial per-task training corpus.
2. **Student evaluation** — student adapters generate on held-out commits, measure per-task baseline.
3. **Student self-improvement** — student generates per-task outputs on training commits, successes/failures become new training data, mixed with teacher data (accumulate, never replace).
4. **Iteration** — repeat step 3 with fresh commits from new repos to maintain diversity.

**Harness repo corpus:**

The harness uses the same repos already curated for fail-fast style training ([Section 6.7](#67-fail-fast-training-corpus)). Their *code* trains the fail-fast coding style, their *commit histories* feed the plan validation harness. Dual purpose, one corpus. See `references/fail_fast_research.md` for the full repo list.

---

## 7. Distillation Strategy

With 22 adapters across 3 judgment patterns, the distillation strategy must account for fundamentally different data needs per pattern. The key insight: **binary tasks (A1-A9) may not need teacher distillation at all.** They produce clean ground-truth labels from pipeline outcomes. The data-scarce structured enumeration tasks (A10-A14) are where teacher distillation matters most.

### 7.1 Distillation Need by Pattern

| Pattern | Tasks | Volume per run | Ground truth quality | Teacher distillation need |
|---------|-------|---------------|---------------------|--------------------------|
| Binary judgment | A1-A9 | 10-100+ per task | High — binary outcome is unambiguous | **Low.** Ground truth from pipeline outcomes suffices. Teacher adds volume but not quality. |
| Structured enum | A10-A14 | ~1 per part | Medium — structural quality is assessable but subjective | **High.** Volume multiplier. These tasks produce too few organic examples for LoRA training. |
| Design/generation | A15-A22 | ~1 per step/fn | Medium — execution validates output but quality varies | **Medium.** Cold-start datasets provide initial volume. Teacher improves quality and diversity. |

### 7.2 Teacher Models

| Pattern | Target Model | Primary Teacher | Secondary |
|---------|-------------|----------------|-----------|
| Binary (A1-A9) | 0.6B | Ground truth from pipeline outcomes | Qwen3.5-397B for volume bootstrapping only |
| Structured enum (A10-A14) | 1.7B | Qwen3.5-397B-A17B | DeepSeek-V3.2 (cross-validation) |
| Design/generation (A15-A22) | 1.7B | Qwen3-Coder-Next-80B-A3B | OpenCodeInstruct dataset |

**Binary judgment distillation (A1-A9)**: The primary training signal for binary tasks is not teacher output — it's pipeline outcome labels. When the pipeline runs a task and succeeds or fails, every binary judgment call in that run gets a ground-truth label: "was this file actually needed?" is answered by whether execute succeeded without it. Teacher distillation adds volume during cold start but should be phased out in favor of organic labels once the pipeline generates enough runs.

**Structured enum distillation (A10-A14)**: These are the most data-starved tasks. Each produces ~1 example per part per pipeline run. Teacher distillation is the primary volume multiplier: run the teacher on diverse commit-derived tasks to generate hundreds of (task, enumeration/grouping/targeting) pairs. The teacher's output is the training target. This is where the API budget goes.

**Design/generation distillation (A15-A22)**: Mixed strategy. Cold-start datasets (CommitPackFT, SRI Tuning, OpenCodeInstruct) provide initial SFT volume for A17 and A18. Teacher distillation adds higher-quality examples. The plan validation harness ([Section 6.9](#69-plan-validation-harness)) provides execution-validated examples that neither cold-start datasets nor teacher distillation alone can match.

**Tokenizer incompatibility**: Qwen3.5 uses a 250K vocabulary vs Qwen3-1.7B's 150K. This means only response-level SFT (train on teacher's text output), not logit-level distillation (KL divergence on token probabilities). Fallback for logit distillation if needed: Qwen3-235B (same tokenizer family).

### 7.3 Per-Task Training Configurations

**Binary judgment LoRAs (A1-A9, 0.6B base):**

| # | Task | Technique | Rank | Examples | Time (RTX 5090) |
|---|------|-----------|------|----------|-----------------|
| A1-A9 | All binary tasks | SFT | 8-16 | 2-5K each | ~5-15 min each |

Training is fast because: (a) small model (0.6B), (b) tiny context windows (~1-4K), (c) simple output (yes/no). All 9 adapters can be trained in under 2 hours total.

**Structured enumeration LoRAs (A10-A14, 1.7B base):**

| # | Task | Technique | Rank | Examples | Time (RTX 5090) |
|---|------|-----------|------|----------|-----------------|
| A10 | Change point enum | SFT | 16-32 | 3-5K | ~15-30 min |
| A11 | Part grouping | SFT | 16-32 | 3-5K | ~15-30 min |
| A12 | Symbol targeting | SFT | 16-32 | 3-5K | ~15-30 min |
| A13 | Interface enum | SFT | 16-32 | 2-3K | ~10-20 min |
| A14 | Task analysis | SFT | 16-32 | 3-5K | ~15-30 min |

These tasks depend most on teacher distillation for volume.

**Design/generation LoRAs (A15-A22, 1.7B base):**

| # | Task | Technique | Rank | Examples | Time (RTX 5090) |
|---|------|-----------|------|----------|-----------------|
| A15 | Step design | SFT + DPO | 32-64 | 5-10K | ~30-60 min |
| A16 | Header generation | SFT | 16-32 | 2-3K | ~10-20 min |
| A17 | Implement (per-step) | SFT + DPO | 32-64 | 10-50K | ~1-3 hrs |
| A18 | Function implement | SFT + DPO | 32-64 | 5-20K | ~30-90 min |
| A19 | Test implement | SFT + DPO | 32-64 | 3-10K | ~20-60 min |
| A20 | Adjustment | SFT | 32-64 | 1-3K | ~10-20 min |
| A21 | Documentation | SFT | 16-32 | 2-5K | ~10-30 min |
| A22 | Refilter | SFT | 16-32 | 2-3K | ~10-20 min |

A17 (implement) is the largest adapter by training volume — it benefits from CommitPackFT, OpenCodeInstruct subsamples, SRI Tuning, and harness data combined.

**Total estimated training time**: ~4-8 hours for all 22 adapters on RTX 5090. The base fine-tunes (two runs: 0.6B at 4K + 1.7B at 32K) are separate, estimated at 2-4 hours each.

### 7.4 Training Infrastructure

**Deployment context**: the full fine-tune may be outsourced to external GPU hardware; LoRA training runs locally. Post-migration to the air-gapped machine, all training (LoRA only) runs locally with no internet access. See [Section 12](#12-deployment-lifecycle) for the full lifecycle.

**Framework**: **Unsloth** is the primary training framework — 2-5x faster training with 70-80% less VRAM via custom Triton kernels, explicit Qwen3 support with pre-quantized models on HuggingFace, and one-line Ollama/GGUF export. **LLaMA-Factory** is the alternative. Axolotl and raw PEFT/TRL are fallbacks.

**LoRA hyperparameters** (synthesized from PLoRA, QLoRA, and Unsloth documentation):
- **Quantization**: QLoRA 4-bit NF4 + double quantization.
- **Target modules**: All 7 linear layers (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`).
- **Alpha**: 2x rank (e.g., rank 32 → alpha 64).
- **Learning rate**: 2e-4 with cosine scheduler.
- **Optimizer**: `adamw_8bit` for memory efficiency.
- **Dropout**: 0 (enables Unsloth kernel optimizations; add 0.05 only if overfitting observed).
- **Epochs**: 1-3 with early stopping.
- **Precision**: bf16 throughout.

**Qwen-specific gotchas**:
- EOS token `<|im_end|>` must be set explicitly (common source of infinite generation bugs).

**VRAM requirements** (QLoRA 4-bit, gradient checkpointing, sequence length 2048):

| Model | Batch Size 1 | Batch Size 2 | Batch Size 4 |
|-------|:-----------:|:-----------:|:-----------:|
| Qwen3-0.6B | 2-3 GB | 3-4 GB | 4-6 GB |
| Qwen3-1.7B | 3-5 GB | 4-6 GB | 6-9 GB |

A 24 GB RTX 5090 runs LoRA on the 1.7B at batch size 4+ comfortably. The 0.6B is trivial. Training times are fast: LoRA fine-tuning the 1.7B takes ~1-2 days for all 13 adapters, the 0.6B trains all 9 adapters in hours.

**LoRA-to-deployment conversion** (three paths):
1. **Unsloth auto-export**: `save_pretrained_gguf()` + `save_pretrained_ollama()` — auto-generates Modelfile with correct Qwen ChatML template.
2. **Manual merge**: PEFT `merge_and_unload()` → `llama.cpp/convert_hf_to_gguf.py` → Modelfile → `ollama create`. For merged-model deployment.
3. **Adapter-only GGUF**: `llama.cpp/scripts/convert_lora_to_gguf.py` → reference via vLLM `--lora-modules`. For per-request adapter selection.

**Estimated cost**: ~$80-200 total teacher inference API calls (primarily for A10-A14 structured enum distillation). ~4-8 hrs total local GPU LoRA training + ~4-8 hrs base fine-tune on RTX 5090.

---

## 8. Execution Priority

With 22 adapters across 3 patterns, the old 3-tier priority (Plan/Code → Retrieval → Planning DPO) is obsolete. The new ordering principle: **train the simplest, most data-rich adapters first to validate the infrastructure; train the highest-impact adapters second; train the most data-scarce adapters last.**

This inverts the old priority where planning was Tier 1. Planning is no longer a monolithic base-model concern — it's decomposed into 7 per-task adapters (A6, A7, A10-A13, A15) whose data scarcity puts them later in the queue.

### 8.1 Priority Tiers

**Tier 1 — Binary judgment adapters (A1-A9): train first.**

| Priority | What | Why first |
|----------|------|-----------|
| 1a | Base model fine-tunes (0.6B at 4K, 1.7B at 32K) | All LoRAs stack on top. Coding style + context reduction + IN2. Two runs. Outsource if needed. |
| 1b | Plan validation harness infrastructure | Everything downstream depends on this: auto-venv, commit filtering, pipeline integration. Engineering time, no GPU time. |
| 1c | Binary judgment adapters (A1-A9) | Simplest tasks, most data, fastest to train. Validate 0.6B viability early. If 0.6B fails binary judgment, the two-tier cascade collapses — discover this in week 1, not week 5. Ground truth from synthetic pipeline runs is sufficient — no teacher API calls needed. |

**Why binary first**: Binary tasks have three unique advantages: (1) highest data volume (10-100+ per pipeline run), (2) clearest quality signal (binary outcome is unambiguous), (3) fastest to train (0.6B, 8-16 rank, tiny context). Training them first validates the entire LoRA infrastructure (training scripts, evaluation pipeline, adapter deployment, vLLM integration) on the easiest possible targets. Any infrastructure bugs surface here, where iteration is cheap.

**Tier 2 — Design/generation adapters (A15-A22): train second.**

| Priority | What | Why second |
|----------|------|------------|
| 2a | A17 Implement + A18 Function implement | Highest impact on output quality. Most training data available (CommitPackFT, OpenCodeInstruct, SRI Tuning). Binary quality signal (validation pass or fail). |
| 2b | A15 Step design + A19 Test implement | Core planning and testing. DPO pairs from harness. |
| 2c | A16 Header gen + A20 Adjustment + A21 Documentation + A22 Refilter | Supporting tasks. Lower impact, lower priority. |

**Why design/generation second**: These produce the pipeline's visible output — code edits, plans, tests. Improving them directly improves task success rate. Cold-start datasets provide immediate training data volume without teacher API calls. The harness (built in Tier 1b) supplies execution-validated DPO pairs.

**Tier 3 — Structured enumeration adapters (A10-A14): train last.**

| Priority | What | Why last |
|----------|------|----------|
| 3a | A14 Task analysis | Smallest stage, lowest risk. Indirect quality signal. Defer until retrieval adapters (A1-A5) are stable. |
| 3b | A10 Change point enum + A11 Part grouping | Lowest data volume (~1 per pipeline run). Most reliance on teacher distillation. Train only after teacher API budget is allocated and harness is generating enough examples. |
| 3c | A12 Symbol targeting + A13 Interface enum | Moderate data, moderate impact. Train after A10-A11 provide structural planning capability. |

**Why structured enum last**: These are the most data-starved tasks, the most reliant on teacher distillation, and the ones where quality is hardest to evaluate automatically. Training them last means: (1) the harness has been running long enough to accumulate organic examples, (2) teacher distillation campaigns have generated volume, (3) downstream adapters (A15-A22) are already trained, so the quality signal for structured enums is cleaner (good enum + good execution = good plan; good enum + bad execution = execution problem).

### 8.2 Decision Gates

Do not advance to the next tier until the current tier's adapters pass evaluation ([Section 9](#9-evaluation-methodology)). Specifically:

- **Gate 1→2**: Base fine-tunes validated on held-out data. All 9 binary adapters show measurable improvement over base model on held-out binary judgment tasks. 0.6B viability confirmed (or rejected, triggering cascade collapse to 1.7B-only). Harness infrastructure operational.
- **Gate 2→3**: A17 and A18 show measurable improvement in pass@1 on held-out tasks. Pipeline end-to-end success rate (plan → implement → validate) is stable enough to generate reliable quality signals for structured enum tasks.

### 8.3 Estimated Timeline

| Tier | Engineering time | GPU time | Bottleneck |
|------|-----------------|----------|------------|
| 1 | 2-3 weeks | ~3-6 hrs | Harness infrastructure (1b). Base fine-tunes (1a). Binary adapter training is trivial. |
| 2 | 2-3 weeks | ~3-6 hrs | Data curation for A17/A18. DPO pair generation via harness. |
| 3 | 1-2 weeks | ~1-3 hrs | Teacher distillation campaigns for A10-A14. Volume is the bottleneck. |

**Total**: 5-8 weeks of engineering time. Faster than the old strategy because: (1) binary adapters train in hours, not days, (2) no monolithic planning base fine-tune, (3) harness automates DPO pair generation. The main bottleneck shifts from "planning capability in the base" to "teacher distillation volume for structured enum tasks."

---

## 9. Evaluation Methodology

Training without evaluation is flying blind. Each of the 22 tasks needs a held-out evaluation set with clear pass/fail criteria, built *before* the first training run.

### 9.1 Per-Task Evaluation Sets

**Binary judgment adapters (A1-A9):**

Binary tasks have natural evaluation: precision and recall on held-out judgment decisions. The ground truth is derivable from commit diffs.

| # | Task | Eval set source | Size | Primary metric | Pass threshold |
|---|------|----------------|------|----------------|----------------|
| A1 | Scope judgment | Synthetic: task + file → was file in diff? | 500-1K | Precision + recall | ≥ 85% recall, ≥ 70% precision |
| A2 | Precision pass 1 | Synthetic: task + symbol → was symbol in diff? | 500-1K | Precision + recall | ≥ 80% recall, ≥ 65% precision |
| A3 | Precision pass 2 | Synthetic: task + pass-1 survivor → was it primary? | 300-500 | Accuracy | ≥ 75% |
| A4 | Precision pass 3 | Synthetic: task + non-primary → was it supporting? | 300-500 | Accuracy | ≥ 75% |
| A5 | Similarity judgment | Synthetic: function pairs → were they co-modified? | 300-500 | Precision + recall | ≥ 70% both |
| A6 | Part dependency | Synthetic: part pairs → was dep edge correct? | 200-400 | Accuracy | ≥ 80% |
| A7 | Step dependency | Synthetic: step pairs → was dep edge correct? | 200-400 | Accuracy | ≥ 80% |
| A8 | KB pattern relevance | Synthetic: stub + KB section → domain match? | 200-400 | Precision + recall | ≥ 70% both |
| A9 | Routing | Synthetic: task + stage → was stage useful? | 200-400 | Accuracy | ≥ 75% |

**Structured enumeration adapters (A10-A14):**

These require set-level evaluation: did the enumeration cover all actual items?

| # | Task | Eval set source | Size | Primary metric | Pass threshold |
|---|------|----------------|------|----------------|----------------|
| A10 | Change point enum | Commit diffs → expected change points | 100-200 | Change point recall | ≥ 80% |
| A11 | Part grouping | Multi-file commits → expected groupings | 50-100 | Grouping coherence (no cross-part entanglement) | Subjective + downstream success |
| A12 | Symbol targeting | Parts → expected symbols | 100-200 | Symbol recall | ≥ 80% |
| A13 | Interface enum | Feature commits → expected interfaces | 50-100 | Compile success rate | ≥ 70% |
| A14 | Task analysis | Commit messages → expected file targets | 200-500 | File target recall | ≥ 80%, ≥ 60% precision |

**Design/generation adapters (A15-A22):**

These need end-to-end evaluation: does the generated output work?

| # | Task | Eval set source | Size | Primary metric | Pass threshold |
|---|------|----------------|------|----------------|----------------|
| A15 | Step design | Harness: held-out commits | 50-100 | Step execution success rate | ≥ base model |
| A16 | Header generation | C repos: held-out headers | 50-100 | Compile success rate | ≥ 80% |
| A17 | Implement | SWE-bench Verified + held-out diffs | 100-300 | pass@1 | Measurable improvement over base |
| A18 | Function implement | Held-out function bodies | 100-200 | Compile + test pass rate | ≥ base model |
| A19 | Test implement | Held-out test commits | 50-100 | Tests compile + catch known-bad | ≥ base model |
| A20 | Adjustment | Held-out adjustment scenarios | 30-50 | Revised steps succeed rate | ≥ base model |
| A21 | Documentation | Held-out doc commits | 50-100 | AST doc verification pass | ≥ 80% |
| A22 | Refilter | Held-out oversized contexts | 50-100 | Execute success with kept subset | ≥ base model |

**Eval set construction** depends only on Phase 1 (indexer), the bootstrapping pipeline ([Section 6](#6-bootstrapping-from-external-repo-history)), and the plan validation harness ([Section 6.9](#69-plan-validation-harness)). Reserve 10-20% of harness-qualifying repos exclusively for evaluation — no overlap with training data.

### 9.2 Evaluation Protocol

**Before every training run:**
1. Run the eval set against the current best adapter (or base model for the first run). Record baseline metrics.

**After every training run:**
2. Run the same eval set against the new adapter. Record metrics.
3. Compare against baseline. If any primary metric regresses, the new adapter is rejected.
4. If metrics improve, promote the new adapter to active. Archive the old adapter (do not delete).

**Per-task regression detection:**
- Each task's eval is independent. An improvement in A17 does not excuse a regression in A1.
- End-to-end pipeline eval (plan → implement → validate on held-out tasks) runs after any individual adapter change, to catch cross-task interaction effects.
- With 22 adapters, interaction effects are a real concern. A binary adapter (A1) that becomes too aggressive on scope inclusion changes what the downstream enum (A10) and design (A15) adapters see. The end-to-end eval is the safety net.

### 9.3 What the Eval Set Cannot Catch

Some failure modes are invisible to automated evaluation:

- **Structural planning subtlety**: A10 (change point enum) that scores well on recall but produces change points at the wrong granularity (too fine → too many parts; too coarse → monolithic parts). Mitigated by downstream success metrics, but requires spot-checking.
- **Style drift**: the model produces correct code that violates coding style principles. Mitigated by the fail-fast corpus training, but should be spot-checked manually.
- **Distribution shift**: the eval set may not cover the task distribution the agent encounters in real use. Mitigated by repo diversity in eval set construction.

Manual review of a random sample (5-10 examples per task per training round) is not optional. With 22 tasks, this means 110-220 manual reviews per round — a meaningful time investment, but necessary.

---

## 10. Self-Improvement Guardrails

Empirical limits on self-improvement at small scale, informed by recent research:

> **Revision (Feb 2026):** The original "sub-6B fails to bootstrap" and "sub-7B plateaus after ~2 iterations" guardrails assumed monolithic planning prompts with high cognitive complexity per call. Planning decomposition (binary sub-tasks via `run_binary_judgment()`) fundamentally changes this: each LLM call now produces 1 bit of information (binary judgment) or a small structured output (enumeration, grouping), not a complex multi-file plan. A 1.7B model producing reliable binary judgments is a much easier problem than a 4B model producing full plans in one call. The guardrails below are revised to account for this, but empirical validation during Phase 4 bootstrapping is still needed.

**Hard limits**:
- Self-improvement plateaus after ~2 iterations at small scale (SWE-Gym, ICML 2025), though planning decomposition may extend this ceiling by reducing per-call complexity.
- Sub-2B models may struggle with structured outputs (A10-A14) even after decomposition — the 1.7B's viability for these tasks needs empirical validation. Binary classification (0.6B, A1-A9) is lower risk.
- Recursive training on synthetic data leads to model collapse (Shumailov et al., Nature 2024) without careful data management.
- Full GRPO/PPO training is NOT feasible on consumer hardware. DPO from self-generated trajectories is practical.

**Mandatory practices**:
- **Accumulate data, never replace**: each training round adds new examples to the pool; never discard previous rounds.
- **Always train from frozen base, never merge-then-retrain**: LoRA adapters are trained independently from the base model checkpoint. Never merge an adapter into base weights and then train a new adapter on the merged model.
- **10-20% replay data**: every training batch includes examples from earlier rounds to prevent catastrophic forgetting.
- **Per-task self-improvement**: with 22 adapters, self-improvement can proceed independently per task. A binary adapter (A1) can enter self-improvement while a design adapter (A17) is still on teacher data. This is more granular than the old "stage-level" self-improvement.

**Timeline** (ordered by increasing risk):
1. Pure teacher distillation (structured enum A10-A14 + design/generation A15-A22) and ground-truth labels (binary A1-A9).
2. Mixed: on-policy data with teacher scoring (binary tasks first — simplest to validate).
3. Execution feedback loop (code validation as signal for A17, A18 LoRAs).
4. Planning decomposition self-improvement: student adapters generate per-task outputs through the harness, successes/failures become training data per adapter. Same harness infrastructure, same validation.

**Practical thresholds**:
- ~500 quality trajectories can produce measurable gains (SWE-Gym finding). With 22 tasks generating ~6-10 training pairs per harness run, 500 trajectories yield ~3,000-5,000 per-task training pairs — well above the minimum for most adapters.
- Repo diversity matters more than volume for transfer (Hybrid-Gym, 2026). The harness makes adding repos trivial.
- SelfCodeAlign (NeurIPS 2024) achieves self-alignment from 3B to 33B without teacher. Suggests a hybrid: teacher distillation initially, on-policy data once baseline capability is established.

### 10.5 Adapter Versioning and Rollback

A training round that makes things worse must be reversible without retraining. With 22 adapters, versioning discipline is critical.

**Versioning scheme:** Every trained adapter gets a version tag: `{task}-v{N}` (e.g., `scope-v1`, `implement-v3`, `change-point-enum-v2`). The version number increments monotonically. The curated DB `adapter_metadata` table tracks: task name, version, training timestamp, eval metrics snapshot, active/inactive status, file path to adapter weights.

**Active adapter selection:** Exactly one adapter version per task is marked `active` in curated DB. The `ModelRouter` resolves task → active adapter at runtime. Switching the active adapter is a single DB update — no model reloading needed if using vLLM with multiple adapters loaded.

**Rollback protocol:**
1. After a training run, the evaluation protocol ([Section 9.2](#92-evaluation-protocol)) compares the new adapter against the current active version.
2. If any primary metric regresses, the new adapter is marked `inactive`. The previous version remains `active`.
3. If regression is detected after promotion (e.g., end-to-end eval catches a cross-task interaction), rollback to the previous version is immediate.

**Retention policy:** Never delete previous adapter versions. Disk cost for LoRA adapters is trivial (~20-50 MB each, ~440 MB total for all 22 at one version). Retraining cost is not. Retain all versions indefinitely for regression analysis, recovery from compounding errors, and audit trail.

**Base model checkpoints** follow the same versioning. Context-reduced bases are tagged `{model}-{context}-base-v{N}` (e.g., `0.6b-4k-base-v1`, `1.7b-32k-base-v1`). If a base needs to be retrained, all LoRA adapters on the old base are invalidated — they must be retrained on the new base.

---

## 11. Evidence Base and Advanced Techniques

**Per-stage LoRA validation**: IBM Granite Intrinsics Library demonstrated 6 LoRA adapters for RAG pipeline stages with no cross-stage degradation, plus **Activated LoRA (aLoRA)** achieving 20-35x speedup per adapter invocation via KV cache reuse. META-LoRA showed benefits more pronounced for small models. **R-LoRA** (EMNLP 2025 Findings) explicitly tested on **Qwen2.5-3B**, showing multi-head LoRA outperforms vanilla LoRA on multi-task benchmarks. MTL-LoRA found task-specific transformation matrices consistently outperform both single-task and vanilla multi-task LoRA, mitigating the "seesaw effect." Note: no published project has done per-stage LoRA for coding agent pipelines — this is genuinely novel. Scaling from 6 adapters (Granite) to 22 is the primary research risk.

**Adapter routing**: LORAUTER (January 2026) routes queries to appropriate adapters via task embeddings, scaling to 1500+ adapters. LoGo (November 2025) performs training-free, instance-specific selection and merging from a LoRA pool. For our pipeline where task routing is deterministic (the orchestrator knows which task it's running), simple adapter selection via the vLLM `model` parameter is sufficient — no learned routing needed.

**Relevance judgment training**: **Self-RAG** (ICLR 2024) trains models with special reflection tokens (`[Relevant]`, `[Irrelevant]`, etc.) using GPT-4-generated critic labels — applicable to binary judgment tasks A1-A5. **ARES** (NAACL 2024) fine-tunes LLM judges on triples with only ~150 human-annotated validation datapoints — relevant to the data-scarce structured enum tasks A10-A14.

**Edit-sequence training**: **LintSeq** (ICLR 2025) decomposes complete programs into synthetic edit sequences using a linter (no LLM needed), producing instruction + file-state + diff-sequence tuples. The **D3 dataset** (3.6M examples) extends LintSeq with LLM-powered instruction labeling. **SRI Tuning** (January 2026) trains models to generate search-and-replace edit instructions — directly aligned with A17's output format.

**Self-improvement at scale**: **SWE-Gym** (ICML 2025) found plateau after two iterations. **SWE-smith** (NeurIPS 2025) extended to 50K task instances. **R2E-Gym** (COLM 2025) derives training environments from version-control commits — closest system to our commit-history bootstrapping. **Hybrid-Gym** (February 2026) found repo diversity more important than volume. **Lingma SWE-GPT** (Alibaba) trained 7B/72B Qwen models, with the **7B model resolving 18.20% of SWE-bench Verified** — proof small models can work with proper training.

**Small-model bootstrapping**: **SelfCodeAlign** (NeurIPS 2024) achieves self-alignment without teacher from 3B to 33B. **LADDER** improved Llama 3B from 2% to 82% on integration problems using generate→solve→verify→learn with explicit curriculum. The **Code Graph Model** (May 2025) fine-tunes LLMs with LoRA to understand repository-level structure.

**SAD (Structured Agent Distillation)**: If SFT + DPO underperforms for structured enum tasks (A10-A14), segment output into `[STRUCTURE]` and `[CONTENT]` spans with different loss weights. Apply curriculum learning (simple enumerations → complex multi-file decompositions).

---

## 12. Deployment Lifecycle

The system goes through four phases before reaching its final operating state. Each phase has distinct infrastructure, network access, and training capabilities.

### 12.1 Phase Overview

```
Development Machine          External Training        Air-Gapped Machine
(connected, current)         (outsourced GPU)         (final deployment)
        │                          │                         │
   Build Phase 4 ──────────► Full fine-tune(s) ─────►       │
   Build harness               (0.6B + 1.7B)                │
   Validate under supervision      │                         │
   Teacher distillation            │                         │
   (A10-A14 primarily)             │                         │
   Harness campaigns               │                         │
        │                          │                         │
        └── Migration (one-time transfer) ──────────► System operational
                                                             │
                                           Self-improvement loop (student-only)
                                           22 per-task adapters iterate independently
                                           New repos via USB (ongoing)
```

### 12.2 Development Phase (Connected Machine)

**Network access**: full internet. **Purpose**: build, validate, and front-load teacher signal.

1. **Implement Phase 4** — self-training infrastructure: harness pipeline, data curation, training scripts, evaluation methodology for 22 tasks.
2. **Teacher distillation campaigns** — all teacher API calls happen during this phase. Focus on A10-A14 (structured enum, most data-starved). Binary tasks (A1-A9) rely on pipeline outcome labels. Design/generation tasks (A15-A22) use cold-start datasets + harness.
3. **Harness validation** — run teacher-driven and student-driven harness campaigns. Verify per-task quality isolation.
4. **Curate initial repo corpus** — clone, filter, package repos for transfer.

### 12.3 Outsourced Training

**Network access**: full internet (external machine). **Purpose**: full fine-tune iterations that exceed the development machine's GPU capacity.

Two full fine-tune runs:
1. **0.6B base (4K context)** — coding style + context reduction + IN2.
2. **1.7B base (32K context)** — coding style + context reduction (if needed) + IN2.

**LoRA training** does not require outsourcing — QLoRA on 0.6B and 1.7B models fits comfortably on consumer GPUs.

### 12.4 Migration

**One-time transfer** of the complete system to the air-gapped machine.

**Migration checklist — what goes onto the air-gapped machine:**

| Category | Contents |
|----------|----------|
| **Pipeline code** | Phases 1-4 source, all dependencies vendored or pre-installed |
| **Models** | Fine-tuned 0.6B base (4K) + 1.7B base (32K) + generalist 1.7B (full window) + all 22 LoRA adapters + base checkpoints |
| **Training tooling** | Unsloth, PEFT, PyTorch, CUDA toolkit — all pinned versions, fully installed |
| **Inference server** | vLLM (primary, 22 adapters) + Ollama (fallback) |
| **Initial repo corpus** | Packaged repos with pre-cached wheels and manifests |
| **Databases** | Raw DB (training corpus from development phase), curated DB (indexed repos + adapter metadata for 22 tasks) |
| **Evaluation sets** | Held-out harness commits for ongoing per-task evaluation |

### 12.5 Post-Migration Operations

**Network access**: none. **Purpose**: autonomous self-improvement from new repo data.

- **Inference**: vLLM serves both base models + active LoRA adapters (up to 22 concurrent).
- **Pipeline execution**: Phases 1-3 operate on real tasks.
- **Self-improvement**: student-driven harness on pre-loaded repos. Per-task LoRA training locally. Per-task evaluation locally. Adapter versioning and rollback locally.
- **No teacher access**: all teacher distillation happened pre-migration. Post-migration training is student-only.
- **Per-task iteration**: each of the 22 adapters can be retrained independently. A binary adapter can iterate daily; a design adapter iterates on multi-day cycles.

### 12.6 New Data Ingestion Protocol

New curated repos are the only external input post-migration. They enter through a dedicated air-gapped laptop that serves as a hardware data diode.

**Protocol**:
1. **Preparation (connected side)**: clone repo, run commit filtering, collect dependency wheels, generate manifest, package.
2. **Transfer to USB**: write the repo package to a USB drive. Only the repo package — no executables.
3. **Air-gapped laptop**: reads the USB, writes to the air-gapped machine's storage. No persistent storage on the laptop.
4. **Air-gapped machine**: validates the manifest, creates venv from cached wheels, integrates into harness corpus. New commits become available for per-task self-improvement campaigns.

**What does NOT go through the USB**: model weights, training scripts, pipeline code, configuration changes, software updates. The system is frozen at migration time. Only training data (curated repos) enters.

**Repo package validation**: manifest schema check, wheel completeness, test command dry-run, no unexpected file types.
