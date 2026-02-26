# LoRA Training Strategy

Per-stage LoRA training targets, base model full fine-tuning (coding style + role specialization + planning capability + context window reduction + positional robustness), deployment architecture, training data sources (bootstrapping + self-improvement + cold-start datasets + plan artifacts + plan validation harness), distillation strategy, execution priority, evaluation methodology, training infrastructure, and self-improvement guardrails.

---

## 1. Per-Stage LoRA Training Targets

**Qwen3-4B LoRAs (reasoning/retrieval):**

| Stage | Training Pair | Quality Signal | Technique | Rank | Examples |
|-------|--------------|----------------|-----------|------|----------|
| Task Analysis | task description -> structured task analysis | did retrieval find the right files? | SFT | 16-32 | 3-5K |
| Scope | task + candidates -> relevance classification | were the included files actually needed? | SFT | 16-32 | 2-4K |
| Precision | task + symbols -> tier classification | did the selected symbols lead to correct code? | SFT + curriculum | 16-32 | 3-5K |

**Qwen2.5-Coder-3B LoRAs (coding):**

| Stage | Training Pair | Quality Signal | Technique | Rank | Examples |
|-------|--------------|----------------|-----------|------|----------|
| Execute (Code) | context + task -> code edits | did the code pass validation? | SFT + DPO | 32-64 | 10-50K |

**Qwen3-4B optional refinement LoRA (planning DPO):**

| Stage | Training Pair | Quality Signal | Technique | Rank | Examples |
|-------|--------------|----------------|-----------|------|----------|
| Planning DPO | context + task -> plan (accept/reject pairs) | plan validation harness outcome ([Section 6.9](#69-plan-validation-harness)) | DPO | 16-32 | 1-5K preference pairs |

**Note on planning capability**: Planning (meta-plan decomposition, step-plan sequencing, test-plan coverage) is the reasoning model's primary role. Planning capability lives in the **base model full fine-tune** ([Section 2](#2-base-model-full-fine-tuning)), not in per-stage LoRA adapters. The reasoning model learns all three planning types as part of role specialization, eliminating adapter-swapping during the planning-heavy orchestrator loop. The optional DPO LoRA above is a refinement layer — preference alignment on top of an already-capable base — trained only if base fine-tune evaluation shows planning quality needs improvement.

**Training pair format**: NL-to-Code instruction format (natural language description → code) significantly outperforms Code-to-Code format across all benchmarks (OpenCodeInstruct, 2025). All training pairs — including synthetic pairs from commit histories — should use natural language task descriptions as input, not code-in → code-out.

Training data curation must separate examples by base model -- a Qwen3-4B reasoning LoRA cannot be applied to Qwen2.5-Coder-3B and vice versa.

---

## 2. Base Model Full Fine-Tuning

Before LoRA adapters are trained, each base model undergoes a full fine-tune. This is not optional — it prepares the base for its specific role in the pipeline. The full fine-tune combines five objectives in a single training run, all compatible since they don't conflict:

1. **Coding style** — fail-fast error handling, no defensive patterns, narrow exception clauses, rich contextual error messages. Trained from the fail-fast corpus ([Section 6.7](#67-fail-fast-training-corpus)).
2. **Role specialization** — bias the base toward its pipeline role (planning, coding, generalist/retrieval). Different base models may receive different data mixes emphasizing their target task distribution.
3. **Planning capability** — the reasoning model's primary role is planning. All three planning types are trained as part of the base: **meta-plan** (task decomposition into independent parts with dependency edges), **step-plan** (detailed implementation step sequencing within a part), and **test-plan** (test coverage planning given code changes). This is not a LoRA concern — planning is the model's core competency, and baking it into the base (a) uses all planning examples to update all parameters rather than low-rank projections, maximizing learning from scarce planning data, (b) eliminates adapter-swapping during the planning-heavy orchestrator loop (meta_plan → N×part_plan → N×test_plan → adjustments), and (c) ensures every retrieval LoRA stacked on top inherits planning-aware attention patterns for free. Training data comes from three sources: synthetic plans from commit diffs ([Section 6.3](#63-synthetic-pipeline-run-generation)), real plan artifacts from collaborative repos ([Section 6.8](#68-existing-plan-artifacts-from-collaborative-repos)), and execution-validated plans from the plan validation harness ([Section 6.9](#69-plan-validation-harness)). Teacher distillation ([Section 7](#7-distillation-strategy)) provides volume. See [Section 2.6](#26-planning-capability-training-details) for planning-specific training details.
4. **Context window reduction** — reduce `max_position_embeddings` and retrain RoPE frequencies to match each role's actual operating range. This is a free rider on the training run that produces compounding gains when LoRAs are stacked on top. See the rationale and detailed analysis below.
5. **Positional robustness (IN2 training)** — train the model to attend to critical information at any position within the reduced window, not just the beginning and end. IN2 (INformation-INtensive) training synthesizes QA data where answers require attending to specific ~128-token segments scattered throughout the context. Applied to Mistral-7B, this produced FILM-7B with positional robustness comparable to GPT-4-Turbo while maintaining short-context performance. The root cause of lost-in-the-middle degradation is insufficient training supervision — IN2 directly addresses it. For the pipeline, curated context has no natural "important stuff first" ordering; symbols from different files appear in whatever order assembly produces. The model must attend uniformly across the window.

### 2.1 Why Context Window Reduction Matters

**Shorter context windows produce better output.** This is not an efficiency claim — it is a quality claim backed by converging empirical evidence (see `research_reviews/context_window_size_research.md` for the full literature review):

- Context length alone degrades performance **13.9–85%** even when retrieval is perfect and irrelevant tokens are completely masked (Du et al., 2025). The model isn't distracted by noise — the sheer computational burden of longer sequences interferes with reasoning.
- Most LLMs effectively utilize only **10–20%** of their context on reasoning tasks (BABILong, NeurIPS 2024). Open-source models demonstrate effective context of **less than 50%** of training length (RULER, COLM 2024).
- At 32K tokens, **11 of 12 tested models dropped below 50%** of their short-context performance. The effective context length maintaining ≥85% of baseline was **≤2K tokens** for most models (NoLiMa, 2025).
- RAG + Llama-3-8B (8K context) achieved perfect retrieval accuracy up to 2M tokens, while GPT-4o failed beyond 128K. OP-RAG achieved **38% better F1 with 60% fewer tokens** than full-context Llama3.1-70B.

The industry's race to extend context windows — 4K to 128K to 256K — has outpaced models' ability to use that context. For this project, context reduction is not a cost optimization. It is the mechanism by which the pipeline produces better results.

**This is the same thesis as the N-prompt pipeline itself.** The pipeline's core claim is that a 32K window at ~100% signal relevance beats a 200K window at 10-15% utilization. The research above validates this claim empirically: shorter, focused contexts paired with intelligent retrieval outperform naive long-context approaches on most practical tasks. Context reduction at the model level and context curation at the pipeline level are the same strategy applied at different layers.

**Why compression techniques are architecturally incompatible:** The research literature offers a rich toolkit for context compression — hard prompt compression (LLMLingua), soft token methods (gisting), KV-cache surgery (SnapKV). These are impressive engineering, but they are incompatible with this project's transparency architecture. Every compression technique inserts an opaque decision layer between the retrieval pipeline and the model: a BERT model silently dropping tokens, learned vectors replacing readable text, attention selectively masking KV pairs. The traceability chain breaks — logged context no longer equals received context, which means the self-improvement loop trains on corrupted pairs. The training pipeline would see "the model received X and produced Y" when the model actually received "X minus whatever the compressor removed." Every quality signal, every feedback loop, every evaluation metric assumes that what was logged is what the model saw. Compression violates that assumption. For systems that don't train from their own logs, compression is a powerful optimization. For this system, it is architecturally incompatible. Context reduction must happen at the model level (RoPE retraining) and the pipeline level (curated retrieval with logged decisions), not at an intermediate compression layer.

**What the pipeline actually uses:**

| Role | What goes in the window | Actual content range | Utilization at 64K | Utilization at 128K |
|------|------------------------|---------------------|--------------------|---------------------|
| Task Analysis | task description + repo file tree + environment brief | ~2-8K | 3-12% | 2-6% |
| Scope | task summary + candidate files + judgment prompt | ~8-24K | 12-37% | 6-19% |
| Precision | task summary + file symbols + classification prompt | ~8-24K | 12-37% | 6-19% |
| Execute — Plan | full curated ContextPackage + plan prompt | ~16-32K | 25-50% | 12-25% |
| Execute — Code | full curated ContextPackage + code edit prompt | ~16-32K | 25-50% | 12-25% |

Even Execute — the heaviest stage — uses at most 50% of a 64K window. Task Analysis uses 3-12%. Given the evidence above, this is not waste to be tolerated — these stages are *operating in the range where models perform best*. The pipeline architecture already targets the sweet spot. Context reduction at the model level sharpens this further by training the model's attention patterns for the actual operating range rather than a 4-16x larger range it will never see.

**Emergent codebase discipline:** Tight context windows don't just constrain the model — they constrain the code the model can operate on. If an Execute stage has a 32K budget and must fit multiple files plus framing plus the task description, any single file that consumes 20K of that budget is a pipeline-breaking problem. The system naturally pressures toward small, focused files because that's what fits through the pipeline. This creates a reinforcing loop: small files → better retrieval precision (less noise per candidate) → tighter context assembly → which in turn rewards small files. The architecture enforces the coding style it needs to function well, without requiring a linter rule or style guide to mandate it.

**What unused context capacity costs (beyond the quality degradation above):**

1. **Wasted attention computation.** Self-attention is O(n²) in sequence length. A 4096-token model requires 64x less computation than a 32K-token model. In practice, 16x context increase produces 50x latency increase (Adnan et al., 2024). The model's learned attention patterns were shaped during pretraining to distribute across the full positional range — when 75-95% of those positions are always empty, the distributions don't match operating conditions.

2. **Diluted positional resolution.** RoPE positional encodings are frequency-distributed across the full `max_position_embeddings` range. A model trained for 128K positions spreads its positional resolution thinly over a range it will never use, rather than concentrating it in the 8-32K range where all the actual content lives.

3. **Wasted KV cache.** KV-cache memory scales linearly with sequence length. At 3-4B scale the absolute savings from reducing 128K → 32K are modest (~200-400 MB), but the relative savings matter for concurrent adapter serving — fitting 4 adapters in VRAM instead of 2 is a 2x throughput gain.

4. **Looser LoRA fit.** A LoRA adapter fine-tuned on top of a base model inherits the base's attention patterns. If the base was trained for 128K but only ever sees 8-32K during LoRA training, the adapter is compensating for a positional distribution mismatch rather than focusing entirely on the task. A LoRA on a context-optimized base starts from a better foundation.

### 2.2 Per-Role Context Window Targets

| Role | Actual max content | Target `max_position_embeddings` | Headroom | Rationale |
|------|-------------------|----------------------------------|----------|-----------|
| Task Analysis | ~2-8K | 8K | 0-300% | Smallest stage. Only task + metadata. |
| Scope / Precision | ~8-24K | 32K | 33-300% | Candidate batches vary. 32K covers worst case. |
| Execute — Plan | ~16-32K | 32K | 0-100% | Full curated context. Tightest fit. |
| Execute — Code | ~16-32K | 32K | 0-100% | Full curated context. Same as Plan. |

The targets provide enough headroom for worst-case inputs while eliminating the 2-8x overcapacity of stock base models.

**Shared bases:** Scope, Precision, and both Execute roles share the 32K target, meaning they can share a single context-reduced base. Task Analysis at 8K gets its own. This means two full fine-tune runs total (8K base + 32K base), not five.

### 2.3 Why Context Reduction Compounds with LoRAs

- The base model's attention patterns are retrained for the actual operating range. Every subsequent LoRA inherits these tighter patterns for free.
- Smaller KV cache per layer — the model physically cannot attend beyond the new limit, freeing VRAM permanently. The freed VRAM goes to larger batch sizes, more concurrent LoRA adapters in vLLM, or reduced swapping.
- A LoRA fine-tuned on a context-optimized base learns task-specific behavior without also needing to compensate for positional distribution mismatch. This is a strictly better starting point than a LoRA on a generic 128-256K base.

### 2.4 Practical Notes

- **Irreversible per base.** The model loses long-context capability. Retrain from original weights if the target range needs to change. This means base checkpoints must be versioned — if a future pipeline change requires a stage to see more context, the checkpoint is the recovery path.
- **Shrink ratio matters.** 128K → 32K (4x reduction) is well within safe range. 128K → 8K (16x) shifts the RoPE frequency distribution significantly — validate on a held-out set before committing. Ablate 8K vs 16K for Task Analysis before choosing the aggressive target.
- **Unsloth supports full fine-tuning** on 24GB VRAM for 3-4B models. Training time is hours, not minutes — but this is a one-time cost per base model release, amortized across all LoRA adapters trained on top of it.

> **Revision note (Feb 2026):** The specific `max_position_embeddings` targets above are based on current Qwen2.5/Qwen3 model specifications. When Qwen 3.5 small models release (expected with 64K+ base context), the reduction ratios and RoPE retraining parameters will need recalculation. The architecture and rationale are stable; the specific numbers are not. Revisit this section at that time.

### 2.5 Full-Window Generalist

One base model is kept at the original `max_position_embeddings` (128-256K) without context reduction. This model receives the coding style and generalist role fine-tuning but retains the full context window. It serves three purposes that context-reduced specialists cannot:

1. **Audit** — feed an entire pipeline run's logs into one context window to verify the traceability chain end-to-end. Specialists at 8-32K cannot hold a full run's worth of decisions.
2. **A/B testing** — compare specialist output against the unrestricted base to validate that context reduction actually improves (or at least doesn't degrade) quality. Without this baseline the compounding claim is unmeasured. This is the critical validation step: if context-reduced models don't outperform the generalist on their stage's task, the reduction is not justified.
3. **Fallback** — if a task exceeds a specialist's reduced window, route to the generalist rather than failing or retraining.

The generalist does not replace specialists in the pipeline — it runs alongside them as infrastructure. It can also serve as the base for future LoRA experiments where the full context range is needed.

### 2.6 Planning Capability Training Details

Planning is the reasoning model's dominant task. The orchestrator loop makes more planning LLM calls per task than any other type: meta_plan, N×part_plan, N×test_plan, N×adjustment. Including planning in the base fine-tune rather than as separate LoRA adapters is an architectural decision with specific implications.

**Three planning types, one base:**

| Planning Type | Input | Output | Core Reasoning |
|---------------|-------|--------|----------------|
| Meta Plan | full task + curated context | `MetaPlan` (parts with dependency edges) | Task decomposition, scope partitioning, dependency analysis |
| Step Plan | part description + curated context | `PartPlan` (ordered implementation steps) | Step sequencing, file targeting, change ordering |
| Test Plan | part description + code diff + curated context | `PartPlan` (test steps) | Coverage analysis, edge case identification, test strategy |

All three share deep reasoning substrate (understanding code structure, estimating change scope, ordering operations, identifying risks) but have distinct input distributions and output targets. The full fine-tune's all-parameter updates can represent all three without the rank constraints of LoRA.

**Why not separate LoRAs per planning type:**

1. **Data scarcity.** Planning data is the hardest to source. Splitting 8-15K examples across 2-3 LoRAs means 3-5K per adapter in low-rank projections. The full fine-tune uses all examples to update all parameters — strictly more learning per example at sub-7B scale.
2. **Deployment cost.** Separate planning LoRAs require adapter swaps during the orchestrator loop (meta_plan → part_plan → code → test_plan → test_code → adjustment, per part). With planning in the base, adapter swaps are reserved for the genuinely different tasks: retrieval judgment LoRAs for scope/precision, and the coding LoRA on a different base model entirely.
3. **Shared reasoning depth.** The three planning types share more reasoning substrate than they differ. Unlike scope-vs-precision (genuinely different classification tasks), meta-plan-vs-step-plan are hierarchical levels of the same skill. All-parameter training captures this shared structure; low-rank projections may not.

**When to add the DPO refinement LoRA:**

The optional planning DPO LoRA ([Section 1](#1-per-stage-lora-training-targets)) is trained only if evaluation shows the planning-aware base underperforms on DPO-style preference criteria. It stacks on the frozen planning-aware base — the base provides capability, the LoRA provides preference alignment. This mirrors the SelfCodeAlign finding: base models benefit most from alignment with their own data distribution after baseline capability is established.

**Planning-specific quality signals** differ by type and are tracked separately in the plan validation harness ([Section 6.9](#69-plan-validation-harness)):

| Planning Type | Quality Signal | Measurable? |
|---------------|---------------|-------------|
| Meta Plan | Were parts well-scoped? Did parts need restructuring? | Indirect — downstream step success rate per part |
| Step Plan | Did steps execute cleanly? Were dependencies correct? | Direct — per-step code generation success |
| Test Plan | Did tests catch real issues? Was coverage adequate? | Direct — test pass rate, mutation testing |

---

## 3. Per-Stage Model Overrides and Deployment Architecture

Each pipeline stage can specify a model name override via `[models.overrides]` in config. The stage registry resolves: stage name -> override (if exists) -> role-based default -> base model.

**Phases 1-3 (MVP)**: Ollama or any configured provider. Single model at a time, no adapter switching needed.

**Phase 4 deployment target**: vLLM with per-request LoRA adapter selection. Ollama cannot hot-swap LoRA adapters per request (each adapter is a separate model tag requiring full unload/reload). vLLM's `--lora-modules` flag enables multiple adapters loaded simultaneously with per-request selection via the `model` field in the OpenAI-compatible API.

- **Primary**: vLLM server — quantized base model (~2-2.5 GB VRAM) + up to 8 LoRA adapters (~20-50 MB each) + KV cache = ~3-5 GB total VRAM on a 24 GB GPU.
- **Fallback**: llama-server (llama.cpp) with LoRA support.

The `[models.overrides]` stage→model mapping stays the same regardless of provider — only the inference server changes. Override values become adapter identifiers (e.g., `scope-v1`) that the provider translates to its native adapter selection mechanism.

---

## 4. Raw DB as Per-Stage Training Corpus

Each stage's training data can come from two sources:

**Logged runs (self-improvement)**: Extracted from `retrieval_llm_calls` filtered by:
- `call_type` (which stage produced it)
- linked `task_runs.success` (was the overall task successful?)
- optionally `retrieval_decisions` (was this specific decision part of a successful run?)

**Synthetic pairs (bootstrapping)**: Generated from external repo commit histories via the bootstrapping pipeline ([Section 6](#6-bootstrapping-from-external-repo-history)). These provide training data before the agent has produced any real runs.

Both sources produce datasets in the same format and are stored with a `source` field to allow mixing and weighting during training.

---

## 5. Training Artifact Storage

- Training plans and curated datasets go in raw DB (they're generated outputs, same as enrichment outputs).
- Adapter metadata (which stage, performance metrics, active/inactive) goes in curated DB (it's verified configuration the pipeline reads at runtime).

---

## 6. Bootstrapping from External Repo History

The self-improvement loop has a cold-start problem: Phase 4 training needs quality-signal-rich data, but Phase 3's output quality depends on models that haven't been trained yet. External repos break this dependency.

### 6.1 Data Source

Clone mature, well-tested repos from GitHub. Ideal repos have:
- Clear commit messages (natural task descriptions).
- Small, focused bug-fix and feature commits (clean before/after pairs).
- Stable dependency graphs and well-established symbol structure (high-quality indexer output).
- Passing CI (implicit quality signal: the fix stuck, tests passed, it got merged).

**Domain distribution**: The training corpus should span **6+ distinct domains** with no more than 20% from any single domain. This prevents the model from conflating domain idioms (parser visitor patterns, attrs-ecosystem conventions) with the target coding style. Include repos from authors across at least 5 different organizations, vary codebase sizes from 1K to 50K LOC, and mix library code with tool code.

**Anti-patterns**: Web frameworks (Django, Flask, FastAPI), HTTP clients (requests, httpx), task queues (Celery), and message brokers look like good candidates — they have clean architecture and sophisticated exception hierarchies — but embed deeply defensive patterns (top-level catch-all handlers, silent fallbacks, `.get()` with defaults everywhere). Any project whose primary job is keeping a long-running process alive will be defensive at its boundaries. Exclude these from the training corpus.

A curated fail-fast repository corpus with specific repo recommendations and an AST-based heuristic scorer for programmatic identification is documented in `research_reviews/fail_fast_research.md`. See also [Section 6.7](#67-fail-fast-training-corpus).

### 6.2 Commit Filtering Criteria

**Extraction tool**: PyDriller for commit traversal and diff extraction. **CommitChronicle** (JetBrains Research) provides a reproducible collection pipeline built on PyDriller with deduplication and outlier filtering. The **D3 paper** is the best architectural reference for LLM-powered instruction labeling of code edit sequences at the 1-3B parameter range. Note: no end-to-end commit→training-pair tool exists — expect 1-2 weeks of custom engineering for the full pipeline.

Not every commit is a useful training example. Filter for:
- **Diff size bounds**: single-file or small-cluster changes (e.g. 1-5 files, < 200 lines changed). Massive refactors are noise. CommitPackFT (NeurIPS 2023 Workshop) found that single-file commits produce the highest-quality instruction-completion pairs.
- **Descriptive messages**: commits with imperative-mood messages starting with action verbs ("Fix," "Add," "Verify") serve as synthetic task descriptions. Apply Verb-Direct-Object pattern filtering via NLTK/spaCy POS tagging to retain only actionable messages. Skip sub-3-word messages and generic tokens ("wip", "misc", "update").
- **Non-merge commits**: merge commits conflate multiple logical changes.
- **Language match**: filter to Python/TS/JS files matching the indexer's current capability.
- **Exclude automated commits**: filter out dependabot updates, version bumps, CI config changes, and documentation-only commits.

**Message regeneration**: For commits with poor messages but good diffs, use **OpenCommit** or a local Ollama model to regenerate descriptions from diffs. The **OMG paper** (ACM 2024) shows that ReAct prompting with broader software context dramatically improves generated descriptions over diff-only approaches. Apply LLM-as-judge scoring to filter generated descriptions, keeping only high-scoring ones.

### 6.3 Synthetic Pipeline Run Generation

From a filtered commit, reverse-engineer what a correct pipeline run *should have* looked like:

| Stage | Synthetic Training Pair | Derivation |
|-------|------------------------|------------|
| Task Analysis | commit message -> structured task description | The commit message is the task; the diff tells you what files/symbols were relevant. |
| Scope | task + repo files -> file relevance classification | Files changed in the diff are "relevant"; their direct dependencies are "supporting"; unrelated files are "irrelevant". |
| Precision | task + file symbols -> symbol tier classification | Symbols modified in the diff are "primary"; symbols referenced by modified code are "supporting"; others are "excluded". |
| Execute (Code) | context + task -> code edits | The diff itself is the target output. Context is the pre-commit state of relevant files. |
| Meta Plan | context + task -> task decomposition | Derive parts from the diff: group file changes into independent logical units, infer dependency edges from change ordering. |
| Step Plan | context + part -> implementation steps | Derive steps from the diff within each part: which symbols to change, in what order, at what granularity. |
| Test Plan | context + diff -> test plan | If the commit includes test changes: separate test file changes as the target, code changes as the input context. |

Note: synthetic planning pairs from diffs are post-hoc reconstructions with weaker signal than the plan validation harness ([Section 6.9](#69-plan-validation-harness)), which generates and validates plans forward. Use synthetic pairs for SFT volume; use harness pairs for high-quality SFT and DPO.

**LintSeq** (ICLR 2025) generates synthetic edit sequences from existing code using only a linter — no LLM needed for data generation. It decomposes complete programs into instruction + file-state + diff-sequence tuples. Models trained on LintSeq edit sequences show 3-6 point pass@1 improvements on HumanEvalFix. Apply LintSeq to transform any existing code dataset (e.g., OpenCodeInstruct subsamples) into edit-sequence format aligned with our search-and-replace output.

**Instruction format**: All synthetic training pairs should use **NL-to-Code format** (natural language task description → code output) rather than Code-to-Code format (code input → code output). OpenCodeInstruct (2025) demonstrated that NL-to-Code significantly outperforms Code-to-Code across all benchmarks. This means commit messages (or regenerated descriptions) should serve as the instruction, not the pre-commit code state.

This is a distinct data curation step from `cra curate-data` (which curates from the agent's own logged runs). It may be implemented as a preprocessing pipeline or as an additional mode.

### 6.4 Quality Signal Mapping

External repo commits provide quality signals analogous to what the self-improvement loop gets from logged runs:

| Self-Improvement Signal | External Repo Equivalent |
|------------------------|--------------------------|
| `task_runs.success` | commit was merged / tests passed |
| retrieval decision correctness | did the diff touch the files scope would have found? |
| symbol selection correctness | did the diff modify the symbols precision would have selected? |
| code generation correctness | does the generated diff match the actual commit diff? |

### 6.5 Storage

Synthetic training pairs from external repos are stored in the same raw DB tables as self-improvement training data (`training_datasets`, `training_plans`). A `source` field distinguishes them:

- `source = "synthetic"`: generated from external repo commit history.
- `source = "logged"`: curated from the agent's own logged runs.

This allows training to mix both data sources as the agent matures.

### 6.6 Cold-Start Datasets

Open datasets that provide immediate training data without commit-history mining:

| Dataset | Size | License | Use |
|---------|------|---------|-----|
| **OpenCodeInstruct** | 5M examples | CC BY 4.0 | Subsample 100-500K for code generation SFT. NL-to-Code format outperforms Code-to-Code. Safest license. |
| **CommitPackFT** | 702K pairs | Apache 2.0 | Real-world code change pairs (commit message + diff). Directly usable for execute stage training. |
| **CodeFeedback-Filtered** | 156K examples | — | Pre-filtered by Qwen-72B for complexity ≥4/5. High-complexity, excellent size for LoRA. |
| **Magicoder OSS-Instruct** | 75K examples | — | Generated from real OSS code snippets as seeds. Orthogonal to Evol-Instruct — combine both. |
| **Evol-Instruct-Code** | 80-110K examples | — | Progressively evolved from Code Alpaca. Proven results on WizardCoder (ICLR 2024). |
| **OpenCoder SFT Stage 2** | 375K examples | — | Quality-filtered with test cases for RL. Well-suited for LoRA. |
| **D3** | 3.6M examples | — | LintSeq + LLM instruction labeling from The Stack. Tested on Llama 3.2 1B and 3B. |
| **SWE-bench** | 2,294 instances | MIT | Validated patches with task descriptions. High-quality but small — use for evaluation and DPO preference pairs. |
| **SRI Tuning** | 20K examples | Apache 2.0 | Search-and-replace instruction examples. Directly aligned with our code edit format. |

**LintSeq** (ICLR 2025) is not a dataset but an algorithm that transforms any existing instruction+code dataset into edit-sequence format. Apply to OpenCodeInstruct or OSS-Instruct subsamples to generate edit-format training data without an LLM.

Both open datasets and commit-history mining complement each other and both depend only on Phase 1. Open datasets provide immediate volume; commit-history mining provides project-specific examples with richer dependency context.

### 6.7 Fail-Fast Training Corpus

Style-targeted training data is a distinct concern from functional-capability training. The code generation LoRA should be biased toward fail-fast error handling (custom exception hierarchies, rich contextual error messages, narrow `except` clauses, no silent failure) to align with the project's coding style principles.

A curated 25-repo corpus spanning 6 domains (parsers/compilers, validation/type checking, structured data, developer tooling, small focused libraries, frameworks with good error design) is documented in `research_reviews/fail_fast_research.md`. Top-tier repos include strictyaml, attrs, cattrs, LibCST, structlog, Black, typeguard, and Zod (TS). The corpus targets 5,000-10,000 high-quality commit pairs filtered for error-handling improvements.

**Programmatic identification**: An AST-based heuristic scorer using 8 weighted metrics (bare-except ratio, blind-except ratio, try-except-pass count, specific exception ratio, dict["key"] vs .get() ratio, assert density, raise density, exception chaining ratio) can rank candidate repos before manual review. Repos scoring 75-100 are strong candidates. This scorer can be built as part of the Phase 4 bootstrapping tooling to expand the corpus beyond the manually curated list.

**DPO preference pairs**: The CodeFavor approach (pre-commit code as rejected, post-commit code as accepted) is directly applicable. Repos like cattrs have commits showing progression from generic `raise Exception(...)` to custom exception classes — ideal before/after training pairs for preference optimization.

### 6.8 Existing Plan Artifacts from Collaborative Repos

A distinct and immediately available data source: repositories where an AI assistant (Claude, Codex, etc.) generated plans that were committed to the repo as files, and a human then followed those plans to produce implementations. This is common in AI-assisted development workflows where the developer saves plans alongside code.

**Why this is uniquely valuable for planning training:**

Unlike synthetic plan generation from commit diffs ([Section 6.3](#63-synthetic-pipeline-run-generation)), these are *real plans that were actually used*. They capture:
- The plan structure and reasoning as it was actually produced, not reverse-engineered.
- The task description (or a close approximation) that prompted the plan, typically present in the plan file or commit message.
- Step ordering, file targets, and rationale — information that is lost when working backwards from a diff.
- Implicit human approval signal: the plan was good enough that the developer chose to follow it rather than discard it.

**Natural quality labeling from git history:**

The commits following a plan file provide outcome signals without manual labeling:

| Git history pattern | Quality label | Training use |
|--------------------|--------------| -------------|
| Plan → clean implementation, no rework | Good | SFT positive example. DPO accept. |
| Plan → implementation → small bug-fix commits | Fine | SFT positive example (plan was structurally sound). Weak DPO accept — or exclude from DPO to avoid noise. |
| Plan → implementation → significant rewrite or new replacement plan | Bad | DPO reject. Pair with the replacement plan (if one exists) as accept. |
| Plan → partial implementation → abandoned | Bad | DPO reject, but only if an alternative exists. Otherwise discard (no accept to pair with). |

**The fine-but-buggy boundary:** Plans that led to bug-fix commits require judgment: was the bug a foreseeable consequence of a plan error, or an edge case the plan couldn't reasonably anticipate? Default to treating these as SFT-positive (the plan structure was sound) and excluding them from DPO (the preference signal is ambiguous). An LLM-as-judge pass can refine this classification if the volume warrants it.

**Context reconstruction:**

The plan file provides the output side of the training pair. The input side (task description + curated context) requires reconstruction:

1. **Task description**: extract from the plan file header, the commit message that introduced the plan, or the conversation context if available.
2. **Repository state**: check out the commit immediately before the plan was added. This is the codebase the planner saw.
3. **Curated context**: run the indexer and retrieval pipeline against that pre-plan repo state with the extracted task description. This produces an approximation of the context window the planner had — not exact (the AI assistant saw different/more context), but structurally equivalent for training purposes.

**Availability:** This data source requires no bootstrapping pipeline, no commit filtering, and no synthetic generation. It requires only: (a) identifying repos that contain plan files, (b) writing an extraction script that walks git history to find plan-commit → implementation-commit sequences, (c) running the context reconstruction pipeline from step 3 above.

**Scale expectation:** Tens to low hundreds of plan→outcome trajectories per repo, depending on development history. Small volume compared to commit-history mining, but high quality per example — each one is a real planning decision with a real outcome. 50-200 high-quality plan trajectories with natural outcome labels may be more valuable than 5K synthetic pairs derived from diffs, because the plans reflect actual planning decisions rather than post-hoc reconstruction.

**Relationship to other data sources:** This complements teacher distillation ([Section 7](#7-distillation-strategy)), synthetic plan generation ([Section 6.3](#63-synthetic-pipeline-run-generation)), and the plan validation harness ([Section 6.9](#69-plan-validation-harness)). Use plan artifacts for SFT examples with natural human-validated quality labels. Use teacher distillation for volume and coverage. Use synthetic plans for breadth. Use the harness for execution-validated pairs with exact distribution match. Plan artifacts are the only source that captures *how humans actually plan* (as opposed to how a teacher model plans) — this diversity of planning styles is valuable for the base fine-tune.

### 6.9 Plan Validation Harness

The highest-quality planning data source: forward-generate plans through our actual pipeline, execute them against real repos, and validate with tests. Unlike Section 6.3 (post-hoc reconstruction from diffs) and Section 6.8 (extraction from existing artifacts), this approach produces execution-validated training pairs with zero distribution mismatch between training and inference — the model sees exactly the context our pipeline would produce.

**Core loop:**

1. **Select commit.** From a filtered external repo ([Section 6.2](#62-commit-filtering-criteria)), choose a feature/enhancement commit with passing tests.
2. **Snapshot pre-commit state.** Check out the parent commit. This is the repo state the planner will see.
3. **Index and retrieve.** Run our Phase 1 indexer and Phase 2 retrieval pipeline against the pre-commit repo state, using the commit message (or regenerated description) as the task. This produces a `ContextPackage` — the exact curated context our pipeline would generate.
4. **Generate plan.** Feed the `ContextPackage` + task description to the planner (teacher model during bootstrapping, student model during self-improvement). The planner produces a plan in our format (MetaPlan → PartPlans → steps).
5. **Execute plan.** Run the plan through our Phase 3 implementation pipeline — code generation for each step, patch application, the full orchestrator loop.
6. **Validate.** Run the repo's existing test suite against the modified code. Supplement with our own generated tests if coverage is insufficient ([see below](#test-supplementation)).
7. **Record outcome.** Store the `(context, plan, outcome)` triple. Success → SFT positive example, DPO accept. Failure → DPO reject.

**Why this works:**

- **Distribution match.** The training pair input is our curated context, not raw files or a different retrieval system's output. The student model at inference time sees exactly this kind of context. Zero distribution mismatch.
- **Ground truth exists.** The commit was merged and tests passed — the task is known to be solvable. If the planner fails, the failure is real, not an impossible task.
- **Binary quality signal.** Tests pass or they don't. No proxy metrics, no LLM-as-judge ambiguity.
- **Automated DPO at scale.** Run the planner N times per commit with temperature variation. Some succeed, some fail. Same task, different plan quality → natural DPO preference pairs with controlled task difficulty. This is higher quality than pairing successes from one task against failures from a different task.
- **Multiple valid plans.** The planner's approach may differ from the original commit's. If it passes tests, it's a valid positive example regardless of the path taken — generating diverse successful plans, not reconstructions of the original approach.

**Isolating plan quality from execution quality:**

When a plan fails, the failure could be in the plan (wrong decomposition, wrong targets), in code generation (plan was fine, coding botched it), or in retrieval (pipeline missed critical context). Mislabeled DPO pairs poison the dataset.

Two mitigation strategies (use both):

1. **Strong code model for execution.** During bootstrapping, use the teacher model for code generation too. If the teacher can't implement a plan, the plan is more likely flawed. This doesn't eliminate coding failures, but reduces them significantly.
2. **Plan-diff structural alignment.** Compare the generated plan against the actual commit diff without executing: file target overlap (did the plan target the right files?), symbol target overlap (did it target the right functions?), dependency ordering consistency (does the step order match the diff's logical structure?). A structurally sound plan that fails execution is likely a coding failure, not a planning failure. A structurally misaligned plan is a genuine planning failure regardless of execution outcome. Use alignment score as an additional signal when labeling DPO pairs.

**Commit selection for planning types:**

Different commit patterns exercise different planning capabilities:

| Commit Pattern | Planning Type Exercised | Selection Heuristic |
|---------------|------------------------|---------------------|
| Multi-file feature addition | Meta Plan (decomposition) | 4+ files changed, new symbols added |
| Single-module enhancement | Step Plan (sequencing) | 1-3 files, modifying existing symbols |
| Test suite addition or expansion | Test Plan (coverage) | Test files changed, production code unchanged |
| Mixed feature + tests | All three | Production files + test files changed |

Prioritize mixed feature+tests commits — they exercise the full planning pipeline (meta-plan → step-plan → test-plan) in a single training example.

**Test supplementation:**

External repos vary in test coverage. When the repo's existing tests are insufficient to validate a plan's implementation:

1. **Generate tests from the ground-truth diff.** The actual commit shows what changed — generate tests targeting those changes using the teacher model. These tests are validated against the *original* post-commit state (they must pass there) before being used to validate the planner's implementation.
2. **Import our test generation pipeline.** If the orchestrator's test-plan phase is operational, use it to generate tests for the planner's implementation. This exercises the test-planning capability and produces additional training data for it.
3. **Fallback: structural validation.** If test generation is unavailable, compare the planner's output structurally against the ground-truth diff (file targets, symbol modifications, import changes). This produces weaker signal than test execution but is available from day one.

**Infrastructure requirements:**

- **Auto-created isolated environments.** The harness reads the repo's dependency spec (`pyproject.toml`, `requirements.txt`, `setup.py`, `setup.cfg`) and auto-creates a Python venv per repo. On the connected machine during development, dependencies install from PyPI normally. Post-migration on the air-gapped machine ([Section 12](#12-deployment-lifecycle)), dependencies install from a **pre-cached package directory** shipped alongside the repo — a directory of `.whl` files collected via `pip download` during repo preparation. No Docker daemon, no container orchestration, no image management. The harness creates the venv, installs from the local cache (`pip install --no-index --find-links ./packages/`), runs the tests, and destroys the venv when done.
- **Repo package format.** Each curated repo is packaged as a self-contained directory:
  ```
  repo_package/
    repo/                    # full git clone
    packages/                # pre-cached .whl files for all dependencies
    manifest.json            # filtered commit list, test command, Python version, metadata
  ```
  The manifest is produced by the commit filtering pipeline ([Section 6.2](#62-commit-filtering-criteria)) during repo preparation on the connected machine. The harness reads the manifest and operates on the listed commits — no discovery at runtime.
- **Parallelizable.** Each commit is independent. Run N commits across M repos concurrently. The pipeline (index → retrieve → plan → execute → validate) is CPU/GPU bound, not I/O bound — parallelism is limited by compute, not by external dependencies.
- **Storage.** Each `(context, plan, outcome)` triple includes: curated context (~8-32K tokens), full plan JSON, execution logs, test output, structural alignment scores. Store in raw DB with `source = "harness"` to distinguish from other data sources.

**Scale expectation:**

From the existing fail-fast repo corpus ([see below](#harness-repo-corpus)), conservatively 100-300 qualifying commits per repo across 14 Python repos:
- ~1,400-4,200 total qualifying commits from the existing corpus
- ~30-50% success rate with teacher → ~500-2,100 successful plans (SFT positives)
- ~50-70% failure rate → DPO reject candidates (after filtering for plan-quality failures)
- Multiple runs per commit (3 temperature variations) → 4,200-12,600 `(context, plan, outcome)` triples
- Adding new repos post-migration expands this further

This produces enough data for both the base fine-tune's planning objective and the optional DPO refinement LoRA, from the existing curated corpus alone — no additional repo sourcing needed for the initial campaign.

**Relationship to other data sources:**

| Data Source | Volume | Quality | Cost | Planning Signal | Distribution Match |
|---|---|---|---|---|---|
| Section 6.3 — reconstructed from diffs | High (thousands) | Medium (post-hoc) | Low (no execution) | Structural only | Low (no pipeline context) |
| Section 6.8 — real plan artifacts | Low (tens to hundreds) | High (human-validated) | Low (extraction only) | Real but sparse | Medium (approximate context) |
| **Section 6.9 — plan validation harness** | **Medium-high (thousands)** | **High (execution-validated)** | **Medium (pipeline + execution)** | **End-to-end, automated** | **Exact (our pipeline context)** |

Use all three: Section 6.3 for SFT volume (most plans, weakest signal), Section 6.8 for real-world plan structure (fewest but most natural), Section 6.9 for execution-validated SFT and DPO pairs (highest quality per example, exact distribution match).

**Self-improvement loop integration:**

The harness architecture is deliberately model-agnostic in the planner slot. During bootstrapping ([Section 7](#7-distillation-strategy)), the teacher model generates plans. Once the student model's base fine-tune includes planning capability, the same harness runs with the student model — producing on-policy training data where the student generates plans, executes them, and learns from the outcomes. The infrastructure is identical; only the planner model changes. This makes the harness the primary vehicle for the planning self-improvement loop ([Section 10](#10-self-improvement-guardrails)):

1. **Teacher bootstrapping** — teacher generates plans, builds the initial training corpus.
2. **Student evaluation** — student generates plans on held-out commits, measures baseline.
3. **Student self-improvement** — student generates plans on training commits, successes/failures become new training data, mixed with teacher data (accumulate, never replace).
4. **Iteration** — repeat step 3 with fresh commits from new repos to maintain diversity.

The ~500 quality trajectory threshold for measurable gains (SWE-Gym finding) is reachable from a single automated harness campaign. Repo diversity injection (Hybrid-Gym finding) is built into the harness design — adding a new repo to the campaign is adding a repo package and a manifest, not redesigning the pipeline.

**Harness repo corpus:**

The harness does not require a separate repo curation effort. The repos already curated for fail-fast style training ([Section 6.7](#67-fail-fast-training-corpus), detailed in `research_reviews/fail_fast_research.md`) and LoRA training data (`research_reviews/lora_training_git_data.md`) serve double duty: their *code* trains the fail-fast coding style, their *commit histories* feed the plan validation harness. These repos were selected for maturity, clean commits, and strong test suites — exactly the harness criteria. A documentation quality assessment across all 48 repos (`research_reviews/repo_documentation_quality.md`) provides additional signal on commit hygiene, test coverage, and maintainer culture.

**Primary harness candidates** (Python, MIT/Apache/BSD, pytest, mature commit history, right size):

| Repo | Files | Why it's good for the harness | Planning types exercised |
|------|-------|-------------------------------|-------------------------|
| **attrs** | ~15 | Excellent commit hygiene, 100% coverage requirement, focused feature commits | Step Plan (single-module enhancements) |
| **cattrs** | ~20 | Commits showing error handling progression (generic → custom exceptions), ideal before/after | Step Plan, DPO pairs |
| **structlog** | ~50 | 100% test coverage, signed releases, async-safe features added incrementally | Step Plan, Test Plan |
| **Black** | ~50+ | Large feature set, AST safety verification, comprehensive fuzz testing | Meta Plan (multi-file features), Step Plan |
| **LibCST** | ~150 | 1,788+ tests (Hypothesis fuzz), rich parser features, well-scoped commits | Meta Plan, Step Plan, Test Plan |
| **typeguard** | ~30-50 | Three instrumentation modes, `xfail_strict=true` in tests, focused scope | Step Plan, Test Plan |
| **svcs** | small | 100% coverage, ResourceWarning safety, clean API evolution | Step Plan |
| **stamina** | small | Well-documented commits, backoff algorithm improvements | Step Plan |
| **Click** | medium | Exemplary exception hierarchy, feature additions with tests | Step Plan, Test Plan |
| **strictyaml** | ~15 | Story-based tests, 18+ exception types, philosophy-driven commits | Step Plan |
| **parso** | ~30-40 | Dual-mode parser, fail-fast by default with optional recovery | Step Plan |
| **msgspec** | medium | Strict mode, structured serialization features added incrementally | Step Plan, Test Plan |
| **Hypothesis** | large | Extensive fuzz testing, complex feature additions | Meta Plan, Step Plan (MPL-2.0 — check license) |
| **Werkzeug** | medium | Gold-standard exception hierarchies, feature-rich history | Meta Plan, Step Plan (defensive boundary caveat) |

**Dual purpose, one corpus:** these repos are packaged once in the repo package format (repo + wheels + manifest). The same package serves both the harness (commit-by-commit plan generation and validation) and the fail-fast style training (code extraction for the coding style objective in the base fine-tune). No additional curation needed — the fail-fast research already filtered for exactly the repos the harness needs.

**Commit density estimate:** the Tier 1 fail-fast repos (11 Python + 4 TypeScript) collectively have thousands of qualifying feature/enhancement commits. Even conservatively — 100-300 harness-qualifying commits per repo, 14 repos — that's 1,400-4,200 commits available for the harness before adding any new repos. With 3 temperature runs per commit, that's 4,200-12,600 `(context, plan, outcome)` triples from the existing corpus alone.

**Expansion path:** post-migration, new repos enter via the data ingestion protocol ([Section 12.6](#126-new-data-ingestion-protocol)). Each new repo simultaneously expands both the harness corpus (more commits for planning training) and the style corpus (more code for fail-fast training). The AST-based heuristic scorer from `research_reviews/fail_fast_research.md` can be used to pre-screen candidate repos on the connected side before packaging for USB transfer.

---

## 7. Distillation Strategy

LoRA adapters and the base fine-tune's planning component are trained via teacher-student distillation: large teacher models generate high-quality training examples, which are used to fine-tune the small local models. For planning, the teacher generates plans through the plan validation harness ([Section 6.9](#69-plan-validation-harness)), producing execution-validated training pairs.

### 7.1 Teacher Models

| Stage | Target Model | Primary Teacher | Secondary |
|-------|-------------|----------------|-----------|
| 1-3 (Task Analysis, Scope, Precision) | Qwen3-4B | Qwen3.5-397B-A17B (Apache 2.0, $0.11/M input) | — |
| Base fine-tune — Planning | Qwen3-4B | Qwen3.5-397B thinking mode | DeepSeek-V3.2 (cross-validation) |
| 4 (Execute — Code) | Qwen2.5-Coder-3B | Qwen3-Coder-Next-80B-A3B | OpenCodeInstruct dataset |

**Planning teacher role**: The teacher generates plans through the plan validation harness ([Section 6.9](#69-plan-validation-harness)) — our pipeline provides the context, the teacher produces the plan, execution validates it. The teacher also generates CoT reasoning traces for all three planning types (meta-plan, step-plan, test-plan). These feed into the base fine-tune's planning objective, not a separate LoRA.

**Tokenizer incompatibility**: Qwen3.5 uses a 250K vocabulary vs Qwen3-4B's 150K. This means only response-level SFT (train on teacher's text output), not logit-level distillation (KL divergence on token probabilities). Fallback for logit distillation if needed: Qwen3-235B (same tokenizer family as Qwen3-4B).

### 7.2 Per-Stage Training Configurations

**LoRA adapters (retrieval stages + code execution):**

| Stage | Technique | Rank | Examples | Time (RTX 4090) |
|-------|-----------|------|----------|-----------------|
| 1 Task Analysis | SFT | 16-32 | 3-5K | ~15-30 min |
| 2 Scope (relevance) | SFT | 16-32 | 2-4K | ~10-25 min |
| 3 Precision (symbol selection) | SFT + curriculum | 16-32 | 3-5K | ~15-30 min |
| 4 Execute — Code | SFT + DPO | 32-64 | 10-50K | ~1.5-3.5 hrs |

**Base fine-tune planning component** (part of the full fine-tune run, [Section 2.6](#26-planning-capability-training-details)):

| Planning Type | Technique | Examples | Source |
|---------------|-----------|----------|--------|
| Meta Plan (decomposition) | CoT-SFT | 3-5K | Harness ([6.9](#69-plan-validation-harness)) + teacher distillation + synthetic ([6.3](#63-synthetic-pipeline-run-generation)) |
| Step Plan (sequencing) | CoT-SFT | 3-8K | Harness + teacher distillation + synthetic |
| Test Plan (coverage) | CoT-SFT | 2-5K | Harness + teacher distillation |

**Optional planning DPO LoRA** (trained only if base evaluation warrants it):

| Stage | Technique | Rank | Examples | Time (RTX 4090) |
|-------|-----------|------|----------|-----------------|
| Planning DPO | DPO | 16-32 | 1-5K preference pairs | ~15-45 min |

**Total estimated training time**: ~2-3.5 hours for all 4 LoRA stages on an RTX 4090 with QLoRA rank-32 and sequence length 2048. The base fine-tune (including planning) is a separate run estimated at 4-8 hours. RTX 3090 adds ~60-70% to these times. RTX 4080 (16GB) adds ~100%.

**Planning iteration risk:** Planning has a high-dimensional output space with no ground truth. The plan validation harness ([Section 6.9](#69-plan-validation-harness)) substantially de-risks this by automating preference pair generation — but the fundamental challenges remain:

- **Subtle planning degradation is the primary risk.** A bad planning base produces plans that look reasonable but miss edge cases, order steps wrong, or scope incorrectly. Detection requires end-to-end evaluation (plan → implement → validate), not just plan-level metrics. The harness provides this evaluation automatically.
- **CoT trace quality is hard to evaluate automatically.** A reasoning trace can look coherent but lead to a subtly wrong plan. Manual inspection of a sample (10-20 examples per planning type per training round) is unavoidable.
- **Isolating plan quality from coding quality** is essential for DPO pairs. The harness addresses this via strong-model execution and plan-diff structural alignment ([Section 6.9](#69-plan-validation-harness)).
- **Realistic wall-clock estimate**: 1-2 weeks for the initial base fine-tune planning component (data generation via harness is automated, but evaluation and hyperparameter tuning require iteration). The optional DPO LoRA adds 1-2 weeks if needed. This is faster than the previous estimate (2-4 weeks) because the harness automates preference pair generation — the previous bottleneck.

### 7.3 Training Infrastructure

**Deployment context**: the full fine-tune is outsourced to external GPU hardware; LoRA training runs locally. Post-migration to the air-gapped machine, all training (LoRA only) runs locally with no internet access. See [Section 12](#12-deployment-lifecycle) for the full lifecycle. All tooling below must be installed and frozen before migration.

**Framework**: **Unsloth** is the primary training framework — 2-5× faster training with 70-80% less VRAM via custom Triton kernels, explicit Qwen2.5-Coder and Qwen3 support with pre-quantized models on HuggingFace, and one-line Ollama/GGUF export. **LLaMA-Factory** is the alternative (Web UI, Qwen3 templates, direct Ollama Modelfile export). Axolotl and raw PEFT/TRL are fallbacks for advanced use cases only.

**LoRA hyperparameters** (synthesized from PLoRA, QLoRA, and Unsloth documentation):
- **Quantization**: QLoRA 4-bit NF4 + double quantization.
- **Target modules**: All 7 linear layers (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`). QLoRA paper confirms targeting all layers is critical for matching full fine-tuning performance.
- **Alpha**: 2× rank (e.g., rank 32 → alpha 64). Unsloth recommends alpha ≥ rank.
- **Learning rate**: 2e-4 with cosine scheduler (10× higher than full fine-tuning).
- **Optimizer**: `adamw_8bit` for memory efficiency.
- **Dropout**: 0 (enables Unsloth kernel optimizations; add 0.05 only if overfitting observed).
- **Epochs**: 1-3 with early stopping. Multi-epoch training on instruction datasets can decrease performance (Raschka).
- **Precision**: bf16 throughout.

**Qwen-specific gotchas**:
- EOS token `<|im_end|>` must be set explicitly (common source of infinite generation bugs).
- Qwen2.5-Coder `pad_token` was incorrectly set to `<|endoftext|>` in the original release, causing infinite generations — use Unsloth's corrected HuggingFace versions.

**VRAM requirements** (QLoRA 4-bit, gradient checkpointing, sequence length 2048):

| Model | Batch Size 1 | Batch Size 2 | Batch Size 4 |
|-------|:-----------:|:-----------:|:-----------:|
| Qwen2.5-Coder-3B | 4-6 GB | 5-7 GB | 7-10 GB |
| Qwen3-4B | 5-7 GB | 6-8 GB | 8-12 GB |

An 8 GB GPU handles QLoRA on the 3B model at batch size 1. A 24 GB RTX 4090 runs LoRA 16-bit with batch size 4+ on both models.

**LoRA-to-deployment conversion** (three paths):
1. **Unsloth auto-export**: `save_pretrained_gguf()` + `save_pretrained_ollama()` — auto-generates Modelfile with correct Qwen ChatML template, calls `ollama create` internally. Easiest path.
2. **Manual merge**: PEFT `merge_and_unload()` → `llama.cpp/convert_hf_to_gguf.py` → Modelfile with `FROM ./model.gguf` → `ollama create`. For Ollama merged-model deployment.
3. **Adapter-only GGUF**: `llama.cpp/scripts/convert_lora_to_gguf.py` → reference via Ollama `ADAPTER` directive or vLLM `--lora-modules`. For per-request adapter selection.

**Estimated cost**: ~$80-200 total teacher inference API calls (increased from previous estimate due to harness-driven plan generation campaigns). ~2-3.5 hrs total local GPU LoRA training + ~4-8 hrs base fine-tune on RTX 4090 (see Section 7.2 for per-stage breakdown).

---

## 8. Execution Priority

The full training strategy involves multiple base fine-tunes, 4 per-stage LoRA adapters, an optional planning DPO LoRA, multiple data sources, and multiple training techniques. This section defines the order in which to execute them. The ordering principle: **train what you can measure first, train what you can't measure last.**

### 8.1 Priority Tiers

**Tier 1 — Train first (highest data quality, clearest signal):**

| Priority | What | Why first |
|----------|------|-----------|
| 1a | Plan validation harness infrastructure | Everything downstream depends on this: auto-venv repo environments, commit filtering, pipeline integration. Engineering time, no GPU time. |
| 1b | Base model full fine-tune (32K), **including planning capability** | All LoRAs stack on top of this. Planning data from harness ([6.9](#69-plan-validation-harness)) + teacher distillation + synthetic ([6.3](#63-synthetic-pipeline-run-generation)) + plan artifacts ([6.8](#68-existing-plan-artifacts-from-collaborative-repos)). The model can plan out of the box after this step. Outsourced to external GPU ([Section 12.3](#123-outsourced-training)). |
| 2 | Execute — Code LoRA | Most training data available (CommitPackFT, OpenCodeInstruct, SRI Tuning). Binary quality signal (tests pass or they don't). Highest immediate impact on pipeline output. |
| 3 | Scope LoRA | File relevance classification has ground truth from commit diffs. Small dataset (2-4K). Fast to train and validate. Directly improves Execute context quality. |

**Tier 2 — Train second (good signal, moderate iteration):**

| Priority | What | Why second |
|----------|------|------------|
| 4 | Precision LoRA | Symbol-level classification. Ground truth derivable from diffs but noisier than file-level scope. Curriculum training adds complexity. |
| 5 | Task Analysis LoRA | Smallest stage, lowest risk. Quality signal is indirect (did downstream retrieval find the right files?). Worth deferring until Scope and Precision are stable so the signal is cleaner. |
| 6 | Base model full fine-tune (8K) | Only needed for Task Analysis. Defer until Task Analysis LoRA training is imminent. |

**Tier 3 — Train if needed (refinement, optional):**

| Priority | What | Why last |
|----------|------|----------|
| 7 | Planning DPO LoRA (optional) | Preference alignment on the planning-aware base. Only trained if Tier 1 evaluation shows the base fine-tune's planning underperforms on DPO-style preference criteria. DPO pairs generated automatically by the plan validation harness ([Section 6.9](#69-plan-validation-harness)). Faster iteration than the previous strategy (harness automates what was previously the bottleneck). |

Note: planning moved from Tier 3 (weakest signal, most iteration) to **Tier 1** (base fine-tune). The plan validation harness automates preference pair generation, and planning capability in the base means the model can plan from day one. The optional DPO LoRA in Tier 3 is refinement, not capability — the system functions without it.

### 8.2 Decision Gates

Do not advance to the next tier until the current tier's adapters pass evaluation ([Section 9](#9-evaluation-methodology)). Specifically:

- **Gate 1→2**: Base fine-tune must show planning capability across all three types (meta-plan, step-plan, test-plan) on harness-derived evaluation set. Execute-Code LoRA must show measurable improvement over base model on held-out tasks (pass@1 or test pass rate). Scope LoRA must improve file retrieval precision/recall vs. base model on synthetic commit tasks.
- **Gate 2→3**: Pipeline end-to-end success rate (plan → implement → validate) must be stable. Planning DPO pairs are generated by the harness from Tier 1-2 pipeline runs — if those runs are unreliable, the preference data is garbage.

### 8.3 Estimated Timeline

| Tier | Engineering time | GPU time | Bottleneck |
|------|-----------------|----------|------------|
| 1 | 2-3 weeks | ~6-10 hrs | Harness infrastructure (1a) + base fine-tune with planning (1b) + data curation. Harness engineering and teacher data generation run in parallel with commit filtering for code/retrieval datasets. |
| 2 | 1-2 weeks | ~1-2 hrs | Curriculum design for Precision, indirect signal for Task Analysis |
| 3 | 0-2 weeks | ~0-1 hrs | Optional. Only if base planning evaluation warrants DPO refinement. Harness generates preference pairs automatically — previous bottleneck is eliminated. |

**Total**: 3-7 weeks of engineering time. Tier 1 is longer than before (harness infrastructure + base fine-tune planning component), but Tier 3 shrinks from 2-4 weeks to 0-2 weeks because the harness automates preference pair generation. Net effect: similar total time, but the model has planning capability from Tier 1 instead of waiting until Tier 3.

---

## 9. Evaluation Methodology

Training without evaluation is flying blind. Each stage needs a held-out evaluation set with clear pass/fail criteria, built *before* the first training run. Evaluation is not optional and not deferred — it is a prerequisite for Tier 1 training.

### 9.1 Per-Stage Evaluation Sets

| Stage | Eval set source | Size | Primary metric | Pass threshold |
|-------|----------------|------|----------------|----------------|
| Task Analysis | Synthetic from commit history: commit message → expected file targets | 200-500 | File target recall (did the analysis identify the files the commit actually touched?) | ≥ 80% recall, ≥ 60% precision |
| Scope | Synthetic from commit history: task + repo → file relevance ground truth | 200-500 | Classification accuracy (relevant/supporting/irrelevant vs. ground truth from diff) | ≥ 75% accuracy, ≥ 85% relevant-recall |
| Precision | Synthetic from commit history: task + files → symbol tier ground truth | 200-500 | Tier classification accuracy (primary/supporting/excluded vs. diff-derived ground truth) | ≥ 70% accuracy, ≥ 80% primary-recall |
| Execute — Code | SWE-bench Verified subset + held-out commit diffs | 100-300 | pass@1 (generated code passes validation) | Measurable improvement over base model |
| Planning — Meta Plan | Plan validation harness on held-out multi-file feature commits | 50-100 | Part decomposition quality: file target coverage, dependency correctness, downstream step success rate | ≥ base model (pre-fine-tune) success rate |
| Planning — Step Plan | Plan validation harness on held-out single-module commits | 100-200 | Step execution success rate (per-step code generation passes) | ≥ base model success rate |
| Planning — Test Plan | Plan validation harness on held-out commits with test changes | 50-100 | Test generation coverage: does the generated test plan exercise the changed code? | ≥ base model success rate |

**Eval set construction** depends only on Phase 1 (indexer), the bootstrapping pipeline ([Section 6](#6-bootstrapping-from-external-repo-history)), and the plan validation harness ([Section 6.9](#69-plan-validation-harness)). The harness produces both training data and evaluation data from the same infrastructure — but the eval set must be strictly held out (no overlap with training data, no overlap with repos used for training). Reserve 10-20% of harness-qualifying repos exclusively for evaluation.

**Planning evaluation uses the harness directly.** Unlike retrieval stages (where evaluation is a classification comparison), planning evaluation requires end-to-end execution: plan → implement → validate. The harness already does this. The eval set is a set of held-out commits run through the harness with the model under evaluation as the planner. This means planning evaluation is automated and repeatable — no manual judgment of plan quality needed for the primary metric.

### 9.2 Evaluation Protocol

**Before every training run:**
1. Run the eval set against the current best adapter (or base model for the first run). Record baseline metrics.

**After every training run:**
2. Run the same eval set against the new adapter. Record metrics.
3. Compare against baseline. If any primary metric regresses, the new adapter is rejected (see [Section 10.5](#105-adapter-versioning-and-rollback)).
4. If metrics improve, promote the new adapter to active. Archive the old adapter (do not delete).

**Per-stage regression detection:**
- Each stage's eval is independent. An improvement in Execute-Code does not excuse a regression in Scope.
- End-to-end pipeline eval (plan → implement → validate on held-out tasks) runs after any individual adapter change, to catch cross-stage interaction effects.

### 9.3 What the Eval Set Cannot Catch

Some failure modes are invisible to automated evaluation:

- **Planning subtlety**: a plan that scores well on structure metrics but leads to brittle implementations. Mitigated by end-to-end eval, but not eliminated.
- **Style drift**: the model produces correct code that violates coding style principles (defensive patterns creeping in, silent error swallowing). Mitigated by the fail-fast corpus training, but should be spot-checked manually.
- **Distribution shift**: the eval set may not cover the task distribution the agent encounters in real use. Mitigated by repo diversity in eval set construction, but periodic eval set expansion is necessary.

Manual review of a random sample (10-20 examples per stage per training round) is not optional. Automate what you can, inspect what you can't.

---

## 10. Self-Improvement Guardrails

Empirical limits on self-improvement at sub-7B scale, informed by recent research:

**Hard limits**:
- Self-improvement plateaus after ~2 iterations at sub-7B scale (SWE-Gym, ICML 2025).
- Sub-6B models fail to bootstrap via self-generated data (STaR finding) — external teacher signal is mandatory.
- Recursive training on synthetic data leads to model collapse (Shumailov et al., Nature 2024) without careful data management.
- Full GRPO/PPO training is NOT feasible on consumer hardware (DeepSWE required 64 H100s for 6 days). DPO from self-generated trajectories is practical — single-round DPO with coarse filtering achieves RL-level results with lower compute.

**Mandatory practices**:
- **Accumulate data, never replace**: each training round adds new examples to the pool; never discard previous rounds.
- **Always train from frozen base, never merge-then-retrain**: LoRA adapters are trained independently from the base model checkpoint. Never merge an adapter into base weights and then train a new adapter on the merged model.
- **10-20% replay data**: every training batch includes examples from earlier rounds to prevent catastrophic forgetting.
- **Planning self-improvement uses the harness**: the plan validation harness ([Section 6.9](#69-plan-validation-harness)) is the primary vehicle for planning self-improvement. Swap the teacher for the student model in the harness planner slot — the infrastructure is identical, only the model changes. This provides execution-validated on-policy data without any new engineering.

**Timeline** (ordered by increasing risk):
1. Pure teacher distillation (retrieval stages 1-3 + code execution) and teacher-driven harness (planning in base fine-tune)
2. Mixed: on-policy data with teacher scoring (retrieval stages 1-3 first)
3. Execution feedback loop (code validation as signal for code LoRA)
4. Student-driven harness (planning self-improvement): swap student model into harness planner slot, generate on-policy plan data, mix with teacher data. Same harness infrastructure, same validation, same quality signals. Planning self-improvement still transitions last — planning degradation is the hardest to detect — but the harness automates what was previously manual iteration.

**Practical thresholds**:
- ~500 quality trajectories can produce measurable gains (SWE-Gym finding) — sets a lower bar for initial self-improvement rounds than expected. The harness can generate 500 trajectories from a single automated campaign across 2-3 repos.
- Repo diversity in the training set matters more than volume for transfer to new tasks (Hybrid-Gym, 2026). Inject fresh task diversity (new repos, new problem types) to break through plateaus rather than collecting more data from the same distribution. The harness makes this trivial: adding a new repo is adding a Docker environment and a commit filter.
- SelfCodeAlign (NeurIPS 2024) achieves self-alignment without any teacher model on models from 3B to 33B, with the finding that base models benefit most from alignment with their own data distribution. This suggests a hybrid approach: teacher distillation for initial rounds, then on-policy data once baseline capability is established. The harness implements exactly this transition — same infrastructure, different planner model.

### 10.5 Adapter Versioning and Rollback

A training round that makes things worse must be reversible without retraining. This requires explicit adapter versioning.

**Versioning scheme:** Every trained adapter gets a version tag: `{stage}-v{N}` (e.g., `scope-v1`, `execute-code-v3`). The version number increments monotonically. The curated DB `adapter_metadata` table tracks: stage name, version, training timestamp, eval metrics snapshot, active/inactive status, file path to adapter weights.

**Active adapter selection:** Exactly one adapter version per stage is marked `active` in curated DB. The `ModelRouter` resolves stage → active adapter at runtime. Switching the active adapter is a single DB update — no model reloading needed if using vLLM with multiple adapters loaded.

**Rollback protocol:**
1. After a training run, the evaluation protocol ([Section 9.2](#92-evaluation-protocol)) compares the new adapter against the current active version.
2. If any primary metric regresses, the new adapter is marked `inactive`. The previous version remains `active`. The new adapter is retained on disk (for analysis) but never enters the pipeline.
3. If regression is detected after promotion (e.g., end-to-end eval catches a cross-stage interaction), rollback to the previous version is immediate: set `active=false` on current, `active=true` on previous. No retraining, no data loss.

**Retention policy:** Never delete previous adapter versions. Disk cost for LoRA adapters is trivial (~20-50 MB each). Retraining cost is not. Retain all versions indefinitely for:
- Regression analysis (compare v3 vs v2 vs v1 on specific failure cases).
- Recovery from compounding errors (if v3 was trained on data from a pipeline running v2, and v2 turns out to be subtly bad, rollback to v1 and regenerate training data).
- Audit trail (which adapter was active when a specific task was run).

**Base model checkpoints** follow the same versioning. Context-reduced bases are tagged `{role}-base-v{N}` (e.g., `reasoning-32k-base-v1`). If a base needs to be retrained (e.g., context target changes), all LoRA adapters on the old base are invalidated — they must be retrained on the new base. This is the cost of context reduction being irreversible, and why base model changes should be rare and deliberate.

---

## 11. Evidence Base and Advanced Techniques

**Per-stage LoRA validation**: IBM Granite Intrinsics Library demonstrated 6 LoRA adapters for RAG pipeline stages with no cross-stage degradation, plus **Activated LoRA (aLoRA)** achieving 20-35× speedup per adapter invocation via KV cache reuse. META-LoRA showed benefits more pronounced for small models. **R-LoRA** (EMNLP 2025 Findings) explicitly tested on **Qwen2.5-3B**, showing multi-head LoRA outperforms vanilla LoRA on multi-task benchmarks — direct evidence for our target model. MTL-LoRA found task-specific transformation matrices consistently outperform both single-task and vanilla multi-task LoRA, mitigating the "seesaw effect." Note: no published project has done per-stage LoRA for coding agent pipelines — this is genuinely novel.

**Adapter routing**: LORAUTER (January 2026) routes queries to appropriate adapters via task embeddings, scaling to 1500+ adapters. LoGo (November 2025) performs training-free, instance-specific selection and merging from a LoRA pool. For our pipeline where stage routing is deterministic (the orchestrator knows which stage it's in), simple adapter selection via the vLLM `model` parameter is sufficient — no learned routing needed.

**Relevance judgment training**: **Self-RAG** (ICLR 2024) trains models with special reflection tokens (`[Relevant]`, `[Irrelevant]`, `[Fully supported]`, etc.) using GPT-4-generated critic labels — the model learns to output structured judgment tokens inline. **ARES** (NAACL 2024) fine-tunes LLM judges on (query, passage, answer) triples with only ~150 human-annotated validation datapoints. Both adapt directly to code symbol relevance classification for the scope and precision stages.

**Edit-sequence training**: **LintSeq** (ICLR 2025) decomposes complete programs into synthetic edit sequences using a linter (no LLM needed), producing instruction + file-state + diff-sequence tuples. Models trained on LintSeq show 3-6 point pass@1 improvements on HumanEvalFix. The **D3 dataset** (3.6M examples from The Stack) extends LintSeq with LLM-powered instruction labeling, tested on Llama 3.2 1B and 3B. **SRI Tuning** (January 2026) trains models to generate search-and-replace edit instructions, tested on Qwen2.5-Coder-Base — directly aligned with our output format.

**Self-improvement at scale**: **SWE-Gym** (ICML 2025) used LoRA via Unsloth to train Qwen-2.5-Coder-7B, finding plateau after two iterations. **SWE-smith** (NeurIPS 2025) extended to 50K task instances, training SWE-agent-LM-32B to 40.2% on SWE-bench Verified. **R2E-Gym** (COLM 2025) introduces SWE-GEN, deriving training environments from version-control commits via back-translation — closest system to our commit-history bootstrapping. **Hybrid-Gym** (February 2026) found repo diversity more important than volume for transfer. **Lingma SWE-GPT** (Alibaba) trained 7B/72B Qwen models, with the **7B model resolving 18.20% of SWE-bench Verified** — proof small models can work with proper training.

**Small-model bootstrapping**: **SelfCodeAlign** (NeurIPS 2024) achieves self-alignment without teacher from 3B to 33B. **LADDER** improved Llama 3B from 2% to 82% on integration problems using generate→solve→verify→learn with explicit curriculum. The **Code Graph Model** (May 2025) fine-tunes LLMs with LoRA to understand repository-level structure using graph-to-text conversion.

**SAD (Structured Agent Distillation)**: If CoT-SFT + DPO underperforms for planning, an advanced technique to try: segment plan output into `[REASONING]` and `[PLAN]` spans with different loss weights (higher weight on the plan structure, lower on reasoning traces). Apply curriculum learning (single-file plans → multi-file plans → complex dependency chains).

---

## 12. Deployment Lifecycle

The system goes through four phases before reaching its final operating state. Each phase has distinct infrastructure, network access, and training capabilities.

### 12.1 Phase Overview

```
Development Machine          External Training        Air-Gapped Machine
(connected, current)         (outsourced GPU)         (final deployment)
        │                          │                         │
   Build Phase 4 ──────────► Full fine-tune(s) ─────►       │
   Build harness                   │                         │
   Validate under supervision      │                         │
   Teacher distillation            │                         │
   Harness campaigns               │                         │
        │                          │                         │
        └── Migration (one-time transfer) ──────────► System operational
                                                             │
                                           Self-improvement loop (student-only)
                                           New repos via USB (ongoing)
```

### 12.2 Development Phase (Connected Machine)

**Network access**: full internet. **Purpose**: build, validate, and front-load teacher signal.

1. **Implement Phase 4** — self-training infrastructure: harness pipeline, data curation, training scripts, evaluation methodology. All developed and tested here.
2. **Teacher distillation campaigns** — all teacher API calls (Qwen3.5-397B) happen during this phase. Generate teacher plans through the harness ([Section 6.9](#69-plan-validation-harness)), collect CoT reasoning traces, build the initial training corpus. This is a finite activity — teacher access ends at migration.
3. **Harness validation** — run teacher-driven and student-driven harness campaigns under supervision. Verify: environment auto-creation works reliably, plan quality isolation is accurate, DPO pairs are correctly labeled, evaluation metrics are meaningful.
4. **Curate initial repo corpus** — clone repos, run commit filtering ([Section 6.2](#62-commit-filtering-criteria)), collect dependency wheels (`pip download`), generate manifests, package into the repo package format ([Section 6.9](#69-plan-validation-harness)). This corpus transfers with the system at migration time.

### 12.3 Outsourced Training

**Network access**: full internet (external machine). **Purpose**: full fine-tune iterations that exceed the development machine's GPU capacity.

The full fine-tune ([Section 2](#2-base-model-full-fine-tuning)) — including the planning capability objective — runs on a more capable external machine. This may happen more than once as data mix, hyperparameters, and evaluation results drive iteration.

**Workflow**:
1. Prepare training data on the development machine (datasets, harness output, teacher distillation traces).
2. Transfer training data + scripts to the external machine.
3. Run full fine-tune. Evaluate on the external machine.
4. Export model artifacts in deployment format (GGUF for Ollama, HuggingFace format for vLLM, or both).
5. Transfer artifacts back to the development machine for integration testing.
6. If evaluation fails: adjust and repeat from step 1.

**LoRA training** does not require outsourcing — QLoRA on 3-4B models fits comfortably on consumer GPUs ([Section 7.3](#73-training-infrastructure)). Only the full fine-tune is outsourced.

### 12.4 Migration

**One-time transfer** of the complete system to the air-gapped machine. After this point, the air-gap is established and the connected machine is no longer involved.

**Migration checklist — what goes onto the air-gapped machine:**

| Category | Contents |
|----------|----------|
| **Pipeline code** | Phases 1-4 source, all dependencies vendored or pre-installed |
| **Models** | Fine-tuned base model(s) (GGUF + HF format), all LoRA adapters, base model checkpoints for recovery |
| **Training tooling** | Unsloth, PEFT, PyTorch, CUDA toolkit — all pinned versions, fully installed. No post-migration `pip install` |
| **Inference server** | Ollama and/or vLLM, fully configured |
| **Initial repo corpus** | Packaged repos with pre-cached wheels and manifests ([Section 6.9](#69-plan-validation-harness) format) |
| **Databases** | Raw DB (training corpus from development phase), curated DB (indexed repos + adapter metadata) |
| **Evaluation sets** | Held-out harness commits for ongoing evaluation ([Section 9](#9-evaluation-methodology)) |

**Post-migration verification**: run the full self-improvement loop once end-to-end (harness → LoRA training → evaluation → promote/rollback) on the initial repo corpus to confirm everything works in the air-gapped environment before relying on it.

### 12.5 Post-Migration Operations

**Network access**: none. **Purpose**: autonomous self-improvement from new repo data.

The air-gapped machine runs the complete pipeline and self-improvement loop independently:

- **Inference**: Ollama or vLLM serves the fine-tuned base + active LoRA adapters.
- **Pipeline execution**: Phases 1-3 operate on real tasks.
- **Self-improvement**: student-driven harness on pre-loaded repos. LoRA training locally. Evaluation locally. Adapter versioning and rollback ([Section 10.5](#105-adapter-versioning-and-rollback)) locally.
- **No teacher access**: all teacher distillation happened pre-migration. Post-migration training is student-only, which is consistent with the self-improvement guardrails ([Section 10](#10-self-improvement-guardrails)) — teacher distillation for initial rounds, on-policy data once baseline capability is established.

### 12.6 New Data Ingestion Protocol

New curated repos are the only external input post-migration. They enter through a dedicated air-gapped laptop that serves as a hardware data diode.

**Protocol**:
1. **Preparation (connected side)**: clone repo, run commit filtering, collect dependency wheels via `pip download`, generate manifest, package into the repo package format. Validate the package is complete (all wheels present, manifest correct, test command verified).
2. **Transfer to USB**: write the repo package to a USB drive. The USB contains only the repo package — no executables, no scripts, no other data.
3. **Air-gapped laptop**: reads the USB, writes the repo package to the air-gapped machine's storage. **The laptop never copies or retains data** — it is a transfer medium only. No persistent storage of repo contents on the laptop between transfers.
4. **Air-gapped machine**: the harness detects the new repo package, validates the manifest, creates a venv from the pre-cached wheels, and integrates the repo into the harness corpus. New commits become available for self-improvement campaigns.

**What does NOT go through the USB**: model weights, training scripts, pipeline code, configuration changes, software updates. The system is frozen at migration time. Only training data (curated repos) enters.

**Repo package validation**: the air-gapped machine validates each incoming repo package before integration:
- Manifest schema check (required fields present, commit hashes valid)
- Wheel completeness (all dependencies in `packages/` resolve against the manifest's requirements)
- Test command dry-run (venv creation + `pip install` succeeds from local cache)
- No unexpected file types (reject executables, archives, non-source files outside `packages/`)
