# LoRA Training Strategy

Per-stage LoRA training targets, deployment architecture, training data sources (bootstrapping + self-improvement + cold-start datasets), distillation strategy, training infrastructure, and self-improvement guardrails.

---

## 1. Per-Stage LoRA Training Targets

**Qwen3-4B LoRAs (reasoning):**

| Stage | Training Pair | Quality Signal | Technique | Rank | Examples |
|-------|--------------|----------------|-----------|------|----------|
| Task Analysis | task description -> structured task analysis | did retrieval find the right files? | SFT | 16-32 | 3-5K |
| Scope | task + candidates -> relevance classification | were the included files actually needed? | SFT | 16-32 | 2-4K |
| Precision | task + symbols -> tier classification | did the selected symbols lead to correct code? | SFT + curriculum | 16-32 | 3-5K |
| Execute (Plan) | context + task -> plan | did the plan lead to successful implementation? | CoT-SFT + DPO | 32-64 | 8-15K |

**Qwen2.5-Coder-3B LoRAs (coding):**

| Stage | Training Pair | Quality Signal | Technique | Rank | Examples |
|-------|--------------|----------------|-----------|------|----------|
| Execute (Code) | context + task -> code edits | did the code pass validation? | SFT + DPO | 32-64 | 10-50K |

**Note**: Stage 4 (Execute Plan) requires two-phase training (CoT-SFT then DPO) due to its high-dimensional output space — planning has no ground truth and requires structured reasoning. Rank 32-64 is recommended based on ablation evidence (r=16 optimal for code generation; planning's higher-dimensional output warrants more capacity but r=128 appears excessive at sub-7B scale). Ablate rank early.

**Training pair format**: NL-to-Code instruction format (natural language description → code) significantly outperforms Code-to-Code format across all benchmarks (OpenCodeInstruct, 2025). All training pairs — including synthetic pairs from commit histories — should use natural language task descriptions as input, not code-in → code-out.

Training data curation must separate examples by base model -- a Qwen3-4B reasoning LoRA cannot be applied to Qwen2.5-Coder-3B and vice versa.

---

## 2. Per-Stage Model Overrides and Deployment Architecture

Each pipeline stage can specify a model name override via `[models.overrides]` in config. The stage registry resolves: stage name -> override (if exists) -> role-based default -> base model.

**Phases 1-3 (MVP)**: Ollama or any configured provider. Single model at a time, no adapter switching needed.

**Phase 4 deployment target**: vLLM with per-request LoRA adapter selection. Ollama cannot hot-swap LoRA adapters per request (each adapter is a separate model tag requiring full unload/reload). vLLM's `--lora-modules` flag enables multiple adapters loaded simultaneously with per-request selection via the `model` field in the OpenAI-compatible API.

- **Primary**: vLLM server — quantized base model (~2-2.5 GB VRAM) + up to 8 LoRA adapters (~20-50 MB each) + KV cache = ~3-5 GB total VRAM on a 24 GB GPU.
- **Fallback**: llama-server (llama.cpp) with LoRA support.

The `[models.overrides]` stage→model mapping stays the same regardless of provider — only the inference server changes. Override values become adapter identifiers (e.g., `scope-v1`) that the provider translates to its native adapter selection mechanism.

---

## 3. Raw DB as Per-Stage Training Corpus

Each stage's training data can come from two sources:

**Logged runs (self-improvement)**: Extracted from `retrieval_llm_calls` filtered by:
- `call_type` (which stage produced it)
- linked `task_runs.success` (was the overall task successful?)
- optionally `retrieval_decisions` (was this specific decision part of a successful run?)

**Synthetic pairs (bootstrapping)**: Generated from external repo commit histories via the bootstrapping pipeline ([Section 5](#5-bootstrapping-from-external-repo-history)). These provide training data before the agent has produced any real runs.

Both sources produce datasets in the same format and are stored with a `source` field to allow mixing and weighting during training.

---

## 4. Training Artifact Storage

- Training plans and curated datasets go in raw DB (they're generated outputs, same as enrichment outputs).
- Adapter metadata (which stage, performance metrics, active/inactive) goes in curated DB (it's verified configuration the pipeline reads at runtime).

---

## 5. Bootstrapping from External Repo History

The self-improvement loop has a cold-start problem: Phase 4 training needs quality-signal-rich data, but Phase 3's output quality depends on models that haven't been trained yet. External repos break this dependency.

### 5.1 Data Source

Clone mature, well-tested repos from GitHub. Ideal repos have:
- Clear commit messages (natural task descriptions).
- Small, focused bug-fix and feature commits (clean before/after pairs).
- Stable dependency graphs and well-established symbol structure (high-quality indexer output).
- Passing CI (implicit quality signal: the fix stuck, tests passed, it got merged).

**Domain distribution**: The training corpus should span **6+ distinct domains** with no more than 20% from any single domain. This prevents the model from conflating domain idioms (parser visitor patterns, attrs-ecosystem conventions) with the target coding style. Include repos from authors across at least 5 different organizations, vary codebase sizes from 1K to 50K LOC, and mix library code with tool code.

**Anti-patterns**: Web frameworks (Django, Flask, FastAPI), HTTP clients (requests, httpx), task queues (Celery), and message brokers look like good candidates — they have clean architecture and sophisticated exception hierarchies — but embed deeply defensive patterns (top-level catch-all handlers, silent fallbacks, `.get()` with defaults everywhere). Any project whose primary job is keeping a long-running process alive will be defensive at its boundaries. Exclude these from the training corpus.

A curated fail-fast repository corpus with specific repo recommendations and an AST-based heuristic scorer for programmatic identification is documented in `research_reviews/fail_fast_research.md`. See also [Section 5.7](#57-fail-fast-training-corpus).

### 5.2 Commit Filtering Criteria

**Extraction tool**: PyDriller for commit traversal and diff extraction. **CommitChronicle** (JetBrains Research) provides a reproducible collection pipeline built on PyDriller with deduplication and outlier filtering. The **D3 paper** is the best architectural reference for LLM-powered instruction labeling of code edit sequences at the 1-3B parameter range. Note: no end-to-end commit→training-pair tool exists — expect 1-2 weeks of custom engineering for the full pipeline.

Not every commit is a useful training example. Filter for:
- **Diff size bounds**: single-file or small-cluster changes (e.g. 1-5 files, < 200 lines changed). Massive refactors are noise. CommitPackFT (NeurIPS 2023 Workshop) found that single-file commits produce the highest-quality instruction-completion pairs.
- **Descriptive messages**: commits with imperative-mood messages starting with action verbs ("Fix," "Add," "Verify") serve as synthetic task descriptions. Apply Verb-Direct-Object pattern filtering via NLTK/spaCy POS tagging to retain only actionable messages. Skip sub-3-word messages and generic tokens ("wip", "misc", "update").
- **Non-merge commits**: merge commits conflate multiple logical changes.
- **Language match**: filter to Python/TS/JS files matching the indexer's current capability.
- **Exclude automated commits**: filter out dependabot updates, version bumps, CI config changes, and documentation-only commits.

**Message regeneration**: For commits with poor messages but good diffs, use **OpenCommit** or a local Ollama model to regenerate descriptions from diffs. The **OMG paper** (ACM 2024) shows that ReAct prompting with broader software context dramatically improves generated descriptions over diff-only approaches. Apply LLM-as-judge scoring to filter generated descriptions, keeping only high-scoring ones.

### 5.3 Synthetic Pipeline Run Generation

From a filtered commit, reverse-engineer what a correct pipeline run *should have* looked like:

| Stage | Synthetic Training Pair | Derivation |
|-------|------------------------|------------|
| Task Analysis | commit message -> structured task description | The commit message is the task; the diff tells you what files/symbols were relevant. |
| Scope | task + repo files -> file relevance classification | Files changed in the diff are "relevant"; their direct dependencies are "supporting"; unrelated files are "irrelevant". |
| Precision | task + file symbols -> symbol tier classification | Symbols modified in the diff are "primary"; symbols referenced by modified code are "supporting"; others are "excluded". |
| Execute (Code) | context + task -> code edits | The diff itself is the target output. Context is the pre-commit state of relevant files. |
| Execute (Plan) | context + task -> plan | Derive a plan from the diff: which files to change, what changes, in what order. |

**LintSeq** (ICLR 2025) generates synthetic edit sequences from existing code using only a linter — no LLM needed for data generation. It decomposes complete programs into instruction + file-state + diff-sequence tuples. Models trained on LintSeq edit sequences show 3-6 point pass@1 improvements on HumanEvalFix. Apply LintSeq to transform any existing code dataset (e.g., OpenCodeInstruct subsamples) into edit-sequence format aligned with our search-and-replace output.

**Instruction format**: All synthetic training pairs should use **NL-to-Code format** (natural language task description → code output) rather than Code-to-Code format (code input → code output). OpenCodeInstruct (2025) demonstrated that NL-to-Code significantly outperforms Code-to-Code across all benchmarks. This means commit messages (or regenerated descriptions) should serve as the instruction, not the pre-commit code state.

This is a distinct data curation step from `cra curate-data` (which curates from the agent's own logged runs). It may be implemented as a preprocessing pipeline or as an additional mode.

### 5.4 Quality Signal Mapping

External repo commits provide quality signals analogous to what the self-improvement loop gets from logged runs:

| Self-Improvement Signal | External Repo Equivalent |
|------------------------|--------------------------|
| `task_runs.success` | commit was merged / tests passed |
| retrieval decision correctness | did the diff touch the files scope would have found? |
| symbol selection correctness | did the diff modify the symbols precision would have selected? |
| code generation correctness | does the generated diff match the actual commit diff? |

### 5.5 Storage

Synthetic training pairs from external repos are stored in the same raw DB tables as self-improvement training data (`training_datasets`, `training_plans`). A `source` field distinguishes them:

- `source = "synthetic"`: generated from external repo commit history.
- `source = "logged"`: curated from the agent's own logged runs.

This allows training to mix both data sources as the agent matures.

### 5.6 Cold-Start Datasets

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

### 5.7 Fail-Fast Training Corpus

Style-targeted training data is a distinct concern from functional-capability training. The code generation LoRA should be biased toward fail-fast error handling (custom exception hierarchies, rich contextual error messages, narrow `except` clauses, no silent failure) to align with the project's coding style principles.

A curated 25-repo corpus spanning 6 domains (parsers/compilers, validation/type checking, structured data, developer tooling, small focused libraries, frameworks with good error design) is documented in `research_reviews/fail_fast_research.md`. Top-tier repos include strictyaml, attrs, cattrs, LibCST, structlog, Black, typeguard, and Zod (TS). The corpus targets 5,000-10,000 high-quality commit pairs filtered for error-handling improvements.

**Programmatic identification**: An AST-based heuristic scorer using 8 weighted metrics (bare-except ratio, blind-except ratio, try-except-pass count, specific exception ratio, dict["key"] vs .get() ratio, assert density, raise density, exception chaining ratio) can rank candidate repos before manual review. Repos scoring 75-100 are strong candidates. This scorer can be built as part of the Phase 4 bootstrapping tooling to expand the corpus beyond the manually curated list.

**DPO preference pairs**: The CodeFavor approach (pre-commit code as rejected, post-commit code as accepted) is directly applicable. Repos like cattrs have commits showing progression from generic `raise Exception(...)` to custom exception classes — ideal before/after training pairs for preference optimization.

---

## 6. Distillation Strategy

Per-stage LoRA adapters are trained via teacher-student distillation: large teacher models generate high-quality training examples, which are used to fine-tune the small local models.

### 6.1 Teacher Models

| Stage | Target Model | Primary Teacher | Secondary |
|-------|-------------|----------------|-----------|
| 1-3 (Task Analysis, Scope, Precision) | Qwen3-4B | Qwen3.5-397B-A17B (Apache 2.0, $0.11/M input) | — |
| 4 (Execute — Plan) | Qwen3-4B | Qwen3.5-397B thinking mode | DeepSeek-V3.2 (cross-validation) |
| 5 (Execute — Code) | Qwen2.5-Coder-3B | Qwen3-Coder-Next-80B-A3B | OpenCodeInstruct dataset |

**Tokenizer incompatibility**: Qwen3.5 uses a 250K vocabulary vs Qwen3-4B's 150K. This means only response-level SFT (train on teacher's text output), not logit-level distillation (KL divergence on token probabilities). Fallback for logit distillation if needed: Qwen3-235B (same tokenizer family as Qwen3-4B).

### 6.2 Per-Stage Training Configurations

| Stage | Technique | Rank | Examples | Time (RTX 4090) |
|-------|-----------|------|----------|-----------------|
| 1 Task Analysis | SFT | 16-32 | 3-5K | ~15-30 min |
| 2 Scope (relevance) | SFT | 16-32 | 2-4K | ~10-25 min |
| 3 Precision (symbol selection) | SFT + curriculum | 16-32 | 3-5K | ~15-30 min |
| 4 Execute — Plan | CoT-SFT + DPO | 32-64 | 8-15K | ~30-90 min |
| 5 Execute — Code | SFT + DPO | 32-64 | 10-50K | ~1.5-3.5 hrs |

**Total estimated training time**: ~2-4 hours for all 5 stages on an RTX 4090 with QLoRA rank-32 and sequence length 2048. RTX 3090 adds ~60-70% to these times. RTX 4080 (16GB) adds ~100%.

Stage 4 (planning) gets two-phase training because planning has a high-dimensional output space with no ground truth. Phase 1: CoT-SFT on teacher reasoning traces. Phase 2: DPO with plan quality as the preference signal (successful implementation vs failed). Rank 32-64 is recommended — ablation evidence shows r=16 optimal for code generation, and while planning's output space is larger, r=128 appears excessive at sub-7B scale. **Ablate rank early** (compare 32 vs 64 on a held-out evaluation set) before committing.

### 6.3 Training Infrastructure

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

**Estimated cost**: ~$60-150 total teacher inference API calls. ~2-4 hrs total local GPU training on RTX 4090 (see Section 6.2 for per-stage breakdown).

---

## 7. Self-Improvement Guardrails

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
- **Planning stage transitions to self-improvement LAST**: planning is the hardest stage with the most room for subtle degradation. It stays on teacher distillation longest.

**Timeline** (ordered by increasing risk):
1. Pure teacher distillation (Stages 1-5)
2. Mixed: on-policy data with teacher scoring (Stages 1-3 first)
3. Execution feedback loop (code validation as signal)
4. Production feedback (real task outcomes as signal, planning stage last)

**Practical thresholds**:
- ~500 quality trajectories can produce measurable gains (SWE-Gym finding) — sets a lower bar for initial self-improvement rounds than expected.
- Repo diversity in the training set matters more than volume for transfer to new tasks (Hybrid-Gym, 2026). Inject fresh task diversity (new repos, new problem types) to break through plateaus rather than collecting more data from the same distribution.
- SelfCodeAlign (NeurIPS 2024) achieves self-alignment without any teacher model on models from 3B to 33B, with the finding that base models benefit most from alignment with their own data distribution. This suggests a hybrid approach: teacher distillation for initial rounds, then on-policy data once baseline capability is established.

---

## 8. Evidence Base and Advanced Techniques

**Per-stage LoRA validation**: IBM Granite Intrinsics Library demonstrated 6 LoRA adapters for RAG pipeline stages with no cross-stage degradation, plus **Activated LoRA (aLoRA)** achieving 20-35× speedup per adapter invocation via KV cache reuse. META-LoRA showed benefits more pronounced for small models. **R-LoRA** (EMNLP 2025 Findings) explicitly tested on **Qwen2.5-3B**, showing multi-head LoRA outperforms vanilla LoRA on multi-task benchmarks — direct evidence for our target model. MTL-LoRA found task-specific transformation matrices consistently outperform both single-task and vanilla multi-task LoRA, mitigating the "seesaw effect." Note: no published project has done per-stage LoRA for coding agent pipelines — this is genuinely novel.

**Adapter routing**: LORAUTER (January 2026) routes queries to appropriate adapters via task embeddings, scaling to 1500+ adapters. LoGo (November 2025) performs training-free, instance-specific selection and merging from a LoRA pool. For our pipeline where stage routing is deterministic (the orchestrator knows which stage it's in), simple adapter selection via the vLLM `model` parameter is sufficient — no learned routing needed.

**Relevance judgment training**: **Self-RAG** (ICLR 2024) trains models with special reflection tokens (`[Relevant]`, `[Irrelevant]`, `[Fully supported]`, etc.) using GPT-4-generated critic labels — the model learns to output structured judgment tokens inline. **ARES** (NAACL 2024) fine-tunes LLM judges on (query, passage, answer) triples with only ~150 human-annotated validation datapoints. Both adapt directly to code symbol relevance classification for the scope and precision stages.

**Edit-sequence training**: **LintSeq** (ICLR 2025) decomposes complete programs into synthetic edit sequences using a linter (no LLM needed), producing instruction + file-state + diff-sequence tuples. Models trained on LintSeq show 3-6 point pass@1 improvements on HumanEvalFix. The **D3 dataset** (3.6M examples from The Stack) extends LintSeq with LLM-powered instruction labeling, tested on Llama 3.2 1B and 3B. **SRI Tuning** (January 2026) trains models to generate search-and-replace edit instructions, tested on Qwen2.5-Coder-Base — directly aligned with our output format.

**Self-improvement at scale**: **SWE-Gym** (ICML 2025) used LoRA via Unsloth to train Qwen-2.5-Coder-7B, finding plateau after two iterations. **SWE-smith** (NeurIPS 2025) extended to 50K task instances, training SWE-agent-LM-32B to 40.2% on SWE-bench Verified. **R2E-Gym** (COLM 2025) introduces SWE-GEN, deriving training environments from version-control commits via back-translation — closest system to our commit-history bootstrapping. **Hybrid-Gym** (February 2026) found repo diversity more important than volume for transfer. **Lingma SWE-GPT** (Alibaba) trained 7B/72B Qwen models, with the **7B model resolving 18.20% of SWE-bench Verified** — proof small models can work with proper training.

**Small-model bootstrapping**: **SelfCodeAlign** (NeurIPS 2024) achieves self-alignment without teacher from 3B to 33B. **LADDER** improved Llama 3B from 2% to 82% on integration problems using generate→solve→verify→learn with explicit curriculum. The **Code Graph Model** (May 2025) fine-tunes LLMs with LoRA to understand repository-level structure using graph-to-text conversion.

**SAD (Structured Agent Distillation)**: If CoT-SFT + DPO underperforms for planning, an advanced technique to try: segment plan output into `[REASONING]` and `[PLAN]` spans with different loss weights (higher weight on the plan structure, lower on reasoning traces). Apply curriculum learning (single-file plans → multi-file plans → complex dependency chains).
