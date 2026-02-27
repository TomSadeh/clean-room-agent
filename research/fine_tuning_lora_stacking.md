# Full fine-tuning + LoRA stacking for specialist code classifiers

**The full fine-tune → LoRA stacking pattern is proven, practical, and essentially the same pipeline every major AI lab uses for RLHF alignment.** Applying LoRA adapters on top of a fully fine-tuned model is trivially supported by the PEFT library, validated by the SFT→DPO/RLHF workflow used in ChatGPT, Llama, and Qwen, and formally studied through ReLoRA (ICLR 2024) for iterative merge-and-retrain. The hypothesis that a fully fine-tuned specialist with LoRA-based self-improvement will dramatically outperform a base model with LoRA alone is well-supported — but the optimal architecture likely involves a hybrid: encoder models for the two pure-classification stages and a causal LM only for JSON parsing. No existing coding agent uses a fine-tuned small model for file relevance classification, making this approach a genuine innovation over the expensive LLM-prompting methods used by Agentless, CodeMonkeys, and Moatless Tools.

> **Architecture revision (Feb 2026):** This report's references to "fully fine-tuned Qwen3-4B specialist" should be read as applying to the current primary target: **Qwen3-1.7B**. Planning decomposition reduced per-call complexity enough that the 4B is likely eliminated. The full FT + LoRA stacking methodology described here applies identically to the 1.7B — at lower VRAM and faster training times. See `protocols/design_records/binary_decomposition_and_model_tiers.md`.

---

## 1. Full fine-tuning a 4B model on 24GB is tight but achievable

A 4B parameter model in bf16 requires approximately **8 GB for model weights, 8 GB for gradients, and 24–48 GB for optimizer states** (AdamW stores three copies: fp32 weights, first and second moments). Total without optimization: ~64 GB. This makes naive full fine-tuning impossible on a 24GB GPU.

Three strategies bring it within reach. First, **8-bit optimizers** (via bitsandbytes) cut optimizer memory roughly in half, to ~24 GB. Second, **gradient checkpointing** reduces activation memory by ~50% at the cost of 20–30% slower training. Third, **DeepSpeed ZeRO-2 with CPU offload** moves optimizer states to system RAM, requiring ≥64 GB of CPU memory but dramatically reducing GPU pressure. The DeepSpeed memory estimator confirms a 3B model with ZeRO-2 offload needs only ~6–16 GB of GPU memory for parameters and gradients, with activations on top.

**Unsloth** is the most practical path for single-GPU full fine-tuning. It explicitly supports `full_finetuning = True` for Qwen3-4B with optimized Triton kernels, claiming 70% VRAM reduction. Community reports show a TinyLlama model using ~10 GB via Unsloth versus ~24 GB with standard Transformers. For Qwen3-4B with short sequences (~512 tokens), batch size 1, gradient checkpointing, and Unsloth optimizations, **24 GB is plausible but tight** — expect to be within 1–2 GB of the limit.

The more reliable approach is **renting a cloud A100 80GB** for a few hours. Current pricing as of early 2026:

| Provider | GPU | Price/hr |
|----------|-----|----------|
| Vast.ai | A100 80GB | $0.80–$1.50 |
| Thunder Compute | A100 80GB | $0.78 |
| Lambda Labs | A100 40GB | $1.29 |
| RunPod (community) | H100 80GB | $1.99 |

For 10K–50K classification examples at ~512-token sequences, 3 epochs, full fine-tuning on an A100 takes **1–15 hours**. Total cost: **$5–$50**. This is cheap enough to not be a meaningful constraint. For comparison, LoRA training on the same dataset fits comfortably on a 24GB GPU at ~10–14 GB VRAM (QLoRA at ~5–7 GB) and trains **2–5× faster** than full fine-tuning on the same hardware.

**Framework support** is excellent. Unsloth, Axolotl (Qwen3 since October 2025), LLaMA-Factory, DeepSpeed via HuggingFace Trainer, and Torchtune all support Qwen3 full fine-tuning. Axolotl offers the most configurability with YAML-based configs; Unsloth offers the best single-GPU optimization.

### Catastrophic forgetting is irrelevant — possibly beneficial

For a specialist whose only job is classification, **forgetting general capabilities is not a problem and may be desirable**. The model becomes a "classification appliance" that fully dedicates its capacity to the target task. Research confirms catastrophic forgetting occurs in 1B–7B models during continual instruction tuning (Luo et al., 2023), but this matters only if the model needs retained capabilities. One caveat: if the model should output brief reasoning alongside classifications, some general language facility is needed. In that case, mixing 5–10% general instruction data into training or limiting to 1–2 epochs preserves sufficient capability.

---

## 2. Classification with causal LMs: generate labels as text, not classification heads

Three architectural approaches exist for classification with decoder-only models. **Keeping the causal LM head** (generating labels like `"relevant"` as text) is the standard approach and the only one compatible with GGUF/Ollama deployment. **Replacing with a classification head** (linear layer on last hidden state) gives the fastest inference — single forward pass, no autoregressive generation — but is incompatible with llama.cpp, Ollama, and standard GGUF tooling. **Structured JSON generation** combines classification with metadata and, when paired with constrained decoding, guarantees valid output format.

The evidence strongly favors **keeping the causal LM head with constrained decoding**. A Predibase case study found that combining LoRA fine-tuning with Outlines constrained decoding achieved the best results — correct content AND guaranteed schema compliance. The SLOT paper (Amazon, May 2025) showed a fine-tuned Mistral-7B with constrained decoding achieved **99.5% schema accuracy**, outperforming Claude-3.5-Sonnet by 25 percentage points. Even a fine-tuned Llama-3.2-1B matched larger proprietary models.

For the output schema, place the **reasoning field before the classification label**: `{"reason": "file contains API endpoint definitions used by target module", "classification": "relevant"}`. This leverages the autoregressive nature — the model's reasoning tokens influence the classification token prediction, functioning as implicit chain-of-thought. This pattern costs only 10–20 extra tokens and measurably improves accuracy.

### Full FT vs LoRA: the gap narrows for classification

A controlled Mistral-7B benchmark (December 2025) showed full fine-tuning at **100% accuracy** versus LoRA at 93.8% and QLoRA at 94.5% on 25-category resume classification. The "LoRA vs Full Fine-tuning: An Illusion of Equivalence" paper (arXiv:2410.21228, October 2024) found that for sequence classification tasks, LoRA matched full fine-tuning on task accuracy despite having fundamentally different learned weight structures. The Anyscale Llama-2 study confirmed LoRA performs "nearly on par" with full fine-tuning for text classification and SQL generation, though full FT still wins on math reasoning.

**For a 3-class classification task, expect LoRA to come within 1–6 percentage points of full fine-tuning.** Whether that gap matters depends on the volume — at millions of classifications per day, even 2% more errors compound significantly. Full fine-tuning is justified when maximum accuracy is the goal and the ~$10–50 cloud training cost is acceptable.

---

## 3. LoRA on top of full fine-tuning: a proven, production-ready pattern

### Technical feasibility is trivial

LoRA treats any model weights as a "base model" and injects trainable low-rank decomposition matrices. The PEFT library documentation confirms there is **zero restriction** on whether the model is pretrained or fully fine-tuned. HuggingFace staff have provided explicit working code for the pattern: load fine-tuned model → `get_peft_model()` with new LoRA config → train → `merge_and_unload()`.

### The RLHF/DPO pipeline IS this pattern

The strongest evidence is that the entire RLHF alignment pipeline — used by OpenAI, Meta, Alibaba (Qwen), and every major lab — is precisely "full fine-tune → LoRA for incremental improvement":

- **InstructGPT/ChatGPT**: Pretrain → SFT → RLHF (with LoRA on SFT model)
- **StackLLaMA** (HuggingFace tutorial): SFT with LoRA → merge → RLHF with LoRA on merged SFT model
- **Cerebras DPO**: SFT → DPO using LoRA, showing "noticeable improvement in downstream accuracy"
- **"LoRA is All You Need for Safety Alignment"** (2025): LoRA on reasoning-tuned models achieves safety parity with full-model alignment, even at **rank 1**
- **OpenAI DPO Cookbook**: Recommends "SFT followed by DPO" as best practice, noting it "stabilizes training"

This is not experimental or novel — it is the **dominant paradigm** in modern LLM training.

### Iterative stacking (merge → retrain → merge → retrain) works

**ReLoRA** (ICLR 2024) formally studied the exact pattern of training LoRA → merging into base weights → reinitializing new LoRA → training again. Key findings: successive low-rank updates accumulate into high-rank updates (since rank(A+B) ≤ rank(A) + rank(B)), consistently outperforming single-shot LoRA, with performance comparable to full-rank training at 1.3B scale. **Chain of LoRA** (COLA, arXiv:2401.04151) also validated this iterative merge pattern using residual learning.

Critical engineering requirements for stable iterative stacking:

- **Reset optimizer state** (>90% pruning) between iterations — old Adam moments push the new adapter toward the same subspace, preventing exploration of new directions
- **Learning rate warm-up** after each merge-and-reinitialize cycle
- **Keep models in full/half precision** (bf16/fp16) during merge steps — quantized-to-full-precision merges can introduce precision mismatches that degrade quality
- **Monitor for quality degradation** via a held-out validation set — the "Illusion of Equivalence" paper warns that LoRA introduces "intruder dimensions" in spectral structure that accumulate over sequential fine-tuning

### Advantages over repeated full fine-tuning for self-improvement loops

For the specific use case of a self-improvement loop generating new training data in batches, LoRA offers four decisive advantages over re-doing full fine-tuning each iteration. **Less catastrophic forgetting**: LoRA freezes base weights, preserving the specialist knowledge encoded during initial full fine-tuning. **Lower compute cost**: training ~0.1% of parameters means each iteration takes minutes, not hours. **Rollback capability**: if a LoRA iteration degrades quality, discard it and try different data or hyperparameters without touching the base. **Better for small data increments**: LoRA is more data-efficient and less prone to overfitting on the small batches typical of self-improvement loops.

### LoRA hyperparameters on a fine-tuned base

No published research specifically compares hyperparameters on fine-tuned versus pretrained bases, but the RLHF/DPO literature provides practical guidance. Use **rank 8–16** (sufficient for incremental improvement; even rank 1 works for alignment per the safety alignment paper). Apply LoRA to **all linear layers** (QLoRA paper finding). Set **alpha equal to rank** as baseline. Use a **lower learning rate** (1e-5 to 5e-5) than for LoRA on a pretrained base (1e-4), since the model is already well-adapted. Consider **KL-divergence anchoring** (target KL ≈ 0.03–0.10) to prevent excessive drift from the fine-tuned base.

**Verdict on the core question: Full FT + LoRA stacking is a proven, production-ready pattern with high confidence (9/10).** The only genuinely novel aspect is the self-improvement data generation loop, not the FT+LoRA mechanics.

---

## 4. One fine-tuned base with three LoRAs is optimal — but Ollama complicates it

### Architecture comparison

| Architecture | Disk | VRAM | Switch time | Quality ceiling |
|---|---|---|---|---|
| 1 pretrained base + 3 LoRAs | ~2.7 GB | ~2.5–3 GB | <1s (adapter swap) | Limited by LoRA rank |
| 3 separate fully fine-tuned models | ~7.5 GB | ~2.5–3 GB | 3–10s (full load) | Highest per-task |
| **1 fine-tuned base + 3 LoRAs** | **~2.7 GB** | **~2.5–3 GB** | **<1s (adapter swap)** | **High (FT base + LoRA)** |

The third option — one code-classification-tuned base with three task-specific LoRA adapters — is theoretically best. The shared base encodes general code understanding, and lightweight adapters (~10–50 MB each at rank 8–16) specialize per task.

### Ollama's LoRA limitations are the practical bottleneck

Ollama supports LoRA adapters via `ADAPTER` in Modelfiles, but with significant restrictions: **only one adapter per model**, **no hot-swapping** (each adapter requires a separate Modelfile treated as a distinct model), and switching between adapters triggers a **full model reload (3–10 seconds)**. The underlying llama.cpp already supports per-request LoRA selection, but Ollama hasn't exposed this capability yet.

**Practical workarounds** for the three-task pipeline:

- **Simplest**: Merge each LoRA into the fine-tuned base, creating three separate GGUF models. Accept the 3× disk space (~7.5 GB total) and model switching latency. Set `OLLAMA_KEEP_ALIVE=-1` and `OLLAMA_MAX_LOADED_MODELS=3` to keep all three in memory simultaneously (feasible on 24 GB with Q4 quantized models at ~2.5–3 GB each).
- **Better**: Use **llama.cpp server directly** instead of Ollama — it supports per-request LoRA adapter selection with true hot-swapping and serves multiple adapters simultaneously.
- **Best**: Use **vLLM** with S-LoRA, which can serve thousands of adapters simultaneously with ~3% overhead for 30 adapters. Overkill for three adapters, but eliminates all switching latency.
- **Simplest possible**: Train a **single multi-task model** with task-routing prompt prefixes (e.g., `[FILE_RELEVANCE]`, `[SYMBOL_SELECT]`, `[TASK_ANALYSIS]`). Research shows consolidated multi-task fine-tuned models match single-task specialist performance, and a fine-tuned Phi-3-Mini (3.8B) outperformed GPT-4o on classification tasks in one study. This eliminates all adapter management.

### Model size threshold for full FT vs LoRA

Below ~1.5B parameters, full fine-tuning becomes trivially feasible on consumer hardware and is generally preferable to LoRA — the proportional overhead of adapter computation becomes larger while LoRA's memory-saving advantage diminishes. At 4B, full FT requires aggressive optimization or cloud GPUs. At **0.5B, full fine-tuning needs only ~6 GB** in fp16, making it effortless on any modern GPU.

---

## 5. Training data: 10K–50K examples for full FT, with careful class balancing

### Dataset size thresholds

| Dataset size | Recommendation |
|---|---|
| <1K examples | Use LoRA/QLoRA only; full FT will overfit |
| 1K–5K | LoRA preferred; full FT possible with strong regularization |
| **5K–10K** | **Either works; LoRA is safer for quick iteration** |
| **10K–50K** | **Full FT becomes viable and may outperform LoRA** |
| 50K+ | Full FT clearly preferable for maximum adaptation |

Quality matters more than quantity. Well-curated 5K examples routinely outperform noisy 50K. Multiple sources confirm that classification is an "easier" fine-tuning task requiring less data than generation.

### Class imbalance strategy

In production, the natural distribution is heavily skewed (perhaps 95% of files are irrelevant). Training on this raw distribution teaches the model to always predict "irrelevant." Instead, **curate a training set at roughly 3:1 to 5:1 irrelevant-to-relevant ratio**, even if the real ratio is 100:1. Prioritize **hard negatives** — files that look superficially relevant but aren't (same directory, similar naming, but unrelated functionality). Keep the test set at the natural distribution for realistic evaluation, and use **F1-score per class** rather than accuracy as the primary metric.

### Bootstrapping from git commits requires noise-aware training

The core challenge: "file was changed in commit" ≠ "file is relevant to the described task." A commit touching files A, B, and C for "add user authentication" includes the auth module (truly relevant), a config file (tangentially relevant), and a formatting change committed alongside (irrelevant noise). Expected noise rate: **20–40% for positive labels**, ~5–10% for negatives.

Mitigation strategies: filter to single-purpose, clean commits; link commits to issue/PR descriptions for better task labels; **manually validate 500–1K examples as gold standard**; use confidence-based bootstrapping (train initial model, score its own training data, filter low-confidence examples, retrain). The "negative mining" approach is particularly clean: files NOT touched in any commit for a given task type are strong negatives with low noise.

Training on at least **20–50 diverse repositories** across different domains (web apps, libraries, CLI tools, data pipelines) is critical to prevent learning repo-specific shortcuts.

---

## 6. Quantization barely impacts classification; merged weights eliminate LoRA overhead

### Quantization is safe for classification specialists

Classification tasks are **more resilient to quantization than generation**. A study on fine-tuned 1B models for e-commerce intent classification (October 2025) found GPTQ 4-bit and GGUF quantization maintained **99% accuracy**. RunPod's guide confirms 8-bit quantization typically causes 1–3% accuracy degradation, with classification seeing less impact than generation. The AQUA-LLM framework showed quantization alone degraded accuracy to 52–67%, but **fine-tuning + quantization restored accuracy to 98–100%**.

**Recommended pipeline**: Full fine-tune → merge all weights → quantize to **GGUF Q4_K_M or Q5_K_M** → deploy via Ollama. Expected classification accuracy loss: **<1%**.

### Merged weights = zero inference overhead

A merged LoRA model (via `merge_and_unload()`) is mathematically identical to a standard model — W' = W + αBA/r — with **zero additional inference computation**. Unmerged LoRA adds 10–30% latency to time-to-first-token and roughly 40–60% overhead per layer. Since the deployment scenario uses a single specialist (not multi-tenant), always merge before deployment.

### Constrained decoding speeds up short outputs

For classification outputs of ~20–50 tokens, Ollama's built-in structured output (via llama.cpp GBNF grammars) **actually speeds up generation** by reducing the token search space. When only one valid token exists (JSON scaffolding like `{`, `"`, `:`), those steps can be skipped entirely. Pass `format=YourPydanticModel.model_json_schema()` in the Ollama API. Set `temperature: 0` and `num_predict: 100–150` for deterministic, bounded classification.

---

## 7. Encoder models deserve serious consideration for two of three stages

### The speed differential is enormous

For pure classification, encoder models like **CodeBERT** (125M params) or **UniXcoder** (125M) process input in a **single forward pass** — no autoregressive token generation. A CodeBERT classification pass on 512 tokens takes **2–5 ms on GPU** versus **50–500 ms for a 4B causal LM** generating tokens. That's **10–100× faster** at ~1/32nd the model size.

UniXcoder achieves 97.6 F1 on clone detection and leads on code search MRR. CodeBERT-based classifiers reach **97.2% accuracy on code vulnerability classification**. These are not toy models — they represent the state of the art for encoder-based code understanding.

### The optimal hybrid architecture

A tiered approach outperforms using a single 4B model for everything:

- **Stage 1 (File Relevance)**: Code embedding model (Jina Code V2 at 137M, or CodeRankEmbed at 137M) for coarse retrieval of top-50 candidates → fine-tuned UniXcoder cross-encoder (125M, ~5 ms/pair) for precise classification. Total: **<500 MB, <250 ms for 50 files**.
- **Stage 2 (Symbol Selection)**: Fine-tuned UniXcoder or CodeBERT with multi-label classification head. **125M params, ~2–5 ms per symbol batch**.
- **Stage 3 (Task Analysis / JSON parsing)**: This stage requires generation — only here does a causal LM make sense. Fine-tuned Qwen3-4B (or even Qwen2.5-Coder-1.5B) with structured output. **~100–200 ms per task**.

Without the hybrid: classifying 1,000 files with Qwen3-4B takes ~50–100 seconds. With hybrid: embedding retrieval (5 ms) + cross-encoder classification of 50 candidates (250 ms) = **200–400× speedup** on Stage 1.

### Smaller causal models for code classification

If staying with causal LMs for all stages, **Qwen2.5-Coder-1.5B** deserves strong consideration — it's pretrained specifically on code, supports full fine-tuning on a 24GB GPU with ~18 GB VRAM, and benchmarks show well-tuned 1B models outperform prompted 8B models. **Qwen3-0.6B** showed the largest fine-tuning gains ("highest tunability") in the Distil Labs 12-model benchmark. For binary classification, 0.5B models work well with sufficient training data; for structured JSON output, 1B+ is recommended.

### Reward model / continuous scoring alternative

Instead of discrete classification, a cross-encoder scoring model outputs a continuous **0.0–1.0 relevance score** for code-task pairs. This allows flexible thresholds without retraining, naturally produces rankings, and follows the proven architecture of search rerankers. A ZeroEntropy benchmark found dedicated cross-encoder rerankers achieve ~0.78 NDCG@10, outperforming GPT-5 variants on reranking tasks. Train with Sentence Transformers v4 on `microsoft/codebert-base-mlm`, using MSE or contrastive loss with hard negatives.

---

## 8. This fills a genuine gap in existing coding agent architectures

### No existing agent uses fine-tuned small models for file relevance

Every current coding agent handles context selection through either manual user selection, BM25/embedding retrieval, or expensive LLM prompting:

- **CodeMonkeys** (Stanford, 2025) reads **every Python file** with Qwen2.5-Coder-32B and labels each "relevant" or "not relevant" — an expensive brute-force approach that a fine-tuned small model could replace at a fraction of the cost
- **Agentless** uses hierarchical LLM prompting for file → class → function localization, costing ~$0.34 per issue
- **Moatless Tools** uses structural code analysis with Faiss indexing, avoiding neural classification entirely
- **Cursor** uses AST chunking + embedding retrieval

The user's approach of training a small specialist model for file relevance classification is a **natural optimization** of CodeMonkeys' expensive approach and represents a genuine gap in the current ecosystem.

### Closest precedents

**SPIN** (Self-Play Fine-Tuning, ICML 2024) provides the closest precedent for the self-improvement loop: the model generates training data, learns to distinguish self-generated from ground-truth responses, and iterates. SPIN outperformed DPO with GPT-4 preference data. **Self-Instruct** (ACL 2023) bootstrapped 52K instruction examples from 175 seeds, though 46% of generated data had quality issues — underscoring the need for strong filtering.

For the two-stage training pattern, **LLaVA-RLHF** provides an explicit precedent: full SFT for feature alignment → LoRA-only RLHF on top. **StatLLaMA** (2025) demonstrated multi-stage pipelines (continual pretraining → SFT → DPO → downstream fine-tuning) and warned that downstream task fine-tuning must use **extremely low intensity** to avoid forgetting earlier training.

### Directly reusable resources

Starting points for building code relevance classifiers:
- **Base models**: `microsoft/codebert-base-mlm` (encoder, 125M), `microsoft/unixcoder-base` (encoder, 125M), `Salesforce/codet5-base` (encoder-decoder, 220M), `Qwen/Qwen2.5-Coder-1.5B` (causal)
- **Reranker training**: Sentence Transformers v4 cross-encoder training pipeline, `BAAI/bge-reranker-base` as starting checkpoint
- **Self-improvement**: SPIN codebase (github.com/uclaml/SPIN) with LoRA implementation
- **Iterative LoRA**: ReLoRA merge-and-reinitialize pattern, Chain of LoRA (COLA)
- **Datasets**: CodeSearchNet (2M NL-code pairs), CoIR benchmark for evaluation

---

## Practical blueprint and gaps requiring experimentation

### Recommended implementation path

**Phase 1 — Baseline** (1–2 days): QLoRA fine-tune Qwen3-1.7B on 1K–5K manually labeled examples per task. Deploy merged + quantized on Ollama. This establishes the quality baseline with minimal investment. *(Original target was Qwen3-4B; 1.7B fits more easily on consumer GPUs.)*

**Phase 2 — Full specialization** (1 week): Full fine-tune Qwen3-1.7B on local 24GB GPU or cloud A100 (~$5–20) with 10K–50K examples. Compare accuracy against Phase 1. Simultaneously, fine-tune CodeBERT/UniXcoder cross-encoders for Stages 1–2 and compare speed/accuracy tradeoff. *(Full fine-tuning of 1.7B is more feasible on consumer hardware than the original 4B target.)*

**Phase 3 — Self-improvement loop**: Apply LoRA (rank 8–16) on the full fine-tuned base using production disagreements as training signal. Merge after each successful iteration (validation accuracy improved). Reset optimizer state between iterations. Target monthly retraining cycles.

### Gaps where experimentation is needed

No published work exists on **fine-tuning small models specifically as code file relevance judges** — this is genuinely novel. The classification-accuracy gap between full FT and LoRA for code-specific tasks at the 3–4B scale needs empirical measurement on your data. The quality degradation rate over multiple LoRA merge-and-retrain iterations has been studied for pre-training (ReLoRA) but not for classification self-improvement loops. Whether 0.5B–1B models can match 4B performance on nuanced code relevance decisions (where understanding module boundaries, import graphs, and semantic relationships matters) requires direct comparison. Finally, the noise rate in commit-bootstrapped training data and how much filtering is needed for stable training is dataset-specific.

### Summary confidence levels

| Question | Confidence | Status |
|---|---|---|
| Can LoRA be applied on fully fine-tuned model? | **10/10** | Trivially proven |
| Does the FT→LoRA pattern work well? | **9/10** | Industry standard (RLHF) |
| Can iterative merge→retrain stack? | **8/10** | Proven (ReLoRA, ICLR 2024) |
| Will full FT + LoRA beat base + LoRA alone? | **8/10** | Strong theoretical + empirical support |
| Will Qwen3-1.7B work for code classification? | **7/10** | Smaller than original 4B target; decomposed tasks reduce per-call complexity |
| Is self-improvement via production disagreements viable? | **7/10** | SPIN validates concept; filtering quality is key unknown |
| Can 0.5B–1B replace 4B for classification stages? | **6/10** | Plausible but unproven for code relevance |

The core hypothesis is sound. The pattern is not uncharted territory — it is the backbone of modern LLM alignment. The novelty lies in applying it to code classification with a self-improvement loop, which combines well-established techniques in a configuration that hasn't been published but has no known technical barriers.