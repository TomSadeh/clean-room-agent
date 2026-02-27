# Qwen3 0.6B and 1.7B: Small Model Assessment

Date: 2026-02-27

## Summary

Qwen3's smallest dense models achieve a ~2x efficiency gain over Qwen2.5: the 1.7B matches Qwen2.5-3B across most benchmarks. Both support hybrid thinking/non-thinking mode via strong-to-weak distillation. The 0.6B is too fragile for code generation but viable for narrow classification tasks after fine-tuning. The 1.7B is the minimum viable option for code-adjacent work.

## Architecture

Both are dense transformers. Same architecture as larger Qwen3 (GQA, SwiGLU, RoPE, RMSNorm), plus QK LayerNorm new in Qwen3. Differ only in width.

| Parameter | 0.6B | 1.7B |
|---|---|---|
| Total params | 0.6B (752M Ollama) | 1.7B (2.03B Ollama) |
| Non-embedding params | 0.44B | 1.4B |
| Hidden size | 1024 | 2048 |
| FFN size | 3072 | 6144 |
| Layers | 28 | 28 |
| Q heads | 16 | 16 |
| KV heads | 8 | 8 |
| Head dim | 128 | 128 |
| GQA ratio | 2:1 | 2:1 |
| Vocabulary | 151,936 | 151,936 |
| Context (native) | 32K | 32K |
| RoPE theta | 1,000,000 | 1,000,000 |
| Tied embeddings | Yes | Yes |

Critical observation: at 0.6B, the embedding matrix consumes ~0.16B parameters (27% of total). The actual "reasoning capacity" is 0.44B non-embedding parameters. This is a consequence of the 151K vocabulary designed for multilingual coverage — the vocabulary is oversized for a model this small.

## Training

- **Pretraining corpus**: ~36 trillion tokens, 119 languages. ~2x Qwen2.5's 18T.
- **Three-stage pretraining**: Stage 1 (>30T tokens, 4K context, general). Stage 2 (~5T tokens, STEM/code/reasoning focus). Stage 3 (long-context extension to 32K).
- **Post-training (instruct)**: Strong-to-weak distillation from larger Qwen3 frontier models. Off-policy distillation in both thinking and non-thinking modes. This is how small models acquire reasoning capability they cannot learn from direct training alone.

## Benchmarks

### Base Models

| Benchmark | 0.6B | 1.7B |
|---|---|---|
| MMLU | 52.8 | 62.6 |
| BBH | 41.5 | 54.5 |
| GSM8K | 59.6 | 75.4 |
| MATH | 32.4 | 43.5 |
| EvalPlus (code) | 36.2 | 52.7 |
| MBPP (code) | 36.6 | 55.4 |
| MultiPL-E (code) | 24.6 | 42.7 |
| CRUX-O (code) | 27.0 | 36.4 |

### Instruct + Thinking Mode

| Benchmark | 0.6B |
|---|---|
| MMLU-Redux | 55.6 |
| MATH-500 | 77.6 |

Thinking mode provides ~10-20 point improvements on math and code at this scale. MATH-500 jumps from ~32 to 77 on 0.6B. But slight degradation on pure retrieval/factual tasks (thinking doesn't help when the knowledge isn't there).

### vs Qwen2.5

**Qwen3-1.7B matches Qwen2.5-3B** across most benchmarks. **Qwen3-0.6B significantly exceeds Qwen2.5-0.5B**, especially on STEM and coding. Roughly 2x efficiency improvement — same capability at half the parameters.

## Quantization Sensitivity

This is load-bearing for deployment decisions.

### 0.6B

| Quantization | MMLU | Perplexity | Assessment |
|---|---|---|---|
| BF16 (baseline) | 47.1 | 20.9 | Reference |
| Q8 (W8A16) | 47.0 | ~20.9 | Near lossless |
| Q4 (W4A16) | 42-44 | 25-27 | ~10% MMLU drop. Notable. |
| Q3 | ~25 | — | Collapsed. Not viable. |
| Q2 | — | — | Useless. |

### 1.7B

| Quantization | MMLU | Assessment |
|---|---|---|
| BF16 | ~60 | Reference |
| Q8 | 59.8-60.0 | Negligible impact |
| Q4 | 52.5-55.7 | 7-9% drop. Moderate. |
| Q3 | ~27 | Severe. Not viable. |

**Key finding**: Activation quantization (W8A8) is particularly damaging compared to weight-only quantization (W8A16). At 0.6B, Q4 already causes meaningful degradation. At 1.7B, Q4 is tolerable but not ideal. **For fine-tuning deployment: prefer Q8 or BF16 for the 0.6B, Q4 is acceptable for 1.7B.**

## Fine-tuning

- **Unsloth**: Both supported. 2x speed, 70% less VRAM. 1.7B FP8 GRPO runs on 5GB VRAM.
- **Tunability**: DistilLabs benchmarked 12 small models across 8 tasks. Qwen3 models "consistently deliver the strongest results after fine-tuning." The 0.6B showed the largest post-fine-tuning gains (high tunability). 1.7B average rank 4.44/12 after tuning.
- **Recommended mix**: 75% reasoning data, 25% non-reasoning data to preserve reasoning capability during specialization.
- **Tokenizer pruning**: The Qwen Tokenizer Pruner project exists specifically because the 151K vocab is oversized for small models. Pruning could recover parameter budget for actual computation.

## Ollama / llama.cpp

| Model | Command | Size (Q4) |
|---|---|---|
| 0.6B | `ollama run qwen3:0.6b` | 523 MB |
| 1.7B | `ollama run qwen3:1.7b` | 1.4 GB |

Official GGUF from Qwen on HuggingFace. Must set `num_ctx` explicitly in Ollama (defaults to 2048, model supports 40960).

## Implications for Jane

> **Architecture revision (Feb 2026):** Planning decomposition (binary judgment per item, multi-stage meta/part plans) reduced per-call complexity enough that 1.7B is now the **primary model for all pipeline roles**, not just a fast classifier. The 4B is likely eliminated — its reasoning advantage over 1.7B is marginal once planning tasks are decomposed into smaller cognitive steps. The 0.6B remains viable for binary classification stages. See `protocols/design_records/binary_decomposition_and_model_tiers.md`.

### Primary Model for All Pipeline Stages (Revised Role)

With planning decomposition, every pipeline stage — including plan generation and code implementation — now consists of smaller, structured sub-tasks (enumeration, binary yes/no judgments, grouping). These decomposed tasks fall within the 1.7B's demonstrated capability range:

- **Scope judgment**: Binary relevant/irrelevant per file with reason. Narrow task. 1.7B with LoRA handles this directly.
- **Precision classification**: Four-class (primary/supporting/type_context/excluded) per symbol. Still constrained and within 1.7B range.
- **Routing**: Select from ~5 stage names given task summary. Trivial classification.
- **Decomposed planning**: Enumeration → binary dependency judgments → grouping. Each sub-call is a structured, bounded task — no single call requires the deep multi-step reasoning that motivated the original 4B selection.
- **Code implementation**: Search/replace edits guided by a decomposed plan. The plan provides sufficient structure that the implementation calls are bounded.

Inference speed advantage: ~3-4x faster than 4B on same hardware. For the audit protocol's 16 reference tasks, this means audit runs complete faster, tighter iteration loops.

### 0.6B for Binary Classification

The 0.6B can't generate code or plans, but a fine-tuned 0.6B doing binary relevance filtering on pre-filtered candidates is a much simpler task than general generation. With planning decomposition producing many binary yes/no judgment calls, the 0.6B becomes a natural fit for the highest-volume, lowest-complexity stage of the pipeline — binary relevance and binary dependency decisions.

### Negative Transfer Consideration

The CLAUDE.md architecture decision notes that code-specialized models have Python priors that interfere with C. At 0.6B/1.7B, the model has less capacity to hold conflicting priors — a generalist base may actually be cleaner for LoRA specialization than a code-specialized one at this scale. The Qwen3 generalist + LoRA path avoids the negative transfer problem entirely.

### Training Compute Impact

With 2x RTX 5090:
- 1.7B LoRA: ~1-2 days (vs ~7-8 days for 4B). Fast iteration.
- 0.6B LoRA: ~hours. Can experiment rapidly.
- The self-improvement loop (Phase 4) benefits enormously from faster training cycles. More iterations per unit time = faster convergence.
- Eliminating the 4B removes the most compute-intensive training target entirely.

### Model Architecture (Revised)

Previous plan: Qwen2.5-Coder-3B (coding) + Qwen3-4B (reasoning).

Current architecture: **Qwen3-1.7B (primary, all roles)** + **Qwen3-0.6B (optional, binary classification)**. The 4B is likely eliminated — planning decomposition reduced per-call complexity below the threshold where the 4B's additional capacity provides measurable benefit. The two-model architecture is simpler to train, deploy, and maintain than the original three-tier cascade.

This remains config-only in the current architecture — the ModelRouter supports per-stage model resolution. No code changes needed, just config.toml entries.

### Context Window

32K native for both. Matches the project's target of "32K window at ~100% signal relevance." The smaller models can't effectively use more context anyway — attention quality degrades with length at this scale. 32K is the right ceiling.

## Open Questions

1. **Tokenizer pruning**: The 151K vocab is 27% of 0.6B parameters. Pruning to a code-focused ~32K vocab could double the effective reasoning capacity. But this requires re-training from scratch (or significant continued pretraining). Worth investigating if 0.6B becomes the filter model.
2. **Thinking mode overhead**: For classification tasks, how many thinking tokens does the model generate? If it spends 500 tokens thinking about a binary relevant/irrelevant decision, that's wasted latency. May want to use non-thinking mode for simple classifications.
3. **Distillation quality at 0.6B**: The instruct models are distilled, not directly trained. How robust is the distilled reasoning under domain shift (general → code retrieval)? Fine-tuning may override the distilled behavior.

## Sources

- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- [Qwen3 Blog Post](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen3-1.7B Model Card](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Qwen3-0.6B config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- [Qwen3-1.7B config.json](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json)
- [Qwen3 Quantization Study (arXiv:2505.02214)](https://arxiv.org/html/2505.02214v1)
- [Unsloth Qwen3 Fine-tuning Guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)
- [DistilLabs Small Model Benchmark](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning)
- [Qwen Tokenizer Pruner](https://github.com/KaihuaTang/Qwen-Tokenizer-Pruner)
