# Fine-tuning Qwen3-4B-Instruct-2507: a comprehensive guide

> **Architecture note (Feb 2026):** The 4B model is under evaluation for elimination from the pipeline. Planning decomposition reduced per-call complexity enough that Qwen3-1.7B may handle all roles. This research remains valid for cases where the 4B is retained for complex reasoning, or as a reference for fine-tuning methodology applicable to any Qwen3 model. See `protocols/design_records/binary_decomposition_and_model_tiers.md`.

**Qwen3-4B-Instruct-2507 is one of the most capable small language models available for fine-tuning today, and parameter-efficient methods like QLoRA make it trainable on a single consumer GPU with as little as 5 GB of VRAM.** Independent benchmarks by Distillabs across 8 tasks and 12 models ranked it the top-performing small model for fine-tuning — outperforming even the larger Qwen3-8B. The "2507" release (August 6, 2025) split the original hybrid-thinking Qwen3-4B into separate Instruct and Thinking models, extended native context to **262K tokens**, and significantly improved instruction following and alignment. This guide covers every practical dimension of fine-tuning this model: architecture details, all major adaptation methods, hardware requirements, tooling, tradeoffs, and hard-won community discoveries about gotchas that can silently break your training runs.

---

## The architecture under the hood

Qwen3-4B-Instruct-2507 is a **4.02-billion parameter** dense transformer built on Grouped Query Attention (GQA), SwiGLU MLPs, and RMSNorm. Its architectural specifics matter for configuring LoRA targets and estimating memory.

| Specification | Value |
|---|---|
| Layers | 36 |
| Hidden size | 2560 |
| Query heads / KV heads | 32 / 8 (4:1 GQA ratio) |
| Head dimension | 128 |
| Intermediate (MLP) size | 9728 |
| Vocabulary size | 151,936 |
| Max context (native) | 262,144 tokens |
| RoPE theta | 5,000,000 |
| Normalization | RMSNorm (ε=1e-6) with per-head Q/K norms |
| Activation | SiLU (SwiGLU) |
| Tied embeddings | Yes (input embedding = output lm_head) |
| Precision | bfloat16 |
| Minimum transformers version | ≥ 4.51.0 |

A notable design choice: the Q projection maps from hidden size 2560 to 4096 (32 heads × 128 dim), meaning the attention dimension is larger than the hidden dimension. The O projection maps back from 4096 to 2560. K and V projections map to 1024 (8 heads × 128 dim). Each of the 36 layers contains approximately **100.9M parameters** across seven linear projections (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) plus four RMSNorm layers.

The tokenizer uses byte-level BPE with ChatML-style special tokens: `<|im_start|>` (151644), `<|im_end|>` (151645), `<|endoftext|>` (151643), plus `<think>` (151667) and `</think>` (151668) for thinking-mode models. Chat formatting follows the standard ChatML template with `<|im_start|>role` and `<|im_end|>` delimiters. Crucially, **no default system prompt** is used in Qwen3 (unlike Qwen2.5).

## What changed in the 2507 release

The "2507" designation refers to a July 2025 training vintage, released August 6, 2025. Two config-level changes differentiate it from the original April 2025 Qwen3-4B: `max_position_embeddings` jumped from **40,960 to 262,144**, and `rope_theta` increased from **1,000,000 to 5,000,000**, enabling native 256K context without YaRN scaling.

The most consequential change was philosophical. The original Qwen3-4B was a hybrid model supporting both thinking (chain-of-thought with `<think>...</think>` blocks) and non-thinking modes via `enable_thinking=True/False`. The 2507 update **split this into two specialized models**: Qwen3-4B-Instruct-2507 (non-thinking only) and Qwen3-4B-Thinking-2507 (thinking only). This simplifies fine-tuning considerably for the Instruct variant — there are no thinking tokens to worry about, no mode toggles to preserve, and no risk of accidentally breaking the thinking/non-thinking switch. The Instruct-2507 model also received substantial improvements in instruction following, multilingual coverage, tool use, and alignment with human preferences.

For fine-tuning, this split means you should choose your base model deliberately: use the Instruct variant for standard task adaptation, and the Thinking variant if you need chain-of-thought reasoning.

## LoRA: the default starting point

Low-Rank Adaptation remains the most popular method for fine-tuning Qwen3-4B, offering an excellent balance of quality, speed, and memory efficiency. The community and official documentation have converged on well-tested configurations.

**Recommended LoRA configuration:**

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                    # rank: 8 (official) or 16 (community default)
    lora_alpha=32,          # typically 2-4× rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

The official ms-swift (Alibaba's SWIFT framework) recommends **r=8, α=32, learning rate 1e-4**, targeting all linear layers. Community practitioners often use r=16 or r=32 for more complex tasks, with Distillabs achieving top benchmark results using **r=64 with lr=5e-5**. The alpha-to-rank ratio matters: setting α=2r (e.g., α=32 for r=16) is a common heuristic, though some practitioners use α=r. For higher ranks (64+), consider **rsLoRA** (`use_rslora=True` in PEFT), which replaces the α/r scaling with α/√r for more stable training.

Target modules should always include all seven linear projections. Early Apple MLX implementations defaulted to only q_proj and v_proj, yielding just **0.28% trainable parameters** versus the expected ~3.5% — a bug that produced visibly worse results. Always specify all modules explicitly or use the `all-linear` shortcut.

**VRAM for LoRA (16-bit base):** approximately **10–14 GB** depending on rank and sequence length, fitting comfortably on an RTX 3090 or 4090. Trainable parameters range from ~1.5M (r=8, attention only) to ~26M (r=64, all linear layers), representing 0.04% to 0.65% of total model parameters. LoRA adapter files are tiny: **6–200 MB** depending on rank, compared to the 8 GB full model.

**Qwen3-specific LoRA considerations:** DeepSpeed ZeRO-3 is incompatible with LoRA on Qwen3 — it breaks gradient flow with a "tensors does not require grad" error. **Use ZeRO-2 instead.** Also ensure you call `model.gradient_checkpointing_enable()` and `model.enable_input_require_grads()` for proper gradient flow through the adapters.

## DoRA: better quality at modest overhead

Weight-Decomposed Low-Rank Adaptation (DoRA) decomposes pretrained weights into magnitude and direction components, applying LoRA only to the directional component while training the magnitude vector separately. This mimics full fine-tuning's learning dynamics more closely than standard LoRA. Enabling it requires a single flag change:

```python
dora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=True,  # single flag enables DoRA
)
```

DoRA's advantage is that it can achieve comparable or better results with **half the rank** of standard LoRA. On LLaMA-family models, DoRA showed +3.7% on commonsense reasoning at 7B scale and +4.4% at LLaMA3-8B. No Qwen3-specific DoRA benchmarks exist yet, but the architectural similarity suggests comparable gains.

The tradeoff is overhead: without caching, DoRA adds **~139% training time and 4% memory**. With PEFT's `DoraCaching()` context manager, this drops to **17% more time but 41% more memory**. A known PEFT issue (#1692) causes excess VRAM from fp32 upcasting in weight norm computation — a community Triton kernel fix saved ~2 GB on large models. After training, DoRA adapters can be merged via `model.merge_and_unload()` with zero inference overhead.

**QDoRA** (DoRA + quantization) works with bitsandbytes 4-bit quantized weights, adding roughly **15% overhead per step** versus QLoRA. FSDP + QDoRA is documented to work; DeepSpeed ZeRO-2 + QDoRA has reported compatibility issues.

## QLoRA: fine-tuning on consumer hardware

Quantized LoRA makes Qwen3-4B trainable on GPUs with as little as **5–6 GB of VRAM** — a Colab T4 or RTX 3060 will suffice. The base model weights are quantized to 4-bit NormalFloat (NF4) while LoRA adapters train in bf16.

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # optimal for normally-distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,       # saves ~0.4 bits/param additional
)
```

**NF4** is strongly preferred over FP4 — it is information-theoretically optimal for the weight distributions in neural networks. **Double quantization** (`bnb_4bit_use_double_quant=True`) quantizes the quantization constants themselves, always worth enabling. After quantization, Qwen3-4B's weights occupy approximately **2–2.5 GB**, with LoRA adapters, optimizer states, and activations bringing total VRAM to roughly **5–8 GB** at batch size 1 and 2048 sequence length.

| QLoRA Configuration | Estimated VRAM |
|---|---|
| 4-bit, r=16, seq_len=2048 | ~5–6 GB |
| 4-bit, r=32, seq_len=2048 | ~6–7 GB |
| 4-bit, r=64, seq_len=4096 | ~8–10 GB |
| 8-bit, r=32, seq_len=2048 | ~7–8 GB |

Community experience from the dad-joke fine-tuning project (Qwen3-32B on 2× RTX 5090) found that **8-bit showed no noticeable quality improvement over 4-bit** while consuming substantially more VRAM. For Qwen3-4B, 4-bit QLoRA is the pragmatic default.

Known quantization issues: QLoRA is **incompatible with DeepSpeed ZeRO-3 and FSDP** — the official Qwen training scripts explicitly warn about this. MoE variants (like Qwen3-30B-A3B) have additional complications: the full 16-bit model must be downloaded before on-the-fly 4-bit conversion, causing RAM/disk pressure. For dense models like Qwen3-4B, this is not a concern. Ensure library versions: `transformers>=4.51.0`, `bitsandbytes>=0.43.0`, `peft>=0.11.1`.

## Full fine-tuning: when adapters are not enough

Full fine-tuning updates all 4.02B parameters and achieves the highest quality ceiling, particularly for domain adaptation with large datasets or continued pre-training. The cost is substantial memory.

| Configuration | Estimated VRAM |
|---|---|
| BF16 + standard AdamW | ~64–70 GB |
| BF16 + 8-bit AdamW | ~38–44 GB |
| BF16 + 8-bit AdamW + gradient checkpointing | ~32–38 GB |

The memory breakdown: 8 GB for bf16 weights, 8 GB for gradients, 24–48 GB for optimizer states (depending on 8-bit vs fp32 AdamW), plus 2–10 GB for activations. **Gradient checkpointing** reduces activation memory by 60–70% at the cost of ~20–30% slower training. An **8-bit AdamW optimizer** (`optim="adamw_8bit"`) halves optimizer state memory with negligible quality impact.

Full fine-tuning is preferable in three scenarios: large-scale domain adaptation with 50K+ samples, continued pre-training on billions of tokens (where the LoRA quality gap is most pronounced), and when training on distributions very different from the pre-training data. Interestingly, recent research (Shuttleworth et al., 2024) found that full fine-tuning actually preserves pre-training spectral structure better than LoRA in continual learning, because LoRA introduces "intruder dimensions" that accumulate across sequential tasks.

For distributed training, **DeepSpeed ZeRO-2** is recommended for multi-GPU LoRA and **ZeRO-3** for full fine-tuning. FSDP works with `fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer` and `FULL_SHARD` strategy. CPU offloading extends capacity further but slows training 2–5× and requires at least 64 GB system RAM. **Always use bf16** — it matches Qwen3's pre-training precision, and fp16 can cause training instability.

## Combining and stacking adapters

LoRA and DoRA cannot be applied simultaneously to the same module — they are mutually exclusive adapter types (DoRA uses LoRA internally for its directional updates). However, PEFT supports several powerful composition strategies.

**Weighted merging** combines multiple trained adapters into one using `add_weighted_adapter()` with methods like TIES (task-arithmetic with sparsification), DARE (Drop And REscale), or simple linear averaging. **Runtime adapter mixing** via `set_adapters(["adapter_1", "adapter_2"], adapter_weights=[0.7, 0.3])` allows real-time blending without merging. The most practical workflow for **iterative fine-tuning** is merge-and-extend: merge adapter A into the base weights with `model.merge_and_unload()`, save the merged model, then train a new LoRA adapter B on top. This can be repeated indefinitely.

A recommended iterative pipeline from community experience: start with **KL-anchored SFT** (LoRA, KL penalty ≈ 0.03–0.10) to learn the target format without destroying base capabilities, then follow with **DPO** (lr ≈ 2e-5, β starting at 0.05 and swept up to 0.20) for preference alignment. Only tune LoRA rank and alpha after finding a stable β value.

## Hardware requirements at a glance

| Method | Min VRAM | Recommended GPU | Training Speed (est.) |
|---|---|---|---|
| QLoRA 4-bit (r=16) | ~5–6 GB | RTX 3060 12GB / T4 16GB | ~1,000–2,000 tok/s |
| QLoRA 4-bit + Unsloth | ~4–5 GB | RTX 4090 24GB | ~2,000–3,500 tok/s |
| LoRA 16-bit (r=32) | ~12–14 GB | RTX 4090 24GB | ~2,500–4,000 tok/s |
| Full FT (bf16 + 8-bit optim) | ~38–44 GB | A100 80GB | ~1,500–2,500 tok/s |
| Full FT (2-GPU ZeRO-2) | ~20–22 GB/GPU | 2× RTX 4090 | ~2,500–4,500 tok/s |

Training time estimates for common scenarios on an RTX 4090 with QLoRA: **1K samples ≈ 15–30 minutes**, **10K samples ≈ 2–4 hours**, **50K samples ≈ 8–16 hours** (3 epochs each). Full model checkpoints are ~8 GB in bf16; LoRA adapters are 25–200 MB depending on rank. Optimizer states add 24–48 GB per checkpoint for full fine-tuning but are negligible for LoRA.

Storage planning: keep at least **30 GB free** for a typical LoRA run (model download + adapter checkpoints + tokenized dataset cache). Full fine-tuning checkpoints with optimizer states can exceed **100 GB each** — budget accordingly.

## The tooling ecosystem

Six frameworks provide mature Qwen3-4B-Instruct-2507 support, each with distinct strengths.

**Unsloth** is the top recommendation for beginners and single-GPU users. It claims **2× faster training and 70% less VRAM** through custom Triton kernels, and provides free Colab notebooks with pre-quantized models (`unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`). Known issue: loading with `load_in_8bit=True` can fail (#3501) — use 4-bit instead. Also verify the chat template matches official Qwen3, as Unsloth may insert `<think>` tokens even for the non-thinking Instruct-2507 model (#3383).

**Axolotl** excels for production and multi-GPU deployments. It supports FSDP1/FSDP2, DeepSpeed, N-D parallelism, sample packing, Cut Cross Entropy (Apple), and optimized LoRA Triton kernels. The `chat_template: qwen3` option works out of the box. Claims 30%+ faster training and cost savings over baseline HuggingFace. Requires `axolotl>=0.9.1`.

**LLaMA-Factory** provides the widest algorithm variety (PPO, DPO, KTO, ORPO, GaLore, DoRA, and more) with a built-in **web UI** for no-code training. Pre-built config: `llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml`. Early versions had a Liger Kernel compatibility issue with Qwen3 (#7965), resolved in later updates. A "fix qwen3 loss" patch was also merged (#7923).

**ms-swift** is Alibaba's first-party framework with official Qwen team backing. Its unique strength is **Megatron integration** enabling tensor, pipeline, context, and expert parallelism — claimed **10× faster than transformers for MoE models**. The official best-practices document recommends: `lora_rank=8, lora_alpha=32, target_modules=all-linear, lr=1e-4, packing=True, use_liger_kernel=True, attn_impl=flash_attn`.

**Hugging Face TRL/PEFT** serves as the baseline reference implementation. TRL's documentation uses Qwen3 as its primary example model, and `trl-lib/Qwen3-4B-LoRA` exists as a pre-trained adapter example. Most flexible for custom training loops and research.

**torchtune** added Qwen3 support in May 2025 but has notable bugs: compiled Qwen3 runs **4× slower than eager mode** (PyTorch issue #156103), and inference from LoRA checkpoints can produce garbage output (#2866). Less mature than alternatives for Qwen3 specifically.

## Critical gotchas the community discovered

Practitioners have identified several issues that can silently degrade or break training. These represent hours of collective debugging.

**Chat template asymmetry in multi-turn data.** Qwen3's template creates different formatting for the last versus intermediate assistant turns — the last turn may include `<think></think>` tags (even in non-thinking mode) while earlier turns do not. The official workaround: split multi-turn trajectories into individual examples and strip all think tags from history turns, training only on the final response.

**The silent tokenizer EOS change.** A tokenizer update changed the EOS token from `<|im_end|>` to `<|endoftext|>`. The chat template still uses `<|im_end|>` as the turn delimiter, but it may no longer be recognized as EOS — causing the model to never learn when to stop generating. Additionally, PAD and EOS tokens can collide, masking EOS during training. **Always explicitly set `eos_token='<|im_end|>'`** in your training configuration and use the tokenizer serialized with your checkpoints.

**TRL `assistant_only_loss` incompatibility.** Qwen3's tokenizer chat template is missing the `{% generation %}` Jinja2 keyword that TRL's SFTTrainer expects for `assistant_only_loss=True`. This causes a RuntimeError. Workaround: modify the template to include generation markers, or disable assistant-only loss.

**Thinking mode preservation.** For the original hybrid Qwen3-4B (not the 2507 Instruct split), fine-tuning on non-reasoning data can permanently break the thinking mode. The Unsloth team recommends a **75% reasoning + 25% non-reasoning** data mix to maintain both capabilities. For tool-calling fine-tuning specifically, ms-swift issue #4835 found that the `loss_scale='ignore_empty_think'` setting is required to preserve capabilities when think tags are empty, and a very conservative learning rate of **5e-6** with weight decay 0.1 prevents tool-pattern overfitting.

**Unsloth template mismatch for 2507.** Unsloth may insert `<think>` tokens during training for the non-thinking Instruct-2507 model but not during inference, creating a train/inference inconsistency (#3383). Verify your template matches the official Qwen3 specification.

## Quality, cost, and forgetting tradeoffs

The quality hierarchy runs from full fine-tuning (highest ceiling) through LoRA 16-bit (~90–95% of full FT quality) to QLoRA 4-bit (~80–90%). These gaps narrow with smaller datasets and widen with larger ones or distribution shifts. Distillabs found that **fine-tuning quality matters more than base model size** — a well-tuned Qwen3-4B matched a 120B+ teacher model on 7 of 8 benchmarks.

Catastrophic forgetting risk is counterintuitive. LoRA keeps base weights frozen, providing natural protection for single tasks. But research shows LoRA introduces "intruder dimensions" — new high-ranking singular vectors that accumulate across sequential fine-tuning tasks, eventually degrading base capabilities. Full fine-tuning, despite updating all weights, actually preserves pre-training spectral structure better in continual learning scenarios. For practical mitigation, use **KL divergence penalty during SFT** (KL ≈ 0.03–0.10) as a "do-no-harm anchor," keep training epochs low (1–3), and include diverse data that exercises the model's original capabilities.

The cost optimization is straightforward: start prototyping with **QLoRA 4-bit via Unsloth** on whatever GPU you have. Once you've validated your data pipeline and task formulation, scale to LoRA 16-bit on better hardware for production quality. Reserve full fine-tuning for domain adaptation at scale or continued pre-training where the LoRA gap is most meaningful.

## Conclusion

Qwen3-4B-Instruct-2507 represents a sweet spot in the current open-model landscape: small enough to fine-tune on consumer hardware via QLoRA (**~5 GB VRAM**), yet capable enough to match models 30× its size when properly adapted. The 2507 release's split into separate Instruct and Thinking variants eliminated the most complex fine-tuning hazard — accidentally breaking the thinking-mode toggle — making the Instruct variant particularly straightforward to work with.

Three insights stand out from synthesizing community experience. First, **target all seven linear layers**, not just attention — the quality difference is dramatic and the memory cost is modest. Second, **chat template and tokenizer bugs are the most common silent failure mode** — always verify EOS token configuration, check for `<think>` tag injection mismatches, and serialize your exact tokenizer with checkpoints. Third, **DeepSpeed ZeRO-2 is the safe choice** for distributed LoRA training on Qwen3; ZeRO-3 breaks gradient flow with LoRA adapters, and FSDP is incompatible with QLoRA. For anyone starting today, the fastest path to results is Unsloth with QLoRA 4-bit, rank 8–16, targeting all linear modules, at learning rate 1e-4 — a configuration that runs on free Colab instances and has the most community validation behind it.