# Three ways to train Qwen: LoRA, full fine-tuning, and from scratch

**LoRA adapts a frozen Qwen model by training ~1–2% of parameters on a single 24GB GPU; full fine-tuning updates every weight across multiple A100s; building from scratch gives total control but demands serious engineering.** Each paradigm serves fundamentally different goals — task adaptation, deep behavioral change, or pre-training — and the choice cascades into every decision from dataset format to hardware budget. This report provides a code-oriented, practical comparison across dataset preparation, infrastructure, implementation patterns, and architectural trade-offs for Qwen2.5 and Qwen3 models up to 7–8B parameters within the PyTorch/HuggingFace ecosystem.

---

## Qwen architecture foundations that shape every training decision

Before comparing paradigms, understanding Qwen's architecture is essential because it directly determines target modules, memory profiles, and tokenizer configuration.

All Qwen2.5 and Qwen3 models are **decoder-only causal transformers** using Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), SwiGLU activations, and RMSNorm. Qwen2.5-7B specifically has **28 layers, 28 query heads, 4 key-value heads, hidden size 3584, intermediate size 18944, and vocabulary size 152064**. Qwen3-8B increases this to 36 layers, 32 query heads, 8 KV heads, hidden size 4096 — and removes the attention QKV bias present in Qwen2.5 while adding QK LayerNorm for training stability.

The tokenizer across both families is a Byte-level BPE (BBPE) with ~151,646 actual tokens (Qwen2.5) or ~151,669 tokens (Qwen3), padded to 151,936/152,064 in the embedding matrix for GPU-efficient tensor shapes. The chat template follows ChatML format using `<|im_start|>` and `<|im_end|>` special tokens. Qwen3 adds `<think>` and `</think>` tokens for its reasoning mode.

**Critical tokenizer gotcha:** The `pad_token` must never be set to `<|endoftext|>` — this causes infinite generation loops during fine-tuning. For instruct models, set `pad_token = tokenizer.eos_token` (which is `<|im_end|>`, token ID 151645). In base models, `<|im_start|>` and `<|im_end|>` embeddings are untrained random noise — do not use ChatML formatting with base models without also training the embedding layer.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.pad_token = tokenizer.eos_token   # <|im_end|>, NOT <|endoftext|>
tokenizer.padding_side = "right"            # right for training, left for batched inference with FA2
```

The seven linear projection layers in each transformer block — `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention) and `gate_proj`, `up_proj`, `down_proj` (SwiGLU MLP) — are the primary targets for LoRA adaptation and the layers that dominate memory in full fine-tuning.

---

## Dataset preparation diverges early across the three paradigms

All three approaches use the same fundamental chat format for supervised fine-tuning (SFT), but diverge sharply in scale, preprocessing depth, and tooling.

### Data format: the common foundation

For SFT tasks, all three paradigms consume conversational data in ChatML-compatible format. The most common representations are:

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Explain LoRA."},
  {"role": "assistant", "content": "LoRA is a parameter-efficient..."}
]}
```

Frameworks also accept Alpaca format (`instruction`/`input`/`output` fields) and ShareGPT format (`conversations` with `from`/`value` fields). The tokenizer's `apply_chat_template()` method converts any of these into the correct `<|im_start|>role\ncontent<|im_end|>` sequence. For pre-training in paradigm 3, the format shifts to raw text in JSONL, pre-tokenized into binary files.

### LoRA: small, high-quality datasets win

LoRA can produce meaningful results with as few as **a few hundred high-quality examples**, with a practical sweet spot of **1,000–10,000 examples** for task-specific adaptations like style, tone, or output format. Community experience shows a Qwen2.5-3B fine-tuned on 47K examples (12K original + 35K augmented) achieved 8.2x performance improvement in a single epoch. Because LoRA modifies fewer parameters, it is more sensitive to data noise and inconsistency than full fine-tuning — consistent formatting and high per-example quality matter more than volume.

Data loading uses the HuggingFace `datasets` library with on-the-fly tokenization:

```python
from datasets import load_dataset
dataset = load_dataset("json", data_files="train.jsonl", split="train")

def format_fn(examples):
    texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in examples["messages"]]
    return {"text": texts}

dataset = dataset.map(format_fn, batched=True)
```

### Full fine-tuning: 10x more data, same format

Full fine-tuning updates every parameter, making it both more capable of deep behavioral change and more prone to catastrophic forgetting. **Typical dataset sizes range from 10K to 100K+ examples** — sometimes much more for domain adaptation. The data format and tokenization pipeline are identical to LoRA, but preprocessing parallelism becomes important: set `preprocessing_num_workers=16` or higher.

Quality requirements are paradoxically both more and less forgiving: full fine-tuning can absorb noisier data without overfitting due to the larger parameter budget, but low-quality data causes more lasting damage because every parameter is updated. The standard practice is to start with smaller high-quality datasets and gradually increase.

### Custom training harness: pre-tokenize everything

For pre-training or large-scale training from scratch, on-the-fly tokenization becomes a bottleneck. The standard approach is **offline pre-tokenization** into memory-mapped binary files. Megatron-LM's `preprocess_data.py` converts JSONL into `.bin` + `.idx` files; Qwen3-Coder uses a similar `binarize_data.py` producing `.mmap` files. A 2TB raw corpus can compress to ~25GB after pre-tokenization.

```python
# Memory-mapped dataset for pre-training (Qwen3-Coder pattern)
import numpy as np
class MMAPDataset(torch.utils.data.Dataset):
    def __init__(self, path, seq_length):
        self.seq_length = seq_length
        self.input_ids = np.memmap(f"{path}.input_ids.mmap", dtype=np.int32, mode='r',
                                    shape=(n_samples, seq_length))
    def __getitem__(self, idx):
        return {"input_ids": torch.from_numpy(self.input_ids[idx].astype(np.int64))}
```

Megatron-LM preprocessing command for Qwen:
```bash
python tools/preprocess_data.py \
    --input corpus.jsonl --output-prefix qwen_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen2.5-7B \
    --workers 48 --append-eod --dataset-impl mmap
```

| Aspect | LoRA | Full Fine-Tuning | Custom Harness |
|--------|------|-----------------|----------------|
| **Typical dataset size** | 1K–10K examples | 10K–100K+ examples | Billions to trillions of tokens |
| **Format** | JSONL (ChatML/Alpaca/ShareGPT) | Same as LoRA | Pre-tokenized binary (.bin/.idx/.mmap) |
| **Tokenization** | On-the-fly via HF datasets | On-the-fly, parallelized | Offline pre-tokenization |
| **Quality sensitivity** | Very high (fewer params to absorb noise) | Moderate | Moderate (massive scale averages out) |
| **Key tools** | HF `datasets`, `apply_chat_template` | Same + multiprocessing | Megatron preprocess, numpy memmap |

---

## Hardware and infrastructure scale by orders of magnitude

The memory equation for training a 7B parameter model is straightforward in principle but varies dramatically across paradigms due to what gets stored in GPU memory.

### LoRA: one consumer GPU is enough

LoRA freezes the base model and trains small rank-decomposition matrices, so only the adapter parameters, their optimizer states, and their gradients consume "training memory." The frozen model weights still occupy VRAM, but quantization (QLoRA) slashes this.

| Configuration | Qwen2.5-7B VRAM | Hardware |
|--------------|-----------------|----------|
| **QLoRA (4-bit NF4)** | **10–16 GB** | Single RTX 3090/4090 24GB ✅ |
| **LoRA (16-bit)** | 20–24 GB | Single RTX 4090 24GB (tight), A100 40GB ✅ |
| **LoRA (16-bit, all 7 targets)** | 24–35 GB | A100 40GB or 80GB ✅ |

With Unsloth optimizations, QLoRA fine-tuning of Qwen2.5-7B can run in as little as **~7 GB VRAM on a Tesla T4**. The key enablers are 4-bit NF4 quantization via bitsandbytes, gradient checkpointing, and batch size 1 with gradient accumulation.

**Essential frameworks:** PEFT (LoRA adapters), bitsandbytes (quantization), transformers (model/tokenizer), trl (SFTTrainer), accelerate (multi-GPU), flash-attn (Flash Attention 2).

QLoRA configuration for bitsandbytes:
```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use fp16 for QLoRA if bf16 unsupported
)
```

**Important constraint:** QLoRA only supports fp16 compute, not bf16. FSDP and ZeRO-3 are incompatible with QLoRA; use ZeRO-2 for multi-GPU QLoRA training.

### Full fine-tuning: multi-GPU territory

Full fine-tuning stores the complete model in trainable form plus Adam optimizer states (two fp32 copies of every parameter) plus gradients. The memory breakdown for Qwen2.5-7B (7.6B parameters):

- Model weights (bf16): **~15 GB**
- Optimizer states (fp32 momentum + variance): **~58 GB**
- Gradients (bf16/fp32): **~15–30 GB**
- Activations (varies by batch/sequence): **~5–20 GB**
- **Total: ~60–120 GB** depending on configuration

This means full fine-tuning **cannot fit on a single GPU** for 7B models. The minimum viable setups:

| Setup | ZeRO Stage | Feasibility |
|-------|-----------|-------------|
| 2× A100 80GB | ZeRO-2 | Works for short sequences (≤1024) |
| 4× A100 80GB | ZeRO-2/3 | Comfortable up to 4096 tokens |
| 4× A100 40GB | ZeRO-3 + gradient checkpointing | Possible but tight |
| 8× A100 40GB | ZeRO-3 | Full flexibility |
| 1× A100 80GB | — | ❌ Out of memory |

DeepSpeed ZeRO-3 is the standard distributed strategy, sharding model parameters, gradients, and optimizer states across GPUs. ZeRO-2 shards only optimizer states and gradients, reducing inter-GPU communication but requiring more per-GPU memory. **For multi-node setups, ZeRO-2 is preferred** because ZeRO-3's parameter all-gather operations are expensive over network interconnects.

**Estimated cloud costs:** 4× A100 80GB on budget providers (RunPod, Lambda) costs $5–7/hour. A typical 7B full fine-tuning run on 50K examples takes 10–24 hours, totaling **$50–170**.

### Custom harness: cluster-scale infrastructure

Pre-training a 7B model from scratch requires processing trillions of tokens. Qwen2.5 was trained on **18 trillion tokens**; Qwen3 on **36 trillion tokens**. The infrastructure requirements are:

- **Minimum for serious pre-training:** 8× A100 80GB (single node), yielding ~1 month for 1T tokens
- **Recommended:** 32–64× A100/H100 GPUs across multiple nodes
- **Full-scale (matching Qwen's training):** 256–1024+ H100s
- **Network:** InfiniBand (200–400 Gbps) strongly recommended for multi-node; 10GbE is viable but slow for clusters >8 GPUs
- **Communication backend:** NCCL (NVIDIA), with NVLink providing 600–900 GB/s intra-node bandwidth on H100s

The parallelism strategy for 7B typically combines **FSDP (data parallelism with sharding)** across all GPUs, without tensor parallelism (unnecessary at 7B scale). Larger models (70B+) add tensor parallelism within nodes and pipeline parallelism across nodes.

**Key frameworks:** Megatron-LM (NVIDIA's gold standard, 41–48% MFU on H100), TorchTitan (Meta's PyTorch-native platform, ICLR 2025), GPT-NeoX (EleutherAI, battle-tested in academia), NeMo (NVIDIA, higher-level wrapper around Megatron Core).

---

## Code architecture: from ten lines to ten thousand

The implementation complexity is the starkest difference between paradigms. LoRA fine-tuning can be accomplished in under 50 lines; full fine-tuning adds distributed configuration; a custom harness requires implementing every component from scratch.

### LoRA: PEFT + SFTTrainer in ~40 lines

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. Model + quantization
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=torch.bfloat16),
    device_map="auto", attn_implementation="flash_attention_2",
)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# 2. LoRA configuration
peft_config = LoraConfig(
    r=64, lora_alpha=16, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # ~0.5-2% trainable

# 3. Dataset
dataset = load_dataset("json", data_files="train.jsonl", split="train")
def fmt(examples):
    return {"text": [tokenizer.apply_chat_template(m, tokenize=False)
                     for m in examples["messages"]]}
dataset = dataset.map(fmt, batched=True)

# 4. Train
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset, peft_config=peft_config,
    args=SFTConfig(
        output_dir="./qwen-lora", per_device_train_batch_size=1,
        gradient_accumulation_steps=4, num_train_epochs=1,
        learning_rate=2e-4, optim="paged_adamw_32bit",
        bf16=True, gradient_checkpointing=True,
        lr_scheduler_type="cosine", warmup_steps=10, logging_steps=10,
    ),
)
trainer.train()
model.save_pretrained("./qwen-lora-adapter")
```

**Key LoRA hyperparameters for Qwen:**

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| **rank (r)** | 16–64 (start with 32) | Higher = more capacity, more memory |
| **alpha** | 2× rank | Sebastian Raschka's research confirms this sweet spot |
| **dropout** | 0.05 (0 for Unsloth) | Set 0 when using optimized kernels |
| **learning rate** | 1e-4 to 5e-4 | 2e-4 is the most common default |
| **target modules** | All 7 linear layers | Minimum: just `q_proj`, `v_proj` |
| **optimizer** | `paged_adamw_32bit` (QLoRA) | Offloads optimizer state pages to CPU |

To merge the adapter back into the base model for deployment:
```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "./qwen-lora-adapter")
merged = model.merge_and_unload()  # LoRA weights folded into base — zero inference overhead
merged.save_pretrained("./qwen-merged")
```

### Full fine-tuning: add DeepSpeed and distributed launch

The code structure is similar but replaces PEFT configuration with DeepSpeed ZeRO and requires multi-GPU launching.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.gradient_checkpointing_enable()  # Essential for memory

training_args = TrainingArguments(
    output_dir="./qwen-full-ft",
    per_device_train_batch_size=1,       # Memory-limited
    gradient_accumulation_steps=8,       # Effective batch = 8 × num_gpus
    num_train_epochs=2,
    learning_rate=1e-5,                  # 10-20x lower than LoRA
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    deepspeed="ds_z3_config.json",       # ZeRO-3 configuration file
    optim="adamw_torch_fused",
    max_grad_norm=1.0,
    save_steps=500, save_total_limit=3,
    logging_steps=10, report_to="wandb",
)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset,
                  data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
trainer.train()
```

The **DeepSpeed ZeRO-3 configuration** (saved as `ds_z3_config.json`):
```json
{
  "bf16": {"enabled": "auto"},
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

Launch with: `torchrun --nproc_per_node=4 train.py --deepspeed ds_z3_config.json`

**Key difference from LoRA:** The learning rate drops to **1e-5** (versus 2e-4 for LoRA) because updating all 7.6 billion parameters requires much smaller steps to avoid catastrophic forgetting. Weight decay increases to 0.1 for regularization.

### Custom harness: every component is manual

A ground-up training loop requires implementing gradient accumulation, mixed precision, learning rate scheduling, checkpointing, distributed communication, and evaluation — all without the HuggingFace Trainer abstraction.

```python
import torch, math, os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# === Distributed setup ===
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# === Model (can use HF model class, just not HF Trainer) ===
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                              torch_dtype=torch.bfloat16).to(local_rank)
model = DDP(model, device_ids=[local_rank])  # Or wrap with FSDP for 7B

# === Optimizer + scheduler ===
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                               weight_decay=0.1, fused=True)

def cosine_lr(step, warmup=2000, total=100000, max_lr=3e-4, min_lr=3e-5):
    if step < warmup: return max_lr * step / warmup
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay))

# === Training loop ===
grad_accum_steps = 8
for step, batch in enumerate(dataloader):
    input_ids = batch['input_ids'].to(local_rank)
    labels = batch['labels'].to(local_rank)

    # Update learning rate
    lr = cosine_lr(step // grad_accum_steps)
    for pg in optimizer.param_groups: pg['lr'] = lr

    # Forward + backward with mixed precision
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(input_ids).logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        ) / grad_accum_steps
    loss.backward()

    # Step every grad_accum_steps micro-batches
    if (step + 1) % grad_accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Checkpoint (save model, optimizer, RNG states)
    if (step + 1) % save_interval == 0:
        torch.save({
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'rng': torch.random.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(),
        }, f"checkpoint_{step}.pt")
```

For production pre-training, this raw loop is typically replaced by **Megatron-LM** or **TorchTitan**, which handle 3D/4D parallelism (tensor parallel + pipeline parallel + data parallel + context parallel), fused CUDA kernels, efficient indexed datasets, and FP8 training. Adapting Megatron-LM for Qwen requires configuring its TransformerConfig:

```python
from megatron.core.transformer.transformer_config import TransformerConfig
config = TransformerConfig(
    num_layers=28, hidden_size=3584, num_attention_heads=28,
    num_query_groups=4,          # GQA: 4 KV heads
    ffn_hidden_size=18944,       # SwiGLU intermediate
    hidden_act='swiglu',
    pipeline_dtype=torch.bfloat16,
)
```

---

## How the three paradigms compare on fundamental trade-offs

### Memory, gradients, and parameter updates differ at the core

The fundamental distinction is what gets updated and what stays frozen:

**LoRA** injects small trainable matrices (rank `r`) into each target layer. For Qwen2.5-7B with r=64 across all 7 linear layers, this adds roughly **160M trainable parameters** (~2% of total). The base model's 7.4B parameters remain frozen in VRAM (or quantized to 4-bit), and only the adapter parameters require optimizer states and gradient storage. The effective gradient computation backpropagates through the full model graph but only updates the low-rank matrices, making memory proportional to `2 × r × (d_in + d_out) × num_layers × num_targets`.

**Full fine-tuning** computes gradients for every parameter. Adam stores two fp32 state tensors (momentum and variance) per parameter, meaning 7.6B parameters require ~58GB just for optimizer states. This is why the total VRAM scales to ~4× the model's bf16 size (15GB weights + 58GB optimizer + 15GB gradients + activations).

**Custom pre-training** faces the same memory profile as full fine-tuning but at vastly greater data scale. The engineering challenge shifts from memory optimization to throughput: maximizing tokens-per-second-per-GPU through efficient data pipelines, communication overlap, and kernel fusion. Megatron-LM achieves **41–48% Model FLOPs Utilization (MFU)** on H100 clusters through these optimizations.

### When each approach makes sense

**Choose LoRA when** you need to adapt a pre-trained Qwen model to a specific task, style, or domain — and the adaptation is primarily behavioral rather than knowledge-heavy. LoRA excels at teaching a model to follow a particular output format, adopt a persona, or handle a specialized task taxonomy. It trains in hours on a single GPU, costs under $10 in compute, and the adapter can be merged for zero inference overhead or swapped at runtime for multi-task serving. The limitation is that LoRA cannot fundamentally restructure the model's knowledge or capabilities — it bends behavior, not understanding.

**Choose full fine-tuning when** you need deep domain adaptation, significant capability changes, or must modify the model's core behavior across a wide range of tasks. Full fine-tuning is appropriate for creating a domain-specific foundation model (medical, legal, financial), for continued pre-training on proprietary corpora, or when LoRA quality proves insufficient. The cost is 10–50x higher in compute, requires multi-GPU infrastructure, and risks catastrophic forgetting if data is poorly curated.

**Choose a custom training harness when** you are pre-training from scratch, continuing pre-training at very large scale, need custom parallelism strategies, are training on proprietary hardware or infrastructure, or require optimizations that existing frameworks cannot provide. This is the domain of AI labs (Alibaba for Qwen, Meta for Llama), research institutions (EleutherAI), and companies with dedicated ML infrastructure teams. The engineering overhead is massive — you must implement or integrate gradient accumulation, mixed precision, distributed communication, checkpointing, data loading, evaluation, and monitoring — but the payoff is complete control and maximum throughput.

### Cost and time comparison for Qwen2.5-7B

| Metric | LoRA (QLoRA) | Full Fine-Tuning | Custom Pre-Training |
|--------|-------------|-----------------|-------------------|
| **Hardware** | 1× RTX 4090 24GB | 4× A100 80GB | 8–64× H100 80GB |
| **VRAM used** | ~12 GB | ~60–80 GB total | ~60–80 GB per GPU |
| **Training time (10K examples)** | 1–3 hours | 4–12 hours | N/A (pre-training) |
| **Training time (1T tokens)** | N/A | N/A | ~1 month (8× A100) |
| **Cloud cost** | $2–10 | $50–170 | $10K–100K+ |
| **Trainable parameters** | ~160M (2%) | 7.6B (100%) | 7.6B (100%) |
| **Engineering complexity** | Low | Medium | Very high |
| **Risk of catastrophic forgetting** | Very low | Moderate | N/A (starting fresh) |

---

## Framework ecosystem for Qwen training

Several frameworks provide battle-tested implementations for each paradigm, all with native Qwen support.

**LLaMA-Factory** is the most popular general-purpose framework, officially recommended by the Qwen team. It supports LoRA, QLoRA, full fine-tuning, and RL methods (DPO, PPO, GRPO) through YAML configuration. Use `template: qwen` for Qwen2/2.5 or `template: qwen3` for Qwen3. A typical LoRA config:

```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
stage: sft
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_target: all            # Auto-detects all 7 linear layers
template: qwen
dataset: your_dataset
cutoff_len: 4096
learning_rate: 1.0e-4
bf16: true
deepspeed: examples/deepspeed/ds_z2_config.json
```

**Unsloth** claims 2x faster training and 70% less VRAM compared to standard HuggingFace + Flash Attention 2, achieved through custom Triton kernels and optimized gradient checkpointing. It excels on single-GPU setups: Qwen2.5-7B in 4-bit runs in ~7GB VRAM. For Qwen3 reasoning fine-tuning, Unsloth recommends a **75/25 ratio of reasoning to non-reasoning data** to preserve thinking capabilities.

**Axolotl** provides advanced features like sample packing (multiple short sequences per training example for throughput), fused LoRA kernels for SwiGLU and QKV attention, and native DeepSpeed/FSDP integration. Its `lora_target_linear: true` flag automatically targets all linear layers.

**MS-SWIFT** (ModelScope) is Alibaba's own framework with first-class Qwen support, including Megatron-SWIFT integration for MoE training with expert parallelism (~10x acceleration). It supports training on NVIDIA, Ascend, AMD, and Intel hardware.

For custom pre-training, **Megatron-LM** remains the gold standard for throughput at scale, while **TorchTitan** (Meta, ICLR 2025) offers a more modern PyTorch-native alternative with FSDP2, torch.compile integration, and Float8 training support.

---

## Scaling the Qwen architecture down: from 1B to 4K parameters

The Qwen2/Qwen3 architecture is a standard decoder-only causal transformer, so its components scale down mechanically — but practical limits emerge quickly, especially around vocabulary size.

### The vocabulary wall

The single biggest obstacle to tiny Qwen-style models is the **tokenizer vocabulary size**. Qwen's 151,936–152,064 token vocabulary creates an embedding matrix of `vocab_size × hidden_size` parameters, plus a matching `lm_head` output projection (Qwen does not tie these by default). This means:

| Hidden size | Embedding params (untied) | Embedding params (tied) | Realistic total model |
|-------------|--------------------------|------------------------|-----------------------|
| 4096 | ~1.24B | ~622M | 7–8B (Qwen's actual 7B/8B) |
| 1024 | ~311M | ~156M | ~600M–1B |
| 512 | ~156M | ~78M | ~150–300M |
| 128 | ~39M | ~19.5M | ~25–50M |
| 32 | ~9.7M | ~4.9M | ~6–12M |
| 4 | ~1.2M | ~608K | ~700K–1.5M |

With Qwen's native vocabulary, **4K total parameters is physically impossible** — the embedding table alone is 608K params even at hidden_size=4 with weight tying. The absolute floor with Qwen's tokenizer is around **1–2M parameters**. To reach 4K–100K parameter models, you need a **custom, much smaller vocabulary** (256–2,000 tokens), at which point you're using the same transformer architecture but not Qwen's tokenizer.

### What scales down cleanly

**RMSNorm** scales perfectly — it's just a learned per-channel scale vector. Works identically at hidden_size=32 as at hidden_size=4096.

**RoPE (Rotary Position Embeddings)** scales well. It operates on dimension pairs in the head dimension, so head_dim must be even (which it always is). Works at head_dim=8 or even head_dim=4, though very small head dimensions reduce positional encoding expressiveness.

**SwiGLU MLP** scales fine. The `gate_proj`/`up_proj` → multiply → `down_proj` structure works at any size, with the standard intermediate ≈ 8/3 × hidden ratio. At hidden_size=64, intermediate would be ~170.

**Causal attention** scales fine mechanically. Flash Attention 2 requires head_dim ≤ 256, but standard scaled dot-product attention works at any size.

### What breaks or becomes pointless at small scale

**Grouped Query Attention (GQA)** — Qwen's efficiency optimization where multiple query heads share fewer KV heads (e.g., 28Q / 4KV in Qwen2.5-7B). At small scale, GQA becomes either pointless or impossible. With only 2–4 total attention heads, it collapses to either standard multi-head attention or multi-query attention, and the memory savings that motivate GQA at 7B are negligible at 10M. Additionally, `num_query_heads` must be divisible by `num_kv_heads`, constraining head count options. **Below ~100M params, just use standard MHA** (num_kv_heads = num_query_heads).

**QK LayerNorm (Qwen3)** adds per-layer overhead that's proportionally larger in tiny models. Not harmful, but unnecessary — training instability from exploding gradients is not a concern when your model is too small to exhibit it.

**The untied lm_head** — at 7B the lm_head is ~2% of total params. At 5M total params with a 152K vocabulary, the lm_head could be **60%+ of the model**. Always tie weights at small scale: `config.tie_word_embeddings = True`.

### Practical configurations at each scale

```python
# ~1B params (comparable to Qwen2.5-0.5B, which ships officially)
config_1b = {
    "hidden_size": 1024, "num_hidden_layers": 24,
    "num_attention_heads": 16, "num_key_value_heads": 4,  # GQA still useful
    "intermediate_size": 2816, "vocab_size": 151936,
    "tie_word_embeddings": False,  # Can afford the separate lm_head
}

# ~100M params
config_100m = {
    "hidden_size": 512, "num_hidden_layers": 12,
    "num_attention_heads": 8, "num_key_value_heads": 8,  # Standard MHA
    "intermediate_size": 1365, "vocab_size": 151936,
    "tie_word_embeddings": True,  # Should tie at this scale
}

# ~10M params
config_10m = {
    "hidden_size": 128, "num_hidden_layers": 6,
    "num_attention_heads": 4, "num_key_value_heads": 4,
    "intermediate_size": 341, "vocab_size": 151936,
    "tie_word_embeddings": True,  # Must tie
}

# ~1M params (requires smaller vocabulary)
config_1m = {
    "hidden_size": 64, "num_hidden_layers": 4,
    "num_attention_heads": 2, "num_key_value_heads": 2,
    "intermediate_size": 170, "vocab_size": 4096,  # Custom small tokenizer
    "tie_word_embeddings": True,
}

# ~4K-50K params (toy/research only, custom tiny vocabulary)
config_tiny = {
    "hidden_size": 16, "num_hidden_layers": 2,
    "num_attention_heads": 2, "num_key_value_heads": 2,
    "intermediate_size": 42, "vocab_size": 64,  # Byte-level or character-level
    "tie_word_embeddings": True,
}
```

### Instantiating a small Qwen-architecture model

The Qwen2 model class from HuggingFace accepts any valid configuration, so you can instantiate arbitrary sizes:

```python
from transformers import Qwen2Config, Qwen2ForCausalLM

config = Qwen2Config(
    hidden_size=512, num_hidden_layers=12,
    num_attention_heads=8, num_key_value_heads=8,
    intermediate_size=1365, vocab_size=151936,
    max_position_embeddings=4096, tie_word_embeddings=True,
    rms_norm_eps=1e-6, use_sliding_window=False,
)
model = Qwen2ForCausalLM(config)
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
# Roughly ~100M params
```

This gives the exact same architecture (RoPE, SwiGLU, RMSNorm, causal attention) as Qwen2.5-7B, just smaller. It can be trained with any of the three paradigms described above.

### Where architecture choice stops mattering

Below about **100M parameters**, the specific architectural choices (GQA vs MHA, SwiGLU vs GELU, RMSNorm vs LayerNorm) have marginal impact compared to data quality and training methodology. The Qwen architecture will work, but so will a vanilla GPT-2 style transformer. The reason to stick with Qwen's architecture at small scale is usually **research consistency** (ablation studies, scaling law experiments) or codebase uniformity across model sizes.

Below **1M parameters** with a full-sized vocabulary, most of the parameter budget goes to the embedding table rather than the transformer layers. At that point, either shrink the vocabulary significantly or consider whether a transformer is even the right architecture for the task.

---

## Conclusion

The three paradigms form a clear spectrum of investment versus capability. LoRA is the right default for most practitioners — it turns a $2 compute budget and a few thousand examples into a task-specialized model within hours. Full fine-tuning unlocks deeper adaptation at **10–50x the cost**, justified when domain transformation or capability expansion is the goal. Building from scratch is reserved for organizations that need to control the entire training pipeline, whether for pre-training new models, implementing novel research, or optimizing for proprietary infrastructure.

For Qwen specifically, the ecosystem is mature: all three paradigms are well-supported through official tooling (ms-swift), community frameworks (LLaMA-Factory, Axolotl, Unsloth), and low-level libraries (Megatron-LM, TorchTitan). The key Qwen-specific details that trip up practitioners — the pad_token bug, untrained chat tokens in base models, QLoRA's fp16-only constraint, ZeRO-3/QLoRA incompatibility, and the attention bias difference between Qwen2.5 and Qwen3 — are well-documented but easy to miss. Start with LoRA using the code patterns above, validate your results, and escalate to full fine-tuning only when the parameter-efficient approach hits its ceiling.