# C-Based ML Training Infrastructure — Source Survey

**Date:** 2026-02-27

## Critical References (start here)

### 1. llm.c — Karpathy
- **URL:** https://github.com/karpathy/llm.c
- **What:** Complete GPT-2/GPT-3 pretraining in C/CUDA. ~5,000 lines. Mixed precision (BF16/FP16), multi-GPU via NCCL, AdamW, gradient accumulation, checkpointing.
- **Performance:** ~7% faster than PyTorch Nightly. GPT-2 124M reproduced in 90 min / $20 on 8xH100.
- **Status:** Active, 29K+ stars, MIT license.
- **Gaps:** GPT-2 only (no MoE, no linear attention), data parallelism only (no tensor/pipeline parallelism), no LoRA.

### 2. LibNC — Fabrice Bellard
- **URL:** https://bellard.org/libnc/
- **What:** C tensor library with full autograd, ADAM optimizer, CUDA support (Ampere, hardware BF16). Dynamic computation graph. No external dependencies. Fully deterministic. Used for GPT-2 training (GPT2TC).
- **Why it matters:** Closest thing to "PyTorch in C." Proven in production by the creator of FFmpeg and QEMU.

### 3. llama.cpp LoRA fine-tuning
- **URL:** https://github.com/ggml-org/llama.cpp (finetune-lora.cpp)
- **What:** First LoRA training in C/C++. Masked-loss, ADAM optimizer, backward kernels for CUDA/Metal/Vulkan. Works with quantized GGUF models.
- **Why it matters:** Proves LoRA training is feasible in C. Study the backward pass implementation.

### 4. GGML
- **URL:** https://github.com/ggml-org/ggml
- **What:** C tensor library powering llama.cpp. 16-bit float, integer quantization (4/5/8-bit), autograd (WIP), ADAM/L-BFGS optimizers. Multi-backend: ARM NEON, AVX/AVX2, CUDA, Metal, Vulkan.
- **Status:** Very active. Training support still maturing.

### 5. ThunderKittens — Stanford Hazy Research
- **URL:** https://github.com/HazyResearch/ThunderKittens
- **What:** CUDA-embedded DSL with tile primitives. Matches cuBLAS for GEMM, outperforms FlashAttention-3 backward by 10-40%. Header-only. Supports Hopper, Blackwell, FP8.
- **Why it matters:** Best framework for writing custom training kernels without raw CUDA pain.

## CUDA Kernel References

| Project | URL | Focus |
|---------|-----|-------|
| FlashAttention (Dao-AILab) | https://github.com/Dao-AILab/flash-attention | Efficient attention (Hopper/Blackwell) |
| FlashMLA (DeepSeek) | https://github.com/deepseek-ai/FlashMLA | Multi-head Latent Attention kernels |
| CUTLASS (NVIDIA) | https://github.com/NVIDIA/cutlass | Foundation GEMM/conv library |
| Transformer Engine (NVIDIA) | https://github.com/NVIDIA/TransformerEngine | FP8/FP4 training, framework-agnostic C++ API |
| FlashInfer | https://github.com/flashinfer-ai/flashinfer | Paged/sparse attention, C++ header-only API |
| LeetCUDA | https://github.com/xlite-dev/LeetCUDA | 200+ educational kernels, progressive difficulty |
| SGEMM_CUDA (Boehm) | https://github.com/siboehm/SGEMM_CUDA | Step-by-step GEMM optimization to 95% cuBLAS |

## Educational / Smaller References

| Project | URL | Focus |
|---------|-----|-------|
| mla.c | https://github.com/thomaschlt/mla.c | Multi-head Latent Attention in C (DeepSeek-V3) |
| llama2.c | https://github.com/karpathy/llama2.c | Single-file C inference (~700 lines) |
| yalm | https://github.com/andrewkchan/yalm | C++/CUDA inference, zero libraries, excellent blog |
| cmicrograd | https://github.com/MilanSuk/cmicrograd | Autograd engine in C |
| tensor.h | https://github.com/apoorvnandan/tensor.h | Minimal C tensor library |
| gpu.cpp | https://github.com/AnswerDotAI/gpu.cpp | Portable GPU compute via WebGPU |

## Key Blog Posts and Papers

| Title | URL |
|-------|-----|
| Reproducing GPT-2 124M in llm.c ($20, 90 min) | https://github.com/karpathy/llm.c/discussions/481 |
| Reproducing GPT-2 1.6B in llm.c ($672, 24h) | https://github.com/karpathy/llm.c/discussions/677 |
| llm.c State of the Union | https://github.com/karpathy/llm.c/discussions/344 |
| CUDA GEMM optimization (Boehm) | https://siboehm.com/articles/22/CUDA-MMM |
| Fast LLM Inference From Scratch (yalm) | https://andrewkchan.dev/posts/yalm.html |
| FlashAttention line-by-line | https://www.stephendiehl.com/posts/flash_attention/ |
| FlashAttention from scratch | https://lubits.ch/flash/Part-1 |
| Mixed Precision Training (NVIDIA, ICLR 2018) | https://arxiv.org/abs/1710.03740 |
| ThunderKittens paper | https://arxiv.org/html/2410.20399v1 |
| ThunderKittens 2.0 (Blackwell, FP8, multi-GPU) | https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2 |
| KernelBench: Can LLMs Write GPU Kernels? | https://arxiv.org/html/2502.10517v1 |

## What Already Exists in C/CUDA

- Full pretraining loop (llm.c)
- Mixed precision BF16/FP16 training (llm.c)
- Distributed data parallel via NCCL (llm.c)
- AdamW with LR scheduling (llm.c)
- LoRA fine-tuning (llama.cpp)
- Tensor library with autograd (GGML, LibNC)
- Flash attention kernels (cuDNN, FlashAttention, ThunderKittens)
- Data tokenization/loading (llm.c)
- Model checkpointing (llm.c)
- Quantized inference (llama.cpp)

## Gaps — Would Need to Be Built

- **MoE training in C** — no known implementation
- **Tensor/pipeline parallelism in C** — only in Python frameworks (Megatron-LM)
- **DPO/RLHF training loops in C** — all implementations Python
- **Architecture-agnostic C training framework** — llm.c is GPT-2 specific
- **Teacher-student distillation in C** — all implementations Python
- **Training data curation pipeline in C** — filtering, dedup, quality scoring
- **LoRA on custom architectures in C** — llama.cpp's LoRA is tied to GGUF format
- **DeltaNet / linear attention training kernels** — novel architecture, no C implementation
