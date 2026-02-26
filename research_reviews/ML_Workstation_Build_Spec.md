# ML Workstation — Endgame Build Spec

**Purpose:** Self-contained ML workstation for the clean-room agent's self-improvement loop. Dual RTX 5090 for: LoRA/QLoRA/DPO training, full fine-tuning of 3-4B base models, training mini-models from scratch (up to ~3.5B), plan validation harness inference, and all agent inference. Target: fully air-gapped post-migration.

**OS:** Ubuntu Server 24.04 LTS + i3 window manager (X11)

**Shop:** Plonter (plonter.co.il), KSP (ksp.co.il)

---

## Components

| Component | Model | Price (₪) | Notes |
|---|---|---|---|
| **CPU** | AMD Ryzen 9 9900X 12C/24T (Box, no fan) | 1,768 | Harness runs tests (CPU-bound) |
| **Motherboard** | ASUS ProArt X870E-Creator WiFi (AM5, ATX) | 2,603 | 2× PCIe 5.0 x16 (x8/x8 bifurcation) |
| **RAM** | 2× Kingston Fury Beast 32GB DDR5-6400 CL32 | 4,772 | 64GB total |
| **GPU 1** | RTX 5090 32GB (cheapest compact AIB) | ~13,000 | See GPU selection notes |
| **GPU 2** | RTX 5090 32GB (cheapest compact AIB) | ~13,000 | Same model as GPU 1 |
| **Storage** | Samsung 990 Pro 2TB NVMe (MZ-V9P2T0BW) | 1,822 | OS + models + active repos |
| **Storage 2** | Samsung 990 Pro 2TB NVMe | ~1,000 | Datasets, raw DB, checkpoints |
| **Case** | Antec Performance 1 FT White (Full Tower, E-ATX, Mesh) | 704 | 400mm GPU clearance |
| **PSU** | be quiet! Dark Power Pro 13 1600W Titanium | ~2,500 | 2× 12V-2x6, ATX 3.1 |
| **CPU Cooler** | Thermalright Peerless Assassin 120 Digital ARGB White | 390 | |
| **Peripherals** | Logitech MK120 Wired USB Keyboard + Mouse | 63 | |
| | | | |
| **Total** | | **~41,622** | |

**Pricing note (Feb 2026):** RTX 5090 supply is severely constrained globally. Israeli market prices for cheapest AIB models (MSI Ventus 3X OC, Gigabyte Windforce OC) are ₪12,800–13,000 per card. Premium models run ₪14,000–16,800. Prices will likely decrease as supply stabilizes. PSU price is estimated from US pricing ($460) with typical Israeli markup.

---

## Power Budget

| Component | TDP (W) | ML Steady-State (W) |
|---|---|---|
| GPU 1 (RTX 5090) | 575 | ~490 (undervolted) |
| GPU 2 (RTX 5090) | 575 | ~490 (undervolted) |
| CPU (9900X PBO) | 170 | ~120 |
| System (RAM, NVMe, fans) | ~80 | ~80 |
| **Total** | **~1,400** | **~1,180** |

1600W PSU at 80+ Titanium (~94% efficiency at full load) delivers 1600W output. Peak system draw of ~1,400W is 87.5% — within safe operating range. ML training is sustained steady-state load (not transient spikes), and undervolting both GPUs by 10-15% (99% performance retention per [Tom's Hardware benchmarks](https://www.tomshardware.com/pc-components/gpus/what-sort-of-power-supply-do-you-actually-need-for-an-rtx-5090)) brings steady-state to ~1,180W (74% PSU load — comfortable).

---

## GPU Selection

RTX 5090 cards vary significantly in physical size. For dual-GPU, thickness matters — the ProArt X870E-Creator's two PCIe 5.0 x16 slots are spaced ~3 slots apart.

| Model | Length | Thickness | Slots | Dual-GPU Fit |
|---|---|---|---|---|
| NVIDIA FE | 304mm | 48mm | 2-slot | Ideal (~1 slot gap) but unavailable |
| ASUS ProArt | ~310mm | ~55mm | 2.5-slot | Good (~0.5 slot gap) |
| MSI Ventus 3X OC | 325mm | 67mm | 3-slot | Tight (zero gap) — thermal risk |
| Gigabyte Windforce OC | 342mm | 67mm | 3-slot | Tight (zero gap) — thermal risk |

**Recommendation:** Prioritize 2-slot or 2.5-slot models for dual-GPU. If only 3-slot models are available, apply aggressive undervolt and set custom fan curves. Both GPUs will run sustained ML workloads — airflow between cards is critical.

**Israeli market availability (Feb 2026):**
- Gigabyte Windforce OC 32G: ~₪12,800 (KSP, TMS)
- MSI Ventus 3X OC: ~₪13,000 (Plonter)
- MSI Gaming Trio OC: ~₪13,000–14,000
- ASUS TUF Gaming: ~₪15,850 (Ivory)
- ASUS ROG Astral: ~₪16,790 (Ivory)

---

## ML Training Capabilities

| Operation | VRAM Used | Duration (10K examples) |
|---|---|---|
| Inference (3B + 4B, fp16, concurrent) | ~14 GB | — |
| QLoRA on 4B model | ~8–12 GB | 1–3 hours |
| LoRA (fp16) on 4B model | ~12–18 GB | 2–6 hours |
| DPO on 4B model (QLoRA) | ~10–14 GB | 2–4 hours |
| Full fine-tune 3B (ZeRO-3, 2 GPUs) | ~20–25 GB/GPU | 4–12 hours |
| Full fine-tune 4B (ZeRO-3, 2 GPUs) | ~25–33 GB/GPU | 6–16 hours |
| Mini-model 100M from scratch | ~2–4 GB | Minutes–hours |
| Mini-model 1B from scratch | ~15–20 GB | Hours–days |
| Mini-model 3.5B from scratch (ZeRO-3) | ~25–30 GB/GPU | Days |

**Max model size trainable from scratch:** ~3–3.5B parameters (ZeRO-3 across both GPUs).

**Max model size for QLoRA adaptation:** ~30B+ parameters (single GPU, 4-bit).

---

## BIOS Configuration (First Boot)

- Enable AMD EXPO for DDR5-6400 (or manually set 6000MHz if FCLK 2133MHz is unstable)
- Disable CSM, enable Above 4G Decoding, enable Resizable BAR
- Enable IOMMU
- Enable PBO with Curve Optimizer -10 to -15 all cores
- Set PCIe slots to Gen5 (not Auto) for consistent x8/x8 bifurcation

## Software Setup

1. Install Ubuntu Server 24.04 LTS + HWE kernel
2. Install i3 + Xorg + lightdm
3. Install NVIDIA driver 590-open (required for Blackwell RTX 5090)
4. CUDA 12.8+ (bundled with PyTorch, no system-wide install needed)
5. Install uv for Python package management
6. PyTorch 2.9+ with CUDA 12.8 index
7. Key ML packages: transformers, peft, trl, bitsandbytes, accelerate, datasets, unsloth, deepspeed
8. DeepSpeed ZeRO-3 config for multi-GPU full fine-tuning

## System Tuning

- Kernel params: `amd_pstate=active pcie_aspm=off iommu=pt`
- `vm.swappiness=10`, `vm.max_map_count=1048576`
- NVMe scheduler: `none`
- Mount with `noatime,nodiratime`
- `nvidia-smi -pm 1` (persistence mode) for both GPUs
- Undervolt both GPUs: `nvidia-smi -pl 490` (85% TDP, ~99% performance)
- Lock GPU clocks for consistent training throughput
- `CUDA_VISIBLE_DEVICES=0,1` for training; `=0` or `=1` for single-GPU tasks
- `NCCL_P2P_DISABLE=1` if PCIe P2P causes issues (x8/x8 bifurcation)

## Verification Script

```python
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2, f"Expected 2 GPUs, found {torch.cuda.device_count()}"
for i in range(2):
    assert torch.cuda.get_device_capability(i) == (12, 0), f"GPU {i}: not Blackwell"
    props = torch.cuda.get_device_properties(i)
    vram_gb = props.total_mem / 1e9
    assert vram_gb > 30, f"GPU {i}: only {vram_gb:.1f} GB VRAM"
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}, {vram_gb:.1f} GB")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"Total VRAM: {sum(torch.cuda.get_device_properties(i).total_mem for i in range(2)) / 1e9:.1f} GB")

# Verify inter-GPU communication
import torch.distributed as dist
# For quick NCCL check: torchrun --nproc_per_node=2 verify.py
```

---

## Remaining Upgrade Path

| Phase | Action | Est. Cost |
|---|---|---|
| **Optional** | CPU upgrade → Ryzen 9 9950X 16C/32T (drop-in AM5) | ~₪2,500–3,000 |
| **Optional** | RAM → 128GB (swap to 2× 64GB DDR5-6400 sticks) | ~₪5,000–6,000 |

The dual RTX 5090 configuration is the endgame GPU setup. No further GPU upgrades planned.

## Key Design Decisions

**Motherboard (ProArt X870E-Creator):** Two CPU-direct PCIe 5.0 x16 slots with x8/x8 bifurcation — the only AM5 board with real dual-GPU capability. PCIe 5.0 x8 provides ~32 GB/s per direction, sufficient for DeepSpeed ZeRO-3 inter-GPU communication (much less bandwidth-hungry than NVLink-dependent workloads).

**GPU (2× RTX 5090 32GB):** 64GB total VRAM enables full fine-tuning of 3-4B base models with ZeRO-3 (no more outsourcing), training mini-models up to ~3.5B from scratch, and all LoRA/QLoRA/DPO training. The 1.8 TB/s memory bandwidth per card accelerates both inference (bandwidth-bound for autoregressive generation) and training throughput. Blackwell tensor cores with FP4/FP8 support for future quantized training.

**PSU (1600W Titanium):** Sized for dual RTX 5090 (2× 575W TDP) + 9900X (170W) + system (80W) = ~1,400W peak. Must have 2× 12V-2x6 connectors for the two GPUs. The be quiet! Dark Power Pro 13 1600W is ATX 3.1 compliant with dual 12VHPWR cables included. 80+ Titanium efficiency means less waste heat at sustained ML loads.

**RAM (DDR5-6400 CL32, 64GB):** Sufficient for dataset loading, multi-process harness operations, and model training. 128GB upgrade path exists if needed for larger dataset preprocessing.

**Dual NVMe:** Separate drives for OS+models (990 Pro #1) and data+checkpoints (990 Pro #2). Prevents I/O contention during training when checkpoints are written to disk while data loading continues.

**Case (Antec Performance 1 FT):** Full tower with 400mm GPU clearance and high-airflow mesh front. Critical for thermal management with dual 575W GPUs. Front-to-back airflow path with 4 included fans.

**Dual-GPU thermal strategy:** Undervolt both GPUs to ~490W (85% TDP) for ~99% performance. This drops total system draw from ~1,400W to ~1,180W, reducing heat output by ~220W. Set aggressive fan curves prioritizing GPU thermals. If using 3-slot cards (zero inter-card gap), monitor GPU 1 temperatures closely — it receives pre-heated air from GPU 2's exhaust.

---

## Cost Comparison: Current Spec vs Endgame

| Component | Current Spec | Endgame | Delta |
|---|---|---|---|
| GPU | 1× RTX 5060 Ti 16GB (₪2,225) | 2× RTX 5090 32GB (~₪26,000) | +₪23,775 |
| PSU | Antec HCG1200 1200W (₪1,072) | be quiet! DPP13 1600W (~₪2,500) | +₪1,428 |
| Storage | 1× 2TB NVMe (₪1,822) | 2× 2TB NVMe (~₪2,822) | +₪1,000 |
| Everything else | ₪10,300 | ₪10,300 | — |
| **Total** | **₪15,419** | **~₪41,622** | **+₪26,203** |
