# ML Fine-Tuning Workstation — Final Build Spec (Plonter)

**Purpose:** Fine-tuning small LLMs (up to 4B parameters) via LoRA/QLoRA, with full infrastructure for future dual-GPU upgrade.

**OS:** Ubuntu Server 24.04 LTS + i3 window manager (X11)

**Shop:** Plonter (plonter.co.il)

---

## Components

| Component | Model | Price (₪) |
|---|---|---|
| **CPU** | AMD Ryzen 9 9900X 12C/24T (Box, no fan) | 1,768 |
| **Motherboard** | ASUS ProArt X870E-Creator WiFi (AM5, ATX) | 2,603 |
| **RAM** | 2× Kingston Fury Beast 32GB DDR5-6400 CL32 (KF564C32BWEA-32) | 4,772 |
| **GPU** | Gigabyte RTX 5060 Ti 16GB Windforce 2 (GV-N506TWF2-16GD) | 2,225 |
| **Storage** | Samsung 990 Pro 2TB NVMe (MZ-V9P2T0BW) | 1,822 |
| **Case** | Antec Performance 1 FT White (Full Tower, E-ATX, Mesh) | 704 |
| **PSU** | Antec HCG1200 Pro 1200W Platinum (ATX 3.1) | 1,072 |
| **CPU Cooler** | Thermalright Peerless Assassin 120 Digital ARGB White | 390 |
| **Peripherals** | Logitech MK120 Wired USB Keyboard + Mouse | 63 |
| | | |
| **Total** | | **15,419** |

---

## BIOS Configuration (First Boot)

- Enable AMD EXPO for DDR5-6400 (or manually set 6000MHz if FCLK 2133MHz is unstable)
- Disable CSM, enable Above 4G Decoding, enable Resizable BAR
- Enable IOMMU
- Enable PBO with Curve Optimizer -10 to -15 all cores

## Software Setup

1. Install Ubuntu Server 24.04 LTS + HWE kernel
2. Install i3 + Xorg + lightdm
3. Install NVIDIA driver 590-open (required for Blackwell/RTX 5060 Ti)
4. CUDA 12.8+ (bundled with PyTorch, no system-wide install needed)
5. Install uv for Python package management
6. PyTorch 2.9+ with CUDA 12.8 index
7. Key ML packages: transformers, peft, trl, bitsandbytes, accelerate, datasets, unsloth

## System Tuning

- Kernel params: `amd_pstate=active pcie_aspm=off iommu=pt`
- `vm.swappiness=10`, `vm.max_map_count=1048576`
- NVMe scheduler: `none`
- Mount with `noatime,nodiratime`
- `nvidia-smi -pm 1` (persistence mode)
- Lock GPU clocks for consistent training throughput

## Verification Script

```python
import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 0)  # Blackwell
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

---

## Upgrade Path

| Phase | Action | Est. Cost |
|---|---|---|
| **Near-term** | Add second 2TB NVMe for dataset storage | ~₪600–700 |
| **Medium-term** | Swap GPU → RTX 5090 or equivalent (Slot 1) | TBD |
| **Medium-term** | Move RTX 5060 Ti to Slot 2 for display/inference | Free (reuse) |
| **Optional** | CPU upgrade → Ryzen 9 9950X (drop-in AM5) | ~₪2,500–3,000 |

## Key Design Decisions

**Motherboard (ProArt X870E-Creator):** Two CPU-direct PCIe 5.0 x16 slots with x8/x8 bifurcation for real dual-GPU capability. The original X670E-PRO's second slot was only PCIe 4.0 x4 through chipset — unusable for dual-GPU training.

**GPU (RTX 5060 Ti 16GB):** Blackwell architecture, 16GB GDDR7, better memory bandwidth and newer tensor cores than the originally planned 4060 Ti 16GB — at essentially the same price. Requires driver ≥590 and CUDA 12.8+.

**RAM (DDR5-6400 CL32):** Higher-binned than the originally planned DDR5-6000 CL36. Run at 6400MHz with EXPO if FCLK 2133MHz is stable; otherwise set 6000MHz manually for guaranteed 1:1 FCLK ratio.

**PSU (1200W Platinum):** Sized for future dual-GPU (RTX 5090 at 575W TDP + 5060 Ti + 9900X). Close to limits with full dual-GPU load but adequate.

**Dual-GPU strategy:** Primary GPU (future 5090) in Slot 1 for training. Secondary (5060 Ti) in Slot 2 for display output, keeping primary GPU's full VRAM free. `CUDA_VISIBLE_DEVICES` for isolation.
