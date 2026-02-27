# Building the optimal ML fine-tuning workstation on Linux

**Ubuntu Server 24.04 LTS with i3 window manager is the strongest configuration for this hardware**, combining bulletproof NVIDIA/CUDA support with minimal resource overhead. This pairing delivers the widest ML community support, longest maintenance window (5 years LTS), and near-zero VRAM waste — critical when every megabyte counts for fine-tuning LLMs on 16GB of GPU memory. One finding demands immediate attention: the ASUS Prime X670E-PRO WiFi's second PCIe slot runs at only **PCIe 4.0 x4 through the chipset**, making true dual-GPU training impractical on this motherboard. The rest of this report covers every dimension of the build in depth, with a concrete post-install guide at the end.

---

## Distribution ranking: why Ubuntu LTS wins for ML workstations

The distribution choice for an ML workstation reduces to a single question: how much friction exists between you and a working CUDA stack? Ubuntu 24.04 LTS answers this better than any alternative.

**Ubuntu Server 24.04 LTS** sits at the top because NVIDIA treats it as their primary Linux target. The CUDA Installation Guide officially supports it, `ubuntu-drivers` auto-detects the correct driver, and the `cuda-keyring` package provides clean access to NVIDIA's apt repository. The HWE kernel (currently **6.14** in 24.04.3, with 6.17 coming in 24.04.4) delivers full Zen 5/AM5/X670E support. Every major ML tutorial, Docker image, and PyTorch guide assumes Ubuntu. When something breaks at 2 AM during a training run, Ubuntu's community will have an answer.

**Fedora Server 42** earns second place for users who value fresh toolchains. It ships kernel **6.14**, GCC 15, and Python 3.13+ out of the box. NVIDIA support through RPM Fusion's `akmod-nvidia` works well, and NVIDIA's own repo offers precompiled driver streams with a DNF plugin that **blocks kernel upgrades when no matching driver exists** — an elegant safety mechanism. The downside is a ~13-month support window per release and occasional GCC/NVCC version mismatches requiring compatibility packages.

**Arch Linux** appeals to power users who want the latest everything on a single rolling system. The `nvidia-open` package in official repos is well-maintained, and the ArchWiki's NVIDIA and CUDA documentation is among the best anywhere. The risk is real, though: kernel 6.19 broke NVIDIA 580.x builds in early 2026, and rolling-release users who updated before the fix had non-functional GPUs. Arch demands active maintenance and a btrfs+snapper safety net.

**Pop!_OS 24.04** deserves mention for its NVIDIA ISO that delivers working proprietary drivers from first boot — zero configuration. It shares Ubuntu's package base, so ML guides transfer directly. The new COSMIC desktop (Rust-based, Wayland-only) is still maturing, which introduces some uncertainty.

| Criterion | Ubuntu 24.04 | Fedora 42 | Arch | Pop!_OS | Debian 13 | NixOS 25.05 | openSUSE TW |
|---|---|---|---|---|---|---|---|
| NVIDIA/CUDA | ★★★★★ | ★★★★ | ★★★★ | ★★★★★ | ★★★ | ★★ | ★★ |
| ML community | ★★★★★ | ★★★★ | ★★★ | ★★★★ | ★★★ | ★★ | ★★ |
| Maintenance burden | ★★★★★ | ★★★ | ★★ | ★★★★ | ★★★★★ | ★★★★★ | ★★★★ |
| Ryzen 9000 support | ★★★★ | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ | ★★★★ | ★★★★★ |
| Rollback capability | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ | ★★★★★ | ★★★★★ |

**NixOS** offers the best reproducibility story — atomic rollbacks, declarative configuration, instant generation switching — but CUDA on NixOS remains painful. The non-FHS store layout forces workarounds like `buildFHSEnv` sandboxes or Docker containers. Many users report giving up on NixOS specifically because of CUDA friction. **openSUSE Tumbleweed** should be avoided for ML: NVIDIA's CUDA toolkit doesn't officially support it, drivers frequently lag kernel updates, and community documentation is thin. **Debian 13** (kernel 6.12) works but offers fewer ML-specific resources than Ubuntu.

---

## i3 on X11 is the right GUI choice for NVIDIA ML workstations

The window manager decision comes down to a fundamental trade-off: Wayland's modernity versus X11's battle-tested NVIDIA compatibility. For a production ML workstation, **X11 wins decisively**.

**i3** consumes roughly **~150–250 MB system RAM** total (including Xorg) and only **~50–150 MB GPU VRAM** at 1080p–1440p resolution. Compare this to GNOME at 700–1000 MB RAM and **500–2000 MB VRAM** (one user measured 1.5 GB idle VRAM on Pop!_OS with GNOME), or KDE at nearly **2 GB VRAM** idle at 4K. On a 16GB GPU where every megabyte matters for training, i3's overhead is negligible.

The Wayland compositors — Sway and Hyprland — have improved dramatically with NVIDIA driver 555+'s explicit sync support, but neither officially supports NVIDIA proprietary drivers. Sway requires the `--unsupported-gpu` flag and community-patched wlroots builds. Hyprland has extensive community documentation but mixed stability reports. More critically, the Arch Wiki documents a **Wayland VRAM leak** on NVIDIA where compositors like Sway and Hyprland can consume ~2.5 GB of idle VRAM without the `GLVidHeapReuseRatio` workaround. This is unacceptable for ML workloads.

The practical ML workflow on i3 is excellent: tiling windows naturally organize multiple terminals for training jobs, `nvtop` monitoring, and log watching. A dedicated workspace holds Firefox for documentation and Weights & Biases. All GPU monitoring tools (`nvidia-smi`, `nvtop`, `gpustat`, `nvitop`) work identically across all environments since they're terminal-based.

**The ideal setup for maximum VRAM**: if this machine will serve primarily as a training server accessed via SSH, run **headless with no GUI at all** — zero VRAM overhead. Install `nvidia-headless-580` instead of the full driver package. Use SSH + tmux for persistent sessions, VS Code Remote for editing, and JupyterLab via SSH tunnel. You can always install i3 later if local GUI access becomes important.

For NVIDIA driver 570+ on X11, the required configuration is minimal:

```
# /etc/modprobe.d/nvidia.conf
options nvidia_drm modeset=1
# fbdev=1 is default in 570+
```

---

## NVIDIA driver management: open modules and smart pinning

The NVIDIA driver landscape shifted significantly in 2025. **Open kernel modules (`nvidia-open`) are now the default and recommended path** for all Turing+ GPUs (RTX 20-series and newer). As of driver R590 (December 2025), Arch Linux switched its main `nvidia` package to open modules, and they're mandatory for Blackwell GPUs (RTX 5090). Performance is equivalent to proprietary modules within ~1% for CUDA compute workloads — these share the same proprietary userspace components and GSP firmware; only the kernel module layer is open-source (dual MIT/GPL licensed).

The recommended installation path on Ubuntu uses NVIDIA's official repository:

```bash
# Add NVIDIA's repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install open kernel modules + driver
sudo apt install nvidia-driver-580-open

# Pin the driver branch
sudo apt install nvidia-driver-pinning-580
```

**Kernel update breakage** is the primary maintenance risk. When a new kernel introduces ABI changes (as kernel 6.19 did for NVIDIA 580.x), DKMS module builds fail, leaving the system without GPU acceleration. The defenses are layered:

- **Always keep the previous kernel installed** — boot into it from GRUB if the new one breaks
- **Pin working combinations** with `apt-mark hold linux-image-generic nvidia-driver-580-open` until ready to update
- **Use btrfs + snapper** (or Timeshift) for instant pre-update snapshots with one-command rollback
- **Prefer LTS kernels** (6.6, 6.12) over bleeding-edge for production stability

For CUDA toolkit management, **install only the driver system-wide**. PyTorch's pip wheels bundle their own CUDA runtime libraries — you don't need a system-wide CUDA toolkit unless compiling custom CUDA extensions. The recommended CUDA version for early 2026 is **CUDA 12.8**, supported by PyTorch 2.9.x and 2.10.0 and compatible with driver 580.x. When upgrading to RTX 5090, you'll need driver ≥590 (which requires open kernel modules) and CUDA 12.8+ for Blackwell's sm_120 compute capability.

---

## The multi-GPU bottleneck you need to know about

The most consequential finding for future planning: **the ASUS Prime X670E-PRO WiFi cannot provide adequate bandwidth for two GPUs**. The board's PCIe slot layout is:

- **Slot 1** (CPU-direct): PCIe 5.0 x16 — **~64 GB/s** bidirectional
- **Slot 2** (chipset-routed): PCIe 4.0 x4 — **~8 GB/s** bidirectional (1/8th of Slot 1)

The Ryzen 9 9900X provides **28 total PCIe 5.0 lanes** from the CPU (24 usable + 4 for chipset downlink), with 16 lanes dedicated exclusively to the primary GPU slot. There is **no x8/x8 bifurcation option**. A second GPU in Slot 2 communicates through the X670E chipset at a fraction of the bandwidth, making distributed training (DDP, model parallelism) with gradient synchronization between GPUs impractical.

The viable dual-GPU strategy on this board is **task isolation, not distributed training**: place the RTX 5090 in Slot 1 for primary ML compute, and relegate the RTX 4060 Ti to Slot 2 for display output, light inference, or data preprocessing. Connect monitors to the 4060 Ti to keep the 5090's full VRAM available for training. Use `CUDA_VISIBLE_DEVICES=0` to isolate training jobs to the primary GPU.

Mixed GPU models (Ada Lovelace + Blackwell) work under a single driver version — driver 570+ supports both architectures. CUDA handles different compute capabilities transparently. However, DDP across mismatched GPUs is inherently problematic regardless of bandwidth: every training step waits for the slowest GPU.

**If true multi-GPU distributed training is the goal**, the platform upgrade path is AMD Threadripper PRO (WRX90) with **128 PCIe 5.0 lanes** supporting multiple x16 GPU slots, or select X870E boards that offer PCIe bifurcation.

Power planning for dual GPU: the RTX 5090 draws **575W TDP** (spikes higher), the 4060 Ti draws 160W, and the 9900X draws ~170W under PBO. Total system power approaches **~1,000W**, requiring a **1,300W+ ATX 3.0/3.1 PSU** with native 12V-2x6 connectors. Thermal management becomes critical — combined GPU heat output exceeds 735W, demanding a full tower with excellent airflow.

---

## ML environment setup: uv, Unsloth, and the 16GB VRAM sweet spot

**uv** has emerged as the recommended Python package manager for ML workflows in 2025-2026. It's **10-100x faster than pip**, produces deterministic lockfiles (`uv.lock`), and handles PyTorch's CUDA index requirements cleanly. The PyTorch conda channel is officially deprecated as of PyTorch 2.6 — pip/uv is now the canonical installation method.

```toml
# pyproject.toml for ML project
[project]
dependencies = [
    "torch>=2.9.0,<2.10",
    "transformers>=5.0,<6.0",
    "peft>=0.14,<0.15",
    "trl>=0.28,<0.29",
    "bitsandbytes>=0.49,<0.50",
    "accelerate>=1.3,<2.0",
    "datasets>=3.2,<4.0",
]

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

For fine-tuning on 16GB VRAM, **Unsloth is the most impactful single tool recommendation**. It delivers **2x faster training with 70% less VRAM** through custom Triton kernels, padding-free training, and dynamic 4-bit quantization. Concrete examples: Qwen3-4B trains on just **3.9 GB VRAM** with Unsloth; models that barely fit with standard QLoRA become comfortable.

The 16GB VRAM budget enables QLoRA fine-tuning of these models:

- **Llama 3.2 3B / Phi-4 Mini 3.8B / Qwen3-4B**: ~4–7 GB, comfortable with room for larger batches
- **Mistral 7B**: ~6–10 GB in 4-bit, fits with small batch sizes
- **Llama 3.1 8B**: ~8–12 GB, tight — requires gradient checkpointing and batch_size=1-2
- **Qwen3-30B-A3B** (MoE, 3B active): ~17.5 GB — requires Unsloth optimizations

Flash Attention 2 is **fully supported on RTX 4060 Ti** (Ada Lovelace, sm_89) for both fp16 and bf16. PyTorch 2.9+ also includes SDPA with a FlashAttention backend built in, so many models benefit without installing the separate `flash-attn` package.

For containerized workflows, Docker with NVIDIA Container Toolkit (v1.18.2) provides complete isolation. The key Docker Compose detail: set `shm_size: '8gb'` for PyTorch DataLoader shared memory, and mount a persistent HuggingFace cache volume to avoid re-downloading models.

---

## System tuning that actually moves the needle

Not all optimizations matter equally. These are ranked by impact for ML training throughput on this specific hardware.

**High impact — configure these first:**

The **AMD P-State driver** should run in active mode with performance energy preference: `amd_pstate=active` on the kernel command line, then `echo performance > /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference`. This lets hardware manage frequency scaling biased toward maximum clocks during training. **NVIDIA persistence mode** (`nvidia-smi -pm 1`) eliminates the 1–3 second GPU initialization delay between CUDA calls. **Lock GPU clocks** (`nvidia-smi --lock-gpu-clocks=2100,2700 --lock-memory-clocks=9001`) for consistent training throughput — memory bandwidth on the 4060 Ti's 128-bit bus is often the bottleneck. **Disable PCIe ASPM** (`pcie_aspm=off`) — NVIDIA developer forums document "GPU has fallen off the bus" errors on ASUS Prime boards with ASPM enabled.

**Medium impact — configure during initial setup:**

Set `vm.swappiness=10` (prefer RAM over swap while maintaining an OOM safety net), `vm.max_map_count=1048576` (critical for large model memory maps), and NVMe I/O scheduler to `none` (the Samsung 990 Pro has its own internal scheduling; kernel schedulers add overhead). Mount with `noatime,nodiratime` to eliminate access-time writes during data loading. Set read-ahead to **2048 KB** for sequential dataset loading.

**BIOS settings that matter:**

- **Enable AMD EXPO** for DDR5-6000 with FCLK at 2000 MHz (1:1 ratio)
- **Enable Resizable BAR**: requires CSM disabled + Above 4G Decoding enabled
- **Enable IOMMU** with `iommu=pt` kernel parameter for passthrough mode
- **PBO settings**: Enable with Curve Optimizer at -10 to -15 all cores for better efficiency. Stock PPT is 162W; motherboard limits allow higher sustained boost

The Ryzen 9 9900X's **2 CCDs with 6 cores each** expose as a single NUMA node by default. For advanced tuning, pin the training process to CCD0 cores (`taskset -c 0-5,12-17 python train.py`) and let data loading workers use CCD1, keeping L3 cache locality. Inter-CCD latency is ~180ns, though GPU DMA transfers go through the IOD regardless of CCD, so the impact is modest.

**The complete kernel command line:**
```
amd_pstate=active pcie_aspm=off iommu=pt transparent_hugepage=always
```

---

## Post-install guide for the recommended configuration

This guide assumes Ubuntu Server 24.04 LTS with i3, targeting a working ML environment from bare metal.

**Phase 1 — Base system (30 minutes):**
Install Ubuntu Server 24.04 LTS from USB (minimal installation, no desktop). Immediately install the HWE kernel: `sudo apt install linux-generic-hwe-24.04`. Reboot. Install i3 and essentials: `sudo apt install i3 xorg lightdm alacritty firefox thunar`. Enable lightdm: `sudo systemctl enable lightdm`.

**Phase 2 — NVIDIA driver (15 minutes):**
Add NVIDIA's repository via `cuda-keyring`, install `nvidia-driver-580-open`, pin with `nvidia-driver-pinning-580`. Reboot. Verify with `nvidia-smi`. Enable persistence: `sudo systemctl enable nvidia-persistenced`.

**Phase 3 — System optimization (10 minutes):**
Apply kernel parameters to GRUB, create `/etc/sysctl.d/99-ml-training.conf` with the tuning values above, create the udev rule for NVMe scheduler. Enable btrfs snapshots with snapper (format root as btrfs during installation). Run `sudo update-grub && sudo sysctl --system`. Reboot.

**Phase 4 — ML environment (20 minutes):**
Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`. Create a project: `uv init ml-finetune --python 3.12`. Configure `pyproject.toml` with PyTorch CUDA 12.8 index. Install dependencies: `uv add torch transformers peft trl bitsandbytes accelerate datasets unsloth`. Install Docker + NVIDIA Container Toolkit for containerized workflows.

**Phase 5 — Verify everything:**
```python
import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (8, 9)  # Ada Lovelace
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

**Before every training run**, execute the GPU optimization script: enable persistence mode, set power limit to 165W, lock clocks, and take a snapper snapshot. This configuration maximizes what's achievable on the RTX 4060 Ti 16GB while providing a clean upgrade path to high-end NVIDIA GPUs — just swap the card, update the driver if needed, and your entire software stack continues working.