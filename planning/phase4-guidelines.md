# Phase 4 Guidelines — Self-Improvement Loop

**Status:** Pre-planning. Phase 3 complete, refactoring in progress (Batches 4-6).
**Nature:** High-level roadmap and decision principles, not a task list.

---

## 1. What Phase 4 Is

Phase 4 closes the loop: the agent's logged activity becomes training data that improves the agent. It also opens the bootstrapping path: external repos provide training signal before the agent has produced any real runs.

Phase 4 builds four things:
1. **Plan validation harness** — automated plan generation + execution + test validation against external repos
2. **Training infrastructure** — data extraction, curation, LoRA training, adapter management
3. **Two new pipeline modes** — `cra train-plan` (analyze logs → training plan) and `cra curate-data` (extract training pairs)
4. **Inference server migration** — Ollama → vLLM for per-request LoRA adapter selection

Phase 4 does NOT modify the retrieval pipeline or orchestrator. It reads from what they produce (raw DB) and writes adapters that improve how they perform.

---

## 2. Prerequisites

Before Phase 4 work begins:
- [ ] Refactoring Batches 4-6 complete (runner decomposition, cross-module cleanup)
- [ ] All transparency audit findings resolved (14/14 — currently 9/14 done)
- [ ] Enough logged orchestrator runs in raw DB to validate data extraction queries (even manual test runs count)
- [ ] Hardware: consumer GPU available for LoRA training (RTX 5060 Ti is sufficient for QLoRA)

Phase 4's bootstrapping path depends only on Phase 1 (indexer). Self-improvement depends on Phase 3 logs.

---

## 3. Work Streams

Three parallel tracks. They share infrastructure but can be developed independently.

### Stream A: Harness Infrastructure (highest priority)

The plan validation harness is the single most important piece — it provides training data for every stage, enables automated DPO pair generation, and becomes the self-improvement vehicle post-migration.

**What to build:**
- Repo package format: `repo/` (git clone) + `packages/` (pre-cached .whl files) + `manifest.json` (filtered commits, test command, Python version)
- Auto-venv creation from pre-cached wheels (no Docker, no internet)
- Commit filtering pipeline: PyDriller extraction → message scoring (verb-initial, descriptive) → diff size filtering → test existence check. Expect ~96% attrition from raw commits.
- Harness runner: per-commit loop of index → retrieve → plan → execute → validate → record outcome
- Plan-diff structural alignment scorer: isolates plan quality from coding quality by checking file target overlap, symbol overlap, dependency ordering against the actual commit diff
- Test supplementation: generate tests from the diff when repo's existing coverage is insufficient

**Initial repo corpus:** 14 fail-fast Python repos identified in the research (attrs, cattrs, structlog, Black, Hypothesis, typeguard, beartype, LibCST, parso, svcs, strictyaml, stamina, mypy, rich). Expected yield: 1,400-4,200 qualifying commits → 4,200-12,600 training triples with multi-temperature runs.

**Key principle:** The harness uses the real pipeline (Phase 1 indexer + Phase 2 retrieval + Phase 3 orchestrator). Zero distribution mismatch between training and inference. This is the harness's primary advantage over synthetic data.

### Stream B: Training Pipeline

**Base fine-tune (outsourced):**
- Two base models: Qwen3-4B (reasoning/planning) and Qwen2.5-Coder-3B (coding)
- Planning capability baked into the Qwen3-4B base fine-tune (not LoRA) — all three planning types (meta-plan, step-plan, test-plan) via CoT-SFT
- 32K context window reduction for most stages (empirically validated: shorter windows produce better output)
- Coding style from the 25-repo fail-fast corpus
- Outsourced to external GPU (4× A100 80GB or equivalent). One-time activity, possibly iterated 2-3 times.

**LoRA adapters (on-premises, consumer GPU):**

| Adapter | Model | Technique | Data Source | Priority |
|---|---|---|---|---|
| Scope (file relevance) | Qwen3-4B | SFT | Commit diffs (files changed = ground truth) | Tier 1 |
| Execute-Code | Qwen2.5-Coder-3B | SFT + DPO | CommitPackFT + harness successes/failures | Tier 1 |
| Precision (symbol tier) | Qwen3-4B | SFT + curriculum | Commit diffs (symbols modified = ground truth) | Tier 2 |
| Task Analysis | Qwen3-4B | SFT | Logged task analyses + teacher distillation | Tier 2 |
| Planning DPO (optional) | Qwen3-4B | DPO | Harness preference pairs | Tier 3 |

**Training framework:** Unsloth + QLoRA (4-bit, all 7 linear layers). ZeRO-2 only for distributed — ZeRO-3 breaks gradient flow with LoRA on Qwen3. Explicit `eos_token='<|im_end|>'` to avoid the silent Qwen3 tokenizer bug.

**Cold-start datasets:** CommitPackFT (702K pairs, Apache 2.0), OpenCodeInstruct (subsample 100-500K), SRI Tuning (search-and-replace format). These bootstrap the Execute-Code adapter before the harness produces data.

### Stream C: Pipeline Modes + Inference Migration

**`cra train-plan`:** Reads raw DB, analyzes logged runs, identifies weak stages, produces a training plan artifact. Uses log analysis retrieval stages (new) instead of code retrieval stages.

**`cra curate-data`:** Guided by a training plan, extracts and formats training pairs from raw DB. Filters, balances, deduplicates, converts to JSONL. Stores in raw DB `training_datasets` table.

**`LogAnalysisAPI`:** The Phase 4 equivalent of `KnowledgeBase`. Reads from raw DB, provides queries for logged LLM calls, retrieval decisions, task outcomes, cross-referenced by `task_id`.

**Inference server migration:** Ollama → vLLM (or llama-server as fallback). Required for per-request LoRA adapter routing — Ollama cannot hot-swap adapters. The `LLMClient.complete()` interface doesn't change; only the transport layer switches to `provider = "openai_compat"`.

---

## 4. Execution Order

```
                    ┌─────────────────┐
                    │  Stream A:      │
                    │  Harness Infra  │──────┐
                    └────────┬────────┘      │
                             │               │
                    ┌────────▼────────┐      │
                    │  Repo corpus    │      │  ┌──────────────────┐
                    │  packaging      │      │  │  Stream B:       │
                    └────────┬────────┘      │  │  Cold-start data │
                             │               │  │  (CommitPackFT)  │
                    ┌────────▼────────┐      │  └────────┬─────────┘
                    │  Teacher-driven │      │           │
                    │  harness runs   │◄─────┘  ┌────────▼─────────┐
                    └────────┬────────┘         │  Execute-Code    │
                             │                  │  LoRA (Tier 1)   │
                    ┌────────▼────────┐         └────────┬─────────┘
                    │  Outsourced     │                  │
                    │  base fine-tune │                  │
                    └────────┬────────┘         ┌────────▼─────────┐
                             │                  │  Scope LoRA      │
                    ┌────────▼────────┐         │  (Tier 1)        │
                    │  Gate 1: Base   │         └──────────────────┘
                    │  planning eval  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───────┐ ┌───▼──────┐ ┌─────▼──────────┐
     │ Precision LoRA │ │ Task     │ │ Stream C:      │
     │ (Tier 2)       │ │ Analysis │ │ Pipeline modes │
     └────────────────┘ │ LoRA     │ │ + vLLM         │
                        └──────────┘ └────────┬───────┘
                                              │
                                     ┌────────▼────────┐
                                     │  Gate 2: E2E    │
                                     │  pipeline eval  │
                                     └────────┬────────┘
                                              │
                                     ┌────────▼────────┐
                                     │  Migration to   │
                                     │  air-gapped     │
                                     └────────┬────────┘
                                              │
                                     ┌────────▼────────┐
                                     │  Self-improving  │
                                     │  loop active    │
                                     └─────────────────┘
```

**Gate 1:** Base model demonstrates planning capability across all three types (meta-plan, step-plan, test-plan) on held-out harness commits. Code LoRA and Scope LoRA pass stage-level evaluation.

**Gate 2:** End-to-end pipeline success rate is stable. `cra train-plan` and `cra curate-data` produce valid outputs. Adapter versioning and rollback work correctly.

---

## 5. Decision Principles

**Harness before training.** The harness produces every type of training data (SFT, DPO, evaluation). Build it first, validate it produces quality data, then train.

**Base fine-tune before LoRAs.** LoRAs are stacked on the base. A better base lifts all adapters. Planning goes in the base because it's the reasoning model's core competency and data is scarce (all parameters updated = more learning per example).

**Teacher distillation is finite.** All Qwen3.5-397B API calls happen pre-migration on the connected machine. After air-gap, no teacher access. Front-load teacher-driven harness campaigns.

**Accumulate, never replace.** Each training round adds to the data pool. Previous rounds are never discarded. Training always starts from frozen base checkpoint. 10-20% replay data in every batch to prevent forgetting.

**Repo diversity over volume.** Adding a new repo to the harness is worth more than collecting 10x more data from existing repos (Hybrid-Gym finding). Target 6+ domains, max 20% from any single domain.

**Expect plateau.** Self-improvement plateaus after ~2 iterations at sub-7B scale (SWE-Gym). Break plateaus by injecting new repo diversity, not by running more iterations on the same distribution.

**Chronological evaluation only.** Random train/test splits produce results 57-94% more optimistic than chronological splits. All evaluation uses temporal holdout.

---

## 6. Self-Improvement Guardrails

- Sub-6B models cannot bootstrap from self-generated data alone — external teacher signal is mandatory for initial training
- Recursive training on synthetic data leads to model collapse without data management (Shumailov et al., Nature 2024)
- Full GRPO/PPO is not feasible on consumer hardware — DPO from self-generated trajectories is the practical alternative
- Never merge a LoRA into base weights and retrain from the merged model
- Every adapter version is kept (20-50 MB each) — rollback is a DB update, not retraining
- If any primary metric regresses after adapter promotion, immediately rollback

---

## 7. Multi-Use Repo Corpus

Every repo the system ingests serves multiple purposes simultaneously. The harness is not just a plan validator — it's the universal data extraction engine. A single well-chosen repo can provide coding style training, planning data, knowledge base content, and domain-specific capability all at once.

### Use categories

Each repo can serve one or more of these roles:

| Role | What it provides | Where it goes |
|---|---|---|
| **P** — Planning training | Commit history → harness → plan generation/validation → SFT + DPO pairs | Raw DB (harness triples) |
| **C** — Code style training | Source code patterns → base fine-tune + Execute-Code LoRA | Training datasets |
| **K** — Knowledge base | Indexed + enriched → curated DB → retrieved during agent tasks | Curated DB |
| **S** — Self-referential | The repo teaches the agent a skill it needs to build/improve itself | Knowledge base + training |

The **S** role is the most powerful: the agent studies code that does what it needs to do, then uses that knowledge to do it itself.

### Repo categories and concrete examples

**1. Fail-fast Python repos** — P + C
Already identified (14 repos). Dual-purpose: code trains fail-fast style, commits feed the planning harness.
- attrs, cattrs, structlog, svcs, stamina, strictyaml, Black, LibCST, parso, typeguard, beartype, Hypothesis, mypy, rich

**2. LoRA / fine-tuning repos** — P + C + K + S
The agent needs to write and improve its own training code. These repos teach it how.
- **Unsloth** — optimized QLoRA training, Triton kernels, Qwen support. The agent's own training framework.
- **PEFT** — LoRA adapter implementation, merging, stacking. Core to the adapter architecture.
- **trl** — SFTTrainer, DPO training, reward modeling. The agent's alignment training tools.
- **axolotl** — advanced fine-tuning (sample packing, fused kernels). Alternative patterns.
- **LLaMA-Factory** — YAML-driven training, multi-method support. The agent could learn to auto-configure training.

**3. From-scratch training repos** — P + C + K + S
These teach the agent to build its own training infrastructure — directly enabling the C-native trainer.
- **llm.c** (Karpathy) — GPT-2 training in pure C. The direct template for the C-native trainer.
- **nanoGPT** (Karpathy) — minimal PyTorch GPT training. Clean reference implementation.
- **minGPT** (Karpathy) — even more minimal. Good for understanding core patterns.
- **tinygrad** (geohot) — tiny ML framework from scratch. Shows how to build a framework, not just use one.
- **micrograd** (Karpathy) — autograd engine in Python. The microgpt.py Value class expanded.

**4. CUDA / GPU programming repos** — P + C + K + S
The agent needs these to write its own CUDA kernels for the C-native trainer.
- **llama.cpp** — inference in C/C++ with CUDA kernels. Quantization, KV cache, attention kernels.
- **ggml** — tensor library underlying llama.cpp. Low-level GPU compute patterns.
- **whisper.cpp** — another C++ inference engine. Different architecture, same patterns.
- **CUTLASS** (NVIDIA) — CUDA templates for matrix operations. The building blocks for custom kernels.
- **flash-attention** — the actual Flash Attention CUDA implementation. Critical for efficient attention.
- **ThunderKittens** (Stanford) — embedded DSL for CUDA kernels. Next-gen kernel writing patterns.

**5. Inference server repos** — P + K + S
The agent migrates from Ollama to vLLM. Understanding internals helps debug and configure.
- **vLLM** — the target inference server. Per-request LoRA routing, PagedAttention, continuous batching.
- **llama-server** (part of llama.cpp) — fallback inference server with LoRA support.
- **text-generation-inference** (HuggingFace) — alternative server, good reference for batching strategies.

**6. Data pipeline repos** — P + C + K + S
The `cra curate-data` mode needs to process, filter, and format training data efficiently.
- **datasets** (HuggingFace) — the data loading library the agent uses. Understanding internals helps optimize.
- **datatrove** (HuggingFace) — large-scale data processing. Filtering, dedup, quality scoring patterns.
- **dolma** (AI2) — data curation toolkit. Pre-training data pipeline patterns at scale.

**7. AST / parser repos** — P + C + K + S
The agent's indexer is built on AST parsing. These repos improve its own parsing capabilities.
- **tree-sitter** — the core parsing library. C repo with grammar definitions.
- **tree-sitter-python**, **tree-sitter-c**, **tree-sitter-cuda** — language grammars. Adding C/CUDA parsing for the knowledge base expansion.
- **LibCST** — Python CST manipulation. Already in the fail-fast corpus.
- **parso** — Python parser. Already in the fail-fast corpus.

**8. Testing / evaluation repos** — P + C + K + S
The harness runs tests. The agent evaluates its own adapters. These teach it how.
- **pytest** — the test framework. Understanding internals helps generate better tests.
- **Hypothesis** — property-based testing. Already in fail-fast corpus. Test generation patterns.
- **lm-evaluation-harness** (EleutherAI) — LLM evaluation framework. Patterns for adapter evaluation.
- **bigcode-evaluation-harness** — code-specific evaluation. SWE-bench, HumanEval runners.

**9. Git / VCS repos** — P + C + K
The harness uses PyDriller for commit extraction. The agent does git operations.
- **PyDriller** — commit mining library. The agent's own commit filtering pipeline uses this.
- **gitpython** — Git operations from Python. The agent's git workflow.
- **dulwich** — pure-Python Git implementation. Useful if gitpython is insufficient.

**10. Database repos** — P + C + K
The three-database architecture uses SQLite. Understanding internals helps optimize.
- **sqlite** (C source) — the actual SQLite implementation. ~150K lines of battle-tested C.
- **SQLAlchemy** — ORM patterns. Not directly used but relevant for query optimization.
- **peewee** — lightweight ORM. Patterns for the agent's own DB query layer.

**11. CLI / TUI repos** — P + C + K
The agent's CLI uses Click. These teach UI patterns.
- **click** — the CLI framework used by `cra`. Understanding internals helps extend it.
- **typer** — modern CLI framework. Alternative patterns.
- **rich** — terminal formatting. Already in fail-fast corpus.
- **textual** — TUI framework. Potential for interactive agent interfaces.

**12. Quantization / optimization repos** — K + S
The agent deploys quantized models and may learn to quantize its own mini-models.
- **bitsandbytes** — NF4 quantization for QLoRA. The agent's own quantization tool.
- **GPTQ** — post-training quantization. Alternative approach for mini-model deployment.
- **llama.cpp quantization tools** — GGUF format, various quantization methods.

**13. C reference / systems repos** — P + C + K
Mature C codebases that teach systems programming patterns for the C-native trainer.
- **Redis** — clean C codebase, excellent patterns for data structures, event loops.
- **SQLite** (again) — the gold standard for robust, well-tested C.
- **jq** — C-based JSON processor. Clean, focused C codebase.
- **zstd** (Facebook) — compression in C. High-performance numerical patterns.

**14. Documentation / reference material** — K only
Not code repos — indexed as knowledge base content for retrieval.
- C programming books (K&R, Modern C)
- CUDA Programming Guide (NVIDIA)
- PyTorch internals documentation
- Transformer architecture papers (Attention Is All You Need, Flash Attention, RoPE)
- SQLite internals documentation

### The self-referential flywheel

The most valuable repos are those where studying the code teaches the agent to improve itself:

```
Agent studies Unsloth → learns to write training code
  → writes better training pipeline → trains better adapters
    → produces better code → writes even better training code → ...

Agent studies llm.c → learns C/CUDA training patterns
  → writes C-native trainer → trains mini-models faster
    → mini-models replace LLM calls → pipeline runs faster
      → more data → better models → ...

Agent studies vLLM → understands inference serving
  → optimizes its own deployment → lower latency
    → more tasks per hour → more training data → ...

Agent studies tree-sitter → improves its own parser
  → better AST indexing → better retrieval → better context
    → better code output → ...
```

Each self-referential repo creates a feedback loop. The more of these the system ingests, the more capabilities it has to improve itself.

### Corpus sizing

Not all repos need full packaging (wheels, manifest). The tiers:

| Tier | What | Packaging | Count |
|---|---|---|---|
| **Full harness** | Python repos with tests, commit history usable for planning training | repo + packages + manifest | 25-40 repos |
| **Knowledge base only** | C/CUDA repos, reference material, repos without usable test suites | repo only (indexed + enriched) | 20-30 repos |
| **Documentation** | Books, guides, papers | chunked text (indexed + enriched) | 10-15 sources |

Total: ~60-85 sources. All enter the air-gapped machine via USB. The harness auto-detects new full-harness repos from their `manifest.json`.

---

## 8. The Long Game (Post-Phase 4)


These are not Phase 4 deliverables. They are horizons that Phase 4 enables.

**Knowledge base expansion:** Index reference documentation (C books, CUDA docs, algorithm references) alongside code. The retrieval pipeline is content-agnostic — extend the indexer with document chunking and enrichment. Tree-sitter already has C/CUDA grammars. Worth a research spike: manually chunk one book, run through enrichment, evaluate retrieval quality.

**Mini-model creation:** The agent trains small specialist models (10M-500M) to replace LLM calls in the pipeline. A 10M retrieval classifier runs in milliseconds vs seconds for a 4B model call. Requires: the training infrastructure from Phase 4 + the ability to define and train custom architectures.

**C-native training:** The microgpt.py algorithm reimplemented in C/CUDA. Eliminates PyTorch dependency, enables custom kernels optimized for the exact hardware and model architecture. Self-contained training binary: data in, model out. On the air-gapped machine, this means no framework dependencies at all.

**The cascade:** Mini-models replace LLM calls → pipeline runs faster → more data collected → better models → faster pipeline. The C-native trainer accelerates the inner loop. The knowledge base expansion gives the agent new capabilities (writing C, CUDA, new languages). Each capability feeds the next.

---

## 9. What's Different From the Training Strategy

This document is a guideline for *how to approach Phase 4*. The training strategy (`planning/training-strategy.md`) is the detailed technical reference for *what to train and how*. They don't conflict — this document organizes the strategy into an execution roadmap.

Key clarifications this document adds:
- Explicit three-stream parallel structure (harness, training, pipeline modes)
- Execution flow diagram with gates
- Decision principles distilled from the research reviews
- The long game vision connecting Phase 4 to the C-native trainer and knowledge base expansion
