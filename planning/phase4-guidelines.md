# Phase 4 Guidelines — Self-Improvement Loop

**Status:** Pre-planning. Phase 3 complete, refactoring complete (Batches 0-6).
**Nature:** High-level roadmap and decision principles, not a task list.
**Companion documents:** `repo-corpus.md` (full repo catalog), `training-strategy.md` (training details)

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
- [x] Refactoring Batches 0-6 complete
- [ ] All transparency audit findings resolved (14/14 — currently 9/14 done)
- [ ] Enough logged orchestrator runs in raw DB to validate data extraction queries (even manual test runs count)
- [ ] Hardware: consumer GPU available for LoRA training (RTX 5060 Ti is sufficient for QLoRA)

Phase 4's bootstrapping path depends only on Phase 1 (indexer). Self-improvement depends on Phase 3 logs.

---

## 3. Work Streams

Three parallel tracks. They share infrastructure but can be developed independently.

### Stream A: Harness Infrastructure (highest priority)

The harness is a **training data factory**, not just a plan validator. It provides training data for every stage, enables automated DPO pair generation, and becomes the self-improvement vehicle post-migration. Once the agent can be its own teacher, this runs fully offline on the air-gapped machine.

**What to build:**
- Repo package format: `repo/` (git clone) + `packages/` (pre-cached dependencies) + `manifest.json` (filtered commits, validation command, language/runtime version)
- Auto-environment creation from pre-cached dependencies (no Docker, no internet)
- Commit filtering pipeline: PyDriller extraction → message scoring (verb-initial, descriptive) → diff size filtering → validation existence check. Expect ~96% attrition from raw commits.
- Harness runner: per-commit loop of index → retrieve → plan → execute → validate → record outcome
- Multi-temperature teacher runs: same commit prompted at temps 0.3/0.6/0.9/1.2. Multiplies output 3-4x and produces natural DPO preference pairs (plan A at temp 0.6 passed, plan B at temp 1.2 failed = DPO pair with no human labeling)
- Plan-diff structural alignment scorer: isolates plan quality from coding quality by checking file target overlap, symbol overlap, dependency ordering against the actual commit diff
- Test supplementation: generate tests from the diff when repo's existing coverage is insufficient

**Validation is broader than test suites.** Test suites are the simplest oracle, but not the only one. The harness can validate by:
- Running tests (`pytest`, `make test`, `cargo test`, `go test`)
- Running the actual code (spin up a server, hit endpoints, check responses)
- Agentic debugging loops (error → hypothesis → add logging → re-run → narrow down → fix)
- Writing integration tests and property checks against observable behavior
- Performance profiling (before/after measurement = objective signal)

A repo without tests isn't a dead end — the agent writes the verification, runs it, and the outcome is still binary signal. Every agentic verification loop is itself training data: traces of HOW the agent reasoned about the problem.

**Repo corpus:** See `repo-corpus.md` for the full catalog (~170 sources across 36 categories). Not limited to Python — any repo with a deterministic validation path is harness-compatible. The corpus is finite but the data is not: same repos, infinite variations through temperature, reasoning paths, and verification strategies.

**Key principles:**
- The harness uses the real pipeline (Phase 1 indexer + Phase 2 retrieval + Phase 3 orchestrator). Zero distribution mismatch between training and inference.
- Harness data is the PRIMARY training source. External datasets (CommitPackFT) serve as cold-start bridges only. Once harness produces volume, it dominates.
- Domain diversity over volume within any single domain. Each new repo domain teaches new reasoning patterns that compound.

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

**Cold-start datasets:** CommitPackFT (702K pairs, Apache 2.0) bootstraps the Execute-Code adapter before the harness produces data. Once harness volume is sufficient, CommitPackFT becomes 10-20% replay data to prevent distribution narrowing. OpenCodeInstruct and SRI Tuning are optional breadth insurance, not primary sources — the base Qwen models already have general coding ability from pre-training.

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

**Harness data over external datasets.** The distribution mismatch argument is decisive — training on data from the exact pipeline, validated by real tests, on real repos, beats any external dataset collected under different assumptions. CommitPackFT is a cold-start bridge; harness data is the primary source once available.

**Teach HOW to think, not WHAT.** The diverse repo corpus exposes reasoning patterns — when to decompose, when to iterate, when to approximate. A regression in an econ model and a least-squares fit in physics are the same reasoning shape in different domains. Breadth of domains matters more than depth in any single domain.

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

**Full catalog:** See `repo-corpus.md` for the complete repo listing (~170 sources across 36 categories).

Every repo the system ingests serves multiple purposes simultaneously. The harness is a universal data extraction engine. A single well-chosen repo provides coding style training, planning data, knowledge base content, and domain-specific reasoning patterns all at once.

### Use roles

| Role | What it provides | Where it goes |
|---|---|---|
| **P** — Planning training | Commit history → harness → plan generation/validation → SFT + DPO pairs | Raw DB (harness triples) |
| **C** — Code style training | Source code patterns → base fine-tune + Execute-Code LoRA | Training datasets |
| **K** — Knowledge base | Indexed + enriched → curated DB → retrieved during agent tasks | Curated DB |
| **S** — Self-referential | The repo teaches the agent a skill it needs to build/improve itself | Knowledge base + training |

### The data factory principle

The corpus is finite but the data is not. Same repos, infinite variations:
- Multi-temperature teacher runs produce different reasoning paths for the same commit
- Each path either passes or fails validation — natural DPO pairs
- Agentic verification loops (debugging, profiling, integration testing) generate reasoning traces
- Every run produces training signal about HOW to approach problems, not just WHAT to produce

Once the agent can be its own teacher, this runs fully offline. Given enough compute time, data is unlimited.

### Corpus categories (36 total)

**Self-referential (Categories 1-18):** Fail-fast Python, LoRA/fine-tuning, from-scratch training, CUDA/GPU, inference servers, data pipeline, AST/parser, testing/evaluation, Git/VCS, database, CLI/TUI, quantization, C reference/systems, documentation, Rust-for-Python-tooling, diff/patch/code transformation, GPU kernel synthesis, video games/game engines.

**Real-world domains (Categories 19-26):** Web/HTTP/APIs, scientific computing, cryptography/security, networking/protocols, compilers/language implementation, image/audio/media, embedded/hardware-adjacent, concurrency/async patterns.

**Applied mathematics (Categories 27-36):** Economics/econometrics, ML/statistical learning, physics simulation, chemistry/molecular simulation, biology/bioinformatics, applied math libraries, quantitative finance, control systems/robotics, signal processing/DSP, operations research/optimization.

**Scientific papers (Category 37):** Methods papers converted to markdown via the paper pipeline. Optimization, numerical methods, statistical learning, finance, causal inference foundations.

### Corpus sizing

| Tier | What | Count |
|---|---|---|
| **Full harness** | Repos with validation paths (tests, runnable code, observable behavior) | ~80 repos |
| **Knowledge base only** | C/Rust repos, repos without usable validation, reference material | ~65 repos |
| **Documentation + papers** | Books, guides, converted scientific papers | ~25 sources |
| **Total** | | **~170 sources** |

All enter the air-gapped machine via USB. The harness auto-detects new full-harness repos from their `manifest.json`.

### Corpus expansion protocol

New repos are identified and evaluated periodically. See Section 10 for the search protocol.

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

---

## 10. Corpus Expansion Protocol

**Design record:** `protocols/design_records/corpus_search_protocol.md`

The repo corpus is a living document. New repos are discovered and evaluated periodically to inject fresh reasoning patterns and break training plateaus. This protocol is run by us (human + Claude Code sessions), not by Jane.

### Triggers

- **Opportunistic:** we're in a session and have bandwidth
- **Plateau-triggered:** training metrics stall, need new reasoning patterns
- No fixed calendar

### Process (per run)

1. **Gap analysis first** (5 min) — scan `repo-corpus.md` category counts. Identify the 2-3 thinnest categories or domains with no coverage.

2. **Targeted search** — for each gap, in priority order:
   - Awesome lists for the domain
   - Papers-with-code filtered by repo active 1+ year post-publication
   - Dependency mining from existing corpus repos
   - GitHub topic/language search
   - Quick scan of trending (5 min max)

3. **Three-gate evaluation** — in order, all must pass:

   **Gate 1 — Domain transparency:** Does this domain's methodology practice transparency about its reasoning limitations? Inherently verifiable domains (math, CS, engineering) pass. Domains with formal hedging methods (epidemiology, Bayesian statistics, experimental physics) pass — they teach reasoning under uncertainty. Domains that obscure limitations, make unfalsifiable claims, or use circular reasoning are rejected. This is the project's core principle ("Transparency Is Load-Bearing") applied to domain selection.

   **Gate 2 — Implementation quality (high bar):** Clean code. Decent commit history. Documentation that explains the why. Actively maintained or feature-complete. License doesn't matter (air-gapped, nothing leaves).

   **Gate 3 — Novelty (low bar):** "Could Jane learn something from this that she couldn't learn from what we already have?" Even marginal variance counts — same reasoning shape in a different domain context is valuable. Only reject if exact same reasoning, same domain, same approach as an existing corpus repo. Variance compounds; hard category boundaries between reasoning patterns are not needed.

4. **Tier assignment:**
   - Has any validation path (tests, runnable code, observable output) → Full harness
   - Clean code but no validation path → KB only
   - Paper/documentation → Convert with opendataloader-pdf, index

5. **Record:**
   - Accepted → add to `repo-corpus.md` with category, license, tier, one-line reasoning-pattern note
   - Rejected → add to "Rejected" section at bottom of `repo-corpus.md` with one-line reason
   - Papers → queue for opendataloader-pdf conversion

### Scientific papers and document conversion

**PDF conversion tool:** `opendataloader-pdf` (MPL-2.0) — rule-based PDF → markdown/JSON. Local, no GPU, deterministic (XY-Cut++ layout analysis). Table detection, heading hierarchy, AI safety filters. Serves both knowledge base population and training data extraction.

Run on:
- Methods papers accompanying newly added repos
- Foundational papers in domains where the corpus has code but no theory
- Recent survey papers that reference techniques used across multiple corpus repos
- PDF documentation bundled with corpus repos (API docs, architecture guides, specifications)
