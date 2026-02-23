# Phase 4: Validation + Benchmarking

## Context

This phase validates the built system from Phases 1-3. It measures retrieval quality and solve performance, and runs thesis-level comparisons.

Phase 4 does not build core solve-path features. It evaluates them.

---

## Inputs From Previous Phases

- Phase 1: Curated DB (indexed knowledge base), raw DB (indexing run metadata), connection factory.
- Phase 2: Retrieval pipeline + context packages. Raw DB retrieval decisions. Session DB retrieval state.
- Phase 3: Solve harness (`cra solve`). Raw DB attempt records, task results, validation results, LLM outputs.

---

## Database Interaction Model

| DB | Access | Purpose |
|----|--------|---------|
| Curated | **Never touches** | Phase 4 does not modify or directly query the curated DB |
| Raw | **Read-only** | Primary data source for all analysis: retrieval decisions, attempt records, task results, validation results, LLM outputs |
| Session | **Never touches** | Reads archived session copies from raw DB if needed |

Phase 4 is a **read-only consumer** of raw DB data. It analyzes logged decisions and results from Phases 1-3 to produce metrics and reports. Raw->curated derivation analysis (identifying which raw signals should be promoted) is a key Phase 4 output that informs future automation.

---

## Validation Scope

- Retrieval quality evaluation (`cra evaluate`) for file/symbol precision-recall and token efficiency.
- SWE-bench integration and execution pipeline.
- Stress-test matrix runs and comparison reporting.
- Reproducibility, checkpointing, and result artifact management.

---

## Project Structure Additions

```
src/
  clean_room/
    bench/
      __init__.py
      swe_bench.py                # SWE-bench instance loading + adaptation
      runner.py                   # Benchmark runner
      metrics.py                  # Score aggregation + comparison
      report.py                   # Report/table generation
      configs.py                  # Stress matrix definitions
tests/
  test_swe_bench.py
  test_runner.py
```

---

## Step 1: Retrieval Validation Harness

**Delivers**:
- `cra evaluate` wired to Phase 2 retrieval outputs.
- File/symbol precision/recall, token efficiency, budget utilization metrics.

**Data source**: Reads logged retrieval decisions from **raw DB** (not re-running retrieval). Compares logged decisions against ground truth to compute precision/recall without requiring live retrieval runs.

**Gate**:
- Evaluation cases run consistently and metrics are emitted per-case and aggregate.

---

## Step 2: SWE-bench Integration

**Delivers**:
- SWE-bench instance loading/prep utilities.
- Prediction artifact generation from `solve` outputs.
- Strict separation from gold patch data during solving.

**Requirements**:
- Use SWE-bench conventions (`base_commit`, `problem_statement`, `test_patch`).
- Ensure patch format is compatible with SWE evaluator.

---

## Step 3: Benchmark Runner

**Delivers**:
- `cra bench` for batched runs across selected configs and instances.
- Resume/checkpoint support.

**Requirements**:
- Deterministic sampling with fixed seed.
- Durable artifact writes after every task run.
- Clear failure categorization (infrastructure vs model vs apply/validation).

---

## Step 4: Metrics + Report

**Delivers**:
- `cra report` with config comparisons.
- Internal solve metrics and official SWE evaluation metrics kept separate.

**Requirements**:
- Distinct metric namespaces:
  - Internal: patch-apply success, local validation success, attempts/tokens/latency.
  - Official: SWE evaluator pass rate.
- Comparison tables include exact model IDs and run parameters.

**Raw DB analysis**:
- Aggregate retrieval decision patterns across runs (which signals drove correct inclusions vs false positives).
- Identify derivation candidates: raw signals that consistently predict solve success and should be promoted to curated DB.
- Extract fine-tuning data from raw DB: task descriptions, retrieval decisions, LLM outputs, and success/failure labels for future per-stage LoRA training.

---

## Stress Test Matrix

- Config A: small model + baseline context.
- Config B: small model + pipeline context.
- Config C: larger model + baseline context.
- Config D: larger model + pipeline context.

All comparisons must keep temperature, retry limit, and output format aligned.

---

## Verification (Phase 4 Gate)

```bash
# Retrieval evaluation
cra evaluate --cases eval_cases.json --repo /path/to/repo

# Pilot benchmark run
cra bench --configs A,B,C,D --instances 10 --output results/pilot/

# Report generation
cra report results/pilot/
```

**Gate criteria**:
1. Benchmark runs complete with resumable artifacts.
2. Reports show per-config results with reproducible run metadata.
3. Official SWE evaluation can be executed on produced predictions.
4. Thesis comparisons (B vs A, D vs C, B vs C) are reported from evaluation outputs, not build-phase assumptions.

---

## Notes

- Phase 4 owns thesis validation.
- Phase 3 completion is a prerequisite, not evidence of benchmark superiority.
- Phase 4 is a **read-only consumer** - it never writes to any DB. All data comes from raw DB records logged by Phases 1-3.
- Raw->curated derivation is analyzed but **not automated** in Phase 4. Phase 4 identifies what should be promoted; the promotion mechanism is a post-Phase 4 decision.
- Fine-tuning data extraction from raw DB is a Phase 4 deliverable, feeding into the long-term per-stage LoRA adapter goal.

