# Phase 4: Validation + Benchmarking

## Context

This phase validates the built system from Phases 1-3. It measures retrieval quality and solve performance, and runs thesis-level comparisons.

Phase 4 does not build core solve-path features. It evaluates them.

---

## Inputs From Previous Phases

- Phase 1: Indexing + knowledge base.
- Phase 2: Retrieval pipeline + context package generation.
- Phase 3: Solve harness (`cra solve`) with pipeline and baseline modes.

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
