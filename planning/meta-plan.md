# Clean Room Agent Meta-Plan

## Purpose

This document defines the top-level execution order and ownership boundaries across phases.

---

## Phase Split

1. **Phase 1: Knowledge Base + Indexer Build**
- Build deterministic indexing, schema, parsers, dependency graph, git metadata, and query API.
- Gate: `cra index` and data/query integrity are stable.

2. **Phase 2: Retrieval Pipeline Build**
- Build task analysis, scoring, scoping, precision extraction, budget enforcement, and `cra retrieve`.
- Gate: retrieval runs end-to-end and output is budget-compliant and actionable.

3. **Phase 3: Agent Harness Build**
- Build `cra solve`: prompting, generation, parse/apply/validate loop, retries, and baseline mode.
- Gate: solve path works end-to-end in both pipeline and baseline modes.

4. **Phase 4: Validation + Benchmarking**
- Run formal evaluation: retrieval metrics, SWE-bench integration, benchmark runner, reporting.
- Gate: reproducible benchmark runs and thesis comparisons from evaluation outputs.

---

## Hard Boundaries

- Phases 1-3 are **build phases**.
- Phase 4 is the **validation phase**.
- Thesis proof claims (e.g., B > A) belong only to Phase 4 outputs.

---

## Dependency Order

Phase 1 -> Phase 2 -> Phase 3 -> Phase 4

No phase should pull validation gates from a later phase.
