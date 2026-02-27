# Documentation Audit: Model Architecture & Training Strategy Update

## Context for Future Claude

Three architectural shifts have converged that make large portions of the documentation suite outdated. This audit updates all affected docs to reflect current reality.

### Shift 1: Two Models → Single 1.7B

The original architecture specified two base models:
- Qwen2.5-Coder-3B for coding (pattern matching)
- Qwen3-4B for reasoning/planning (harder cognitive tasks)

This collapsed first to a single generalist model (design record: `single_generalist_model.md`), then to Qwen3-1.7B which benchmarks as a straight upgrade to the 3B coder. The 4B model is now largely redundant — planning decomposition (Shift 2) reduced the cognitive complexity of each LLM call to the point where 1.7B handles it.

The three-tier cascade in `binary_decomposition_and_model_tiers.md` (0.6B / 1.7B / 4B) should be revised: the 4B tier may be eliminated entirely, leaving 0.6B for cheap binary classification and 1.7B for everything else (structured outputs, code generation, remaining binary judgments).

### Shift 2: Planning Decomposition

The monolithic planning prompts (meta-plan: 6+ cognitive tasks; part-plan: 5 tasks) have been decomposed into atomic sub-tasks:

**Meta-plan decomposition:**
1. Change point enumeration (structured JSON) — "What files/symbols need to change?"
2. Part grouping (structured JSON) — "Group these change points into logical parts"
3. Part dependency (binary yes/no per pair) — "Does part B depend on part A?"

**Part-plan decomposition:**
1. Symbol targeting (structured JSON) — "Which symbols need modification for this part?"
2. Step design (structured JSON) — "Design implementation steps for these targets"
3. Step dependency (binary yes/no per pair) — "Does step B depend on step A?"

This uses the same `run_binary_judgment()` infrastructure as retrieval classification. The binary dependency calls dominate volume (N*(N-1) per plan level). Each call is the same format as a scope/precision classification.

Key files: `src/clean_room_agent/execute/decomposed_plan.py` (new module), config flag `orchestrator.decomposed_planning`.

### Shift 3: Training Data Economics Resolved

The original problem: classification data grew multiplicatively per task, planning data grew linearly (1:1). The planning model needed the most training but generated the least data.

After decomposition, a single task with 3 parts and 4 steps/part generates ~50 LLM calls instead of 4. The binary dependency calls are structurally identical to retrieval classification calls. Planning and retrieval now produce the same data format (binary judgment) at similar volumes.

Additional insight: the classifier can self-train on its own outputs filtered by downstream task success (outcome-based SFT). The audit protocol's reference tasks with ground truth (must_contain/must_exclude) provide exact verification.

This means:
- No training data scarcity for any pipeline stage
- One unified binary judgment training format across all stages
- Self-distillation viable for classification (abundant data + verification signal)
- Planning self-improvement timeline shortened from "months" to "weeks"
- Sub-6B self-improvement guardrails need revision (1.7B may self-improve successfully given the decomposition)

---

## Files to Audit

Organized by priority. Update each file to reflect the three shifts above. Do NOT rewrite files wholesale — make surgical updates that correct outdated claims while preserving the document's structure and purpose.

### Priority 1: Core Architecture (must be consistent)

1. **`CLAUDE.md`** — Lines referencing:
   - "two base models: Qwen2.5-Coder-3B for coding, Qwen3-4B for reasoning/planning" → single Qwen3-1.7B (+ optional 0.6B for binary classification)
   - "Multi-Model Architecture" section → simplify or rename
   - ModelRouter description → simplify (role-based routing may be unnecessary with one model)
   - "sub-7B scale" / "sub-6B fails to bootstrap" guardrails → revisit given decomposition
   - Phase 4 LoRA description → update base model references

2. **`README.md`** — Line 17 references "Qwen2.5-Coder-3B-Instruct for coding and Qwen3-4B-Instruct-2507 for reasoning/planning"

3. **`planning/meta-plan.md`** — Section 3 has explicit model assignment table (Qwen2.5-Coder-3B vs Qwen3-4B). Budget section references model-specific context windows. Section 16 LoRA integration references both models.

4. **`planning/training-strategy.md`** (~90KB, the big one) — Per-stage LoRA targets reference both base models. Training pair specs, techniques, rank recommendations all model-specific. Distillation strategy references Qwen3.5-397B as teacher for two student models. Self-improvement guardrails section needs revision. Compute estimates need updating (1.7B trains in 1-2 days, not 4B at ~4 months).

5. **`planning/phase4-guidelines.md`** (26 pages) — Stream B adapter tiers reference both base models explicitly. Training framework sections, cold-start datasets, self-improvement guardrails all need updating. The three-tier cascade and compute estimates need revision.

### Priority 2: Design Records (architectural decisions)

6. **`protocols/design_records/single_generalist_model.md`** — Decided on Qwen3-4B as single generalist. Now superseded by 1.7B. Update the decision and rationale (the negative transfer argument still holds; the model size reduction is the new development).

7. **`protocols/design_records/binary_decomposition_and_model_tiers.md`** — Three-tier cascade (0.6B / 1.7B / 4B). The 4B tier may be eliminated. Update to reflect planning decomposition making Tier 2 redundant.

8. **`protocols/design_records/infrastructure_self_improvement.md`** — Validation gauntlet and training data hygiene sections may reference model-specific assumptions.

9. **`protocols/design_records/per_task_adapter_map.md`** — Per-stage adapter tracking assumes multi-model architecture.

### Priority 3: Planning & Pipeline Docs

10. **`planning/pipeline-and-modes.md`** — Section 3.3 ModelRouter, model context windows, Phase 4 vLLM migration. Simplify routing if single model.

11. **`planning/cli-and-config.md`** — `[models]` config section with coding_model/reasoning_model entries. May simplify to single model entry.

12. **`planning/phase3-implementation.md`** — Line 90: "Qwen2.5-Coder-3B codes reliably given small enough tasks. The planner (Qwen3-4B) uses the full N-prompt pipeline to decompose work." Also orchestration loop references model assignments.

13. **`planning/schemas.md`** — DDL for Phase 4 training tables may reference model-specific columns or assumptions.

14. **`planning/binary-decomposition-plan.md`** — References three-tier model cascade.

15. **`planning/scaffold-then-implement-plan.md`** — May reference model assignments for plan vs implement.

### Priority 4: Research Docs (update conclusions, preserve data)

16. **`research/qwen3_small_models.md`** — Benchmark data is still valid. Update the "Implications for Jane" section to reflect that 1.7B is now the primary (not just a "fast filter"), and 4B is likely eliminated.

17. **`research/qwen3_4b_2507_comperhensive_finetuning.md`** — Fine-tuning analysis for a model that may no longer be used. Add a note at top pointing to the architectural decision, but preserve the research.

18. **`research/distillation_pipeline_report_v2.md`** — Teacher→student distillation targets need updating (teacher→1.7B, not teacher→4B).

19. **`research/fine_tuning_lora_stacking.md`** — LoRA composition may simplify with single base model.

20. **`research/lora_training_git_data.md`** — Training data generation references may assume two models.

21. **`research/training_on_git_research.md`** — Same as above.

22. **`research/c_training_infrastructure.md`** — Compute estimates reference 3B and 4B models. Update to 1.7B.

### Priority 5: Infrastructure (compute estimates)

23. **`infrastructure/ML_Workstation_Build_Spec.md`** — Training compute estimates for 3B and 4B models. Update to 1.7B (and 0.6B if retained). VRAM estimates change. Max model size considerations change.

24. **`infrastructure/linux_ml_environment.md`** — May reference model-specific CUDA/driver requirements.

### Priority 6: Other References

25. **`references/repo-corpus.md`** — Training data catalog. The corpus itself doesn't change, but any notes about which models consume which data may need updating.

26. **`protocols/governance/evolutionary_economics_of_self_improvement.md`** — Population evolution framework. If model assumptions are embedded, update them.

27. **`protocols/enforcement/transparency-enforcement-prompt.md`** — May reference model capabilities or limitations.

---

## Audit Rules

1. **Preserve document structure.** Make targeted edits, not rewrites. Each doc has a purpose and audience — don't disrupt that.

2. **Research docs: preserve data, update conclusions.** Benchmark numbers, experimental results, and literature findings are permanent. Only update the "implications" or "conclusions" sections that interpret the data for our architecture.

3. **Archived docs: do not modify.** Files in `archive/` are historical records. Leave them as-is.

4. **Design records: supersede, don't delete.** If a decision is superseded, add a "Status: Superseded by [new decision]" header and brief note. Keep the original rationale — it documents the reasoning chain.

5. **Flag uncertainties.** If the 4B model's elimination is not yet confirmed (user said "almost redundant"), use conditional language: "likely eliminated" or "under evaluation." Don't state as fact what's still being validated.

6. **Update MEMORY.md last.** After all docs are updated, update the memory file to reflect current architecture.

7. **Cross-reference consistency.** After individual updates, verify that CLAUDE.md, meta-plan.md, and training-strategy.md all agree on: base model(s), model sizes, training timelines, LoRA targets, and self-improvement guardrails.

8. **Read before editing.** Read each file fully before making changes. Understand the document's role and audience. Some files are authoritative specs, others are research notes, others are implementation guides — edit accordingly.

---

## Verification Checklist

After completing all edits, verify:

- [ ] No remaining references to "Qwen2.5-Coder-3B" as an active model (only in historical/research context)
- [ ] No remaining references to "two base models" or "dual model" as current architecture
- [ ] All compute estimates updated from 4B/3B timelines to 1.7B timelines
- [ ] Self-improvement guardrails updated to account for planning decomposition
- [ ] Training strategy reflects unified binary judgment format across all stages
- [ ] ModelRouter documentation reflects simplified (or eliminated) routing
- [ ] Config documentation reflects simplified [models] section
- [ ] Planning decomposition mentioned in relevant architecture sections
- [ ] Training data economics (multiplicative binary data) documented in training strategy
- [ ] Classifier self-distillation pathway documented
- [ ] All design records have correct Status fields
- [ ] MEMORY.md updated to reflect current architecture
- [ ] No contradictions between CLAUDE.md, meta-plan.md, and training-strategy.md
