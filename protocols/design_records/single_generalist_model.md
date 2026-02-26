# Single Generalist Model — Design Record

**Date:** 2026-02-26
**Status:** Decided
**Depends on:** `c_language_choice.md`

## Problem Statement

The current architecture specifies two base models: a coding-specialized model (Qwen2.5-Coder-3B) and a generalist reasoning model (Qwen3-4B). This split was justified by the assumption that coding tasks benefit from a coding-specialized model.

The C language decision (see `c_language_choice.md`) invalidates that assumption. The coding model's specialization is overwhelmingly Python/JavaScript — languages that are massively overrepresented in coding benchmarks and training corpora. For C, this specialization provides no advantage. Worse, it actively interferes.

## Decision

Use a single generalist model for both coding and reasoning roles. The current two-role config structure (`coding` and `reasoning` in `[models]`) is retained — both are set to the same model. When Qwen 3.5 generalist models are released, evaluate those as the unified base.

No code changes required. `ModelRouter` already supports identical model strings for both roles.

## Rationale

### Negative transfer from coding specialization

The coding model's weights are saturated with Python idioms: exception handling patterns, dynamic typing assumptions, implicit memory management, `import`-based dependency resolution. These are not neutral knowledge that sits unused when writing C. They are anti-patterns that actively pull the model toward wrong outputs.

A generalist model has weaker priors across the board, but those priors are not pulling in the wrong direction. For C, a blank slate is easier to write on than one with deeply etched wrong answers.

### LoRA efficiency

Fine-tuning a coding-specialized model on C data must first overcome entrenched Python priors before moving weights toward C. Fine-tuning a generalist model on C data moves weights directly toward C — no unlearning phase.

Same rank, same training budget, but more of the gradient goes to useful learning instead of fighting against the base weights. The LoRA efficiency hypothesis from `c_language_choice.md` (underrepresented distributions have more headroom) is strengthened: the generalist model has headroom without interference, while the coding model has headroom blocked by negative transfer.

### Simplification

One model instead of two: one set of LoRA adapters, one model to serve, one context window size to budget for, simpler routing logic. The two-role distinction remains in config for future flexibility but carries no architectural weight when both roles resolve to the same model.

## What This Does Not Change

- The `ModelRouter` and `[models]` config structure — both roles still exist, both can be set independently if needed in the future.
- The pipeline architecture — stages are model-agnostic by design.
- The per-stage LoRA adapter strategy — adapters target pipeline stages, not model roles.
- The retrieval pipeline, orchestrator, or any other component.

## Validation

- **Pre-training baseline:** Run the same C task N times with identical curated context, once with the coding model and once with the generalist. Compare output quality (correctness, memory safety, idiomatic C). If the generalist matches or beats the coder on C tasks despite weaker benchmark scores overall, the negative transfer hypothesis is confirmed before any fine-tuning investment.
- **Immediate:** Set both `coding` and `reasoning` to the same generalist model in `config.toml`. Run the existing test suite. No failures expected (config-only change).
- **Phase 4:** Compare matched LoRAs trained on the generalist base vs. the coding-specialized base, both fine-tuned on C data. The generalist base should show larger improvement delta per training sample, confirming the negative transfer hypothesis. If it doesn't, revisit.
