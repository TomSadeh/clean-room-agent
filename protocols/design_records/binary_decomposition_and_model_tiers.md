# Binary Decomposition and Model Tier Architecture

Date: 2026-02-27

## Decision

1. Every pipeline judgment that can be reliably reduced to a binary decision should be.
2. Binary decisions go to the smallest model that can handle them (0.6B).
3. Code generation uses Qwen3-1.7B as the base for LoRA fine-tuning.
4. Three-tier model cascade: 0.6B (binary classification) → 1.7B (code generation + structured classification) → 4B (reasoning/planning).

## Why Binary Decomposition

### Information-theoretic argument

A binary decision is 1 bit of information. A 0.6B model producing 1 bit reliably is a fundamentally easier problem than a 4B model producing a structured multi-file classification in one call. The output space is minimal, the error mode is simple (wrong bit), and the task complexity matches the model's capacity.

The current batch judgment pattern sends groups of candidates to one LLM call and asks for a structured JSON response classifying all of them. The alternative: N independent binary calls to a 0.6B, each asking "is this file relevant to this task? yes/no." This:

- **Parallelizes trivially.** Independent calls, no sequential dependency.
- **Errors are independent.** A wrong judgment on file 7 doesn't contaminate the judgment on file 12. In a batched call, the model's reasoning about one candidate influences all subsequent candidates — errors correlate within the batch.
- **Each call has the simplest possible output space.** The model can't produce a malformed JSON structure if the output is one token.

### Auditability argument

A binary decision is the atomic unit of auditability. It sits at the far left of the auditability U-curve — fully deterministic output format, trivially verifiable.

- **Human review**: A reviewer can check a binary decision in seconds. Input, yes/no, done.
- **Automated metrics**: The audit protocol computes must-contain recall — that's counting binary decisions (file present or not). No ambiguity.
- **Error attribution**: A binary decision that was wrong points to exactly one cause: this input, this label, wrong. A multi-class batch error is ambiguous — which part of the prompt caused which misclassification?
- **Training signal**: A wrong binary decision is a clean training pair. Flip the label. A wrong batch classification requires decomposing which elements were wrong and why, which may not be possible without re-running the call in isolation.

Compare auditing a batched JSON response where the model classified 15 files simultaneously with interleaved reasoning. Which classification was wrong? Did the batch size exceed the model's effective attention span? Was the ordering within the batch influential? None of these questions arise with binary decisions.

### Composability

Binary decisions compose upward: recall = mean of binary presence checks. Precision = mean of binary relevance checks. Every aggregate metric decomposes cleanly into its constituent binary decisions.

They don't decompose downward: a multi-class classification in a batch is a black box. You can measure the batch's aggregate accuracy but you can't attribute errors to individual decisions within it without isolation testing.

## Why Qwen3-1.7B for Code Generation

### Performance parity at half the parameters

Qwen3-1.7B matches Qwen2.5-3B across most benchmarks (see `research_reviews/qwen3_small_models.md`). EvalPlus 52.7, MBPP 55.4. The 2x efficiency gain from Qwen3's training (36T tokens, three-stage pipeline) means the 1.7B has absorbed more knowledge per parameter than the 2.5-era 3B.

### No negative transfer

The `single_generalist_model.md` design record established that code-specialized models have Python priors that actively interfere with C code generation. Qwen2.5-Coder-3B is Python-heavy by training distribution. Qwen3-1.7B is a generalist — clean surface for LoRA specialization on C without fighting pre-existing Python habits.

### Trainable on available hardware

With 2x RTX 5090: LoRA fine-tuning the 1.7B takes ~1-2 days. Compare ~7-8 days for 4B. This means:
- Phase 4 self-improvement loop iterates 4-5x faster.
- More experiments per unit time = faster convergence.
- Can afford to discard and retrain (the "pension" mechanism from population evolution) without catastrophic time cost.

### Thinking mode available

Qwen3-1.7B supports hybrid thinking/non-thinking mode. For code generation steps where chain-of-thought helps (complex edits, multi-file reasoning), thinking mode is available. For simple edits, non-thinking mode saves tokens. This is configurable per-call, no architecture changes needed.

## Three-Tier Model Cascade

| Tier | Model | Role | Task type |
|---|---|---|---|
| 0 | Qwen3-0.6B | Binary classifier | File relevance (yes/no), simple routing |
| 1 | Qwen3-1.7B | Code generator + structured classifier | Code edits, symbol classification, plan steps |
| 2 | Qwen3-4B | Reasoning/planning | Meta-plan decomposition, complex judgment, teacher distillation |

The cascade is **volume-inverted**: tier 0 handles the most calls (50+ file judgments per task), tier 1 handles moderate calls (plan steps, implementations), tier 2 handles few calls (meta-plan, adjustment). Compute cost per task decreases because the expensive model is used least.

This requires zero code changes. The ModelRouter already supports per-stage model resolution via config.toml:

```toml
[models.stages.scope_judgment]
model = "qwen3:0.6b"

[models.stages.precision]
model = "qwen3:1.7b"

[models.coding]
model = "qwen3:1.7b"

[models.reasoning]
model = "qwen3:4b"
```

## Implications for Phase 4

### Faster self-improvement loop

The 1.7B trains in 1-2 days. The 0.6B trains in hours. This means:
- LoRA adapters for binary classifiers can be iterated daily.
- Code generation adapters iterate on 2-day cycles.
- The "population evolution" architecture benefits enormously — more generations per unit time.

### Cleaner training signal

Binary decisions produce clean training pairs (input, correct_label). No need to extract training signal from ambiguous multi-class batch outputs. The raw DB logs every binary decision with full I/O — each is a potential training example with unambiguous ground truth once validated.

### Distillation path

Teacher (4B) → student (1.7B for code, 0.6B for classification) is a natural distillation cascade. The teacher produces high-quality judgments during planning; the student models are fine-tuned to replicate those judgments on their specific narrow tasks.

## Risks

1. **0.6B may not be reliable enough for binary classification.** The quantization sensitivity data shows Q4 degrades meaningfully at 0.6B. If the binary classifier's error rate is too high even after fine-tuning, the tier 0 model may need to be 1.7B, collapsing to two tiers.
2. **1.7B may be too small for complex C code generation.** The code benchmarks are Python-heavy. C code generation (manual memory management, pointer arithmetic, undefined behavior awareness) may require more capacity. This will be tested empirically during Phase 4 bootstrapping.
3. **Thinking mode token overhead.** At 1.7B, thinking tokens compete with context tokens in a 32K window. If the model generates 2K thinking tokens per call, that's ~6% of the context budget per LLM call. Need to measure empirically and potentially disable thinking for simple classification calls.
