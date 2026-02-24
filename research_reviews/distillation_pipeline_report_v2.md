# Distilling Coding Agents into 3-4B Local Models (v2)

**Per-stage LoRA distillation from large open-weight teachers into Qwen3-4B and Qwen2.5-Coder-3B is technically feasible, legally safe, and surprisingly affordable — roughly $60–150 total.** The primary teacher is now **Qwen3.5-397B-A17B** (released Feb 16, 2026), which scores 83.6 on LiveCodeBench v6, 76.4 on SWE-bench Verified, and 91.3 on AIME'26 — a substantial leap over its predecessor Qwen3-235B. All Qwen models remain Apache 2.0. The per-task LoRA architecture has no direct precedent in published work — it would be a genuinely novel contribution — but all architectural primitives (independent LoRA training, mixture-of-LoRA routing) are proven. **Planning distillation is the critical path.** It's the hardest stage to bootstrap, the hardest to evaluate, and the stage where teacher quality matters most. This report gives it proportionate depth.

---

## 1. Teacher Model Selection

### Primary teacher: Qwen3.5-397B-A17B

Qwen3.5 was released February 16, 2026 and represents a generational leap. It's a 397B MoE with only 17B active parameters per forward pass, built on a hybrid architecture combining Gated Delta Networks (linear attention) with standard gated attention. Alibaba claims it outperforms GPT-5.2, Claude Opus 4.5, and Gemini 3 Pro on 80% of evaluated benchmarks — independent verification is still pending as of this writing (one week post-release).

Key benchmarks relevant to the pipeline:

| Benchmark | Qwen3.5-397B | Qwen3-235B (prior) | Delta |
|---|---|---|---|
| LiveCodeBench v6 | 83.6 | ~70-74 | +12-14 pts |
| SWE-bench Verified | 76.4 | not reported | — |
| AIME'26 | 91.3 | 81.4 (AIME'25) | +10 pts |
| GPQA Diamond | 88.4 | 81.1 | +7 pts |
| BFCL-V4 (tool use) | top tier | 70.8 (v3) | improved |
| IFBench (instruction following) | 76.5 | — | — |

**Why this matters for planning distillation specifically:** The model's post-training was heavily focused on agentic workflows — Alibaba scaled RL across "million-agent environments with progressively complex task distributions." The agentic benchmarks (BFCL-V4, VITA-Bench, DeepPlanning, Tool-Decathlon, MCP-Mark) show the strongest gains over Qwen3. This is exactly the capability profile needed for generating high-quality structured change plans.

The model also supports 262K context natively (1M via the hosted Qwen3.5-Plus variant), which matters for plan generation where large code context windows are essential.

### Coding teacher: Qwen3-Coder-Next-80B-A3B

For Stage 5 (code generation targeting Qwen2.5-Coder-3B), the newly released **Qwen3-Coder-Next** is the most interesting teacher. It's an 80B MoE with only 3B active parameters, built on the Qwen3-Next architecture, and achieves over 70% on SWE-bench Verified using SWE-Agent scaffolding. It was specifically trained with agentic signals — large-scale executable task synthesis, environment interaction, and RL from environment feedback. It sits on a strong Pareto frontier for cost-effective agent deployment, matching models with 10-20× more active parameters.

However, as you noted, Stage 5 is not the focus. The recommended approach there is straightforward SFT distillation from either Qwen3-Coder-Next or Qwen2.5-Coder-32B, supplemented with the NVIDIA OpenCodeInstruct dataset (5M pairs, CC BY 4.0, tested at 3B+).

### Architecture mismatch note

Qwen3.5 uses a fundamentally different architecture from Qwen3 — Gated DeltaNet layers, 512 total experts (10 routed + 1 shared per token), and a **250K vocabulary** (vs Qwen3's 150K). This means **tokenizers are incompatible** — logit-level distillation from Qwen3.5 into Qwen3-4B is not possible. For response-level SFT distillation (the primary approach), this doesn't matter — you're training on text outputs, not logits. But it does close the door on the one technique where same-family matching was a real advantage.

If logit distillation proves necessary (e.g., for stabilizing Stage 4 per the Ministral 3B finding), the fallback is **Qwen3-235B-A22B**, which shares the same tokenizer as your Qwen3-4B target.

### Updated teacher assignment

| Pipeline stage | Target model | Primary teacher | Secondary/fallback | Rationale |
|---|---|---|---|---|
| 1 — Task Analysis | Qwen3-4B | Qwen3.5-397B (thinking mode) | — | Best structured output; strong tool use scores |
| 2 — Relevance | Qwen3-4B | Qwen3.5-397B (non-thinking) | — | Classification; concise output needed |
| 3 — Symbol Selection | Qwen3-4B | Qwen3.5-397B (non-thinking) | Qwen2.5-Coder-32B for code-specific knowledge | Code comprehension depth |
| 4 — Plan Generation | Qwen3-4B | **Qwen3.5-397B (thinking mode)** | DeepSeek-V3.2 for cross-teacher filtering | **Critical stage — see Section 3** |
| 5 — Code Generation | Qwen2.5-Coder-3B | Qwen3-Coder-Next or Qwen2.5-Coder-32B | OpenCodeInstruct dataset | Not the focus of this report |

---

## 2. Licensing: Still Clean

Qwen3.5 is Apache 2.0, confirmed on the GitHub repo. No change from Qwen3. The full licensing landscape from v1 of this report remains accurate:

| Model family | License | Distillation status |
|---|---|---|
| **Qwen2.5 / Qwen3 / Qwen3.5** | Apache 2.0 | ✅ Fully permitted — no restrictions on outputs or derivatives |
| **DeepSeek R1 / V3.x** | MIT | ✅ Explicitly permitted — README says "including distillation" |
| **Llama 3.x** | Community | ✅ With conditions — naming, 700M MAU cap |
| **Mistral Large 3** | Apache 2.0 | ✅ Fully permitted |

**Qwen→Qwen distillation remains unambiguously legal.** Apache 2.0 grants perpetual, worldwide, royalty-free rights to reproduce, prepare derivative works, sublicense, and distribute. No output restrictions. Alibaba's own Qwen3.5 release used strong-to-weak distillation internally.

API cost: Qwen3.5-Plus via Alibaba Cloud Model Studio is priced at ~$0.11/M input tokens — absurdly cheap. Together AI and DeepInfra don't host Qwen3.5 yet (one week old), but will likely follow. DeepSeek V3.2 official API remains the cheapest secondary option.

---

## 3. Planning Distillation: The Critical Path

This is the hardest problem in the pipeline. Planning is hard to distill for three interconnected reasons that don't apply to the other stages:

**Problem 1: No ground truth.** For code generation, you can run tests. For classification stages, you can verify structure. For plan generation, there's no execution-based signal — a plan's quality is only revealed when downstream code generation succeeds or fails, and even then attribution is ambiguous. Did the plan fail, or did the code generator fail to execute a good plan?

**Problem 2: High-dimensional output space.** A plan isn't a classification or a code snippet. It's a structured document specifying which files to change, what changes to make, in what order, with what rationale. The space of "correct" plans for a given task is huge, and many valid plans look completely different from each other. This makes SFT on a single teacher output more fragile than for constrained-output stages.

**Problem 3: Reasoning depth at 4B.** The Ministral 3B team (Mistral, January 2026) found that at 3B scale, vanilla SFT on reasoning traces produced brittle, verbose models. Logit distillation from a teacher was necessary to stabilize training before RL could be applied. At 4B you have slightly more headroom, but this finding is a warning flag for Stage 4 specifically.

### 3.1 Plan representation matters enormously

Before generating any training data, the plan schema itself needs careful design. The representation determines what the student model learns — and at 4B, representational clarity is load-bearing.

**Recommended plan schema:**

```json
{
  "task_summary": "Brief restatement of what needs to change and why",
  "affected_files": [
    {
      "path": "src/module/handler.py",
      "role": "primary",  // primary | supporting | type_stub
      "changes": [
        {
          "symbol": "class RequestHandler.process()",
          "action": "modify",  // modify | add | delete | rename
          "description": "Add timeout parameter, propagate to downstream calls",
          "depends_on": [],
          "depended_by": ["src/module/client.py:Client.send()"]
        }
      ]
    }
  ],
  "execution_order": ["handler.py", "client.py", "tests/test_handler.py"],
  "rationale": "Change propagates from handler→client due to signature change"
}
```

Key design principles:

- **Explicit dependency edges.** The `depends_on` / `depended_by` fields force the teacher (and eventually the student) to reason about change propagation — the core difficulty in repository-level planning. Microsoft's CodePlan (FSE 2024) showed that framing repo-level coding as a planning problem with dependency analysis enabled 5/7 repositories to pass validity checks, while baselines without planning couldn't pass any.

- **Structured but not over-specified.** The plan shouldn't contain actual code — that's Stage 5's job. It should specify *what* changes at *what* symbols in *what* order. Over-specification (including code in the plan) couples stages and makes the plan harder to learn.

- **Code-form reasoning before the plan.** The CodePlan paper (ICLR 2025, not the Microsoft FSE one) showed that generating pseudocode-style reasoning plans as an intermediate step improved multi-step reasoning by 25.1% averaged across 13 benchmarks. For Stage 4, the teacher should generate a CoT reasoning trace *before* the structured plan, and the student should be trained on both.

### 3.2 Generating high-quality plan training data

The data generation pipeline for plans is more complex than for the other stages because of the ground-truth problem. Here's the multi-stage approach:

**Step 1: Source diverse real planning problems.**

Extract real code change tasks from:
- Your indexed repositories: git commits that touch 2+ files, filtered for meaningful changes (not just formatting)
- CommitPackFT (702K real-world code change pairs, open license)
- SWE-bench instances (2,294 real GitHub issues with validated patches)

For each, construct: task description + pre-change codebase state + relevant file/symbol metadata. This is the input to Stage 4.

**Step 2: Generate plans from Qwen3.5-397B in thinking mode.**

Use Qwen3.5's thinking mode (enables extended reasoning before the final answer). The prompt should:
- Provide the task description and full code context
- Request the plan in the exact JSON schema above
- Ask for explicit reasoning about dependency propagation
- Generate N=5 candidate plans per task (temperature 0.8)

**Step 3: Filter plans through downstream execution.**

This is the key insight: **plan quality can be measured indirectly by executing the plan.** For each generated plan:
1. Feed the plan + code context to the teacher model (or a strong code model) as a Stage 5 input
2. Execute the generated code against available test suites
3. Score the plan by the code's test pass rate

Plans that lead to successful code generation are "good plans." Plans where the code fails are either bad plans or suffered from code generation failures — but across N=5 plans per task, the differential signal is informative.

**Step 4: Cross-teacher agreement filtering.**

Generate plans from both Qwen3.5 and DeepSeek-V3.2 (MIT license). Plans where both teachers agree on the same files, symbols, and execution order are higher confidence. Disagreements flag either ambiguous tasks (potentially valuable hard examples) or teacher errors (discard).

**Step 5: Construct preference pairs for DPO.**

From the N=5 candidates per task, create preference pairs:
- **Chosen**: plan that led to highest test pass rate
- **Rejected**: plan that led to lowest test pass rate (but was structurally valid)

This gives you DPO training data "for free" from the same generation run.

### 3.3 Structured Agent Distillation for plans

The Structured Agent Distillation (SAD) framework (arxiv 2505.13820, September 2025) is directly relevant. SAD segments teacher trajectories into `[REASON]` and `[ACT]` spans and applies segment-specific losses to each. On ALFWorld, HotPotQA, and WebShop, SAD consistently outperformed token-level and imitation learning baselines with minimal performance drop during compression.

For Stage 4, adapt SAD's approach:
- Segment the plan output into `[REASONING]` (the CoT trace explaining *why* these changes are needed) and `[PLAN]` (the structured JSON output)
- Apply higher loss weight to the `[PLAN]` span — the structural output matters more than the exact wording of the reasoning
- Apply a curriculum: train first on single-file plans (easy), then multi-file with simple dependency chains, then complex multi-file with branching dependencies

The Routine framework (arxiv 2507.14447, July 2025) provides additional validation. They used structured planning templates to guide multi-step tool execution in enterprise scenarios. After training Qwen3-14B on just 4,209 planning examples, they achieved 88.2% accuracy. With domain-specific distillation on only 537 samples, a student model reached 95.5% accuracy — approaching GPT-4o's 96.3%. This suggests that **structured plan formats dramatically reduce the number of examples needed** because the format constrains the output space.

### 3.4 Plan-aware training configuration

For Stage 4 specifically, the training setup differs from the other stages:

**Phase 1 — CoT-SFT (primary):**
- Train on (task + context, reasoning_trace + structured_plan) pairs
- Higher rank LoRA (128) to capture planning reasoning
- 8K-15K high-quality examples (filtered through execution)
- 3 epochs, learning rate 1.5e-4

**Phase 2 — DPO on execution-ranked plans:**
- Use preference pairs from Step 5 above
- Smaller learning rate (5e-5)
- 1 epoch
- This teaches the model to discriminate between plans that work and plans that don't

**Phase 3 (if SFT proves brittle) — Logit distillation stabilization:**
- Fall back to Qwen3-235B (same tokenizer as Qwen3-4B)
- Use TAID or GKD for on-policy logit alignment
- This was the technique that stabilized Ministral 3B

**Evaluation protocol for plans:**

This is the gap that requires custom tooling. No standard benchmark exists for evaluating plan quality in your specific pipeline. Build a custom eval set:

1. **Structural validity** (automated): Does the plan parse? Does it reference real files and symbols from the context? Are dependency edges consistent?
2. **Downstream execution** (automated): Feed the plan to your Stage 5 model. Do the generated edits pass tests?
3. **Plan-ground-truth overlap** (automated, for tasks with known solutions): For SWE-bench tasks, compare plan's affected files/symbols against the ground truth patch. Metrics: file recall, symbol recall, ordering consistency.
4. **LLM-as-judge** (semi-automated): Have Qwen3.5 score the student's plan on completeness, ordering correctness, and rationale quality. Calibrate against downstream execution results.

Target: ≥70% downstream execution pass rate on held-out tasks. Below 50% means the plan LoRA needs more data or a different training approach.

### 3.5 The plan quality ceiling problem

Your hypothesis is correct: planning benefits most from distillation because it's the stage with the least available signal for self-improvement. But this also means it's the stage most bounded by teacher quality.

Evidence on capability ceilings:
- "Distilling Step-by-Step" (Hsieh et al., ACL 2023) showed a 770M T5 outperforming 540B PaLM via rationale distillation — a 700× compression. But this was on tasks with clear correct answers, not open-ended planning.
- GKD's self-distillation experiments showed student surpassing teacher on GSM8K — but GSM8K has verifiable answers.
- For planning specifically, there's no published evidence of a distilled student exceeding the teacher, because there's no agreed-upon benchmark for plan quality.

**Practical ceiling estimate:** Expect your 4B planner to reach 60-80% of Qwen3.5's planning quality on initial distillation, improvable to 70-85% with DPO and execution feedback. Beyond that, you'd need either a better teacher, domain-specific RL with execution rewards, or human feedback.

The Qwen3.5 upgrade is particularly valuable here because it pushes the teacher ceiling higher — the agentic RL training Alibaba did is exactly the kind of capability that's hardest to reproduce through distillation and most valuable to inherit.

---

## 4. Distillation Techniques Matched to Each Stage

### Stages 1-3 (Classification/Extraction → Qwen3-4B): Standard SFT

These stages have clear correct outputs and constrained output spaces. Standard SFT on teacher-generated input-output pairs is the most efficient approach. Adding DPO or CoT to classification tasks increases inference latency without proportional quality gains.

One exception: for Stage 3 (symbol selection), a curriculum ordering from easy symbols to ambiguous borderline cases helps the student learn decision boundaries.

| Stage | Technique | Examples needed | Training time (24GB GPU) |
|---|---|---|---|
| 1 — Task Analysis | SFT | 3K-5K | ~4 hrs |
| 2 — Relevance | SFT | 2K-4K | ~3 hrs |
| 3 — Symbol Selection | SFT + curriculum | 3K-5K | ~4 hrs |

### Stage 4 (Plan Generation → Qwen3-4B): CoT-SFT + DPO + SAD

See Section 3 above. This is a multi-phase process:

1. CoT-SFT on execution-filtered teacher plans (8K-15K examples, ~8-12 hrs)
2. DPO on execution-ranked preference pairs (1 epoch, ~4 hrs)
3. Optional logit stabilization via Qwen3-235B if brittle

### Stage 5 (Code Generation → Qwen2.5-Coder-3B): SFT + DPO

Not the focus. Standard SFT on teacher code + DPO with execution-ranked preferences. NVIDIA OpenCodeInstruct provides 5M pre-made pairs. The Qwen3-Coder-Next-80B-A3B is the strongest available teacher for this stage.

### Key papers on distillation techniques

| Paper | Venue | Key result |
|---|---|---|
| **Structured Agent Distillation** | arxiv 2505.13820 | Segment-specific [REASON]/[ACT] losses outperform token-level distillation for agent trajectories |
| **Routine** | arxiv 2507.14447 | 537 structured plan samples → 95.5% accuracy (near GPT-4o's 96.3%) |
| **CodePlan** (ICLR 2025) | ICLR 2025 | Code-form plans improve multi-step reasoning 25.1% across 13 benchmarks |
| **CodePlan** (Microsoft FSE 2024) | FSE 2024 | Dependency-aware planning: 5/7 repos pass validity, 0/7 without planning |
| DeepSeek-R1 distillation | arxiv 2501.12948 | SFT on 800K CoT traces: distillation > RL for small models |
| TAID | ICLR 2025 Spotlight | Outperforms all prior KD methods; 2-10× faster |
| GKD | ICLR 2024 | On-policy distillation; 6× over supervised KD |
| Zephyr | NeurIPS 2023 | dSFT+dDPO: 7B outperforms 70B Llama2-Chat |
| MiniLLM | ICLR 2024 | Reverse KLD prevents mode-averaging; student can exceed teacher Rouge-L |
| Distilling Step-by-Step | ACL 2023 | 770M outperforms 540B PaLM via rationale distillation |

---

## 5. Cost Estimate (Updated for Qwen3.5)

Assuming 50K prompts, 5× overgeneration for Stage 4 (more plans needed for preference pair construction), 3× for other stages. ~500 input + ~800 output tokens per pair on average.

| Component | Cost | Notes |
|---|---|---|
| **Qwen3.5-Plus API** (Stages 1-4) | **$20-60** | ~$0.11/M input, ~$0.44/M output; absurdly cheap |
| **DeepSeek V3.2 API** (Stage 4 secondary) | **$10-30** | Cross-teacher filtering; cache hits reduce cost |
| **Qwen3-Coder-Next or Qwen2.5-Coder-32B** (Stage 5) | **$20-50** | Via DeepInfra or similar |
| **Local training** (all 5 LoRAs) | **$0** | Your 24GB GPU; ~35-40 hrs total |
| **Total** | **$50-140** | |

The Qwen3.5 pricing is so aggressive that the entire teacher inference budget is dominated by the code generation stage, not the planning stage. This means you can afford much higher overgeneration for Stage 4 — generate 10× candidates per task instead of 3×, giving you richer preference pairs.

---

## 6. Existing Precedents

**No published project has trained separate LoRA adapters for different coding pipeline stages and composed them.** This is a genuine gap and a genuinely novel contribution.

Closest precedents:

- **DeepSeek-R1 distilled models**: SFT on 800K CoT traces into Qwen2.5 and Llama bases. Proved cross-family distillation works. No planning-specific distillation.
- **Nanbeige4-3B-Thinking**: 3B model beating 10× larger models on reasoning via Dual-Level Preference Distillation. Proved 3B is sufficient capacity for complex reasoning.
- **Routine (2025)**: Structured planning distillation achieving 95.5% accuracy from just 537 domain-specific samples. Most directly relevant to Stage 4.
- **NVIDIA OpenCodeInstruct**: 5M coding pairs tested at 3B+. Most relevant to Stage 5.
- **TT-LoRA MoE** (SC'25): Independently trained LoRA experts composed via lightweight router. Architectural precedent for per-stage adapter composition.
- **rStar-Coder**: 1.5B model achieving 40.1% on LiveCodeBench (outperforming R1-Distill-7B and GPT-4o) through verified training data. Proved extreme compression is viable with quality data.

---

## 7. Self-Improvement Transition

The transition from teacher-distilled to self-improving is viable but **requires strict guardrails** — especially for Stage 4 where the evaluation signal is weakest.

### Model collapse is the top risk

Shumailov et al. (Nature, 2024) showed recursive training on synthetic data causes consistent loss of diversity. Gerstgrasser et al. (2024) proved that **accumulating data (never replacing it)** bounds error regardless of iteration count.

For planning specifically, the collapse risk is higher than for other stages because:
- Plans are more diverse in valid output space → synthetic data narrows this faster
- No execution-based filter for plan diversity (only for plan correctness)
- At 4B, the model has less capacity to maintain distributional diversity

### Recommended self-improvement timeline

| Phase | Timing | Activity | Focus for Stage 4 |
|---|---|---|---|
| Pure teacher distillation | Weeks 1-4 | CoT-SFT + DPO from teacher data | Establish planning baseline |
| Mixed on-policy | Weeks 4-8 | Student generates plans, teacher scores; 70/30 mix | GKD-style with Qwen3.5 as scorer |
| Execution feedback | Weeks 8-16 | Plans scored by downstream code success | GRPO with test pass rate as reward |
| Production feedback | Ongoing | Log successful production plans; periodic retrain | Never remove original teacher data |

**The planning stage should transition to self-improvement last** — after code generation and classification stages are stable. A bad planner poisons everything downstream.

### When student data becomes more valuable than teacher data

After ~10K production tasks with execution feedback, filtered successful plans (plans that led to code that passed tests) become more valuable than generic teacher plans because they're on-distribution for your specific codebase patterns. But: always keep the original teacher data in the training mix.

Expect 3-5 useful improvement iterations before diminishing returns. SPIN showed most gains in iterations 0-2. STaR loops stall when the model can't solve new problems.

---

## 8. Recommended End-to-End Pipeline

### Step 1: Teacher setup
- Primary: Qwen3.5-Plus via Alibaba Cloud Model Studio (~$0.11/M input)
- Secondary (Stage 4 cross-check): DeepSeek-V3.2 official API
- Coding teacher: Qwen3-Coder-Next or Qwen2.5-Coder-32B via DeepInfra

### Step 2: Synthetic data generation
- Extract real change tasks from indexed repos (git commits touching 2+ files)
- Supplement with CommitPackFT and SWE-bench instances
- Construct per-stage prompts from real codebase context
- Target: 50K diverse prompts, weighted heavily toward Stage 4

### Step 3: Teacher inference
- Qwen3.5-Plus Batch API for all reasoning stages
- **Stage 4: generate N=10 candidate plans per task** (the API is cheap enough)
- Temperature 0.8 for diversity; save all metadata
- For Stage 4: parallel generation from both Qwen3.5 and DeepSeek-V3.2

### Step 4: Filtering (Stage 4 gets special treatment)
**Classification stages (1-3):** Format validation → consistency check → heuristic dedup
**Planning stage (4):**
1. Structural validity check (parses? references real files/symbols?)
2. Cross-teacher agreement scoring (Qwen3.5 vs DeepSeek)
3. **Downstream execution test**: feed each plan to code generation teacher, run tests
4. Rank plans by execution success rate
5. Construct DPO preference pairs from best/worst plans per task
6. Diversity sampling to avoid mode collapse in training data

**Code generation stage (5):** Execution-based filtering (run generated code against tests)

Target: 25K-40K high-quality pairs total, with 8K-15K for Stage 4 specifically.

### Step 5: Per-stage LoRA training
Train 5 independent LoRA adapters using Unsloth on local 24GB GPU with QLoRA.

**Stage 4 specific config:**
- Rank: 128 (higher than other stages)
- Alpha: 256
- Target modules: ALL linear layers
- Phase 1: CoT-SFT, lr=1.5e-4, 3 epochs (~8-12 hrs)
- Phase 2: DPO on execution-ranked preference pairs, lr=5e-5, 1 epoch (~4 hrs)
- Max sequence length: 8192 (plans with reasoning traces are longer)

**Other stages:** Rank 64, lr=2e-4, 2-3 epochs, 4-8 hrs each.

### Step 6: Ollama deployment
Export each LoRA to GGUF via llama.cpp. Separate Ollama model files per stage, or single base model with adapter hot-swapping. Pipeline controller routes inputs to appropriate adapter.

### Step 7: Logging and evaluation
Log every query, output, and execution outcome. Custom eval suite:
- Stages 1-3: structural correctness metrics
- **Stage 4: downstream execution pass rate (primary metric), file/symbol recall vs ground truth (secondary), LLM-as-judge scores (tertiary)**
- Stage 5: test pass rate

### Step 8: Self-improvement (Stage 4 last)
Start self-improvement on Stages 1-3 and 5 first. Only begin Stage 4 self-improvement after:
- Code generation model is stable (reliable execution signal for plan evaluation)
- 10K+ production tasks logged with execution outcomes
- Baseline plan accuracy ≥50% downstream execution pass rate

---

## 9. Gaps Requiring Custom Tooling

| Gap | Impact | Custom work needed |
|---|---|---|
| Per-task LoRA routing at inference | Medium | Python orchestrator for adapter selection per pipeline stage |
| Plan evaluation benchmark | High | Custom eval set from indexed repos with ground truth patches |
| Execution sandbox for plan filtering | High | Docker-based sandbox that takes a plan → generates code → runs tests |
| CoT stripping at inference | Low | Strip reasoning traces before passing plan output to Stage 5 |
| Cross-teacher agreement scorer | Medium | Script to align and compare Qwen3.5 vs DeepSeek plan outputs |

The **execution sandbox** is the most important — it's the bridge that converts the "no ground truth for plans" problem into a measurable signal. Without it, plan quality remains subjective.

---

## 10. Risk Summary

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| **4B model insufficient for planning** | **High** | **Medium** | Logit distillation via Qwen3-235B if SFT brittle; fall back to 7-8B |
| **Plan quality ceiling** | Medium | High | Accept 60-80% of teacher quality; focus on common patterns |
| Mode collapse in iterative plan improvement | High | Medium | Data accumulation policy; diversity monitoring; bounded iterations |
| Legal risk from Qwen3.5 distillation | Very low | Very low | Apache 2.0 — unambiguous |
| Qwen3.5 API availability outside Alibaba Cloud | Low | Medium | Fall back to Qwen3-235B via Together AI until broader hosting |
| Teacher API cost overruns | Very low | Very low | Qwen3.5-Plus pricing is negligible |
| Quality loss from quantization | Low | Low | Q4_K_M retains 95-99% quality |
| Catastrophic forgetting in per-stage LoRAs | Medium | Low | Per-stage architecture naturally isolates risk |

**The single highest risk is Stage 4 planning quality at 4B scale.** Everything else has known mitigations. If the 4B planner proves insufficient, the recommended escape hatch is moving to Qwen3-8B for that one stage — still fast on consumer hardware, and the per-stage architecture means you only retrain one adapter.

---

## Appendix: Key URLs and Resources

| Resource | URL | License |
|---|---|---|
| Qwen3.5-397B-A17B | https://huggingface.co/Qwen/Qwen3.5-397B-A17B | Apache 2.0 |
| Qwen3.5 GitHub | https://github.com/QwenLM/Qwen3.5 | Apache 2.0 |
| Qwen3.5 Blog | https://qwen.ai/blog?id=qwen3.5 | — |
| Qwen3-Coder-Next | https://qwen.ai/blog?id=qwen3-coder-next | Apache 2.0 |
| Alibaba Cloud Model Studio | https://modelstudio.alibabacloud.com/ | — |
| OpenCodeInstruct (NVIDIA) | https://huggingface.co/datasets/nvidia/OpenCodeInstruct | CC BY 4.0 |
| Structured Agent Distillation | https://arxiv.org/abs/2505.13820 | — |
| Routine framework | https://arxiv.org/abs/2507.14447 | — |
| CodePlan (ICLR 2025) | https://openreview.net/forum?id=dCPF1wlqj8 | — |
| CodePlan (Microsoft FSE 2024) | https://github.com/microsoft/CodePlan | MIT |
| TAID | ICLR 2025 Spotlight | — |
| Unsloth | https://github.com/unslothai/unsloth | Apache 2.0 |
| LLaMA-Factory | https://github.com/hiyouga/LLaMA-Factory | Apache 2.0 |
| CommitPackFT | https://huggingface.co/datasets/bigcode/commitpackft | Open |
| DeepSeek-R1 | https://huggingface.co/deepseek-ai/DeepSeek-R1 | MIT |
