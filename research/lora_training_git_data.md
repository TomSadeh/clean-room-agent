# Building coding LoRAs for a self-improving agent pipeline

**The open-source ecosystem now supports every piece of this pipeline — but no turnkey system connects them.** Training per-stage LoRA adapters on Qwen3-1.7B (primary) and Qwen3-0.6B (binary classification) is practical on a single consumer GPU using Unsloth with QLoRA (3–5 GB VRAM for the 1.7B model), with total training time around 1–3 hours for five pipeline stages on an RTX 4090.

> **Architecture revision (Feb 2026):** The original model targets (Qwen2.5-Coder-3B + Qwen3-4B) have been replaced by Qwen3-1.7B (primary, all roles) and Qwen3-0.6B (binary classification). Planning decomposition reduced per-call complexity enough that the 4B is likely eliminated and the code-specialized 3B is replaced by the generalist 1.7B (avoiding negative transfer from Python priors). VRAM requirements and training times are lower. See `protocols/design_records/binary_decomposition_and_model_tiers.md`.

The critical architectural finding: **Ollama cannot hot-swap LoRA adapters per request**, so the deployment target should shift to vLLM or llama.cpp's `llama-server`, both of which support true per-request adapter routing with near-zero switching cost. The evidence strongly favors task-specific LoRAs over a single general-purpose adapter, particularly for small models where capacity is limited — IBM's Granite Intrinsics Library and META-LoRA both demonstrate this at production scale. The weakest link in the pipeline is dataset creation from git commits: no end-to-end tool exists, and custom building is required to bridge PyDriller extraction through LLM-powered instruction generation to training-ready format.

---

## 1. Unsloth wins for consumer-GPU LoRA training

Four frameworks dominate LoRA fine-tuning for small models, each with distinct strengths. **Unsloth** (github.com/unslothai/unsloth, ~30K+ stars, actively updated through February 2026) is the clear choice for this setup: it delivers 2–5× faster training with 70–80% less VRAM via custom Triton kernels, has explicit Qwen2.5-Coder and Qwen3 support with pre-quantized models on HuggingFace, and provides one-line Ollama/GGUF export via `model.save_pretrained_gguf()`. **LLaMA-Factory** (github.com/hiyouga/LlamaFactory, 67K+ stars, ACL 2024 paper) is the strongest alternative, offering a Web UI (LlamaBoard), Qwen3 templates (`qwen3` and `qwen3_nothink`), and direct Ollama Modelfile export since February 2025. **Axolotl** (github.com/axolotl-ai-cloud/axolotl) excels at multi-GPU training with DeepSpeed/FSDP but adds unnecessary complexity for single-card setups. **HuggingFace PEFT + TRL** provides maximum flexibility for custom training loops but requires assembling multiple components manually.

All four frameworks support the Qwen3 model family including 0.6B and 1.7B. (The original Qwen2.5-Coder-3B and Qwen3-4B targets are also supported if needed as fallbacks.) One critical compatibility note: Unsloth identified and fixed a bug where Qwen2.5-Coder's `pad_token` was incorrectly set to `<|endoftext|>`, causing infinite generations — use Unsloth's uploaded HuggingFace versions which include this fix.

**Realistic VRAM requirements** for QLoRA 4-bit training with gradient checkpointing and sequence length 2048:

| Model | Batch Size 1 | Batch Size 2 | Batch Size 4 |
|-------|:-----------:|:-----------:|:-----------:|
| Qwen3-0.6B | 2–3 GB | 3–4 GB | 4–6 GB |
| Qwen3-1.7B | 3–5 GB | 4–6 GB | 6–9 GB |
| ~~Qwen2.5-Coder-3B~~ *(original)* | 4–6 GB | 5–7 GB | 7–10 GB |
| ~~Qwen3-4B~~ *(original)* | 5–7 GB | 6–8 GB | 8–12 GB |

An **8 GB GPU** handles QLoRA on the 1.7B model at batch size 2+. A **24 GB RTX 4090** runs LoRA 16-bit with batch size 4+ on the 1.7B comfortably, and can run multiple training experiments concurrently.

The recommended LoRA configuration, synthesized from the PLoRA paper (arXiv:2508.02932), QLoRA paper, and Unsloth documentation:

- **Rank**: 16 (sweet spot; 32 for complex generation tasks). HackerNoon ablation study confirms r=16 optimal for code.
- **Alpha**: 32 (2× rank). Unsloth recommends alpha ≥ rank; Frontiers paper (2026) suggests ratios up to 4:1.
- **Target modules**: All linear layers — `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. The QLoRA paper confirms targeting all layers is critical for matching full fine-tuning performance. MLP modules show higher-rank updates during code/math training.
- **Learning rate**: 2e-4 with cosine scheduler (10× higher than full fine-tuning, per SkyRL documentation).
- **Optimizer**: `adamw_8bit` for memory efficiency.
- **Dropout**: 0 (enables Unsloth kernel optimizations; add 0.05 if overfitting observed).
- **Epochs**: 1–3 with early stopping. Sebastian Raschka found multi-epoch training on instruction datasets can *decrease* performance.

**The LoRA-to-Ollama conversion workflow** has three paths. The easiest: Unsloth's `save_pretrained_gguf()` followed by `save_pretrained_ollama()`, which auto-generates the Modelfile with correct Qwen ChatML template and calls `ollama create` internally. The manual path: merge LoRA via PEFT's `merge_and_unload()`, convert to GGUF via `llama.cpp/convert_hf_to_gguf.py`, write a Modelfile with `FROM ./model.gguf`, and run `ollama create`. For adapter-only deployment: convert the LoRA adapter to GGUF via `llama.cpp/scripts/convert_lora_to_gguf.py`, then reference it with Ollama's `ADAPTER` directive. HuggingFace also provides a web-based converter at huggingface.co/spaces/ggml-org/gguf-my-lora.

---

## 2. Commit-to-training-pair pipelines require custom assembly

No single open-source tool converts git commit histories into instruction-completion training data. The pipeline must be built from existing components, and the **D3 paper** (OpenReview, github.com/upiterbarg/d3) is the best architectural reference — it demonstrates LLM-powered instruction labeling of code edit sequences at exactly the 1–3B parameter range.

**For extraction**, **PyDriller** (github.com/ishepard/pydriller, ⚠️ pre-Aug 2025) is the standard Python framework for mining git repositories. It provides `source_code_before`, `source_code` (after), `diff`, `diff_parsed` (added/deleted lines as tuples), `changed_methods` (method-level granularity via Lizard), and metadata filtering by date, branch, file type, and author. For scale, Google BigQuery's GitHub Public Dataset allows SQL-based filtering of millions of commits before downloading — CarperAI used this to extract 19 million commits filtered to >100 stars and 22 languages, producing 1.086 billion tokens. **CommitChronicle** (github.com/saridormi/commit_chronicle, ⚠️ pre-Aug 2025, JetBrains Research) provides a reproducible collection pipeline built on PyDriller with deduplication and outlier filtering, producing 10.7M commits across 20 languages.

**For synthetic instruction generation from diffs**, two approaches stand out. **OSS-Instruct** (Magicoder, ICML 2024, arXiv:2312.02120) samples open-source code snippets as seeds and prompts an LLM to generate coding problems inspired by them — the same principle adapts to using commit diffs as seeds instead. **LintSeq** (ICLR 2025, arXiv:2410.02749, github.com/upiterbarg/lintseq) decomposes complete programs into synthetic edit sequences using a linter, producing instruction + file-state + diff-sequence tuples without needing an LLM for data generation. Models trained on LintSeq edit sequences show 3–6 point pass@1 improvements on HumanEvalFix. The **D3 dataset** (github.com/upiterbarg/d3) extends LintSeq with LLM-powered instruction labeling, producing 3.6 million examples (8 billion tokens) from The Stack — tested on Llama 3.2 1B and 3B.

**OpenCodeInstruct** (arXiv:2504.04030, NVIDIA) contributes a critical finding: **NL-to-Code instruction format significantly outperforms Code-to-Code format** across all benchmarks. This means instruction-completion pairs like "Add error handling to this function" → fixed code will train better than raw code-in → code-out pairs.

For **handling low-quality commit messages**, the recommended pipeline chains several steps. First, filter mechanically: remove merge commits, commits touching only non-code files, messages under 3 words, and bot/automated commits. Apply Verb-Direct-Object pattern filtering via NLTK/spaCy POS tagging to retain only actionable messages ("Add error handling", "Fix null pointer"). For commits with poor messages, use **OpenCommit** (github.com/di-sukharev/opencommit, 16K+ stars) or a local Ollama model to regenerate descriptions from diffs. The **OMG paper** (ACM 2024, "Only diff Is Not Enough") shows that ReAct prompting with broader software context dramatically improves generated descriptions over diff-only approaches. Finally, apply LLM-as-judge scoring (following OpenCodeInstruct and D3's approach) to filter generated descriptions, keeping only high-scoring ones.

The best **dataset format** for this pipeline is **Alpaca-style** (`instruction` / `input` / `output`) for single-turn code editing tasks — universally supported by Unsloth, Axolotl, LLaMA-Factory, and PEFT/TRL. For conversational agent training, use **ChatML/OpenAI** format (`messages` array with `role`/`content`), which aligns with Qwen's native template. CarperAI's special-token format (`<NME> filename <BEF> before_code <MSG> commit_message <DFF> diff`) with loss masking on the `<DFF>` portion is an alternative for diff-prediction training.

**Key gap**: No end-to-end tool exists for git commits → instruction-completion training pairs. Custom building bridges PyDriller extraction → quality filtering → LLM enrichment → format conversion. Budget 1–2 weeks of engineering.

---

## 3. Curated existing datasets accelerate the cold start

The landscape of open-source coding instruction datasets is rich enough to bootstrap training immediately, without waiting for commit-history pipelines to be built.

**Top recommendations for LoRA fine-tuning on 3–4B models**, ranked by quality and suitability:

- **OpenCodeInstruct** (huggingface.co/datasets/nvidia/OpenCodeInstruct, 5M examples, CC BY 4.0 ✅, April 2025 ⚠️): The largest open-access coding instruction dataset, including unit tests, execution feedback, and LLM quality scores. Fine-tuning Qwen2.5-Coder-3B+ showed substantial improvements on HumanEval, MBPP, LiveCodeBench, and BigCodeBench. Subsample 100K–500K for LoRA. **The safest license for any use case.**
- **CodeFeedback-Filtered-Instruction** (huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction, 156K examples, 2024 ⚠️): Curated from four source datasets and filtered by Qwen-72B-Chat for complexity ≥4/5. Pre-filtered, high-complexity, excellent size for LoRA.
- **Magicoder OSS-Instruct 75K** (huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K, Dec 2023 ⚠️): Generated from real open-source code snippets as seeds. MagicoderS-CL-7B trained on this data surpassed ChatGPT on HumanEval+. Orthogonal to Evol-Instruct — combine both for best results.
- **Evol-Instruct-Code** (nickrosh/Evol-Instruct-Code-80k-v1, 80K–110K examples, 2023 ⚠️): Progressively evolved from Code Alpaca. Proven results on WizardCoder (ICLR 2024, arXiv:2306.08568).
- **OpenCoder SFT Stage 2** (huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2, 375K examples, arXiv:2411.04905): Quality-filtered with test cases for RL, well-suited for LoRA.

For **code edit/diff datasets**, the options are thinner. **LintSeq** provides an algorithm to transform any existing instruction+code dataset into edit-sequence format — apply it to OpenCodeInstruct or OSS-Instruct data. **SRI (Search-and-Replace Instruction) Tuning** (arXiv:2601.13384, January 2026 ✅) trains models to generate search-and-replace edit instructions, tested on Qwen2.5-Coder-Base models with 20K SRI samples. The **SWE-bench training split** (2,294 instances, issue description → patch) provides highest-quality real-world edit data but is small. **CarperAI's diff models** (huggingface.co/CarperAI/diff-codegen-350m-v2, 2023 ⚠️) demonstrate the unified-diff training format but the dataset itself was not fully released.

**No large RAG-augmented coding training dataset exists.** The field uses RAG at inference time, not training time. **CodeRAG-Bench** (arXiv:2406.14497, Carnegie Mellon/UW) and **CrossCodeEval** (NeurIPS 2023, github.com/amazon-science/cceval, 10K cross-file completion examples) serve as evaluation frameworks. For training, the recommendation is to synthesize RAG training data by sampling (retrieved_files, task_description, solution) triples from real repositories.

**Realistic benchmark expectations for 1.7B dense models** after LoRA fine-tuning:

> *Note: Original benchmarks below were for Qwen2.5-Coder-3B. Qwen3-1.7B base benchmarks differ (see `research/qwen3_small_models.md`). Post-LoRA expectations should be calibrated against 1.7B baselines, not 3B.*

| Benchmark | Base Qwen2.5-Coder-3B *(original ref)* | Expected Post-LoRA (1.7B) |
|-----------|:---------------------:|:------------------:|
| HumanEval | 84.1% | 75–82% |
| MBPP+ | 62.4% | 55–62% |
| LiveCodeBench | ~14.2% | 12–18% |
| BigCodeBench Full | 35.8% | 30–36% |
| SWE-bench Verified | <5% (no agent) | <10% (with agent scaffold) |

LiveCodeBench (livecodebench.github.io, contamination-free, continuously updated) and BigCodeBench (ICLR 2025, bigcode-bench.github.io) are the recommended evaluation benchmarks. CrossCodeEval directly tests RAG-augmented cross-file completion. SWE-bench is aspirational for dense 3B models — Qwen3-Coder-Next achieves 70.6% but is an 80B MoE model with only 3B active parameters, architecturally very different.

---

## 4. Per-stage specialization is validated, especially for small models

The evidence strongly supports training separate LoRA adapters for different pipeline stages rather than a single general-purpose adapter. The benefit is **more pronounced for smaller models** where capacity is limited.

**IBM's Granite Intrinsics Library** (arXiv:2504.12397, github.com/IBM/activated-lora, April 2025 ⚠️) is the closest production analogue to this pipeline design. IBM trained 6 separate LoRA adapters on Granite 3.2/3.3 8B Instruct, each specialized for a RAG pipeline stage: query rewrite, answerability detection, hallucination detection, uncertainty quantification, citation generation, and jailbreak detection. Their evaluation showed **no performance degradation from combining multiple intrinsics** versus maintaining separate models. Their **Activated LoRA (aLoRA)** architecture achieves 20–35× speedup per adapter invocation by reusing KV cache — critical for multi-adapter pipelines.

**META-LoRA** (arXiv:2510.11598, October 2025 ✅) provides the strongest direct evidence for small models: "Performance gains introduced by the task-specific adaptation stage are **more pronounced when using the smaller model (LLaMA2-7B)** and the more limited amount of fine-tuning data." **R-LoRA** (EMNLP 2025 Findings) explicitly tested on **Qwen2.5-3B**, showing multi-head LoRA outperforms vanilla LoRA on multi-task benchmarks. **MTL-LoRA** (arXiv:2410.09437, ⚠️ pre-Aug 2025) found task-specific transformation matrices consistently outperform both single-task and vanilla multi-task LoRA, mitigating the "seesaw effect" where improving one task hurts another. HuggingFace's TGI blog cites Predibase evidence that "task-specific LoRAs with a base like Mistral-7B" can outperform GPT-4 on specialized tasks.

For **routing between adapters**, recent work provides several options. **LORAUTER** (arXiv:2601.21795, January 2026 ✅) routes queries to appropriate adapters via task embeddings, scaling to 1500+ adapters. **LoGo** (arXiv:2511.07129, November 2025 ✅) performs training-free, instance-specific selection and merging from a LoRA pool, outperforming training-based baselines. For the user's pipeline where stage routing is deterministic (the orchestrator knows which stage it's in), simple adapter selection via the vLLM model parameter is sufficient — no learned routing needed.

For the **relevance judgment stage**, **Self-RAG** (ICLR 2024, arXiv:2310.11511, github.com/AkariAsai/self-rag) provides the most directly applicable approach: train the model with special tokens (`[Relevant]`, `[Irrelevant]`, `[Fully supported]`, etc.) using GPT-4-generated critic labels. The model learns to output structured judgment tokens inline. **ARES** (NAACL 2024, arXiv:2311.09476) fine-tunes LLM judges on (query, passage, answer) triples with only ~150 human-annotated validation datapoints. Both approaches adapt directly to code symbol relevance classification.

For the **planning stage**, a blog post at krasserm.github.io/2024/05/31/planner-fine-tuning (⚠️ pre-Aug 2025) demonstrates fine-tuning Mistral-7B with QLoRA on 2,780 synthetic trajectories from GPT-4-based agent simulation — the 7B fine-tuned planner matched GPT-4 performance. The **Code Graph Model** (arXiv:2505.16901, May 2025 ⚠️) fine-tunes LLMs with LoRA to understand repository-level code structure using graph-to-text conversion and modified attention masks.

**Recommended per-stage architecture**:

- **Relevance judgment LoRA** (Qwen3-0.6B or 1.7B): Self-RAG special-token approach, ~2K–5K labeled examples, rank 16–32. Binary relevance is viable on 0.6B.
- **Code generation LoRA** (Qwen3-1.7B): Edit-format training (LintSeq/SRI), larger dataset (10K–50K), rank 32–64
- **Planning LoRA** (Qwen3-1.7B): Synthetic trajectories from strong teacher model, ~2K–5K examples, rank 16–32. Decomposed planning reduces per-call complexity.
- **Task analysis LoRA** (Qwen3-1.7B): Code structure + decomposition training, ~5K–10K examples, rank 16–32

**Key gap**: No published work exists on per-stage LoRA specifically for coding agent pipelines with these exact stages. IBM's work targets RAG pipelines, not code editing. The cross-stage ablation (per-stage vs. single combined LoRA) must be run by the user.

---

## 5. Self-improvement loops plateau after two iterations at this scale

The most important finding for the bootstrapping strategy: **SWE-Gym** (ICML 2025, arXiv:2412.21139, github.com/SWE-Gym/SWE-Gym) explicitly used **LoRA via Unsloth** for training and found that self-improvement with Qwen-2.5-Coder-7B **plateaus after two iterations of rejection sampling fine-tuning**. This sets realistic expectations for the closed loop.

SWE-Gym's pipeline — deploy model → sample trajectories on 2,438 real Python tasks → filter 491 successes → rejection sampling fine-tune → redeploy — is the closest existing implementation to the user's desired loop. **SWE-smith** (NeurIPS 2025 D&B Spotlight, arXiv:2504.21798, github.com/SWE-bench/SWE-smith) extends this to 50K task instances from 128 repos, training SWE-agent-LM-32B to **40.2% Pass@1 on SWE-bench Verified** using 5,000 expert trajectories from Claude 3.7 Sonnet. **R2E-Gym** (COLM 2025, arXiv:2504.07164, github.com/R2E-Gym/R2E-Gym) introduces **SWE-GEN**, a pipeline that derives training environments directly from version-control commits using back-translation and automated test generation — the closest system to "distillation from commit history" without requiring human-written PRs or tests. **Hybrid-Gym** (arXiv:2602.16819, February 2026 ✅) produces 4.4K trajectories from 762 repos at 0.07¢/example (16× cheaper than SWE-smith), with the critical finding that **repo diversity in the training set is more important than volume** for transfer to SWE-bench.

For approaches that work at the 3–4B parameter scale, **SelfCodeAlign** (NeurIPS 2024, arXiv:2410.24198) achieves self-alignment **without any teacher model** on models from 3B to 33B, with the key finding that "base models benefit more from alignment with their own data distribution." The pipeline extracts coding concepts from seed snippets, generates instructions, generates responses with test cases, executes in sandbox, and filters passing examples. **LintSeq** generates synthetic edit sequences from existing code using only a linter (no LLM needed for data generation), validated from 150M to 14B parameters. **LADDER** (arXiv:2503.00735) improved Llama 3B from 2% to 82% on integration problems using a generate→solve→verify→learn cycle with explicit curriculum.

A critical warning: the original **STaR paper** (NeurIPS 2022) states that **"sub-6B models generally fail to bootstrap"** reasoning capabilities. This applies even more directly to the 1.7B/0.6B targets than to the original 3B/4B plan. The mitigation: use teacher-generated training data (from a stronger model or API) for the initial bootstrap rounds, then switch to self-generated data once the model has enough baseline capability. Planning decomposition partially mitigates this by reducing the reasoning depth required per call.

**DPO from self-generated trajectories** is practical on consumer hardware. Single-round DPO with coarse filtering achieves RL-level results with lower compute (arXiv:2503.12854). The approach: collect successful and failed agent trajectories, pair them for preference optimization, and train with DPO. **Full GRPO/PPO training is NOT feasible on consumer hardware** — DeepSWE used 64 H100s for 6 days.

Among open-source coding agents: **OpenHands** (formerly OpenDevin) trains its own models, including OpenHands LM 32B on Qwen-2.5-Coder-32B using SWE-Gym trajectories, achieving 37% on SWE-bench Verified. **SWE-Agent** created SWE-smith for data generation and released SWE-agent-LM-32B. **Aider does NOT train models** — it focuses purely on prompt engineering and edit formats. **Lingma SWE-GPT** (Alibaba, arXiv:2411.00622) trains Qwen-based 7B/72B models by simulating code submission activities, with the **7B model resolving 18.20% of SWE-bench Verified** — notable proof that relatively small models can work with proper training.

**Recommended bootstrapping pipeline for consumer hardware**:

1. **Cold start**: Train on existing datasets (OpenCodeInstruct subsample + domain-specific data from commit history pipeline)
2. **Trajectory collection**: Run agent on tasks, log all runs with full context. Use a stronger API model (Claude/GPT) for initial expert trajectories if budget allows, OR use the small model itself and aggressively filter for successes
3. **Rejection sampling SFT**: Keep only successful trajectories. Train LoRA from frozen base with Unsloth. ~500 quality trajectories can produce measurable gains (SWE-Gym finding)
4. **Optional DPO round**: Pair successes with failures for preference optimization. One round is sufficient.
5. **Iterate with fresh tasks**: Self-improvement plateaus after 2–3 rounds on the same task distribution. Inject fresh task diversity (new repos, new problem types) to break through plateaus.

---

## 6. Deploy with vLLM, not Ollama

**Ollama does not support per-request LoRA hot-swapping.** Each adapter must be baked into a separate model tag via the `ADAPTER` directive in a Modelfile. GitHub Issue #9548 (opened March 2025) requests this feature and remains open with no implementation timeline. For a multi-stage pipeline calling different adapters per step, Ollama requires unloading/reloading between model tags — adding seconds of latency per stage transition.

**vLLM** (docs.vllm.ai) is the recommended alternative. It supports true per-request LoRA selection via the `model` parameter in its OpenAI-compatible API, dynamic loading/unloading via `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` endpoints, and concurrent multi-LoRA batching with `--max-loras N`. A quantized Qwen3-1.7B base (~1.4 GB at Q4) plus 5 rank-32 LoRA adapters (~15–40 MB each) plus KV cache fits in approximately **2–4 GB total VRAM**, leaving ample headroom on even an 8 GB GPU. The server launch command:

```bash
vllm serve Qwen/Qwen3-1.7B \
    --quantization awq --enable-lora --max-loras 8 --max-lora-rank 64 \
    --lora-modules stage1=./adapters/stage1 stage2=./adapters/stage2
```

Pipeline orchestration then becomes trivial — each stage calls the OpenAI client with a different `model` parameter selecting the appropriate adapter, with zero switching cost.

**llama.cpp's `llama-server`** is the fallback for minimal-dependency or 8 GB GPU deployments. Per-request LoRA selection was merged in PR #10994 (January 2025) — requests specify `"lora": [{"id": 0, "scale": 1.0}]` to select among pre-loaded adapters. Native GGUF quantization runs the 3B model in ~2.5 GB. **SGLang** supports multi-LoRA via S-LoRA/Punica kernels but is more complex to set up. **TGI** (HuggingFace) supports multi-LoRA since v2.0.6 via `adapter_id` in requests. **LoRAX** (github.com/predibase/lorax) was purpose-built for serving hundreds of LoRA adapters with just-in-time loading, but is overkill for a single-user pipeline.

**Training time estimates** for QLoRA rank-32 on sequence length 2048, using Unsloth:

| GPU | 1.7B Model, 10K examples | 1.7B Model, 50K examples | 0.6B Model, 10K examples |
|-----|:------------------------:|:------------------------:|:------------------------:|
| RTX 4090 (24GB) | 10–30 min | 1–2.5 hr | 5–15 min |
| RTX 3090 (24GB) | 15–45 min | 1.5–4 hr | 8–25 min |
| RTX 4080 (16GB) | 20–50 min | 2–4.5 hr | 10–30 min |

*(Original estimates for 3B/4B targets: RTX 4090 15-60 min per 10K examples. The 1.7B is faster.)*

For a 5-stage pipeline with 10K examples per stage on an RTX 4090: **roughly 1–3 hours total training time**.

**On multi-round fine-tuning degradation**: LoRA does not prevent catastrophic forgetting (confirmed by Biderman et al.'s "LoRA Learns Less and Forgets Less" and multiple follow-up studies). The critical practice: **always train independent LoRA adapters from the same frozen base model, never merge-then-retrain**. Sequential merge-retrain cycles compound quantization artifacts and cause progressive degradation after 2–3 rounds. For iterative self-improvement rounds on the same task, always retrain from the original base model with updated/expanded data — include 10–20% replay data from previous rounds to maintain past capabilities. For advanced continual learning, **CURLoRA** (arXiv:2408.14572) uses CUR decomposition, and **I-LoRA** (arXiv:2402.18865) maintains a dual-memory framework.

---

## Where the ecosystem has gaps

Several components of this pipeline have no existing open-source solutions and require custom engineering:

1. **End-to-end commit-to-training-pair pipeline**: PyDriller handles extraction, LLMs handle enrichment, but no tool connects them. Budget 1–2 weeks to build the glue code from extraction through quality filtering through instruction generation to training-format output.

2. **Code symbol relevance training data**: Self-RAG and ARES address document-level relevance, but function/class-level relevance classification for code lacks published datasets. Generate synthetic training data by having a strong model label code symbols against task descriptions.

3. **Sub-6B self-bootstrapping**: STaR-style bootstrapping is unvalidated below 6B parameters. Use teacher-generated data for the first 1–2 rounds before attempting self-generated training data.

4. **Per-stage LoRA ablation for coding agents**: No published comparison of per-stage vs. single combined LoRA for coding pipelines specifically. Run this ablation early to validate the architecture.

5. **Scaffolding + weight self-improvement**: SICA (arXiv:2504.15228) improves agent scaffolding code autonomously but doesn't update model weights. No published system combines both approaches — this pipeline could be novel.

6. **Ollama adapter hot-swap**: If Ollama is a hard deployment constraint, the workaround is merging each LoRA into a separate model copy (each becoming its own Ollama tag like `coding-agent:planner`, `coding-agent:generator`). This works but wastes disk space and adds switching latency. The better path is switching to vLLM or llama-server for the multi-adapter inference layer while potentially keeping Ollama for other uses.

The practical starting point: begin with Unsloth + QLoRA training on Qwen3-1.7B with a subsample of OpenCodeInstruct, deploy initial adapters on vLLM with per-request switching, build the commit-history extraction pipeline in parallel, and establish the evaluation loop on LiveCodeBench + BigCodeBench before investing in self-improvement infrastructure.