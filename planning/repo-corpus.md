# Repo Corpus — Training & Knowledge Base

**Status:** Draft. Compiling candidates for Phase 4 harness, base fine-tune, and knowledge base.
**Companion to:** `phase4-guidelines.md` Section 7 (Multi-Use Repo Corpus)

---

## Use Roles

Each repo serves one or more roles:

| Role | What it provides | Destination |
|---|---|---|
| **P** — Planning training | Commit history → harness → plan generation/validation → SFT + DPO pairs | Raw DB (harness triples) |
| **C** — Code style training | Source code patterns → base fine-tune + Execute-Code LoRA | Training datasets |
| **K** — Knowledge base | Indexed + enriched → curated DB → retrieved during agent tasks | Curated DB |
| **S** — Self-referential | The repo teaches the agent a skill it needs to build/improve itself | Knowledge base + training |

## Packaging Tiers

| Tier | What | Packaging | Requirement |
|---|---|---|---|
| **Full harness** | Repos with tests, commit history usable for planning training | repo + packages + manifest | Tests pass, commits filterable |
| **Knowledge base only** | C/Rust repos, reference material, repos without usable test suites | repo only (indexed + enriched) | Clean code, permissive license |
| **Documentation** | Books, guides, papers | chunked text (indexed + enriched) | Relevant to agent capabilities |

**Maximize full harness.** Every full-harness repo is a training data factory. The automated teacher-student pipeline (index → retrieve → plan → execute → validate) produces SFT triples and DPO pairs with zero human involvement. Multi-temperature teacher runs (0.3/0.6/0.9/1.2) on the same commit multiply output 3-4x and produce natural preference pairs (plan A passed tests at temp 0.6, plan B failed at temp 1.2 = DPO pair).

**Validation is broader than tests.** Test suites are the simplest oracle, but not the only one. The agent can validate by:
- Running the actual code (spin up a server, hit endpoints, check responses)
- Agentic debugging loops (error → hypothesis → add logging → re-run → narrow down → fix)
- Writing its own integration tests and property checks against observable behavior
- Performance profiling (before/after measurement = objective signal)
- Any deterministic verification the agent can invent and execute

This collapses the harness/KB-only distinction. A repo without tests isn't a dead end — the agent writes the verification, runs it, and the outcome is still binary signal. A C repo with no test suite but a `main()` that produces observable output is harness-compatible if the agent can determine what "correct" looks like. Every agentic verification loop (debugging, profiling, integration testing) is itself training data — traces of HOW the agent reasoned about the problem, not just WHAT it produced.

**Prioritize making non-Python repos harness-compatible** where feasible (C repos with `make test` or observable `main()`, Rust repos with `cargo test`, Go repos with `go test`). The more domains in the harness, the more reasoning patterns in the training data.

---

## Category 1: Fail-Fast Python (P + C)

High-quality Python repos with explicit error handling, no silent failures, good test suites. Dual-purpose: code trains fail-fast style, commits feed the planning harness.

| Repo | License | Notes | Tier |
|---|---|---|---|
| hynek/attrs | MIT | Dataclass alternative, strict validation | Full harness |
| python-attrs/cattrs | MIT | Structured/unstructured conversion for attrs | Full harness |
| hynek/structlog | MIT/Apache-2.0 | Structured logging, explicit configuration | Full harness |
| hynek/svcs | MIT | Service locator, minimal API | Full harness |
| hynek/stamina | MIT | Retry library, explicit failure handling | Full harness |
| crdoconnor/strictyaml | MIT | Type-safe YAML parsing, no implicit coercion | Full harness |
| psf/black | MIT | Deterministic code formatter | Full harness |
| Instagram/LibCST | MIT | Python CST manipulation | Full harness |
| davidhalter/parso | MIT | Python parser, error recovery | Full harness |
| agronholm/typeguard | MIT | Runtime type checking | Full harness |
| beartype/beartype | MIT | O(1) runtime type checking | Full harness |
| HypothesisWorks/hypothesis | MPL-2.0 | Property-based testing | Full harness |
| python/mypy | MIT | Static type checker | Full harness |
| Textualize/rich | MIT | Terminal formatting, renderables | Full harness |
| astral-sh/ruff | MIT | Fastest Python linter/formatter (Rust). Massive test suite, deterministic, fail-fast | Full harness |
| jcrist/msgspec | BSD-3 | Fast serialization/validation, zero deps, C extension core | Full harness |
| pydantic/pydantic | MIT | Dominant validation library, v2 Rust core, strict mode | Full harness |
| samuelcolvin/dirty-equals | MIT | Small clean test utility, well-tested | Full harness |
| astral-sh/uv | MIT/Apache-2.0 | Ultra-fast package manager (Rust), strict error handling | KB only |

---

## Category 2: LoRA / Fine-Tuning (P + C + K + S)

The agent writes and improves its own training code. These repos teach it how.

| Repo | License | Notes | Tier |
|---|---|---|---|
| unslothai/unsloth | Apache-2.0 | Optimized QLoRA, Triton kernels, Qwen support. The agent's own training framework | Full harness |
| huggingface/peft | Apache-2.0 | LoRA adapter implementation, merging, stacking | Full harness |
| huggingface/trl | Apache-2.0 | SFTTrainer, DPO, reward modeling | Full harness |
| OpenAccess-AI-Collective/axolotl | Apache-2.0 | Advanced fine-tuning, sample packing, fused kernels | Full harness |
| hiyouga/LLaMA-Factory | Apache-2.0 | YAML-driven training, multi-method support | Full harness |
| meta-pytorch/torchtune | BSD-3 | PyTorch-native post-training, pure PyTorch, clean modular LoRA/QLoRA recipes | Full harness |
| pytorch/torchtitan | BSD-3 | PyTorch-native distributed pre-training, clean parallelism abstractions | Full harness |
| huggingface/nanotron | Apache-2.0 | Minimalistic 3D-parallel LLM training, small codebase | Full harness |
| predibase/lorax | Apache-2.0 | Multi-LoRA serving, hot-swap adapters. Directly relevant to Phase 4 adapter routing | Full harness |

---

## Category 3: From-Scratch Training (P + C + K + S)

Teach the agent to build training infrastructure. Directly enabling the C-native trainer.

| Repo | License | Notes | Tier |
|---|---|---|---|
| karpathy/llm.c | MIT | GPT-2 training in pure C. Direct template for C-native trainer | KB only |
| karpathy/nanoGPT | MIT | Minimal PyTorch GPT training | Full harness |
| karpathy/minGPT | MIT | Even more minimal GPT | Full harness |
| tinygrad/tinygrad | MIT | Tiny ML framework from scratch | KB only |
| karpathy/micrograd | MIT | Autograd engine in Python | Full harness |
| karpathy/nanochat | MIT | Full-stack ChatGPT: pretrain + SFT + RLHF + inference in ~8K lines | Full harness |
| tanishqkumar/beyond-nanogpt | MIT | ~100 modern DL techniques from scratch (KV cache, spec decoding, diffusion) | KB only |
| lucidrains/x-transformers | MIT | Clean modular transformer variants (RoPE, ALiBi, flash attn). Single-author consistency | Full harness |
| huggingface/transformers | Apache-2.0 | Cherry-pick modeling files only (Qwen, Llama, GPT2). Too large for full ingest | KB only |

---

## Category 4: CUDA / GPU Programming (P + C + K + S)

The agent needs these to write CUDA kernels for the C-native trainer.

| Repo | License | Notes | Tier |
|---|---|---|---|
| ggml-org/llama.cpp | MIT | C/C++ inference with CUDA kernels. Quantization, KV cache, attention | KB only |
| ggml-org/ggml | MIT | Tensor library underlying llama.cpp | KB only |
| ggerganov/whisper.cpp | MIT | C++ inference engine, different architecture, same patterns | KB only |
| NVIDIA/cutlass | BSD-3 | CUDA templates for matrix operations | KB only |
| Dao-AILab/flash-attention | BSD-3 | Flash Attention CUDA implementation | KB only |
| HazyResearch/ThunderKittens | Apache-2.0 | Embedded DSL for CUDA kernels | KB only |
| triton-lang/triton | MIT | Python-to-GPU kernel compiler. The Python-CUDA bridge trajectory | KB only |
| NVIDIA/cuda-samples | BSD-3 | Official CUDA SDK samples, every major feature documented | KB only |
| ScalingIntelligence/KernelBench | MIT | 250 PyTorch-to-CUDA tasks. Benchmark + training data for kernel generation | KB only |
| NVIDIA/cuda-python | Apache-2.0 | Python bindings for CUDA, Tile programming model | KB only |

---

## Category 5: Inference Servers (P + K + S)

The agent migrates from Ollama to vLLM. Understanding internals helps debug and configure.

| Repo | License | Notes | Tier |
|---|---|---|---|
| vllm-project/vllm | Apache-2.0 | Target inference server. Per-request LoRA, PagedAttention, continuous batching | KB only |
| ggml-org/llama.cpp | MIT | Fallback inference server (llama-server) with LoRA support | KB only |
| huggingface/text-generation-inference | Apache-2.0 | Alternative server, batching strategy reference | KB only |
| BerriAI/litellm | MIT | Unified LLM API abstraction, 100+ providers. Relevant to LLM client layer | Full harness |
| turboderp-org/exllamav2 | MIT | Fast inference for consumer GPUs. Mixed-precision quant, lean codebase | KB only |
| NVIDIA/TensorRT-LLM | Apache-2.0 | State-of-art GPU optimization. Cherry-pick relevant modules | KB only |

---

## Category 6: Data Pipeline (P + C + K + S)

The `cra curate-data` mode processes, filters, and formats training data.

| Repo | License | Notes | Tier |
|---|---|---|---|
| huggingface/datasets | Apache-2.0 | Data loading library. Understanding internals helps optimize | Full harness |
| huggingface/datatrove | Apache-2.0 | Large-scale data processing. Filtering, dedup, quality scoring | Full harness |
| allenai/dolma | Apache-2.0 | Data curation toolkit. Pre-training data pipeline patterns | Full harness |
| ChenghaoMou/text-dedup | Apache-2.0 | Text deduplication (MinHash, SimHash, exact). Curate-data needs this | Full harness |
| google-research/deduplicate-text-datasets | Apache-2.0 | ExactSubstr dedup in Rust. Canonical implementation | KB only |
| pola-rs/polars | MIT | Rust+Python DataFrame, Arrow-native. Faster than pandas for large-scale processing | KB only |

---

## Category 7: AST / Parser (P + C + K + S)

The agent's indexer is built on AST parsing. These improve its own parsing capabilities.

| Repo | License | Notes | Tier |
|---|---|---|---|
| tree-sitter/tree-sitter | MIT | Core incremental parsing library (C) | KB only |
| tree-sitter/tree-sitter-python | MIT | Python grammar | KB only |
| tree-sitter/tree-sitter-c | MIT | C grammar | KB only |
| tree-sitter/tree-sitter-cuda | MIT | CUDA grammar | KB only |
| tree-sitter/tree-sitter-rust | MIT | Rust grammar (agent's tooling is Rust-backed) | KB only |
| tree-sitter/tree-sitter-json | MIT | JSON grammar (primary LLM output format) | KB only |
| tree-sitter/tree-sitter-toml | MIT | TOML grammar (config format) | KB only |
| Instagram/LibCST | MIT | Python CST manipulation (also in Category 1) | Full harness |
| davidhalter/parso | MIT | Python parser (also in Category 1) | Full harness |

---

## Category 8: Testing / Evaluation (P + C + K + S)

The harness runs tests. The agent evaluates its own adapters.

| Repo | License | Notes | Tier |
|---|---|---|---|
| pytest-dev/pytest | MIT | The test framework. Understanding internals helps generate better tests | Full harness |
| HypothesisWorks/hypothesis | MPL-2.0 | Property-based testing (also in Category 1) | Full harness |
| EleutherAI/lm-evaluation-harness | MIT | LLM evaluation framework. Adapter evaluation patterns | Full harness |
| bigcode-project/bigcode-evaluation-harness | Apache-2.0 | Code-specific evaluation. SWE-bench, HumanEval runners | Full harness |
| SWE-bench/SWE-bench | MIT | Canonical coding agent evaluation + (issue, patch) training pairs | Full harness |
| pytest-dev/pytest-xdist | MIT | Distributed test execution, clean plugin architecture | Full harness |
| tox-dev/tox | MIT | Test automation across Python versions | Full harness |

---

## Category 9: Git / VCS (P + C + K)

The harness uses PyDriller for commit extraction. The agent does git operations.

| Repo | License | Notes | Tier |
|---|---|---|---|
| ishepard/pydriller | Apache-2.0 | Commit mining library. Agent's commit filtering pipeline uses this | Full harness |
| gitpython-developers/GitPython | BSD-3 | Git operations from Python | Full harness |
| jelmer/dulwich | Apache-2.0/GPL-2.0 | Pure-Python Git implementation | Full harness |
| GitoxideLabs/gitoxide | MIT/Apache-2.0 | Pure Rust Git implementation. Modern, idiomatic, fast | KB only |

---

## Category 10: Database (P + C + K)

The three-database architecture uses SQLite.

| Repo | License | Notes | Tier |
|---|---|---|---|
| sqlite/sqlite | Public domain | The actual SQLite implementation. ~150K lines of battle-tested C | KB only |
| sqlalchemy/sqlalchemy | MIT | ORM patterns. Query optimization reference | Full harness |
| coleifer/peewee | MIT | Lightweight ORM. Patterns for agent's own DB layer | Full harness |
| tursodatabase/libsql | MIT | Fork of SQLite with extensions (server mode, replication, encryption) | KB only |
| rqlite/rqlite | MIT | Distributed SQLite with Raft consensus (Go) | KB only |

---

## Category 11: CLI / TUI (P + C + K)

The agent's CLI uses Click.

| Repo | License | Notes | Tier |
|---|---|---|---|
| pallets/click | BSD-3 | The CLI framework used by `cra` | Full harness |
| fastapi/typer | MIT | Modern CLI framework, alternative patterns | Full harness |
| Textualize/rich | MIT | Terminal formatting (also in Category 1) | Full harness |
| Textualize/textual | MIT | TUI framework. Potential for interactive interfaces | Full harness |
| Textualize/trogon | MIT | Auto-generates TUIs from Click CLI apps | Full harness |

---

## Category 12: Quantization / Optimization (K + S)

The agent deploys quantized models and may quantize its own mini-models.

| Repo | License | Notes | Tier |
|---|---|---|---|
| bitsandbytes-foundation/bitsandbytes | MIT | NF4 quantization for QLoRA. The agent's own quantization tool | KB only |
| AutoGPTQ/AutoGPTQ | MIT | GPTQ post-training quantization | KB only |
| casper-hansen/AutoAWQ | MIT | AWQ 4-bit quant, archived but complete. Outperforms GPTQ | KB only |
| huggingface/optimum | Apache-2.0 | Hardware optimization toolkit. ONNX/Intel/AMD bridges | Full harness |

---

## Category 13: C Reference / Systems (P + C + K)

Mature C codebases teaching systems programming patterns.

| Repo | License | Notes | Tier |
|---|---|---|---|
| redis/redis | BSD-3 | Clean C: data structures, event loops | KB only |
| sqlite/sqlite | Public domain | Gold standard for robust, well-tested C (also in Category 10) | KB only |
| jqlang/jq | MIT | C-based JSON processor. Clean, focused | KB only |
| facebook/zstd | BSD-3 | Compression in C. High-performance numerical patterns | KB only |
| jart/cosmopolitan | ISC | Build-once run-anywhere C library. Exceptional code quality. Vectorized ops | KB only |
| DaveGamble/cJSON | MIT | Ultra-lightweight JSON parser in ANSI C. Single file | KB only |
| ibireme/yyjson | MIT | Fastest JSON library in C. Pure C, single header+source | KB only |
| antirez/kilo | BSD-2 | Text editor in 1000 lines of C by Redis author | KB only |
| DoctorWkt/acwj | MIT | Self-hosting C compiler in documented steps | KB only |

---

## Category 14: Documentation / Reference (K only)

Not code repos — indexed as knowledge base content for retrieval.

| Source | Notes |
|---|---|
| C programming books (K&R, Modern C) | Core language reference |
| CUDA Programming Guide (NVIDIA) | GPU programming reference |
| PyTorch internals documentation | Framework internals |
| Transformer architecture papers | Attention Is All You Need, Flash Attention, RoPE |
| SQLite internals documentation | Database internals |
| Triton tutorials (in triton-lang repo) | Annotated Python GPU kernels (matmul, fused softmax, flash attn) |
| karpathy/nn-zero-to-hero | Building GPT from scratch, companion repo with code |

---

## Category 15: Rust-for-Python-Tooling / PyO3 (P + C + K + S)

The Rust-Python bridge. The agent's tools (ruff, uv, polars) are already Rust-backed.

| Repo | License | Notes | Tier |
|---|---|---|---|
| PyO3/pyo3 | MIT/Apache-2.0 | THE Rust-Python FFI library. Used by ruff, polars, pydantic-core | KB only |
| PyO3/maturin | MIT/Apache-2.0 | Build/publish Rust-based Python packages | KB only |
| pydantic/pydantic-core | MIT | Pydantic validation core in Rust. Real production Rust-Python code | KB only |
| huggingface/tokenizers | Apache-2.0 | Fast tokenizer in Rust with Python bindings | KB only |
| huggingface/safetensors | Apache-2.0 | Safe tensor serialization. Rust core, Python bindings, zero-copy | KB only |

---

## Category 16: Diff / Patch / Code Transformation (P + C + K + S)

The agent's core output is code patches. Fundamental operations of diffing and transforming source code.

| Repo | License | Notes | Tier |
|---|---|---|---|
| google/diff-match-patch | Apache-2.0 | Canonical Myers diff algorithm. Python + C implementations | KB only |
| mitsuhiko/python-unidiff | MIT | Parse unified diff data. Small, focused | Full harness |
| bowler-dev/bowler | MIT | Safe code refactoring on LibCST. AST-based transforms | Full harness |

---

## Category 17: GPU Kernel Synthesis (K + S)

Emerging intersection of CUDA and self-improvement. Teaching the agent to write GPU kernels.

| Repo | License | Notes | Tier |
|---|---|---|---|
| ScalingIntelligence/KernelBench | MIT | 250 PyTorch-to-CUDA tasks. Benchmark + training data | KB only |
| RLsys-Foundation/TritonForge | Apache-2.0 | SFT+RL for PyTorch → optimized Triton kernels | KB only |
| meta-pytorch/KernelAgent | BSD-3 | Autonomous GPU kernel generation. Parallel workers + verify stages | KB only |

---

## Category 18: Video Games / Game Engines (P + C + K)

Deceptively complex domains. State machines, event systems, physics, rendering, entity management. Even simple games have non-trivial architecture.

| Repo | License | Notes | Tier |
|---|---|---|---|
| raysan5/raylib | zlib | Clean C99 API, zero-dep, exhaustive examples. THE C game framework | KB only |
| orangeduck/Corange | BSD-2 | Full engine in pure C: rendering, physics, animation, assets | KB only |
| floooh/sokol | zlib | STB-style single-header libs: graphics, audio, windowing | KB only |
| SanderMertens/flecs | MIT | ECS with 8,500 tests. Data-oriented design in C99 | KB only |
| slembcke/Chipmunk2D | MIT | 11K LOC 2D physics engine. Spatial algorithms, numerical stability | KB only |
| pythonarcade/arcade | MIT | Modern 2D game lib, higher-level than pygame | Full harness |
| pyglet/pyglet | BSD-3 | Pure Python multimedia, ctypes-to-system. No compiled extensions | Full harness |

---

## Category 19: Web / HTTP / APIs (P + C + K)

The agent will work on web projects.

| Repo | License | Notes | Tier |
|---|---|---|---|
| encode/starlette | BSD-3 | ASGI foundation under FastAPI. Small, focused, well-tested | Full harness |
| pallets/flask | BSD-3 | Canonical micro-framework. Part of clean Pallets ecosystem | Full harness |
| encode/httpx | BSD-3 | 99% test coverage. Sync+async HTTP client. Pluggable transport | Full harness |
| pallets/werkzeug | BSD-3 | Low-level WSGI toolkit. HTTP parsing, routing | Full harness |
| fastapi/fastapi | MIT | Type-hint-driven APIs. Thin layer over Starlette+Pydantic | Full harness |

---

## Category 20: Scientific Computing / Numerical (P + C + K)

Numerical libraries, simulations, data analysis.

| Repo | License | Notes | Tier |
|---|---|---|---|
| numpy/numpy (core) | BSD-3 | Cherry-pick ndarray/ufunc C core. Buffer mgmt, SIMD, type dispatch | KB only |
| team-simpy/simpy | MIT | Discrete-event simulation in ~3K LOC using generators | Full harness |
| scipy/scipy (subpackages) | BSD-3 | Cherry-pick optimize, sparse. Each subpackage is self-contained | KB only |
| matplotlib/matplotlib (core) | BSD | Artist/Figure/Axes hierarchy = textbook composite pattern | KB only |

---

## Category 21: Cryptography / Security (P + C + K)

Clean crypto implementations, security tools.

| Repo | License | Notes | Tier |
|---|---|---|---|
| LoupVaillant/Monocypher | CC0/BSD-2 | <2K SLOC. Complete crypto suite. Every line auditable. Constant-time | KB only |
| jedisct1/libsodium | ISC | Production-grade NaCl successor. Extreme correctness | KB only |
| bearssl.org (BearSSL) | MIT | Full TLS in ~20KB compiled. "Cleanest C code I've ever seen" | KB only |
| PyCQA/bandit | Apache-2.0 | AST-based Python security scanner. Plugin architecture | Full harness |

---

## Category 22: Networking / Protocols (P + C + K)

Protocol implementations, async networking.

| Repo | License | Notes | Tier |
|---|---|---|---|
| libuv/libuv | MIT | Reference event loop. Powers Node.js. epoll/kqueue/IOCP/io_uring | KB only |
| curl/curl (libcurl) | MIT | Modular protocol implementations. State machine patterns | KB only |
| civetweb/civetweb | MIT | Embeddable HTTP server in C. Small, clean | KB only |
| encode/httpcore | BSD-3 | HTTP transport layer under httpx. Connection pooling, HTTP/2 | Full harness |

---

## Category 23: Compilers / Language Implementation (P + C + K)

Actual language design and implementation beyond AST parsing.

| Repo | License | Notes | Tier |
|---|---|---|---|
| lua/lua (lua.org) | MIT | 24K LOC complete language. Compiler + VM + GC + stdlib. The benchmark | KB only |
| wren-lang/wren | MIT | Language VM in <4K semicolons of C99. Fibers/coroutines in core | KB only |
| pocketpy/pocketpy | MIT | Python 3.x interpreter for game scripting. Two-file distribution | KB only |
| 8l/qbe | MIT | 14K LOC compiler backend. SSA, register alloc, instruction selection | KB only |
| DoctorWkt/acwj | MIT | Self-hosting C compiler in documented steps (also in Category 13) | KB only |
| bakpakin/Fennel | MIT | Lisp that compiles to Lua. Complete language in a single file | KB only |

---

## Category 24: Image / Audio / Media / Document Processing (P + C + K + S)

Media manipulation, codecs, document parsing, processing.

| Repo | License | Notes | Tier |
|---|---|---|---|
| nothings/stb | PD/MIT | Single-header C media libraries. JPEG/PNG decode, font rasterization | KB only |
| mackron/miniaudio | PD/MIT-0 | Complete audio stack in one C file. Cross-platform device abstraction | KB only |
| python-pillow/Pillow | HPND | Standard Python image library. C extensions for performance | Full harness |
| opendataloader-project/opendataloader-pdf | MPL-2.0 | PDF → LLM-ready markdown/JSON. Rule-based XY-Cut++, table detection, heading hierarchy, AI safety filters. Local, no GPU. **Tool dependency**: used for scientific paper conversion and training data extraction | Full harness |

---

## Category 25: Embedded / Hardware-Adjacent (P + C + K)

Resource-constrained programming, real-time patterns.

| Repo | License | Notes | Tier |
|---|---|---|---|
| micropython/micropython | MIT | Python for MCUs. 256KB code + 16KB RAM. Embedded C + Python internals | KB only |
| Mbed-TLS/mbedtls | Apache-2.0 | TLS for constrained systems. Modular build, memory-conscious | KB only |
| platformio/platformio-core | Apache-2.0 | Cross-platform embedded build system. Plugin architecture | Full harness |

---

## Category 26: Concurrency / Async Patterns (P + C + K)

Cross-cutting: the agent needs to understand concurrent code.

| Repo | License | Notes | Tier |
|---|---|---|---|
| python-trio/trio | MIT/Apache-2.0 | Structured concurrency. Nurseries enforce task lifetime correctness | Full harness |
| MagicStack/uvloop | MIT/Apache-2.0 | 2-4x faster asyncio loop. Cython wrapping libuv | Full harness |
| agronholm/anyio | MIT | Event-loop-agnostic async abstraction. One API, multiple backends | Full harness |

---

## Corpus Totals

| Tier | Count |
|---|---|
| Full harness | ~55 repos |
| Knowledge base only | ~55 repos |
| Documentation | ~15 sources |
| **Total** | **~125 sources** |

---

## APPLIED MATHEMATICS — MATH THROUGH REAL-WORLD USAGE

The agent learns math by seeing it used in context, not from abstract textbooks. A regression in an economics model, differential equations in a physics simulation, optimization in portfolio allocation. The retrieval pipeline + knowledge base handle the rest — these repos are seeds.

**Paper conversion tool:** `opendataloader-pdf` (MPL-2.0) — rule-based PDF → LLM-ready markdown/JSON. Local, no GPU, deterministic. Table detection, heading hierarchy, AI safety filters. Use for all scientific papers and PDF documentation below. Also applicable for extracting training data from any PDF-format sources.

---

## Category 27: Economics / Econometrics (P + C + K)

Regressions, time series, causal inference, structural estimation. Math in economic context.

| Repo | License | Math it teaches | Tier |
|---|---|---|---|
| statsmodels/statsmodels | BSD-3 | OLS/GLS/WLS regression, GLMs (logit/probit/Poisson), ARIMA/SARIMAX, VAR, state space (Kalman filter), MLE, kernel density estimation | Full harness |
| QuantEcon/QuantEcon.py | MIT | Markov chains (transition matrices, stationary distributions), dynamic programming (value/policy iteration), game theory (Nash equilibrium), linear-quadratic control | Full harness |
| py-why/dowhy | MIT | Causal inference (do-calculus, IV, propensity score matching, IPW), graphical models (DAGs, d-separation), regression discontinuity, diff-in-diff | Full harness |
| uber/causalml | Apache-2.0 | CATE estimation, uplift modeling, meta-learners (S/T/X-learner), doubly robust estimation, propensity scoring | Full harness |
| py-econometrics/pyfixest | MIT | High-dimensional fixed effects (within-transformation, FWL), 2SLS, Poisson regression, cluster-robust variance, wild cluster bootstrap | Full harness |

---

## Category 28: General ML / Statistical Learning (P + C + K)

Algorithm implementations where the math is visible. Not frameworks — the algorithms themselves.

| Repo | License | Math it teaches | Tier |
|---|---|---|---|
| scikit-learn/scikit-learn | BSD-3 | Linear/logistic regression, SVMs (QP, kernel trick), decision trees (entropy, Gini), k-means (Lloyd's), PCA (eigendecomp, SVD), gradient boosting, Gaussian processes, regularization (L1/L2) | Full harness |
| eriklindernoren/ML-From-Scratch | MIT | Every classical ML algorithm in pure NumPy. Gradient computations, loss functions, optimization steps all visible. ~8K LOC | KB only |
| karpathy/micrograd | MIT | Reverse-mode autodiff, computational graphs, chain rule, gradient descent. ~200 LOC, 100% math | Full harness |
| HIPS/autograd | MIT | Forward+reverse mode autodiff, Jacobian/VJP products, higher-order derivatives over arbitrary NumPy code. Predecessor to JAX | Full harness |
| jmschrei/pomegranate | MIT | Probability distributions (exponential family), GMMs (EM algorithm), HMMs, Bayesian networks, Markov chains, factor graphs | Full harness |

---

## Category 29: Physics Simulation (P + C + K)

ODEs, PDEs, vector calculus, linear algebra in physical context.

| Repo | License | Math it teaches | Tier |
|---|---|---|---|
| zwicker-group/py-pde | MIT | PDEs (diffusion, wave, advection), finite differences, spatial discretization (Cartesian/spherical/cylindrical), Runge-Kutta time stepping, boundary conditions | Full harness |
| sfepy/sfepy | BSD-3 | Finite element method (weak form, element assembly, shape functions), linear elasticity, heat transfer, Navier-Stokes, sparse linear systems | Full harness |
| qutip/qutip | BSD-3 | Quantum mechanics (Schrodinger/master equation), density matrices, tensor products, operator algebra, Lindblad dynamics, spectral decomposition, matrix exponentials | Full harness |
| SciML/DifferentialEquations.jl | MIT | ODE solvers (Euler, RK, BDF, SDIRK), SDEs (Euler-Maruyama, Milstein), DAEs, sensitivity analysis (adjoint methods), adaptive step control, stiffness detection | KB only |

---

## Category 30: Chemistry / Molecular Simulation (P + C + K)

Numerical integration, optimization, statistical mechanics in molecular context.

| Repo | License | Math it teaches | Tier |
|---|---|---|---|
| pyscf/pyscf | Apache-2.0 | Hartree-Fock (SCF iteration, Fock matrix construction), DFT, electron integrals (Gaussian basis), MP2 perturbation theory, matrix diagonalization | Full harness |
| openmm/openmm | MIT/LGPL | Molecular dynamics (Verlet/leapfrog integration), force fields (Lennard-Jones, Coulomb), Langevin dynamics, constraint algorithms (SHAKE/SETTLE), Ewald summation | KB only |
| lanl/PYSEQM | BSD-3 | Semi-empirical QM (AM1/PM3) in PyTorch. Autodiff applied to quantum chemistry — overlap between ML math and physical chemistry | Full harness |

---

## Category 31: Biology / Bioinformatics (P + C + K)

Probability, HMMs, dynamic programming, differential equations in biological context.

| Repo | License | Math it teaches | Tier |
|---|---|---|---|
| scikit-bio/scikit-bio | BSD-3 | Sequence alignment (Smith-Waterman, Needleman-Wunsch = dynamic programming), phylogenetics (distance matrices, neighbor-joining), diversity metrics (Shannon entropy, Simpson index), ordination (PCoA) | Full harness |
| hmmlearn/hmmlearn | BSD-3 | Hidden Markov models (forward-backward, Viterbi, Baum-Welch/EM), GMMs, transition probability matrices, log-sum-exp tricks. ~5K LOC, very focused | Full harness |
| biopython/biopython | BSD-like | Sequence statistics, scoring matrices (PAM/BLOSUM), structural biology (superimposition via SVD, rotation matrices), population genetics (Hardy-Weinberg) | Full harness |

---

## Category 32: Applied Mathematics Libraries (P + C + K)

Numerical methods as code. ODE/PDE solvers, optimization, linear algebra, signal processing.

| Repo | License | Math it teaches | Tier |
|---|---|---|---|
| jax-ml/jax | Apache-2.0 | Autodiff (forward+reverse), JIT compilation of numerical code, vmap vectorization, linear algebra, FFT, splittable PRNGs. NumPy + calculus | KB only |
| scipy/scipy | BSD-3 | Optimization (BFGS, L-BFGS-B, Nelder-Mead, trust-region), linear algebra (LU, QR, SVD, Cholesky), signal processing (FFT, filter design), ODE solvers (RK45, BDF), interpolation (splines, RBF), sparse matrices, special functions | KB only |
| numpy/numpy | BSD-3 | Array operations (broadcasting), linear algebra (matmul, determinants, eigenvalues), FFT, random distributions. The foundation | KB only |

---

## Category 33: Finance / Quantitative Finance (P + C + K)

Stochastic calculus, Monte Carlo, optimization, time series in financial context.

| Repo | License | Math it teaches | Tier |
|---|---|---|---|
| robertmartin8/PyPortfolioOpt | MIT | Mean-variance optimization (Markowitz efficient frontier), covariance estimation (Ledoit-Wolf shrinkage), Black-Litterman, risk parity, convex optimization via CVXPY | Full harness |
| luphord/nelson_siegel_svensson | MIT | Nelson-Siegel yield curve model, parametric curve fitting, nonlinear least squares. ~1K LOC, math fully exposed | Full harness |
| domokane/FinancePy | GPL-3.0 | Black-Scholes (PDE + analytical), Hull-White/Vasicek/CIR interest rate models, Monte Carlo simulation, finite differences for PDE pricing, Greeks computation, exotic options | Full harness |

---

## Category 34: Control Systems / Robotics (P + C + K)

Linear algebra, differential equations, optimization, probability in control context.

| Repo | License | Math it teaches | Tier |
|---|---|---|---|
| python-control/python-control | BSD-3 | Transfer functions, state-space models, Bode/Nyquist, root locus, LQR/LQG, Kalman filter, controllability/observability, PID tuning, matrix Riccati equations | Full harness |
| rlabbe/filterpy | MIT | Kalman filter (predict-update, covariance propagation), Extended KF (Jacobian linearization), Unscented KF (sigma points), particle filter (sequential Monte Carlo). ~8K LOC | Full harness |
| rlabbe/Kalman-and-Bayesian-Filters-in-Python | MIT/CC-BY | Full mathematical derivations interleaved with code. Bayesian probability, Gaussian distributions, matrix algebra, prediction-correction. 15 notebooks | KB only |

---

## Category 35: Signal Processing / DSP (P + C + K)

Fourier transforms, convolution, linear algebra, information theory in signal context.

| Repo | License | Math it teaches | Tier |
|---|---|---|---|
| librosa/librosa | ISC | STFT, MFCCs, chromagrams, spectral features, beat tracking (autocorrelation), harmonic-percussive separation, constant-Q transform | Full harness |
| scipy.signal (part of scipy) | BSD-3 | Filter design (Butterworth, Chebyshev, elliptic), convolution/correlation, FFT/IFFT, spectral density estimation (Welch), window functions, resampling | KB only |

---

## Category 36: Operations Research / Optimization (P + C + K)

Optimization theory applied to real scheduling, routing, allocation problems.

| Repo | License | Math it teaches | Tier |
|---|---|---|---|
| cvxpy/cvxpy | Apache-2.0 | Convex optimization (DCP), LP, QP, SDP, SOCP, MIP, duality (Lagrangian, KKT), problem canonicalization. Stanford Boyd's group | Full harness |
| google/or-tools | Apache-2.0 | LP (simplex, interior point), MIP (branch and bound), constraint programming, vehicle routing (TSP), network flow, scheduling | KB only |
| Pyomo/pyomo | BSD-3 | Algebraic modeling, linear/nonlinear/integer programming, stochastic programming, model decomposition | Full harness |
| coin-or/pulp | MIT | LP formulation (objectives, constraints, bounds), integer programming, dual values. Simplest LP modeler, ~5K LOC | Full harness |

---

## Scientific Papers for Knowledge Base (Category 37)

Convert via the pipeline at `C:\Users\User\Documents\GitHub\claude-knowledge-repo`. These are methods papers — practical, not pure theory.

### Optimization & Learning
| Paper | Why |
|---|---|
| Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization" | The dominant optimizer. Adaptive learning rates, moment estimation, bias correction |
| Nocedal & Wright (2006). "Numerical Optimization" (Ch 6-7: BFGS, CG) | Workhorses of scipy.optimize. Quasi-Newton methods, line search, trust regions |
| Boyd & Vandenberghe (2004). "Convex Optimization" | Foundation of CVXPY. Duality, KKT, interior point methods. Free online |

### Statistical Learning
| Paper | Why |
|---|---|
| Hastie, Tibshirani & Friedman (2009). "Elements of Statistical Learning" | Mathematical foundation behind scikit-learn. Bias-variance, regularization, boosting. Free online |
| Rumelhart, Hinton & Williams (1986). "Learning representations by back-propagating errors" | Backpropagation. Chain rule on computational graphs. Implemented in micrograd/autograd/tinygrad |

### Numerical Methods & Simulation
| Paper | Why |
|---|---|
| Dormand & Prince (1980). "A family of embedded Runge-Kutta formulae" | RK45 (DoPri5) — default ODE solver in SciPy/MATLAB. Adaptive step-size control |
| Kalman (1960). "A New Approach to Linear Filtering and Prediction Problems" | Foundation of state estimation. Recursive Bayesian, covariance propagation, matrix Riccati |
| Metropolis et al. (1953). "Equation of State Calculations by Fast Computing Machines" | Birth of Monte Carlo methods. MCMC, detailed balance, acceptance probability |

### Finance & Economics
| Paper | Why |
|---|---|
| Black & Scholes (1973). "The Pricing of Options and Corporate Liabilities" | Foundation of quant finance. SDEs → PDEs → option pricing |
| Pearl (2009). "Causality" (or 2000 overview paper) | Foundation of DoWhy/causalml. Do-calculus, DAGs, counterfactuals |

---

## Corpus Totals

| Tier | Count |
|---|---|
| Full harness | ~80 repos |
| Knowledge base only | ~65 repos |
| Documentation + papers | ~25 sources |
| **Total** | **~170 sources** |

---

## Standout Gems (highest learning per LOC)

These repos deliver exceptional training signal relative to their size:

- **Monocypher** — 2K lines of perfect C. Complete crypto suite
- **wren** — complete language VM in <4K semicolons of C99
- **QBE** — 14K-line compiler backend (vs LLVM's 10M)
- **Lua** — 24K lines containing a production language
- **nanochat** — full training pipeline (pretrain+SFT+RLHF+inference) in ~8K lines
- **httpx** — 99% test coverage, clean async Python
- **trio** — structured concurrency as a design lesson
- **flecs** — 8,500 tests, data-oriented design in C
- **antirez/kilo** — complete text editor in 1000 lines of C
- **cJSON** — JSON parser in one ANSI C file
- **micrograd** — reverse-mode autodiff in ~200 lines. 100% math
- **filterpy** — Kalman filter variants, ~8K LOC, linear algebra + probability
- **nelson_siegel_svensson** — yield curve fitting in ~1K LOC, nonlinear optimization
- **hmmlearn** — HMMs in ~5K LOC, EM algorithm + probability
- **ML-From-Scratch** — every classical ML algo in pure NumPy, ~8K LOC
- **QuantEcon.py** — dynamic programming + Markov chains + game theory, ~15K LOC
- **py-pde** — PDE solving with visible finite differences, ~20K LOC
