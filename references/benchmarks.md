# The definitive catalog of AI coding benchmarks

**Over 100 AI coding benchmarks now exist**, spanning code generation, software engineering, competitive programming, debugging, security, translation, and more. The overwhelming majority — roughly 85% — use **ground-truth evaluation** via unit tests or execution-based verification, making coding one of the most rigorously measurable domains in AI. This catalog documents **102 distinct benchmarks** organized by category, with metadata on evaluation type, size, language coverage, and provenance.

The landscape has exploded since OpenAI's HumanEval (2021), which established the pass@k paradigm. The field has since moved toward contamination-resistant designs (LiveCodeBench, EvoCodeBench), repository-level realism (SWE-bench family), and massive multilingual coverage (McEval with 40 languages, mHumanEval with 204 natural languages). A key finding: **"vibed" evaluation remains rare in coding benchmarks** — the executability of code makes objective verification the norm, with only a handful of benchmarks (Scale SEAL, some security evaluations) relying on LLM-as-judge or subjective scoring.

---

## Code generation: the foundational benchmarks

These benchmarks evaluate a model's ability to produce correct code from natural language specifications. Nearly all use unit-test-based pass@k evaluation — the gold standard for ground-truth assessment.

| # | Benchmark | Size | Languages | Evaluation | GT/Vibed | Year | Creator |
|---|-----------|------|-----------|------------|----------|------|---------|
| 1 | **HumanEval** | 164 problems | Python | Unit tests (pass@k); avg 7.7 tests/problem | Ground-truth | 2021 | OpenAI |
| 2 | **HumanEval+** (EvalPlus) | 164 problems, **80× more tests** | Python | Unit tests; ~764 tests/problem median | Ground-truth | 2023 | UIUC (Liu et al.) |
| 3 | **MBPP** | 974 tasks (427 sanitized) | Python | Unit tests; 3 tests/problem | Ground-truth | 2021 | Google DeepMind |
| 4 | **MBPP+** (EvalPlus) | 378–399 problems, **35× more tests** | Python | Unit tests; ~108 tests/problem | Ground-truth | 2024 | UIUC (Liu et al.) |
| 5 | **HumanEval Pro / MBPP Pro** | ~564 problems | Python | Unit tests (pass@1); self-invoking tasks | Ground-truth | 2024 | Yu et al. |
| 6 | **BigCodeBench** | 1,140 tasks (148 Hard) | Python | Unit tests; 5.6 tests/task avg, 99% branch coverage | Ground-truth | 2024 | BigCode / Monash Univ. |
| 7 | **DS-1000** | 1,000 problems across 7 libraries | Python | Unit tests + surface-form constraints; 1.8% false-accept rate | Ground-truth | 2022 | CMU / Meta AI |
| 8 | **DSCodeBench** | 1,000 problems, 10 libraries | Python | Unit tests | Ground-truth | 2025 | Multi-institutional |
| 9 | **APPS** | 10,000 problems (3 difficulty tiers) | Python | Unit tests; 131K+ test cases total | Ground-truth | 2021 | UC Berkeley (Hendrycks et al.) |
| 10 | **ClassEval** | 100 classes, 412 methods | Python | Unit tests; 33.1 tests/class avg, 98% branch coverage | Ground-truth | 2023 | Fudan University |
| 11 | **CoderEval** | 460 tasks (230 Python + 230 Java) | Python, Java | Unit tests in Docker; project-level execution | Ground-truth | 2023 | Huawei |
| 12 | **ODEX** | 945 NL-code pairs | Python (code); EN/ES/JA/RU (intents) | Unit tests; 1,707 human-written test cases | Ground-truth | 2022 | CMU NeuLab |
| 13 | **EvoCodeBench** | 275 samples from 25 repos | Python | Unit tests + dependency recall metric | Ground-truth | 2024 | Peking University |
| 14 | **NaturalCodeBench** | 402 problems (201 Py + 201 Java) | Python, Java | Unit tests; avg 9.3 tests/problem | Ground-truth | 2024 | Tsinghua / Zhipu AI |
| 15 | **AixBench** | 336 samples | Java | Unit tests (175) + manual evaluation (161) | Mixed | 2022 | Baidu / aiXcoder |
| 16 | **CanAiCode** | Varies (leaderboard) | Multiple | Sandboxed code execution | Ground-truth | 2023+ | Community |
| 17 | **SciCode** | 80 main → 338 subproblems, 6 domains | Python | Scientist-annotated unit tests; 3 rounds of validation | Ground-truth | 2024 | CMU / UIUC et al. |
| 18 | **ResearchCodeBench** | TBD | Python | Unit tests for research code | Ground-truth | 2025 | Hua et al. |
| 19 | **EvoEval** | ~828 problems (7 transformations of HumanEval) | Python | Unit tests (pass@k) | Ground-truth | 2024 | Xia et al. |
| 20 | **WebApp1K** | 1,000 problems across 20 web domains | JavaScript (React) | Unit tests (Jest); TDD-based pass@k | Ground-truth | 2024 | OneKQ Lab (Yi Cui) |
| 21 | **FullStackBench** | 3,374 problems, 11 domains | 16 languages | Unit tests via SandboxFusion | Ground-truth | 2024 | ByteDance Seed |

**HumanEval** remains the most-cited benchmark but is increasingly viewed as saturated — top models exceed 95% pass@1. **BigCodeBench** and **LiveCodeBench** have emerged as the preferred modern alternatives, offering harder tasks and contamination resistance. The **EvalPlus** project (HumanEval+, MBPP+) demonstrated that simply adding more test cases drops model scores by 10–20 percentage points, exposing how narrow original test suites were.

---

## Multi-language code generation benchmarks

These extend code generation evaluation beyond Python to dozens of programming and natural languages.

| # | Benchmark | Size | PL Coverage | NL Coverage | Evaluation | GT/Vibed | Year | Creator |
|---|-----------|------|-------------|-------------|------------|----------|------|---------|
| 22 | **MultiPL-E** | 164+ problems/lang | **18+ PLs** (Rust, Go, Swift, Lua, R, etc.) | English | Unit tests (pass@k) | Ground-truth | 2022 | Northeastern Univ. |
| 23 | **HumanEval-X** | 820 (164 × 5 langs) | Python, C++, Java, JS, Go | English | Unit tests (manually rewritten/lang) | Ground-truth | 2023 | Tsinghua (THUDM) |
| 24 | **HumanEvalPack** | 164 × 6 langs × 3 tasks | Python, JS, Java, Go, C++, Rust | English | Unit tests (synthesis/repair); LLM eval (explanation) | Mixed | 2023 | BigCode / Muennighoff |
| 25 | **McEval** | ~16,000 samples | **40 PLs** (5 paradigms, 11 domains) | English | Unit tests; human-written (not translated) | Ground-truth | 2024 | Multi-institutional |
| 26 | **xCodeEval** | 25M examples, ~7,500 problems, 7 tasks | **11 PLs** (C, C++, C#, Go, Java, JS, Kotlin, PHP, Python, Ruby, Rust) | English | Unit tests via ExecEval engine | Ground-truth | 2023 | NTU NLP |
| 27 | **CodeScope** | Multi-dataset, 8 tasks | **43 PLs** (14 with execution) | English | Execution-based via MultiCodeEngine | Ground-truth | 2024 | UCSB / Zhejiang / Alibaba |
| 28 | **HumanEval-XL** | 22,080 prompts (80 × 23 NLs × 12 PLs) | 12 PLs | **23 NLs** | Unit tests (pass@1) | Ground-truth | 2024 | Peng et al. |
| 29 | **mHumanEval** | 164 base × 204 NLs × 25 PLs | **25 PLs** | **204 NLs** | Unit tests (pass@1) | Ground-truth | 2025 | Raihan et al. (NAACL 2025) |
| 30 | **CRUXEval-X** | 800 functions × multiple PLs | C++, Java, JS, and others | — | Exact match vs. execution | Ground-truth | 2024 | CAS / HKUST |

**McEval** stands out for covering 40 programming languages with original (non-translated) test cases. **mHumanEval** achieves the widest natural language coverage at 204 languages, useful for evaluating how well models understand coding tasks described in low-resource languages.

---

## Competitive programming benchmarks

These target algorithmic problem-solving at interview to Olympiad difficulty, typically using hidden test suites or online judges.

| # | Benchmark | Size | Languages | Evaluation | GT/Vibed | Year | Creator |
|---|-----------|------|-----------|------------|----------|------|---------|
| 31 | **CodeContests** | ~13,000+ problems | C++, Python, Java | Unit tests (~200 private tests/problem) | Ground-truth | 2022 | Google DeepMind (AlphaCode) |
| 32 | **LiveCodeBench** | 1,000+ problems (continuously growing) | Python (eval); multi-lang platforms | Hidden test cases; time-segmented to prevent contamination | Ground-truth | 2024+ | MIT / UC Berkeley |
| 33 | **USACO Benchmark** | 307 problems (Bronze → Platinum) | Python, C++, Java | Unit tests; 10–20 tests/problem | Ground-truth | 2024 | Princeton NLP |
| 34 | **CodeElo** | Codeforces problems (800–2400+ rating) | C++, Python, Java | **Live Codeforces judge** + Elo rating system | Ground-truth | 2025 | Alibaba / Qwen team |
| 35 | **ProBench** | Multi-platform (CF, Luogu, Nowcoder) | C++, Java, Python | Online platform submission; bilingual (EN/CN) | Ground-truth | 2025 | Yang, Jin, Shi et al. |
| 36 | **TACO** | 25,443+ problems | Python | Unit tests; topic-classified (36 categories) | Ground-truth | 2023 | Li et al. |
| 37 | **LeetCodeDataset** | ~2,100 problems; 256 post-July-2024 | Multi-language via LeetCode | LeetCode online judge | Ground-truth | 2025 | Various |
| 38 | **UA-Code-Bench** | 500 problems (5 difficulty levels) | Python | Online judge (Eolymp platform) | Ground-truth | 2025 | Ukrainian researchers |

**CodeElo** is notable for submitting model outputs directly to the Codeforces online judge and producing human-comparable Elo ratings, making it the most ecologically valid competitive programming evaluation. **LiveCodeBench** solves the contamination problem by continuously adding new problems with timestamps.

---

## Agentic and software engineering benchmarks

These evaluate models on realistic software engineering tasks — from fixing real GitHub issues to building entire libraries. This is the fastest-growing category.

| # | Benchmark | Size | Languages | Evaluation | GT/Vibed | Year | Creator |
|---|-----------|------|-----------|------------|----------|------|---------|
| 39 | **SWE-bench** | 2,294 tasks from 12 repos | Python | Unit tests (fail-to-pass + pass-to-pass) in Docker | Ground-truth | 2023 | Princeton NLP |
| 40 | **SWE-bench Lite** | 300 tasks (easier subset) | Python | Unit tests | Ground-truth | 2024 | Princeton NLP |
| 41 | **SWE-bench Verified** | 500 tasks (human-validated) | Python | Unit tests; each task verified by 3 annotators | Ground-truth | 2024 | Princeton NLP + OpenAI |
| 42 | **SWE-bench Multimodal** | 617 tasks from 17 JS libraries | JavaScript / TypeScript | Unit tests + pixel-diff screenshot comparison | Ground-truth | 2024 | Princeton NLP / Meta FAIR |
| 43 | **SWE-bench Pro** | 1,865 tasks across 41 repos | Multiple (Python + others) | Unit tests; includes proprietary code | Ground-truth | 2025 | Scale AI |
| 44 | **SWE-bench Live** | Continuously growing | Python | Unit tests; monthly updates | Ground-truth | 2025 | Princeton NLP |
| 45 | **SWE-PolyBench / Multi-SWE-bench** | Varies | Java, Rust, and others | Unit tests (SWE-bench paradigm) | Ground-truth | 2025 | Various |
| 46 | **SWE-Lancer** | Tasks valued at $1M total | Multiple | Task completion evaluation | Ground-truth | 2025 | Research team |
| 47 | **Commit0** | 57 Python libraries (write from scratch) | Python | Unit tests (pytest); linting with ruff | Ground-truth | 2024 | Univ. of Toronto / Vector Institute |
| 48 | **RE-Bench** | 7 deep ML research environments | Python | Automated scoring (speed, performance metrics) vs. 61 human experts | Ground-truth | 2024 | METR |
| 49 | **MLE-Bench** | 75 Kaggle competitions | Python | Kaggle grading scripts; medal thresholds | Ground-truth | 2024 | OpenAI |
| 50 | **R2E** | 137 repositories | Python | Equivalence tests (same outputs as ground-truth function) | Ground-truth | 2024 | UC Berkeley |
| 51 | **FEA-Bench** | Multiple tasks (lite + full) | Python | Unit tests (feature implementation, not bug fixing) | Ground-truth | 2025 | Multi-institutional (ACL 2025) |
| 52 | **FeatureBench** | Not fully specified | Multiple | Unit tests | Ground-truth | 2025 | Research team |
| 53 | **GitTaskBench** | 54 expert-designed tasks | Multiple | Expert-designed task evaluation | Ground-truth | 2025 | UCAS |
| 54 | **Aider Polyglot** | 225 Exercism exercises | C++, Go, Java, JS, Python, Rust | Unit tests; 2 attempts (second with error feedback) | Ground-truth | 2024 | Aider (Paul Gauthier) |
| 55 | **BFCL** (Berkeley Function Calling) | 2,000+ question-function pairs | Python, Java, JS, REST API | AST matching + executable verification | Ground-truth | 2024+ | UC Berkeley (Gorilla project) |
| 56 | **AgentBench** | 8 environments (OS, DB, web, etc.) | Multi-domain | Task-specific success metrics | Ground-truth | 2023 | Tsinghua (THUDM) |
| 57 | **WebArena** | 812 tasks | Web interaction | Functional correctness (task completion) | Ground-truth | 2023 | CMU |
| 58 | **VisualWebArena** | 910 tasks | Multimodal web | Execution-based visual grounding | Ground-truth | 2024 | CMU |

The **SWE-bench family** dominates this category with at least 7 variants. **SWE-bench Verified** (500 human-validated tasks) has become the de facto standard for evaluating coding agents, while **SWE-bench Multimodal** extends to JavaScript/visual software. **Commit0** is uniquely ambitious — it requires agents to write entire libraries from scratch given only specification documents.

---

## Repository-level code completion benchmarks

These evaluate models' ability to complete code using cross-file context from real repositories.

| # | Benchmark | Size | Languages | Evaluation | GT/Vibed | Year | Creator |
|---|-----------|------|-----------|------------|----------|------|---------|
| 59 | **RepoBench** | Large-scale (3 tasks: Retrieval, Completion, Pipeline) | Python, Java | Exact Match, Edit Similarity, CodeBLEU | Ground-truth | 2023 | UC San Diego |
| 60 | **CrossCodeEval** | Thousands of instances | Python, Java, TypeScript, C# | Exact Match, Edit Similarity, Identifier Match | Ground-truth | 2023 | Amazon Science |
| 61 | **RepoEval** | Multiple tasks from 14 repos | Python | Edit Similarity + unit tests (function-level) | Ground-truth | 2023 | Zhang et al. (EMNLP 2023) |
| 62 | **RepoMasterEval** | 288 code snippets | Python, TypeScript | Enhanced unit testing (reintegrate into repo + run tests) | Ground-truth | 2024 | Peng et al. (ASE 2025) |
| 63 | **DevBench** (SW Lifecycle) | Multi-stage projects | Multiple | Unit tests + structured metrics per stage | Ground-truth | 2024 | OpenCompass / Shanghai AI Lab |
| 64 | **DevEval** | Repository-aligned tasks | Multiple | Comprehensive annotations | Ground-truth | 2024 | ACL 2024 Findings |

---

## Code reasoning and understanding benchmarks

These test whether models can predict program behavior, understand code semantics, or answer questions about code — without necessarily generating new code.

| # | Benchmark | Size | Languages | Evaluation | GT/Vibed | Year | Creator |
|---|-----------|------|-----------|------------|----------|------|---------|
| 65 | **CRUXEval** | 800 functions (input + output prediction) | Python | Exact match vs. executed ground truth | Ground-truth | 2024 | MIT / Meta AI |
| 66 | **CRUXEval-X** | 800 × multiple PLs | C++, Java, JS, others | Exact match across languages | Ground-truth | 2024 | CAS / HKUST |
| 67 | **CodeMind** | 5,395 programs (3 reasoning tasks) | Python, Java | Execution prediction + static analysis | Ground-truth | 2024 | UIUC |
| 68 | **CodeApex** | Large-scale, 3 tasks | C++ (code); EN/CN (bilingual) | MCQ accuracy + unit tests + compilation | Ground-truth | 2023 | Research team |
| 69 | **InfiBench** | 234 Stack Overflow questions | 15 PLs | Keyword matching, blank filling, unit tests, dialogue similarity | Ground-truth | 2024 | InfiCoder Team |
| 70 | **CodeXGLUE** | 14 datasets, 10 tasks | Python, Java, Go, Ruby, JS, PHP, C/C# | Mixed (BLEU, exact match, execution, accuracy) | Ground-truth | 2021 | Microsoft Research |

**CRUXEval** has become the standard for testing code reasoning — it asks models to predict what a function outputs given an input (or vice versa). Models that score well on code generation often struggle here, revealing that "writing code" and "understanding code" are partially independent capabilities.

---

## Code repair and debugging benchmarks

These evaluate the ability to find and fix bugs in existing code — from single-line errors to complex real-world defects.

| # | Benchmark | Size | Languages | Evaluation | GT/Vibed | Year | Creator |
|---|-----------|------|-----------|------------|----------|------|---------|
| 71 | **DebugBench** | 4,253 instances (4 bug categories, 18 types) | C++, Java, Python | Unit tests (LeetCode test cases) | Ground-truth | 2024 | Tsinghua / THUNLP |
| 72 | **DebugEval** | ~5,712 instances (4 tasks) | Python, C++, Java | Unit tests (repair) + accuracy (localization) | Ground-truth | 2024 | Univ. of Michigan |
| 73 | **Defects4J** | **854 bugs** (v2.0) from 17 projects | Java | Unit tests (fail-to-pass with developer test suites) | Ground-truth | 2014+ | Univ. of Washington (René Just) |
| 74 | **QuixBugs** | 40 programs (one-line bugs) | Python, Java | Unit tests | Ground-truth | 2017 | MIT |
| 75 | **BugsInPy** | 493 bugs from 17 projects | Python | Unit tests (Defects4J-like infrastructure) | Ground-truth | 2020 | Singapore Mgmt Univ. |
| 76 | **FixEval** | Large-scale (~6.5M submissions) | Java, Python | Execution-based test pass/fail | Ground-truth | 2022 | Virginia Tech |
| 77 | **ConDefects** | 2,879 faulty programs (1,254 Java + 1,625 Python) | Java, Python | Unit tests; designed for LLM-era leakage-resistance | Ground-truth | 2023 | Wu et al. |
| 78 | **MdEval** (Multilingual Debug) | ~3,900 samples | **20 PLs** | Pass@1 for repair; accuracy for localization | Ground-truth | 2024 | ByteDance / Alibaba |
| 79 | **RunBugRun** | **~450,000 bugs** across 8 languages | C, C++, Java, Python, JS, Ruby, Go, PHP | Unit tests (Defects4J-like) | Ground-truth | 2023 | INRIA |
| 80 | **Defects4C** | 248 buggy + 102 vulnerable functions | C, C++ | Unit tests | Ground-truth | 2024 | Academic researchers |
| 81 | **GitBug-Java** | 199 bugs from 55 repos (2023 commits only) | Java | Unit tests via GitHub Actions | Ground-truth | 2024 | KTH (Silva, Monperrus) |
| 82 | **CodeEditorBench** | 7,961 tasks (debug, translate, polish, etc.) | C++, Java, Python | Online Judge; avg 44 tests/task | Ground-truth | 2024 | Multi-institutional |
| 83 | **CVE-Bench** | Multiple CVE instances | Python, JS, PHP | Execution-based repair rate | Ground-truth | 2025 | NAACL 2025 |

**RunBugRun** is the largest repair benchmark at ~450K bugs across 8 languages. **Defects4J** remains the most-cited gold standard for Java APR despite being over a decade old. **ConDefects** and **GitBug-Java** explicitly address data leakage for LLM evaluation by using only recent (post-training-cutoff) bugs.

---

## Security and vulnerability detection benchmarks

This category shows the most evaluation diversity — some use ground-truth execution, while others rely on static analyzers with known false-positive rates, making evaluation quality more variable.

| # | Benchmark | Size | Languages | Evaluation | GT/Vibed | Year | Creator |
|---|-----------|------|-----------|------------|----------|------|---------|
| 84 | **CyberSecEval** (v1–v4) | Hundreds of test cases (varies by version) | Python, C, C++, Java, others | Static analysis (CWE-based) + keyword judgment + LLM-as-judge (MITRE tests) | **Mixed** | 2023–2025 | Meta AI (Purple Llama) |
| 85 | **SecurityEval** | 121 prompts for 69 CWEs | Python | Static analysis (Bandit, CodeQL) | **Mixed** (known FP/FN issues) | 2022 | s2e-lab (Siddiq, Santos) |
| 86 | **LLMSecEval** | 150 NL prompts | Python, C | Static analysis (CodeQL for 18 of Top 25 CWEs) | **Mixed** | 2023 | Tony et al. (MSR 2023) |
| 87 | **SVEN** | Multiple CWE scenarios | C, C++ | Static analysis + HumanEval functional correctness | Ground-truth | 2023 | ETH Zurich |
| 88 | **SecCodePLT** | 1,345 problems, 27 CWEs | Python | Functionality + vulnerability test suites (paired) | Ground-truth | 2024 | Yang et al. |
| 89 | **CWEval** | Multiple CWE-mapped tasks | Python | Combined unit testing + dynamic security testing | Ground-truth | 2025 | Academic researchers |
| 90 | **SecVulEval** | 25,440 function samples, 5,867 CVEs | C, C++ | Multi-agent pipeline; F1 at statement level | Ground-truth | 2025 | Ahmed et al. |
| 91 | **PrimeVul** | Thousands of functions | C, C++ | Binary classification (vulnerable/not) | Ground-truth | 2024 | Academic researchers |
| 92 | **Devign** | ~27,318 functions | C | Binary classification (accuracy, F1) | Ground-truth | 2019 | Zhou et al. |
| 93 | **Big-Vul** | ~188,000 functions | C, C++ | Binary classification (CVE-mapped) | Ground-truth | 2020 | Fan et al. |
| 94 | **VADER** | Not fully specified | Multiple | Human evaluation + automated metrics | **Mixed** | 2025 | Academic researchers |

Security benchmarks are the most methodologically divided category. Older benchmarks like **SecurityEval** and **LLMSecEval** rely on static analyzers (Bandit, CodeQL) that have known false-positive and false-negative issues, making their evaluation partially "vibed." Newer benchmarks like **CWEval** and **SecCodePLT** address this by combining execution-based functionality tests with dynamic security testing — a dual ground-truth approach.

---

## Code translation benchmarks

These evaluate the ability to translate code between programming languages or deep learning frameworks.

| # | Benchmark | Size | Languages | Evaluation | GT/Vibed | Year | Creator |
|---|-----------|------|-----------|------------|----------|------|---------|
| 95 | **CodeTransOcean** | 270,507 examples | **45 PLs** + 4 DL frameworks | BLEU, CodeBLEU, Exact Match, DSR@K (execution) | Ground-truth | 2023 | Alibaba (EMNLP 2023) |
| 96 | **G-TransEval** | 400 translation pairs (4-level taxonomy) | Python, C++, Java, C#, JS | Unit tests + CodeBLEU | Ground-truth | 2023 | Jiao et al. (ASE 2023) |
| 97 | **AVATAR** | 9,515 problems; 250 with unit tests | Java, Python | Execution accuracy + CodeBLEU; unit tests for subset | Ground-truth | 2023 | UCLA / Microsoft |
| 98 | **TransCoder-test** | ~850 functions | Python, Java, C++ | Unit tests (execution-based) | Ground-truth | 2020 | Facebook AI Research |
| 99 | **XLCoST** | Large-scale parallel dataset | C++, Java, Python, C#, JS, PHP, C | BLEU, CodeBLEU, execution-based | Ground-truth | 2022 | Zhu et al. |
| 100 | **RepoTransBench** | Multiple repository-level tasks | Multiple (Python, Java, C++, Rust, etc.) | Execution-based test suites | Ground-truth | 2024 | Academic researchers |

**CodeTransOcean** is the largest translation benchmark by far at 270K examples across 45 languages. Its novel **DSR@K** (Debugging Success Rate) metric is noteworthy — it measures whether translated code can be made to work with k debugging attempts, reflecting real-world translation workflows.

---

## Fill-in-the-middle and code completion benchmarks

| # | Benchmark | Size | Languages | Evaluation | GT/Vibed | Year | Creator |
|---|-----------|------|-----------|------------|----------|------|---------|
| 101 | **SAFIM** | 17,720 examples from 8,590 files | Multiple (Python, Java, C++, etc.) | Exact match with syntax-aware post-processing | Ground-truth | 2024 | Academic researchers |
| 102 | **HumanEval-Infilling** | 164 code files | Python | Exact match; execution-based | Ground-truth | 2022 | OpenAI (Bavarian et al.) |

---

## Robustness, hallucination, and code review benchmarks

A few benchmarks target less conventional but important dimensions:

- **ReCode** (2023, Amazon Science): Applies 30+ semantic-preserving perturbations to HumanEval/MBPP prompts to test robustness. Ground-truth via unit tests. Covers Python with extensions to Java, C++, JS.
- **CodeHalu** (2024/2025, Alibaba/UC Berkeley): First systematic benchmark for code hallucinations with **8,883 samples** across 4 hallucination types. Ground-truth via execution-based dynamic detection. Accepted at AAAI 2025.
- **CodeSearchNet** (2019, GitHub): ~6M functions for code search and summarization across 6 languages. Ground-truth via function-docstring pairs. Uses BLEU and F1.
- **ContextCRBench** (2025, ByteDance): 67,910 code review samples from 90 repos across 9 languages. Mixed evaluation — LLM-validated with 85%+ human agreement.
- **Scale SEAL** (2024, Scale AI): 1,000 expert-curated coding prompts. **Vibed** — uses expert human evaluation and LLM-as-judge, making it one of the few purely subjective coding benchmarks.

---

## The ground-truth vs. vibed landscape

The coding benchmark ecosystem is overwhelmingly ground-truth. Of the 102 benchmarks cataloged:

- **~85 benchmarks (83%)** use purely ground-truth evaluation — unit tests, execution-based verification, exact match against known outputs, or online judge systems
- **~12 benchmarks (12%)** use mixed evaluation — typically combining execution-based metrics with static analysis, BLEU scores, or LLM-as-judge components
- **~5 benchmarks (5%)** lean toward vibed evaluation — primarily in security (CyberSecEval's MITRE compliance tests), code review (ContextCRBench), general assessment (Scale SEAL), and markdown formatting (MDEval)

This ground-truth dominance is unique among AI evaluation domains. **Code is inherently verifiable** — a function either passes its test suite or it doesn't. This makes coding benchmarks more reliable than most NLP benchmarks, though caveats exist: test suites can be incomplete (motivating EvalPlus), static analysis has false positives (affecting security benchmarks), and match-based metrics like BLEU poorly correlate with functional correctness.

## Conclusion

Three trends define the current benchmark landscape. First, **contamination resistance** has become a first-class concern — LiveCodeBench, EvoCodeBench, ConDefects, and GitBug-Java all explicitly design around LLM training data leakage. Second, the field is rapidly moving from **function-level to repository-level** evaluation, with the SWE-bench family (7+ variants), Commit0, and RepoBench leading this shift. Third, **multi-language coverage** has expanded dramatically, from HumanEval's Python-only scope to McEval's 40 languages and mHumanEval's 204 natural languages. The most conspicuous gap remains in security evaluation, where the field lacks a universally accepted execution-based benchmark — CWEval and SecCodePLT are promising but not yet widely adopted. For practitioners choosing benchmarks, BigCodeBench (practical generation), SWE-bench Verified (agentic SE), and LiveCodeBench (contamination-free algorithmic tasks) represent the current best-in-class trifecta.