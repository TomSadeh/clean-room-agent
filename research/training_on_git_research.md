# Training code LLMs from Git history

**Git history represents one of the richest and most underutilized signals for training code language models.** Unlike static code snapshots, Git repositories encode how software evolves—diffs, commit messages, code reviews, merge conflicts, and temporal dependencies—providing natural supervision for tasks ranging from program repair to commit message generation. Since 2020, researchers have assembled datasets exceeding 67 terabytes from platforms like Software Heritage and GitHub, developed change-aware architectures that outperform general code models on editing tasks by 20%+, and demonstrated that commit messages serve as naturally occurring instruction-tuning data across 350+ programming languages. This report synthesizes the full landscape of methods, datasets, architectures, and practical tooling for leveraging Git history in LLM training, covering academic research and industry practice from 2020 through early 2025.

---

## How researchers extract and clean Git-derived training data

Training data from Git repositories falls into several categories: **commit diffs** (line-level changes in unified diff format), **file snapshots** at specific versions, **commit messages** (natural language paired with code changes), **pull request data** (multi-turn review discussions), **GitHub issues**, **blame history**, and **merge conflict records**. The extraction process typically begins with large-scale archival sources and proceeds through increasingly aggressive filtering.

**Software Heritage** has emerged as the canonical source for transparent, traceable code data. This UNESCO/Inria initiative archives 22+ billion unique source files across 345+ million projects, using a Merkle DAG structure with persistent identifiers (SWHIDs) that enable precise data provenance. The Stack v2 and StarCoder2 both source their data from Software Heritage rather than scraping GitHub directly. Software Heritage launched the CodeCommons project (€5M French government funding) specifically to provide ethically sourced, pre-cleaned code collections for AI training.

**PyDriller** (Spadini, Aniche & Bacchelli, ESEC/FSE 2018) is the standard Python framework for mining individual repositories, extracting commits, modifications, diffs, and source code with ~50% less code than raw GitPython. **GHTorrent** (Gousios, MSR 2013) mirrors GitHub's event streams into MongoDB/MySQL, covering users, repos, commits, pull requests, and issues—though it has known GDPR compliance issues since 2018 and less complete data after 2017. **GHArchive** collects GitHub's public event stream and served as the primary source for BigCode's pull request and issue data. **GrimoireLab** (Dueñas et al., 2021) from the CHAOSS/Linux Foundation project supports 30+ data sources with identity management via SortingHat.

The filtering pipeline for Git-derived data is aggressive. StarCoder's pipeline removes merge commits, bot commits (username keyword matching for "bot", "ci", "dependabot"), very short messages (<200 characters), and auto-generated code (files where encoded substrings exceed 50% of content). **Near-deduplication via MinHash LSH** is the standard approach: documents are decomposed into 5-gram shingles, MinHash signatures computed (128–256 permutations), and LSH groups similar documents with a Jaccard threshold typically between 0.7 and 0.85. BigCode's research confirmed that near-deduplication significantly boosts downstream model performance across all experiments. Approximately **40% of permissively licensed files in The Stack v2 were near-duplicates**. Additional filters target line length extremes, low alphanumeric percentages, HTML text-to-code ratios, and file size limits for data formats like JSON/YAML. PII redaction uses BigCode's StarPII NER model trained on 12,099 annotated files containing 22,950 entities (names, emails, API keys, SSH keys, passwords, IP addresses).

---

## Pre-training on code changes versus static snapshots

The field divides into two paradigms: large decoder-only models trained on static code snapshots (Codex, CodeLlama, DeepSeek-Coder) and smaller encoder-decoder models explicitly pre-trained on code changes (CCT5, CommitBART, CoditT5, CodeReviewer). The evidence consistently shows that **change-aware pre-training dramatically outperforms static-code models on editing and change-understanding tasks**, while static-snapshot models remain more versatile for general code generation.

**CCT5** (Lin et al., ESEC/FSE 2023) is the most comprehensive change-oriented pre-trained model. Built on CodeT5 (223M parameters), it was further pre-trained on 1.5M+ pairwise code changes and commit messages (39.6GB) from 35K repositories using five specialized tasks: masked language modeling for code changes, masked language modeling for commit messages, code-to-message generation, structure-aware code diff generation, and message-to-code generation. CCT5 achieved a **BLEU score of 22.06% on commit message generation**—a 22% relative improvement over CodeT5 (18.11%) and 20% over CodeReviewer (18.45%). A 2024 follow-up study found that even LLMs with parameter-efficient fine-tuning achieve only comparable, not superior, performance to the much smaller CCT5 on change-related tasks.

**CommitBART** (Liu et al., arXiv 2022; ACM TOSEM 2024) collected 7.99 million commits across seven languages and pre-trained a PLBART-based model using six tasks spanning denoising, cross-modal generation, and contrastive learning. It introduced special segment tokens ([MSG], [CODE], [NEG], [POS]) to distinguish commit messages from deleted and added code, and used contrastive objectives to align code change embeddings with message embeddings. **CoditT5** (Zhang et al., ASE 2022) took a different approach, introducing an edit-plan pre-training objective where the model generates explicit edit operations (keep, delete, replace-old, replace-new, insert) before producing the target sequence. This yielded up to **19.35% improvement** when combined with standard generation models through reranking, demonstrating that edit-based and generation-based approaches are complementary. **CodeReviewer** (Li et al., 2022, Microsoft) was pre-trained on 7.9 million pull requests using four tasks including diff tag prediction and line-level masking, achieving 8%+ F1 improvement on code change quality estimation.

Among large models, **StarCoder** (BigCode, 2023) included 32GB of Git commits in its 783GB training mix alongside 86 programming languages, 54GB of GitHub issues, and 13GB of Jupyter notebooks. For commit data, only 20% used full file content; the remaining 80% sampled a window of 0–32 lines around changed lines. **StarCoder2** (BigCode, 2024) expanded to include pull request diffs generated by comparing base and head commits retrieved from Software Heritage. **DeepSeek-Coder** (2024) innovated with repository-level organization—parsing file dependencies within repos and rearranging files based on dependency order—but did not use Git diffs explicitly. **Codex** (OpenAI, 2021) trained on 159GB of Python from 54 million GitHub repositories using only static code snapshots.

---

## Fine-tuning with commits as natural instructions

The most influential fine-tuning approach treats Git commits as naturally occurring instruction-tuning data. **OctoPack** (Muennighoff et al., ICLR 2024) created **CommitPack**, a 4TB dataset of Git commits across 350 programming languages extracted from GitHub's BigQuery dump. The filtered subset, **CommitPackFT** (2GB), applies strict quality filters: commit messages must start with instructive verbs ("Fix", "Add", "Update"), consist of multiple words, and not contain external references, yielding roughly 5,000 high-quality samples per language. The commit message serves as the instruction and the before/after code serves as input/output. **OctoCoder**—StarCoder-16B fine-tuned on CommitPackFT plus the OASST conversational dataset—achieved **46.2% pass@1 on HumanEval**, the best among models not trained on proprietary OpenAI outputs. This demonstrated that Git commits provide diverse, multilingual instruction data without requiring synthetic generation from closed-source models.

**Astraios** (BigCode, 2024) systematically compared seven parameter-efficient fine-tuning methods (full fine-tuning, LoRA, four adapter variants, prefix tuning, and prompt tuning) across four StarCoderBase scales (1B to 16B) using CommitPackFT + OASST. **LoRA offered the best cost-performance tradeoff**, while full fine-tuning led in absolute performance. Prefix tuning performed poorly for StarCoder architectures, and larger models showed reduced robustness and security after fine-tuning.

For code review specifically, **CodeReviewer** fine-tuning enables quality estimation (71.5% F1), review comment generation, and code refinement simultaneously. The **D-ACT** approach (SANER 2023) integrates token-level diff information and revealed a critical evaluation issue: **performance drops 57–94% under chronological train/test splitting** compared to random splitting, meaning most published results dramatically overestimate real-world performance on code review tasks.

---

## Downstream tasks spanning the software lifecycle

Git history data powers a remarkably broad set of downstream applications, each exploiting different aspects of version control data.

**Commit message generation** has progressed from retrieval-based methods (NNGen, 2018) through neural machine translation (ATOM, 2020) to LLM-based approaches. ERICommiter (Xue et al., IEEE TSE 2024) demonstrated that GPT-4o with a single retrieved example achieves **89% BLEU improvement** over zero-shot. CoRaCMG (2025) showed DeepSeek-R1 achieving 76% BLEU and 71% CIDEr improvements with retrieval augmentation. Human evaluation consistently favors LLM-generated messages over human-written ones (judged best in 78% of cases in one 2024 study), suggesting automated metrics underestimate LLM quality.

**Automated program repair** learns from bug-fixing commits. TFix (Berabi et al., ICML 2021) fine-tuned T5-large on GitHub commits fixing ESLint errors, achieving ~67% fix rate on JavaScript. ChatRepair (2024, ISSTA) used conversational LLM interaction with few-shot examples from project history, fixing 162 of 337 bugs at $0.42 each. AlphaRepair (2022, FSE) pioneered zero-shot repair using CodeBERT's masked language model without any fine-tuning on bug-fixing data.

**Merge conflict resolution** uses historical merge commits as training data. DeepMerge (Dinella et al., FSE 2021, Microsoft) achieved 78% accuracy using edit-aware embeddings and pointer networks on JavaScript conflicts—**9× higher than structured merge tools**. MergeBERT (Svyatkovskiy et al., ESEC/FSE 2022, Microsoft) scaled to 220,000 real-world conflicts across four languages, achieving 63–68% accuracy using a multi-input BERT variant that formulates resolution as classification over primitive merge patterns.

**Vulnerability detection from patch history** exploits security-fixing commits. Datasets include Devign (Zhou et al., NeurIPS 2019, 26K functions), Big-Vul (Fan et al., MSR 2020, 3,754 vulnerabilities), and DiverseVul (Chen et al., 2023, 18,945 vulnerable functions across 150 CWE types). However, **label noise is pervasive—17.4% to 50% of "vulnerable" labels may be incorrect**. PRIMEVUL (2024) revealed that StarCoder2's F1 drops from 68.26% on Big-Vul to just 3.09% on its higher-quality benchmark, exposing severe overestimation in prior work.

**Just-in-time defect prediction** classifies commits as defect-introducing using the SZZ algorithm for labeling. CC2Vec (Hoang et al., ICSE 2020), DeepJIT (Hoang et al., MSR 2019), and BiCC-BERT (2025) all learn from commit features, though a sobering finding from an ISSTA 2021 replication study showed that a logistic regression model using just "added-line-number" as a feature can outperform deep models while being **81,000× faster**.

---

## Architectures designed to understand code changes

Standard transformer architectures struggle with code diffs because they were designed for static sequences, not structured change representations. Several specialized approaches address this gap.

The foundational work is **"Learning to Represent Edits"** (Yin, Neubig, Allamanis et al., ICLR 2019), which introduced an edit encoder that takes (before, after) code pairs and produces fixed-length edit representation vectors. The key architectural insight is a bottleneck design: by limiting the edit encoder's output capacity while allowing free access to the original code, the model captures only the salient edit information. Edit representations of semantically similar changes cluster in embedding space, enabling unsupervised pattern discovery. **"Learning Structural Edits via Incremental Tree Transformations"** (Yao et al., ICLR 2021) extended this to AST-level edits, modeling changes as consecutive tree transformations with imitation learning, achieving **28% relative gains** over sequential models.

**CC2Vec** (Hoang et al., ICSE 2020) introduced a hierarchical attention architecture processing diffs at four levels: tokens → lines → hunks → files, with parallel RNN branches for added and removed code and comparison functions to identify differences. The model is trained with commit messages as supervision and produces general-purpose code change vectors applicable to multiple downstream tasks.

For input representation, three main strategies have emerged:

- **Unified diff with special tokens**: CCT5 uses [ADD] and [DEL] tokens to mark line types, avoiding duplication of unchanged code. CodeReviewer processes complete diff hunks including @@ headers and explicitly trains the model to predict diff tags (ADD/DEL/KEEP).
- **Edit plan representation**: CoditT5 generates sequences of explicit edit operations (INSERT, DELETE, REPLACE, KEEP) before the target, forcing the model to reason about what changed rather than regenerating everything.
- **Token-level diff marking**: D-ACT annotates individual changed tokens within lines, providing finer-grained change information than line-level markers. This approach improved code review performance by 17–82% compared to line-level approaches.

**GumTree** (Falleri et al., ASE 2014) remains the standard tool for fine-grained AST differencing, producing edit scripts with insert, remove, update, and move operations. Multiple neural approaches use GumTree-generated edit scripts as structured input. Most models remain constrained to **512-token context windows**, forcing processing at the method or hunk level rather than full commits. Repository-level change understanding is handled by agent-based systems like CodePlan (Microsoft, FSE 2024) rather than end-to-end neural models.

---

## Temporal ordering matters more than most realize

The temporal structure of Git history—commits ordered chronologically, code evolving over time—provides important learning signals, but also creates evaluation pitfalls that much of the literature has ignored.

The most impactful finding comes from **D-ACT** (SANER 2023), which demonstrated that switching from random to chronological train/test splitting causes **57–94% performance drops** on code review transformation tasks. AutoTransform lost 92% of its reported accuracy; TufanoT5 dropped 57%. This means most published results on code change tasks are unrealistically optimistic. Chronological splitting—training only on commits that preceded the test set temporally—is essential for realistic evaluation but remains underused.

**CodePlan** (Bairi et al., FSE 2024, Microsoft) explicitly leverages both spatial context (cross-file dependencies via static analysis) and temporal context (history of edits to the repository) for LLM prompting. Ablation studies showed both context types contribute significantly—without temporal context, models lack awareness of the evolutionary trajectory of the code. CodePlan successfully handled changes spanning 2–97 files per repository on tasks like package migration, while baselines without temporal context failed entirely.

Curriculum learning—ordering training examples from easy to hard based on commit properties—has shown mixed results for code models. A 2025 study (Khant et al.) found that CodeT5 exhibited catastrophic forgetting and shortcut learning under curriculum learning, with performance saturating after only the first quartile of training. However, a 2024 study on FIM code completions identified AST node types where completions fail frequently and used these as curriculum examples, achieving gains especially for smaller models. The evidence suggests curriculum learning based on commit history may benefit small models but offers diminishing returns for larger ones.

---

## Eight major datasets and how they were built

The landscape of Git-derived datasets has expanded dramatically since 2020, with the largest now exceeding 67 terabytes.

| Dataset | Year | Size | Languages | Git Data Used | Key Feature |
|---------|------|------|-----------|---------------|-------------|
| **The Stack v2** | 2024 | 67.5TB raw, ~900B tokens | 600+ | Files, commits, PRs, issues | Software Heritage–sourced, SWHID-traceable |
| **CommitPack** | 2023 | 4TB | 350 | Commit diffs + messages | Natural instruction-tuning data |
| **The Stack v1** | 2022 | 6.4TB (3.1TB permissive) | 358 | File snapshots | First large-scale opt-out mechanism |
| **StarCoderData** | 2023 | 783GB (250B tokens) | 86 + issues + commits | Code + 32GB commits + 54GB issues | Multi-modal training mix |
| **CommitBench** | 2024 | 1.66M examples | 6 | Diffs + messages | Bot-filtered, cross-project splits |
| **MegaDiff** | 2021 | 663K diffs | Java | Fixing commit diffs | Categorized by diff size (1–40 lines) |
| **CodeSearchNet** | 2019 | 2M pairs, 6M functions | 6 | Function-level snapshots | Expert relevance annotations |
| **DiverseVul** | 2023 | 18,945 vulnerable functions | C/C++ | Security-fixing commits | 150 CWE types, 7,514 commits |

**The Stack v2** represents the current state of the art in dataset construction. Built entirely from the Software Heritage archive, it includes not just code files but GitHub issues (from GHArchive, with bot filtering and quality controls), pull request diffs (generated by comparing base/head commits), Jupyter and Kaggle notebooks, code documentation, and natural language datasets for math and coding. All data is identifiable via SWHIDs for provenance traceability. The deduplication pipeline uses 5-grams with Jaccard threshold 0.7, prioritizing files from higher-star repos.

**CommitPack** is uniquely valuable because it transforms Git history into instruction-tuning format without synthetic data generation. Its filtered subset CommitPackFT applies verb-initial commit message filters and restricts to single-file, permissively licensed commits, yielding high-quality instruction-response pairs across more languages than any synthetic alternative covers. **CommitBench** addressed dataset quality issues in the commit message generation literature by implementing five-stage preprocessing (deduplication, bot removal, language normalization, diff-length filtering, language balancing) and enforcing cross-project train/test splits.

---

## Navigating the practical challenges

Several challenges consistently emerge across the literature and require careful handling in any Git-based training pipeline.

**Data quality remains the primary bottleneck.** After filtering, approximately 63% of raw commits are discarded as noise. Commit messages are frequently uninformative—53% don't follow verb-direct-object structure, 18% contain multiple sentences, and many are single words like "fix" or "update." Applying message length (≤30 tokens) and diff length (≤100 tokens) filters to one 1.8M-commit dataset reduced it to just 75K usable examples, a **96% reduction**. Vulnerability datasets face even worse quality issues, with mislabeling rates of 17–50%.

**Licensing exposure is legally unresolved.** The GitHub Copilot class-action lawsuit (filed November 2022) had most copyright claims dismissed by July 2024, but **breach-of-license and breach-of-contract claims continue**. Best practice is to use only permissively licensed code with automated detection (go-license-detector) and implement opt-out mechanisms. BigCode's "Am I in The Stack?" tool processed 44 opt-out requests before StarCoder training and updates approximately quarterly.

**Benchmark contamination is pervasive and underappreciated.** Studies found **8–18% of HumanEval overlaps with pre-training data** in major datasets. A 13B model fine-tuned on rephrased HumanEval samples matched GPT-4 performance, demonstrating contamination severity. Simple n-gram decontamination (the standard approach) fails against paraphrasing, translation, or dead-code injection. HumanEval-T, which generates combinatorial variants of problems, showed **5–14 percentage point drops** across all tested LLMs. LiveCodeBench (Jain et al., 2024) addresses this by collecting problems from competitive programming contests posted after model training cutoffs.

**Information leakage between train and test splits** from the same repository inflates results because models exploit project-specific conventions. Cross-project splitting (enforced by CommitBench) and chronological splitting (demonstrated critical by D-ACT) are both necessary for realistic evaluation. The recommended pipeline combines both: split first by project, then chronologically within each partition.

---

## Recommended pipeline for building Git-based training data

Based on the StarCoder2, CommitBench, and BigCode best practices, the following pipeline represents current consensus:

1. **Source from Software Heritage** or GHArchive for provenance traceability via SWHIDs
2. **License filter** using go-license-detector or ScanCode, retaining only permissive licenses (MIT, Apache-2.0, BSD)
3. **Process opt-out requests** from dedicated developer communication channels
4. **Exact deduplication** via SHA-256 hashing (fast first pass)
5. **Near-deduplication** via MinHash LSH with 5-grams, 128+ permutations, Jaccard threshold ≥0.7, prioritizing higher-star repositories
6. **Bot and CI filtering** via username keyword matching ("bot", "[bot]", "dependabot", "renovate", "greenkeeper")
7. **Auto-generated code removal** using regex detection for encoded data, checking HTML visible text ratios, and size-filtering data formats
8. **PII redaction** using StarPII or equivalent NER model trained on code-specific PII annotations
9. **Secrets scanning** via detect-secrets, Gitleaks, or TruffleHog (regex + entropy detection)
10. **Benchmark decontamination** using exact string match on both solutions and prompts from HumanEval, MBPP, and other evaluation sets, supplemented by semantic matching
11. **Quality filtering** for commits: message length, V-DO structure checks, diff size limits, single-file restriction for instruction-tuning data
12. **Language balancing** for evaluation sets; windowed context (0–32 lines around changes) for training efficiency

Key tools include BigCode's bigcode-dataset repository (extraction, filtering, decontamination scripts), PyDriller for targeted repository mining, fastdedup or datatrove for deduplication at scale, and BigCode's PII library for redaction. For evaluation, the combination of HumanEval/MBPP (standard benchmarks), LiveCodeBench (contamination-free), BigCodeBench (compositional tasks), CommitBench (commit-specific), and CodeXGLUE (multi-task) provides comprehensive coverage.

---

## Conclusion

The research landscape for training code LLMs from Git history reveals several clear patterns and open opportunities. **Change-aware pre-training using relatively small encoder-decoder models (CCT5, CoditT5, CodeReviewer, ~223M parameters) consistently outperforms much larger general-purpose models on editing and change-understanding tasks**, establishing that how code evolves is fundamentally different from what code looks like at any point in time. The OctoPack line of work proved that Git commits serve as high-quality, naturally occurring instruction data spanning hundreds of languages without relying on synthetic generation from proprietary models—a finding with significant practical implications for training open-weight models.

Three underappreciated insights deserve emphasis. First, temporal evaluation reveals that **most published results on code change tasks are dramatically overoptimistic**, with performance dropping 57–94% under realistic chronological splitting. Any practitioner building on this literature should re-evaluate baseline numbers accordingly. Second, the complementarity between edit-based and generation-based models (demonstrated by CoditT5's reranking experiments) suggests that ensemble approaches combining change-aware specialist models with large general-purpose LLMs represent a promising but underexplored direction. Third, the vulnerability detection literature's label noise problems (17–50% mislabeling rates) and PRIMEVUL's finding of F1 dropping from 68% to 3% on higher-quality benchmarks suggest that Git-derived security datasets require substantially more investment in label quality before claims of effective automated vulnerability detection can be trusted.

The field is moving toward repository-level understanding (CodePlan, DeepSeek-Coder's dependency-aware organization) and agent-based architectures that treat Git history as navigable context. The 512-token context window limitation that constrains most current change-aware models will likely be addressed as long-context architectures mature, potentially enabling end-to-end models that reason over entire commit histories rather than isolated hunks. The infrastructure is maturing—Software Heritage provides traceable archival data, BigCode's pipeline offers reproducible preprocessing, and datasets like CommitPack and The Stack v2 provide the scale needed for pre-training. The primary remaining challenges are evaluation methodology (temporal splitting, contamination), data quality (commit message informativeness, vulnerability labels), and the fundamental architectural question of how to efficiently encode structured change information at scale.