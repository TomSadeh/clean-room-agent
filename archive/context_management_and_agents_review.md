# The state of coding agents and context management in early 2026

**Context curation — not model capability — is the binding constraint for most production LLM applications, and this is especially visible in coding agents where the gap between benchmark performance and real-world utility remains stubbornly wide.** This survey covers two interrelated fronts: the architecture of every major open-source coding agent harness, and the research landscape on LLM context management. The findings converge on a single insight: every successful coding agent is fundamentally a context engineering system that happens to call an LLM, and the harnesses that perform best are those that most carefully control what enters and exits the context window. The white space — what nobody is building — centers on principled context curation pipelines, "default-deny" context architectures, and formal frameworks for measuring the marginal value of each token in context.

---

## Part 1: Open source coding agent harness survey

### How twelve harnesses manage context reveals a common architecture

Every major coding agent follows a surprisingly similar skeleton: a system prompt, some form of project context, tool definitions, and conversation history, all packed into a single context window that the LLM processes on each turn. The differences lie in *how* each harness curates what fills that window.

**Aider** pioneered the most sophisticated orchestrator-directed context strategy. Its **repo map** uses tree-sitter to parse every file into an AST, builds a NetworkX graph of identifier references between files, and runs **PageRank with personalization** to rank files by relevance to the current chat. The personalization vector weights chat files at +100/len(fnames), and edge multipliers boost identifiers mentioned in conversation (×50) while penalizing private names (×0.1). A binary search finds the maximum tag set fitting within the token budget (default **1,024 tokens**, expanding to ~8,192 when no files are active). For overflow, Aider uses a **weak model** (Haiku-class) to summarize chat history when it exceeds a configurable threshold, while recent turns remain verbatim. Aider never silently truncates — it reports API errors when limits are exceeded, leaving the user to manage context via `/drop`, `/clear`, or `.aiderignore`.

**SWE-Agent** takes the opposite approach: entirely model-directed exploration with no pre-computed context. Its core innovation is the **Agent-Computer Interface (ACI)** — purpose-built commands (`open`, `search_file`, `search_dir`, `edit`) replacing raw shell access. Context management uses **observation masking**: the `Last5Observations` processor collapses all but the five most recent tool outputs to single-line summaries, preserving the action taken but dropping verbose output. Failed edits that are subsequently fixed are removed entirely from history. The NeurIPS 2024 paper showed **51.7%** of trajectories have at least one failed edit, but agents recover 90.5% of the time. Critically, **mini-SWE-agent** — a 100-line Python script with only bash access — achieves **65%+** on SWE-bench Verified, suggesting much of SWE-Agent's tooling complexity may be unnecessary with newer models.

**OpenHands** implements the most sophisticated compaction system. Its **LLMSummarizingCondenser** triggers when events exceed a configurable threshold, keeping initial events (system prompts, first messages) verbatim while summarizing everything else via LLM. An **emergency fallback** cuts event history in half when a `ContextWindowExceededError` fires. Once condensation activates, average cost per turn drops to **less than half** the baseline. The design target is maintaining a fixed ~32K token context, making it viable for open-source models.

**Claude Code** operates within a 200K token window with a hierarchical allocation: **~5–15K** for the dynamically-assembled system prompt, **~1–10K** for CLAUDE.md files, **~500–2,000** per MCP server for tool schemas, **~40–45K** reserved for response generation (thinking + output), leaving **~140–150K** for conversation and tool results. Auto-compact triggers at **75–92%** usage. The compaction agent analyzes the conversation and replaces old messages with summaries, marking pruned tool outputs with `[compacted: tool output removed]`. Subagents are instructed to re-read files when encountering these markers. A notable design detail: with `ENABLE_TOOL_SEARCH=auto`, MCP tool schemas load on-demand rather than at session start, reducing initial overhead by **~85%**.

**Cursor** employs a VS Code fork architecture giving it full editor internals access. Its key innovation is the **apply model** — the main agent produces a "semantic diff" (changed code with insertion-point comments), and a cheaper, faster model writes the actual file. The system prompt is deliberately **static** (no personalized text), enabling full advantage of Anthropic's prompt caching. Cursor rules are NOT appended to the system prompt but appear as named descriptions fetched on-demand via a `fetch_rules()` tool call — a retrieval-based approach to prompt engineering. Cursor 2.0 runs up to **8 parallel agents** using git worktree isolation.

**Cline** calculates its effective context limit as `Math.max(contextWindow - 40_000, contextWindow * 0.8)`, reserving roughly 20–25% overhead. Its **Auto Compact** fires at ~80% usage, generating an LLM summary preserving decisions and code changes. A **duplicate file read optimization** replaces redundant content with `[DUPLICATE FILE READ]` notices. The team explicitly moved away from RAG/embedding-based retrieval toward letting the model direct its own context gathering — what they call "agentic search."

**Continue** differs by offering three modes. In Chat mode, users manually select context via `@` mentions (no tools sent). In Agent mode, all tools ship with each request and the model directs its own exploration. Its indexing stack combines **LanceDB** (vector embeddings via transformers.js, run locally), **tree-sitter** (AST parsing), and **ripgrep** (text search). A distinctive architectural choice: Continue implements "system message tools" — converting tool schemas into XML included in the system message rather than using native tool-calling APIs — providing universal compatibility across all models regardless of tool-calling support.

**OpenCode** (anomalyco/opencode, **100K+ GitHub stars**) mirrors Claude Code's architecture closely. Auto-compact triggers at ~95% of the model's context window. The compaction prompt explicitly asks for "a detailed prompt for continuing our conversation... what we did, what we're doing, which files we're working on, and what we're going to do next." Seven built-in agents include a dedicated compaction agent, explore agent (fast, read-only), and plan agent (no editing tools). OpenCode supports the `AGENTS.md` convention and has a plugin ecosystem including persistent memory via Letta.

**OpenClaw** (formerly Clawdbot, **217K+ GitHub stars**) is a conversation-first autonomous agent using messaging platforms (WhatsApp, Telegram, Slack) as its primary interface, not primarily a coding agent. Its **3-tier memory** uses a Filing Cabinet (persistent Docker volume), Notepad (session context), and Rolodex (config). Context overflow is handled by truncating oversized tool outputs and compacting oldest tool-result messages.

**Devon** and **Mentat** represent earlier, less mature approaches. Devon's development has slowed; it reads ~100 lines at a time with basic session-scoped state. Mentat (AbanteAI) has been archived as a CLI tool and pivoted to a GitHub App service; it originally relied on manual file inclusion with an optional **8,000-token** auto-context via embeddings.

### The tool landscape is converging on a standard set

Across all harnesses, a remarkably consistent tool vocabulary has emerged:

| Capability | Aider | SWE-Agent | OpenHands | Claude Code | Cursor | Cline | Continue | OpenCode |
|---|---|---|---|---|---|---|---|---|
| File read | Implicit | `open` | `FileEditorTool` | `Read` | `read_file` | `read_file` | `read_file` | `read` |
| File edit | Text format | `edit` (line range) | `FileEditAction` | `Edit` | `edit_file` (semantic diff) | `replace_in_file` | `edit_existing_file` | `edit` |
| Search (text) | Via repo map | `search_file/dir` | Bash | `Grep` | `grep_search` | `search_files` | `grep_search` | `grep` |
| Search (files) | Via repo map | `find_file` | Bash | `Glob` | `file_search` | `list_files` | `glob_search` | `glob` |
| Shell | `/run` | Blocked (custom cmds) | `CmdRunAction` | `Bash` | `run_command` | `execute_command` | `run_terminal_command` | `bash` |
| Web | No | No | `BrowserInteractiveAction` | `WebSearch`/`WebFetch` | `web_search` | `browser_action` | `fetch_url_content` | `web_fetch` |
| Subagent | No (architect mode only) | No | `AgentDelegateAction` | `Task` | Parallel agents | `new_task` | Background agents | `task` |

**Tool injection strategy** divides into two camps. Most harnesses (Cline, Claude Code, OpenCode) include all tool definitions in every request. Continue and Cursor are exceptions — Continue uses system-message XML for universal model compatibility, while Cursor makes rules fetchable on-demand via tool calls. Claude Code's lazy-loading MCP tools represent the most sophisticated approach: tool schemas appear only when needed.

**Tool failure handling** is universally basic. SWE-Agent runs a linter after every edit, rejecting syntactically invalid changes. Claude Code and Cursor cap linter-fix loops at 3 iterations. Most harnesses simply return error output to the LLM and let it retry. No harness implements sophisticated failure classification (distinguishing tool formatting errors from reasoning errors from environment errors) or adaptive retry strategies.

### Memory systems range from nonexistent to hierarchical

**Claude Code** has the most layered memory: enterprise policy → project `.claude.md` → project rules (`.claude/rules/*.md` with glob-based conditional loading) → user memory (`~/.claude/CLAUDE.md`) → auto-memory (per-project `MEMORY.md` with first 200 lines loaded per session, topic files loaded on-demand). Cross-session persistence is supported via `claude --continue`.

**Cline's Memory Bank** uses `.clinerules` markdown files specifying architecture plans and tribal knowledge. **Continue's** rules system (`.continue/rules/*.md`) provides hierarchically-scoped persistent instructions. **OpenCode** stores sessions in SQLite and supports memory plugins. **Aider** has `.aider.conventions.md` for project-specific instructions but no cross-session learning.

**No harness implements genuine cross-project learning.** Every agent starts functionally fresh on each new project, despite the fact that patterns in debugging, testing, and code organization transfer across codebases. The closest approach is SWE-Agent's trajectory files (`.traj`), which can serve as demonstrations for future runs, and Voyager's transferable skill library (from game environments, not coding).

### Pipeline structures increasingly favor multi-agent delegation

The field has moved decisively toward multi-agent architectures in 2025-2026. **Claude Code** runs specialized subagents: Explore (Haiku 4.5, read-only), Plan (Sonnet/Opus, read-only), and custom agents defined as Markdown files. Each subagent operates in a **fresh context window**, and only the condensed result flows back — preventing context pollution of the main conversation. **Cursor 2.0** runs up to 8 parallel agents with git worktree isolation. **OpenCode** and **Cline** support task delegation to child sessions. **OpenHands** pioneered delegation between agent types (CodeActAgent → BrowsingAgent).

**Aider's architect mode** represents a different multi-model pattern: an architect model describes the solution in plain text, then an editor model translates to specific file edits. This achieved **85%** on Aider's benchmark with o1-preview + DeepSeek as editor — the highest score at the time.

The key insight from multi-agent patterns is that **context isolation is the primary benefit**, not parallelism. Each subagent's fresh context window ensures it processes only the information relevant to its subtask. Anthropic reported **90% improvement** from multi-agent architectures, but at **15× token cost**.

### Model dependency varies from fully agnostic to locked-in

**Fully model-agnostic** (via litellm or equivalent): Aider, SWE-Agent, OpenHands, Cline, Continue, OpenCode, OpenClaw. These support 30–100+ providers including local models via Ollama.

**Provider-specific**: Claude Code (Anthropic only, uses Sonnet/Opus for reasoning, Haiku for summarization/compaction). Cursor supports multiple providers but relies on a proprietary apply model and tab-completion model.

**Performance with smaller models** degrades in predictable ways: inability to follow structured output formats (Mentat's diff format), exhaustion of context windows (Cline requires more aggressive truncation), and reduced tool-calling reliability. Continue's system-message-tools approach mitigates tool-calling compatibility issues. SWE-Agent's ACI design specifically helps smaller models by constraining the action space.

### Where agents break: reasoning versus context is a false dichotomy

Failure analysis reveals that the "reasoning vs. context" framing misses the real dynamic. From SWE-bench Pro trajectory analysis:

- **Frontier models** fail primarily on **wrong solutions** (35–52% of failures) — semantically plausible but fundamentally incorrect code. This looks like a reasoning failure, but is often traceable to incomplete context about requirements, edge cases, or architectural constraints.
- **Smaller models** fail primarily on **tool formatting, context overflow, and infinite exploration loops** — clearly context/information management failures.
- **Both** suffer from **context poisoning**: once an agent spends thousands of tokens exploring a wrong path, it struggles to ignore that exploration even when redirected. As one HN commenter noted: "A next-token predictor can't 'forget' context."

The **MAST taxonomy** (Berkeley, 1,642 traces across 7 multi-agent frameworks) found failures distributed roughly equally across specification/system design (~37%), inter-agent misalignment (~31%), and task verification/termination (~31%). Overall failure rates ranged from **41% to 86.7%**.

The **planner-coder gap** (arXiv:2510.10460) found that semantically equivalent inputs cause **7.9%–83.3%** of previously solved problems to fail when passed between planning and coding agents — information loss during multi-stage transformation is a major unaddressed failure mode.

A critical benchmark insight: scaffolding matters enormously. GPT-4's SWE-bench Lite performance varied from **2.7% to 28.3%** depending on scaffold — a **>10× difference** from the same model. SWE-bench Verified scores with the Live-SWE-agent scaffold reach **79.2%** (Claude Opus 4.5), versus ~45% for the same model without sophisticated scaffolding. The harness, not the model, is often the binding constraint.

### Seven gaps no current harness addresses

1. **"Default deny" context architecture.** Every harness is additive — retrieve, then stuff, then hope the model ignores noise. No system treats context as a secure perimeter where information must be explicitly validated before admission.

2. **Hierarchical abstraction-level navigation.** No agent can dynamically zoom between project → module → class → function abstraction levels while maintaining coherent context at each level.

3. **Codebase-specific learned retrieval.** All harnesses use embedding similarity, AST-based graph analysis, or lexical search. None adapts a retrieval model to a specific project's structure during a session.

4. **Semantic deduplication.** When an agent reads the same file three times across a long session, most harnesses (except Cline's duplicate detection) include it three times. No system detects semantic redundancy across tool outputs.

5. **Cross-session persistent learning.** No harness accumulates repository-specific "muscle memory" — learning that certain test patterns, debugging strategies, or code patterns work well in a specific codebase. Claude Code's auto-memory is an early step.

6. **Context provenance tracking.** When an agent hallucinates, there is no way to trace which context element contributed. No system tracks where each piece of context originated, how it was transformed, or its reliability.

7. **Formal verification integration.** Type checkers, static analyzers, and theorem provers remain afterthoughts rather than first-class tools integrated into the agent's planning loop.

The most striking emerging pattern is **Live-SWE-agent**, which starts with only bash access and **synthesizes its own tools at runtime** — creating editors, search utilities, and domain-specific tools on-the-fly. It achieves **79.2%** on SWE-bench Verified without any pre-defined tool set, suggesting the entire tool-definition paradigm may be an unnecessary constraint.

---

## Part 2: LLM context management — research and practice

### The empirical case against naive context stuffing is overwhelming

The research literature has converged on a clear finding: **adding more context hurts more than it helps, and the degradation is non-linear.**

The seminal **"Lost in the Middle"** paper (Liu et al., Stanford/Meta, TACL 2024) established that LLM performance follows a **U-shaped curve** — models perform best when relevant information sits at the beginning or end of context, with accuracy dropping ~20% for middle-positioned content. Follow-up work by Hsieh et al. (ACL 2024) traced this to an **intrinsic U-shaped attention bias** in transformer architectures, proposing a calibration mechanism that improves RAG performance by up to **15 percentage points**.

The **RULER benchmark** (NVIDIA, COLM 2024) found that of models claiming ≥32K context, **only half maintain satisfactory performance at 32K** on complex tasks. Performance degradation follows an **approximately linear trend on a log scale** within trained context lengths. GPT-4 showed the least degradation (15.4 points from 4K to 128K) but still degraded significantly.

**Context Rot** (Chroma Research, July 2025) tested 18 frontier models and found performance degrades on "deliberately simple tasks with constant complexity" as input length increases. On the **NoLiMa benchmark**, 11 of 12 models dropped below **50%** of their short-context performance at 32K tokens. Standard needle-in-a-haystack tests mask this because they only test lexical retrieval; introducing semantic matching or distractors reveals the real degradation curve.

The distraction effect is quantified precisely. **GSM-DC** (Yang et al., EMNLP 2025) showed accuracy degrades as distractor count increases following a **power-law trend whose exponent grows with reasoning depth** — meaning the penalty for irrelevant context compounds as tasks get harder. **GSM-IC** (Shi et al., ICML 2023) found fewer than **30%** of base problems were solved consistently after adding topically-adjacent distractors.

Even when context is "perfectly" curated, length alone degrades performance. Du et al. (2025) showed that with 100% exact-match retrieval, performance still drops **13.9%–85%** as input length increases — even when irrelevant tokens are replaced with whitespace or masked entirely. The mechanism appears to be attention dilution: since attention is zero-sum, adding tokens monotonically increases noise in representations regardless of content relevance.

### Model-side solutions partially address the problem

**Sparse attention** offers the most promising model-side approach. DeepSeek's **Native Sparse Attention (NSA)** combines token compression, token selection, and sliding window in a hardware-aligned design, pre-trained from scratch (27B parameters, 260B tokens). NSA **outperforms full attention** on MMLU, BBH, GSM8K, and HumanEval — suggesting sparsity doesn't just save compute but actually improves quality by reducing attention noise. **MInference** reduces pre-filling latency by **up to 10×** on A100s while maintaining accuracy at 1M tokens by identifying and exploiting three natural attention patterns (A-shape, Vertical-Slash, Block-Sparse).

**Alternative architectures** show mixed results. **Mamba** (Gu & Dao, COLM 2024) achieves linear-time scaling with 5× throughput over transformers, and Mamba-3B matches transformer models twice its size. However, SSMs generally underperform transformers on precise recall tasks. The practical solution is **hybrid architectures**: **Jamba** (AI21 Labs) interleaves Mamba and transformer layers with MoE, achieving effective 256K context with 8× smaller KV cache. **RWKV-X** combines RWKV-7 with sparse attention blocks, achieving near-perfect accuracy on 64K passkey retrieval with linear complexity.

**Context compression** techniques achieve remarkable ratios. **LLMLingua** (Microsoft Research) compresses prompts up to **20× with only 1.5 percentage point loss** on GSM8K using small-model perplexity scoring. **Gist tokens** (Mu et al., NeurIPS 2023) achieve **26× compression** and 40% FLOPs reduction. **AutoCompressors** (Princeton, EMNLP 2023) achieve **30:1 compression** using recursive segment compression. However, a 2025 comprehensive study found compressed representations "struggle to reconstruct original content" — they work for gist-level tasks but fail at precise information retrieval.

The most conceptually interesting model-side approach is **self-retrieval via attention**. **InfiniRetri** (Ye et al., 2025) leverages LLMs' own attention information to perform retrieval across unlimited-length inputs, outperforming all KV-cache compression methods — not just reducing costs but improving effectiveness. **SnapKV** demonstrated that "LLMs know what you are looking for before generation" — attention patterns are stable and predictive, enabling intelligent token selection before generation begins.

### Application-side solutions define the practical state of the art

RAG pipelines have evolved through four generations: **Naive RAG** (retrieve → concatenate → generate), **Advanced RAG** (query rewriting + reranking), **Modular RAG** (swappable components), and now **Agentic RAG** (autonomous agents directing retrieval dynamically). The current production standard combines hybrid search (BM25 + dense retrieval), reranking, and agent-directed retrieval decisions.

**Prompt caching** has become a critical production technique. Anthropic's implementation offers **90% cost reduction** on cached tokens with manual `cache_control` breakpoints. OpenAI provides automatic prefix caching with **50–90%** discounts. The key constraint is **exact prefix matching** — any token mismatch forces recomputation. This creates a strong architectural incentive to place static content (system prompt, policies, few-shot examples) at the beginning and dynamic content at the end. Cursor exploits this by making its entire system prompt static, enabling maximal cache hit rates.

**Context window budgeting** in production follows common patterns: reserve 25–50% for output, distribute the remainder across system prompt, retrieved documents, and conversation history. The **ACE framework** (Stanford/SambaNova, arXiv:2510.04618) formalizes this with a Generator→Reflector→Curator pipeline treating contexts as evolving "playbooks," achieving **+10.6%** on agent benchmarks by treating context as a first-class optimizable resource.

**System 2 Attention** (Weston & Sukhbaatar, 2023) is perhaps the most direct application-side approach to the context curation problem: a first LLM call regenerates the input context with irrelevant portions removed, then a second call processes the cleaned context. This is the closest existing work to a principled multi-prompt context curation pipeline.

### The research gaps form a clear map of unexplored territory

**Pre-filtering as an alternative to RAG.** The most significant gap identified. All current RAG systems are additive: retrieve, then optionally filter. **FILCO** (Wang et al., 2023) is the closest work — it trains models to filter retrieved contexts using lexical overlap and information-theoretic approaches, reducing prompt lengths by **up to 64%**. But no work frames context management as "default deny" — where nothing enters context unless explicitly validated. The security analogy (allowlist vs. blocklist) is obvious but unresearched.

**Multi-prompt context curation pipelines.** Beyond S2A's two-stage approach and ACE's three-stage pipeline, no work systematically studies using chains of prompts where each stage distills context for the next. No standardized metrics exist for measuring context quality at each pipeline stage. No formal optimization framework determines how many curation steps are worth the compute cost.

**Separation of retrieval and reasoning with specialized models.** Despite the trend toward model routing (cheap models for classification, expensive for reasoning), no research systematically assigns retrieval to one model and reasoning to another, optimized for their respective tasks. **RAFT** (UC Berkeley, 2024) teaches a single model to distinguish relevant from distractor documents, improving performance by **up to 76%** over baselines, but doesn't separate the responsibilities across different systems.

**Growing knowledge bases from agent activity.** **Voyager** (2023) built ever-growing skill libraries for Minecraft agents; **A-MEM** (2025) uses the Zettelkasten method for interconnected knowledge networks; **Reflexion** stores textual reflections from failures. But no open-source coding agent implements persistent, self-updating knowledge bases that accumulate codebase-specific understanding over time.

**Context-aware retrieval** is perhaps the most surprising absence. Current retrievers select documents based on query-document similarity but have **no awareness of what's already in context**. A retriever that knows the model already has file A should retrieve different information than one where the model has no prior context — but no system implements this.

### Pressure-testing the thesis: context curation versus model capability

The thesis — "the primary bottleneck in LLM application performance is not model capability but context curation" — finds **strong support for production applications but faces meaningful counterevidence at the frontier**.

**Supporting evidence is compelling for typical use cases.** Pinecone demonstrated that RAG preserves **95% of accuracy with only 25% of tokens** — a 4:1 compression ratio. AppFolio achieved a jump from **40% to 80%** performance through dynamic few-shot prompting (a context curation technique), not model upgrades. Fine-tuned ModernBERT outperformed Claude Haiku by **30% in accuracy** while being **69× faster** and **225× cheaper**. The GPT-4 SWE-bench scaffold variation (**2.7% to 28.3%** from the same model) provides the starkest evidence that context management can produce 10× performance differences with fixed model capability.

**Counterevidence is strong for frontier tasks.** Scaling laws demonstrate model size has profound, predictable effects. On SWE-bench with standardized scaffolds, model capability still dominates rankings — Claude Opus 4.5 consistently beats smaller models regardless of scaffold. Larger models are **more context-responsive**, not less: Google Research (2023) showed overriding prior knowledge is an emergent ability of scale. On SWE-bench Pro (harder, enterprise-grade problems), context augmentation yields only modest gains, suggesting model reasoning is the binding constraint for genuinely difficult tasks.

**The most accurate formulation** is that context curation and model capability operate on different parts of the difficulty spectrum. For the median production task, marginal improvements in context curation deliver higher ROI than model upgrades. For tasks at the frontier of model capability, no amount of context curation compensates for insufficient reasoning ability. **Qwen3-Coder-Next** (3B active parameters) achieving SWE-bench-Pro performance comparable to models with **10–20× more parameters** demonstrates that the Pareto frontier of capability-per-token is advancing rapidly, making context curation even more important as smaller models become viable.

The **SWE-ContextBench** (February 2026) provides the most direct validation: correctly selected, summarized experience improves resolution accuracy and substantially reduces runtime and token cost — but **unfiltered or incorrectly selected experience provides limited or negative benefits**. This is the thesis in microcosm.

---

## Conclusion: the field is converging, but the hardest problems are untouched

Three insights emerge from mapping both landscapes simultaneously.

**First, coding agents are converging on a common architecture** — agentic loop, model-directed file exploration, LLM-based context compaction, multi-agent delegation with context isolation, hierarchical memory files. The variation between Claude Code, OpenCode, Cline, and OpenHands is now largely in implementation quality rather than architectural novelty. The fact that mini-SWE-agent (100 lines, bash only) achieves 65%+ on SWE-bench Verified suggests that with frontier models, most scaffolding complexity is overhead rather than value-add.

**Second, the research literature has thoroughly documented the problem but barely begun solving it.** We know context degrades performance non-linearly, that irrelevant information actively harms reasoning, that models use their context windows with a strong positional bias, and that longer context has diminishing returns. We have prototype solutions (sparse attention, context compression, S2A filtering). But no system combines these findings into a principled, end-to-end context curation pipeline.

**Third, the largest gaps cluster around treating context as a first-class engineered artifact.** Nobody is building "default deny" context architectures. Nobody is training specialized retrieval models that are aware of what's already in context. Nobody is formally measuring the marginal value of each token. Nobody is building systems where agents accumulate genuinely useful codebase-specific knowledge across sessions. The researcher building a coding agent harness today has a clear roadmap: the competitive edge is not in better tools or cleverer prompts, but in **the pipeline that decides what the model sees before it starts thinking**. Every token that enters the context window should earn its place — and right now, in every harness surveyed, most of them don't.