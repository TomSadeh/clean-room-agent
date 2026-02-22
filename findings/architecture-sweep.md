# OpenClaw Architecture Sweep

**Date**: 2026-02-22
**Purpose**: Reconnaissance only — understand architecture, identify intervention points, assess feasibility.

---

## What OpenClaw Is

OpenClaw is a personal AI assistant platform — a local-first, self-hosted gateway that connects LLMs to messaging channels and tools. It is not a coding agent in the narrow sense; it has a coding agent as one of 53 bundled skills, but the core is a general-purpose agent runtime.

- 217k GitHub stars, MIT license
- TypeScript monorepo (pnpm), Node >= 22
- Core SDK: `@mariozechner/pi-coding-agent` + `@mariozechner/pi-agent-core`
- Gateway: WebSocket control plane at `ws://127.0.0.1:18789`
- Channels: WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Teams, Matrix, Google Chat, WebChat

---

## Repository Structure

| Directory | Purpose |
|-----------|---------|
| `src/agents/` | Core agent runtime (418 files) — context, tools, skills, model selection, sessions |
| `src/agents/pi-embedded-runner/` | Main agent execution engine (41 files) — run loop, compaction, system prompt, tool truncation |
| `src/agents/tools/` | Tool definitions (74 files) — memory, web, browser, messaging, sessions, cron |
| `src/agents/skills/` | Skill loading and configuration (16 files) |
| `src/memory/` | Persistent memory system (72 files) — embeddings, vector search, hybrid retrieval |
| `src/sessions/` | Session management (8 files) — history limits, model overrides |
| `src/providers/` | Provider-specific auth (10 files) — GitHub Copilot, Google, Qwen |
| `src/channels/` | Messaging channel integrations |
| `src/config/` | Configuration loading, validation, Zod schemas |
| `src/gateway/` | WebSocket gateway control plane |
| `skills/` | 53 bundled skill packages |
| `docs/` | 28 concept docs |

---

## Context Management — Five Layers

### Layer 1: System Prompt Assembly
**Files**: `src/agents/system-prompt.ts`, `src/agents/pi-embedded-runner/system-prompt.ts`

Rebuilt from scratch on every run. Assembled from modular sections:
- Identity, tooling, safety, skills (compact list only), memory instructions
- Injected context files: `AGENTS.md`, `SOUL.md`, `TOOLS.md`, `IDENTITY.md`, `USER.md`, `HEARTBEAT.md`, `BOOTSTRAP.md`
- Each context file truncated to 20k chars, total cap 150k chars
- Three modes: `"full"` (main agent), `"minimal"` (sub-agents), `"none"` (bare identity)

### Layer 2: Conversation History Limits
**File**: `src/agents/pi-embedded-runner/history.ts`

- `limitHistoryTurns()` counts backward through messages by user turns
- Per-DM and per-channel limits configurable in `openclaw.json`

### Layer 3: Tool Result Truncation
**File**: `src/agents/pi-embedded-runner/tool-result-truncation.ts`

- `MAX_TOOL_RESULT_CONTEXT_SHARE`: 30% of context window per result
- `HARD_MAX_TOOL_RESULT_CHARS`: 400,000 characters absolute ceiling
- ~4 chars = 1 token heuristic
- Truncates at newline boundaries when possible

### Layer 4: `transformContext` Hook (Primary Intervention Point)
**File**: `src/agents/pi-embedded-runner/tool-result-context-guard.ts`

Intercepts the full message array just before every LLM call:

```typescript
export function installToolResultContextGuard(params: {
  agent: GuardableAgent;
  contextWindowTokens: number;
}): () => void {
  mutableAgent.transformContext = (async (messages, signal) => {
    const transformed = originalTransformContext
      ? await originalTransformContext.call(mutableAgent, messages, signal)
      : messages;
    enforceToolResultContextBudgetInPlace({ messages: contextMessages, ... });
    return contextMessages;
  });
}
```

Key constants:
- `CONTEXT_INPUT_HEADROOM_RATIO`: 0.75 (75% of context window for input)
- `SINGLE_TOOL_RESULT_CONTEXT_SHARE`: 0.5 (no single result exceeds 50%)
- Two-phase enforcement: truncate oversized individual results, then compact oldest tool results until total fits budget

### Layer 5: LLM-Based Compaction
**File**: `src/agents/pi-embedded-runner/compact.ts`

- Auto-triggers when approaching limits
- Uses the LLM itself to summarize older turns
- Runs a "memory flush" turn first to save important info to disk
- Summary persists in session JSONL

### Context Window Discovery
**File**: `src/agents/context.ts`

Lazy-loads context window sizes from Pi SDK's `ModelRegistry`, user config overrides in `models.json`, and when multiple providers expose the same model, prefers the smaller window to avoid overestimation.

---

## Model Integration

OpenClaw is highly model-agnostic. 25+ providers supported:

| Provider | Env Var | Notes |
|----------|---------|-------|
| Anthropic | `ANTHROPIC_API_KEY` | Recommended default |
| OpenAI | `OPENAI_API_KEY` | Including Codex |
| Google | `GEMINI_API_KEY` | Also Vertex |
| **Ollama** | `OLLAMA_API_KEY` | **Local models — key integration** |
| **vLLM** | `VLLM_API_KEY` | **Local serving** |
| Groq | `GROQ_API_KEY` | |
| OpenRouter | `OPENROUTER_API_KEY` | Multi-provider gateway |
| LiteLLM | `LITELLM_API_KEY` | Proxy layer |
| + 15 more | Various | Mistral, Together, xAI, Bedrock, etc. |

### Wiring Qwen via Ollama

Trivial. Already first-class:

```bash
export OLLAMA_API_KEY="ollama-local"
ollama pull qwen3:32b
```

In `~/.openclaw/openclaw.json`:
```json
{
  "agents": {
    "defaults": {
      "model": { "primary": "ollama/qwen3:32b" }
    }
  }
}
```

Models referenced as `"provider/model"` strings. Supports aliases, allowlists, per-session overrides.

---

## Memory System

**Files**: `src/memory/` (72 files)

### Storage
- Plain Markdown in `~/.openclaw/workspace/`
- `memory/YYYY-MM-DD.md` — daily append-only logs
- `MEMORY.md` — curated long-term facts

### Indexing
- Chunks Markdown into ~400-token segments with 80-token overlap
- Vectors stored in SQLite + `sqlite-vec` at `~/.openclaw/memory/<agentId>.sqlite`
- Multiple embedding providers: OpenAI, Gemini, Voyage, local LLaMA via `node-llama.ts`

### Retrieval (Hybrid Search)
**File**: `src/memory/hybrid.ts`

- BM25 keyword matching (SQLite FTS5) + vector similarity
- Configurable `vectorWeight` and `textWeight`
- Post-processing: MMR re-ranking for diversity, temporal decay (30-day half-life)
- Results include file path, line ranges, snippets, scores

### Agent Interface
- `memory_search` tool: semantic search over indexed files
- `memory_get` tool: targeted line-range reads
- Automatic memory flush before compaction

---

## Skills System

**Files**: `src/agents/skills/` (16 files), `skills/` (53 bundled skills)

- Three types: bundled, managed (from ClawhHub), workspace (user-defined)
- Each skill has a `SKILL.md` with frontmatter metadata
- **Lazy loading**: Skills appear as a compact list in the system prompt (name + description only). Full instructions loaded on-demand when the model reads the `SKILL.md`. Keeps base context small.
- Eligibility checks: OS compatibility, binary/env/config requirements, explicit enable/disable

---

## Intervention Surfaces

### Surface A: `transformContext` Hook (Best for our experiment)
**File**: `src/agents/pi-embedded-runner/tool-result-context-guard.ts`

Receives complete message array after history assembly, before API call. Chains composably. This is the established pattern for context manipulation in OpenClaw.

### Surface B: System Prompt Builder
**File**: `src/agents/system-prompt.ts`

`buildAgentSystemPrompt()` accepts `contextFiles` array. Could dynamically compute which files to inject based on the task.

### Surface C: Memory System (Best for retrieval quality)
**Files**: `src/memory/manager.ts`, `src/memory/hybrid.ts`

The hybrid search infrastructure is already there. Could upgrade retrieval quality from the inside: better embeddings, task-aware re-ranking, domain-specific grounding injected into the index.

### Surface D: Workspace Context Files
Auto-injected files (`AGENTS.md`, `SOUL.md`, etc.) could be dynamically rewritten before a run. Coarse but simple.

---

## Assessment

### Opportunity
OpenClaw's current context management is **reactive** — truncate and compact when things overflow. The three-prompt strategy would make it **proactive** — curate what enters in the first place. The `transformContext` hook and the memory system's hybrid search provide two composable intervention points: smarter retrieval feeding into smarter context assembly.

### A/B Feasibility
Same model, same tasks, toggle the three-prompt pipeline on/off. Metrics from existing session JSONL logs: token counts, turn counts, tool calls, first-attempt correctness. The coding-agent skill is the natural test surface.

### Key Advantage
The infrastructure for retrieval-augmented context already exists in OpenClaw — it's just used for personal memory, not for proactive task-context curation. The three-prompt strategy would repurpose and extend this infrastructure rather than building from scratch.
