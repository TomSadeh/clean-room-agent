# [ARCHIVED] Conversation Summary — MapGen V3 Review, Auto-GM Vision, and the N-Prompt Context Architecture

**Date**: 2026-02-22
**Participants**: Tom (Aaron Institute / Auto-GM co-founder) and Claude (Opus 4.6)
**Context**: What started as a MapGen V3 meta-plan review evolved into a wide-ranging discussion that surfaced a potentially significant general-purpose architecture for LLM context management.

---

## Part 1: MapGen V3 Meta-Plan Review

### Documents Reviewed
- `mapgen_v3_meta_plan.md` — The full meta-plan for rebuilding Auto-GM's map generation system from scratch
- `mapgen_v3_algorithm_recommendation.md` — Algorithm selection appendix recommending Template-Driven Topology + Constraint-Based Placement + Compiled Bindings

### Architecture Assessment
The chosen architecture (template-driven hybrid) is well-reasoned and appropriate for the project's constraints. It correctly rejects graph grammars (too complex for MVP, conflicts with curated-bindings-only requirement), WFC (no global structure awareness, requires grid topology), and Voronoi (overkill for tactical map scale of 20-40 cells). The meta-plan's phasing and gating structure is solid — mandatory audits before schema work, compiler validation before runtime, explicit contract boundaries between LLM and procedural system.

### Identified Gaps and Problems (by severity)

**Structural / Architectural Gaps:**
1. **Abstract spatial archetypes (e.g., ForestClearing) have no geometry generation step.** These are used as spatial containers for `within` and `bisects` constraints but nowhere does the plan describe how their bounding polygons are created. This is load-bearing and entirely unspecified.
2. **A* pathfinding on a continuous coordinate space.** The plan commits to continuous (non-grid) placement but specifies A*/Dijkstra for path generation, which requires a discrete graph. Needs either a temporary discretized grid or a different algorithm (RRT, spline-based waypoint generation).
3. **Placement ordering is underspecified for complex dependency chains.** The algorithm says "sort by required=true first, then by constraint dependencies" but complex templates create co-dependent placement chains that need explicit topological sort with defined strategies. This is the actual hard part of Phase 2.
4. **Map dimensions have no specified source.** Phase 2 takes map dimensions as input but `intent_v3` doesn't include them and the meta-plan doesn't specify where they come from.

**Design Gaps:**
5. **Modifiers have no execution path.** Intent JSON includes modifiers but the algorithm never describes how they affect anything.
6. **Two-biome maps have no spatial model.** Max two biomes per map for MVP, but no description of how biome boundaries work spatially.
7. **Gameplay traits aren't threaded through the algorithm.** The trait catalog is defined but binding tables don't include trait columns and module resolution only returns asset IDs.
8. **Compiler can't validate geometric satisfiability.** Can validate binding completeness but not spatial constraint satisfiability — the doc should be explicit about this distinction.

**Technical Concerns:**
9. **Poisson disk sampling doesn't take a count parameter.** Standard Bridson's algorithm takes minimum radius, not count.
10. **Seed derivation is ad-hoc and inconsistent** across the algorithm doc, conflicting with the meta-plan's clean policy.
11. **No map-edge buffer policy** for objects near boundaries.
12. **Decor scatter for linear features is ambiguous** — paths are linear, not regions.

**Smaller Items:**
13. Open Question #2 (dynamic LLM template generation) should be a closed "No" for MVP.
14. Template composition should be decided now to avoid schema rework.
15. A* cost function references terrain/obstacles that don't exist yet at path-placement time.
16. No performance characteristics for the constraint solver; no early detection of geometrically impossible placements.

### Overall Assessment
Phase 2 is carrying the most implicit complexity and deserves its own sub-design doc before implementation. The rest of the plan is well-structured.

---

## Part 2: Auto-GM Product Vision and Strategy

### Project Status
Auto-GM is approximately one month from a friends pre-alpha. Current state:
- Tactical combat nearly finished (grappling and edge cases remaining)
- Four classes (Fighter, Rogue, Cleric, Wizard) implemented to level 3 with subclass selection
- Four races (Human, Halfling, Dwarf, Elf) implemented
- ~20 spells implemented with full spell effects
- HDYWTDT (How Do You Want To Do This) fully working with companion memory integration
- Companion banter references past events ("the way you cut off that orc chief's head...")
- Big dice rolling on death saves, crits, and skill checks
- Knowledge system with perfect retrieval using local model operational

### Remaining for Pre-Alpha
- MapGen V3 (the system reviewed in Part 1)
- A scripted general arc for the alpha with scripted encounters
- Playtesting of procedural quest templates

### Three-Mode Product Model
The product doc (`Auto-GM_Summary_v3_Structure_revised.md`) establishes three first-class play modes:
- **Procedural**: Generate a world, play forever. Stories emerge from system interactions.
- **Authored**: Hand-crafted campaigns with intentional storytelling and designed encounters.
- **Hybrid**: Creator writes the highlights, the game fills in the rest. Equal billing, not a footnote.

### Core Differentiators
1. **"Every AI DM cheats. We don't."** — Rules are real, dice are real, the game can't fudge outcomes. This is the lead differentiator.
2. **Tactical combat** — Absent from every current AI DM product. Requires spatial state, rule enforcement, deterministic mechanics, and meaningful maps.
3. **Constrained chaos** — The LLM provides creative direction (chaos), deterministic systems enforce mechanical integrity (constraints). Neither alone is D&D. Together they might be.
4. **Scoped NPC knowledge** — NPCs know what they should know and nothing more. The deeper moat — every competitor claims smart combat, almost nobody claims scoped NPC knowledge.

### Narrative Architecture
Procedural narrative works through reactive template instantiation:
- The system detects emergent patterns in player behavior (liked an NPC, repeated interactions of a certain type)
- Relationships get promoted (Best Friend, Mentor, Romantic Interest) based on interaction patterns
- The Game Director instantiates quest arc templates (hint → crisis → rescue → showdown) with content drawn from actual game state
- The knowledge system with perfect retrieval feeds exactly the right context to the LLM for each narrative beat
- Result: feels authored because the emotional stakes were written by the player's own choices

### Monetization Strategy
- **Sell the game itself** — simplest answer, aligns with "runs on your machine, no subscription" positioning
- **Early Access** viable because the procedural base provides genuine replayability from day one
- **EA pricing**: Start with 4 classes, 4 races, levels 1-6; raise price as content is added (Factorio/Valheim model)
- **Marketplace** is growth multiplier, not core revenue — campaigns ($10-20), content packs ($3-5)
- The procedural base sets the quality floor: "if the free procedural system is better than your campaign, nobody buys it"
- **Creator ecosystem**: Two types of creation (authored narrative content + procedural template authoring) that compose across creators without coordination

### Architecture Advantages for Environmental Interaction
Because map spec JSON is single source of truth with swappable sprites/textures at runtime:
- Fireball lands → grass texture swaps to fire, gameplay traits update (difficult terrain → fire damage zone)
- Shatter breaks wall → wall object removed from spec, cover trait removed, replaced with rubble (difficult terrain)
- This is generic through the trait/material system, not individually scripted like BG3
- Potentially surpasses BG3 tactically through procedural environmental interaction

### Key Quote on Aesthetic Strategy
VTT maps are genuinely beautiful in their own register. The visual bar is achievable with curated asset sets and good art direction rather than a 200-person art team. The LLM narrator bridges imagination to mechanics — "the Fireball detonates against the wooden barricade — the planks splinter and catch fire." Players need functional clarity, not cinematic fidelity. The art budget goes to readability. Every hour saved on visual fidelity goes to mechanical depth.

### Development Methodology
Tom and Itay are self-described non-programmers building one of the most architecturally complex game types (procedurally generated, LLM-driven, tactical combat RPG) through:
- Rigorous upfront planning with mandatory gates between phases
- Documentation discipline ("like someone points a gun at our head")
- Aggressive agent monitoring despite limited programming knowledge
- Leveraging their domain expertise (deep 5e mechanics knowledge) as the quality check agents can't provide
- Key principle: "באג בדיזיין, זין בדיבאג" — bugs caught in design save exponentially more time than debugging after implementation

---

## Part 3: The N-Prompt Context Management Architecture

### Core Thesis
**The primary bottleneck in LLM application performance is not model capability but context curation.** Controlling what enters the context window matters more than model size, architecture improvements, or prompting technique. The context window should be used for reasoning, not retrieval. Separate the two and both get better.

### Origin of the Thesis
Validated in Auto-GM's knowledge system: a 4B local model with perfect retrieval (curated context, multi-stage filtering) outperforms much larger models with naive full-context approaches. The insight: most LLM failures are not reasoning failures, they are information failures — the model had the wrong context, not the wrong capability. This parallels human cognition: most human errors are also retrieval failures, not reasoning failures (a doctor misdiagnoses because the right differential didn't surface from memory, not because they can't reason through symptoms).

### The Architecture

**Components:**
1. **A centralized knowledge base** — everything the system has ever done, learned, or encountered, structured and indexed. Self-maintaining: grows automatically with every task the agent performs.
2. **A retrieval system** — deterministic where possible, targeted AI-assisted where necessary. Not semantic search (embedding similarity hoping to capture relevance). Deterministic metadata extraction first, AI only for ambiguous items, grounded with confirmed metadata.
3. **A pipeline orchestrator** — n prompts, where n scales with task complexity. Early stages filter and ground. Later stages reason and execute. Each prompt starts clean with curated context.
4. **N swappable LoRA adapters** — one per pipeline stage, fine-tuned for that stage's specific job (filtering, grounding, executing, etc.). Same base model in memory, tiny adapter swap between stages. Retrainable cheaply as the knowledge base evolves.

**What this eliminates from the context window:**
- Irrelevant project files (retrieval handles this)
- Tool definitions (orchestrator injects only the needed tool with exact syntax, only when needed)
- Tool call history (no persistent conversation, each prompt starts clean)
- System prompt overhead (minimal per stage, task-specific)
- Conversation history (no conversation to accumulate)
- Context summaries/compaction (nothing accumulates, nothing degrades)

**Result:** A 32k context window at nearly 100% signal relevance, versus a 200k window at maybe 10-15% effective utilization. The small model with the pipeline has more *useful* context than the large model with the ocean.

### Why Current Approaches Are Insufficient

**What the research community is doing (model-side solutions):**
- Dynamic attention weighting, sparse attention mechanisms
- Retrieval-augmented attention
- Context compression/distillation
- Architectural improvements (Mamba, RWKV, state-space models)
- All of these try to make the model better at finding relevant items in a noisy context

**What practitioners are doing (application-side solutions):**
- RAG pipelines with embedding-based semantic search
- Reranking retrieved documents
- Recursive summarization
- Elaborate documentation systems (CLAUDE.md, skills, custom commands)
- All of these are either lossy, stale-prone, or still fundamentally "stuff context and hope"

**What nobody is doing (the gap):**
- Pre-filtering as a principled alternative to retrieval-augmented generation
- Multi-prompt pipelines where early prompts curate context for later prompts
- Separation of retrieval and reasoning as distinct responsibilities handled by different systems
- Knowledge bases that grow from agent activity as an alternative to static documentation
- Cross-project knowledge transfer through a unified centralized knowledge base

### Scaling Properties

**Cross-project knowledge:**
The centralized knowledge base isn't project-scoped. Every project the agent works on enriches the DB. A bug fix pattern discovered in one project surfaces when a similar pattern appears in a completely different project. This is how senior engineers actually work — they carry a mental library of every bug they've ever fixed across every job. This system externalizes and makes retrievable what currently walks out the door when senior engineers leave.

**Cold start via git history:**
Every project with a git history already has a structured log of every decision, bug fix, refactor, and file relationship. Deterministic extraction populates the knowledge base immediately. Point the system at every repo you've ever touched and the agent starts its first task with your entire engineering career as its knowledge base. No onboarding, no documentation to write.

**Team scaling:**
Multiple engineers' git histories go into the same knowledge base. A new hire's agent performs like a senior engineer on day one — not because the model is smart, but because the retrieval system has access to the entire team's institutional knowledge.

**Domain agnosticism:**
The architecture is not coding-specific. Medical diagnosis, legal research, financial analysis, scientific research — any domain where the core problem is "get the right information in front of a reasoning system at the right time." The pipeline depth (n) scales with domain complexity. The LoRAs specialize per domain and per stage. The knowledge base schema adapts to domain-specific metadata.

### Competitive Implications

This architecture is being treated as a competitive advantage, not a publication. Key reasons:
1. To the best of current knowledge, nobody in the research community is working on this specific combination of ideas
2. The insight is necessary but not sufficient — the specific implementation (retrieval logic, grounding patterns, pipeline orchestration, knowledge base schema) is earned through extensive trial and error
3. If validated across domains, the advantage compounds over time as the knowledge base grows
4. Publication gives competitors the framing for free; keeping it means Auto-GM's agents perform better than competitors without an obvious explanation

### Implications for LLM Persistent Memory
If this architecture works as theorized, it provides a general solution to LLM persistent memory. The model never holds state — it never will. But if the right information enters the context window at the right time, every time, the model's experience is indistinguishable from having memory. This is better than human memory, which is a lossy, biased, emotionally-weighted retrieval system that degrades over time and can't be searched.

---

## Action Items

1. **Immediate**: Finish Auto-GM pre-alpha (maps, scripted alpha arc, quest template playtesting)
2. **Immediate**: Codex applies MapGen V3 review findings with codebase context
3. **Pre-alpha gate**: Friends playtest, specifically the harsh AI-skeptic TTRPG player
4. **Post-alpha**: Run deep research prompts (agent harness survey + context management literature survey)
5. **Post-alpha**: Probe OpenClaw architecture to understand current state of the art (and what not to do)
6. **Post-alpha**: Design and build own harness — knowledge base, retrieval system, pipeline orchestrator
7. **Post-alpha**: Experiment with small fine-tuned coding model (Qwen Coder class) + context management pipeline vs. frontier model with naive context
8. **Long-term**: Cross-project knowledge base seeded from git history across all repos
9. **Long-term**: N-LoRA pipeline with per-stage fine-tuned adapters
10. **Long-term**: Domain generalization beyond coding

---

## Key Principles (Referenced Throughout)

- **באג בדיזיין, זין בדיבאג** — A bug caught in design saves exponentially more than debugging after implementation
- **Constrained Chaos** — Creative freedom bounded by mechanical consequence; the LLM provides chaos, deterministic systems provide constraints
- **The room, not the model** — Performance comes from what's in the context window, not the model's capability
- **Results and capabilities, never mechanisms** — Show what the system does, never explain how
- **The model's job is reasoning, not retrieval** — Separate the two and both get better


