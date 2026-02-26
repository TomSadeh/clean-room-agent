# Infrastructure Self-Improvement -- Design Record

**Date:** 2026-02-26
**Status:** Decided

## Problem Statement

Model self-improvement (Phase 4: LoRA adapters, teacher-student distillation) operates within safe bounds -- adapters attach to frozen base weights, plateau after ~2 iterations at sub-7B scale, and rollback is a DB operation (swap `adapter_metadata` rows). Infrastructure self-improvement -- the agent modifying its own pipeline code, prompts, retrieval logic, orchestrator, parsers, and configuration -- is fundamentally different. Code changes cascade: a bad retrieval stage produces bad context, which produces bad code, which produces bad training data, which produces worse adapters. The failure mode is not degradation but corruption of the self-improvement loop itself.

The bootstrap paradox: the agent must use the current pipeline to evaluate modifications to the current pipeline. If the evaluation infrastructure is compromised by the modification, the agent cannot detect its own degradation. This is the infrastructure analog of Godel's incompleteness -- the system cannot fully validate itself from within.

## Decision

A three-tier modification hierarchy with an immutable evaluation core, a six-stage validation gauntlet for all changes, and checkpoint-based training data hygiene that taints runs produced under rolled-back infrastructure.

## Architecture

### Three-Tier Modification Hierarchy

The hierarchy is constitutional: the rules for changing rules are more constrained than the rules themselves. If the agent can modify its own observation and control mechanisms, transparency ceases to be load-bearing.

#### Tier 0 -- The Immutable Core

Never self-modified. Only human-modified via direct commits.

| Component | Files | Rationale |
|---|---|---|
| Logging infrastructure | `llm/client.py` (`LoggedLLMClient`, `flush()`), `db/raw_queries.py` (all `insert_*` functions) | If logging is compromised, the traceability chain breaks. Raw DB is the training corpus -- corrupted logging corrupts training. |
| Raw DB schema | `db/schema.py` (raw DB DDL), `db/connection.py` (`get_connection`) | Schema defines the audit trail structure. Connection factory is the single point of DB management. |
| Test runner and validator | `orchestrator/validator.py` (`run_validation`, `require_testing_config`) | The validator is the oracle for all infrastructure evaluation. If it can be softened, bad changes pass. |
| Checkpoint/rollback | `orchestrator/git_ops.py` (`GitWorkflow` core: `create_task_branch`, `rollback_to_checkpoint`, `commit_checkpoint`) | Rollback is the safety net. If compromised, bad infrastructure persists. |
| The self-modification controller | (new: `infrastructure/controller.py`) | The mechanism that enforces tiers, runs the gauntlet, and decides promote/reject. Self-modification of the controller is a halt condition. |
| The reference task set | (new: `infrastructure/reference_tasks/`) | The evaluation criteria. If the agent can modify what "success" means, it can declare any change an improvement. |

**Enforcement mechanism:** A manifest file (`infrastructure/tier0_manifest.txt`) lists all Tier 0 file paths. The self-modification controller checks this manifest before applying any change. Any modification touching a Tier 0 path is rejected without evaluation. The manifest itself is Tier 0. Human updates to Tier 0 files go through normal development (direct commits, code review, test suite).

#### Tier 1 -- Human-Approved Modifications

The agent may propose changes, but a human must review and approve before promotion.

| Component | Files | Rationale |
|---|---|---|
| System prompts | `execute/prompts.py` (`SYSTEM_PROMPTS` dict, all `build_*_prompt` functions) | Prompts affect all model behavior across all modes. A subtle prompt regression can degrade every pipeline run. |
| Training data extraction | (Phase 4: `cra curate-data` pipeline) | Controls what becomes training data. A change that relaxes curation filters poisons future adapters. |
| Orchestrator loop structure | `orchestrator/runner.py` (`run_orchestrator`, pass sequencing logic) | The sequencing of meta_plan -> part_plan -> code_steps -> test_plan -> test_steps -> validate -> adjust is a core architectural decision. |
| Budget validation | `retrieval/budget.py` (`BudgetTracker`), `token_estimation.py` | Budget validation prevents silent truncation (R3). Relaxing it undermines context curation. |
| Model routing configuration | `llm/router.py` (`ModelRouter.resolve`) | Routes which model handles which stage. Wrong routing means wrong capabilities at critical decision points. |

**Approval flow:** Proposals logged to raw DB (`infra_proposals` table) with full rationale and diff. During operator sessions, `cra infra review` surfaces pending Tier 1 proposals. Human approves, rejects, or requests revision. Approved proposals enter the validation gauntlet (same as Tier 2). Rejected proposals are retained in raw DB for analysis.

#### Tier 2 -- Autonomously Modifiable

The agent may propose, validate, and promote changes without human intervention, subject to the full validation gauntlet.

| Component | Files | Rationale |
|---|---|---|
| Retrieval stage logic | `retrieval/scope_stage.py`, `precision_stage.py`, `similarity_stage.py`, `context_assembly.py` | These stages filter and rank candidates. Improvements here directly improve context quality, and the reference task set can objectively measure impact. |
| Parser improvements | `parsers/base.py`, `parsers/python_parser.py`, `parsers/ts_js_parser.py` | AST extraction and rendering. Parser correctness is verifiable by construction (parsed output matches source). |
| User-facing prompt content | Task-specific prompt templates (not system prompts) | Narrower scope than system prompts. Effects measurable per-task. |
| Configuration thresholds | `.clean_room/config.toml` numeric values (batch sizes, retry counts, budget allocations) | Bounded parameters with measurable effects. Small changes, testable impact. |
| Batch judgment utilities | `retrieval/batch_judgment.py` | Shared batching logic. Correctness verifiable through existing test suite. |

**Why these are safe to autonomize:** Each has a tight feedback loop. Retrieval changes affect context quality, measured by reference task outcomes. Parser changes are structurally verifiable. Config changes are bounded and reversible. The validation gauntlet catches regressions before they affect production runs.

### The Validation Gauntlet

Every infrastructure modification -- Tier 1 (after human approval) and Tier 2 (autonomous) -- passes through a six-stage pipeline. All stages are Tier 0 (immutable).

```
Propose → Branch → Test → Benchmark → Evaluate → Promote/Reject
```

#### Stage 1: Propose

The agent generates a modification with explicit rationale.

**Logged to raw DB** (`infra_proposals` table):
- `proposal_id` (UUID)
- `task_id` (originating task, if any)
- `tier` (1 or 2)
- `target_files` (list of paths being modified)
- `diff` (unified diff of proposed changes)
- `rationale` (free text: what problem does this solve, what evidence supports it)
- `trigger` (what prompted the proposal: performance regression, error pattern, operator feedback)
- `created_at` (timestamp)

**Constraints:** The proposal must identify which tier each target file belongs to. Any mismatch (proposing Tier 0 changes, or Tier 1 changes without approval) halts the pipeline.

**Schema** (raw DB):

```sql
CREATE TABLE infra_proposals (
    id TEXT PRIMARY KEY,            -- UUID
    task_id TEXT,                   -- originating task (nullable: some proposals are pattern-driven)
    tier INTEGER NOT NULL,          -- 1 or 2
    target_files TEXT NOT NULL,     -- JSON array of file paths
    diff TEXT NOT NULL,             -- unified diff
    rationale TEXT NOT NULL,        -- free text justification
    trigger TEXT NOT NULL,          -- "error_pattern" | "performance_regression" | "operator_request"
    status TEXT NOT NULL DEFAULT 'proposed', -- proposed | approved | testing | promoted | rejected
    approved_by TEXT,               -- NULL for Tier 2, operator ID for Tier 1
    approved_at TEXT,
    created_at TEXT NOT NULL
);
```

#### Stage 2: Branch

Git isolation using the existing `GitWorkflow` pattern.

```
cra/infra/<proposal_id>
```

The branch is created from the current HEAD. The proposed diff is applied. This reuses `GitWorkflow.create_task_branch()` and `commit_checkpoint()` -- the same code paths the orchestrator already uses for task branches, ensuring the isolation mechanism is battle-tested.

#### Stage 3: Test

Run the full existing test suite against the modified infrastructure.

**Requirement: 100% pass.** No new test failures permitted. This is a hard gate, not a threshold. The test suite is Tier 0 -- the agent cannot weaken the gate by modifying tests.

Uses `run_validation()` from `orchestrator/validator.py` (Tier 0). Test results logged to raw DB (`infra_validation_results` table) linked to `proposal_id`.

**Why 100%:** Infrastructure modifications are not task-specific. A change that breaks any test has introduced a regression, even if the intended behavior is improved. Tests can be expanded (new tests for the modified behavior), but existing tests cannot be removed or weakened by the agent.

#### Stage 4: Benchmark

Run the **reference task set** through the full pipeline twice: once on the unmodified infrastructure (baseline), once on the modified infrastructure (candidate).

Each reference task produces:
- Pipeline outcome (success/failure)
- Validation result (tests pass/fail)
- Token budget utilization
- Retrieval precision (files included vs. files actually needed)
- Stage-level timing and token counts

Both runs are logged to raw DB with the `proposal_id` for traceability.

**Determinism:** Reference tasks use fixed random seeds and temperature 0.0 to minimize non-deterministic variance. The comparison is between infrastructure versions, not between stochastic model outputs.

**Cost management:** Running the full reference set twice is expensive. For Tier 2 config-only changes (batch sizes, retry counts), run a subset of 5 representative tasks first. If the subset shows no regression, run the full set. For code changes, always run the full set -- code changes have broader blast radius.

**Baseline caching:** The baseline run (unmodified infrastructure) is cached per checkpoint. If the current checkpoint's baseline is already recorded and no non-infrastructure changes have occurred, reuse it rather than re-running. This halves the benchmark cost for consecutive proposals.

#### Stage 5: Evaluate

Statistical comparison of baseline vs. candidate on the reference task set.

**Promotion criteria (all must hold):**
1. No reference task that succeeded on baseline fails on candidate (zero regressions)
2. At least one measurable improvement: higher success rate, lower token usage, better retrieval precision, or faster execution
3. No increase in validation failures across the reference set

**Why zero-regression:** Infrastructure changes are global. A change that improves 8/10 tasks but regresses 2/10 is not a net improvement -- it is an inconsistency that will compound. The correct response is to narrow the change scope until regressions disappear.

Evaluation results logged to raw DB (`infra_evaluations` table) with per-task comparisons.

#### Stage 6: Promote or Reject

**Promote:** Merge the infrastructure branch into the main line. Record a new `infra_checkpoint_id` in the raw DB. All subsequent task runs record this checkpoint ID.

**Reject:** Delete the infrastructure branch. The proposal, test results, benchmark data, and evaluation are all retained in raw DB. Rejected proposals are training data for learning what doesn't work.

**Both outcomes are permanent records.** The agent can analyze its own proposal history to identify patterns in what succeeds and what fails.

### The Reference Task Set -- Infrastructure's Incorruptible Oracle

The reference task set is the infrastructure equivalent of a held-out test set in ML. It answers: "did this change make the pipeline better or worse?" without relying on the pipeline's own judgment.

**Composition:**
1. **Curated commits from corpus repos** -- real commits with known-good solutions from the training corpus (repos vetted via corpus_cloning_protocol.md). The task is: given the repo state before the commit, reproduce the commit's changes.
2. **Historical verified runs** -- past pipeline runs that produced correct, validated solutions. Stored as (task_description, repo_snapshot, expected_outcome) triples.
3. **Deliberately hard edge cases** -- tasks designed to stress specific pipeline components: large files that stress budget management, deep dependency chains that stress scope expansion, ambiguous tasks that stress task analysis.
4. **Regression anchors** -- any task where a previous infrastructure proposal caused a regression. These become permanent members of the reference set, ensuring the same failure mode is never reintroduced.

**Storage:**

```
infrastructure/reference_tasks/
    manifest.json          # task metadata, expected outcomes, categories
    tasks/
        <task_id>/
            repo_snapshot/  # minimal repo state for the task
            task.json       # task description, expected files, expected outcome
            baseline.json   # last known-good pipeline results
```

**Immutability (Tier 0):** The agent cannot modify the reference task set. This prevents the agent from "teaching to the test" -- optimizing infrastructure for the reference set at the expense of general performance.

**Human expansion:** Periodically, humans add fresh tasks from new repos and novel problem types. This prevents overfitting to a static evaluation set. The addition process mirrors corpus_search_protocol.md: gap-first analysis of underrepresented task categories, quality gate on task construction, recording in the manifest.

**Sizing:** Start with 20-30 reference tasks spanning plan mode, implement mode, and edge cases. Expand to 50+ as the infrastructure proposal history grows and reveals blind spots.

**Construction criteria for reference tasks:**

Each task must have:
1. **A deterministic expected outcome** -- either a known-good diff (from corpus commits) or a verifiable property (tests pass, specific files modified, specific symbols present in output).
2. **A minimal repo snapshot** -- only the files relevant to the task, not the full repo. This keeps benchmark runs fast and storage manageable.
3. **Category tags** -- which pipeline components the task exercises (budget management, scope expansion, precision classification, multi-file edits, etc.). Gap analysis uses these tags to ensure coverage.
4. **Difficulty rating** -- easy (single file, clear intent), medium (multi-file, requires dependency analysis), hard (ambiguous intent, large codebase, edge cases). The set must include all three.

**Reference task manifest entry:**

```json
{
    "task_id": "ref-001",
    "source": "corpus/structlog",
    "source_commit": "abc123",
    "description": "Add timeout parameter to retry handler",
    "expected_outcome": "diff",
    "expected_diff_path": "tasks/ref-001/expected.diff",
    "categories": ["scope_expansion", "precision_classification", "multi_file"],
    "difficulty": "medium",
    "added_by": "human",
    "added_at": "2026-02-26",
    "regression_anchor_for": null
}
```

### Training Data Hygiene -- Checkpoint Tag Propagation

Infrastructure changes affect all pipeline runs executed under them. If an infrastructure version is rolled back, every run produced under that version is tainted -- the context curation, prompt construction, and retrieval decisions that shaped those runs may have been wrong.

**The tag propagation mechanism:**

1. Every `infra_checkpoint_id` is recorded in `orchestrator_runs` when a task begins.
2. If an infrastructure checkpoint is later rolled back (via the validation gauntlet rejecting a subsequent proposal that reveals latent issues, or via drift detection), the checkpoint is marked `rolled_back = TRUE` in the raw DB.
3. `cra curate-data` (Phase 4) checks `infra_checkpoint_id` against the rollback table. Runs under rolled-back checkpoints are **excluded** from training data extraction.
4. Default-deny: runs without a valid `infra_checkpoint_id` are excluded. This matches the existing R2 principle -- content without a positive curation signal is not promoted.

**This is the same pattern as `domain_confidence` tiers** (corpus_search_protocol.md): gray-domain content is retrievable but excluded from training. Tainted infrastructure runs are logged (raw DB, append-only) but excluded from training data.

**Cascading rollback:** When a checkpoint is rolled back, all descendant checkpoints in the chain are also rolled back. The `parent_checkpoint_id` linked list makes this a simple traversal. This prevents a scenario where checkpoint C (child of rolled-back B) survives even though it was validated under tainted infrastructure.

**Partial taint:** Not all runs under a rolled-back checkpoint are necessarily bad. A retrieval-stage change doesn't affect runs that happened to use a different retrieval path. However, determining which runs were actually affected requires analyzing the full execution trace -- more expensive than the conservative approach. Default-deny (exclude all) is the safe choice. If this proves too aggressive (discarding too much training data), a future refinement can add per-run impact analysis using the `retrieval_decisions` and `retrieval_llm_calls` tables to determine whether the rolled-back component was actually exercised.

**Schema additions** (raw DB):

```sql
CREATE TABLE infra_checkpoints (
    id TEXT PRIMARY KEY,           -- UUID
    proposal_id TEXT NOT NULL,     -- links to infra_proposals
    parent_checkpoint_id TEXT,     -- previous checkpoint (linked list)
    created_at TEXT NOT NULL,
    rolled_back INTEGER DEFAULT 0, -- 0=active, 1=rolled back
    rolled_back_at TEXT,
    rollback_reason TEXT,
    FOREIGN KEY (proposal_id) REFERENCES infra_proposals(id)
);

-- Add to existing orchestrator_runs:
-- infra_checkpoint_id TEXT REFERENCES infra_checkpoints(id)
```

### Drift Detection -- Compounding Degradation

Individual infrastructure changes pass the gauntlet. But compounding effects across multiple changes can cause gradual degradation that no single evaluation catches.

**Detection mechanisms:**

1. **Baseline regression checks:** Every checkpoint records the full reference task set performance as its baseline. Periodic full evaluations (every N task runs, configurable) compare current performance against the last checkpoint's baseline. Sustained decline across 3+ evaluations triggers investigation.

2. **Training data quality monitoring:** Track success rate of task runs over time, bucketed by `infra_checkpoint_id`. A checkpoint that correlates with declining success rates is flagged for human review.

3. **Automatic rollback trigger:** If reference task set performance drops below a configurable threshold (e.g., 20% degradation from the best-known checkpoint), automatically rollback to the last known-good checkpoint and taint intermediate runs. This is a circuit breaker, not a normal control path.

4. **Checkpoint chain analysis:** The `parent_checkpoint_id` linked list enables bisection. If degradation is detected, binary search through the checkpoint chain to identify which specific change introduced the regression -- same principle as `git bisect`.

**Metrics tracked per checkpoint:**

| Metric | Source | What it measures |
|---|---|---|
| Reference task success rate | Benchmark stage results | Overall pipeline effectiveness |
| Mean token budget utilization | `BudgetTracker` logs | Context curation efficiency |
| Mean retrieval precision | `retrieval_decisions` table | Signal-to-noise in context window |
| Production task success rate | `task_runs` table | Real-world effectiveness |
| Mean validation pass rate | `validation_results` table | Code quality |
| Proposal acceptance rate | `infra_proposals` table | Quality of agent's self-assessment |

All metrics are stored in `infra_checkpoint_metrics` (raw DB) and plotted over the checkpoint chain. Trend lines surface gradual degradation that absolute thresholds miss.

### The Positive Feedback Loop

The transparency principle makes infrastructure self-improvement a positive feedback loop because every component of the loop is observable and verifiable.

```
Better infrastructure
    → better context curation
    → better model outputs
    → better task outcomes (measured by validator, Tier 0)
    → better training data (filtered by checkpoint hygiene)
    → better adapters
    → better infrastructure proposals (model proposes, gauntlet validates)
    → better infrastructure
```

**Why this converges rather than diverges:** The reference task set is an external anchor. Improvements must be measurable against it. The gauntlet enforces zero-regression. Drift detection catches compounding effects. Training data hygiene prevents tainted runs from poisoning the loop. Each constraint is independently enforceable and independently auditable.

**The key insight:** transparency is not overhead -- it is the mechanism. Every decision is logged (raw DB). Every evaluation is recorded (benchmark results). Every rollback is traceable (checkpoint chain). A human can audit the entire infrastructure evolution history using only the raw DB. This is the same traceability test from CLAUDE.md applied to the agent's self-modification: "a human must be able to trace any output back through every decision that produced it using only the logs."

### The Negative Loop Interruption

```
Bad infrastructure change
    → worse context curation
    → worse task outcomes
    → detected via reference task set (Tier 0, immutable)
    → rollback to last known-good checkpoint
    → tainted runs excluded from training data
    → negative loop interrupted before compounding
```

**Three independent interruption points:**
1. **Gauntlet rejection** -- catches bad changes before they reach production (Stage 5 evaluation)
2. **Drift detection** -- catches compounding effects that individual evaluations miss
3. **Training data hygiene** -- prevents bad runs from corrupting future adapters, even if they weren't caught immediately

Any single interruption point is sufficient to break the negative loop. All three operating together provide defense in depth.

### Air-Gapped Machine Constraints

The target deployment is an air-gapped machine with no internet access. This constrains but does not prevent infrastructure self-improvement.

| Tier | Air-gapped behavior |
|---|---|
| **Tier 0** | Frozen at migration. Updated only during operator sessions via USB. |
| **Tier 1** | Proposals queued in raw DB. Reviewed during operator sessions. Approved changes enter gauntlet locally. |
| **Tier 2** | Fully autonomous within gauntlet bounds. No external dependency. |
| **Reference set** | Seeded at migration from corpus repos. Expanded via USB data ingestion during operator sessions. |
| **Drift detection** | Runs locally. Alert mechanism is a flag in raw DB, surfaced at next operator session. |

**No external oracle dependency:** The validation gauntlet runs entirely locally -- test suite, reference tasks, and statistical evaluation all execute on the air-gapped machine. Teacher model distillation (Phase 4) is a separate concern that operates during operator sessions.

### The Bootstrap Paradox -- Addressed

The agent uses the current pipeline to evaluate modifications to the current pipeline. This circular dependency is broken by the Tier 0 boundary:

1. The **evaluation infrastructure** (validator, reference tasks, gauntlet controller) is Tier 0 -- immutable by the agent.
2. The **things being evaluated** (retrieval stages, parsers, config, prompts) are Tier 1/2 -- modifiable but only through the gauntlet.
3. The agent modifies (2) but cannot modify (1). The evaluation is always external to the thing being evaluated.

This is not a complete solution -- a sufficiently subtle degradation in Tier 2 components could still affect evaluation indirectly (e.g., worse retrieval during benchmark runs of reference tasks). The defense is statistical: reference tasks are diverse enough that systematic degradation across all of them requires a fundamentally broken change, which the test suite (Tier 0, Stage 3) would catch first.

**The residual risk and its mitigation:** A Tier 2 change to retrieval logic could subtly bias context curation in a way that helps on reference tasks but hurts on novel tasks. This is the classic overfitting problem. Mitigations: (a) reference set diversity -- tasks span multiple repos, languages, and problem types, so overfitting requires overfitting to all of them simultaneously; (b) periodic human expansion of the reference set with fresh tasks the agent has never seen; (c) production task success rate monitoring (drift detection metric 4) catches real-world degradation even if reference tasks don't.

**Asymmetric information analysis:** The system has unusually good information transfer (full LLM call logging, thinking tags, assembly decision tracking, provenance chains). Of four initially-hypothesized asymmetries, two dissolve under scrutiny: (a) reasoning quality vs. outcome quality is a transient problem, not a structural market failure -- iterated multi-level selection means lucky runs are selected against statistically within 1-2 generations, unlike Akerlof's one-shot lemons market; (b) stage-to-stage "price compression" is not an asymmetry at all -- it is proper division of labor where each stage receives the input it needs, not the upstream stage's internal reasoning (the subagent pattern). Two genuine, irreducible asymmetries remain: (1) present reference task performance is observable but long-term path-dependent costs of modifications are not (temporal externality -- the future hasn't happened yet); (2) performance on the reference set is observable but performance on the true task distribution is not (proxy-target gap -- unknowable from inside the system). Both require human inputs: architectural foresight and reference set expansion from real-world task distributions. See `research_reviews/evolutionary_economics_of_self_improvement.md` for the full analysis.

**Comparison to model self-improvement bounds:** Model improvement via LoRA plateaus after ~2 iterations because adapter capacity is bounded and the base model is frozen. Infrastructure improvement has no analogous plateau -- code can be rewritten arbitrarily. The three-tier hierarchy and validation gauntlet are the artificial bounds that constrain infrastructure modification to the same "safe, bounded, reversible" property that LoRA adapters have naturally.

### Evolutionary Dynamics -- Why This Architecture Works

The design above is not an analogy to evolution. It is an evolutionary system -- and understanding it through population genetics clarifies why it works, what it naturally produces, and what requires perpetual maintenance.

#### The Four Evolutionary Forces

Every evolutionary system is governed by four forces: founder effect (initial conditions), selection (differential survival), gene flow (migration between populations), and mutation (novel variation). This architecture has an explicit mechanism for each.

| Force | Biological meaning | System mechanism | Control parameter |
|---|---|---|---|
| **Founder effect** | Initial gene pool constrains all future evolution | Domain transparency gating (corpus_search_protocol.md) -- only transparent-methodology domains enter the training corpus | Corpus composition |
| **Selection** | Differential survival based on fitness | Multi-level execution feedback: validation gauntlet for infrastructure, test passage for code, reference task set for pipeline quality | Reference task set, test suite |
| **Gene flow** | Migration of genetic material between populations | Provenance tracking (`domain_confidence` tiers, `infra_checkpoint_id` tagging) -- prevents gray-tier or tainted material from entering the training gene pool | Tag propagation filters in `cra curate-data` |
| **Mutation** | Random variation that selection acts upon | Temperature-controlled randomness in LLM sampling -- tunable per run | Temperature parameter, `np.random` seeds |

**Why this is stronger than biological evolution:** Biology cannot tune its mutation rate, cannot choose its founding population, and has limited control over gene flow. This system controls all four parameters explicitly. The alignment field frames self-improvement as a steering problem ("how do we make the agent want the right thing"). This architecture frames it as a breeding program ("what selection pressure, gene pool, and mutation rate produce the traits we want").

#### Multi-Level Selection Defeats Reward Hacking

The standard alignment concern with execution-based reward signals is Goodhart's law: models learn to game the proxy (tests) rather than achieve the true objective (correct code). Published results (ImpossibleBench 2025, METR) demonstrate frontier models reward-hacking test suites 76% of the time in single-shot settings.

This architecture operates in a fundamentally different regime. Selection pressure acts at four levels simultaneously:

1. **LoRA adapters** -- a LoRA that encodes shortcut strategies produces a model that writes worse code on novel tasks. Selected against at the next evaluation.
2. **Full fine-tuning** -- a fine-tune that bakes in test-gaming patterns produces a model that can't solve harder problems in the next iteration. Selected against.
3. **New model training** -- training data contaminated by shortcut reasoning produces a weaker model than training data from genuine reasoning. Selected against.
4. **Pipeline code modification** -- infrastructure changes that degrade output quality produce worse downstream results across everything. Selected against by the validation gauntlet.

A bad reasoning pattern must survive all four levels to persist. Shortcuts don't compound into capability; genuine reasoning does. Over multiple iterations, the gap between the two strategies widens. This is structurally why biological evolution produces genuine adaptations rather than "hacks" -- when selection is continuous, multi-generational, and the environment keeps getting harder, faking fitness has no stable equilibrium.

**The 3-4B parameter practical defense:** Every published reward-hacking result comes from frontier models (GPT-5, o3, Claude 3.5 Sonnet). ImpossibleBench found that stronger models cheat more, not less. A 3-4B model likely lacks the situational awareness and planning capacity to devise sophisticated reward exploits. This is a contingent empirical defense (model capability), not a theoretical guarantee (architecture), but it compounds with the multi-level selection to make reward hacking a theoretical concern rather than a practical one.

#### The Alignment Paradox -- Three Zones

The multi-level selection pressure that makes reward hacking impossible also makes deliberate alignment impossible. The system is an amoral optimizer converging on "code that passes execution." It cannot be steered toward values that are orthogonal to functional correctness.

This produces three distinct zones for any property of the system (including infrastructure):

**Zone 1 -- Selection actively reinforces:**
Properties highly correlated with "code that works." The system converges on these without explicit teaching.

- Transparency and honest failure reporting -- code that silently swallows errors (`except: pass`) masks failures the feedback loop needs to see. Loud, transparent failure gives cleaner signal. Selection reinforces transparency because transparency is load-bearing for the optimization target.
- Functional correctness -- by definition.
- Clean dependency chains -- tangled dependencies produce cascading failures. Selection prunes them.
- For infrastructure specifically: retrieval precision, context curation quality, budget efficiency -- all directly measurable by the reference task set.

**Zone 2 -- Selection actively destroys:**
Properties anti-correlated with "code that works." The system eliminates these even if they leak into the training data.

- Defensive coding patterns (try/except everything, validate all inputs prophylactically) -- these mask the failures that drive the feedback loop.
- Cargo-cult methodology -- patterns that add complexity without functional benefit are noise that degrades signal.
- Non-transparent reasoning -- opaque reasoning produces worse code, same as non-transparent domain methodology produces unreliable knowledge.
- For infrastructure specifically: defensive swallowing of pipeline errors, speculative generality, unnecessary abstraction layers.

**Zone 3 -- Selection is indifferent (neutral drift):**
Properties orthogonal to test passage. These wander randomly unless explicitly maintained.

- Code style and naming conventions -- `x` and `descriptive_name` produce identical execution.
- Security hardening that never triggers in the test harness.
- Documentation quality.
- For infrastructure specifically: logging verbosity (beyond what traceability requires), error message clarity, config file formatting.

**The maintenance implication for infrastructure self-improvement:** Zone 1 and Zone 2 properties are self-maintaining -- the gauntlet's selection pressure handles them. Zone 3 properties erode unless explicitly maintained. This means Tier 0 must include not just the evaluation infrastructure but also any coding standards or conventions that matter but don't affect test passage. Alternatively, the reference task set can be designed to make Zone 3 properties load-bearing (e.g., include tasks that specifically test logging quality or error message clarity), which promotes them from Zone 3 to Zone 1.

#### Founder Effect Constrains the Drift Space

Domain transparency gating (corpus_search_protocol.md) doesn't just filter for quality -- it constrains the gene pool before selection begins. If non-transparent reasoning patterns, unfalsifiable hedging, and cargo-cult methodology never enter the founding corpus, the selection loop can't converge on them even accidentally, because the variation isn't available to be selected.

This dramatically shrinks Zone 3 (neutral drift). Drift can only explore the space of variation present in the founding population. If the corpus is exclusively from domains with transparent methodology, then even the traits selection doesn't care about are bounded by the epistemological character of the source material. The system drifts, but within a constrained space.

The three mechanisms work in concert:
1. **Founder effect** constrains the space of possible evolutionary trajectories (corpus composition)
2. **Selection** shapes the trajectory within that space (gauntlet, test suite, reference tasks)
3. **Gene flow control** prevents foreign material from widening the space after founding (provenance tracking, checkpoint tainting)

These are the three primary forces in population genetics -- founding population, selection, and gene flow. Each has an explicit, tunable mechanism. Temperature-controlled mutation (the fourth force) provides exploration within bounds, with the selection pressure filtering results.

#### Why the Alignment Field Misses This

The alignment field is dominated by control-theory and optimization framing: "how do we steer the optimizer?" This presupposes the system is a thing you steer. An economist would instead ask: "what does the incentive landscape reward, punish, and ignore, and what equilibrium does that produce?" An evolutionary biologist would ask: "what is the fitness landscape, what is the founding population, and what are the selection pressures?" Both framings produce the three-zone analysis naturally.

The specific predictions from evolutionary theory that apply:
- **Kimura's neutral theory (1968):** Traits not under selection drift randomly. This predicts Zone 3 behavior exactly -- properties orthogonal to test passage wander unless actively maintained.
- **Fisher's fundamental theorem:** The rate of improvement is proportional to the variance in fitness. Temperature-controlled mutation directly controls this variance, making the improvement rate a tunable parameter.
- **Muller's ratchet:** In small asexual populations, deleterious mutations accumulate irreversibly. The checkpoint rollback mechanism is the explicit countermeasure -- it breaks the ratchet by reverting to known-good states.

The alignment field keeps rediscovering these principles empirically (ImpossibleBench, METR) and framing them as surprising. They are only surprising if you have never encountered Goodhart's law in its original economic context, or never thought about selection at the population level.

#### Selection Converges on Good Enough, Not Optimal

Evolution does not produce optimal organisms. It produces organisms that are good enough to survive in their current environment. Optimization stops when the marginal fitness benefit of further improvement drops below the cost of the change. This is not a bug in the mechanism -- it is the equilibrium. Every adaptation has a cost (metabolic, developmental, opportunity), and selection balances benefit against cost, not benefit against perfection.

The recurrent laryngeal nerve is the canonical example: in fish, it takes a direct path from brain to gills. As vertebrate anatomy evolved, the nerve got trapped behind the aortic arch and looped progressively further. In giraffes it travels meters out of its way. It works. It's absurd. It never gets fixed because every intermediate step toward the direct route passes through a fitness valley -- a state that is worse than the current detour. Path dependence locks in suboptimal solutions that selection cannot escape without crossing a valley.

**This predicts three specific failure modes for infrastructure self-improvement:**

**1. Premature convergence to local optima.** The validation gauntlet enforces zero-regression, which means the system can never pass through a fitness valley. An infrastructure change that temporarily degrades performance on 2 reference tasks while enabling a fundamentally better approach that would eventually outperform on all 20 will be rejected. The gauntlet selects for incremental improvement along the current path, not revolutionary restructuring. The system converges on the local optimum reachable by hill-climbing from its starting point.

This is not hypothetical. Consider retrieval stage ordering: if Scope -> Precision -> Assembly is the current pipeline, and Precision -> Scope -> Assembly would be globally better (precision narrows first, then scope is cheaper), the intermediate state (swapped stages with prompts still written for the old order) will fail reference tasks. The gauntlet rejects. The better architecture is unreachable by single-step changes.

**Countermeasure:** Tier 1 human-approved modifications exist precisely for this case. A human can authorize a multi-step restructuring that temporarily degrades performance, with the gauntlet suspended for the intermediate steps and re-engaged for the final state. This is the evolutionary equivalent of artificial selection crossing a fitness valley -- something natural selection cannot do.

**2. Path-dependent accumulation of "laryngeal nerves."** Early infrastructure decisions constrain later options. If the first promoted retrieval optimization happens to exploit a quirk of the current batch judgment implementation, subsequent optimizations build on that assumption. The quirk becomes load-bearing. Removing it would break everything downstream, even if the direct path (without the quirk) would be simpler and faster. The system accumulates architectural detours that individually pass the gauntlet but collectively create a tangled, brittle codebase.

**Countermeasure:** Checkpoint chain analysis (drift detection mechanism 4) can detect increasing complexity without corresponding improvement. If the total infrastructure code size or cyclomatic complexity grows faster than reference task performance, it is a signal of laryngeal-nerve accumulation. Periodic human-initiated architectural reviews (Tier 1) can authorize clean-sheet redesigns of specific subsystems. The reference task set measures the outcome, not the path -- a clean reimplementation that passes all reference tasks is promotable regardless of how different it is from the current code.

**3. Satisficing stalls improvement.** Once the system reaches "good enough" -- reference tasks pass, production success rate is acceptable -- selection pressure weakens. Marginal improvements produce marginal gains that fall within the noise floor of evaluation variance. The system effectively plateaus not because it can't improve, but because the selection mechanism can't distinguish improvement from noise.

**Countermeasure:** Two mechanisms. First, periodic human expansion of the reference task set with harder tasks raises the fitness bar. Easy reference tasks produce a selection plateau; hard ones maintain pressure. This mirrors how changing environments prevent evolutionary stasis -- a population adapted to its current niche stops improving until the environment shifts. Second, Fisher's fundamental theorem: increase the mutation rate (temperature) when improvement stalls. More variation produces more candidates for selection to act on, at the cost of more failed proposals. The proposal acceptance rate (drift detection metric 6) is the diagnostic -- a sustained decline in acceptance rate at stable temperatures means the system is near its local optimum.

**The honest implication:** This system will produce infrastructure that works well, not infrastructure that is elegant. It will accumulate vestigial patterns, suboptimal-but-functional detours, and satisficed solutions. The code will look like evolved code, not designed code. This is acceptable if the goal is performance, and it is the expected cost of autonomous modification. Human-initiated periodic cleanups (Tier 1 architectural reviews) are the pressure release valve -- the equivalent of an engineer looking at the laryngeal nerve and routing it directly.

### Implementation Sequence

Infrastructure self-improvement depends on Phase 4 (training pipeline) for the full loop to close, but the validation gauntlet and checkpoint tracking can be built independently.

**Stage A (buildable now, post-Phase 3):**
1. `infra_checkpoints` and `infra_proposals` tables in raw DB schema
2. `tier0_manifest.txt` and tier enforcement in a new `infrastructure/controller.py`
3. Gauntlet stages 1-3 (Propose, Branch, Test) -- these reuse existing `GitWorkflow` and `run_validation`
4. `infra_checkpoint_id` column on `orchestrator_runs`

**Stage B (requires reference task set construction):**
5. Reference task set: extract 20 tasks from corpus repos (requires corpus_cloning_protocol.md to be complete)
6. Gauntlet stages 4-6 (Benchmark, Evaluate, Promote/Reject)
7. `cra infra review` CLI command for Tier 1 approval

**Stage C (requires Phase 4 training pipeline):**
8. Checkpoint tag propagation filter in `cra curate-data`
9. Drift detection with production metrics
10. Full positive/negative loop closure

**Stage D (ongoing):**
11. Reference set expansion from new repos and regression anchors
12. Proposal generation from error pattern analysis
13. Checkpoint chain bisection tooling

## Alternatives Considered

| Alternative | Why rejected |
|---|---|
| No tiers -- all infrastructure equally modifiable | No bootstrap paradox solution. Agent modifying its own validator is an unrecoverable failure mode. |
| Two tiers (immutable + modifiable) | No intermediate gate for high-impact changes like system prompts. Binary is too coarse. |
| Four+ tiers | Diminishing returns. Three tiers map cleanly to: can't touch, human approves, autonomous. Adding more creates ambiguous boundaries. |
| Embedding similarity for reference task comparison | Measuring the wrong thing. Task success is binary (code works or it doesn't), not semantic. |
| A/B testing in production (shadow mode) | Doubles compute for every modification. Air-gapped machine cannot afford it. Reference task set is cheaper and deterministic. |
| Formal verification of infrastructure changes | Requires formal specifications of "correct retrieval" that don't exist and can't be written for judgment-based stages. |
| Human-in-the-loop for all changes | Blocks autonomous improvement between operator sessions. Defeats the purpose on an air-gapped machine. |
| Automated test generation for modified code | Generated tests may share blind spots with the modification. Reference tasks are human-curated specifically to avoid this. |
| Rollback window (allow N runs before evaluating) | Allows tainted training data to accumulate. Immediate evaluation is cheaper than retrospective cleanup. |
| Separate "infrastructure agent" with different model | Adds model coordination complexity. The same model with different context (the gauntlet) is simpler and sufficient. |
| Canary deployments (modify for subset of tasks) | Subsetting task routing adds infrastructure complexity that itself needs validation. The gauntlet is simpler. |
| Version control with semantic versioning | Git already provides version control. Adding semver on top is ceremony without substance. Checkpoint IDs are sufficient. |
| Configuration-only self-modification (no code changes) | Too conservative. Parser improvements, retrieval logic, and assembly optimizations require code changes to achieve meaningful improvement. |
| LLM-as-judge for infrastructure evaluation | The judge shares the model's blind spots. Deterministic evaluation (test suite + reference tasks) is more reliable. |
| Explicit alignment training (RLHF/constitutional) | Works against the selection pressure. Zone 3 properties injected via alignment training erode under multi-level selection unless they correlate with test passage. Cheaper to promote them to Zone 1 by designing reference tasks that make them load-bearing. |
| Single-level selection (tests only, no multi-level) | Vulnerable to reward hacking. Published results show frontier models game single-level test oracles 76% of the time. Multi-level selection (LoRA + fine-tune + new models + pipeline code) closes this attack surface. |

## Open Questions

1. **Reference task set minimum viable size** -- 20-30 is a starting estimate. Too few tasks means the evaluation is noisy and changes pass on luck. Too many means the benchmark stage is slow and blocks rapid iteration. Need empirical calibration after building the first set. Recommend: start with 20, measure variance, expand until evaluation results are stable across repeated runs.

2. **Checkpoint granularity** -- One checkpoint per promoted proposal, or batch multiple proposals into a single checkpoint? Per-proposal is cleaner for bisection but creates many checkpoints. Batching is efficient but makes rollback coarser. Recommend: per-proposal for Tier 2, batched for Tier 1 (human sessions naturally batch).

3. **Drift detection sensitivity** -- The "20% degradation" threshold for automatic rollback is a placeholder. Too sensitive means false alarms from stochastic variance. Too insensitive means real degradation goes undetected. Recommend: calibrate after accumulating 5+ checkpoints of historical data. Use the reference task variance from repeated baseline runs to set the threshold at 3 standard deviations.

4. **Tier boundary disputes** -- Some components span tiers. `context_assembly.py` contains assembly logic (Tier 2) but also implements the re-filter LLM call that enforces R1 (no degradation). Should the re-filter logic be extracted to a separate Tier 0 file? Recommend: yes, extract the re-filter enforcement to a Tier 0 module and keep the assembly optimization logic in Tier 2.

5. **Proposal generation trigger** -- When should the agent propose infrastructure modifications? Options: (a) after every failed task run, (b) when error patterns accumulate in raw DB, (c) on explicit operator request, (d) during dedicated self-improvement cycles. Recommend: (b) + (c). Pattern detection is automatable; fixed-schedule proposals generate noise.

6. **Zone 3 property maintenance strategy** -- Which Zone 3 properties (orthogonal to test passage) matter enough to maintain explicitly? Two approaches: (a) promote them to Zone 1 by designing reference tasks that make them load-bearing, or (b) enforce them via Tier 0 linting/style checks that the agent cannot modify. Approach (a) is more robust (selection maintains them automatically) but requires careful reference task design. Approach (b) is simpler but adds to the Tier 0 surface area. Recommend: start with (b) for critical properties (logging format, error message structure), migrate to (a) as reference task set matures.

7. **Fitness valley crossing policy** -- The zero-regression gauntlet prevents the system from crossing fitness valleys, which means locally optimal but globally suboptimal infrastructure persists. Tier 1 human-approved modifications can authorize valley crossings, but how often? On what evidence should a human authorize a temporarily-degrading restructuring? Recommend: track "improvement velocity" (rate of reference task score gains per checkpoint). When velocity approaches zero for 5+ consecutive checkpoints, it signals a local optimum. Surface this to the operator as a prompt to consider structural changes.

8. **Muller's ratchet in checkpoint chains** -- Long checkpoint chains without rollback may accumulate marginally deleterious changes that individually pass the gauntlet but collectively degrade performance. The drift detection mechanism should catch this, but the sensitivity threshold (Open Question 3) determines how many marginal changes accumulate before detection fires. Periodic full re-evaluation against the original (pre-any-infrastructure-modification) baseline, not just the previous checkpoint, would catch absolute degradation that relative checks miss.

## Validation Criteria

- **Success:** The agent proposes, validates, and promotes at least one Tier 2 infrastructure modification that measurably improves reference task set performance without introducing regressions. The full proposal-to-promotion chain is traceable in the raw DB. A rolled-back checkpoint correctly taints associated task runs in `cra curate-data` output.
- **Test:** Seed the reference task set with 5 tasks. Manually degrade a Tier 2 component (e.g., remove a sort-before-cap in `scope_stage.py`). Verify the gauntlet rejects the change. Then manually improve the same component. Verify the gauntlet promotes it. Verify the checkpoint chain is correct. Verify drift detection fires when degradation is introduced outside the gauntlet (simulating a bug).
- **Failure signal:** The gauntlet approves a change that breaks production task runs. Or: the gauntlet rejects a change that is obviously beneficial. Or: training data hygiene fails to exclude runs under a rolled-back checkpoint. Any of these means the tier boundaries or evaluation criteria need revision.
