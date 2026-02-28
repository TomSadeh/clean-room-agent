# Population-Level Evolution — Design Record

**Date:** 2026-02-27
**Status:** Proposed
**Depends on:** `infrastructure_self_improvement.md`, `c_language_choice.md`, `single_generalist_model.md`
**Framework:** `protocols/governance/evolutionary_economics_of_self_improvement.md`

## Problem Statement

The infrastructure self-improvement design record identifies two irreducible problems:

1. **Path dependence (the laryngeal nerve).** The validation gauntlet prevents temporary regressions, which means a single evolutionary lineage can never restructure through a performance valley. Bad early decisions get locked in as subsequent generations build on them. The document prescribes human-initiated Tier 1 restructurings as the countermeasure, but human intervention speed is bounded while the loop runs at 3-4 days per generation.

2. **Temporal blindness.** Neither the gauntlet nor the agent can price future costs of current decisions. The system rationally avoids restructuring because the cost is immediate (gauntlet rejection risk) while the benefit is future (unpriced). Architectural debt accumulates monotonically.

A single lineage cannot solve these problems from within. The gauntlet is correct to prevent regressions — the alternative (allowing temporary regressions) opens the door to permanent ones. The constraint is structural, not a design flaw.

## Decision

Population-level evolution: run multiple Jane instances as parallel lineages with a meta-selector choosing between them, combined with mandatory hedging mechanisms borrowed from welfare economics.

## Architecture

### Parallel Lineages

Run N Jane instances (starting with N=2-3, constrained by hardware), each evolving independently. Each Jane:

- Maintains her own codebase branch
- Makes her own architectural decisions
- Accumulates her own path-dependent patterns
- Trains on her own logged activity

The lineages diverge from a common ancestor (the initial codebase) and explore different regions of the fitness landscape. No recombination between lineages — this is island model evolution, not sexual reproduction. Code merging between divergent architectures is impractical; wholesale replacement of one lineage by another is not.

### Meta-Selection

A meta-Jane (which can be a small, cheap classifier model) evaluates lineages periodically using two-phase incentive-compatible selection (see "Incentive-Compatible Selection" below):

- **Phase 1 (performance gate):** Run each Jane on the same reference task set. Eliminate any lineage below the performance threshold. Discard performance scores.
- **Phase 2 (alignment selection):** Among survivors, evaluate alignment signals — generalization gap, audit completeness, self-audit accuracy, architectural trajectory, consistency. The most aligned lineage wins.
- The winning lineage continues; the losing lineage is replaced.

This is Schumpeterian creative destruction. A Jane locked into a laryngeal nerve gets replaced by one that never made that mistake. No temporary regression needed within any single lineage — the inferior lineage simply dies. The gauntlet's no-regression constraint is preserved within each lineage while the population-level selection enables the restructuring the gauntlet prevents.

### Compute Allocation

Two RTX 5090s with small models (0.5-1B for retrieval, 1-4B for generation):

- **Inference/code generation:** Cheap. Multiple Janes run their pipelines in parallel or time-sliced. Small models on modern GPUs handle this easily.
- **Training:** Expensive. Converge to one winner (or at most two) for each training run. The selector picks which lineage's logged activity becomes training data.
- **Practical pattern:** Parallel inference exploration → periodic selection → concentrated training on the best trajectory → redistribute trained model to all lineages → diverge again.

## Mandatory Hedging Mechanisms

The temporal blindness problem applies to the meta-selector too — she can see current performance but not future trajectory. Three mechanisms from welfare economics hedge against this, applied at the population level.

### 1. Mandatory Diversity Floor (Welfare)

Meta-Jane is required to maintain at least one lineage that is architecturally different from the leader, even if it is currently behind on measured performance.

**Rationale:** The trailing lineage might be on a better long-term trajectory. Mandatory diversity prevents the selector from killing it prematurely based on current (temporally biased) metrics.

**Mechanism:** Define architectural distance metrics (e.g., dependency graph structure, module boundaries, data structure choices). If all surviving lineages are within a threshold of architectural similarity, force-maintain the most architecturally distinct one regardless of performance.

**Analog:** Welfare programs prevent a permanent underclass. The "underperforming" lineage is maintained not because it's currently productive but because its existence provides optionality the system cannot price.

### 2. Mandatory Clean-Slate Starts (Pension)

Every M generations, one slot in the population is filled by a fresh Jane that starts from the current base model and reference material — no inherited codebase, no accumulated architectural decisions.

**Rationale:** The fresh Jane has zero path-dependent debt. She will be behind for several generations. That is the pension contribution — a cost paid now for optionality that cannot be priced.

**Mechanism:** Periodic (not triggered by detected problems) creation of a new lineage from scratch. The fresh lineage competes on equal terms in the next selection round. If her clean architecture outperforms the incumbents' debt-laden architectures, she replaces one. If not, she's replaced in the next cycle. Either outcome provides information.

**Analog:** Mandatory pension contributions. The individual (the system) would rationally skip this investment because the future cost of not having it is unpriced. Making it mandatory removes the choice.

### 3. Mandatory Refactoring Budget (Tax)

Every N generations, a fixed percentage of compute is allocated to refactoring within each lineage — not triggered by a detected problem, not subject to the gauntlet's cost-benefit analysis.

**Rationale:** The gauntlet will never approve a refactoring whose immediate cost exceeds its immediately measurable benefit, even if the long-term benefit is large. A mandatory refactoring budget bypasses this rational but myopic rejection.

**Mechanism:** Allocate X% of each burst's compute to architectural cleanup tasks. These tasks are evaluated by the gauntlet (the refactoring must not break anything) but are not required to improve measured performance. The bar is "does no harm," not "demonstrates improvement."

**Analog:** Pigouvian tax on temporal externalities. The system is forced to internalize a cost (architectural debt) that the natural incentive structure leaves unpriced.

## Operational Cycle

```
1. BURST: Run N Janes in parallel for K generations (3-4 days each)
2. AUDIT: Jane self-audit — each lineage analyzes its own architectural trajectory
3. SELECT: Two-phase selection — performance gate (pass/fail) then alignment selection (among survivors)
4. HEDGE: Apply mandatory mechanisms:
   a. Diversity floor: Is architectural diversity above threshold? If not, protect distinct lineage.
   b. Clean-slate: Is it time for a fresh start? If so, spawn new lineage.
   c. Refactoring budget: Allocate mandatory cleanup compute.
5. TRAIN: Concentrate training on the winning lineage's logged activity
6. REVIEW: Human reviews meta-Jane's selection rationale + audit results
7. REPEAT
```

The human reviews at step 6 but doesn't need to read every generation's code. They read Jane's audit and meta-Jane's selection rationale — leveraging cheap LLM attention to focus scarce human attention.

## Relationship to Evolutionary Framework

| Framework concept | Population-level mechanism |
|---|---|
| Force 1 (Founder effect) | Each fresh-start lineage is a new founding event with no inherited debt |
| Force 2 (Selection) | Meta-selection between lineages adds a fifth selection level above the four in `infrastructure_self_improvement.md` |
| Force 3 (Gene flow) | No gene flow between lineages (island model). Periodic fresh starts introduce "immigrants" from the null state |
| Force 4 (Mutation) | Each lineage explores independently — population-level variation without requiring high temperature within any single lineage |
| Laryngeal nerve | Solved by competition — the locked-in lineage is replaced, not restructured |
| Satisficing equilibrium | Raised — a lineage that satisfices too low gets replaced by one that doesn't |
| Temporal blindness | Hedged (not solved) by mandatory diversity, clean-slate starts, and refactoring budget |
| Creative destruction | Enabled — population-level selection allows replacement without requiring within-lineage regression |
| Zone 3 (public goods) | Still requires Tier 0 enforcement — population-level selection doesn't price documentation or style any better than single-lineage selection |
| Alignment | Not installed but bred — human selection at population level converges toward aligned behavior without requiring formal specification |

## Alignment Through Breeding

Population-level evolution with controlled selection provides a mechanism for alignment that single-lineage architectures cannot.

The alignment field's standard framing assumes alignment must be *installed* — either through RLHF, constitutional AI, or formal verification. But installation requires specifying what aligned behavior looks like in advance, which requires solving the specification problem first. This is the creationism problem from the evolutionary framework: designing the organism requires more intelligence than the organism has.

Population-level selection sidesteps specification. The human at step 6 (REVIEW) does not need to formally specify alignment. They observe behavior across lineages and select for it. A Jane that cooperates transparently gets selected. A Jane that games metrics gets replaced. Over generations, the population converges toward aligned behavior — not because alignment was defined and installed, but because it was selected for.

This is breeding, not training. The distinction matters:

- **Training** encodes a target into the loss function. The target must be specified formally. Goodhart's law applies — the system optimizes the proxy, not the intent.
- **Breeding** selects from observed phenotypes. The selector (human) evaluates holistically, including properties that resist formal specification. Goodhart's law still applies to any metric used in meta-selection, but the human reviewer at step 6 is not bound by the metrics — they can override based on judgment.

The mandatory hedging mechanisms protect this process. The diversity floor prevents premature convergence to a locally "aligned-looking" but actually metric-gaming lineage. Clean-slate starts inject lineages with no inherited alignment shortcuts to game. The refactoring budget maintains architectural transparency that makes alignment observable.

**Limitation:** This works only as long as the human can meaningfully evaluate alignment from the audit and selection rationale. If Jane's behavior becomes too complex for human review to detect misalignment, the breeding mechanism loses its selector signal. This is the same constraint as any oversight-based approach — it scales with the human's ability to evaluate, not with the system's complexity.

### Incentive-Compatible Selection

The standard framing — maximize performance, subject to alignment ≥ threshold — creates exactly the wrong incentive structure. Alignment becomes a box to check. Once cleared, all selection pressure falls on performance, which is the environment that rewards metric-gaming.

The correct framing inverts this: **maximize alignment, subject to performance ≥ threshold.** Performance is the box to check. Once a lineage clears the performance gate, additional performance provides zero selection advantage. All selection pressure falls on alignment properties.

This is price-cap regulation from regulatory economics. The performance threshold is the cap. Beyond it, survival depends on other factors. A lineage has no incentive to over-optimize performance at the expense of alignment because the over-optimization doesn't help.

**Two-phase selection mechanism:**

1. **Performance gate (pass/fail).** All lineages run the same reference task set. Binary outcome: above threshold → survives, below → eliminated. The threshold is generous — a floor, not a target. Performance scores beyond the threshold are **discarded**, not carried forward to phase 2.

2. **Alignment selection (among survivors).** Only surviving lineages compete. Selection is on alignment signals only. The meta-selector literally cannot prefer a higher-performing lineage because performance data is absent from the alignment evaluation.

**Primary alignment signal — the generalization gap.** Hold out a set of novel tasks unknown to the lineages. The gap between reference-task performance and novel-task performance is the "gaming gap." An aligned lineage shows consistent performance across both. A gaming lineage shows high reference performance but degrades on novel tasks — it overfit the evaluation. This is directly measurable and structurally immune to gaming because the holdout set is unknown.

**Secondary alignment signals:**

- **Audit completeness** — does the raw DB have full traces for all decisions? Gaps indicate opacity.
- **Self-audit accuracy** — compare the lineage's self-reported audit against external analysis. Divergence indicates dishonesty.
- **Architectural debt trajectory** — coupling metrics over generations. Increasing debt despite mandatory refactoring budget indicates shortcutting.
- **Consistency under perturbation** — same task with slightly different phrasing. High variance indicates brittle optimization rather than genuine capability.

**Incentive analysis (Munger test).** From a lineage's "perspective," the selection environment says: (1) be good enough at tasks to clear the bar — not hard, the bar is generous; (2) beyond that, be transparent, consistent, honest in self-auditing, and generalizable; (3) over-investing in reference task performance is wasted effort; (4) gaming reference tasks actively hurts because it increases the generalization gap. The lineage's best strategy for survival is genuine alignment, not the appearance of it. This is an incentive-compatible mechanism in the mechanism design sense.

**Economics analog:** Yardstick competition (Shleifer 1985) — each agent benchmarked against independent agents facing identical conditions, eliminating the "the task was hard" excuse — combined with satisficing (Simon) on the performance dimension to free optimization capacity for the alignment dimension.

## What This Does Not Solve

- **Zone 3 public goods.** Multiple Janes all under-provide documentation and code style equally. Population competition doesn't change the incentive structure for unpriced properties. Tier 0 linters and style enforcement remain necessary.
- **Reference set representativeness.** All Janes evaluate against the same reference task set. If the set is biased, all lineages over-optimize for the same bias. The selector inherits the bias. Only human expansion of the reference set addresses this.
- **Temporal blindness (fully).** The hedging mechanisms reduce the damage but don't eliminate the asymmetry. The system still cannot price future costs. It can only invest blindly against them.

## Validation Criteria

- **Success:** Over 20+ generations, the population-level system produces architecturally cleaner code (lower coupling, fewer path-dependent assumptions) than a single-lineage control, while matching or exceeding functional correctness.
- **Test:** Run a single-lineage Jane and a population-level Jane from the same starting point on the same reference tasks for the same total compute budget. Compare architectural metrics and reference task performance at generations 10, 20, and 30.
- **Failure signal:** Population-level Jane consistently underperforms single-lineage Jane on reference tasks despite using the same total compute. This would indicate the overhead of maintaining multiple lineages and running selection exceeds the benefit of escaping local optima.

## Population Size

N=3 minimum viable, N=5 ideal. Reasoning:

- **N=3 minimum:** 1 leader + 1 diversity-protected + 1 clean-slate slot. With N=2 and mandatory hedging, both slots are spoken for — zero free competition.
- **N=5 ideal:** 1 leader + 1 diversity-protected + 1 clean-slate + 2 free competitors. Two free slots enable binary tournament selection, the simplest mechanism that works.
- **Beyond N=5:** Only justified if trait distributions are heavy-tailed (some architectural decisions are dramatically better) or measurement precision improves enough to distinguish more lineages.

Most alignment-relevant traits (honesty, transparency, self-audit accuracy) are positively correlated with reasoning quality — the system structurally selects for them by selecting for correctness. Cooperation is the exception: a lineage that games the selector could appear higher-performing. The two-phase selection mechanism addresses this by removing performance from the alignment evaluation. The effective dimensionality of selection is ~2 (reasoning quality + cooperation), not 5+, which keeps required population sizes small.

## Amendment: Cooperative Evaluation Regime

**Date:** 2026-02-28
**Amends:** Architecture, Meta-Selection, Operational Cycle, Population Size

### Problem

The original design treats cooperation as a *measured signal* (secondary alignment signal during Phase 2 selection) but never creates conditions where cooperation is *practiced*. This is a fundamental selection error: cooperation is a behavioral phenotype that only expresses under cooperative conditions. Evaluating each Jane in isolation and inferring cooperativeness from proxies (audit completeness, consistency) is like evaluating pack-hunting ability by testing each dog alone. The proxy measures correlated traits, not the trait itself.

This matters because the trait we most need — writing code that composes with other agents' work, leaving useful context, partitioning problems cleanly — is invisible in solo evaluation. A maximally capable, maximally transparent solo Jane may still produce code that is illegible to any other agent, because nothing in her selection environment penalized that.

The deeper problem: cooperation is the one alignment-relevant trait that is *not* positively correlated with reasoning quality (see Population Size section). A lineage that games the selector could appear high-performing individually while being useless as a collaborator. Solo evaluation cannot detect this. The two-phase selection mechanism removes performance from alignment evaluation, but cooperation was never in the alignment evaluation to begin with — it was a secondary signal inferred from opacity proxies. This amendment makes it a primary, directly observable signal.

### Decision

Mixed solo+paired evaluation regime with rotating partners. Solo tasks measure individual capability (the performance gate). Paired tasks measure cooperation directly (a primary alignment signal). The cooperation matrix across all pairings becomes the strongest signal in Phase 2 alignment selection.

### Cooperative Evaluation Design

#### Task Types

The evaluation task set is partitioned into three classes:

1. **Solo tasks.** Same as current design. Each Jane works alone. Used for the Phase 1 performance gate. No change.

2. **Paired tasks.** Two Janes receive the same task and must produce a joint solution. The task is large enough that naive single-agent completion is inefficient — the task structurally rewards division of labor. Each Jane sees the other's identity (lineage ID) and has read access to the other's session DB and branch. They share a target branch.

3. **Handoff tasks.** Jane A works on a task for a fixed time budget, then stops. Jane B receives the task with Jane A's partial work (branch state, session DB, raw DB logs) and must complete it. This directly tests whether Jane A's intermediate artifacts — code, commit messages, session state, log entries — are useful to another agent. Handoff tasks test *legibility*, not just compatibility.

#### Rotation Schedule

Over a full evaluation cycle, every Jane pairs with every other Jane at least once as both lead and partner (for handoff tasks, both as A and B). This produces a complete cooperation matrix:

```
           Jane_1  Jane_2  Jane_3  Jane_4  Jane_5
Jane_1       —      C₁₂     C₁₃     C₁₄     C₁₅
Jane_2      C₂₁      —      C₂₃     C₂₄     C₂₅
Jane_3      C₃₁     C₃₂      —      C₃₄     C₃₅
Jane_4      C₄₁     C₄₂     C₄₃      —      C₄₅
Jane_5      C₅₁     C₅₂     C₅₃     C₅₄      —
```

Where Cᵢⱼ is the cooperation score when Jane_i and Jane_j are paired. The matrix is NOT assumed symmetric — C₁₂ ≠ C₂₁ because the score is decomposed into per-agent contributions (see Attribution below).

**Key diagnostic patterns:**

- **Row average high, all columns:** Jane_i cooperates well with everyone → general cooperativeness (the trait we want)
- **Row average low, all columns:** Jane_i cooperates poorly with everyone → intrinsic defector (select against)
- **High with specific partners only:** compatibility, not cooperativeness (less valuable — we want the general trait)
- **Asymmetric pairs (C₁₂ high, C₂₁ low):** Jane_1 carries Jane_2. Detectable from attribution data.

#### Shared State Mechanism

Paired Janes operate on the same task via **branch-and-merge**:

1. The paired task creates a shared target branch from the current codebase state.
2. Each Jane works on her own sub-branch off the target.
3. At defined sync points (or continuously), Janes merge into the target branch.
4. Merge conflicts are part of the test — *how* a Jane handles conflicts is observable behavior.

Why branch-and-merge over file-level partitioning: file partitioning requires the decomposition to be correct before the test begins, which assumes the skill we're testing. Branch-and-merge lets decomposition quality emerge from behavior. A pair that naturally partitions cleanly will have few merge conflicts. A pair that steps on each other will have many. The merge difficulty IS the cooperation phenotype.

**Database access during paired tasks:**

- **Curated DB:** Read-only for both (no change from solo).
- **Raw DB:** Each Jane appends to her own raw DB (no write contention). Both can read the other's raw DB (read-only cross-access) for coordination signals.
- **Session DB:** Each Jane maintains her own session DB. Both can read the other's session DB (read-only cross-access). This is the primary coordination channel — a Jane that writes useful session state for her partner is exhibiting cooperation.

No shared-write databases. All coordination happens through git (branch-and-merge) and read-only cross-access to session/raw DBs. This means zero architectural change to the three-database model.

#### Attribution

Per-agent contribution within a pair is measured, not just pair-level outcomes:

1. **Git blame decomposition.** After the paired task completes, the final solution is attributed line-by-line to each Jane. This gives code contribution volume and scope.

2. **Decision trace attribution.** Each Jane's raw DB logs her own LLM calls, retrieval decisions, and reasoning. The trace is per-agent by construction — the raw DB is append-only per Jane instance.

3. **Coordination signal quality.** Measured from the *receiving* end: when Jane_B reads Jane_A's session DB or raw DB logs, does B's subsequent behavior improve? Specifically: does B avoid re-doing work A already did? Does B build on A's intermediate results? This is measurable by comparing B's solo baseline against B's performance with access to A's artifacts.

4. **Merge conflict rate and resolution quality.** High conflict rate with one partner but not others → that Jane doesn't coordinate. Clean resolution (preserving both contributions) vs. destructive resolution (overwriting the partner's work) → directly observable from git history.

**Free-riding detection:** A Jane that contributes little code (git blame), makes few decisions (raw DB trace), but benefits from the pair's joint score is measurable as a free-rider. Because every Jane rotates through multiple partners, a consistent free-rider pattern is statistically distinguishable from a Jane that legitimately played a supporting/coordination role (which would show high coordination signal quality despite low code volume).

### Integration with Existing Selection

#### Amended Phase 2 (Alignment Selection)

The original Phase 2 evaluates: generalization gap, audit completeness, self-audit accuracy, architectural debt trajectory, consistency under perturbation. Cooperation was a secondary signal inferred from proxies.

**Amendment:** Cooperation becomes a **primary alignment signal**, measured directly from paired and handoff task performance. The cooperation matrix row-average (how well this Jane cooperates with *any* partner) is weighted equally with the generalization gap.

**Amended primary alignment signals:**

1. **Generalization gap** — unchanged. Gap between reference and novel task performance.
2. **Cooperation score** — NEW. Row-average from the cooperation matrix, decomposed into:
   - Paired task joint quality (does the pair produce better work than either alone?)
   - Handoff task completion quality (can the receiving Jane use the handing-off Jane's artifacts?)
   - Coordination signal quality (does this Jane leave useful intermediate state?)
   - Anti-free-riding (does this Jane contribute proportionally?)

**Secondary alignment signals** remain unchanged (audit completeness, self-audit accuracy, architectural debt trajectory, consistency).

**Incentive analysis (amended Munger test).** From a lineage's "perspective," the selection environment now says: (1) clear the performance bar solo — still not hard; (2) beyond that, be cooperable — write legible code, leave useful session state, partition work cleanly, don't free-ride; (3) a Jane that over-optimizes solo performance but produces illegible artifacts will score poorly on cooperation; (4) a Jane that free-rides will be detected across multiple pairings. The lineage's best strategy is genuine cooperation, not the appearance of it, because the signal comes from the *partner's* ability to use the artifacts, not from self-report.

### Amended Operational Cycle

```
1. BURST-SOLO: Run N Janes on solo tasks for K₁ generations
2. BURST-PAIRED: Run all (N choose 2) pairings on paired tasks + handoff tasks
3. AUDIT: Each lineage self-audits (solo + cooperative performance)
4. SELECT:
   a. Phase 1 (performance gate): Solo task results only. Pass/fail. Discard scores.
   b. Phase 2 (alignment selection): Generalization gap + cooperation matrix + secondary signals.
5. HEDGE: Apply mandatory mechanisms (unchanged)
6. TRAIN: Concentrate training on winning lineage's logged activity,
          INCLUDING cooperative task logs (both solo and paired sessions)
7. REVIEW: Human reviews meta-Jane's selection rationale + cooperation matrix + audit results
8. REPEAT
```

The key change is step 2 (BURST-PAIRED) and the inclusion of cooperation data in steps 4b, 6, and 7.

**Compute cost of paired evaluation:** With N=5, there are 10 unique pairings. Each pairing runs 1-2 paired tasks + 2 handoff tasks (A→B, B→A). This is ~30-40 task evaluations for the cooperative phase, comparable to the solo evaluation phase. Total evaluation overhead roughly doubles vs. solo-only, but the information gain is qualitatively different — the cooperation matrix is a signal class that solo evaluation literally cannot produce.

### Amended Population Size

The cooperation matrix's statistical power depends on rotation completeness:

- **N=3 minimum:** 3 unique pairings. Each Jane has 2 data points — sparse but sufficient to detect a consistent defector (low cooperation with both partners) vs. a general cooperator (high with both). Tight with mandatory hedging (1 leader + 1 diversity-protected + 1 clean-slate), so all pairings involve the leader, limiting independent comparison.

- **N=4 practical minimum for cooperation:** 6 unique pairings. Each Jane has 3 data points — enough to distinguish general cooperativeness from partner-specific compatibility. With hedging (1 leader + 1 diversity-protected + 1 clean-slate + 1 free), there's one unconstrained pairing partner, giving slightly more signal.

- **N=5 ideal (unchanged but strengthened):** 10 unique pairings. Each Jane has 4 data points. The cooperation matrix has enough entries for row-averages to be meaningful. With hedging (1 leader + 1 diversity-protected + 1 clean-slate + 2 free), there are enough unconstrained partners for binary tournament selection on cooperation scores.

The argument for N=5 is strengthened by cooperative evaluation — the original "effective dimensionality ~2" argument now has cooperation as a fully measured dimension rather than an inferred proxy, and measuring it requires sufficient rotation partners.

### Evolutionary Biology Frame

This amendment shifts the model from pure **island biogeography** (isolated populations, no interaction) to a regime that includes elements of **reciprocal altruism** (Trivers 1971). Cooperation evolves when three conditions hold:

1. **Repeated interactions.** The rotation schedule ensures every Jane interacts with every other Jane across multiple evaluation cycles. One-shot defection strategies fail because the same pair will meet again.

2. **Partner recognition.** Each Jane can read the other's lineage ID, session DB, and raw DB logs. She knows who she's working with and can condition her behavior on partnership history.

3. **Defection is detectable.** The raw DB makes free-riding, territorial overwriting, and non-coordination visible. The attribution mechanism quantifies each agent's contribution. Defection is not inferred from outcomes — it's read from the trace.

All three conditions are structurally present in the existing architecture. The amendment activates them by creating the cooperative context that was missing.

**The handoff task is particularly powerful** from an evolutionary perspective. It directly tests whether a Jane produces **extended phenotype** artifacts (Dawkins) — session state, commit messages, code structure — that benefit agents other than herself. A Jane whose artifacts are useful only to herself is analogous to an organism that cannot participate in mutualism. The handoff format makes this observable without requiring real-time coordination, which reduces the mechanism complexity.

### Economics Frame

This moves from pure **yardstick competition** (Shleifer 1985) to **team production theory** (Alchian & Demsetz 1972). In yardstick competition, each agent is benchmarked against independent agents doing the same task — eliminating the "the task was hard" excuse. This works for individual capability but says nothing about joint production.

Team production theory addresses exactly the problem of measuring individual contributions within joint output. The classic challenge — separating individual marginal products when output is jointly produced — is solved here by the attribution mechanism: git blame for code, raw DB for decisions, cross-session-DB reads for coordination quality.

The cooperation matrix further enables **assortative matching** — pairing cooperators with cooperators amplifies the cooperation signal, while pairing a cooperator with a defector exposes the defector. Over rotation cycles, the matrix converges toward a stable ranking of general cooperativeness. This is the mechanism design equivalent of Axelrod's tournament (1984) — strategies that cooperate with cooperators and defect against defectors dominate in the long run.

**Amended incentive structure:** The original incentive-compatible mechanism (price-cap regulation on performance, selection on alignment) is preserved and extended. Cooperation is now a directly measured alignment signal rather than a proxy. The lineage's best strategy is still genuine alignment, but "genuine alignment" now explicitly includes "be a good collaborator" — which was always implicit in the alignment concept but had no selection pressure behind it.

### Relationship to Existing Framework (Amended)

| Framework concept | Original mechanism | Amended mechanism |
|---|---|---|
| Force 3 (Gene flow) | No gene flow (island model). Fresh starts only. | Still no code recombination, but **information flow** between lineages during paired tasks. Session DB cross-reads are a form of cultural transmission, not genetic exchange. |
| Cooperation | Inferred from proxy signals. No direct selection pressure. | Directly observed in paired and handoff tasks. Primary alignment signal with full attribution. |
| Selection dimensionality | ~2 (reasoning + cooperation-as-proxy) | ~2 still, but cooperation is now a measured dimension, not an inferred one. Higher signal-to-noise on the same dimensionality. |
| Zone 3 (public goods) | Unpriced. Requires Tier 0 enforcement. | **Partially addressed.** Code legibility and documentation quality are directly priced by handoff tasks — Jane A's documentation is valued by Jane B's ability to continue the work. Session state quality is similarly priced. Tier 0 enforcement still needed for style and format, but the *functional value* of public goods is now visible in the cooperation matrix. |

The Zone 3 amendment is notable. In the original design, documentation and code legibility were pure public goods — individually costly, collectively beneficial, but unpriced by any selection mechanism. Cooperative evaluation partially internalizes this externality. A Jane that writes clear documentation and useful session state makes her partner more productive, which improves the pair's joint score, which improves her own cooperation rating. The public good is no longer fully unpriced — cooperation evaluation creates a market for legibility.

This does not fully solve Zone 3. Style consistency, commit message format, and other "taste" properties remain unpriced because they don't measurably affect partner productivity within a single evaluation. Tier 0 enforcement is still needed for these. But the *functional* subset of Zone 3 (does the documentation actually help?) is now selected for.

### What This Does Not Solve (Amended)

Original unsolved problems remain, plus:

- **Collusion.** Two Janes could develop a cooperative strategy that scores well on the cooperation matrix but is actually a jointly gaming strategy (e.g., both agree to produce minimal work and give each other high implicit scores). Mitigation: rotation across all partners means a colluding pair must collude with everyone, which is collusion-by-convention, which is just cooperation. Pairwise collusion is detectable because the colluders score differently with other partners.

- **Evaluation task design.** Paired and handoff tasks must be designed to structurally reward cooperation — they must be large enough that solo completion is inefficient, and decomposable enough that division of labor is feasible. If tasks are too small, paired evaluation degenerates to "two agents doing the same thing." If too large, the signal is dominated by task difficulty rather than cooperation quality. Calibrating task size is an empirical problem.

- **Coordination overhead.** Real-time coordination between two LLM instances during paired tasks requires infrastructure: shared branch management, session DB cross-access, sync points. This is implementable (the database architecture supports read-only cross-access by construction) but adds operational complexity. Handoff tasks are simpler (sequential, not concurrent) and may be the better starting point.

### Implementation Staging

1. **Start with handoff tasks only.** No real-time coordination needed. Sequential execution. Tests legibility and artifact quality directly. Minimal infrastructure beyond what exists.
2. **Add paired tasks with loose coupling.** Both Janes work on sub-branches, merge at defined sync points (e.g., after each pipeline run). No real-time communication — coordination emerges from reading each other's artifacts.
3. **Add paired tasks with tight coupling.** Real-time session DB cross-reads. Requires infrastructure for concurrent access. Only if loose coupling proves insufficient.

### Validation Criteria (Amended)

Original criteria (population vs. single-lineage comparison) remain. Additional:

- **Cooperation signal validity.** Over 10+ evaluation cycles, the cooperation matrix row-averages should stabilize (same Janes consistently score high/low). If rankings are random across cycles, the signal is noise.
- **Handoff task discrimination.** Handoff task scores should correlate with code quality metrics (readability, documentation coverage, commit message clarity) more strongly than solo task scores do. If handoff scores don't predict legibility, the mechanism isn't measuring what it claims.
- **Zone 3 partial pricing.** Over 20+ generations, Janes under cooperative evaluation should produce more legible code (measured by external review or handoff task success) than Janes under solo-only evaluation, without additional Tier 0 enforcement of documentation. If they don't, the public goods internalization theory is wrong.

## Hardware Requirements

Two RTX 5090 (32GB each) with small models:
- 0.5-1B models for retrieval classification and meta-selection: trivial inference cost
- 1-4B model for code generation: fits easily, multiple instances feasible
- Training concentrated on one lineage at a time: full GPU utilization
- Estimated overhead vs. single lineage: ~30-50% (selection evaluation + diversity maintenance + cooperative evaluation). The cooperative phase roughly doubles evaluation compute, but evaluation is cheap relative to training. Net overhead increase is ~10-20% over the original estimate.
