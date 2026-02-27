# Population-Level Evolution — Design Record

**Date:** 2026-02-27
**Status:** Proposed
**Depends on:** `infrastructure_self_improvement.md`, `c_language_choice.md`, `single_generalist_model.md`
**Framework:** `research_reviews/evolutionary_economics_of_self_improvement.md`

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

A meta-Jane (which can be a small, cheap classifier model) evaluates lineages periodically:

- Run each Jane on the same reference task set
- Compare outcomes: correctness, code quality, token efficiency, architectural coupling metrics
- The winning lineage continues; the losing lineage is replaced

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
3. SELECT: Meta-Jane compares lineages on reference tasks + architectural metrics
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

## What This Does Not Solve

- **Zone 3 public goods.** Multiple Janes all under-provide documentation and code style equally. Population competition doesn't change the incentive structure for unpriced properties. Tier 0 linters and style enforcement remain necessary.
- **Reference set representativeness.** All Janes evaluate against the same reference task set. If the set is biased, all lineages over-optimize for the same bias. The selector inherits the bias. Only human expansion of the reference set addresses this.
- **Temporal blindness (fully).** The hedging mechanisms reduce the damage but don't eliminate the asymmetry. The system still cannot price future costs. It can only invest blindly against them.

## Validation Criteria

- **Success:** Over 20+ generations, the population-level system produces architecturally cleaner code (lower coupling, fewer path-dependent assumptions) than a single-lineage control, while matching or exceeding functional correctness.
- **Test:** Run a single-lineage Jane and a population-level Jane from the same starting point on the same reference tasks for the same total compute budget. Compare architectural metrics and reference task performance at generations 10, 20, and 30.
- **Failure signal:** Population-level Jane consistently underperforms single-lineage Jane on reference tasks despite using the same total compute. This would indicate the overhead of maintaining multiple lineages and running selection exceeds the benefit of escaping local optima.

## Hardware Requirements

Two RTX 5090 (32GB each) with small models:
- 0.5-1B models for retrieval classification and meta-selection: trivial inference cost
- 1-4B model for code generation: fits easily, multiple instances feasible
- Training concentrated on one lineage at a time: full GPU utilization
- Estimated overhead vs. single lineage: ~20-30% (selection evaluation + diversity maintenance)
