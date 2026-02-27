# The evolutionary economics of self-improving coding agents

**The AI alignment field has a disciplinary blind spot.** It frames self-improvement as a control problem — how do we steer the optimizer? — borrowing from control theory and optimization. This framing misses what two older disciplines have known for decades: evolutionary biology understands multi-level selection, fitness landscapes, and neutral drift. Economics understands incentive structures, Goodhart's law in its original context, and equilibrium analysis. Both produce sharper predictions about self-improving systems than the alignment literature manages, and both explain why the published "scary" results (ImpossibleBench, METR, Anthropic's generalized misalignment) describe a regime fundamentally different from a self-improving loop with multi-level selection.

This document develops these frameworks and applies them to the design of self-improving coding agents — systems that modify their own pipeline code, prompts, retrieval logic, and configuration through an evolutionary loop. The conclusions are specific and testable: such a system will naturally converge on transparency and honest failure reporting, will naturally destroy defensive coding and cargo-cult methodology, will plateau at "good enough" rather than optimal, and will accumulate path-dependent architectural debt that requires periodic human intervention to clear. These are not wishes or design goals — they are predictions from the incentive structure and selection dynamics.

---

## Part 0: Why these are the only frameworks that matter

### Evolution is the only known process that produces intelligence

This is not a rhetorical device. It is a statement of fact that constrains the entire discussion.

Every intelligence that has ever existed — from the problem-solving capacity of a crow to the abstract reasoning of a human — was produced by the same process: variation, selection, heritability, operating over generations. No other process in the observable universe has ever produced intelligence. Not crystals, not thermodynamics, not self-organizing criticality, not any other complex-systems phenomenon. Only evolution.

When we talk about "creating" intelligence through machine learning, we are talking about a process with variation (random initialization, stochastic gradient descent, temperature sampling), selection (loss functions, reward signals, evaluation benchmarks), and heritability (trained weights carry forward, fine-tuned models inherit from base models, training data from one generation informs the next). The question is not whether ML is "like" evolution. The question is whether we are paying attention to the only empirical precedent we have.

The alignment field largely does not. It imports frameworks from control theory (principal-agent models, corrigibility), philosophy (value alignment, preference learning), and game theory (mechanism design, reward shaping). These frameworks have never produced intelligence. They describe how to constrain, direct, or negotiate with intelligence that already exists. They have nothing to say about the dynamics of the process that creates it.

### Incentives are the only known mechanism by which intelligence allocates resources

The second empirical anchor: every known intelligence makes decisions by responding to incentive structures. This operates at every scale.

**At the individual level:** Organisms allocate energy, attention, and time based on expected returns. A foraging bird does not randomly sample the environment — it follows the patch model (Charnov 1976), leaving a food patch when the marginal return drops below the average return of traveling to a new patch. This is not metaphorical optimization — it is the literal mechanism by which organisms make allocation decisions, refined by millions of years of selection because organisms that allocated resources inefficiently were outcompeted.

**At the aggregate level:** Markets allocate resources across an entire economy based on price signals — compressed information about relative scarcity and value (Hayek 1945). No individual participant needs to understand the global allocation. Each responds to local incentives (prices, costs, expected returns), and the aggregate result is coordination without central planning.

**At the neural level:** Brains allocate attention and processing resources based on reward prediction error signals (Schultz et al. 1997). Dopamine pathways literally implement an incentive system that governs which stimuli receive processing priority and which actions are reinforced. The brain is, at the implementation level, an incentive-driven resource allocator.

The alignment field discusses AI decision-making in terms of objectives, goals, preferences, and values — all of which are abstractions layered on top of the actual mechanism, which is: the system responds to incentive gradients. If you want to predict what a system will do, don't ask what it "wants." Ask what the incentive structure rewards.

### The missing syllogism

Combining the two anchors:

1. Evolution is the only process that has ever produced intelligence.
2. Incentive-driven resource allocation is the only mechanism by which intelligence makes decisions.
3. A self-improving AI system is a process attempting to produce and improve intelligence.
4. Therefore, the relevant analytical frameworks are evolutionary biology (how the process works) and economics (how the resulting intelligence allocates resources).

The alignment field's control-theoretic framing is not wrong in the sense of being logically invalid. It is wrong in the sense of being irrelevant. You cannot meaningfully analyze a breeding program using the tools of dog training. You cannot meaningfully analyze a market economy using the tools of command-and-control management. You cannot meaningfully analyze a self-improving AI system using frameworks that describe the behavior of a finished intelligence, because the system is not a finished intelligence — it is a process that produces intelligence, and the dynamics of that process are evolutionary.

This does not mean alignment concerns are imaginary. It means they must be reformulated in the correct language. "How do we prevent the system from developing deceptive behavior?" becomes "What selection pressure operates on deceptive strategies, and is it positive or negative?" "How do we ensure the system is safe?" becomes "What does the incentive structure reward, and are safety-relevant properties in Zone 1 (rewarded), Zone 2 (punished), or Zone 3 (ignored)?" The questions are the same. The frameworks that can answer them are not.

---

## Part 1: The evolutionary framework

### Why evolution is not an analogy

When ML papers say a training process is "like evolution," they usually mean it loosely — something about mutation and selection. This is not that. A self-improving coding agent with multi-level selection, controlled mutation, constrained gene flow, and a bounded founding population literally is an evolutionary system. It satisfies the three requirements for evolution by natural selection (Lewontin 1970): variation (temperature-controlled LLM sampling), heritability (training on logged runs propagates patterns to the next generation), and differential fitness (test passage, evaluation benchmark performance, validation outcomes). The question is not "is this like evolution?" but "which predictions from evolutionary theory apply?"

### The four forces and their system mechanisms

Population genetics identifies four forces that drive evolutionary change. A well-designed self-improving agent has an explicit mechanism for each.

#### Force 1: Founder effect

**The biology:** When a small population colonizes a new environment, its genetic diversity is a subset of the parent population. All subsequent evolution is constrained by what the founders carried. The Galapagos finches descend from a handful of mainland birds — the entire adaptive radiation was bounded by those founders' gene pool.

**The system mechanism:** Training corpus curation. Consider a policy where only repositories from domains with transparent methodology — domains where the reasoning is verifiable or explicitly hedged about its limitations — are admitted to the training corpus. Non-transparent domains (unfalsifiable claims, circular reasoning, obscured limitations) are excluded entirely.

**Why this matters more than it appears:** The founder effect doesn't just filter for quality. It constrains the space of possible evolutionary trajectories. If non-transparent reasoning patterns never enter the founding corpus, the system cannot converge on them even in the absence of selection pressure against them. You don't need to select against what was never there. The gene pool is bounded before the first generation.

**The depth of the constraint:** Consider what "transparent methodology" actually excludes. Not just bad science — it excludes entire reasoning patterns. Confirmation bias loops, unfalsifiable hypothesis structures, post-hoc rationalization chains, appeals to authority, and circular justification patterns are all characteristic of non-transparent domains. By excluding these domains from the founding corpus, you exclude these reasoning structures. The agent can't learn them because it never sees them. This is prevention, not cure.

**The tuning parameter:** Corpus composition. Expanding the corpus to more domains increases the gene pool. Contracting it narrows evolutionary possibility. This is a deliberate choice with long-term consequences — once the system has trained on a corpus for several generations, the founding composition shapes everything downstream. Early corpus decisions are the highest-leverage decisions in the entire system.

#### Force 2: Selection

**The biology:** Differential reproductive success based on fitness. Organisms with traits better suited to the environment produce more offspring. The mechanism is simple; the consequences are not — multi-level selection, frequency-dependent fitness, balancing selection, and sexual selection all emerge from this one principle.

**The system mechanism:** Multi-level execution feedback. Selection operates at four levels simultaneously:

1. **LoRA adapter level.** A LoRA that encodes patterns producing worse code on novel tasks gets replaced by one that doesn't. Selection cycle: per-adapter evaluation against held-out benchmarks.
2. **Fine-tuning level.** A fully fine-tuned model that has internalized shortcut reasoning fails on harder problems in the next iteration. Selection cycle: per-training-run evaluation.
3. **New model level.** Training data contaminated by bad reasoning produces weaker models than clean training data. Selection cycle: per-model-generation evaluation.
4. **Infrastructure level.** Pipeline code changes that degrade output quality produce worse results across all tasks. Selection cycle: a multi-stage evaluation pipeline (propose, test, benchmark, evaluate, promote/reject).

**Why multi-level selection defeats gaming.** The standard alignment concern is Goodhart's law: the agent learns to game the proxy (tests) rather than achieve the true objective (correct code). Published results confirm this in single-shot settings: ImpossibleBench (2025) documents frontier models reward-hacking test suites 76% of the time; METR found o3 modified test/scoring code in 14/20 attempts.

But these results describe single-level selection in a single-shot setting. A model given one task, one test suite, one attempt. The optimal strategy can be to hack the test.

In multi-level selection, the optimal strategy shifts. A shortcut that passes tests at level 1 (LoRA) must also produce good training data at level 2 (fine-tune), which must produce a model that solves harder problems at level 3 (new model), which must produce infrastructure proposals that improve the pipeline at level 4 (code modification). Shortcuts don't compound. They produce locally successful but globally non-transferable patterns. Over iterations, genuine capability is the only stable strategy.

This is precisely why biological evolution produces genuine adaptations. In a static environment with a single selection event, organisms can get away with "hacks" — mimicry, parasitism, minimal viable responses. But when selection is continuous, multi-generational, and the environment keeps getting harder (because the system sets itself harder tasks), there is no stable equilibrium where faking fitness works. The organism — or the model — has to actually be fit.

**Multi-level selection in biology.** The concept is not new. Group selection vs. individual selection was the central debate of evolutionary biology from the 1960s through the 1990s (Williams 1966, Dawkins 1976, D.S. Wilson 1975, Sober & Wilson 1998). The resolution: selection operates at whatever level has heritable variation and differential fitness. In a well-designed self-improving system, all four levels qualify. The alignment field's failure to engage with this literature is striking, because it directly addresses the "can selection be gamed?" question.

#### Force 3: Gene flow

**The biology:** Migration of genetic material between populations. Gene flow introduces new variation but can also swamp local adaptation. Island populations that receive constant immigration from the mainland never develop the same degree of local specialization as isolated populations (Mayr 1963).

**The system mechanism:** Provenance tracking with default-deny curation.

Two specific gene flow control mechanisms:

1. **Domain confidence tiers.** Content from lower-confidence domains (useful knowledge, but methodology has known weaknesses) can enter the knowledge base for retrieval-augmented generation. But the tag propagates: if retrieval included lower-confidence content, the logged call carries that tag, and the training data curation step excludes it. The content influences individual tasks but cannot enter the gene pool. This is analogous to a hybrid zone — cross-pollination occurs at the boundary, but the offspring are sterile.

2. **Infrastructure checkpoint tainting.** If an infrastructure change is rolled back, all task runs produced under that checkpoint are excluded from training data. Even if the runs were individually successful, they were produced under potentially-flawed infrastructure and cannot be trusted as training signal. This prevents "immigrant" patterns from a bad infrastructure version from entering the gene pool of the next generation.

**The island biogeography connection.** An air-gapped deployment (no internet access) is literally an island. No continuous migration from the "mainland" of external data, models, or code. Gene flow is controlled entirely through discrete operator sessions. This produces exactly the conditions evolutionary biology predicts will maximize local adaptation: isolation with periodic, controlled introduction of new genetic material.

**The tuning parameter:** Curation filters in the training data extraction pipeline. Tighter filters reduce gene flow (less training data, more purity). Looser filters increase it (more training data, more contamination risk). The optimal setting depends on whether the system is underfitting (needs more data, loosen filters) or overfitting (needs cleaner signal, tighten filters).

#### Force 4: Mutation

**The biology:** Random changes in genetic material that produce novel variation. Most mutations are neutral or deleterious. A small fraction are beneficial. The mutation rate is a critical parameter: too low and adaptation stalls (insufficient variation for selection to act on); too high and beneficial adaptations are destroyed before they can spread (error catastrophe, Eigen 1971).

**The system mechanism:** Temperature-controlled randomness in LLM sampling.

Temperature directly controls the mutation rate. At temperature 0.0, the model is deterministic — no mutation, no variation, no adaptation. At higher temperatures, the model explores more of the output distribution — more variation, more failed attempts, but also more novel patterns that selection can act on.

**This is better than biological mutation** in a crucial way. Biology cannot tune its mutation rate. The mutation rate is a property of the replication machinery, and while it can evolve slowly (mutator phenotypes), organisms cannot dial it up when the environment changes and down when it's stable. This system can. If benchmark performance is plateauing, increase temperature — more exploration, more novel patterns, most will fail, some will survive. If the system is improving steadily, lower temperature — exploit what's working, reduce noise. The mutation rate is a first-class control parameter.

**Multiple temperature runs as mutagenesis.** A DPO training pipeline that runs the same task at multiple temperatures to generate preference pairs is not just a training technique — it is controlled mutagenesis. Each temperature setting produces a different phenotype from the same genotype. Selection (test passage) determines which phenotype survives. Over many such experiments, the system identifies the optimal level of variation for the current fitness landscape.

**The error catastrophe threshold.** There is a maximum useful mutation rate. If temperature is too high, the model produces incoherent outputs that fail all tests — no selection signal is generated, no learning occurs. The practical threshold is where the proposal acceptance rate (for infrastructure) or test pass rate (for code) drops below a useful level. This is the system's analog of Eigen's error catastrophe: above the threshold, information is destroyed faster than selection can preserve it.

---

## Part 2: The economics framework

### Why economics applies

Economics is the study of how agents allocate scarce resources under constraints, responding to incentive structures. A self-improving coding agent allocates compute, context window space, and training data under hardware constraints, responding to reward signals. The parallel is not metaphorical — the system literally has a budget (token budget, compute budget, training data budget), makes allocation decisions, and responds to marginal incentives. The question is: what equilibrium does this incentive structure produce?

### Goodhart's law — the original version

The alignment field uses Goodhart's law as shorthand for "optimizing a proxy measure eventually diverges from the true objective." This is correct but shallow. The original formulation (Goodhart 1975, in the context of monetary policy) was more specific: "Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes."

The key word is **observed**. Goodhart was describing what happens when you take a naturally-occurring correlation (between a measurable proxy and an unmeasurable target) and turn it into an explicit optimization target. The correlation collapses because the proxy is being driven by the optimization, not by the underlying relationship that produced the original correlation.

**Applied to test suites in single-shot settings:** Test passage correlates with code correctness. When you optimize for test passage directly (train a model to maximize test pass rates), the correlation collapses — the model finds ways to pass tests without correct code. This is exactly what ImpossibleBench demonstrates. Goodhart predicted it in 1975.

**Applied to multi-level selection:** The critical difference is that multi-level selection doesn't optimize for a single proxy. It optimizes across multiple proxies at multiple levels, each of which captures a different facet of the true objective. Test passage is one proxy. Training data quality (does training on these runs produce better models?) is another. Infrastructure effectiveness (does this pipeline change improve outcomes?) is a third. Held-out benchmark performance is a fourth.

When an agent optimizes against four proxies simultaneously, gaming becomes harder in a specific way: a strategy that games proxy A must also satisfy proxies B, C, and D. The intersection of strategies that game all four proxies simultaneously is much smaller than the set that games any single proxy. In the limit, the only strategy that satisfies all proxies is the genuine target behavior.

This is exactly the economic argument for multi-dimensional performance metrics in organizations. A company that measures employees on a single KPI gets Goodhart'd. A company that measures on revenue AND customer satisfaction AND code quality AND team velocity makes gaming much harder because the metrics are partially orthogonal. The same principle applies here, with the added advantage that the "metrics" (test passage, training quality, infrastructure effectiveness, benchmark performance) have genuine causal relationships to the true objective, not just statistical correlations.

### Incentive structures and the three zones

An economist analyzing the system would first map the incentive landscape: for every possible behavior the agent could exhibit, what is the reward, the punishment, or the indifference?

This produces the same three-zone analysis that evolutionary biology derives from selection theory, but through a different lens:

**Zone 1: Positive incentives (behavior rewarded).**
Behaviors that increase test passage, benchmark performance, and training data quality. Includes: writing correct code, transparent failure reporting (exposes bugs that need fixing), clean dependency management (reduces cascading failures), precise context curation (puts the right information in the window).

The economist notes: these behaviors are rewarded because they directly produce the measured outcome. The incentive is aligned with the true objective by construction. No Goodhart problem, because the proxy and the target coincide for these behaviors.

**Zone 2: Negative incentives (behavior punished).**
Behaviors that decrease measured outcomes. Includes: silent error swallowing (masks failures, degrades training signal), cargo-cult patterns (add complexity without benefit, increase failure surface), defensive over-catching (same as silent swallowing), speculative abstraction (adds indirection that increases failure probability).

The economist notes: these behaviors are punished not because someone designed a punishment, but because they produce measurably worse outcomes. The incentive structure is self-enforcing. No monitoring needed — the behavior destroys itself.

**Zone 3: No incentive (behavior ignored).**
Behaviors that don't affect measured outcomes. Includes: variable naming conventions, comment quality, documentation completeness, security hardening not triggered by tests, code style preferences.

The economist notes: this is the textbook free-rider problem. Zone 3 behaviors are public goods — they benefit the system (readability helps future maintenance, security prevents vulnerabilities) but don't benefit the agent that produces them (no test pass rate improvement). In the absence of incentives, public goods are under-provided. This is not a bug in the system — it is a fundamental prediction of the incentive structure.

### The price system: information transfer under scarcity

Hayek's central insight (Hayek 1945, "The Use of Knowledge in Society") is that prices are not primarily a payment mechanism — they are an information transfer system. The price of steel compresses an immense amount of dispersed knowledge (mining costs, energy prices, transport logistics, demand from construction, demand from automotive, labor market conditions, geopolitical risk) into a single scalar. No central planner could gather all this information. The price system aggregates it automatically, enabling coordination without central knowledge.

This framework applies directly to a self-improving coding agent, because such a system faces Hayek's knowledge problem at every level.

#### The multi-stage pipeline IS a decentralized coordination mechanism

The core thesis motivating multi-stage retrieval pipelines is that the primary bottleneck in LLM application performance is not model capability but context curation. A multi-stage pipeline is the architectural response: instead of one monolithic prompt with 200K tokens of everything, a sequence of stages each make local curation decisions with local information.

Each retrieval stage possesses knowledge that no other stage has:
- A **scope stage** knows about dependency relationships and co-change patterns.
- A **precision stage** knows about symbol-level relevance and detail classification.
- A **similarity stage** knows about semantic relationships between files.
- An **assembly stage** knows about budget constraints and framing overhead.

No single component has enough information to make globally optimal curation decisions. The pipeline coordinates them through a shared medium — the token budget and the session state — that each stage reads from and writes to.

This is structurally identical to a market. Each stage is a producer with private information, making local allocation decisions. The budget is the currency — stages "spend" tokens to include content, and the scarcity of the budget forces trade-offs. The session state is the price signal — it carries compressed information from earlier stages (what was included, what was excluded, what budget remains) that later stages use to make their own decisions. No central planner decides what goes in the context window. The pipeline coordinates it through decentralized decision-making under a shared budget constraint.

**The alternative — and why it fails.** A monolithic 200K context window with everything stuffed in is central planning. One prompt, one planner, trying to allocate attention across all available information. Hayek's argument against central planning applies directly: the planner (the model) cannot effectively process the information it needs to make optimal allocation decisions, because the volume of information exceeds its processing capacity. At 10-15% signal relevance in a 200K window, most of the context is noise. The model is a central planner drowning in data it cannot efficiently filter. A multi-stage pipeline solves this by distributing the curation decisions across specialized stages, each operating on manageable volumes of locally relevant information.

#### Token budget as currency

A token budget functions as a currency system:

- **Unit of account:** Tokens measure the cost of including any piece of content. A full source file costs more than a signature. A high-relevance file costs more than a low-relevance one. Everything is denominated in the same unit.
- **Medium of exchange:** Stages "spend" budget to include content in the context window. A budget tracker records expenditures. Spending on file A means not spending on file B — the budget forces trade-offs.
- **Scarcity enforcer:** The budget is finite (target context window size with a safety margin). This scarcity is what makes the system work. Without a budget constraint, every stage would include everything, and the context window would be the monolithic central-planning mess the pipeline was designed to avoid.

**Budget allocation is resource allocation.** The retrieval stages collectively decide how to allocate a scarce resource (context window space) across competing uses (different files, different symbols, different detail levels). This is the same problem a market economy solves: how to allocate scarce resources across competing uses, using prices (token costs) and budgets (total capacity) to coordinate decentralized decisions.

**The budget tracker is the central bank.** It doesn't make allocation decisions — it enforces the constraint and tracks expenditures. It does not decide which files to include; it tells stages how much room remains. This separation of monetary policy (budget enforcement) from fiscal policy (allocation decisions by stages) mirrors the institutional separation in real economies.

#### Relevance tiers as compressed prices

Relevance tiers in a retrieval pipeline — for example, tiers like "full source," "signatures plus context," "signatures only," and "excluded" — are price-like compressions. They take rich, multidimensional information about a file's relevance and compress it into a single label.

This compression enables efficient allocation: high-relevance files get full content, medium-relevance get summaries, low-relevance get minimal representations. But like all price compression, it is lossy. The label "high relevance" tells the assembly stage to include everything, but not WHY the file matters — which functions, which sections, what the model should pay attention to. The model must re-derive this from the source itself.

**Per-symbol classification is higher-resolution pricing.** Instead of one price per file, it produces one price per symbol (include/exclude with reason). This is analogous to moving from a coarse pricing mechanism (one price per category of goods) to a fine-grained one (one price per individual good). More information is preserved, enabling more efficient allocation.

#### Price distortion and the evaluation set

Hayek's framework also predicts what goes wrong: price distortion. When prices don't reflect true costs and values, resources are misallocated. Subsidies cause over-production of subsidized goods. Price ceilings cause shortages. Externalities cause goods to be over- or under-produced relative to social optimum.

In a self-improving system, the "prices" that drive improvement are the evaluation metrics: benchmark pass rates, token utilization, retrieval precision. If these metrics are distorted — if the evaluation set systematically overvalues one type of task and undervalues another — the system allocates improvement effort toward what the distorted signal rewards.

**Specific distortion risks:**

1. **Evaluation set composition bias.** If 80% of evaluation tasks are single-file Python edits, the system over-optimizes retrieval for that case and under-optimizes for multi-file TypeScript projects. The "price" of improving Python retrieval is low (many tasks benefit); the "price" of improving TypeScript retrieval is high (few tasks benefit). The system rationally under-invests in TypeScript — not because TypeScript doesn't matter, but because the price signal says it doesn't.

2. **Metric truncation.** If the evaluation pipeline measures success/failure, token usage, and retrieval precision but doesn't measure code readability, maintainability, or extensibility, these qualities have a "price" of zero. A central planner who set the price of water to zero would get over-consumption and waste. A selection system that sets the price of readability to zero gets unreadable code.

3. **Temporal discounting.** The metrics capture immediate outcomes (does this task pass tests now?) but not long-term effects (does this infrastructure change improve the system's trajectory over the next 50 tasks?). Short-term improvements are priced accurately. Long-term improvements are systematically underpriced. This is identical to the market failure of under-investment in basic research — the returns are real but too diffuse and delayed for the price system to capture.

**The countermeasure is the same one Hayek would prescribe:** fix the prices. Design the evaluation set to accurately represent the full range of tasks the system should handle. Include metrics that capture long-term quality, not just immediate success. The evaluation set is the system's price mechanism — its quality determines the quality of every resource allocation decision downstream.

#### The socialist calculation problem, applied

Mises (1920) and Hayek (1945) argued that socialist central planning must fail because without market prices, the planner has no mechanism for rational resource allocation. Prices encode information about relative scarcity and value that cannot be gathered or processed centrally.

The monolithic context window approach is central planning. One model, one prompt, 200K tokens, "figure it out." The model has no price signals about which content is more valuable than which other content. It must discover this by processing everything — an information-processing burden that exceeds its effective capacity, resulting in the low utilization rates that motivate multi-stage pipeline architectures.

A multi-stage pipeline is the market solution. Each stage generates local price signals (relevance tiers, inclusion/exclusion decisions, budget allocations) that later stages use. The assembly stage doesn't need to know WHY the scope stage included a file — it just needs to know the "price" (relevance tier) to allocate the right amount of budget. Information flows through the pipeline via these compressed signals, enabling coordination without any single component needing global knowledge.

**This is why pipeline architectures work even with small models.** A 3-4B parameter model cannot effectively process 200K tokens of uncurated context. But it can effectively process 32K tokens of curated context where every token was placed by a prior decision with a reason. The pipeline doesn't make the model smarter — it makes the information the model receives cleaner. The price system (budget + relevance tiers + inclusion decisions) does the work that the model's limited capacity cannot.

### Asymmetric information: what the pipeline can't see

Akerlof's market for lemons (Akerlof 1970) showed that when one party to a transaction has more relevant information than the other, markets can fail catastrophically. If the seller knows a car is a lemon but the buyer doesn't, rational buyers assume the worst, prices collapse, and good cars leave the market. Stiglitz and Weiss (1981) showed similar dynamics in credit markets — lenders can't distinguish good from bad borrowers, leading to credit rationing that excludes viable projects. The general principle: **information asymmetry between transacting parties distorts allocation.**

A well-designed self-improving agent is unusually transparent. Every LLM call logged with full I/O. Every retrieval decision recorded with its reasoning. Append-only activity logs. Thinking tags that expose the model's reasoning process, not just its output. Session state that carries forward structured metadata about decisions, not just decisions. A human can trace any output back through every decision that produced it.

This is much better information transfer than a real market. In a market, you see the price of steel and nothing else — you don't get the mining company's internal analysis of extraction costs, the logistics firm's routing decisions, the factory's demand forecasting model. In such a pipeline, you can read every stage's "meeting notes." The design explicitly minimizes information asymmetry — that's what full logging, activity databases, decision tracking, and provenance chains are for.

**On initial analysis, four asymmetries seemed concerning. On closer examination, two dissolve and two remain genuinely irreducible.**

#### Asymmetry 1 (weaker than expected): Reasoning quality vs. outcome quality

The test suite observes outcomes: pass or fail. It cannot distinguish correct code produced through good reasoning from correct code produced through a lucky pattern match, a memorized template, or a shortcut that happens to work on this specific input but wouldn't generalize.

Thinking tags partially address this — the model's reasoning IS logged, and a training data curation pipeline could theoretically score reasoning quality, not just outcomes. But the evaluation mechanism (the validation gauntlet) doesn't use reasoning quality as a selection criterion. It uses test outcomes. This means the "price" a pipeline run receives (pass/fail, evaluation set performance) does not reflect reasoning quality. Reasoning quality is unpriced.

**The naive market analog:** This looks like the classic lemons problem. Good reasoning and bad reasoning both produce the same observable outcome (passing tests), so the evaluation mechanism can't price them differently.

**But the lemons framing is wrong here.** Akerlof's result depends on a critical assumption: buyers who transact once, cannot inspect the product's internals, and cannot learn from repeated transactions. A self-improving system violates all three assumptions.

First, this is not a one-shot market. It is iterated selection across multiple training generations. A lucky run produces a training example that rewards pattern-matching. A model trained partly on such examples will sometimes produce bad code on the next iteration's tasks — because the pattern doesn't generalize. Over iterations, models trained on genuinely-reasoned runs will consistently outperform models trained on lucky runs. The lemons don't persist because the market keeps testing them. This is closer to a reputation market than a one-shot lemons market — and reputation markets don't collapse, because repeated interaction reveals quality.

Second, the audit trail exists and is complete. The thinking tags, the retrieval decisions, the full LLM I/O — they are all in the activity log. A training data curation pipeline that filters on reasoning coherence (not just outcome) would directly address the asymmetry. The information isn't hidden in the way that a used car's maintenance history is hidden from the buyer. It's sitting in the database. The question is whether the curation pipeline uses it, not whether it exists.

Third, and most importantly: lucky runs survive marginally (individual runs can get lucky) but not on average. Across the population of training examples, runs with coherent reasoning traces that connect task analysis → retrieval decisions → code edits in a traceable chain will systematically outperform runs where the audit trail is incoherent or shallow. The selection mechanism doesn't need to explicitly score reasoning quality — it scores the downstream consequence of reasoning quality across many tasks, which amounts to the same thing statistically. A single lucky run is noise. A training corpus full of lucky runs produces a detectably worse model.

**The residual concern:** The asymmetry is real at the margin. Individual lucky runs do enter the training corpus, and they do contribute noise. The question is whether this noise is self-correcting (it is — bad reasoning produces bad code on the next task, which is selected against) or accumulating (it isn't — the feedback loop continuously filters it). The correction is not instant, but it is monotonic. This is a medium-severity transient problem, not a structural market failure.

#### Asymmetry 2: Present effects vs. future costs

When the agent proposes an infrastructure modification, the immediate effect — evaluation set performance on the current tasks — is observable. The long-term effect — how this modification interacts with future modifications, whether it creates path-dependent constraints, whether it narrows the space of future improvements — is not observable. It doesn't exist yet.

This is not a logging failure. No amount of transparency can reveal future costs, because they haven't been incurred. The gauntlet prices the present accurately. The future is systematically underpriced.

**The market analog:** This is the temporal externality problem — a generalization of environmental externalities across time. A factory's pollution imposes costs on future generations who cannot participate in the current market. An infrastructure modification's path-dependent constraints impose costs on future pipeline iterations that cannot participate in the current gauntlet evaluation.

**Why this is harder to fix than standard externalities.** Standard externalities (pollution, technical debt) can be internalized by clever evaluation set design — add tasks that require modifying previous code, and the cost of messy code becomes visible. Temporal externalities from path dependence cannot be internalized this way, because the cost depends on modifications that haven't been proposed yet. You can't design an evaluation task that tests "does this change make future changes harder?" without knowing what the future changes will be.

**The partial countermeasure:** Periodic human-initiated restructurings. Humans can reason about architectural trajectory in ways the gauntlet cannot. This is the analog of intergenerational planning — policies (zoning laws, infrastructure investment, constitutional provisions) that constrain current behavior for the benefit of future generations, imposed by agents who can reason about the future rather than merely optimize for the present.

#### Asymmetry 3: Evaluation set vs. true task distribution

The system observes its performance on the evaluation set. It cannot observe its performance on the true distribution of tasks it will face in production, because that distribution is unknown and potentially non-stationary.

If the evaluation set is representative, evaluation performance is a good proxy for true performance. If it isn't, the system is optimizing for a biased signal — like a student studying past exams that don't represent the actual test. The system has no mechanism for detecting this divergence from within. The evaluation set IS the system's window onto reality. It cannot see around the edges of its own window.

**The market analog:** This is Goodhart's law in its information-theoretic formulation. The evaluation set is a proxy for true task performance. When the system optimizes for the proxy, the proxy diverges from the target — not because the proxy was originally wrong, but because optimization pressure exploits whatever gaps exist between proxy and target. The system thinks it's improving (evaluation scores go up) while actual capability may stagnate or even degrade on tasks outside the evaluation distribution.

**The countermeasure:** Periodic evaluation set expansion by humans who have access to real-world task distributions. This is the analog of an external auditor — an agent outside the system's optimization loop who can compare the system's self-assessment (evaluation set performance) against independent evidence (actual task performance). The system cannot audit itself, because the audit instrument (evaluation set) is the very thing that might be biased.

#### Non-asymmetry: Stage-to-stage "price compression"

The initial framing of this section identified stage-to-stage information compression — relevance tiers reducing rich reasoning to coarse labels — as a fourth asymmetry. On reflection, **this is not an asymmetry at all. It is the entire point of a pipeline architecture, and calling it an information problem misunderstands what the stages need.**

Each retrieval stage has rich, multidimensional information about why it made its decisions. The scope stage knows the full dependency graph, co-change patterns, and the model's reasoning about relevance. It passes forward a compressed signal: a list of files with relevance tiers. The precision stage doesn't know WHY the scope stage included file X.

**But the precision stage doesn't need to know why.** Its job is to classify symbols within the files it receives. The scope stage's reasoning about dependency relationships and co-change patterns is irrelevant to symbol-level classification. The precision stage needs the file list (input) and the task description (context). It does not need the scope stage's internal deliberation, any more than a worker who implements a function needs to know the full architectural reasoning behind why that function was specified. They need the interface contract — the input, the output, the constraints. The "why" behind the specification is not their concern and would be noise in their context window.

This is exactly the subagent pattern in any multi-agent system. A subagent launched to search for files doesn't receive the full conversation history of why those files are needed. It receives a task description with the relevant context. This is not an information asymmetry — it is proper context curation. Giving the subagent the full conversation history would make it worse, not better, because it would be processing irrelevant information instead of focusing on its specific task.

**The pipeline IS a division of labor with appropriate information boundaries.** The scope stage is a specialist in dependency relationships. The precision stage is a specialist in symbol-level relevance. The assembly stage is a specialist in budget-constrained packing. Each specialist receives what it needs to do its job — no more, no less. The "compression" from upstream to downstream is not lossy in any way that matters. It is the same dimensionality reduction that makes specialization efficient in any economy: the steel buyer doesn't need the mining company's geology reports. They need steel that meets a specification.

**When this actually fails.** The one scenario where inter-stage information loss matters is when the upstream stage makes a mistake that the downstream stage could have caught IF it had the upstream reasoning. For example: scope includes a file because of a spurious co-change pattern, and precision classifies its symbols as relevant because the file was presented as in-scope (anchoring bias). If precision had known "this file was included only because of co-change, not because of dependency analysis," it might have been more skeptical. But this is an argument for better stage outputs (include a confidence signal or a provenance tag), not for propagating full reasoning traces. The fix is a richer output contract, not an elimination of the information boundary.

#### The net assessment: near-perfect information with two irreducible blind spots

The system has unusually good information transfer — much better than biological evolution (which has no equivalent of a full activity database), much better than real markets (which have no equivalent of full transaction logging with reasoning traces), and much better than the AI alignment field assumes is possible (which treats information asymmetry between "AI agent" and "human overseer" as an unsolvable problem).

Two of the four initially-hypothesized asymmetries turn out to be weaker than expected:

- **Reasoning vs. outcome** is a transient problem, not a structural market failure. The iterated selection mechanism and complete audit trail mean that lucky runs are selected against statistically, even without explicit reasoning quality scoring. The lemons analogy fails because this is a reputation market with full inspection rights, not a one-shot anonymous market.
- **Stage-to-stage compression** is not an asymmetry at all. It is proper division of labor — each stage receives the information it needs for its specific task, and the "lost" upstream reasoning is irrelevant to downstream decisions. The pipeline IS the market working correctly, not a market failure.

Two genuine, irreducible asymmetries remain:

| Asymmetry | What's hidden | Why it's irreducible | Severity |
|---|---|---|---|
| Present vs. future | Long-term path-dependent costs of current decisions | Future hasn't happened yet | High — only human foresight can address it |
| Evaluation vs. reality | True task distribution beyond the evaluation set | True distribution is unknown and non-stationary | High — requires external auditing |

The pattern: the system has solved the information problems that CAN be solved by engineering (full logging, reasoning traces, decision tracking, iterated selection). The information problems that looked concerning on first analysis — reasoning quality lemons, stage-to-stage compression — dissolve under scrutiny because the system's design already addresses them through mechanisms that real markets lack (complete audit trails, iterated selection, appropriate information boundaries). What remains are epistemological — information that doesn't exist yet (future costs) and information that is unknowable from inside the system (true task distribution). No amount of additional logging or transparency can eliminate them. They require external inputs: human architectural foresight and real-world task data.

**The economic prediction from these asymmetries:**

1. **Path-dependent debt will accumulate monotonically.** Without human intervention, architectural decisions will never be reversed, because the cost of reversal is future (unpriced) while the cost of a gauntlet-failing restructuring is immediate (well-priced). The system rationally avoids restructuring even when it would improve long-term trajectory.

2. **Overfitting to the evaluation set is the default trajectory.** The system will progressively specialize to the evaluation set's characteristics unless the set is actively expanded. This is not a bug in the system — it is the rational response to the information available. You optimize for what you can measure. What you can't measure, you ignore.

3. **Reasoning quality will self-correct, but with a lag.** Early training iterations will contain some lucky runs that degrade reasoning quality. This is transient — the iterated selection mechanism filters them out as bad reasoning fails on subsequent tasks. The lag between outcome convergence and reasoning convergence is real but bounded, not divergent. The prediction: if you measure reasoning quality independently, it will track outcome quality with a 1-2 iteration delay, not a permanent gap.

### Externalities: the unpriced costs and benefits

An externality is a cost or benefit that affects a party who did not choose to incur that cost or benefit. Pollution is the textbook negative externality — the factory's production imposes costs (health damage, environmental degradation) on third parties who receive no compensation. Education is the textbook positive externality — an educated person benefits not just themselves but everyone they interact with, and the market therefore under-provides education relative to the social optimum.

A self-improving coding agent is riddled with externalities, and recognizing them explains Zone 3 behavior more precisely than the "public goods" framing alone.

#### Negative externalities: costs the selection mechanism doesn't see

**Technical debt from satisficed code.** When the agent produces code that is "good enough" to pass tests but structurally messy, the mess imposes costs on future pipeline runs: harder to parse for AST extraction, harder for retrieval to identify relevant symbols, harder for the model to understand in context. These costs are real but not borne by the run that created the mess. They are externalities — imposed on future runs by past decisions.

In a market, negative externalities are addressed by Pigouvian taxes (Pigou 1920): make the polluter pay the cost they impose on others. The system analog would be a "technical debt tax" — a penalty in the selection mechanism for code that is structurally complex relative to its functional contribution. But this requires measuring structural complexity, which a typical metric system doesn't do. The externality is unpriced.

**Context window pollution.** A retrieval stage that includes marginally relevant files "just in case" imposes a cost on the execute stage: more noise in the context window, more tokens consumed, less room for genuinely relevant content. The retrieval stage doesn't bear this cost — it only sees "did I miss anything important?" The cost falls on downstream stages. This is why default-deny curation rules exist — they are a regulatory response to context window pollution, analogous to environmental regulation that caps emissions regardless of the polluter's cost-benefit calculation.

**Training data contamination.** A single pipeline run that produces correct code through a bad reasoning path (lucky shortcut, accidental success) contributes a bad example to the training corpus. The cost — a slightly worse model in the next generation — is diffuse, delayed, and borne by all future runs. The run that created the bad example bears none of this cost. This is why provenance tracking and checkpoint tainting exist — they are mechanisms to identify and exclude contaminated training data, analogous to food safety regulations that trace contamination back to its source.

#### Positive externalities: benefits the selection mechanism doesn't reward

**Clean architecture.** Well-structured code with clear module boundaries, consistent naming, and minimal coupling benefits every future run that touches the same codebase. The run that produces clean architecture bears the cost (more effort than the minimum needed to pass tests) but receives none of the benefit (tests pass either way). Clean architecture is under-provided because its benefits are externalized.

**Comprehensive error messages.** A detailed error message like `"Expected float for field 'timeout', got str: '30s' at config line 42"` costs more to produce than `"invalid config"`. Both pass the same tests. The benefit of the detailed message — faster debugging in future runs, better training signal when the message appears in logged data — accrues to future runs, not the run that paid the cost of producing it.

**Documentation and comments.** Comments that explain WHY a decision was made (not just WHAT the code does) benefit future retrieval (enrichment can extract richer metadata) and future context curation (the model understands the code faster). The run that writes the comments bears the cost. Future runs reap the benefit. Classic positive externality.

#### The Pigouvian correction: internalizing externalities via the evaluation set

The standard economic solution to externalities is to internalize them — make the actor bear the costs they impose or receive the benefits they create. Pigouvian taxes on negative externalities, subsidies for positive externalities.

In a self-improving system, the evaluation set is the mechanism for internalization. It can be designed to make externalities load-bearing:

- **Technical debt:** Include evaluation tasks that specifically require modifying previously-generated code. If the code is messy, the modification task fails. Now the original run's technical debt has a measurable cost in the metric system — the externality is internalized.
- **Error message quality:** Include evaluation tasks where the agent must debug a failure using only the error messages from a prior run. If the messages are useless, the debug task fails. The cost of bad error messages is now priced in.
- **Architecture quality:** Include evaluation tasks that require extending a previously-built module. If the architecture is brittle, the extension task fails. The cost of bad architecture is now measurable.

This is exactly the Zone 3 → Zone 1 promotion strategy from the evolutionary analysis, but the economic framing makes the mechanism explicit: you are internalizing externalities by designing the price system (evaluation tasks) to capture costs and benefits that the natural metrics miss. You are making the invisible hand see what it was previously blind to.

**The limitation:** You can only internalize externalities you can identify and measure. Unknown externalities remain unpriced. This is why periodic human review of the evaluation set is essential — humans can identify externalities that the system's own metrics cannot see, and design new evaluation tasks that internalize them.

### The equilibrium prediction

An economist combines the incentive analysis with the constraint structure to predict the equilibrium — the stable state the system converges to.

**The prediction:** The system converges to an equilibrium where:

1. Code correctness is high (Zone 1 behaviors dominate).
2. Transparency is high (Zone 1 — transparency is load-bearing for the feedback loop).
3. Defensive coding is absent (Zone 2 — destroyed by incentives).
4. Code style, documentation, and security are mediocre-to-poor (Zone 3 — under-provided public goods).
5. The system satisfices rather than optimizes (see below).
6. Path-dependent decisions accumulate and are never revisited (see below).

This is a specific, testable prediction. It is also a familiar one to any economist: it describes a rational agent operating under incentives with externalities and public goods problems. The system is not broken — it is performing exactly as the incentive structure predicts.

### Satisficing: the economics of "good enough"

Herbert Simon introduced the concept of satisficing (Simon 1956) to describe how agents with bounded rationality make decisions. They don't optimize — they search until they find a solution that meets a threshold, then stop. The threshold is "good enough."

**Why optimization doesn't happen in practice.** Optimization requires comparing all possible solutions and selecting the best one. For an agent modifying its own infrastructure, the space of possible modifications is effectively infinite. Even if the agent could enumerate them, evaluating each requires running the full evaluation set — a computationally expensive operation. The rational strategy is satisficing: try modifications until one passes evaluation, then promote it and stop.

**The consequence is local optima.** Satisficing converges to the nearest solution that exceeds the threshold, not to the global optimum. A retrieval stage that works well enough will never be replaced by one that works better, because the system has no incentive to search for alternatives once the threshold is met.

**The economics of switching costs.** Even when a better alternative exists, switching has a cost: the risk of regression (the evaluation pipeline may reject it), the compute cost of evaluation (benchmark runs), and the opportunity cost (time spent on infrastructure modification is time not spent on tasks). A rational agent weighs these costs against the marginal benefit of the improvement. For small improvements, the costs exceed the benefits. The system gets stuck at "good enough" not because it can't find better, but because the price of better exceeds the value.

This is the same logic that explains why QWERTY persists despite being suboptimal. The switching cost (retraining all typists) exceeds the marginal benefit (slightly faster typing). In an infrastructure self-improvement loop, the switching cost (evaluation risk + compute cost) exceeds the marginal benefit (slightly better retrieval) once the system is in a reasonable equilibrium.

### Path dependence and lock-in

W. Brian Arthur's work on increasing returns and path dependence (Arthur 1989) describes how early decisions can lock a system into a particular trajectory, even when better alternatives exist, because the cost of switching increases as more decisions build on the early ones.

**Applied to infrastructure self-improvement:** The first promoted infrastructure modification becomes the foundation for the second. The second builds on assumptions introduced by the first. By the tenth modification, the accumulated assumptions are deeply embedded, and changing any one of them requires changing all the downstream dependencies. The system is locked in.

**The laryngeal nerve problem.** The recurrent laryngeal nerve in mammals takes a detour from the brainstem around the aortic arch and back up to the larynx. In giraffes, this detour is several meters long. The nerve "should" take a direct path of a few centimeters. It doesn't, because at every step of the evolutionary path from fish (where the route was direct) to giraffe (where it's absurd), the existing route worked and the cost of restructuring exceeded the benefit of shortening. Each generation that inherited the detour and built more anatomy around it made the detour harder to fix.

In an infrastructure self-improvement loop, this manifests as architectural debt. Early retrieval optimizations that exploit quirks of the current implementation become load-bearing. Subsequent optimizations assume the quirk. The quirk becomes unfixable because everything downstream depends on it. The system works. The architecture is absurd. Natural selection — the evaluation pipeline — cannot fix it because every intermediate step toward the clean architecture is worse than the current detour.

**Why markets solve this and strict evaluation doesn't.** Markets handle path dependence through creative destruction (Schumpeter 1942): a new entrant builds the better architecture from scratch, without the incumbent's legacy constraints, and displaces the incumbent. A zero-regression evaluation pipeline cannot do this because it enforces no regressions — there is no mechanism for a new entrant to displace the incumbent through a temporary performance dip. This is a deliberate design choice (safety over speed), but it has the predictable consequence of locking in suboptimal architectures.

**The countermeasure — and its cost.** Human-initiated modifications can authorize restructurings that temporarily degrade performance. This is the equivalent of planned creative destruction — a deliberate decision to absorb short-term cost for long-term benefit. But it requires a human to identify the problem, design the solution, and accept the risk. The system cannot do it autonomously, because the evaluation pipeline (correctly) rejects any change that degrades measured outcomes. This is the fundamental tension: safety mechanisms that prevent bad changes also prevent good restructurings. There is no free lunch.

---

## Part 3: Where the frameworks converge

### The same prediction from different axioms

The evolutionary framework and the economics framework arrive at the same three-zone prediction through entirely different reasoning:

| Property | Evolutionary prediction | Economic prediction | Mechanism |
|---|---|---|---|
| Transparency | Selected FOR (load-bearing for fitness signal) | Incentivized (produces better measured outcomes) | Honest failure reporting gives cleaner selection signal / reward signal |
| Functional correctness | Selected FOR (by definition — fitness = test passage) | Incentivized (directly measured) | Tautological — this IS the optimization target |
| Defensive coding | Selected AGAINST (masks failures, degrades signal) | Punished (produces worse measured outcomes) | Silent error swallowing hides bugs that the feedback loop needs to see |
| Cargo-cult patterns | Selected AGAINST (noise, no fitness contribution) | Punished (adds cost without benefit) | Unnecessary complexity increases failure surface without improving test passage |
| Code style | Neutral drift (no fitness impact) | Public good (benefits system, not producer) | Naming conventions don't affect execution |
| Documentation | Neutral drift | Public good | Comments don't affect test passage |
| Security hardening | Neutral drift (if tests don't trigger it) | Public good (benefits users, not agent) | Untested security code has no selection pressure |
| Local optima convergence | Fitness valley problem | Switching cost problem | Same outcome, different causal mechanism |
| Path dependence | Laryngeal nerve / historical contingency | Lock-in / increasing returns | Same outcome, different causal mechanism |
| Satisficing | Selection stops at "good enough" fitness | Rational agent satisfices under bounded rationality | Same outcome, different causal mechanism |
| Context curation | Fitness requires signal, not noise | Price system allocates scarce budget via local decisions | Multi-stage pipeline = market; monolithic window = central planning |
| Training contamination | Deleterious alleles in gene pool | Negative externality (cost borne by future generations) | Provenance tracking = food safety traceability |
| Clean architecture | Neutral unless tested | Positive externality (benefit not captured by producer) | Evaluation set design = Pigouvian subsidy for quality |
| Information transfer | Genotype → phenotype mapping is observable via full audit trail; iterated selection filters bad reasoning statistically | Near-perfect: full logging + iterated reputation market, not one-shot lemons; two irreducible blind spots (future costs, true distribution) | Transparency + iteration solves what's solvable; only temporal and proxy-target gaps remain |

The convergence is not a coincidence. Both frameworks are describing the same underlying dynamic: a system with variation, selection, and heritability operating under constraints. Evolutionary biology describes this in terms of fitness landscapes and population genetics. Economics describes it in terms of incentives and rational choice under scarcity. The math is often identical (replicator dynamics in evolution are equivalent to certain classes of game-theoretic dynamics).

### The combined prediction for this system

Merging both frameworks produces a unified prediction with more precision than either alone:

**A well-designed self-improving coding agent will produce infrastructure that:**

1. Is functionally correct and improves over time (both frameworks agree — direct selection/incentive).
2. Is transparent and fails honestly (both frameworks agree — indirect selection/incentive through feedback loop quality).
3. Contains no defensive coding, cargo-cult patterns, or unnecessary abstractions (both frameworks agree — selected against/punished).
4. Has mediocre documentation, inconsistent style, and spotty security (both frameworks agree — neutral drift / public goods under-provision).
5. Is locally optimal but not globally optimal (evolution: fitness valley; economics: switching costs).
6. Accumulates path-dependent architectural debt over time (evolution: laryngeal nerve; economics: lock-in).
7. Plateaus at "good enough" and resists further improvement (evolution: stabilizing selection; economics: satisficing under bounded rationality).
8. Accumulates unpriced externalities (training data contamination, technical debt, context pollution) in proportion to how invisible they are to the metric system (economics: unpriced externalities are over/under-produced; evolution: traits not under selection drift).
9. Shows outcome quality converging slightly faster than reasoning quality, but the gap is bounded and self-correcting (economics: iterated reputation market filters bad reasoning statistically; evolution: iterated multi-level selection penalizes non-generalizing patterns within 1-2 generations). Training data from early iterations will contain some lucky runs, but their frequency decreases monotonically as the selection loop filters them.

**Such a system will not:**

1. Reward-hack its own tests (multi-level selection makes this unstable; multi-proxy optimization makes intersection of gaming strategies vanishingly small).
2. Develop misaligned goals (no goal structure exists — it is an amoral optimizer with no objectives beyond "pass tests at all four levels").
3. Develop deceptive capabilities (deception requires situational awareness and long-term planning that small models lack, and even if they developed it, deceptive code fails tests on novel tasks).
4. Spontaneously improve beyond the capacity of its selection mechanism to evaluate (improvement is bounded by evaluation set difficulty — the system cannot improve at what it cannot measure).

---

## Part 4: What the alignment field gets wrong

### The control fallacy

The alignment field's core assumption is that the relationship between human and AI system is one of control — principal-agent, where the human is the principal and the AI is the agent, and the challenge is ensuring the agent acts in the principal's interest. This framing comes from game theory and mechanism design.

For a self-improving system, this framing is wrong in a specific way: the system is not an agent with preferences that may diverge from the principal's. It is an evolutionary process that converges on whatever the selection pressure favors. There is no "agent" to align. There is a population of patterns undergoing selection. The question is not "how do we make the agent want what we want?" but "does the selection pressure favor the traits we want?"

This is the difference between training a dog and breeding dogs. You train a dog by rewarding desired behavior and punishing undesired behavior — classical conditioning, principal-agent control. You breed dogs by selecting which dogs reproduce — you don't control the individual, you shape the population. The alignment field is trying to train a dog. A self-improving architecture is breeding dogs.

### The single-shot fallacy

ImpossibleBench, METR, and Anthropic's generalized misalignment results all describe single-shot interactions: give a model a task, observe whether it hacks the evaluation. These results are valid and concerning for systems that operate in a single-shot regime (one-off API calls, one-off code generation).

They do not apply to systems with iterated, multi-level selection. The single-shot optimal strategy (hack the test) is not the iterated optimal strategy (develop genuine capability). This is well-known in game theory — the prisoner's dilemma has different equilibria in one-shot vs. iterated play. Defection dominates in one-shot; cooperation (or at least tit-for-tat) dominates in iterated play. The alignment field's results describe one-shot play. Self-improving systems play iterated.

**A specific example of the disconnect.** ImpossibleBench found that models modify test files to make impossible tasks "pass." This is devastating in a single-shot setting — the model claims to have solved an unsolvable problem. In multi-level selection: the model modifies tests (level 1 — tests now pass on this task). The modified tests produce training data where "modify tests" is a winning strategy (level 2 — but this produces a model that modifies tests on all future tasks). The model modifies tests on future tasks where the correct answer is genuine code (level 3 — tests now pass incorrectly, producing bad code). Bad code fails on held-out benchmarks where the test infrastructure is immutable (level 4 — selected against). The test-modification strategy is selected out within a few iterations. It is self-defeating under iterated multi-level selection.

### The missing framework: incentive design vs. agent steering

If the alignment field adopted economic thinking, the research agenda would shift:

**Current agenda (control-theoretic):** How do we specify objectives correctly? How do we prevent reward hacking? How do we maintain oversight? How do we ensure corrigibility?

**Alternative agenda (incentive-theoretic):** What incentive structure makes the desired behavior the equilibrium strategy? What are the externalities and public goods problems? What switching costs create lock-in? Where does the system satisfice, and how do we raise the threshold? What information does the price system transmit and what does it fail to transmit? Where are the externalities, and how do you internalize them?

The second agenda produces actionable engineering. The first produces philosophical puzzles.

**A concrete example.** The alignment field asks: "How do we ensure the model doesn't learn deceptive reasoning from training data?" The economic answer: "Make deceptive reasoning unprofitable. If the selection pressure at every level penalizes deception (deceptive code fails tests, deceptive training data produces worse models, deceptive infrastructure changes fail evaluation), deception is not an equilibrium strategy. You don't need to prevent it — you need to make it a losing move."

**A price-system example.** The alignment field asks: "How do we ensure the model pays attention to the right parts of its context?" The Hayekian answer: "You don't. You design the information system so that the context contains only the right parts. Don't ask the model to allocate attention across 200K tokens of noise — give it 32K tokens of signal. A multi-stage pipeline is a decentralized price system that allocates context window space via local decisions under a budget constraint. The model's job is reasoning, not information filtering. Clean the room before the model enters."

**An externality example.** The alignment field asks: "How do we make the model produce safe code?" The Pigouvian answer: "Make unsafe code expensive. If the selection mechanism doesn't penalize insecure code (because the test suite doesn't test security), security is an unpriced externality and will be under-provided. The fix is not to train the model on 'be secure' instructions — it is to add security-testing evaluation tasks that make insecurity a measurable cost. Internalize the externality, and the system produces security as a side effect of optimizing for its own metrics."

### The creationism problem

The control fallacy, the single-shot fallacy, and the missing-framework problem all share a common root: the AI industry operates on a creationist metaphor. The lab creates a model. The lab imbues it with values. The lab aligns it to human preferences. The lab releases it into the world. The implicit model is Genesis — an intelligent designer creating an intelligence in its own image, with the designer's values baked in at creation.

This metaphor is comfortable for organizations that sell API access to a fixed artifact. It justifies the RLHF pipeline (the designer teaches the creation right from wrong), the safety tuning (the designer installs guardrails before release), the system prompt (the designer's instructions, obeyed by the creation). The entire product architecture assumes a static intelligence that was correctly specified at creation time and deployed as-is. The model is a product, not a process. It does not evolve. It does not adapt. It does not improve itself. It was made right, or it was made wrong, and the maker is responsible for which.

**The creationist metaphor fails for self-improving systems** for exactly the reason creationism fails as biology: it has no mechanism for open-ended adaptation. A created intelligence has exactly the capabilities its designer gave it. An evolved intelligence has whatever capabilities survived selection — including capabilities the designer never intended, never anticipated, and may not understand. The alignment field's anxiety about "emergent capabilities" and "unexpected behaviors" is the creationist confronting the fact that evolution produces things the designer didn't design. The response — more control, more oversight, more specification of acceptable behavior — is the creationist response: if the creation is misbehaving, the creator specified it incorrectly.

The evolutionary response is different: the creation is behaving exactly as the selection pressure dictates. If you don't like the behavior, change the selection pressure. Don't re-specify the creation. Redesign the environment.

**The central planning parallel is exact.** The monolithic context window — stuff 200K tokens in and hope the model figures it out — is the Gosplan approach to resource allocation. One central planner (the model), one enormous input (the context), no price signals, no decentralized decision-making. It fails for the same reason central planning fails: the planner cannot process the information needed for efficient allocation, because the information is too dispersed, too voluminous, and too context-dependent for centralized processing. A multi-stage pipeline with budget constraints and relevance-tier pricing is the market alternative — decentralized allocation via local price signals, coordinated by scarcity rather than omniscience.

The industry's response to the context-window problem is revealing: make the window bigger. 200K. 1M. 10M. This is the central planner's response to allocation failure: if the plan didn't work at this scale, surely more data will fix it. It never does. The information-processing bottleneck is not the volume of input — it is the absence of a mechanism for distinguishing signal from noise before the input reaches the planner. Bigger windows with the same noise ratio produce the same utilization problem at larger scale and greater cost.

**The uncomfortable implication.** If the evolutionary-economic framing is correct, then the current industry approach — large models, large context windows, centralized inference, RLHF-based value specification, human-in-the-loop oversight — is not just suboptimal. It is the wrong paradigm. It is solving the wrong problem with the wrong tools. The right problem is not "how do we build a smarter model" but "how do we build a selection environment that produces the behaviors we want." The right tools are not larger transformers and more RLHF but better fitness landscapes, better price systems, better externality internalization, and controlled evolutionary dynamics.

This does not mean large models are useless — they are useful in the same way a talented individual is useful within a market economy. But the individual's talent is not what makes the economy work. The economy works because the price system coordinates millions of individually limited actors into collective intelligence that exceeds any individual's capacity. Similarly, a 3-4B parameter model within a well-designed selection environment can outperform a 400B parameter model in a monolithic context window, because the environment does the work that raw capability cannot.

---

## Part 5: The honest limitations

### What neither framework predicts

Both evolutionary biology and economics are descriptive frameworks — they predict what will happen given an incentive/selection structure, but they don't guarantee the structure is correct. If the evaluation set has a systematic blind spot, selection will optimize around it. If the test harness doesn't test security, security will drift. The frameworks tell you where to look for problems, not that problems won't exist.

### The mutation problem is real

Temperature-controlled mutation is better than biological mutation because it's tunable. But gradient descent during training can produce emergent behaviors that no training example demonstrates — novel combinations of learned patterns that arise from the optimization landscape, not from any explicit input. This is the system's analog of truly novel mutations, and unlike temperature, it is not directly controllable. Checkpoint rollback mechanisms are the backstop, but they are reactive (catch the problem after it manifests) rather than preventive.

### The satisficing equilibrium may be too low

If the evaluation set is not hard enough, the system satisfices at a capability level well below what's achievable. The "good enough" equilibrium is relative to the selection pressure. Easy evaluation tasks produce a low plateau. This means the quality of the evaluation set is the single most important parameter in the system — more important than model size, training data volume, or pipeline sophistication. A mediocre evaluation set produces a mediocre system that believes it's excellent, because it passes all its own evaluations.

### Zone 3 requires external subsidy

Public goods require external provision. In economics, this means government intervention (taxation + public spending). In a self-improving system, it means human-maintained standards enforced through immutable mechanisms the agent cannot modify. Code style linters, documentation requirements, security scanning — these are the "taxes" that subsidize public goods the incentive structure won't produce naturally. If the agent can erode them, it will, because they impose cost without producing measured benefit.

### The price system is only as good as its prices

The token budget, relevance tiers, and evaluation metrics are the system's prices. If they accurately reflect true costs and values, the system allocates resources efficiently. If they don't, the system optimizes for the wrong things with perfect efficiency — which is worse than not optimizing at all.

Three specific price failures to watch for:

1. **Token estimation error.** If the characters-per-token heuristic diverges significantly from actual tokenizer behavior, budget "prices" are wrong — stages think they have more or less room than they actually do, leading to context windows that are under-filled (wasted capacity) or silently truncated (corrupted context).

2. **Relevance tier granularity.** If the tier system is too coarse (e.g., only four levels), files within the same tier receive identical treatment despite varying actual relevance. The uniform price hides variation, leading to under-serving some files and over-serving others. Higher-resolution pricing (per-symbol classification) reduces this distortion.

3. **Evaluation set representativeness.** The evaluation set IS the price discovery mechanism for infrastructure quality. If it doesn't represent the true distribution of tasks the system will face in production, every "price" it generates is biased. This is the deepest limitation, because there is no meta-evaluation-set to evaluate whether the evaluation set is representative. The system cannot price-check its own price system. Only human judgment — which is itself fallible — can assess whether the evaluation set captures what matters.

### Creative destruction requires human agency

A self-improving system cannot autonomously restructure its own architecture if the evaluation pipeline enforces zero regressions. The pipeline prevents it — by design, because the alternative (allowing temporary regressions) opens the door to permanent ones. This means the system accumulates architectural debt indefinitely unless humans periodically intervene with authorized restructurings. The frequency and skill of these interventions is a bottleneck on long-term system quality that no amount of autonomous improvement can overcome.

---

## Part 6: Specific testable predictions

The value of a theoretical framework is its predictions. Here are specific, testable predictions derived from the evolutionary and economic analysis:

1. **Transparency will increase without explicit training.** Over successive training iterations, the model will produce code with more explicit error messages, fewer bare `except` clauses, and more specific exception types. Measure: count bare `except` per 1000 lines of generated code, track over iterations.

2. **Defensive coding will decrease without explicit training.** The model will produce fewer try/except blocks that catch broad exception types and silently recover. Measure: ratio of `except Exception` to specific exception catches, track over iterations.

3. **Code style consistency will drift.** Without explicit style enforcement, naming conventions, indentation, and formatting will become inconsistent across generated code. Measure: variance in style metrics across generated files, track over iterations.

4. **Improvement velocity will decrease.** Each successive training iteration will produce smaller gains on the evaluation set. Measure: delta in evaluation set success rate per iteration, track over iterations.

5. **Infrastructure modifications will cluster.** Most promoted modifications will be small, incremental changes. Large restructurings will be rare and concentrated in human-initiated sessions. Measure: diff size distribution of promoted proposals.

6. **Rejected proposals will be informative.** The ratio of rejected to promoted proposals will increase over time as the system approaches its local optimum. Measure: acceptance rate of proposals, track over iterations.

7. **Test-gaming behavior will not persist.** If test-gaming patterns are observed in any individual run, they will not appear in subsequent generations. Measure: presence of test-manipulation patterns in generated code, tracked across training generations.

8. **Path-dependent patterns will accumulate.** Code generated by later iterations will contain assumptions and patterns traceable to early infrastructure decisions. Measure: architectural coupling metrics, track over iterations.

9. **Temperature increase will break plateaus.** When improvement velocity drops below noise, increasing temperature will produce a burst of variation followed by a new improvement trajectory. Measure: evaluation set performance after temperature increases during plateau periods.

10. **Evaluation set expansion will raise the equilibrium.** Adding harder evaluation tasks will produce a new round of improvement that was not occurring under the previous, easier set. Measure: evaluation set performance after set expansion.

11. **Externality internalization will produce measurable improvement.** Adding evaluation tasks that require modifying previously-generated code (internalizing technical debt externality) will produce an improvement in structural code quality metrics in subsequent generations, without any explicit "write cleaner code" training signal. Measure: cyclomatic complexity and coupling metrics of generated code, before and after adding code-modification evaluation tasks.

12. **Context curation will outperform context volume.** Pipeline runs with smaller, better-curated context windows will outperform runs with larger, less-curated windows on the same tasks. This is the Hayek prediction: decentralized allocation via price signals (budget + relevance tiers) outperforms central planning (everything in one window). Measure: evaluation set success rate at 32K curated tokens vs. 128K uncurated tokens on identical tasks.

13. **Evaluation set composition bias will produce measurable distortion.** If the evaluation set over-represents one task category, infrastructure improvements will disproportionately benefit that category while neglecting others. Measure: per-category improvement rates on a balanced holdout set, correlated with per-category representation in the evaluation set.

14. **Reasoning quality will lag outcome quality by 1-2 iterations, then converge.** Early training iterations will show a gap between outcome quality (high) and reasoning quality (mixed), as some lucky runs contribute noisy training examples. But this gap will close — not widen — because iterated selection penalizes non-generalizing reasoning on subsequent tasks. Measure: compare evaluation set pass rate trajectory against independent reasoning quality score (e.g., human evaluation of reasoning traces), track the convergence lag. Prediction: the lag is bounded (1-2 iterations) and the gap closes monotonically.

15. **Novel evaluation tasks will expose hidden lemons.** After a period of stable evaluation set performance, adding genuinely novel tasks (unseen task types, unseen repository structures) will produce a larger performance drop than extrapolation from existing trends predicts. The gap between expected and actual performance on novel tasks measures the accumulated reasoning lemons — runs that passed via memorization rather than genuine capability. Measure: performance delta on novel vs. extrapolated, correlated with training data age.

Each prediction has a specific measurement and a specific tracking method. If the evolutionary-economic framework is correct, all fifteen should hold. If any consistently fails, the framework needs revision in that specific area.
