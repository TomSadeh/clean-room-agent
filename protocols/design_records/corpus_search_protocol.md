# Corpus Search Protocol — Design Record

**Date:** 2026-02-26
**Status:** Decided

## Problem Statement

We need a repeatable, structured process to discover and evaluate new repositories and scientific papers for the training corpus. The corpus must grow over time to inject fresh reasoning patterns, break training plateaus, and expand Jane's domain coverage. This process is run by us (human + Claude Code sessions), not by Jane herself. Currently the corpus exists as a static list built in one brainstorming session — there's no systematic way to find what we're missing, evaluate candidates consistently, or prioritize what to package next.

## Decision

A gap-first, three-gate evaluation protocol run opportunistically in Claude Code sessions.

### Process (per run)

1. **Gap analysis first** (5 min) — scan `repo-corpus.md` category counts. Identify the 2-3 thinnest categories or domains with no coverage.

2. **Targeted search** — for each gap, search using these channels in priority order:
   - Awesome lists for the domain
   - Papers-with-code filtered by repo active 1+ year post-publication
   - Dependency mining from existing corpus repos
   - GitHub topic/language search
   - Quick scan of trending (5 min max)

3. **Evaluate each candidate** — three gates, in order, all must pass:

   **Gate 1 — Domain transparency:** Does this domain's methodology practice transparency about its reasoning limitations?
   - Inherently verifiable (math, CS, engineering — the code compiles or it doesn't) → pass
   - Transparent with formal hedging methods (epidemiology, Bayesian statistics, experimental physics — explicit about what it can and can't conclude) → pass
   - Obscures limitations, unfalsifiable claims, circular reasoning → reject

   **Gate 2 — Implementation quality (high bar):** Clean code. Decent commit history. Documentation that explains the why. Actively maintained or feature-complete. License doesn't matter (air-gapped, nothing leaves).

   **Gate 3 — Novelty (low bar):** "Could Jane learn something from this that she couldn't learn from what we already have?" Even marginal variance counts — same reasoning shape in a different domain context is valuable. Only reject if exact same reasoning, same domain, same approach as an existing corpus repo.

4. **Tier assignment:**
   - Has any validation path (tests, runnable code, observable output) → Full harness
   - Clean code but no validation path → KB only
   - Paper/documentation → Convert with opendataloader-pdf, index

5. **Record:**
   - Accepted → add to `repo-corpus.md` with category, license, tier, one-line reasoning-pattern note
   - Rejected → add to "Rejected" section at bottom of `repo-corpus.md` with one-line reason
   - Papers accompanying new repos → queue for opendataloader-pdf conversion

### Triggers

- Opportunistic: we're in a session and have bandwidth
- Plateau-triggered: training metrics stall, need new reasoning patterns
- No fixed calendar

## Rationale

**Domain transparency gate** is self-consistent with the project's core principle ("Transparency Is Load-Bearing"). The same standard that governs the codebase governs what the agent learns from. Domains that obscure their reasoning limitations teach bad reasoning patterns regardless of code quality. Domains with explicit hedging methodology (epidemiology, evolutionary biology) are valuable BECAUSE of their inference challenges — they teach Jane how to reason under uncertainty, how to hedge, how to be explicit about what she knows and doesn't know.

**Low novelty bar + high quality bar** because variance compounds. A regression in an econ model and a least-squares fit in physics are the same reasoning shape in different contexts — the agent learns to generalize by seeing the gradient between them, not by categorizing them as "distinct" or "redundant." But only well-implemented variance is valuable. Garbage code teaches garbage patterns.

**Gap-first search** because focused effort on weak categories beats exhaustive sweeps. Most runs, most categories are fine.

**No fixed cadence** because the corpus doesn't need feeding on a schedule. It needs feeding when there's a gap or a plateau.

**Three gates in order** because domain gate eliminates entire categories cheaply (no point evaluating code quality in a repo implementing homeopathy dosage calculations). Quality gate is high. Novelty gate is low.

## Alternatives Considered

| Alternative | Why rejected |
|---|---|
| Numeric soft scoring (0-2 per criterion, threshold 6/12) | False precision. Binary pass/fail per gate is clearer |
| Separate candidates file for "maybe" repos | "Maybe" piles grow forever. Binary: add or reject |
| Calendar-based cadence (monthly/quarterly) | Forced busywork when corpus is healthy |
| Full sweep every run (all channels, all categories) | Exhaustive and slow. Gap-first is higher ROI |
| Living taxonomy of reasoning patterns | Requires precision we don't need. Low novelty bar means we don't need crisp pattern boundaries |
| High novelty bar ("must teach a distinct new pattern") | Rejects valuable variance. Similar-but-different reasoning across domains is exactly what compounds |
| Jane runs the search protocol herself | Jane consumes the corpus, doesn't curate it. Curation requires external judgment |
| "Hard science only" domain filter | Too restrictive. Epidemiology, evolutionary biology have sound transparent methodology despite not being "hard" science. The filter is transparency about reasoning limitations, not formality of the discipline |
| Allowlist of approved domains | Premature. Apply judgment per candidate, codify patterns as they emerge |
| Pre-filter script for automated screening | Over-engineering for a process with low volume. Manual for now, script if volume demands it |
| Cherry-picking subdirectories from large repos | Unnecessary complexity. Storage is cheap. Just copy the whole repo |
| Training on domain content directly | Content teaches WHAT, not HOW. Domain content goes to KB for retrieval. Training pipeline only gets repos that teach reasoning patterns through code and commit history |
| Exclude gray domains entirely | Loses useful knowledge. The curated training pipeline (below) solves the leakage problem without exclusion |
| "KB only, never training" as a clean boundary | The boundary leaks: KB content → retrieval → prompt → logged call → raw DB → training data. Any content on the machine eventually influences training through the pipeline |
| Separate "potential bullshit" DB on the machine | Risk/reward wrong. If retrievable, it influences training. If not retrievable, it's dead weight. Non-transparent domains stay off the machine entirely |

## Domain Confidence and Training Data Leakage

The raw-to-curated promotion boundary solves the gray domain problem.

**The leakage path:** Gray content in KB → retrieved into prompt → prompt+response logged to raw DB → raw DB mined for training data. Any content on the machine eventually touches training.

**The solution:** Cut the leakage at the curation step, not at retrieval.

1. Gray domain content enters KB tagged with `domain_confidence: "gray"` (transparent domains get `domain_confidence: "transparent"`)
2. Retrieval pulls gray content into prompts when relevant — Jane can use it for tasks
3. The prompt + response is logged to raw DB as always (append-only, traceability)
4. **The tag propagates** — if retrieval included gray-tagged content, the logged call in raw DB carries that tag
5. `cra curate-data` filters: any logged call that touched gray content is **excluded** from training data extraction
6. Training datasets contain ONLY prompts built entirely from transparent-domain content

**Domain tiers:**

| Domain quality | On the machine? | Retrievable? | Enters training? |
|---|---|---|---|
| **Transparent** (verifiable, or transparent about limitations) | Yes | Yes | Yes |
| **Gray** (useful knowledge, methodology has known weaknesses) | Yes | Yes (tagged) | **No** — excluded at curate-data step |
| **Non-transparent** (obscures limitations, unfalsifiable) | **No** | No | No |

This is the same default-deny principle that governs the retrieval pipeline: content without a positive curation signal doesn't get promoted. `domain_confidence != "transparent"` → exclude from training datasets.

**Implementation requirements:**
- `domain_confidence` field on KB entries
- Tag propagation through retrieval decisions to logged calls in raw DB
- Filter in `cra curate-data` pipeline excluding gray-tainted calls

## Open Questions

1. **Reasoning pattern taxonomy** — deferred to Claude Chat research session. A general taxonomy of thought patterns would help answer the novelty gate more consistently. Not a blocker for running the protocol.
2. **Gray domain boundary in practice** — the transparent/gray distinction requires judgment calls per domain. We'll learn the boundary by doing. Some humanities have rigorous parts and non-rigorous parts within the same discipline — the tag may need to be per-source, not per-domain.

## Validation Criteria

- **Success:** Running the protocol produces a clear list of repos to add, each with tier and reasoning-pattern note. Gap analysis identifies underrepresented categories without deep research. Corpus grows in domain diversity, not just volume.
- **Test:** Run the protocol once on 2-3 thin categories. Did we find repos we wouldn't have found browsing? Did the gates reject what should be rejected? Did it take under 1 hour?
- **Failure signal:** Every run surfaces the same repos. Quality gate lets through garbage or blocks obvious value. Process takes so long nobody runs it.
