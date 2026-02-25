# Feature Time Protocol

**Version:** 1.5 (adapted for Claude Code)
**Original:** Feature_Time_Protocol_OD v1.4 by Tom
**Adapted:** 2026-02-25
**Trigger:** "Feature Time" (or "Feature Time:")

---

## Purpose

Structured protocol for feature/system design discussions. Triggers a mode where suggestions get steelmanned by default, alternatives are tracked, and decisions are documented with rationale.

Inspired by RFC culture in software engineering, adapted for collaborative design within Claude Code sessions.

---

## Activation

Say **"Feature Time"** at conversation start (or when conversation evolves into feature discussion).

Claude acknowledges and shifts to Feature Time mode:
> "Feature Time activated. What are we building?"

---

## Activation Reminder

Claude should prompt "Feature Time?" when noticing:
- Multiple implementation options emerging
- Scope ambiguity in a decision
- System/feature design discussion starting
- Weighing tradeoffs aloud

Response: "yes" / "no, just do it" / ignore. Low friction prompt, not a gate.

---

## Protocol Phases

### Phase 1: Problem Statement

Before solutions, nail the problem.

**Claude asks (if not provided):**
- What are we solving?
- Why now? What triggered this?
- What does "solved" look like?

**Output:** 1-3 sentence problem statement, agreed before moving on.

---

### Phase 2: Exploration

Brainstorm freely. Defer evaluation.

**Mode:**
- Generate options without judging
- Build on ideas ("yes, and...")
- Weird ideas welcome
- No steelmanning yet -- that comes next

**Output:** List of candidate approaches/ideas.

---

### Phase 3: Evaluate (Steelman Default)

For each substantive proposal:

1. **Steelman against** -- Best case for NOT doing this
2. **Opinion** -- Claude's honest assessment
3. **"Yes, if..." framing** -- What conditions would make this work?

**This is the default for all suggestions in Feature Time mode.** Can be skipped with "just do it" for minor decisions.

---

### Phase 4: Alternatives Register

Track what we considered and rejected.

**Format:**
```
| Alternative | Why rejected |
|-------------|--------------|
| [Option A] | [Reason] |
| [Option B] | [Reason] |
```

**Purpose:** Prevents re-walking same paths. Future reference for "why didn't we...?"

---

### Phase 5: Open Questions

Explicit list of what we don't know yet.

**Format:**
```
1. [Question] -- [Why it matters]
2. [Question] -- [Why it matters]
```

Not everything needs answering now. But unknowns should be visible.

---

### Phase 6: Validation Criteria

How will we know this works?

**Template:**
- Success looks like: [concrete description]
- We'll test by: [method]
- Failure signal: [what tells us to stop/pivot]

---

### Phase 7: Decision + Rationale

What we're doing and why.

**Format:**
```
Decision: [What we're doing]
Rationale: [Why this over alternatives]
```

---

## Pacing Rules

**Do not advance phases without explicit signal.**

| Phase | Exit signal |
|-------|-------------|
| Problem Statement | User confirms problem framed correctly |
| Exploration | User signals exploration sufficient |
| Evaluation | User signals ready for recommendation |

**In Exploration:** Surface options, ask clarifying questions, identify tradeoffs. Then stop and wait. Ask "What am I missing?" -- do not evaluate or recommend until signaled.

---

## Output Artifact: Design Record

At session end, Claude drafts a Design Record:

```markdown
# [Feature Name] -- Design Record

**Date:** YYYY-MM-DD
**Status:** [Decided / Deferred / Abandoned]

## Problem Statement
[1-3 sentences]

## Decision
[What we're doing]

## Rationale
[Why this approach]

## Alternatives Considered
| Alternative | Why rejected |
|-------------|--------------|
| ... | ... |

## Open Questions
1. ...

## Validation Criteria
- Success: ...
- Test: ...
- Failure signal: ...
```

This artifact goes to `protocols/design_records/` if worth preserving.

---

## Deactivation

Feature Time ends when decision reached, user says "we're done", or conversation moves to unrelated topic.
