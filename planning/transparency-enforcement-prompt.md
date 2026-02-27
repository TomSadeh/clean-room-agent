# Prompt: Audit and Enforce Fail-Fast / No-Fallback Principles

## Problem

This codebase has a systemic problem: every implementation round introduces silent fallbacks, graceful degradation, and defensive coding patterns that violate the project's core principles. The CLAUDE.md says "no fallbacks and no hardcoded defaults in core logic" and "prefer fail-fast behavior" but these instructions compete against default coding instincts and lose consistently. Fix rounds sometimes introduce *more* violations than they remove.

The most recent example: implementing binary decomposition (Plan A) and scaffold-then-implement (Plan B). The initial implementation introduced:
- `ModelRouter.resolve()` silently falling back from classifier→reasoning when classifier wasn't configured
- A magic threshold `llm.config.context_window < 16384` to auto-activate binary mode instead of explicit config-driven activation
- Scaffold pass catching all exceptions and "falling through to normal step loop" instead of failing
- `getattr(stage, "preferred_role", "reasoning")` — a default that silently hides missing protocol compliance

The fix round removed some of these but introduced a new one:
- Pipeline checking `router.has_role(stage_role)` and silently substituting `"reasoning"` — same fallback pattern, just moved from the router to the pipeline

## What You Need To Do

Two things, in order:

### 1. Audit the current code for violations

Search the entire `src/clean_room_agent/` tree for these specific anti-patterns:

**Pattern A: Silent substitution.** Code that detects a missing/invalid value and silently substitutes a different one. The canonical form is `if not X: X = Y` or `X if condition else Y` where the substitution is not logged as an error and execution continues normally. This includes `getattr(obj, attr, default)` where the default silently changes behavior.

The specific instance to find and fix: `pipeline.py` line ~274-276 where `if not router.has_role(stage_role): stage_role = "reasoning"` silently substitutes reasoning for classifier.

**Pattern B: Catch-and-continue.** `try/except` blocks that catch exceptions and continue execution with degraded behavior instead of re-raising with context. The banned pattern is: catch → log warning → continue/return a default. The correct pattern is: catch → add context → re-raise (or don't catch at all).

**Pattern C: Magic thresholds as feature toggles.** Numeric comparisons used to activate/deactivate features instead of explicit boolean config. Example: `context_window < 16384` to decide binary mode.

**Pattern D: Optional dependencies treated as soft failures.** `try: import X; except ImportError: <degrade>` patterns that silently disable features instead of failing at startup.

For each violation found, classify it:
- **Core logic violation** — must be fixed (the system produces wrong/untraced results)
- **Boundary validation** — acceptable (input validation at system edges, e.g. CLI argument parsing)
- **Intentional design** — acceptable with justification (e.g. R2 default-deny in retrieval is *exclusion*, not substitution)

### 2. Design structural enforcement

The goal: make fallback patterns harder to write than fail-fast patterns. Not more instructions — infrastructure that makes the wrong thing a type error or a missing method.

Concrete ideas to evaluate (pick what works, discard what doesn't):

- **`strict_get(dict, key)` utility** that raises on missing keys, replacing `.get(key, default)` in core logic. If you need a default, you must use a separate `optional_get()` that requires a classification comment.

- **Protocol enforcement for `preferred_role`** — the `RetrievalStage` protocol declares `preferred_role` as a property. Make it abstract (no default implementation) so every stage must explicitly declare its role. If a stage wants reasoning, it says so. No `getattr` with defaults.

- **A "no-fallback" linter** — a pytest-based test that greps `src/` for the banned patterns and fails the build. This is the strongest enforcement because it catches violations before they land. Patterns to flag:
  - `except.*:` followed by `continue` or `return` without `raise` (within 5 lines)
  - `.get(` in files under `src/clean_room_agent/retrieval/`, `src/clean_room_agent/orchestrator/`, `src/clean_room_agent/llm/` (core logic dirs) without an adjacent `# Optional:` or `# Supplementary:` classification comment
  - `getattr(.*,.*,` (3-arg getattr with default) in core logic
  - `if not.*:.*=` substitution patterns

- **Config validation at init time, not use time.** If scaffold_enabled=true requires gcc, check at init. If stages use classifier role, validate classifier is configured when those stages are in the stage list — at pipeline startup, not when the stage runs.

For the linter approach: write it as `tests/test_no_fallbacks.py`. It should be a real test file that runs with `pytest` and fails the build when violations are found. Include an allowlist mechanism for the rare cases where a pattern is intentional (with required justification comment in the allowlist).

## Key Files

Recent changes that need auditing (these are where the violations were introduced):
- `src/clean_room_agent/llm/router.py` — ModelRouter with classifier role
- `src/clean_room_agent/retrieval/pipeline.py` — stage loop role resolution
- `src/clean_room_agent/retrieval/scope_stage.py` — binary judgment path
- `src/clean_room_agent/retrieval/similarity_stage.py` — binary judgment path
- `src/clean_room_agent/retrieval/batch_judgment.py` — run_binary_judgment()
- `src/clean_room_agent/execute/scaffold.py` — scaffold-then-implement (NEW)
- `src/clean_room_agent/orchestrator/runner.py` — scaffold orchestrator integration

But the audit should cover ALL of `src/clean_room_agent/` — the older code likely has pre-existing violations too.

## What Success Looks Like

After this work:
1. Every current violation is either fixed or explicitly allowlisted with justification
2. `pytest tests/test_no_fallbacks.py` passes and will catch future violations
3. The pipeline's classifier→reasoning substitution is replaced with a structural solution (validate at pipeline startup that all stages' preferred roles are configured, or fail)
4. No `getattr` with defaults on protocol-defined properties in core logic
