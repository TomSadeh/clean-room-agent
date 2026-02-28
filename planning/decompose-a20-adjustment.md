# Implementation Plan: Decompose A20 (Adjustment) into Binary Sub-tasks + Focused Finalize

Date: 2026-02-28

## 1. Overview

The current A20 adjustment pass (in `src/clean_room_agent/execute/plan.py` via `execute_plan(..., pass_type="adjustment")`) makes a single 1.7B LLM call that receives:
- A `ContextPackage` from the retrieval pipeline
- A description string ("Remaining steps after {step.id} in part {part.id}")
- `prior_results` containing only the last `StepResult` (success/failure + error_info)
- `cumulative_diff` containing all prior code changes

The LLM must simultaneously: understand what failed, determine which remaining steps are still valid, identify root causes, decide if new steps are needed, and produce a revised step list. This is too much cognitive load for 1.7B.

The decomposition replaces this monolithic call with:

1. **Deterministic failure extraction** -- parse `StepResult` into structured failure categories
2. **Binary step viability** -- per remaining step, "Is this step still valid?"
3. **Binary root cause attribution** -- per (failure, step) pair, "Did this step cause this failure?"
4. **Binary new step detection** -- per unattributed failure, "Does this need a new step?"
5. **Focused 1.7B finalize** -- given all binary verdicts, produce the revised step list

Output remains `PlanAdjustment` -- no downstream changes.

## 2. Data Flow

```
StepResult (from prior step)
    |
    v
[1. extract_failure_signals()] -- deterministic, no LLM
    |
    v
 FailureSignal[]
    |
    +--> [2. run_binary_judgment: step_viability] -- per remaining step
    |        Input: failure signals + step description
    |        Output: verdict_map {step_id: bool}
    |
    +--> [3. run_binary_judgment: root_cause] -- per (failure, step) pair
    |        Input: single failure signal + single step description + cumulative_diff
    |        Output: verdict_map {(failure_idx, step_id): bool}
    |
    v
(only for unattributed failures)
    +--> [4. run_binary_judgment: needs_new_step] -- per unattributed failure
    |        Input: failure signal + all remaining steps summary
    |        Output: verdict_map {failure_idx: bool}
    |
    v
AdjustmentVerdicts (viability, root_causes, new_step_needed)
    |
    v
[5. finalize_adjustment()] -- focused 1.7B call
    Input: verdicts + remaining steps + failure signals + cumulative_diff
    Output: PlanAdjustment (same as before)
```

## 3. New File

**`src/clean_room_agent/execute/decomposed_adjustment.py`**

This follows the exact pattern of `src/clean_room_agent/execute/decomposed_plan.py` -- a module-level public function that replaces the monolithic call, composed of private stage functions.

## 4. New Dataclasses

Add to `src/clean_room_agent/execute/dataclasses.py`:

```python
@dataclass
class FailureSignal(_SerializableMixin):
    """A single categorized failure extracted from a StepResult."""
    category: str     # "compile_error", "test_failure", "patch_failure", "runtime_error", "unknown"
    message: str      # the raw failure text (truncated if necessary)
    source: str       # "error_info", "raw_response", or "step_failed"

    _REQUIRED = ("category", "message", "source")
    _NON_EMPTY = ("category", "message", "source")


@dataclass
class AdjustmentVerdicts(_SerializableMixin):
    """Aggregated binary verdicts from the decomposed adjustment sub-tasks."""
    step_viability: dict[str, bool]         # step_id -> still valid?
    root_causes: dict[str, list[int]]       # step_id -> [failure_indices it caused]
    new_steps_needed: list[int]             # failure_indices that need new steps
    failure_signals: list[FailureSignal]    # the extracted failures (for context)

    _REQUIRED = ("step_viability", "root_causes", "new_steps_needed", "failure_signals")
    _NESTED = {"failure_signals": FailureSignal}
```

## 5. Functions to Add

All in `src/clean_room_agent/execute/decomposed_adjustment.py`:

### 5.1 Public entry point

```python
def decomposed_adjustment(
    context: ContextPackage,
    task_description: str,
    llm: LoggedLLMClient,
    *,
    prior_results: list[StepResult] | None = None,
    remaining_steps: list[PlanStep],
    cumulative_diff: str | None = None,
) -> PlanAdjustment:
    """Generate a PlanAdjustment via decomposed binary sub-tasks.

    1. Extract failure signals deterministically from prior_results
    2. Binary viability per remaining step
    3. Binary root cause per (failure, step) pair
    4. Binary new step detection per unattributed failure
    5. Focused 1.7B finalize with all verdicts

    Returns a PlanAdjustment identical in structure to execute_plan("adjustment").
    """
```

Note the key difference from the current `execute_plan(..., pass_type="adjustment")`: this function takes `remaining_steps: list[PlanStep]` explicitly. The current monolithic call does NOT receive remaining steps explicitly -- the LLM must infer them from context. The decomposed version needs them explicitly for binary judgments.

### 5.2 Stage 1: Deterministic failure extraction

```python
# Category constants
FAILURE_CATEGORY_COMPILE = "compile_error"
FAILURE_CATEGORY_TEST = "test_failure"
FAILURE_CATEGORY_PATCH = "patch_failure"
FAILURE_CATEGORY_RUNTIME = "runtime_error"
FAILURE_CATEGORY_UNKNOWN = "unknown"

_COMPILE_PATTERNS = [
    re.compile(r"error:.*expected|undefined reference|implicit declaration", re.I),
    re.compile(r"gcc|clang|cc1|ld:.*error", re.I),
    re.compile(r"syntax error|parse error|redefinition", re.I),
]
_TEST_PATTERNS = [
    re.compile(r"FAIL|FAILED|AssertionError|assert.*failed", re.I),
    re.compile(r"test_\w+.*FAILED|pytest.*failed", re.I),
]
_PATCH_PATTERNS = [
    re.compile(r"Patch.*failed|search.*not found|could not find", re.I),
]


def extract_failure_signals(
    prior_results: list[StepResult] | None,
) -> list[FailureSignal]:
    """Deterministically extract and categorize failure messages from step results.

    No LLM calls. Uses regex patterns against error_info and raw_response
    to classify failures into categories.

    Returns empty list if prior_results is None or all steps succeeded.
    """
```

Classification logic:
- If `step_result.success is True` and `step_result.error_info is None`: skip (no failure)
- If `step_result.success is False` and `step_result.error_info` is present: classify using regex patterns against `error_info`
- If `step_result.success is False` and `error_info is None`: create `FailureSignal(category="unknown", message="Step failed with no error info", source="step_failed")`
- Matching priority: compile > test > patch > runtime > unknown
- Message truncation: cap each `message` at 500 chars to keep binary prompts small

### 5.3 Stage 2: Binary step viability

```python
def _run_step_viability(
    failure_signals: list[FailureSignal],
    remaining_steps: list[PlanStep],
    task_description: str,
    llm: LoggedLLMClient,
) -> dict[str, bool]:
    """Binary per-step: 'Is step S still valid given these failures? yes/no'

    Uses run_binary_judgment with default_action="excluded" (fail-fast).
    Steps judged invalid will be dropped from the revised plan.
    """
```

Formatting for each item:
```
Failures observed:
- [compile_error] gcc: error: implicit declaration of 'hash_init'
- [test_failure] test_insert FAILED: assertion hash_size == 3 failed

Step to evaluate:
ID: s3
Description: Add hash resize logic
Target files: hash_table.c
Target symbols: hash_resize

Is this step still valid given the failures above? Answer yes or no.
```

The `run_binary_judgment` call uses:
- `system_prompt`: `SYSTEM_PROMPTS["adjustment_step_viability"]` (new)
- `task_context`: formatted failures summary
- `format_item`: formats a single `PlanStep`
- `stage_name`: `"adjustment_step_viability"`
- `item_key`: `lambda step: step.id`
- `default_action`: `"excluded"` (R2 default-deny: if parse fails, drop the step)

**Failure mode**: If ALL steps judged invalid by parse failure (all omitted), fail-fast with ValueError. This preserves the crash-is-signal principle.

### 5.4 Stage 3: Binary root cause attribution

```python
def _run_root_cause_attribution(
    failure_signals: list[FailureSignal],
    remaining_steps: list[PlanStep],
    cumulative_diff: str | None,
    task_description: str,
    llm: LoggedLLMClient,
) -> dict[str, list[int]]:
    """Binary per (failure, step) pair: 'Did step S cause failure F? yes/no'

    Only evaluates pairs where the step was judged viable (still in the plan).
    Returns dict mapping step_id -> list of failure indices it caused.
    """
```

This generates the cross product of (failure_signals x remaining_steps) as items. For N failures and M steps, this is N*M binary calls.

Formatting for each pair:
```
Prior changes:
<prior_changes>{cumulative_diff (capped)}</prior_changes>

Failure:
[compile_error] gcc: error: implicit declaration of 'hash_init'

Step:
ID: s2
Description: Implement hash table insert function
Target files: hash_table.c

Did this step's implementation cause or contribute to this failure? Answer yes or no.
```

The `run_binary_judgment` call uses:
- `system_prompt`: `SYSTEM_PROMPTS["adjustment_root_cause"]` (new)
- `task_context`: cumulative_diff section
- `format_item`: formats a `(FailureSignal, PlanStep)` tuple
- `stage_name`: `"adjustment_root_cause"`
- `item_key`: `lambda pair: f"{pair[0]_idx}:{pair[1].id}"` (using index for failure)
- `default_action`: `"not_attributed"` (R2: if parse fails, don't attribute)

**Optimization**: Only generate pairs for steps that were judged viable in stage 2. Invalid steps are already being dropped -- no point attributing failures to them.

**Optimization**: Only run if there are actual failures. If `failure_signals` is empty, return empty dict.

### 5.5 Stage 4: Binary new step detection

```python
def _run_new_step_detection(
    unattributed_failures: list[tuple[int, FailureSignal]],
    remaining_steps: list[PlanStep],
    task_description: str,
    llm: LoggedLLMClient,
) -> list[int]:
    """Binary per unattributed failure: 'Does this failure require a new step? yes/no'

    Only called for failures that no existing step was attributed as causing.
    Returns list of failure indices that need new steps.
    """
```

This is called only for failures where no step was attributed as root cause in stage 3. If all failures were attributed, this stage is skipped entirely (deterministic short-circuit).

Formatting for each item:
```
Remaining plan steps:
- s3: Add hash resize logic
- s4: Add hash delete function

Unattributed failure:
[test_failure] test_lookup FAILED: key not found after insert

No existing step was identified as causing this failure.
Does this failure require adding a new implementation step? Answer yes or no.
```

The `run_binary_judgment` call uses:
- `system_prompt`: `SYSTEM_PROMPTS["adjustment_new_step"]` (new)
- `task_context`: summary of remaining steps
- `format_item`: formats a single `FailureSignal`
- `stage_name`: `"adjustment_new_step"`
- `item_key`: `lambda pair: pair[0]` (failure index)
- `default_action`: `"no_new_step"` (R2: if parse fails, don't add new step)

### 5.6 Stage 5: Focused 1.7B finalize

```python
def _finalize_adjustment(
    context: ContextPackage,
    task_description: str,
    verdicts: AdjustmentVerdicts,
    remaining_steps: list[PlanStep],
    cumulative_diff: str | None,
    llm: LoggedLLMClient,
) -> PlanAdjustment:
    """Focused 1.7B call: given binary verdicts, produce the revised step list.

    The hard judgment work is done. The model only needs to:
    1. Keep viable steps
    2. Revise steps that caused failures (based on root_causes)
    3. Add new steps for unattributed failures that need them
    4. Produce a coherent step sequence with dependencies

    Returns PlanAdjustment (same output as monolithic adjustment).
    """
```

This uses a new system prompt `SYSTEM_PROMPTS["adjustment_finalize"]` and a specialized prompt builder. The user prompt includes:

```
<adjustment_verdicts>
Step viability:
- s3 (Add hash resize logic): VALID
- s4 (Add hash delete function): INVALID -- will be dropped

Root cause attribution:
- s2 caused failure [0]: [compile_error] implicit declaration of 'hash_init'

New steps needed for:
- Failure [1]: [test_failure] test_lookup FAILED -- unattributed, needs new step
</adjustment_verdicts>

<remaining_steps>
[JSON array of remaining PlanStep objects]
</remaining_steps>

<prior_changes>
{cumulative_diff}
</prior_changes>
```

The finalize call uses `parse_plan_response(..., "adjustment")` to parse the output -- the existing parser and `PlanAdjustment.from_dict()` work unchanged.

Validation: wrap `revised_steps` in synthetic `PartPlan` and call `validate_plan()`, same as the monolithic path in `execute_plan()`.

## 6. New System Prompts

Add to `SYSTEM_PROMPTS` dict in `src/clean_room_agent/execute/prompts.py`:

```python
"adjustment_step_viability": (
    "You are Jane, a plan validity checker. You will be given a list of failures "
    "from a prior implementation step and a single remaining plan step. "
    "Determine whether the step is still valid and should be kept in the plan, "
    "or whether it should be dropped.\n\n"
    "A step is invalid if the failures make it impossible, redundant, or "
    "fundamentally wrong. A step is still valid if it can proceed despite the failures, "
    "possibly with modifications.\n\n"
    "Answer with exactly one word: \"yes\" or \"no\""
),
"adjustment_root_cause": (
    "You are Jane, a failure analyst. You will be given a specific failure message "
    "and a specific implementation step. Determine whether this step's implementation "
    "caused or directly contributed to this failure.\n\n"
    "Consider: does the step's description and target files/symbols match what the "
    "failure message is complaining about? A step causes a failure if its code changes "
    "introduced the error.\n\n"
    "Answer with exactly one word: \"yes\" or \"no\""
),
"adjustment_new_step": (
    "You are Jane, a plan gap analyst. You will be given a failure that no existing "
    "step was identified as causing, along with the remaining plan steps. "
    "Determine whether this failure requires adding a completely new implementation step "
    "to the plan.\n\n"
    "Answer yes only if the failure reveals a gap in the plan -- something not covered "
    "by any existing step. Answer no if the failure will be resolved by revising an "
    "existing step or is transient.\n\n"
    "Answer with exactly one word: \"yes\" or \"no\""
),
"adjustment_finalize": (
    "You are Jane, a plan reviser. You are given binary analysis results that tell you:\n"
    "- Which remaining steps are still valid\n"
    "- Which steps caused specific failures (and need revision)\n"
    "- Which failures need entirely new steps\n\n"
    "Your job is to produce a revised step sequence. This is a synthesis task -- "
    "the analysis is already done.\n\n"
    "Output a JSON object with exactly these fields:\n"
    "- revised_steps: array of step objects, each with:\n"
    "  - id: string -- unique identifier (reuse IDs for kept/revised steps, new IDs for new steps)\n"
    "  - description: string -- what this step accomplishes (revise descriptions for steps that caused failures)\n"
    "  - target_files: array of file paths this step will modify\n"
    "  - target_symbols: array of function/class names to modify\n"
    "  - depends_on: array of step IDs this step depends on\n"
    "- rationale: string -- why these adjustments were made\n"
    "- changes_made: array of strings -- what was changed from the original plan\n\n"
    "Rules:\n"
    "- Drop steps marked as INVALID\n"
    "- Revise descriptions of steps that caused failures to address the root cause\n"
    "- Add new steps for failures that need them\n"
    "- Cannot undo completed steps -- only revise remaining steps\n"
    "- Output only valid JSON"
),
```

## 7. Modifications to Parsers

In `src/clean_room_agent/execute/parsers.py`, add `"adjustment_finalize"` as an alias for `PlanAdjustment`:

```python
elif pass_type in ("adjustment", "adjustment_finalize"):
    return PlanAdjustment.from_dict(data)
```

## 8. Modifications to Orchestrator Runner

In `src/clean_room_agent/orchestrator/runner.py`:

### 8.1 Add config toggle function

```python
def _use_decomposed_adjustment(config: dict) -> bool:
    """Check if decomposed adjustment is enabled. Supplementary, default False."""
    orch = config.get("orchestrator", {})
    if "decomposed_adjustment" not in orch:
        logger.debug("decomposed_adjustment not in config, defaulting to False")
    return bool(orch.get("decomposed_adjustment", False))
```

### 8.2 Modify the adjustment section (lines 1121-1183)

The key change: pass `remaining_steps` explicitly and branch on config toggle.

```python
with LoggedLLMClient(reasoning_config) as adj_llm:
    if _use_decomposed_adjustment(config):
        remaining = sorted_steps[step_idx + 1:]
        adjustment = decomposed_adjustment(
            adj_context, remaining_desc, adj_llm,
            prior_results=[step_result] if step_result else None,
            remaining_steps=remaining,
            cumulative_diff=cumulative_diff or None,
        )
    else:
        adjustment = execute_plan(
            adj_context, remaining_desc, adj_llm,
            pass_type="adjustment",
            prior_results=[step_result] if step_result else None,
            cumulative_diff=cumulative_diff or None,
        )
```

## 9. What Does NOT Change

- **`PlanAdjustment` dataclass** -- output type is identical
- **`parse_plan_response("adjustment")`** -- still works for the finalize call
- **`validate_plan()` on revised_steps** -- still called
- **Downstream consumers**: `sorted_steps[step_idx + 1:] = adjustment.revised_steps` splice, session state serialization, `PassResult` recording, `adjustment_counts` tracking -- all unchanged
- **Error handling pattern**: same `except (ValueError, RuntimeError, OSError) as e:` block wraps the entire adjustment, whether monolithic or decomposed
- **`execute_plan(..., pass_type="adjustment")`** -- preserved as the monolithic fallback

## 10. Config Toggle

Config field: `orchestrator.decomposed_adjustment` (boolean)

Classification: **Supplementary** (non-core, safe fallback to monolithic). Default `false`. Same classification as `decomposed_planning` and `decomposed_scaffold`.

## 11. Per-Task Adapter Map Updates

Update `protocols/design_records/per_task_adapter_map.md`:

- Move A20 from Pattern 4 (single call) to a composite entry
- Add new binary sub-tasks to Pattern 1 table:
  - A20a: **Step viability** -- `adjustment_step_viability` system prompt
  - A20b: **Root cause attribution** -- `adjustment_root_cause` system prompt
  - A20c: **New step detection** -- `adjustment_new_step` system prompt
- Add new finalize task to Pattern 4 table:
  - A20d: **Adjustment finalize** -- `adjustment_finalize` system prompt (reduced complexity vs. A20)
- Update Decomposition Status table to include A20

## 12. Call Volume Analysis

Per adjustment pass, with F failures and S remaining steps:
- Stage 2 (viability): S calls
- Stage 3 (root cause): F * S calls (worst case); F * (viable steps) calls (after stage 2 filtering)
- Stage 4 (new step): 0 to F calls (only unattributed failures)
- Stage 5 (finalize): 1 call

Typical scenario: 1 failure, 3 remaining steps = 3 + 3 + 0-1 + 1 = 7-8 binary calls + 1 finalize. The binary calls are cheap (0.6B candidates) and independent (could be parallelized in Phase 4 with vLLM).

Worst case: 3 failures, 10 remaining steps = 10 + 30 + 0-3 + 1 = 41-44 calls. Still manageable for 0.6B at ~100ms per call.

## 13. Edge Cases and Design Decisions

**Edge case: No failures.** If `prior_results` is None or all steps succeeded, `extract_failure_signals()` returns empty. When failures are empty, all binary stages short-circuit: viability returns all True, root cause returns empty, new step detection is skipped. The function can return early with:
```python
PlanAdjustment(
    revised_steps=list(remaining_steps),
    rationale="No failures detected -- steps unchanged",
    changes_made=[],
)
```

**Edge case: No remaining steps.** If `remaining_steps` is empty, return immediately with empty `PlanAdjustment`. No LLM calls.

**Edge case: Binary parse failures.** For viability, R2 default-deny means the step is dropped (conservative). For root cause, R2 means "not attributed" (conservative -- may trigger new step detection). For new step detection, R2 means "no new step needed" (conservative). If ALL viability judgments fail to parse (all omitted), that is a structural failure -- raise ValueError to match the fail-fast pattern.

**Budget concern: cumulative_diff in root cause prompts.** The cumulative_diff can be large (up to 50K chars). For root cause attribution, the diff is relevant but may blow the 0.6B context window. Solution: truncate diff to the most recent 2K chars per binary call. The stage 5 finalize call (1.7B, 32K window) gets the full diff.

```python
_ROOT_CAUSE_DIFF_CAP_CHARS = 2000  # ~500 tokens for 0.6B binary prompt
```

## 14. Test Plan

New test file: `tests/execute/test_decomposed_adjustment.py`

### Unit tests for `extract_failure_signals()`

```
test_extract_no_failures_returns_empty
test_extract_success_result_returns_empty
test_extract_compile_error_categorization
test_extract_test_failure_categorization
test_extract_patch_failure_categorization
test_extract_unknown_failure_categorization
test_extract_multiple_failures
test_extract_truncates_long_messages
test_extract_none_prior_results_returns_empty
```

### Unit tests for binary stages (mock LLM)

```
test_step_viability_all_valid
test_step_viability_some_invalid
test_step_viability_all_invalid_raises
test_step_viability_empty_steps_returns_empty
test_step_viability_no_failures_all_valid

test_root_cause_single_failure_single_step
test_root_cause_multi_failure_multi_step
test_root_cause_no_attribution
test_root_cause_empty_failures_returns_empty
test_root_cause_skips_invalid_steps

test_new_step_needed
test_new_step_not_needed
test_new_step_skipped_when_all_attributed
test_new_step_empty_unattributed_returns_empty
```

### Unit tests for `_finalize_adjustment()`

```
test_finalize_produces_valid_plan_adjustment
test_finalize_validates_revised_steps_no_cycles
test_finalize_validates_revised_steps_no_duplicates
test_finalize_empty_revised_steps_ok
test_finalize_budget_validation
```

### Integration tests for `decomposed_adjustment()`

```
test_decomposed_adjustment_full_flow
test_decomposed_adjustment_no_failures_returns_unchanged
test_decomposed_adjustment_all_steps_invalid
test_decomposed_adjustment_with_new_steps
test_decomposed_adjustment_matches_plan_adjustment_type
```

### Tests for config toggle in runner

```
test_use_decomposed_adjustment_default_false
test_use_decomposed_adjustment_enabled
test_adjustment_uses_decomposed_when_enabled
test_adjustment_uses_monolithic_when_disabled
```

### Tests for new system prompts

```
test_adjustment_prompts_in_system_prompts_dict
test_build_adjustment_finalize_prompt_budget_validation
test_build_adjustment_finalize_prompt_includes_verdicts
```

Expected test count: approximately 30-35 new tests.

## 15. Implementation Sequence

1. **Add dataclasses** (`FailureSignal`, `AdjustmentVerdicts`) to `src/clean_room_agent/execute/dataclasses.py`
2. **Add system prompts** (4 new entries) to `src/clean_room_agent/execute/prompts.py`
3. **Update parser** (add `"adjustment_finalize"` case) in `src/clean_room_agent/execute/parsers.py`
4. **Create `decomposed_adjustment.py`** with all 5 stage functions + public entry point
5. **Add config toggle** (`_use_decomposed_adjustment`) and branching logic to `src/clean_room_agent/orchestrator/runner.py`
6. **Write tests** in `tests/execute/test_decomposed_adjustment.py`
7. **Update adapter map** in `protocols/design_records/per_task_adapter_map.md`

## Critical Files for Implementation

- `src/clean_room_agent/execute/decomposed_adjustment.py` - New file: core decomposition logic (5 stage functions + public entry point)
- `src/clean_room_agent/execute/dataclasses.py` - Add FailureSignal and AdjustmentVerdicts dataclasses
- `src/clean_room_agent/execute/prompts.py` - Add 4 new system prompts
- `src/clean_room_agent/orchestrator/runner.py` - Config toggle and decomposed branch in adjustment section
- `src/clean_room_agent/execute/decomposed_plan.py` - Reference pattern: follow the exact structure
