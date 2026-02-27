# Implementation Plan: Binary Decomposition and Model Tier Architecture

Design record: `protocols/design_records/binary_decomposition_and_model_tiers.md`

## Context

The current retrieval pipeline batches multiple candidates into single LLM calls (`run_batched_judgment()`) and asks for a structured JSON array response classifying all of them. Three stages use this pattern: scope, precision, and similarity. All retrieval stages currently route through the "reasoning" role (one model for all).

The design record argues for two orthogonal changes:

1. **Binary decomposition** (architectural): every judgment that IS binary gets restructured into independent per-candidate calls. This is not optional. If the decision is yes/no, the code calls `run_binary_judgment()`. Period.
2. **Model tiers** (config): routing different call types to different-sized models (0.6B/1.7B/4B). This uses the existing `[models.overrides]` mechanism. No new roles, no runtime negotiation.

These are separate concerns. Binary decomposition is an architectural fact about how decisions are structured. Model routing is a deployment decision in config.toml.

## Part 1: Binary Decomposition (Architectural)

### Which judgments are binary?

This is a static classification. A judgment's type is a property of the decision being made, not of the model or config.

| Judgment | Question | Binary? | Rationale |
|----------|----------|---------|-----------|
| Scope | "Is this file relevant to the task?" | **Yes** | 1 bit: relevant or not |
| Similarity | "Are these two files related?" | **Yes** | 1 bit: related or not |
| Precision pass 1 | "Is this symbol relevant to the task?" | **Yes** | 1 bit: relevant or excluded |
| Precision pass 2 | "Is this symbol directly involved in the change?" | **Yes** | 1 bit: primary or not |
| Precision pass 3 | "Does this symbol need full source or just signatures?" | **Yes** | 1 bit: supporting or type_context |
| Routing | "Which stages should run?" | **No** | Set selection from available stages |
| Assembly R1 re-filter | "Which files to keep within budget?" | **No** | Constrained subset selection — budget creates interdependence between decisions |

Binary judgments call `run_binary_judgment()`. Non-binary judgments keep their current call patterns. This is hardcoded per call site — no flags, no negotiation, no runtime detection.

**Assembly R1 re-filter is NOT binary** despite appearing to be keep/drop per file. The decision on file A depends on whether file B is kept because of the budget constraint. This is a constrained optimization problem, not N independent decisions. Keep it as a single batched call.

### `run_binary_judgment()` in `batch_judgment.py`

Each candidate gets an independent LLM call. One input, one bit output.

```python
def run_binary_judgment(
    items: Sequence,
    *,
    system_prompt: str,
    task_context: str,          # Task description (shared across all calls)
    llm,
    format_item: Callable,      # (item) -> str for one candidate
    extract_key: Callable,      # (item) -> Hashable
    stage_name: str,
    default_action: str = "excluded",
) -> tuple[dict[Hashable, bool], set[Hashable]]:
    """Independent binary judgment per item. One LLM call per candidate.

    Returns:
        (result_map: key -> True/False, omitted_keys: items that failed to parse)
    """
```

Per-call flow:
1. Build prompt: `task_context + "\n" + format_item(item)`
2. `validate_prompt_budget()` (R3 — trivially satisfied for binary calls)
3. `llm.complete(prompt, system=system_prompt)`
4. Parse response: strip whitespace, lowercase, match against `("yes", "no")`
5. "yes" → True, "no" → False, anything else → R2 default-deny (False), log warning

Key properties:
- One LLM call per item. No batching.
- Sequential for now. Each call is independent — parallelism is a future optimization, not an architectural concern.
- R2 default-deny: unparseable response → False. Logged.
- R3 budget validation: per call. Trivially satisfied when context is task + one candidate.
- Every call logged individually in raw DB. Each binary decision is a separate audit record — one input, one output, one verdict. The atomic unit of auditability.

### Scope stage — direct replacement

`judge_scope()` currently calls `run_batched_judgment()`. Replace with `run_binary_judgment()`. No flag, no conditional:

```python
def judge_scope(candidates, task, llm, kb=None):
    ...
    # Seeds skip judgment (unchanged)
    seeds = [sf for sf in candidates if sf.tier <= 1]
    non_seeds = [sf for sf in candidates if sf.tier > 1]
    ...
    # Binary judgment — one call per candidate
    verdict_map, omitted = run_binary_judgment(
        non_seeds,
        system_prompt=SCOPE_BINARY_SYSTEM,
        task_context=task_header,
        llm=llm,
        format_item=_format,
        extract_key=lambda sf: sf.path,
        stage_name="scope",
        default_action="irrelevant",
    )

    for sf in non_seeds:
        sf.relevance = "relevant" if verdict_map.get(sf.path, False) else "irrelevant"
    ...
```

Binary system prompt:

```
You are a code relevance judge. Given a task and one candidate file,
determine if the file is relevant to the task.
Respond with ONLY "yes" or "no".
```

### Similarity stage — same replacement

Each file pair gets an independent "are these related?" call. Replace `run_batched_judgment()` with `run_binary_judgment()`.

### Precision stage — 3-pass binary cascade

The current 4-class precision decision decomposes into 3 sequential binary passes, each filtering the input for the next.

**Current**: one batched call per chunk → 4-class JSON array `{detail_level: "primary"|"supporting"|"type_context"|"excluded"}`

**Decomposed**:

| Pass | Question | yes → | no → | Input set |
|------|----------|-------|------|-----------|
| 1 | "Is this symbol relevant to the task?" | proceed to pass 2 | **excluded** | All project symbols |
| 2 | "Is this symbol directly involved in the change?" | **primary** | proceed to pass 3 | Symbols that passed pass 1 |
| 3 | "Does this symbol need full source code or just its signature?" | **supporting** | **type_context** | Non-primary symbols from pass 2 |

The cascade is volume-reducing:
- Pass 1 eliminates most symbols (the bulk are excluded) — runs on all project symbols
- Pass 2 runs on the smaller relevant set — separates primary from the rest
- Pass 3 runs on the smallest set — non-primary relevant symbols only

Each pass is binary, independently auditable, produces clean training pairs. The three passes map to concrete rendering boundaries:
- Pass 1: render anything at all?
- Pass 2: render full source (primary priority)?
- Pass 3: render full source (supporting priority) or signatures only (type_context)?

Library symbols still skip all LLM judgment — auto-classified as type_context (R17 unchanged).

**Implementation in `classify_symbols()`**:

```python
def classify_symbols(candidates, task, llm, kb=None):
    ...
    # R17: library symbols skip LLM (unchanged)
    project_candidates = [c for c in candidates if c["file_source"] != "library"]
    ...

    # Pass 1: relevant or excluded?
    pass1_map, _ = run_binary_judgment(
        project_candidates,
        system_prompt=PRECISION_PASS1_SYSTEM,  # "Is this symbol relevant?"
        task_context=task_header,
        llm=llm,
        format_item=_format,
        extract_key=_symbol_key,
        stage_name="precision_pass1",
    )
    relevant = [c for c in project_candidates if pass1_map.get(_symbol_key(c), False)]
    excluded = [c for c in project_candidates if not pass1_map.get(_symbol_key(c), False)]

    # Pass 2: primary or not? (only relevant symbols)
    pass2_map, _ = run_binary_judgment(
        relevant,
        system_prompt=PRECISION_PASS2_SYSTEM,  # "Is this symbol directly involved in the change?"
        task_context=task_header,
        llm=llm,
        format_item=_format,
        extract_key=_symbol_key,
        stage_name="precision_pass2",
    )
    primary = [c for c in relevant if pass2_map.get(_symbol_key(c), False)]
    non_primary = [c for c in relevant if not pass2_map.get(_symbol_key(c), False)]

    # Pass 3: supporting or type_context? (only non-primary relevant symbols)
    pass3_map, _ = run_binary_judgment(
        non_primary,
        system_prompt=PRECISION_PASS3_SYSTEM,  # "Does this symbol need full source or just signatures?"
        task_context=task_header,
        llm=llm,
        format_item=_format,
        extract_key=_symbol_key,
        stage_name="precision_pass3",
    )
    # pass3: yes → supporting (full source), no → type_context (signatures only)
    ...
```

Three system prompts for precision:

```
# Pass 1
You are a code relevance judge. Given a task and one code symbol,
determine if this symbol is relevant to the task.
Respond with ONLY "yes" or "no".

# Pass 2
You are a code relevance judge. Given a task and one code symbol
that is relevant to the task, determine if this symbol is directly
involved in the change (will be modified or is central to the logic).
Respond with ONLY "yes" or "no".

# Pass 3
You are a code relevance judge. Given a task and one code symbol
that provides context for the change, determine if the full source
code is needed (yes) or if just the signature/type definition is
sufficient (no).
Respond with ONLY "yes" or "no".
```

### What `run_batched_judgment()` returns vs `run_binary_judgment()`

The return type changes from `dict[key, dict]` to `dict[key, bool]`. Call sites that previously read `verdict_map[key].get("verdict")` now read `verdict_map.get(key, False)`. Direct refactor at each call site.

The `reason` field from batched JSON responses is gone for binary calls. This is correct: the reason is the input (the candidate metadata). For audit, the input + verdict is sufficient. Self-reported rationale from a small model is noise, not signal.

### `run_batched_judgment()` — kept only for non-binary judgments

After this change, `run_batched_judgment()` is used only by:
- Routing (set selection)
- Assembly R1 re-filter (constrained subset selection)

These are genuinely non-binary: the decision on one item depends on other items (routing selects a compatible set of stages; re-filter selects within a budget). Independent per-item calls cannot express these constraints.

## Part 2: Model Tiers (Config)

### No new roles in ModelRouter

The existing `[models.overrides]` mechanism already supports per-stage model routing. To route binary stages to 0.6B:

```toml
[models.overrides]
scope = {model = "qwen3:0.6b", context_window = 8192, max_tokens = 16}
similarity = {model = "qwen3:0.6b", context_window = 8192, max_tokens = 16}
precision = {model = "qwen3:0.6b", context_window = 8192, max_tokens = 16}
```

This is already implemented in `router.py` lines 91-115. Override can be a string (model tag) or dict with `model` and optional `context_window`. No code changes needed for per-stage routing.

Note: all three precision passes use the same LLM client (resolved once for the "precision" stage). The 3-pass cascade runs within a single stage execution — the pipeline sees one stage, not three. This is correct: the decomposition is internal to precision, not a pipeline-level concern.

### Config template update

Update `create_default_config()` in `config.py` to show the 3-tier model tags and document the override pattern:

```toml
[models]
provider = "ollama"
coding = "qwen3:1.7b"
reasoning = "qwen3:4b"
base_url = "http://localhost:11434"
context_window = 32768

[models.overrides]
# Binary decision stages — route to smallest capable model
# scope = {model = "qwen3:0.6b", context_window = 8192, max_tokens = 16}
# similarity = {model = "qwen3:0.6b", context_window = 8192, max_tokens = 16}
# precision = {model = "qwen3:0.6b", context_window = 8192, max_tokens = 16}
```

The model tier cascade (0.6B → 1.7B → 4B) is a deployment recommendation in config, not an architectural constraint in code. A user could route precision to 4B — binary decomposition still applies, just with a bigger model.

### ModelRouter — one small addition

The `_VALID_ROLES` stays `("coding", "reasoning")`. No "classifier" role. The router already resolves per-stage overrides before falling back to role defaults.

One addition: `max_tokens` support in override dicts (currently only `model` and `context_window` are supported):

```python
elif isinstance(override, dict):
    model_tag = override["model"]
    cw = override.get("context_window", self._context_window[role])
    mt = override.get("max_tokens", self._max_tokens[role])  # NEW
```

Binary calls need `max_tokens = 16` to constrain output length.

## Files to Modify

| File | Change |
|------|--------|
| `src/clean_room_agent/retrieval/batch_judgment.py` | Add `run_binary_judgment()` |
| `src/clean_room_agent/retrieval/scope_stage.py` | Replace `run_batched_judgment` → `run_binary_judgment`, new binary system prompt |
| `src/clean_room_agent/retrieval/similarity_stage.py` | Replace `run_batched_judgment` → `run_binary_judgment`, new binary system prompt |
| `src/clean_room_agent/retrieval/precision_stage.py` | Replace single batched call → 3-pass binary cascade, 3 new system prompts |
| `src/clean_room_agent/llm/router.py` | Add `max_tokens` support in per-stage override dicts |
| `src/clean_room_agent/config.py` | Update default config template with 3-tier override examples |
| `planning/cli-and-config.md` | Document override dict `max_tokens` field |

## Files NOT Modified

- `src/clean_room_agent/retrieval/context_assembly.py` — R1 re-filter stays batched (budget-constrained subset selection, not binary)
- `src/clean_room_agent/retrieval/stage.py` — no protocol changes
- `src/clean_room_agent/retrieval/pipeline.py` — no changes (stages still resolve as "reasoning" + per-stage override; precision cascade is internal to the stage)
- `src/clean_room_agent/execute/` — no execute-stage changes
- `src/clean_room_agent/orchestrator/` — no orchestrator changes
- `src/clean_room_agent/db/` — no schema changes (raw DB already logs model per call; 3 precision passes log as 3 separate stage_names: precision_pass1, precision_pass2, precision_pass3)

## Implementation Order

1. **`run_binary_judgment()`** — new function in batch_judgment.py, with unit tests
2. **Scope stage** — replace batched → binary, new system prompt, update tests
3. **Similarity stage** — same replacement, update tests
4. **Precision stage** — replace batched → 3-pass binary cascade, 3 new system prompts, update tests
5. **Router `max_tokens` in overrides** — small addition, test
6. **Config template** — update default config with 3-tier override examples
7. **Docs** — update cli-and-config.md

## Worked Examples (Pseudocode)

### Example task

```
Task: "Fix the off-by-one error in the token budget calculation that causes
       the last file to be silently dropped from the context window"
Intent: "Bug fix in token budget tracking logic"
Keywords: ["budget", "token", "off-by-one", "context_window"]
Mentioned symbols: ["BudgetTracker", "can_fit"]
```

### Scope: 5 candidates, 2 seeds, 3 non-seeds

```
Input candidates (from tiered expansion):
  [seed]     tier=0  budget.py              (plan artifact)
  [seed]     tier=1  token_estimation.py     (contains BudgetTracker)
  [non-seed] tier=2  context_assembly.py     (imports budget.py)
  [non-seed] tier=3  pipeline.py             (co-changed with budget.py)
  [non-seed] tier=4  scope_stage.py          (metadata match: "budget")

Seeds skip judgment → relevance="relevant", reason="seed (always included)"

Binary calls for 3 non-seeds (sequential, one call each):

--- Call 1 ---
SYSTEM: You are a code relevance judge. Given a task and one candidate file,
        determine if the file is relevant to the task.
        Respond with ONLY "yes" or "no".

USER:   Task: Fix the off-by-one error in the token budget calculation...
        Intent: Bug fix in token budget tracking logic

        File: context_assembly.py (tier=2, language=python, reason=imports budget.py)
        [purpose=assembles context from classified files, domain=retrieval,
         concepts=budget tracking, file rendering]

LLM:    yes

→ verdict_map["context_assembly.py"] = True
→ raw DB: {stage: "scope", input: <prompt>, output: "yes", model: "qwen3:0.6b"}

--- Call 2 ---
USER:   ...
        File: pipeline.py (tier=3, language=python, reason=co-changed with budget.py 4 times)
        [purpose=orchestrates retrieval stages, domain=retrieval,
         concepts=stage sequencing, budget passing]

LLM:    yes

→ verdict_map["pipeline.py"] = True

--- Call 3 ---
USER:   ...
        File: scope_stage.py (tier=4, language=python, reason=metadata match "budget")
        [purpose=scope expansion and judgment, domain=retrieval,
         concepts=tiered expansion, file relevance]

LLM:    no

→ verdict_map["scope_stage.py"] = False

Result:
  budget.py            → relevant (seed)
  token_estimation.py  → relevant (seed)
  context_assembly.py  → relevant (binary: yes)
  pipeline.py          → relevant (binary: yes)
  scope_stage.py       → irrelevant (binary: no)

Total LLM calls: 3 (one per non-seed)
Each logged independently. Each auditable: input + verdict.
```

### Similarity: 3 file pairs

```
Input pairs (from co-change / import overlap):
  (budget.py, token_estimation.py)
  (budget.py, context_assembly.py)
  (token_estimation.py, scope_stage.py)

--- Call 1 ---
SYSTEM: You are a code relationship judge. Given a task and two files,
        determine if these files are related in the context of this task.
        Respond with ONLY "yes" or "no".

USER:   Task: Fix the off-by-one error in the token budget calculation...

        File A: budget.py — BudgetTracker, can_fit(), remaining property
        File B: token_estimation.py — estimate_tokens(), CHARS_PER_TOKEN constants

LLM:    yes

--- Call 2 ---
USER:   ...
        File A: budget.py — BudgetTracker, can_fit(), remaining property
        File B: context_assembly.py — assemble_context(), _refilter_files()

LLM:    yes

--- Call 3 ---
USER:   ...
        File A: token_estimation.py — estimate_tokens(), CHARS_PER_TOKEN constants
        File B: scope_stage.py — expand_scope(), judge_scope()

LLM:    no

Result: groups {budget.py, token_estimation.py, context_assembly.py}, {scope_stage.py}
Total LLM calls: 3
```

### Precision: 3-pass cascade on 8 symbols

```
Input: 8 project symbols from the 4 relevant files, plus 2 library symbols.

Symbols:
  budget.py:         BudgetTracker (class), can_fit (method), remaining (property)
  token_estimation.py: estimate_tokens (func), estimate_tokens_conservative (func),
                       CHARS_PER_TOKEN (const)
  context_assembly.py: assemble_context (func), _refilter_files (func)
  [library]:         httpx.Client (class), sqlite3.Connection (class)

Library symbols → auto type_context (R17, no LLM calls)

=== PASS 1: "Is this symbol relevant to the task?" ===
8 binary calls (one per project symbol)

--- Call 1/8 ---
SYSTEM: You are a code relevance judge. Given a task and one code symbol,
        determine if this symbol is relevant to the task.
        Respond with ONLY "yes" or "no".

USER:   Task: Fix the off-by-one error in the token budget calculation...
        Intent: Bug fix in token budget tracking logic

        Symbol: BudgetTracker (class) in budget.py:15-89
        [calls estimate_tokens, called by assemble_context]
        sig: class BudgetTracker
        doc: Tracks token budget consumption during context assembly

LLM:    yes

--- Call 2/8: can_fit (method) ---     LLM: yes
--- Call 3/8: remaining (property) --- LLM: yes
--- Call 4/8: estimate_tokens ---      LLM: yes
--- Call 5/8: estimate_tokens_conservative --- LLM: yes
--- Call 6/8: CHARS_PER_TOKEN ---      LLM: yes
--- Call 7/8: assemble_context ---     LLM: yes
--- Call 8/8: _refilter_files ---      LLM: no

Pass 1 result: 7 relevant, 1 excluded (_refilter_files)
Pass 1 calls: 8

=== PASS 2: "Is this symbol directly involved in the change?" ===
7 binary calls (only symbols that passed pass 1)

--- Call 1/7 ---
SYSTEM: You are a code relevance judge. Given a task and one code symbol
        that is relevant to the task, determine if this symbol is directly
        involved in the change (will be modified or is central to the logic).
        Respond with ONLY "yes" or "no".

USER:   Task: Fix the off-by-one error in the token budget calculation...

        Symbol: BudgetTracker (class) in budget.py:15-89
        sig: class BudgetTracker
        doc: Tracks token budget consumption during context assembly

LLM:    yes

--- Call 2/7: can_fit ---              LLM: yes   (the off-by-one is likely here)
--- Call 3/7: remaining ---            LLM: yes
--- Call 4/7: estimate_tokens ---      LLM: no    (called by BudgetTracker, not the bug itself)
--- Call 5/7: estimate_tokens_conservative --- LLM: no
--- Call 6/7: CHARS_PER_TOKEN ---      LLM: no
--- Call 7/7: assemble_context ---     LLM: no

Pass 2 result: 3 primary, 4 non-primary
Pass 2 calls: 7

=== PASS 3: "Does this symbol need full source or just signatures?" ===
4 binary calls (only non-primary relevant symbols)

--- Call 1/4 ---
SYSTEM: You are a code relevance judge. Given a task and one code symbol
        that provides context for the change, determine if the full source
        code is needed (yes) or if just the signature/type definition is
        sufficient (no).
        Respond with ONLY "yes" or "no".

USER:   Task: Fix the off-by-one error in the token budget calculation...

        Symbol: estimate_tokens (func) in token_estimation.py:18-25
        sig: def estimate_tokens(text: str) -> int
        doc: Estimate token count from character count

LLM:    yes    (need to see the implementation to verify the constant used)

--- Call 2/4: estimate_tokens_conservative --- LLM: yes
--- Call 3/4: CHARS_PER_TOKEN ---              LLM: no   (it's a constant — signature is the value)
--- Call 4/4: assemble_context ---             LLM: no   (just need to see how it calls BudgetTracker)

Pass 3 result: 2 supporting, 2 type_context
Pass 3 calls: 4

=== FINAL CLASSIFICATION ===

  BudgetTracker              → primary        (pass 2: yes)
  can_fit                    → primary        (pass 2: yes)
  remaining                  → primary        (pass 2: yes)
  estimate_tokens            → supporting     (pass 2: no, pass 3: yes)
  estimate_tokens_conservative → supporting   (pass 2: no, pass 3: yes)
  CHARS_PER_TOKEN            → type_context   (pass 2: no, pass 3: no)
  assemble_context           → type_context   (pass 2: no, pass 3: no)
  _refilter_files            → excluded       (pass 1: no)
  httpx.Client               → type_context   (R17: library, no LLM)
  sqlite3.Connection         → type_context   (R17: library, no LLM)

Total precision LLM calls: 8 + 7 + 4 = 19
Volume reduction: 8 → 7 → 4 (cascade filters at each pass)

Compare batched: would have been 1-2 LLM calls (8 symbols batched).
But: 19 binary calls to a 0.6B are cheaper and faster than 2 batched calls to a 4B.
And: each of the 19 decisions is independently auditable and produces a clean training pair.
```

### Audit trail comparison

```
BATCHED (current):
  raw DB row 1: {stage: "precision", prompt: <2000 tokens of 8 symbols>,
                 response: <JSON array of 8 classifications>, model: "qwen3:4b"}
  → Which classification was wrong? Was the batch size too large? Did ordering matter?
  → Cannot attribute error to a specific symbol without re-running in isolation.

BINARY (new):
  raw DB row 1:  {stage: "precision_pass1", input: "BudgetTracker...", output: "yes"}
  raw DB row 2:  {stage: "precision_pass1", input: "can_fit...",       output: "yes"}
  ...
  raw DB row 8:  {stage: "precision_pass1", input: "_refilter_files...", output: "no"}
  raw DB row 9:  {stage: "precision_pass2", input: "BudgetTracker...", output: "yes"}
  ...
  raw DB row 19: {stage: "precision_pass3", input: "assemble_context...", output: "no"}

  → Wrong classification on can_fit? Row 2 (pass1) + row 10 (pass2) tell you exactly
    what the model saw and what it decided. Flip the label → training pair.
  → No batch ordering effects. No cross-candidate contamination. No ambiguity.
```

## Verification

- `pytest tests/` must pass (binary is now the default for all judgment stages — no fallback)
- Unit tests for `run_binary_judgment()`: mock LLM returns "yes"/"no"/garbage, verify results + R2 logging
- Unit tests for scope stage: verify each candidate gets an independent call (mock LLM call count == candidate count)
- Unit tests for precision: verify 3-pass cascade — pass 1 filters, pass 2 splits primary, pass 3 splits supporting/type_context. Verify call counts decrease across passes (volume-reducing). Verify library symbols skip all 3 passes (R17).
- Unit tests for router: override dict with `max_tokens` resolves correctly
- Manual: run `cra plan` with trace, verify:
  - Scope stage: N individual LLM calls (not batched), each logged separately in raw DB
  - Precision stage: 3 groups of calls logged as precision_pass1/pass2/pass3, each group smaller than the last
  - Each binary decision traceable: input → verdict → outcome
