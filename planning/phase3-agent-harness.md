# Phase 3: Agent Harness + Benchmark

## Context

This is the execution layer of the Clean Room Agent. It takes the curated context package from Phase 2 and feeds it to an LLM to generate code changes. It also provides the naive-context baseline for A/B comparison, the validation loop, and the SWE-bench integration to prove the thesis.

Phase 1 indexes. Phase 2 retrieves. Phase 3 executes and measures.

The gate for Phase 3: run the stress test matrix on SWE-bench Verified and produce a comparison table showing that context curation measurably outperforms naive context with the same model.

---

## Architecture

```
                     ┌──────────────────────────┐
                     │     Benchmark Runner      │
                     │  (SWE-bench instances)    │
                     └────────────┬─────────────┘
                                  │ for each instance × config
                                  ▼
               ┌─────────────────────────────────┐
               │         Agent Harness            │
               │                                  │
               │  ┌───────────┐  ┌─────────────┐ │
               │  │  Pipeline  │  │   Baseline   │ │
               │  │  (Ph1+Ph2) │  │   (Naive)    │ │
               │  └─────┬─────┘  └──────┬──────┘ │
               │        │    context     │        │
               │        └──────┬─────────┘        │
               │               ▼                  │
               │  ┌─────────────────────────┐     │
               │  │    Prompt Builder        │     │
               │  │  context + task → prompt │     │
               │  └────────────┬────────────┘     │
               │               ▼                  │
               │  ┌─────────────────────────┐     │
               │  │      LLM Client          │     │
               │  │  Ollama / Anthropic / OAI │     │
               │  └────────────┬────────────┘     │
               │               ▼                  │
               │  ┌─────────────────────────┐     │
               │  │   Response Parser        │     │
               │  │  LLM output → patch      │     │
               │  └────────────┬────────────┘     │
               │               ▼                  │
               │  ┌─────────────────────────┐     │
               │  │   Patch Application      │     │
               │  │  Apply to worktree       │     │
               │  └────────────┬────────────┘     │
               │               ▼                  │
               │  ┌─────────────────────────┐     │
               │  │     Validation           │     │
               │  │  Tests / lint / type     │     │
               │  └────────────┬────────────┘     │
               │               │                  │
               │        pass?  │  fail?           │
               │          ✓    │  → retry with    │
               │               │    error context  │
               │               ▼                  │
               │          Patch (diff)            │
               └──────────────┬───────────────────┘
                              │
                              ▼
               ┌─────────────────────────────────┐
               │      Evaluation + Reporting      │
               │  SWE-bench eval, comparison      │
               └─────────────────────────────────┘
```

**Two context modes, same execution path.** The only difference between Config A/C (naive) and Config B/D (pipeline) is how context is constructed. Everything downstream — prompt building, LLM call, response parsing, patch application, validation — is identical. This ensures the comparison is fair.

**No conversation accumulation.** Every attempt starts with a fresh prompt. Retries include the previous diff and error output as structured additions, not as conversation history. This is the N-prompt philosophy applied to the retry loop.

---

## Project Structure (additions to Phases 1-2)

```
src/
  clean_room/
    agent/
      __init__.py
      harness.py                  # Top-level agent orchestrator
      prompt_builder.py           # Context + task → model-ready prompt
      llm_client.py               # Unified LLM interface
      response_parser.py          # LLM output → structured edits
      patch.py                    # Apply edits to files
      validation.py               # Run tests, lint, type check
      baseline.py                 # Naive full-context construction
      retry.py                    # Retry logic with error feedback
      dataclasses.py              # Phase 3 data structures
    bench/
      __init__.py
      swe_bench.py                # SWE-bench instance loading + adaptation
      runner.py                   # Benchmark runner (all configs)
      metrics.py                  # Score collection + comparison
      report.py                   # Generate comparison tables/reports
      configs.py                  # Stress test matrix definitions
tests/
  fixtures/
    sample_llm_responses.py       # Canned LLM outputs for parser tests
    sample_patches/               # Known-good patches for application tests
  test_prompt_builder.py
  test_llm_client.py
  test_response_parser.py
  test_patch.py
  test_validation.py
  test_harness.py
  test_baseline.py
  test_swe_bench.py
  test_runner.py
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM client | `httpx` for Ollama, `anthropic`/`openai` SDKs for API models | Local path must work with zero API keys. API SDKs only for reference configs. |
| Output format | Search-and-replace blocks (primary), unified diff (fallback) | S&R blocks are more natural for small models — less formatting overhead, fewer off-by-one errors. Convert to unified diff for SWE-bench submission. |
| Patch isolation | `git worktree` per attempt | Clean rollback on failure, no risk of corrupting the main checkout. Cheap to create/destroy. |
| Retry strategy | Fresh prompt with error context, max 3 attempts | No conversation history. Each attempt gets the full curated context + the previous error. Bounded to prevent runaway token spend. |
| SWE-bench eval | `swebench` Python package for instance loading + Docker evaluation | Standard evaluation infrastructure. Don't reinvent test execution in containers. |
| Baseline approach | Structured naive — files mentioned in issue first, then by directory proximity, truncated to budget | Fair comparison: even the baseline uses basic intelligence about file ordering. Not deliberately sabotaged. |
| Temperature | 0.0 for deterministic comparison, 0.2 for production use | Reproducible benchmarks need deterministic output. Slightly higher temp in production for creativity. |

---

## Data Structures

```python
@dataclass
class LLMConfig:
    """Model configuration for a benchmark run."""
    provider: str                     # "ollama", "anthropic", "openai"
    model: str                        # "qwen2.5-coder:3b", "claude-sonnet-4-20250514", etc.
    temperature: float = 0.0
    max_tokens: int = 4096
    base_url: str | None = None       # Override for Ollama (default: localhost:11434)

@dataclass
class EditBlock:
    """A single code edit extracted from LLM output."""
    file_path: str
    search: str                       # Text to find (for S&R format)
    replace: str                      # Text to replace with
    # Alternative: full unified diff hunk
    diff_hunk: str | None = None

@dataclass
class GenerationResult:
    """Output from a single LLM generation attempt."""
    edits: list[EditBlock]
    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    attempt: int                      # 1-indexed

@dataclass
class ValidationResult:
    """Output from running validation on applied changes."""
    success: bool
    test_output: str | None           # stdout/stderr from test runner
    lint_output: str | None
    type_check_output: str | None
    failing_tests: list[str]          # Names of failing tests

@dataclass
class AttemptRecord:
    """Full record of one generate→apply→validate cycle."""
    attempt: int
    generation: GenerationResult
    patch_applied: bool
    validation: ValidationResult | None   # None if patch failed to apply
    unified_diff: str                     # The diff that was applied (or attempted)

@dataclass
class TaskResult:
    """Full result of the agent solving one task."""
    task_id: str
    config_name: str                  # "A", "B", "C", "D"
    success: bool                     # Final validation passed
    attempts: list[AttemptRecord]
    total_tokens: int                 # Sum across all attempts
    total_latency_ms: int
    final_diff: str | None            # The successful diff, or last attempted

@dataclass
class BenchmarkResult:
    """Aggregate results for one config across all tasks."""
    config_name: str
    model: str
    context_strategy: str             # "naive" or "pipeline"
    tasks: list[TaskResult]
    pass_rate: float                  # % of tasks where success=True
    avg_tokens: float
    avg_attempts: float
    avg_latency_ms: float
```

---

## Implementation Steps

### Step 1: LLM Client

**Delivers**: Unified interface for sending prompts and receiving completions from local and API models.

**Files**:
- `src/clean_room/agent/llm_client.py` — `LLMClient` class
- `src/clean_room/agent/dataclasses.py` — `LLMConfig`, `GenerationResult`
- `tests/test_llm_client.py`

**Interface**:
```python
class LLMClient:
    def __init__(self, config: LLMConfig): ...

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        """Send prompt, return completion with usage stats."""
        ...
```

**Provider implementations**:

1. **Ollama** (primary, local): `POST http://localhost:11434/api/generate` via httpx. Supports both `/api/generate` (completion) and `/api/chat` (chat) endpoints. Use `/api/chat` for chat-tuned models (Qwen-2.5-Coder is chat-tuned). Request format:
   ```json
   {
     "model": "qwen2.5-coder:3b",
     "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
     "stream": false,
     "options": {"temperature": 0.0, "num_predict": 4096}
   }
   ```
   Parse `eval_count` and `prompt_eval_count` from response for token counts. Detect Ollama availability on init — clear error if not running.

2. **Anthropic** (reference): `anthropic` SDK. `client.messages.create(model=..., max_tokens=..., system=..., messages=[...])`. Extract `usage.input_tokens`, `usage.output_tokens`.

3. **OpenAI** (reference): `openai` SDK. `client.chat.completions.create(...)`. Same usage extraction.

**Error handling**: Retry on transient errors (timeout, 503) up to 3 times with exponential backoff. Surface model-specific errors clearly (context window exceeded, rate limit, model not found).

**Verify**: Test with a mock HTTP server that returns canned responses. Integration test with Ollama if available (skip if not).

---

### Step 2: Prompt Builder

**Delivers**: Construct model-ready prompts from a `ContextPackage` (pipeline mode) or raw file contents (baseline mode).

**Files**:
- `src/clean_room/agent/prompt_builder.py` — `build_pipeline_prompt()`, `build_baseline_prompt()`
- `tests/test_prompt_builder.py`

**System prompt** (shared across modes):
```
You are a coding assistant. You will be given a task and relevant code context.
Your job is to produce the exact code changes needed to complete the task.

Output your changes as search-and-replace blocks. For each file you need to modify,
use this format:

<<<< SEARCH file_path
exact lines to find
====
replacement lines
>>>> REPLACE

Rules:
- The SEARCH block must match existing code exactly (including whitespace and indentation)
- Include enough context lines in SEARCH to uniquely identify the location
- You may output multiple search-and-replace blocks for different files or locations
- For new files, use an empty SEARCH block
- Do not include any other text outside the blocks unless explaining your reasoning briefly
```

The S&R format is chosen because:
- Simpler than unified diff for small models (no `@@` line numbers, no `+`/`-` prefixes)
- Each block is self-contained (no need to track hunks)
- Exact match requirement prevents ambiguous edits
- Easy to validate: if SEARCH text isn't found, the edit fails cleanly

**Pipeline prompt** (`build_pipeline_prompt(package: ContextPackage, task: str) -> tuple[str, str]`):
- System: the system prompt above
- User: `render_prompt(package)` from Phase 2 + task description
- Total tokens should be within the `ContextPackage.budget`

**Baseline prompt** (`build_baseline_prompt(repo_path, task, budget) -> tuple[str, str]`):
- System: same system prompt
- User: task description + file contents stuffed to budget (see Step 7 for baseline logic)

**Retry prompt** (`build_retry_prompt(original_prompt, attempt: AttemptRecord) -> tuple[str, str]`):
- System: same system prompt + retry instruction
- User: original context + task + structured error section:
  ```
  ## Previous Attempt (failed)
  ### Changes attempted:
  {unified_diff}

  ### Error output:
  {validation.test_output or validation.lint_output}

  ### What went wrong:
  {brief error classification}

  Please fix the issue and provide corrected search-and-replace blocks.
  Do not repeat changes that were correct — only fix what failed.
  ```

**Token budget awareness**: The prompt builder checks total token count before returning. If the retry section pushes over budget, truncate the error output (keep first and last 50 lines of test output — errors and summaries are usually at the extremes).

**Verify**: Unit tests asserting prompt structure, token budget compliance, retry section formatting.

---

### Step 3: Response Parser

**Delivers**: Extract structured `EditBlock`s from raw LLM output.

**Files**:
- `src/clean_room/agent/response_parser.py` — `parse_response(raw: str) -> list[EditBlock]`
- `tests/fixtures/sample_llm_responses.py`
- `tests/test_response_parser.py`

**Primary format: Search-and-Replace blocks**:
```
<<<< SEARCH path/to/file.py
def old_function():
    return False
====
def old_function():
    return True
>>>> REPLACE
```

Parser logic:
1. Scan for `<<<< SEARCH` markers. Extract file path from the marker line.
2. Collect lines until `====` delimiter → `search` text.
3. Collect lines until `>>>> REPLACE` → `replace` text.
4. Strip trailing whitespace from both blocks.
5. Repeat for all blocks in the response.

**Fallback: Unified diff**: If no S&R blocks found, attempt to parse unified diff format:
```
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,5 +10,5 @@
 def old_function():
-    return False
+    return True
```
Parse hunks into `EditBlock` with `diff_hunk` populated instead of `search`/`replace`.

**Fallback: Fenced code blocks**: If neither format found, look for triple-backtick code blocks with file path annotations. This handles cases where the model ignores format instructions and just produces code:
```python
# path/to/file.py
def old_function():
    return True
```
Detected by checking if the first comment line looks like a file path. These become full-file replacements — less precise but still usable.

**Error handling**:
- Malformed blocks (missing delimiters) → skip with warning, continue parsing remaining
- Empty search block → interpreted as new file creation
- Duplicate file paths in multiple blocks → multiple edits to the same file (applied in order)
- No parseable edits at all → return empty list (caller decides whether to retry)

**Verification**: Test against 15+ canned LLM responses covering: clean S&R, messy S&R (extra whitespace, model commentary mixed in), unified diff, fenced code blocks, mixed formats, and unparseable garbage.

---

### Step 4: Patch Application

**Delivers**: Apply `EditBlock`s to actual files. Git worktree isolation for safe rollback.

**Files**:
- `src/clean_room/agent/patch.py` — `apply_edits()`, `create_worktree()`, `cleanup_worktree()`
- `tests/fixtures/sample_patches/` — Test repos with known states
- `tests/test_patch.py`

**Worktree management**:
```python
def create_worktree(repo_path: Path, ref: str = "HEAD") -> Path:
    """Create an isolated git worktree for applying patches."""
    worktree_dir = repo_path / ".clean_room_worktrees" / f"attempt_{uuid4().hex[:8]}"
    subprocess.run(["git", "worktree", "add", "--detach", str(worktree_dir), ref],
                   cwd=repo_path, check=True)
    return worktree_dir

def cleanup_worktree(repo_path: Path, worktree_dir: Path) -> None:
    """Remove a worktree after use."""
    subprocess.run(["git", "worktree", "remove", "--force", str(worktree_dir)],
                   cwd=repo_path, check=True)
```

**Edit application** (`apply_edits(worktree: Path, edits: list[EditBlock]) -> ApplyResult`):

For S&R edits:
1. Read the target file from the worktree.
2. Find the `search` text in the file content. Use exact string matching first.
3. If exact match fails, try whitespace-normalized matching (collapse runs of spaces/tabs). This handles models that slightly misformat indentation.
4. If still no match, try fuzzy matching with a similarity threshold of 0.9 (difflib `SequenceMatcher`). Log a warning when fuzzy matching is used.
5. Replace the matched text with `replace` text.
6. Write the file back.

For diff hunks:
1. Write the unified diff to a temp file.
2. Apply via `git apply --verbose` in the worktree.
3. If `git apply` fails (context mismatch), try `git apply --3way` for three-way merge.

For full-file replacements:
1. Write the new content directly.

**ApplyResult**:
```python
@dataclass
class ApplyResult:
    success: bool
    applied_edits: list[EditBlock]        # Successfully applied
    failed_edits: list[tuple[EditBlock, str]]  # (edit, error message)
    unified_diff: str                     # git diff of all applied changes
```

After applying all edits, generate `unified_diff` via `git diff` in the worktree. This is the canonical patch format for SWE-bench submission.

**Verify**: Test S&R application (exact match, whitespace-normalized, fuzzy), diff application, new file creation, multi-file edits, and graceful failure on unmatchable search text.

---

### Step 5: Validation

**Delivers**: Run tests, linter, and type checker against the patched worktree. Detect the project's test runner automatically.

**Files**:
- `src/clean_room/agent/validation.py` — `validate(worktree, ...) -> ValidationResult`
- `tests/test_validation.py`

**Test runner detection** (check in order, use first match):
1. `pytest.ini` or `pyproject.toml` with `[tool.pytest]` → `pytest`
2. `package.json` with `scripts.test` → `npm test` (or `pnpm test`, `yarn test` based on lockfile)
3. `Makefile` with `test` target → `make test`
4. Fallback: glob for `test_*.py` or `*_test.py` → `pytest`

**Test execution**:
```python
def run_tests(worktree: Path, timeout: int = 300) -> tuple[int, str, str]:
    """Run detected test suite. Returns (exit_code, stdout, stderr)."""
    runner = detect_test_runner(worktree)
    result = subprocess.run(
        runner.command,
        cwd=worktree,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr
```

**Targeted test execution**: If the task mentions specific test files or test names, run only those first (fast feedback). If they pass, run the full suite as confirmation. This saves significant time on large test suites.

**Linting** (optional, configurable):
- Python: `ruff check` if available
- TypeScript/JavaScript: `eslint` if configured in the project
- Only run if `--lint` flag is set (not default for benchmarks — SWE-bench doesn't require lint pass)

**Type checking** (optional, configurable):
- Python: `mypy` or `pyright` if configured
- TypeScript: `tsc --noEmit`
- Only run if `--type-check` flag is set

**Failing test extraction**: Parse test output to extract individual failing test names. For pytest: parse the short summary section. For Jest/Mocha: parse the failure output. This is useful for retry prompts — telling the model exactly which tests fail is more actionable than dumping raw output.

**SWE-bench mode**: When running under SWE-bench, skip our validation entirely — the `swebench` package handles test execution in Docker containers with the correct environment. Our validation is for standalone use and development.

**Verify**: Test runner detection against sample project structures. Test output parsing for pytest and npm test formats.

---

### Step 6: Retry Logic

**Delivers**: Manage the generate→apply→validate→retry loop. Fresh context each attempt.

**Files**:
- `src/clean_room/agent/retry.py` — `retry_loop()`
- (Tested as part of `test_harness.py` in Step 8)

**Retry loop**:
```python
def retry_loop(
    llm: LLMClient,
    prompt_builder: PromptBuilder,
    initial_prompt: tuple[str, str],     # (system, user)
    worktree_factory: Callable,
    validator: Callable,
    max_attempts: int = 3,
) -> TaskResult:
    attempts = []
    for attempt_num in range(1, max_attempts + 1):
        # Build prompt (initial or retry)
        if attempt_num == 1:
            system, user = initial_prompt
        else:
            system, user = prompt_builder.build_retry_prompt(
                initial_prompt, attempts[-1]
            )

        # Generate
        generation = llm.generate(user, system=system)
        edits = parse_response(generation.raw_response)

        if not edits:
            # Model produced no parseable edits — record and retry
            attempts.append(AttemptRecord(
                attempt=attempt_num, generation=generation,
                patch_applied=False, validation=None, unified_diff=""
            ))
            continue

        # Apply in fresh worktree
        worktree = worktree_factory()
        apply_result = apply_edits(worktree, edits)

        if not apply_result.success:
            cleanup_worktree(worktree)
            attempts.append(AttemptRecord(
                attempt=attempt_num, generation=generation,
                patch_applied=False, validation=None,
                unified_diff=apply_result.unified_diff
            ))
            continue

        # Validate
        validation = validator(worktree)
        attempts.append(AttemptRecord(
            attempt=attempt_num, generation=generation,
            patch_applied=True, validation=validation,
            unified_diff=apply_result.unified_diff
        ))

        if validation.success:
            cleanup_worktree(worktree)
            return TaskResult(success=True, attempts=attempts,
                              final_diff=apply_result.unified_diff, ...)

        cleanup_worktree(worktree)

    # All attempts exhausted
    return TaskResult(success=False, attempts=attempts,
                      final_diff=attempts[-1].unified_diff if attempts else None, ...)
```

**Error classification** for retry prompts:
- **Test failure**: specific tests failed → include failing test names and relevant assertion errors
- **Syntax error**: patch created invalid syntax → include the syntax error with file/line
- **Patch failure**: search text not found → include which edit blocks failed and why
- **Timeout**: tests took too long → suggest the change may have introduced an infinite loop
- **Import error**: missing or circular imports → include the import error

This classification feeds into `build_retry_prompt()` to give the model structured, actionable feedback rather than raw output dumps.

**Depends on**: Steps 1-5

---

### Step 7: Naive Baseline

**Delivers**: Full-context construction for Configs A and C. A fair baseline that represents what a reasonable but unsophisticated agent would do.

**Files**:
- `src/clean_room/agent/baseline.py` — `build_naive_context(repo_path, task, budget) -> str`
- `tests/test_baseline.py`

**The baseline is not deliberately sabotaged.** It uses basic intelligence about file ordering — it just doesn't use the Phase 1/2 pipeline. This ensures the comparison measures the value of structured retrieval, not the difference between "something" and "nothing."

**Naive context construction**:
1. **Extract file hints from task**: Same regex-based file/symbol hint extraction as Phase 2's task analysis (reuse `task_analysis.py`). This is fair — even a naive agent would notice explicit file mentions.

2. **Prioritize files**:
   - Tier 1: Files explicitly mentioned in the task → include in full
   - Tier 2: Files in the same directory as Tier 1 files → include in full
   - Tier 3: Test files matching Tier 1/2 file names → include in full
   - Tier 4: All remaining files, ordered by: (a) path depth (shallower first — top-level files are usually more important), (b) file size ascending (smaller files are more likely utility/config)

3. **Fill budget**:
   - Walk through tiers in order
   - For each file: if adding it stays within budget, include full content
   - If a file would exceed budget: truncate to first N lines that fit, then stop
   - Reserve 512 tokens for task description

4. **Format**: Simple concatenation with file path headers:
   ```
   ## Task
   {task_description}

   ## File: src/auth/handler.py
   ```python
   {file contents}
   ...
   ```

**What the baseline does NOT do** (these are Phase 2's advantages):
- No AST parsing — files are raw text, not structured symbols
- No dependency graph traversal — can't follow imports
- No co-change analysis — no historical signal
- No symbol-level filtering — whole files or nothing
- No relevance scoring — just directory proximity and file size heuristics
- No LLM enrichment metadata — no domain/concept matching
- No token-efficient extraction — a 500-line file goes in whole or not at all

**Verify**: Test that baseline includes mentioned files first, respects token budget, doesn't crash on repos with non-UTF-8 files or binary files.

---

### Step 8: Agent Harness + CLI

**Delivers**: `cra solve "task description" --repo /path` orchestrates the full pipeline. Supports both pipeline and baseline modes.

**Files**:
- `src/clean_room/agent/harness.py` — `solve_task()` top-level orchestrator
- Update `src/clean_room/cli.py` — Add `solve` command
- `tests/test_harness.py`

**CLI interface**:
```bash
# Pipeline mode (default) — uses Phase 1 index + Phase 2 retrieval
cra solve "fix the login validation bug" \
  --repo /path/to/repo \
  --model qwen2.5-coder:3b \
  --budget 32768 \
  --max-attempts 3 \
  --verbose

# Baseline mode — naive full-context
cra solve "fix the login validation bug" \
  --repo /path/to/repo \
  --model qwen2.5-coder:3b \
  --mode baseline \
  --budget 32768

# Dry run — show the prompt without calling the LLM
cra solve "fix the login validation bug" \
  --repo /path/to/repo \
  --dry-run

# Output diff to file
cra solve "fix the login validation bug" \
  --repo /path/to/repo \
  --output patch.diff
```

**Pipeline mode sequence**:
1. Check repo is indexed (`cra index` must have been run). Error if not.
2. Retrieve context: `cra retrieve` (Phase 2) → `ContextPackage`
3. Build prompt from `ContextPackage`
4. Run retry loop: generate → apply → validate → retry if needed
5. Output final diff (or error)

**Baseline mode sequence**:
1. Build naive context from raw repo files
2. Build prompt from naive context
3. Run same retry loop
4. Output final diff

**Verbose mode** (`-v`): Log each stage — task analysis, context construction (with token counts), LLM call (with latency), response parsing (edit count), patch application (success/failure per edit), validation (test results). Critical for debugging and understanding agent behavior.

**Depends on**: Steps 1-7, Phase 1 (indexing), Phase 2 (retrieval)

---

### Step 9: SWE-bench Integration

**Delivers**: Load SWE-bench instances, adapt them for our agent, produce patches in the required format.

**Files**:
- `src/clean_room/bench/swe_bench.py` — `load_instances()`, `prepare_instance()`, `format_patch()`
- `src/clean_room/bench/configs.py` — Stress test matrix definitions
- `tests/test_swe_bench.py`

**Instance loading**: Use the `swebench` Python package (or `datasets` from HuggingFace) to load SWE-bench Verified instances:
```python
from datasets import load_dataset

def load_instances(split: str = "test") -> list[SWEBenchInstance]:
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split=split)
    return [SWEBenchInstance(
        instance_id=row["instance_id"],
        repo=row["repo"],
        base_commit=row["base_commit"],
        problem_statement=row["problem_statement"],
        hints_text=row["hints_text"],
        test_patch=row["test_patch"],
        patch=row["patch"],  # gold patch, for reference only
    ) for row in ds]
```

**Instance preparation**:
1. Clone the repo (or use cached clone) at `base_commit`
2. For pipeline mode: run `cra index` on the cloned repo
3. Use `problem_statement` (+ optionally `hints_text`) as the task description
4. Apply `test_patch` to add the test that should pass (SWE-bench convention)

**Stress test matrix** (from CLAUDE.md):
```python
CONFIGS = {
    "A": StressConfig(
        name="A", label="4B-naive",
        llm=LLMConfig(provider="ollama", model="qwen2.5-coder:3b"),
        mode="baseline", budget=8192  # 4B model has small context
    ),
    "B": StressConfig(
        name="B", label="4B-pipeline",
        llm=LLMConfig(provider="ollama", model="qwen2.5-coder:3b"),
        mode="pipeline", budget=8192
    ),
    "C": StressConfig(
        name="C", label="frontier-naive",
        llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
        mode="baseline", budget=32768
    ),
    "D": StressConfig(
        name="D", label="frontier-pipeline",
        llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
        mode="pipeline", budget=32768
    ),
}
```

Note: model names and budgets are placeholders — the actual values depend on what's available at runtime. The 4B Qwen model is the same one from Auto-GM's knowledge system. The frontier model is whatever's current when we run the benchmark.

**Patch format**: SWE-bench expects a unified diff. Our `apply_result.unified_diff` from Step 4 is already in this format. `format_patch()` ensures it matches SWE-bench's expected format (proper `a/` and `b/` prefixes, correct header).

**Verify**: Test instance loading (mocked dataset), preparation (mocked git clone), patch formatting against SWE-bench's expected format.

---

### Step 10: Benchmark Runner + Reporting

**Delivers**: `cra bench` runs the full stress test matrix and produces comparison tables.

**Files**:
- `src/clean_room/bench/runner.py` — `run_benchmark()`
- `src/clean_room/bench/metrics.py` — `compute_metrics()`, `compare_configs()`
- `src/clean_room/bench/report.py` — `generate_report()`
- Update `src/clean_room/cli.py` — Add `bench` command
- `tests/test_runner.py`

**CLI interface**:
```bash
# Run all 4 configs on SWE-bench Verified (or a subset)
cra bench --configs A,B,C,D --instances 50 --output results/

# Run specific configs
cra bench --configs A,B --instances 10 --output results/

# Run on a specific instance (for debugging)
cra bench --configs B --instance-id "django__django-16379" --output results/

# Resume a previous run (skip already-completed instances)
cra bench --configs A,B --resume results/

# Generate report from existing results
cra report results/
```

**Runner logic**:
1. Load SWE-bench instances (all 500 for Verified, or `--instances N` for a subset — random sample with fixed seed for reproducibility)
2. For each instance × config:
   a. Prepare instance (clone, checkout, index if pipeline mode)
   b. Run agent (`solve_task` from Step 8)
   c. Save `TaskResult` to disk (JSON, one file per instance per config)
   d. Log progress
3. After all runs, compute aggregate metrics

**Parallelism**: Instances within a config run sequentially (one at a time — local LLM is the bottleneck). Different configs can run in parallel if resources allow (e.g., run A and B simultaneously since they use the same local model, but may compete for GPU). By default, sequential.

**Checkpointing**: Each `TaskResult` saved to disk immediately after completion. Runner skips instances that already have results on resume. This is critical for long benchmark runs that may be interrupted.

**Metrics** (per config):
- **Pass rate**: `successful_tasks / total_tasks` — the headline number
- **Pass@1**: Success rate on first attempt only
- **Average attempts**: Mean attempts across all tasks (lower is better)
- **Average tokens**: Mean total tokens consumed per task
- **Average latency**: Mean wall-clock time per task
- **Token efficiency**: `tokens_for_successes / total_tokens` (what fraction of compute was productive)

**Comparison report**:
```
╔══════════╦════════════════╦══════════╦═════════╦═══════════╦═══════════╗
║  Config  ║     Model      ║ Context  ║ Pass@1  ║ Pass (3)  ║ Avg Tokens║
╠══════════╬════════════════╬══════════╬═════════╬═══════════╬═══════════╣
║ A        ║ Qwen-3B        ║ Naive    ║  4.0%   ║   6.0%    ║   3,200   ║
║ B        ║ Qwen-3B        ║ Pipeline ║ 14.0%   ║  20.0%    ║   2,800   ║
║ C        ║ Claude Sonnet  ║ Naive    ║ 32.0%   ║  38.0%    ║  28,000   ║
║ D        ║ Claude Sonnet  ║ Pipeline ║ 40.0%   ║  48.0%    ║  18,000   ║
╚══════════╩════════════════╩══════════╩═════════╩═══════════╩═══════════╝

Thesis Validation:
  B > A: +14.0pp (pipeline helps small model)          ✓
  B ≈ C: -18.0pp (gap remains — see analysis)          ?
  D > C: +10.0pp (pipeline helps large model too)      ✓
```
(Numbers are illustrative, not predictions.)

**SWE-bench evaluation**: After generating all patches, optionally run them through `swebench`'s Docker-based evaluation harness for official scoring. This is separate from our internal validation because SWE-bench uses specific test environments. Command: `swebench.harness.run_evaluation(predictions=..., dataset_name="princeton-nlp/SWE-bench_Verified")`.

**Depends on**: Steps 8-9

---

## Step Dependency Graph

```
Step 1 (LLM Client)
  │
  ├──► Step 2 (Prompt Builder)
  │      │
  │      ├──► Step 3 (Response Parser)
  │      │      │
  │      │      └──► Step 4 (Patch Application)
  │      │             │
  │      │             └──► Step 5 (Validation)
  │      │                    │
  │      │                    └──► Step 6 (Retry Logic)
  │      │                           │
  │      └──► Step 7 (Naive Baseline) │
  │                  │                │
  │                  └──► Step 8 (Agent Harness + CLI) ◄── Step 6
  │                             │
  │                             └──► Step 9 (SWE-bench Integration)
  │                                    │
  │                                    └──► Step 10 (Runner + Reporting)
  │
  [All steps depend on Phases 1 + 2 being complete]
```

Steps 3-5 are a linear chain (parse → apply → validate). Step 7 (baseline) is independent of Steps 3-6 and can be built in parallel.

---

## Verification (Phase 3 Gate)

### Standalone verification:
```bash
# Solve a task on a real repo (pipeline mode)
cra index /path/to/repo
cra solve "fix the broken test in test_auth.py" \
  --repo /path/to/repo --model qwen2.5-coder:3b -v

# Solve same task in baseline mode
cra solve "fix the broken test in test_auth.py" \
  --repo /path/to/repo --model qwen2.5-coder:3b --mode baseline -v

# Verify both produce valid diffs
```

### Benchmark verification:
```bash
# Run a small subset (5-10 instances) across all 4 configs
cra bench --configs A,B,C,D --instances 10 --output results/pilot/

# Generate comparison report
cra report results/pilot/

# Verify:
# - All configs complete without crashes
# - Results files exist for each instance × config
# - At least one config produces some successful patches
# - Comparison table renders correctly
```

### Gate criteria:
1. Agent produces valid patches for at least some tasks in every config
2. Pipeline mode (B) outperforms naive mode (A) with the same small model — **the minimum thesis proof**
3. The comparison is fair: same model, same temperature, same retry count, same output format
4. All results are reproducible (fixed seeds, temperature 0.0)
5. SWE-bench patches are in the correct format for official evaluation
6. Benchmark runner handles interruption and resume gracefully
7. Verbose logging provides enough detail to debug failures

---

## What Phase 3 Does NOT Include

- **Per-stage LoRA adapters**: Long-term optimization. Phase 3 uses the same model for all stages.
- **Cross-project knowledge transfer**: Phase 3 indexes one repo at a time. Cross-project KB is future work.
- **Self-improving tooling**: No runtime tool synthesis. The tool set is fixed.
- **Production deployment**: Phase 3 is a research harness, not a product. No auth, no multi-user, no UI.
