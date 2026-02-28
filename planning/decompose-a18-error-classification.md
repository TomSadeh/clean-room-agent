# Implementation Plan: Decompose A18 Function Implement (Retry Path) with Binary Compiler Error Classification

Date: 2026-02-28

## 1. Overview

**Current state**: The compile-and-retry loop in `src/clean_room_agent/orchestrator/runner.py` (lines 896-989) does a blind retry when per-function compilation fails. When `compile_single_file()` returns `(False, comp_error)`, the raw `comp_error` string is passed to `execute_function_implement()` on the next attempt via the `compiler_error` kwarg. The 1.7B model receives the full compiler error with a generic "fix the error below" instruction and must simultaneously diagnose the error category, determine the fix strategy, and rewrite the function body. This is too much cognitive load for a 1.7B on a retry path.

**Target state**: Insert a compiler error classification step between the compilation failure and the retry, replacing the blind pass-through. The classification pipeline is:

1. **Deterministic pre-filter**: Check the compiler error against a `_SYMBOL_TO_HEADER` mapping and regex patterns. If the error is a known missing include (e.g., `implicit declaration of function 'printf'` maps to `#include <stdio.h>`), fix deterministically by inserting the include -- no LLM call needed, no retry consumed.

2. **Sequential binary LLM calls with short-circuit**: For errors not caught by the deterministic pre-filter, ask up to 3 binary classifiers in sequence, stopping at the first "yes":
   - "Is this a missing include error?" (if yes, ask a focused 1.7B which include is needed)
   - "Is this a signature mismatch?" (if yes, flag for header regeneration)
   - "Is this a missing definition?" (if yes, flag for deferred stub re-processing)
   - Default (all "no"): classify as `logic_error` -- retry with the existing mechanism.

3. **Deterministic recovery actions** per category:
   - `missing_include`: `add_include_to_file()` inserts the `#include` directive at the top of the source file, then re-compiles. If compilation succeeds, the retry attempt is not consumed.
   - `signature_mismatch`: flag on `CompilerErrorClassification` for the orchestrator; the retry passes the diagnostic context "fix signature mismatch" rather than the raw error.
   - `missing_definition`: flag on `CompilerErrorClassification` for the orchestrator; the retry passes diagnostic context "missing dependency definition."
   - `logic_error`: pass the raw compiler error to the existing retry mechanism unchanged.

**Output type**: New `CompilerErrorClassification` dataclass consumed internally by the retry loop. No downstream changes. The `StepResult` produced by `execute_function_implement()` is unchanged.

## 2. Data Flow

```
compile_single_file() returns (False, comp_error)
    |
    v
[1. classify_compiler_error()] -- entry point
    |
    +--> [1a. _deterministic_include_check()]
    |        Input: comp_error string
    |        Output: CompilerErrorClassification with category="missing_include"
    |                and suggested_include set, OR None if not matched
    |
    (if None, continue to binary classifiers)
    |
    +--> [1b. run_binary_judgment: "missing include?"]
    |        Input: comp_error snippet + function context
    |        Output: yes/no
    |        (if yes) -> [1b2. focused LLM: which include?]
    |                     Output: CompilerErrorClassification(missing_include, suggested_include=<header>)
    |
    (if no, continue)
    |
    +--> [1c. run_binary_judgment: "signature mismatch?"]
    |        Input: comp_error snippet + function context
    |        Output: yes/no
    |        (if yes) -> CompilerErrorClassification(signature_mismatch)
    |
    (if no, continue)
    |
    +--> [1d. run_binary_judgment: "missing definition?"]
    |        Input: comp_error snippet + function context
    |        Output: yes/no
    |        (if yes) -> CompilerErrorClassification(missing_definition)
    |
    (if all no)
    |
    +--> CompilerErrorClassification(logic_error)
    |
    v
[2. Recovery action in orchestrator retry loop]
    |
    +--> missing_include (deterministic fix) -> add_include_to_file() -> re-compile
    |       (if re-compile succeeds, skip retry -- don't consume an attempt)
    |
    +--> signature_mismatch -> set compiler_error with diagnostic context -> continue retry
    +--> missing_definition -> set compiler_error with diagnostic context -> continue retry
    +--> logic_error -> set compiler_error = raw comp_error -> continue retry (existing behavior)
```

## 3. Config Toggle

**File**: `src/clean_room_agent/config.py` (after line 128, following `decomposed_scaffold`)

Add a new commented-out config line in the template:
```
'# decomposed_error_classification = false  # Binary compiler error classification on retry (A18 decomposition)\n'
```

**File**: `src/clean_room_agent/orchestrator/runner.py` (after line 85, following `_use_decomposed_scaffold`)

Add a new config check function, following the exact pattern of the existing toggles:

```python
def _use_decomposed_error_classification(config: dict) -> bool:
    """Check if decomposed error classification is enabled. Supplementary, default False."""
    orch = config.get("orchestrator", {})
    if "decomposed_error_classification" not in orch:
        logger.debug("decomposed_error_classification not in config, defaulting to False")
    return bool(orch.get("decomposed_error_classification", False))
```

Classification: **Supplementary** (non-core, safe fallback to blind retry). Default `false`. Same classification as `decomposed_planning` and `decomposed_scaffold`.

Note: This toggle is independent of `decomposed_scaffold`. When `decomposed_scaffold = true` and `decomposed_error_classification = true`, the compile-and-retry loop uses binary classification before retrying. When `decomposed_error_classification = false` (or absent), the existing blind retry is used. When `decomposed_scaffold = false`, the per-function compile loop never runs, so this toggle has no effect.

## 4. New Module

**`src/clean_room_agent/execute/compiler_error_classifier.py`**

This follows the exact pattern of `src/clean_room_agent/execute/decomposed_scaffold.py` and `src/clean_room_agent/execute/decomposed_adjustment.py` (from the A20 plan) -- a module-level public function composed of private stage functions.

## 5. New Dataclass

Add to `src/clean_room_agent/execute/dataclasses.py`:

```python
@dataclass
class CompilerErrorClassification(_SerializableMixin):
    """Result of compiler error classification on the retry path."""
    category: str          # "missing_include", "signature_mismatch", "missing_definition", "logic_error"
    raw_error: str         # the original compiler error string
    suggested_include: str | None = None   # header to add (only for missing_include)
    diagnostic_context: str | None = None  # enriched context for the retry prompt

    _REQUIRED = ("category", "raw_error")
    _NON_EMPTY = ("category", "raw_error")
```

Valid category values as module-level constants in `compiler_error_classifier.py`:

```python
ERROR_CAT_MISSING_INCLUDE = "missing_include"
ERROR_CAT_SIGNATURE_MISMATCH = "signature_mismatch"
ERROR_CAT_MISSING_DEFINITION = "missing_definition"
ERROR_CAT_LOGIC_ERROR = "logic_error"

VALID_ERROR_CATEGORIES = frozenset({
    ERROR_CAT_MISSING_INCLUDE,
    ERROR_CAT_SIGNATURE_MISMATCH,
    ERROR_CAT_MISSING_DEFINITION,
    ERROR_CAT_LOGIC_ERROR,
})
```

## 6. Deterministic Pre-Filter: `_SYMBOL_TO_HEADER` Mapping

```python
# Maps function/macro names to their C standard library headers.
# Only includes unambiguous mappings (one canonical header per symbol).
# Source: C11 standard, Annex B.
_SYMBOL_TO_HEADER: dict[str, str] = {
    # <stdio.h>
    "printf": "stdio.h", "fprintf": "stdio.h", "sprintf": "stdio.h",
    "snprintf": "stdio.h", "scanf": "stdio.h", "fscanf": "stdio.h",
    "sscanf": "stdio.h", "fopen": "stdio.h", "fclose": "stdio.h",
    "fread": "stdio.h", "fwrite": "stdio.h", "fgets": "stdio.h",
    "fputs": "stdio.h", "puts": "stdio.h", "getchar": "stdio.h",
    "putchar": "stdio.h", "fflush": "stdio.h", "fseek": "stdio.h",
    "ftell": "stdio.h", "rewind": "stdio.h", "perror": "stdio.h",
    "remove": "stdio.h", "rename": "stdio.h", "tmpfile": "stdio.h",
    "fgetc": "stdio.h", "fputc": "stdio.h", "ungetc": "stdio.h",
    "feof": "stdio.h", "ferror": "stdio.h", "clearerr": "stdio.h",
    "vprintf": "stdio.h", "vfprintf": "stdio.h", "vsprintf": "stdio.h",
    "vsnprintf": "stdio.h",
    "FILE": "stdio.h", "EOF": "stdio.h",
    "stdin": "stdio.h", "stdout": "stdio.h", "stderr": "stdio.h",
    # <stdlib.h>
    "malloc": "stdlib.h", "calloc": "stdlib.h", "realloc": "stdlib.h",
    "free": "stdlib.h", "atoi": "stdlib.h", "atol": "stdlib.h",
    "atof": "stdlib.h", "strtol": "stdlib.h", "strtoul": "stdlib.h",
    "strtod": "stdlib.h", "strtof": "stdlib.h", "strtold": "stdlib.h",
    "abs": "stdlib.h", "labs": "stdlib.h", "div": "stdlib.h",
    "ldiv": "stdlib.h", "rand": "stdlib.h", "srand": "stdlib.h",
    "exit": "stdlib.h", "abort": "stdlib.h", "atexit": "stdlib.h",
    "system": "stdlib.h", "getenv": "stdlib.h",
    "qsort": "stdlib.h", "bsearch": "stdlib.h",
    "EXIT_SUCCESS": "stdlib.h", "EXIT_FAILURE": "stdlib.h",
    "RAND_MAX": "stdlib.h",
    "NULL": "stdlib.h", "size_t": "stdlib.h",
    # <string.h>
    "memcpy": "string.h", "memmove": "string.h", "memset": "string.h",
    "memcmp": "string.h", "memchr": "string.h",
    "strlen": "string.h", "strcpy": "string.h", "strncpy": "string.h",
    "strcat": "string.h", "strncat": "string.h",
    "strcmp": "string.h", "strncmp": "string.h",
    "strchr": "string.h", "strrchr": "string.h",
    "strstr": "string.h", "strtok": "string.h",
    "strerror": "string.h", "strdup": "string.h",
    # <math.h>
    "sin": "math.h", "cos": "math.h", "tan": "math.h",
    "asin": "math.h", "acos": "math.h", "atan": "math.h",
    "atan2": "math.h", "sinh": "math.h", "cosh": "math.h",
    "tanh": "math.h", "exp": "math.h", "log": "math.h",
    "log10": "math.h", "pow": "math.h", "sqrt": "math.h",
    "ceil": "math.h", "floor": "math.h", "fabs": "math.h",
    "fmod": "math.h", "round": "math.h", "trunc": "math.h",
    "INFINITY": "math.h", "NAN": "math.h", "HUGE_VAL": "math.h",
    "M_PI": "math.h",
    # <ctype.h>
    "isalpha": "ctype.h", "isdigit": "ctype.h", "isalnum": "ctype.h",
    "isspace": "ctype.h", "isupper": "ctype.h", "islower": "ctype.h",
    "toupper": "ctype.h", "tolower": "ctype.h",
    "isprint": "ctype.h", "ispunct": "ctype.h", "isxdigit": "ctype.h",
    # <assert.h>
    "assert": "assert.h",
    # <errno.h>
    "errno": "errno.h", "ERANGE": "errno.h", "EDOM": "errno.h",
    # <limits.h>
    "INT_MAX": "limits.h", "INT_MIN": "limits.h",
    "UINT_MAX": "limits.h", "LONG_MAX": "limits.h",
    "LONG_MIN": "limits.h", "CHAR_MAX": "limits.h",
    "CHAR_MIN": "limits.h", "SCHAR_MAX": "limits.h",
    # <stdbool.h>
    "bool": "stdbool.h", "true": "stdbool.h", "false": "stdbool.h",
    # <stdint.h>
    "int8_t": "stdint.h", "int16_t": "stdint.h", "int32_t": "stdint.h",
    "int64_t": "stdint.h", "uint8_t": "stdint.h", "uint16_t": "stdint.h",
    "uint32_t": "stdint.h", "uint64_t": "stdint.h",
    "SIZE_MAX": "stdint.h", "INT64_MAX": "stdint.h",
    "UINT64_MAX": "stdint.h",
    # <stddef.h>
    "ptrdiff_t": "stddef.h", "offsetof": "stddef.h",
    # <time.h>
    "time": "time.h", "clock": "time.h", "difftime": "time.h",
    "mktime": "time.h", "strftime": "time.h",
    "localtime": "time.h", "gmtime": "time.h",
    "time_t": "time.h", "clock_t": "time.h", "CLOCKS_PER_SEC": "time.h",
    # <signal.h>
    "signal": "signal.h", "raise": "signal.h",
    "SIGINT": "signal.h", "SIGTERM": "signal.h", "SIGSEGV": "signal.h",
    # <setjmp.h>
    "setjmp": "setjmp.h", "longjmp": "setjmp.h", "jmp_buf": "setjmp.h",
}
```

### Regex patterns for the deterministic check

```python
# Matches GCC/Clang implicit declaration warnings.
# Group 1: the function name.
_IMPLICIT_DECL_PATTERN = re.compile(
    r"implicit declaration of function ['\"](\w+)['\"]",
    re.IGNORECASE,
)

# Matches GCC/Clang "undeclared identifier" errors.
# Group 1: the identifier name.
_UNDECLARED_ID_PATTERN = re.compile(
    r"(?:use of undeclared identifier|'(\w+)' undeclared)",
    re.IGNORECASE,
)

# Matches "unknown type name" errors.
# Group 1: the type name.
_UNKNOWN_TYPE_PATTERN = re.compile(
    r"unknown type name ['\"](\w+)['\"]",
    re.IGNORECASE,
)
```

### `_deterministic_include_check()` function

```python
def _deterministic_include_check(
    comp_error: str,
    file_path: str,
    scaffold_content: dict[str, str],
) -> CompilerErrorClassification | None:
    """Check if the compiler error is a known missing include fixable without LLM.

    Scans comp_error for implicit declaration / undeclared identifier / unknown type
    patterns. If the symbol is in _SYMBOL_TO_HEADER and the required #include is
    not already present in the file, returns a classification with suggested_include.

    Returns None if the error does not match any known pattern.
    """
    file_content = scaffold_content.get(file_path, "")

    for pattern in (_IMPLICIT_DECL_PATTERN, _UNDECLARED_ID_PATTERN, _UNKNOWN_TYPE_PATTERN):
        match = pattern.search(comp_error)
        if match:
            symbol = match.group(1)
            if symbol and symbol in _SYMBOL_TO_HEADER:
                header = _SYMBOL_TO_HEADER[symbol]
                include_directive = f"#include <{header}>"
                if include_directive not in file_content:
                    return CompilerErrorClassification(
                        category=ERROR_CAT_MISSING_INCLUDE,
                        raw_error=comp_error,
                        suggested_include=header,
                        diagnostic_context=f"Missing {include_directive} for symbol '{symbol}'",
                    )

    return None
```

## 7. Binary Classifier System Prompts

Add to `SYSTEM_PROMPTS` dict in `src/clean_room_agent/execute/prompts.py`:

```python
"error_missing_include": (
    "You are a C compiler error classifier. You will be given a compiler error "
    "message and the context of a function that failed to compile. "
    "Determine whether the error is caused by a missing #include directive.\n\n"
    "A missing include error typically manifests as:\n"
    "- 'implicit declaration of function'\n"
    "- 'undeclared identifier'\n"
    "- 'unknown type name'\n"
    "- 'incomplete type' for standard library types\n\n"
    "Answer with exactly one word: \"yes\" or \"no\""
),
"error_signature_mismatch": (
    "You are a C compiler error classifier. You will be given a compiler error "
    "message and the context of a function that failed to compile. "
    "Determine whether the error is caused by a function signature mismatch "
    "between the header declaration and the implementation.\n\n"
    "A signature mismatch typically manifests as:\n"
    "- 'conflicting types for'\n"
    "- 'too many arguments' or 'too few arguments'\n"
    "- 'incompatible type for argument'\n"
    "- 'incompatible pointer type'\n"
    "- parameter count or type disagreement between .h and .c\n\n"
    "Answer with exactly one word: \"yes\" or \"no\""
),
"error_missing_definition": (
    "You are a C compiler error classifier. You will be given a compiler error "
    "message and the context of a function that failed to compile. "
    "Determine whether the error is caused by a missing function or variable "
    "definition -- i.e., a symbol that is declared but never defined.\n\n"
    "A missing definition typically manifests as:\n"
    "- 'undefined reference to' (at link time)\n"
    "- calling a function that exists in a header but has no implementation\n"
    "- referencing an extern variable that is never defined\n\n"
    "Note: if the error is clearly a missing #include or a type mismatch, "
    "answer no -- those are different categories.\n\n"
    "Answer with exactly one word: \"yes\" or \"no\""
),
"error_which_include": (
    "You are a C include resolver. You will be given a compiler error that "
    "has been identified as a missing #include problem. Determine which "
    "header file needs to be included.\n\n"
    "Output ONLY the header name (e.g. \"stdio.h\" or \"mylib.h\"). "
    "Do not include the #include directive or angle brackets -- just the filename.\n\n"
    "If this is a system header, use the standard name (e.g. \"stdlib.h\"). "
    "If this is a project-local header, use the relative path (e.g. \"hash_table.h\")."
),
```

These are designed for 0.6B classifiers (binary prompts) except `error_which_include` which is a focused 1.7B call.

## 8. Sequential Binary Classification Function

```python
def classify_compiler_error(
    comp_error: str,
    file_path: str,
    scaffold_content: dict[str, str],
    stub: FunctionStub,
    llm: LoggedLLMClient,
) -> CompilerErrorClassification:
    """Classify a compiler error via deterministic check + sequential binary LLM calls.

    1. Deterministic pre-filter (no LLM): check _SYMBOL_TO_HEADER mapping
    2. Binary: "missing include?" (if yes, focused LLM: "which include?")
    3. Binary: "signature mismatch?"
    4. Binary: "missing definition?"
    5. Default: "logic_error"

    Short-circuits at the first positive classification.
    """
    # Stage 1: Deterministic pre-filter
    det_result = _deterministic_include_check(comp_error, file_path, scaffold_content)
    if det_result is not None:
        logger.info(
            "Deterministic include fix for %s: %s",
            stub.name, det_result.suggested_include,
        )
        return det_result

    # Truncate error for binary prompts (keep it small for 0.6B)
    truncated_error = comp_error[:500] if len(comp_error) > 500 else comp_error

    # Build shared task context
    task_context = (
        f"Function: {stub.name}\n"
        f"File: {file_path}\n"
        f"Signature: {stub.signature}\n"
    )

    # Stage 2: Binary -- missing include?
    is_missing_include = _run_single_binary(
        llm, "error_missing_include", task_context, truncated_error,
        stage_name="error_classify_include",
    )
    if is_missing_include:
        suggested = _resolve_missing_include(comp_error, llm)
        return CompilerErrorClassification(
            category=ERROR_CAT_MISSING_INCLUDE,
            raw_error=comp_error,
            suggested_include=suggested,
            diagnostic_context=f"LLM classified as missing include; suggested: {suggested}",
        )

    # Stage 3: Binary -- signature mismatch?
    is_sig_mismatch = _run_single_binary(
        llm, "error_signature_mismatch", task_context, truncated_error,
        stage_name="error_classify_signature",
    )
    if is_sig_mismatch:
        return CompilerErrorClassification(
            category=ERROR_CAT_SIGNATURE_MISMATCH,
            raw_error=comp_error,
            diagnostic_context=(
                "Compiler error classified as signature mismatch between header "
                "declaration and implementation. Review the function signature in "
                "the header file and ensure the implementation matches exactly."
            ),
        )

    # Stage 4: Binary -- missing definition?
    is_missing_def = _run_single_binary(
        llm, "error_missing_definition", task_context, truncated_error,
        stage_name="error_classify_definition",
    )
    if is_missing_def:
        return CompilerErrorClassification(
            category=ERROR_CAT_MISSING_DEFINITION,
            raw_error=comp_error,
            diagnostic_context=(
                "Compiler error classified as missing definition. A function or "
                "variable is declared but never defined. Check that all required "
                "dependency stubs have been generated."
            ),
        )

    # Stage 5: Default -- logic error
    return CompilerErrorClassification(
        category=ERROR_CAT_LOGIC_ERROR,
        raw_error=comp_error,
        diagnostic_context=None,  # Use raw error as-is for retry
    )
```

### Helper: `_run_single_binary()`

Thin wrapper around `run_binary_judgment()` for a single item, since the sequential short-circuit pattern does not batch items.

```python
def _run_single_binary(
    llm: LoggedLLMClient,
    prompt_key: str,
    task_context: str,
    error_text: str,
    *,
    stage_name: str,
) -> bool:
    """Run a single binary yes/no judgment for one compiler error.

    Uses run_binary_judgment with a 1-element list. Returns the verdict.
    On parse failure, returns False (R2 default-deny: don't classify
    into this category if uncertain).
    """
    system_prompt = SYSTEM_PROMPTS[prompt_key]

    verdict_map, omitted = run_binary_judgment(
        [error_text],
        system_prompt=system_prompt,
        task_context=task_context,
        llm=llm,
        format_item=lambda err: f"\nCompiler error:\n{err}\n\nIs this error of this type?",
        stage_name=stage_name,
        item_key=lambda _: 0,
        default_action="not_classified",
    )

    return verdict_map.get(0, False)
```

### Helper: `_resolve_missing_include()`

```python
def _resolve_missing_include(
    comp_error: str,
    llm: LoggedLLMClient,
) -> str:
    """Focused 1.7B call: given a compiler error classified as missing include, determine which header.

    Returns the header filename (e.g. "stdio.h" or "myproject.h").
    Raises ValueError if LLM returns unparseable output.
    """
    system = SYSTEM_PROMPTS["error_which_include"]
    truncated = comp_error[:800] if len(comp_error) > 800 else comp_error

    validate_prompt_budget(
        truncated, system,
        llm.config.context_window, llm.config.max_tokens,
        "error_which_include",
    )

    response = llm.complete(truncated, system=system)
    header = response.text.strip().strip('"').strip("'").strip("<>")

    if not header or " " in header:
        raise ValueError(
            f"LLM returned unparseable include header: {response.text!r}"
        )

    return header
```

## 9. Deterministic Recovery: `add_include_to_file()`

```python
def add_include_to_file(
    file_path: str,
    header: str,
    scaffold_content: dict[str, str],
    repo_path: Path,
) -> None:
    """Insert #include <header> at the top of a C source file.

    Inserts after any existing #include block to keep includes grouped.
    Updates both scaffold_content dict and the file on disk.

    Raises ValueError if file_path is not in scaffold_content.
    """
    if file_path not in scaffold_content:
        raise ValueError(f"Cannot add include to unknown file: {file_path}")

    content = scaffold_content[file_path]
    include_line = f"#include <{header}>"

    # Don't add if already present
    if include_line in content:
        return

    lines = content.split("\n")
    insert_idx = 0

    # Find the last #include line and insert after it
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#include"):
            insert_idx = i + 1

    lines.insert(insert_idx, include_line)
    new_content = "\n".join(lines)

    # Update both in-memory dict and on-disk file
    scaffold_content[file_path] = new_content
    (repo_path / file_path).write_text(new_content, encoding="utf-8")
```

## 10. Integration Point: Modify the Compile-and-Retry Loop

**File**: `src/clean_room_agent/orchestrator/runner.py`, lines 896-989

The current flow at line 943-953 is approximately:
```python
if not compiled:
    if attempt < max_retries:
        compiler_error = comp_error
        continue
    else:
        # rollback ...
```

The modified flow adds a classification step between `not compiled` and the retry decision. The key change is that `missing_include` with a deterministic fix does NOT consume a retry attempt.

The `use_error_classification` variable is resolved once before the per-function loop:

```python
use_error_classification = use_decomposed and _use_decomposed_error_classification(config)
```

When `use_error_classification` is True and compilation fails:
1. Call `classify_compiler_error()` to categorize the error
2. For `missing_include` with `suggested_include`: call `add_include_to_file()`, re-compile. If re-compile succeeds, don't consume a retry attempt. If re-compile fails, set enriched `compiler_error` and continue retry.
3. For `signature_mismatch` and `missing_definition`: set enriched `compiler_error` with diagnostic context and continue retry.
4. For `logic_error`: set `compiler_error` to raw error and continue retry (existing behavior).

When `use_error_classification` is False, the existing blind retry path is used unchanged.

## 11. What Does NOT Change

- **`execute_function_implement()`** in `scaffold.py` -- signature and behavior unchanged. Still receives `compiler_error` kwarg.
- **`build_function_implement_prompt()`** in `prompts.py` -- unchanged. The `compiler_error` kwarg already renders the error in `<compiler_error>` tags.
- **`FunctionStub` dataclass** -- unchanged.
- **`StepResult` dataclass** -- unchanged.
- **`compile_single_file()`** -- unchanged. Still returns `(bool, str)`.
- **`ScaffoldResult`** -- unchanged.
- **The monolithic scaffold path** -- not affected.
- **All retrieval stages, pipeline, context assembly** -- no changes.
- **`run_binary_judgment()`** -- used as-is, no modifications needed.

## 12. Call Volume Analysis

Per compile failure (per function, per retry attempt):

**Best case (deterministic pre-filter matches)**: 0 LLM calls. `_deterministic_include_check()` matches the symbol in `_SYMBOL_TO_HEADER`, inserts the include, re-compiles. Total cost: 0 LLM calls + 1 re-compilation.

**Typical case (1 binary hit)**: 1-2 binary 0.6B calls (short-circuit at first "yes") + possibly 1 focused 1.7B call (for `_resolve_missing_include`). Total cost: 1-3 LLM calls.

**Worst case (all binaries say "no", logic error)**: 3 binary 0.6B calls + 0 focused calls = 3 LLM calls. Falls through to the existing retry mechanism.

The retry loop runs at most `max_retries` times (typically 1), so the worst case per function is 3 additional binary calls per retry. For a task with 10 functions and 50% compile failure rate, that is ~15 extra binary calls -- negligible for 0.6B at ~100ms each.

## 13. Test Plan

New test file: `tests/execute/test_compiler_error_classifier.py`

### Unit tests for `_deterministic_include_check()`

```
test_implicit_declaration_printf_returns_stdio
test_implicit_declaration_malloc_returns_stdlib
test_unknown_type_uint32_t_returns_stdint
test_undeclared_identifier_strlen_returns_string
test_already_included_returns_none
test_unknown_symbol_returns_none
test_non_matching_error_returns_none
```

### Unit tests for `add_include_to_file()`

```
test_inserts_after_existing_includes
test_inserts_at_top_when_no_includes
test_no_duplicate_if_already_present
test_unknown_file_raises_valueerror
test_updates_scaffold_content_dict
test_updates_file_on_disk
```

### Unit tests for `_run_single_binary()`

```
test_returns_true_on_yes
test_returns_false_on_no
test_returns_false_on_parse_failure
```

### Unit tests for `classify_compiler_error()` with mock LLM

```
test_deterministic_prefilter_short_circuits_llm
test_binary_missing_include_yes
test_binary_signature_mismatch_yes
test_binary_missing_definition_yes
test_all_binaries_no_returns_logic_error
test_short_circuit_stops_after_first_yes
```

### Integration tests (config toggle in runner)

```
test_use_decomposed_error_classification_default_false
test_use_decomposed_error_classification_enabled
```

Expected test count: approximately 20 tests.

## 14. Summary of File Changes

| File | Change |
|------|--------|
| `src/clean_room_agent/execute/compiler_error_classifier.py` | **New file**: `_SYMBOL_TO_HEADER` mapping, regex patterns, `_deterministic_include_check()`, `_run_single_binary()`, `_resolve_missing_include()`, `classify_compiler_error()`, `add_include_to_file()`, error category constants. |
| `src/clean_room_agent/execute/dataclasses.py` | Add `CompilerErrorClassification` dataclass. |
| `src/clean_room_agent/execute/prompts.py` | Add 4 new entries to `SYSTEM_PROMPTS` dict: `error_missing_include`, `error_signature_mismatch`, `error_missing_definition`, `error_which_include`. |
| `src/clean_room_agent/orchestrator/runner.py` | Add `_use_decomposed_error_classification()`. Modify compile-and-retry loop to insert classification step. |
| `src/clean_room_agent/config.py` | Add commented template line for `decomposed_error_classification`. |
| `tests/execute/test_compiler_error_classifier.py` | **New file**: ~20 test functions across 5 test classes. |
| `protocols/design_records/per_task_adapter_map.md` | Update A18 decomposition status to "Done". |

## 15. Implementation Sequence

1. Add `CompilerErrorClassification` dataclass to `src/clean_room_agent/execute/dataclasses.py`
2. Add 4 new system prompts to `src/clean_room_agent/execute/prompts.py`
3. Create `src/clean_room_agent/execute/compiler_error_classifier.py` with all functions
4. Add `_use_decomposed_error_classification()` to `src/clean_room_agent/orchestrator/runner.py`
5. Modify the compile-and-retry loop in `runner.py` to integrate classification
6. Add config template line in `src/clean_room_agent/config.py`
7. Write all tests in `tests/execute/test_compiler_error_classifier.py`
8. Update `protocols/design_records/per_task_adapter_map.md` decomposition status

## Critical Files for Implementation

- `src/clean_room_agent/execute/compiler_error_classifier.py` - New file: core classification logic
- `src/clean_room_agent/orchestrator/runner.py` - Config toggle and integration into compile-and-retry loop
- `src/clean_room_agent/execute/prompts.py` - Add 4 new system prompts
- `src/clean_room_agent/execute/dataclasses.py` - Add CompilerErrorClassification dataclass
- `src/clean_room_agent/retrieval/batch_judgment.py` - Reference pattern: `run_binary_judgment()` used for binary classification calls
