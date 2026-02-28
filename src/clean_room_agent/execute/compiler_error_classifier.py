"""Compiler error classification for the compile-and-retry path (A18 decomposition).

Inserts a classification step between compilation failure and retry, replacing
the blind pass-through of raw compiler errors. Deterministic pre-filter catches
known missing includes without any LLM call. Sequential binary classifiers
short-circuit at the first positive match for remaining errors.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from clean_room_agent.execute.dataclasses import CompilerErrorClassification, FunctionStub
from clean_room_agent.execute.prompts import SYSTEM_PROMPTS
from clean_room_agent.llm.client import LoggedLLMClient
from clean_room_agent.retrieval.batch_judgment import run_binary_judgment
from clean_room_agent.token_estimation import validate_prompt_budget

logger = logging.getLogger(__name__)

# -- Error category constants --

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

# -- Deterministic pre-filter data --

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

# -- Regex patterns for the deterministic check --

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
