"""Test runner and validation for the orchestrator."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import subprocess
from pathlib import Path

from clean_room_agent.db.raw_queries import insert_validation_result
from clean_room_agent.execute.dataclasses import ValidationResult

logger = logging.getLogger(__name__)

# Cap validation output to prevent unbounded context growth in LLM prompts.
# ~1000 tokens conservative (3 chars/token). Full output preserved in raw DB.
_MAX_VALIDATION_OUTPUT_CHARS = 4000


def _cap_output(output: str, label: str) -> str:
    """Truncate output keeping the tail (error summaries are at the bottom)."""
    if len(output) <= _MAX_VALIDATION_OUTPUT_CHARS:
        return output
    truncated = output[-_MAX_VALIDATION_OUTPUT_CHARS:]
    return f"[{label} truncated — showing last {_MAX_VALIDATION_OUTPUT_CHARS} chars]\n{truncated}"


# Patterns for extracting failing test names from common frameworks
_PYTEST_FAIL_PATTERN = re.compile(r"FAILED\s+(\S+)")
_JEST_FAIL_PATTERN = re.compile(r"FAIL\s+(\S+)")


def require_testing_config(config: dict | None) -> dict:
    """Extract [testing] section or raise a hard error.

    Fail-fast: cra solve cannot run without a test command.
    """
    if config is None:
        raise RuntimeError(
            "No config found. Run 'cra init' to create .clean_room/config.toml"
        )
    testing = config.get("testing")
    if testing is None:
        raise RuntimeError(
            "Missing [testing] section in config. "
            "Add [testing] with test_command to .clean_room/config.toml"
        )
    if not testing.get("test_command"):
        raise RuntimeError(
            "Missing or empty test_command in [testing] config. "
            "Set test_command (e.g. 'pytest tests/') in .clean_room/config.toml"
        )
    return testing


def run_validation(
    repo_path: Path,
    config: dict,
    raw_conn: sqlite3.Connection,
    attempt_id: int,
) -> ValidationResult:
    """Run test/lint/type-check commands and log results.

    Only test_command determines success/failure. lint and type_check
    are informational.
    """
    testing_config = require_testing_config(config)
    timeout = testing_config.get("timeout", 120)

    # T64: Bounds-check timeout
    if not isinstance(timeout, int) or timeout <= 0:
        raise RuntimeError(
            f"timeout in [testing] must be a positive integer, got {timeout!r}"
        )

    test_output = _run_command(
        testing_config["test_command"], repo_path, timeout,
    )
    test_success = test_output["returncode"] == 0

    lint_output = None
    if "lint_command" in testing_config:
        lint_result = _run_command(
            testing_config["lint_command"], repo_path, timeout,
        )
        lint_output = lint_result["output"]

    type_check_output = None
    if "type_check_command" in testing_config:
        tc_result = _run_command(
            testing_config["type_check_command"], repo_path, timeout,
        )
        type_check_output = tc_result["output"]

    failing_tests = _extract_failing_tests(test_output["output"])

    # Log full (uncapped) output to raw DB for traceability
    raw_test_output = test_output["output"]
    insert_validation_result(
        raw_conn,
        attempt_id,
        test_success,
        test_output=raw_test_output,
        lint_output=lint_output,
        type_check_output=type_check_output,
        failing_tests=json.dumps(failing_tests) if failing_tests else None,
    )
    raw_conn.commit()

    # Cap outputs for ValidationResult (flows into LLM prompts)
    result = ValidationResult(
        success=test_success,
        test_output=_cap_output(raw_test_output, "test output"),
        lint_output=_cap_output(lint_output, "lint output") if lint_output else None,
        type_check_output=_cap_output(type_check_output, "type check output") if type_check_output else None,
        failing_tests=failing_tests,
    )

    return result


def _run_command(command: str, cwd: Path, timeout: int) -> dict:
    """Run a shell command and capture output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True,
            cwd=str(cwd),
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        return {"returncode": result.returncode, "output": output}
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "output": f"Command timed out after {timeout}s: {command}",
        }


def _extract_failing_tests(output: str) -> list[str]:
    """Best-effort extraction of failing test names from output."""
    failing = []
    # Try pytest pattern
    for match in _PYTEST_FAIL_PATTERN.finditer(output):
        failing.append(match.group(1))
    # Try jest pattern if no pytest matches
    if not failing:
        for match in _JEST_FAIL_PATTERN.finditer(output):
            failing.append(match.group(1))
    return failing
