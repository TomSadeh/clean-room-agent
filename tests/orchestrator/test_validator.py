"""Tests for Phase 3 test runner and validation."""

from unittest.mock import MagicMock, patch

import pytest

from clean_room_agent.db.raw_queries import insert_run_attempt, insert_task_run
from clean_room_agent.execute.dataclasses import ValidationResult
from clean_room_agent.orchestrator.validator import (
    _extract_failing_tests,
    require_testing_config,
    run_validation,
)


class TestRequireTestingConfig:
    def test_valid_config(self):
        config = {"testing": {"test_command": "pytest tests/", "timeout": 120}}
        result = require_testing_config(config)
        assert result["test_command"] == "pytest tests/"

    def test_none_config_raises(self):
        with pytest.raises(RuntimeError, match="No config found"):
            require_testing_config(None)

    def test_missing_testing_section_raises(self):
        with pytest.raises(RuntimeError, match="Missing \\[testing\\]"):
            require_testing_config({"models": {}})

    def test_missing_test_command_raises(self):
        with pytest.raises(RuntimeError, match="Missing or empty test_command"):
            require_testing_config({"testing": {"timeout": 60}})

    def test_missing_timeout_raises(self):
        """A9: missing timeout in [testing] config raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Missing timeout"):
            require_testing_config({"testing": {"test_command": "pytest tests/"}})


class TestExtractFailingTests:
    def test_pytest_failures(self):
        output = """
PASSED tests/test_foo.py::test_good
FAILED tests/test_bar.py::test_bad
FAILED tests/test_baz.py::test_worse
1 passed, 2 failed
"""
        failing = _extract_failing_tests(output)
        assert "tests/test_bar.py::test_bad" in failing
        assert "tests/test_baz.py::test_worse" in failing

    def test_jest_failures(self):
        output = """
PASS src/foo.test.js
FAIL src/bar.test.js
"""
        failing = _extract_failing_tests(output)
        assert "src/bar.test.js" in failing

    def test_no_failures(self):
        output = "3 passed in 0.5s"
        failing = _extract_failing_tests(output)
        assert failing == []


class TestRunValidation:
    def _make_attempt(self, raw_conn):
        run_id = insert_task_run(
            raw_conn, "task-val", "/repo", "implement",
            "model", 32768, 4096, "scope",
        )
        attempt_id = insert_run_attempt(
            raw_conn, run_id, 1, 100, 50, 500, "response", True,
        )
        raw_conn.commit()
        return attempt_id

    @patch("clean_room_agent.orchestrator.validator.subprocess")
    def test_test_passes(self, mock_subprocess, raw_conn, tmp_repo):
        attempt_id = self._make_attempt(raw_conn)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "3 passed in 0.5s"
        mock_result.stderr = ""
        mock_subprocess.run.return_value = mock_result

        config = {"testing": {"test_command": "pytest tests/", "timeout": 120}}
        result = run_validation(tmp_repo, config, raw_conn, attempt_id)

        assert isinstance(result, ValidationResult)
        assert result.success is True
        assert "3 passed" in result.test_output

        # Verify DB record
        row = raw_conn.execute(
            "SELECT * FROM validation_results WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        assert row["success"] == 1

    @patch("clean_room_agent.orchestrator.validator.subprocess")
    def test_test_fails(self, mock_subprocess, raw_conn, tmp_repo):
        import json

        attempt_id = self._make_attempt(raw_conn)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "FAILED tests/test_foo.py::test_bad\n1 failed"
        mock_result.stderr = ""
        mock_subprocess.run.return_value = mock_result

        config = {"testing": {"test_command": "pytest tests/", "timeout": 120}}
        result = run_validation(tmp_repo, config, raw_conn, attempt_id)

        assert result.success is False
        assert "test_bad" in result.failing_tests[0]

        # Verify failing_tests stored as JSON array (T29)
        row = raw_conn.execute(
            "SELECT failing_tests FROM validation_results WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        parsed = json.loads(row["failing_tests"])
        assert isinstance(parsed, list)
        assert any("test_bad" in t for t in parsed)

    @patch("clean_room_agent.orchestrator.validator.subprocess")
    def test_with_lint_and_typecheck(self, mock_subprocess, raw_conn, tmp_repo):
        attempt_id = self._make_attempt(raw_conn)

        def side_effect(cmd, **kwargs):
            result = MagicMock()
            result.stderr = ""
            if "pytest" in cmd:
                result.returncode = 0
                result.stdout = "3 passed"
            elif "ruff" in cmd:
                result.returncode = 1
                result.stdout = "E501 line too long"
            elif "mypy" in cmd:
                result.returncode = 0
                result.stdout = "Success"
            return result

        mock_subprocess.run.side_effect = side_effect

        config = {
            "testing": {
                "test_command": "pytest tests/",
                "lint_command": "ruff check src/",
                "type_check_command": "mypy src/",
                "timeout": 120,
            }
        }
        result = run_validation(tmp_repo, config, raw_conn, attempt_id)

        # Success based only on test_command
        assert result.success is True
        assert result.lint_output == "E501 line too long"
        assert result.type_check_output == "Success"

    @patch("clean_room_agent.orchestrator.validator.subprocess")
    def test_timeout(self, mock_subprocess, raw_conn, tmp_repo):
        import subprocess as real_subprocess

        attempt_id = self._make_attempt(raw_conn)
        mock_subprocess.run.side_effect = real_subprocess.TimeoutExpired("pytest", 10)
        mock_subprocess.TimeoutExpired = real_subprocess.TimeoutExpired

        config = {"testing": {"test_command": "pytest tests/", "timeout": 10}}
        result = run_validation(tmp_repo, config, raw_conn, attempt_id)

        assert result.success is False
        assert "timed out" in result.test_output

    @patch("clean_room_agent.orchestrator.validator.subprocess")
    def test_custom_timeout(self, mock_subprocess, raw_conn, tmp_repo):
        attempt_id = self._make_attempt(raw_conn)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok"
        mock_result.stderr = ""
        mock_subprocess.run.return_value = mock_result

        config = {"testing": {"test_command": "pytest", "timeout": 300}}
        run_validation(tmp_repo, config, raw_conn, attempt_id)

        # Verify timeout was passed to subprocess
        call_kwargs = mock_subprocess.run.call_args[1]
        assert call_kwargs["timeout"] == 300
