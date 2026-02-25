"""Tests for documentation pass: AST verification and execute functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clean_room_agent.execute.dataclasses import PatchEdit, PatchResult, StepResult
from clean_room_agent.execute.documentation import (
    _apply_edits_to_string,
    _language_from_extension,
    execute_documentation_file,
    run_documentation_pass,
    verify_doc_only_edits,
)
from clean_room_agent.llm.client import LLMResponse, ModelConfig


def _make_model_config():
    return ModelConfig(
        model="test-model",
        base_url="http://localhost:11434",
        context_window=32768,
        max_tokens=4096,
    )


# -- verify_doc_only_edits tests --


class TestVerifyDocOnlyEdits:
    def test_accepts_docstring_addition(self):
        original = 'def foo():\n    return 1\n'
        edited = 'def foo():\n    """Return the number one."""\n    return 1\n'
        assert verify_doc_only_edits(original, edited, "python") is True

    def test_accepts_comment_addition(self):
        original = 'x = 1\ny = 2\n'
        edited = '# Set x to 1\nx = 1\n# Set y to 2\ny = 2\n'
        assert verify_doc_only_edits(original, edited, "python") is True

    def test_accepts_comment_modification(self):
        original = '# old comment\nx = 1\n'
        edited = '# new improved comment\nx = 1\n'
        assert verify_doc_only_edits(original, edited, "python") is True

    def test_accepts_module_docstring_addition(self):
        original = 'import os\n\nx = 1\n'
        edited = '"""Module for doing things."""\n\nimport os\n\nx = 1\n'
        assert verify_doc_only_edits(original, edited, "python") is True

    def test_accepts_class_docstring_addition(self):
        original = 'class Foo:\n    x = 1\n'
        edited = 'class Foo:\n    """A class for foo-ing."""\n    x = 1\n'
        assert verify_doc_only_edits(original, edited, "python") is True

    def test_rejects_code_logic_change(self):
        original = 'def foo():\n    return 1\n'
        edited = 'def foo():\n    return 2\n'
        assert verify_doc_only_edits(original, edited, "python") is False

    def test_rejects_variable_rename(self):
        original = 'x = 1\n'
        edited = 'y = 1\n'
        assert verify_doc_only_edits(original, edited, "python") is False

    def test_rejects_import_addition(self):
        original = 'x = 1\n'
        edited = 'import os\nx = 1\n'
        assert verify_doc_only_edits(original, edited, "python") is False

    def test_rejects_mixed_doc_and_code_edits(self):
        original = 'def foo():\n    return 1\n'
        edited = 'def foo():\n    """Added docstring."""\n    return 2\n'
        assert verify_doc_only_edits(original, edited, "python") is False

    def test_rejects_function_signature_change(self):
        original = 'def foo(x):\n    return x\n'
        edited = 'def foo(x, y):\n    """Updated docstring."""\n    return x\n'
        assert verify_doc_only_edits(original, edited, "python") is False

    def test_unsupported_language_returns_false(self):
        assert verify_doc_only_edits("code", "code", "rust") is False

    def test_identical_files_returns_true(self):
        code = 'def foo():\n    """Existing doc."""\n    return 1\n'
        assert verify_doc_only_edits(code, code, "python") is True

    def test_accepts_inline_comment_change(self):
        original = 'x = 1  # old\n'
        edited = 'x = 1  # new explanation\n'
        assert verify_doc_only_edits(original, edited, "python") is True

    def test_accepts_multiline_docstring_change(self):
        original = (
            'def foo():\n'
            '    """Short doc."""\n'
            '    return 1\n'
        )
        edited = (
            'def foo():\n'
            '    """Longer documentation.\n'
            '\n'
            '    Args:\n'
            '        None\n'
            '\n'
            '    Returns:\n'
            '        int: Always 1.\n'
            '    """\n'
            '    return 1\n'
        )
        assert verify_doc_only_edits(original, edited, "python") is True


# -- _language_from_extension tests --


class TestLanguageFromExtension:
    def test_python(self):
        assert _language_from_extension("foo.py") == "python"

    def test_unsupported_js(self):
        assert _language_from_extension("foo.js") is None

    def test_unsupported_ts(self):
        assert _language_from_extension("foo.ts") is None

    def test_unknown(self):
        assert _language_from_extension("foo.rs") is None


# -- _apply_edits_to_string tests --


class TestApplyEditsToString:
    def test_successful_apply(self):
        edit = PatchEdit(file_path="a.py", search="old code", replacement="new code")
        result = _apply_edits_to_string("old code here", [edit])
        assert result == "new code here"

    def test_not_found_returns_none(self):
        edit = PatchEdit(file_path="a.py", search="missing", replacement="new")
        assert _apply_edits_to_string("some code", [edit]) is None

    def test_multiple_matches_returns_none(self):
        edit = PatchEdit(file_path="a.py", search="x", replacement="y")
        assert _apply_edits_to_string("x x", [edit]) is None


# -- execute_documentation_file tests --


class TestExecuteDocumentationFile:
    def test_unsupported_language_returns_success_empty(self, tmp_path):
        llm = MagicMock()
        result = execute_documentation_file(
            "main.rs", tmp_path, "task", "part",
            llm, _make_model_config(),
        )
        assert result.success is True
        assert result.edits == []
        llm.complete.assert_not_called()

    def test_successful_doc_generation(self, tmp_path):
        (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.config = _make_model_config()
        mock_llm.complete.return_value = LLMResponse(
            text=(
                '<edit file="a.py">\n'
                '<search>\ndef foo():\n</search>\n'
                '<replacement>\ndef foo():\n    """Return the number one."""\n</replacement>\n'
                '</edit>'
            ),
            thinking=None,
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=200,
        )

        result = execute_documentation_file(
            "a.py", tmp_path, "task", "part",
            mock_llm, _make_model_config(),
        )
        assert result.success is True
        assert len(result.edits) == 1

    def test_parse_failure_returns_failure(self, tmp_path):
        (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.config = _make_model_config()
        mock_llm.complete.return_value = LLMResponse(
            text='<edit file="a.py">garbage no search tags</edit>',
            thinking=None, prompt_tokens=100, completion_tokens=50, latency_ms=200,
        )

        result = execute_documentation_file(
            "a.py", tmp_path, "task", "part",
            mock_llm, _make_model_config(),
        )
        assert result.success is False
        assert "Failed to parse" in result.error_info

    def test_no_edits_in_response_returns_success(self, tmp_path):
        (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.config = _make_model_config()
        mock_llm.complete.return_value = LLMResponse(
            text="The file already has good documentation. No changes needed.",
            thinking=None, prompt_tokens=100, completion_tokens=20, latency_ms=100,
        )

        result = execute_documentation_file(
            "a.py", tmp_path, "task", "part",
            mock_llm, _make_model_config(),
        )
        assert result.success is True
        assert result.edits == []

    def test_verification_failure_rejects_code_edits(self, tmp_path):
        (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.config = _make_model_config()
        # LLM tries to change code logic
        mock_llm.complete.return_value = LLMResponse(
            text=(
                '<edit file="a.py">\n'
                '<search>\n    return 1\n</search>\n'
                '<replacement>\n    return 2\n</replacement>\n'
                '</edit>'
            ),
            thinking=None, prompt_tokens=100, completion_tokens=50, latency_ms=200,
        )

        result = execute_documentation_file(
            "a.py", tmp_path, "task", "part",
            mock_llm, _make_model_config(),
        )
        assert result.success is False
        assert "Edits modify code logic" in result.error_info

    def test_edit_search_mismatch_returns_failure(self, tmp_path):
        (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.config = _make_model_config()
        mock_llm.complete.return_value = LLMResponse(
            text=(
                '<edit file="a.py">\n'
                '<search>\ndef bar():\n</search>\n'
                '<replacement>\ndef bar():\n    """Doc."""\n</replacement>\n'
                '</edit>'
            ),
            thinking=None, prompt_tokens=100, completion_tokens=50, latency_ms=200,
        )

        result = execute_documentation_file(
            "a.py", tmp_path, "task", "part",
            mock_llm, _make_model_config(),
        )
        assert result.success is False
        assert "did not match" in result.error_info


# -- run_documentation_pass tests --


class TestRunDocumentationPass:
    def test_multiple_files_partial_success(self, tmp_path):
        (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("def bar():\n    return 2\n", encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.config = _make_model_config()

        # First file: success with doc edit. Second file: code edit (rejected)
        mock_llm.complete.side_effect = [
            LLMResponse(
                text=(
                    '<edit file="a.py">\n'
                    '<search>\ndef foo():\n</search>\n'
                    '<replacement>\ndef foo():\n    """Return one."""\n</replacement>\n'
                    '</edit>'
                ),
                thinking=None, prompt_tokens=100, completion_tokens=50, latency_ms=200,
            ),
            LLMResponse(
                text=(
                    '<edit file="b.py">\n'
                    '<search>\n    return 2\n</search>\n'
                    '<replacement>\n    return 3\n</replacement>\n'
                    '</edit>'
                ),
                thinking=None, prompt_tokens=100, completion_tokens=50, latency_ms=200,
            ),
        ]

        results = run_documentation_pass(
            ["a.py", "b.py"], tmp_path, "task", "part",
            mock_llm, _make_model_config(),
        )

        # Only a.py should have succeeded
        assert len(results) == 1
        assert results[0].success is True
        assert "a.py" in results[0].files_modified

    def test_no_edits_needed(self, tmp_path):
        (tmp_path / "a.py").write_text('"""Documented."""\ndef foo():\n    """Good doc."""\n    return 1\n', encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.config = _make_model_config()
        mock_llm.complete.return_value = LLMResponse(
            text="No changes needed.",
            thinking=None, prompt_tokens=100, completion_tokens=20, latency_ms=100,
        )

        results = run_documentation_pass(
            ["a.py"], tmp_path, "task", "part",
            mock_llm, _make_model_config(),
        )
        assert results == []

    def test_unsupported_file_skipped(self, tmp_path):
        (tmp_path / "main.rs").write_text("fn main() {}", encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.config = _make_model_config()

        results = run_documentation_pass(
            ["main.rs"], tmp_path, "task", "part",
            mock_llm, _make_model_config(),
        )
        assert results == []
        mock_llm.complete.assert_not_called()
