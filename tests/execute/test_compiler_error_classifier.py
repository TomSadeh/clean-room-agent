"""Tests for compiler error classification (A18 decomposition)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from clean_room_agent.execute.compiler_error_classifier import (
    ERROR_CAT_LOGIC_ERROR,
    ERROR_CAT_MISSING_DEFINITION,
    ERROR_CAT_MISSING_INCLUDE,
    ERROR_CAT_SIGNATURE_MISMATCH,
    VALID_ERROR_CATEGORIES,
    _deterministic_include_check,
    _resolve_missing_include,
    _run_single_binary,
    add_include_to_file,
    classify_compiler_error,
)
from clean_room_agent.execute.dataclasses import CompilerErrorClassification, FunctionStub
from clean_room_agent.execute.prompts import SYSTEM_PROMPTS
from clean_room_agent.llm.client import LLMResponse, ModelConfig


@pytest.fixture
def model_config():
    return ModelConfig(
        model="test-model",
        base_url="http://localhost:11434",
        context_window=32768,
        max_tokens=4096,
    )


@pytest.fixture
def stub():
    return FunctionStub(
        name="hash_insert",
        file_path="src/hash_table.c",
        signature="int hash_insert(HashTable *ht, const char *key, int value)",
        docstring="Insert a key-value pair into the hash table.",
        start_line=10,
        end_line=20,
        header_file="src/hash_table.h",
    )


@pytest.fixture
def scaffold_content():
    return {
        "src/hash_table.c": '#include "hash_table.h"\n\nint hash_insert(HashTable *ht, const char *key, int value) {\n    return 0;\n}\n',
        "src/hash_table.h": '#ifndef HASH_TABLE_H\n#define HASH_TABLE_H\n\ntypedef struct { int x; } HashTable;\n\n#endif\n',
    }


def _make_mock_llm(model_config, responses):
    """Create a mock LoggedLLMClient that returns responses in order."""
    llm = MagicMock()
    llm.config = model_config
    llm.flush.return_value = []

    response_iter = iter(responses)
    def complete_side_effect(prompt, system=None):
        text = next(response_iter)
        return LLMResponse(
            text=text, thinking=None,
            prompt_tokens=100, completion_tokens=50, latency_ms=100,
        )
    llm.complete.side_effect = complete_side_effect
    return llm


# ---------------------------------------------------------------------------
# Tests for _deterministic_include_check()
# ---------------------------------------------------------------------------


class TestDeterministicIncludeCheck:
    def test_implicit_declaration_printf_returns_stdio(self):
        error = "src/main.c:10:5: warning: implicit declaration of function 'printf'"
        content = {"src/main.c": "int main() { printf(\"hi\"); }"}
        result = _deterministic_include_check(error, "src/main.c", content)
        assert result is not None
        assert result.category == ERROR_CAT_MISSING_INCLUDE
        assert result.suggested_include == "stdio.h"

    def test_implicit_declaration_malloc_returns_stdlib(self):
        error = "src/alloc.c:5:12: warning: implicit declaration of function 'malloc'"
        content = {"src/alloc.c": "void *p = malloc(10);"}
        result = _deterministic_include_check(error, "src/alloc.c", content)
        assert result is not None
        assert result.suggested_include == "stdlib.h"

    def test_unknown_type_uint32_t_returns_stdint(self):
        error = "src/types.c:3:1: error: unknown type name 'uint32_t'"
        content = {"src/types.c": "uint32_t x = 0;"}
        result = _deterministic_include_check(error, "src/types.c", content)
        assert result is not None
        assert result.suggested_include == "stdint.h"

    def test_undeclared_identifier_strlen_returns_string(self):
        error = "src/str.c:4:9: error: 'strlen' undeclared"
        content = {"src/str.c": "int n = strlen(s);"}
        result = _deterministic_include_check(error, "src/str.c", content)
        assert result is not None
        assert result.suggested_include == "string.h"

    def test_already_included_returns_none(self):
        error = "src/main.c:10:5: warning: implicit declaration of function 'printf'"
        content = {"src/main.c": "#include <stdio.h>\nint main() { printf(\"hi\"); }"}
        result = _deterministic_include_check(error, "src/main.c", content)
        assert result is None

    def test_unknown_symbol_returns_none(self):
        error = "src/main.c:10:5: warning: implicit declaration of function 'my_custom_func'"
        content = {"src/main.c": "my_custom_func();"}
        result = _deterministic_include_check(error, "src/main.c", content)
        assert result is None

    def test_non_matching_error_returns_none(self):
        error = "src/main.c:10:5: error: expected ';' after expression"
        content = {"src/main.c": "int x = 5"}
        result = _deterministic_include_check(error, "src/main.c", content)
        assert result is None

    def test_missing_scaffold_file_raises_value_error(self):
        """M3: Missing file_path in scaffold_content is a programming error."""
        error = "src/main.c:10:5: warning: implicit declaration of function 'printf'"
        with pytest.raises(ValueError, match="Scaffold file not in scaffold_content"):
            _deterministic_include_check(error, "src/missing.c", {"src/other.c": ""})


# ---------------------------------------------------------------------------
# Tests for add_include_to_file()
# ---------------------------------------------------------------------------


class TestAddIncludeToFile:
    def test_inserts_after_existing_includes(self, tmp_path):
        content = '#include "hash_table.h"\n\nint main() { return 0; }\n'
        file_path = "src/main.c"
        scaffold = {file_path: content}
        (tmp_path / "src").mkdir()
        (tmp_path / file_path).write_text(content, encoding="utf-8")

        add_include_to_file(file_path, "stdio.h", scaffold, tmp_path)

        assert "#include <stdio.h>" in scaffold[file_path]
        lines = scaffold[file_path].split("\n")
        # stdio.h should be after hash_table.h include
        hash_idx = next(i for i, l in enumerate(lines) if "hash_table.h" in l)
        stdio_idx = next(i for i, l in enumerate(lines) if "stdio.h" in l)
        assert stdio_idx == hash_idx + 1

    def test_inserts_at_top_when_no_includes(self, tmp_path):
        content = "int main() { return 0; }\n"
        file_path = "src/main.c"
        scaffold = {file_path: content}
        (tmp_path / "src").mkdir()
        (tmp_path / file_path).write_text(content, encoding="utf-8")

        add_include_to_file(file_path, "stdlib.h", scaffold, tmp_path)

        lines = scaffold[file_path].split("\n")
        assert lines[0] == "#include <stdlib.h>"

    def test_no_duplicate_if_already_present(self, tmp_path):
        content = '#include <stdio.h>\nint main() { return 0; }\n'
        file_path = "src/main.c"
        scaffold = {file_path: content}
        (tmp_path / "src").mkdir()
        (tmp_path / file_path).write_text(content, encoding="utf-8")

        add_include_to_file(file_path, "stdio.h", scaffold, tmp_path)

        assert scaffold[file_path].count("#include <stdio.h>") == 1

    def test_unknown_file_raises_valueerror(self, tmp_path):
        with pytest.raises(ValueError, match="unknown file"):
            add_include_to_file("nonexistent.c", "stdio.h", {}, tmp_path)

    def test_updates_scaffold_content_dict(self, tmp_path):
        content = "int x;\n"
        file_path = "src/a.c"
        scaffold = {file_path: content}
        (tmp_path / "src").mkdir()
        (tmp_path / file_path).write_text(content, encoding="utf-8")

        add_include_to_file(file_path, "math.h", scaffold, tmp_path)

        assert "#include <math.h>" in scaffold[file_path]

    def test_updates_file_on_disk(self, tmp_path):
        content = "int x;\n"
        file_path = "src/a.c"
        scaffold = {file_path: content}
        (tmp_path / "src").mkdir()
        (tmp_path / file_path).write_text(content, encoding="utf-8")

        add_include_to_file(file_path, "math.h", scaffold, tmp_path)

        disk_content = (tmp_path / file_path).read_text(encoding="utf-8")
        assert "#include <math.h>" in disk_content


# ---------------------------------------------------------------------------
# Tests for _run_single_binary()
# ---------------------------------------------------------------------------


class TestRunSingleBinary:
    def test_returns_true_on_yes(self, model_config):
        llm = _make_mock_llm(model_config, ["yes"])
        result = _run_single_binary(
            llm, "error_missing_include", "Function: test", "error text",
            stage_name="test_stage",
        )
        assert result is True

    def test_returns_false_on_no(self, model_config):
        llm = _make_mock_llm(model_config, ["no"])
        result = _run_single_binary(
            llm, "error_missing_include", "Function: test", "error text",
            stage_name="test_stage",
        )
        assert result is False

    def test_returns_false_on_parse_failure(self, model_config):
        llm = _make_mock_llm(model_config, ["maybe probably"])
        result = _run_single_binary(
            llm, "error_missing_include", "Function: test", "error text",
            stage_name="test_stage",
        )
        assert result is False


# ---------------------------------------------------------------------------
# Tests for classify_compiler_error() with mock LLM
# ---------------------------------------------------------------------------


class TestClassifyCompilerError:
    def test_deterministic_prefilter_short_circuits_llm(self, model_config, stub, scaffold_content):
        error = "src/hash_table.c:15:5: warning: implicit declaration of function 'printf'"
        llm = _make_mock_llm(model_config, [])  # No responses — should not call LLM
        result = classify_compiler_error(error, "src/hash_table.c", scaffold_content, stub, llm)
        assert result.category == ERROR_CAT_MISSING_INCLUDE
        assert result.suggested_include == "stdio.h"
        llm.complete.assert_not_called()

    def test_binary_missing_include_yes(self, model_config, stub, scaffold_content):
        error = "src/hash_table.c:15:5: error: some strange include error"
        # Binary says yes → resolve says "mylib.h"
        llm = _make_mock_llm(model_config, ["yes", "mylib.h"])
        result = classify_compiler_error(error, "src/hash_table.c", scaffold_content, stub, llm)
        assert result.category == ERROR_CAT_MISSING_INCLUDE
        assert result.suggested_include == "mylib.h"

    def test_binary_signature_mismatch_yes(self, model_config, stub, scaffold_content):
        error = "src/hash_table.c:15:5: error: conflicting types for 'hash_insert'"
        # Binary 1 (missing include) says no, binary 2 (sig mismatch) says yes
        llm = _make_mock_llm(model_config, ["no", "yes"])
        result = classify_compiler_error(error, "src/hash_table.c", scaffold_content, stub, llm)
        assert result.category == ERROR_CAT_SIGNATURE_MISMATCH
        assert result.diagnostic_context is not None

    def test_binary_missing_definition_yes(self, model_config, stub, scaffold_content):
        error = "undefined reference to 'helper_func'"
        # Binary 1 no, binary 2 no, binary 3 yes
        llm = _make_mock_llm(model_config, ["no", "no", "yes"])
        result = classify_compiler_error(error, "src/hash_table.c", scaffold_content, stub, llm)
        assert result.category == ERROR_CAT_MISSING_DEFINITION
        assert result.diagnostic_context is not None

    def test_all_binaries_no_returns_logic_error(self, model_config, stub, scaffold_content):
        error = "src/hash_table.c:20:10: error: expected ';' after expression"
        # All 3 binaries say no
        llm = _make_mock_llm(model_config, ["no", "no", "no"])
        result = classify_compiler_error(error, "src/hash_table.c", scaffold_content, stub, llm)
        assert result.category == ERROR_CAT_LOGIC_ERROR
        assert result.diagnostic_context is None

    def test_default_logic_error_logs_warning(self, model_config, stub, scaffold_content, caplog):
        """M4: R2 says log a warning when the default fires."""
        import logging
        error = "src/hash_table.c:20:10: error: expected ';' after expression"
        llm = _make_mock_llm(model_config, ["no", "no", "no"])
        with caplog.at_level(logging.WARNING):
            classify_compiler_error(error, "src/hash_table.c", scaffold_content, stub, llm)
        assert any("default logic_error" in r.message for r in caplog.records)

    def test_short_circuit_stops_after_first_yes(self, model_config, stub, scaffold_content):
        error = "some error"
        # Binary 1 (missing include) says yes → resolve says "string.h"
        llm = _make_mock_llm(model_config, ["yes", "string.h"])
        result = classify_compiler_error(error, "src/hash_table.c", scaffold_content, stub, llm)
        assert result.category == ERROR_CAT_MISSING_INCLUDE
        # Should have made exactly 2 calls: binary + resolve
        assert llm.complete.call_count == 2


# ---------------------------------------------------------------------------
# Tests for _resolve_missing_include()
# ---------------------------------------------------------------------------


class TestResolveMissingInclude:
    def test_strips_quotes_and_brackets(self, model_config):
        llm = _make_mock_llm(model_config, ['"<stdio.h>"'])
        result = _resolve_missing_include("some error", llm)
        assert result == "stdio.h"

    def test_raises_on_empty_response(self, model_config):
        llm = _make_mock_llm(model_config, [""])
        with pytest.raises(ValueError, match="unparseable"):
            _resolve_missing_include("some error", llm)

    def test_raises_on_response_with_spaces(self, model_config):
        llm = _make_mock_llm(model_config, ["I think you need stdio.h"])
        with pytest.raises(ValueError, match="unparseable"):
            _resolve_missing_include("some error", llm)


# ---------------------------------------------------------------------------
# Tests for config toggle
# ---------------------------------------------------------------------------


class TestConfigToggle:
    def test_use_decomposed_error_classification_default_false(self):
        from clean_room_agent.orchestrator.runner import _use_decomposed_error_classification
        assert _use_decomposed_error_classification({}) is False

    def test_use_decomposed_error_classification_enabled(self):
        from clean_room_agent.orchestrator.runner import _use_decomposed_error_classification
        config = {"orchestrator": {"decomposed_error_classification": True}}
        assert _use_decomposed_error_classification(config) is True


# ---------------------------------------------------------------------------
# Tests for dataclass
# ---------------------------------------------------------------------------


class TestCompilerErrorClassification:
    def test_valid_categories(self):
        assert len(VALID_ERROR_CATEGORIES) == 4
        assert ERROR_CAT_MISSING_INCLUDE in VALID_ERROR_CATEGORIES
        assert ERROR_CAT_SIGNATURE_MISMATCH in VALID_ERROR_CATEGORIES
        assert ERROR_CAT_MISSING_DEFINITION in VALID_ERROR_CATEGORIES
        assert ERROR_CAT_LOGIC_ERROR in VALID_ERROR_CATEGORIES

    def test_serialization_roundtrip(self):
        cls = CompilerErrorClassification(
            category=ERROR_CAT_MISSING_INCLUDE,
            raw_error="implicit declaration of function 'printf'",
            suggested_include="stdio.h",
            diagnostic_context="Missing #include <stdio.h>",
        )
        d = cls.to_dict()
        restored = CompilerErrorClassification.from_dict(d)
        assert restored.category == cls.category
        assert restored.raw_error == cls.raw_error
        assert restored.suggested_include == cls.suggested_include
        assert restored.diagnostic_context == cls.diagnostic_context

    def test_required_fields_enforced(self):
        with pytest.raises(ValueError, match="missing required key"):
            CompilerErrorClassification.from_dict({"category": "logic_error"})

    def test_non_empty_enforced(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            CompilerErrorClassification(category="", raw_error="error")


# ---------------------------------------------------------------------------
# Tests for system prompts
# ---------------------------------------------------------------------------


class TestSystemPrompts:
    def test_all_error_prompts_registered(self):
        assert "error_missing_include" in SYSTEM_PROMPTS
        assert "error_signature_mismatch" in SYSTEM_PROMPTS
        assert "error_missing_definition" in SYSTEM_PROMPTS
        assert "error_which_include" in SYSTEM_PROMPTS

    def test_binary_prompts_end_with_yes_no(self):
        for key in ("error_missing_include", "error_signature_mismatch", "error_missing_definition"):
            prompt = SYSTEM_PROMPTS[key]
            assert '"yes" or "no"' in prompt
