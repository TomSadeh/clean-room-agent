"""Tests for decomposed scaffold (multi-stage scaffold generation)."""

import json
from unittest.mock import MagicMock

import pytest

from clean_room_agent.execute.dataclasses import (
    FunctionStub,
    InterfaceEnumeration,
    InterfaceFunction,
    InterfaceType,
    PartPlan,
    PlanStep,
    ScaffoldResult,
)
from clean_room_agent.execute.decomposed_scaffold import (
    _assemble_scaffold_result,
    _generate_deterministic_stubs,
    _run_header_generation,
    _stub_return_for_type,
    decomposed_scaffold,
    select_kb_patterns_for_function,
)
from clean_room_agent.execute.parsers import parse_scaffold_response
from clean_room_agent.llm.client import LLMResponse, ModelConfig
from clean_room_agent.retrieval.dataclasses import (
    BudgetConfig,
    ContextPackage,
    FileContent,
    TaskQuery,
)


@pytest.fixture
def model_config():
    return ModelConfig(
        model="test-model",
        base_url="http://localhost:11434",
        context_window=32768,
        max_tokens=4096,
    )


@pytest.fixture
def context_package():
    task = TaskQuery(
        raw_task="Implement hash table",
        task_id="test-scaffold-001",
        mode="implement",
        repo_id=1,
    )
    return ContextPackage(
        task=task,
        files=[
            FileContent(
                file_id=1, path="src/main.c", language="c",
                content="int main() { return 0; }", token_estimate=10,
                detail_level="primary",
            ),
        ],
        total_token_estimate=10,
        budget=BudgetConfig(context_window=32768, reserved_tokens=4096),
    )


@pytest.fixture
def part_plan():
    return PartPlan(
        part_id="p1",
        task_summary="Implement hash table",
        steps=[
            PlanStep(id="s1", description="Create hash table data structure",
                     target_files=["src/hash.h", "src/hash.c"]),
            PlanStep(id="s2", description="Implement insert and lookup",
                     target_files=["src/hash.c"]),
        ],
        rationale="Hash table implementation",
    )


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


# -- InterfaceEnumeration dataclass tests --

class TestInterfaceEnum:
    def test_valid_json(self, model_config):
        response_json = json.dumps({
            "types": [
                {"name": "HashEntry", "kind": "struct",
                 "fields_description": "key-value pair", "file_path": "hash.h"},
            ],
            "functions": [
                {"name": "hash_insert", "return_type": "int",
                 "params": "HashTable *ht, const char *key, void *value",
                 "purpose": "Insert key-value pair",
                 "file_path": "hash.h", "source_file": "hash.c"},
            ],
            "includes": ["stdlib.h", "string.h"],
        })
        llm = _make_mock_llm(model_config, [response_json])
        result = parse_scaffold_response(response_json, "interface_enum")

        assert isinstance(result, InterfaceEnumeration)
        assert len(result.types) == 1
        assert result.types[0].name == "HashEntry"
        assert len(result.functions) == 1
        assert result.functions[0].name == "hash_insert"
        assert result.includes == ["stdlib.h", "string.h"]

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="JSON"):
            parse_scaffold_response("not valid json", "interface_enum")

    def test_empty_functions_raises(self):
        response_json = json.dumps({
            "types": [
                {"name": "Foo", "kind": "struct",
                 "fields_description": "bar", "file_path": "foo.h"},
            ],
            "functions": [],
            "includes": [],
        })
        with pytest.raises(ValueError, match="no functions"):
            parse_scaffold_response(response_json, "interface_enum")

    def test_unknown_pass_type_raises(self):
        with pytest.raises(ValueError, match="Unknown scaffold pass_type"):
            parse_scaffold_response("{}", "bogus")

    def test_serialization_round_trip(self):
        ie = InterfaceEnumeration(
            types=[InterfaceType(name="Foo", kind="struct",
                                 fields_description="bar", file_path="foo.h")],
            functions=[InterfaceFunction(name="foo_init", return_type="int",
                                         params="void", purpose="Initialize",
                                         file_path="foo.h", source_file="foo.c")],
            includes=["stdio.h"],
        )
        d = ie.to_dict()
        ie2 = InterfaceEnumeration.from_dict(d)
        assert ie2.types[0].name == "Foo"
        assert ie2.functions[0].name == "foo_init"
        assert ie2.includes == ["stdio.h"]


# -- Header generation tests --

class TestHeaderGeneration:
    def test_single_header(self, context_package, model_config):
        enum_result = InterfaceEnumeration(
            types=[InterfaceType(name="Node", kind="struct",
                                 fields_description="data + next pointer",
                                 file_path="list.h")],
            functions=[InterfaceFunction(name="list_push", return_type="void",
                                         params="Node **head, int data",
                                         purpose="Push to front",
                                         file_path="list.h", source_file="list.c")],
            includes=["stdlib.h"],
        )
        # LLM outputs raw header content (no XML wrapping)
        header_response = (
            '#ifndef LIST_H\n#define LIST_H\n'
            'typedef struct Node { int data; struct Node *next; } Node;\n'
            'void list_push(Node **head, int data);\n'
            '#endif'
        )
        llm = _make_mock_llm(model_config, [header_response])
        contents = _run_header_generation(
            enum_result, context_package, "test task", llm, None,
        )
        assert len(contents) == 1
        assert "list.h" in contents
        assert "#ifndef LIST_H" in contents["list.h"]

    def test_multiple_headers(self, context_package, model_config):
        enum_result = InterfaceEnumeration(
            types=[
                InterfaceType(name="A", kind="struct",
                              fields_description="type A", file_path="a.h"),
                InterfaceType(name="B", kind="struct",
                              fields_description="type B", file_path="b.h"),
            ],
            functions=[
                InterfaceFunction(name="a_init", return_type="A *",
                                   params="void", purpose="Init A",
                                   file_path="a.h", source_file="a.c"),
                InterfaceFunction(name="b_init", return_type="B *",
                                   params="void", purpose="Init B",
                                   file_path="b.h", source_file="b.c"),
            ],
            includes=["stdlib.h"],
        )
        # Each call returns raw header content
        responses = [
            '#ifndef A_H\n#define A_H\ntypedef struct A {} A;\n'
            'A *a_init(void);\n#endif',
            '#ifndef B_H\n#define B_H\ntypedef struct B {} B;\n'
            'B *b_init(void);\n#endif',
        ]
        llm = _make_mock_llm(model_config, responses)
        contents = _run_header_generation(
            enum_result, context_package, "test task", llm, None,
        )
        assert len(contents) == 2
        assert llm.complete.call_count == 2
        assert set(contents.keys()) == {"a.h", "b.h"}

    def test_strips_code_fences(self, context_package, model_config):
        """LLM wraps output in markdown fences — we strip them."""
        enum_result = InterfaceEnumeration(
            types=[],
            functions=[InterfaceFunction(name="foo", return_type="int",
                                         params="void", purpose="Foo",
                                         file_path="foo.h", source_file="foo.c")],
            includes=[],
        )
        fenced_response = '```c\n#ifndef FOO_H\nint foo(void);\n#endif\n```'
        llm = _make_mock_llm(model_config, [fenced_response])
        contents = _run_header_generation(
            enum_result, context_package, "test task", llm, None,
        )
        assert "#ifndef FOO_H" in contents["foo.h"]
        assert "```" not in contents["foo.h"]

    def test_empty_response_raises(self, context_package, model_config):
        enum_result = InterfaceEnumeration(
            types=[],
            functions=[InterfaceFunction(name="bar", return_type="void",
                                         params="void", purpose="Bar",
                                         file_path="bar.h", source_file="bar.c")],
            includes=[],
        )
        llm = _make_mock_llm(model_config, [""])
        with pytest.raises(ValueError, match="empty content"):
            _run_header_generation(
                enum_result, context_package, "test task", llm, None,
            )


# -- Deterministic stub generation tests --

class TestDeterministicStubs:
    def test_basic_stub_gen(self):
        header = (
            "#ifndef HASH_H\n#define HASH_H\n"
            "int hash_insert(const char *key, int value);\n"
            "#endif\n"
        )
        enum_result = InterfaceEnumeration(
            types=[], includes=["stdlib.h"],
            functions=[InterfaceFunction(
                name="hash_insert", return_type="int",
                params="const char *key, int value", purpose="Insert",
                file_path="hash.h", source_file="hash.c",
            )],
        )
        stubs = _generate_deterministic_stubs(enum_result, {"hash.h": header})

        assert len(stubs) == 1
        assert "hash.c" in stubs
        assert "return 0;" in stubs["hash.c"]
        assert '#include "hash.h"' in stubs["hash.c"]
        assert '#include <stdlib.h>' in stubs["hash.c"]

    def test_void_return(self):
        header = "void do_nothing(void);\n"
        enum_result = InterfaceEnumeration(
            types=[], includes=[],
            functions=[InterfaceFunction(
                name="do_nothing", return_type="void",
                params="void", purpose="Noop",
                file_path="noop.h", source_file="noop.c",
            )],
        )
        stubs = _generate_deterministic_stubs(enum_result, {"noop.h": header})
        assert len(stubs) == 1
        assert "return;" in stubs["noop.c"]

    def test_pointer_return(self):
        header = "char *get_name(int id);\n"
        enum_result = InterfaceEnumeration(
            types=[], includes=[],
            functions=[InterfaceFunction(
                name="get_name", return_type="char *",
                params="int id", purpose="Get name",
                file_path="names.h", source_file="names.c",
            )],
        )
        stubs = _generate_deterministic_stubs(enum_result, {"names.h": header})
        assert len(stubs) == 1
        assert "return NULL;" in stubs["names.c"]

    def test_includes_from_enum(self):
        header = "int foo(void);\n"
        enum_result = InterfaceEnumeration(
            types=[], includes=["stdio.h", "string.h"],
            functions=[InterfaceFunction(
                name="foo", return_type="int",
                params="void", purpose="Foo",
                file_path="foo.h", source_file="foo.c",
            )],
        )
        stubs = _generate_deterministic_stubs(enum_result, {"foo.h": header})
        assert "#include <stdio.h>" in stubs["foo.c"]
        assert "#include <string.h>" in stubs["foo.c"]


# -- Stub return type helper tests --

class TestStubReturnForType:
    def test_void(self):
        assert _stub_return_for_type("void") == "return;"

    def test_pointer(self):
        assert _stub_return_for_type("char *") == "return NULL;"
        assert _stub_return_for_type("int *") == "return NULL;"

    def test_float(self):
        assert _stub_return_for_type("float") == "return 0.0;"
        assert _stub_return_for_type("double") == "return 0.0;"

    def test_bool(self):
        assert _stub_return_for_type("bool") == "return false;"
        assert _stub_return_for_type("_Bool") == "return false;"

    def test_int(self):
        assert _stub_return_for_type("int") == "return 0;"
        assert _stub_return_for_type("size_t") == "return 0;"


# -- AssembleScaffoldResult tests --

class TestAssembleScaffoldResult:
    def test_correct_file_classification(self):
        header_contents = {"a.h": "header content"}
        stub_contents = {"a.c": "stub content"}
        enum_result = InterfaceEnumeration(
            types=[], functions=[InterfaceFunction(
                name="f", return_type="int", params="void", purpose="F",
                file_path="a.h", source_file="a.c",
            )], includes=[],
        )
        result = _assemble_scaffold_result(header_contents, stub_contents, enum_result)

        assert isinstance(result, ScaffoldResult)
        assert result.success is True
        assert result.header_files == ["a.h"]
        assert result.source_files == ["a.c"]
        assert result.edits == []  # Files written directly

    def test_raw_response_contains_metadata(self):
        enum_result = InterfaceEnumeration(
            types=[], functions=[InterfaceFunction(
                name="f", return_type="int", params="void", purpose="F",
                file_path="a.h", source_file="a.c",
            )], includes=[],
        )
        result = _assemble_scaffold_result({}, {}, enum_result)
        parsed = json.loads(result.raw_response)
        assert parsed["decomposed"] is True


# -- End-to-end decomposed scaffold test --

class TestDecomposedScaffoldEndToEnd:
    def test_full_pipeline(self, context_package, part_plan, model_config, tmp_path):
        # Response 1: interface enumeration
        enum_response = json.dumps({
            "types": [
                {"name": "HashEntry", "kind": "struct",
                 "fields_description": "key + value + next", "file_path": "src/hash.h"},
            ],
            "functions": [
                {"name": "hash_insert", "return_type": "int",
                 "params": "HashTable *ht, const char *key",
                 "purpose": "Insert key",
                 "file_path": "src/hash.h", "source_file": "src/hash.c"},
            ],
            "includes": ["stdlib.h"],
        })
        # Response 2: header generation — raw content (no XML wrapping)
        header_response = (
            '#ifndef HASH_H\n#define HASH_H\n'
            '#include <stdlib.h>\n'
            'typedef struct HashEntry { char *key; struct HashEntry *next; } HashEntry;\n'
            'int hash_insert(HashTable *ht, const char *key);\n'
            '#endif'
        )

        llm = _make_mock_llm(model_config, [enum_response, header_response])
        result = decomposed_scaffold(
            context_package, part_plan, llm,
            repo_path=tmp_path,
        )

        assert isinstance(result, ScaffoldResult)
        assert result.success is True
        assert "src/hash.h" in result.header_files
        # LLM called twice: enum + header gen (stubs are deterministic)
        assert llm.complete.call_count == 2
        # Verify files were written to disk
        assert (tmp_path / "src" / "hash.h").exists()
        assert (tmp_path / "src" / "hash.c").exists()


# -- KB pattern selection tests --

class TestKBPatternSelection:
    @pytest.fixture
    def context_with_kb(self):
        task = TaskQuery(
            raw_task="Implement hash table",
            task_id="test-kb-001",
            mode="implement",
            repo_id=1,
        )
        return ContextPackage(
            task=task,
            files=[
                FileContent(
                    file_id=1, path="src/main.c", language="c",
                    content="int main() { return 0; }", token_estimate=10,
                    detail_level="primary",
                ),
                FileContent(
                    file_id=2, path="kb/knr2/ch6_structures",
                    language="text",
                    content="Hash tables use chaining for collision resolution...",
                    token_estimate=50, detail_level="supporting",
                ),
                FileContent(
                    file_id=3, path="kb/cert_c/arr32",
                    language="text",
                    content="Ensure buffer sizes are consistent...",
                    token_estimate=30, detail_level="supporting",
                ),
            ],
            total_token_estimate=90,
            budget=BudgetConfig(context_window=32768, reserved_tokens=4096),
        )

    @pytest.fixture
    def stub(self):
        return FunctionStub(
            name="hash_insert", file_path="hash.c",
            signature="int hash_insert(HashTable *ht, const char *key)",
            docstring="Insert key into hash table", start_line=10, end_line=15,
        )

    def test_no_kb_files_returns_empty(self, context_package, stub, model_config):
        llm = _make_mock_llm(model_config, [])
        result = select_kb_patterns_for_function(stub, context_package, llm)
        assert result == []
        assert llm.complete.call_count == 0

    def test_all_relevant(self, context_with_kb, stub, model_config):
        # Both KB files judged relevant
        llm = _make_mock_llm(model_config, ["yes", "yes"])
        result = select_kb_patterns_for_function(stub, context_with_kb, llm)
        assert len(result) == 2

    def test_none_relevant(self, context_with_kb, stub, model_config):
        llm = _make_mock_llm(model_config, ["no", "no"])
        result = select_kb_patterns_for_function(stub, context_with_kb, llm)
        assert result == []

    def test_mixed_relevance(self, context_with_kb, stub, model_config):
        llm = _make_mock_llm(model_config, ["yes", "no"])
        result = select_kb_patterns_for_function(stub, context_with_kb, llm)
        assert len(result) == 1
        assert "Hash tables" in result[0]


# -- Enhanced function implement prompt tests --

class TestEnhancedFunctionImplement:
    @pytest.fixture
    def stub(self):
        return FunctionStub(
            name="hash_insert", file_path="hash.c",
            signature="int hash_insert(HashTable *ht, const char *key)",
            docstring="Insert key into hash table", start_line=10, end_line=15,
            header_file="hash.h",
        )

    def test_kb_patterns_in_prompt(self, stub, model_config):
        from clean_room_agent.execute.prompts import build_function_implement_prompt

        scaffold_content = {
            "hash.h": "#ifndef HASH_H\n#define HASH_H\n#endif",
            "hash.c": "int hash_insert(HashTable *ht, const char *key) { return 0; }",
        }
        _, user = build_function_implement_prompt(
            stub, scaffold_content, model_config,
            kb_patterns=["Use chaining for collision resolution"],
        )
        assert "<reference_patterns>" in user
        assert "chaining for collision" in user

    def test_compiler_error_in_retry_prompt(self, stub, model_config):
        from clean_room_agent.execute.prompts import build_function_implement_prompt

        scaffold_content = {
            "hash.h": "#ifndef HASH_H\n#define HASH_H\n#endif",
            "hash.c": "int hash_insert(HashTable *ht, const char *key) { return 0; }",
        }
        _, user = build_function_implement_prompt(
            stub, scaffold_content, model_config,
            compiler_error="hash.c:15: error: undeclared identifier 'foo'",
        )
        assert "<compiler_error>" in user
        assert "undeclared identifier" in user
        assert "Fix the error" in user

    def test_no_extras_when_none(self, stub, model_config):
        from clean_room_agent.execute.prompts import build_function_implement_prompt

        scaffold_content = {
            "hash.h": "#ifndef HASH_H\n#define HASH_H\n#endif",
            "hash.c": "int hash_insert(HashTable *ht, const char *key) { return 0; }",
        }
        _, user = build_function_implement_prompt(
            stub, scaffold_content, model_config,
        )
        assert "<reference_patterns>" not in user
        assert "<compiler_error>" not in user


# -- Decomposed scaffold prompt builder tests --

class TestDecomposedScaffoldPrompt:
    def test_interface_enum_prompt(self, context_package, model_config):
        from clean_room_agent.execute.prompts import build_decomposed_scaffold_prompt

        system, user = build_decomposed_scaffold_prompt(
            context_package, "Build hash table",
            pass_type="interface_enum",
            model_config=model_config,
        )
        assert "interface designer" in system
        assert "Build hash table" in user

    def test_header_gen_prompt(self, context_package, model_config):
        from clean_room_agent.execute.prompts import build_decomposed_scaffold_prompt

        system, user = build_decomposed_scaffold_prompt(
            context_package, "Build hash table",
            pass_type="header_gen",
            model_config=model_config,
            prior_stage_output='{"file_path": "hash.h", "types": [], "functions": []}',
        )
        assert "header file generator" in system
        assert "<prior_analysis>" in user

    def test_unknown_pass_type_raises(self, context_package, model_config):
        from clean_room_agent.execute.prompts import build_decomposed_scaffold_prompt

        with pytest.raises(ValueError, match="Unknown scaffold pass_type"):
            build_decomposed_scaffold_prompt(
                context_package, "test",
                pass_type="bogus",
                model_config=model_config,
            )

    def test_kb_patterns_in_prompt(self, context_package, model_config):
        from clean_room_agent.execute.prompts import build_decomposed_scaffold_prompt

        _, user = build_decomposed_scaffold_prompt(
            context_package, "Build hash table",
            pass_type="header_gen",
            model_config=model_config,
            kb_patterns=["Use open addressing", "Linear probing"],
        )
        assert "<reference_patterns>" in user
        assert "Pattern 1" in user
        assert "Pattern 2" in user


# -- Per-function compilation tests --

class TestPerFunctionCompilation:
    def test_compile_success(self, tmp_path):
        """compile_single_file returns (True, '') on valid C."""
        from clean_room_agent.execute.scaffold import compile_single_file
        import shutil
        if shutil.which("gcc") is None:
            pytest.skip("gcc not available")

        c_file = tmp_path / "test.c"
        c_file.write_text("int main(void) { return 0; }\n")
        success, output = compile_single_file("test.c", tmp_path)
        assert success is True

    def test_compile_failure(self, tmp_path):
        """compile_single_file returns (False, stderr) on invalid C."""
        from clean_room_agent.execute.scaffold import compile_single_file
        import shutil
        if shutil.which("gcc") is None:
            pytest.skip("gcc not available")

        c_file = tmp_path / "bad.c"
        c_file.write_text("int main(void { return 0; }\n")  # missing )
        success, output = compile_single_file("bad.c", tmp_path)
        assert success is False
        assert output  # non-empty error output

    def test_file_not_found(self, tmp_path):
        """compile_single_file returns (False, msg) for missing file."""
        from clean_room_agent.execute.scaffold import compile_single_file
        import shutil
        if shutil.which("gcc") is None:
            pytest.skip("gcc not available")

        success, output = compile_single_file("nonexistent.c", tmp_path)
        assert success is False
        assert "not found" in output


# -- Config helper tests --

class TestUseDecomposedScaffold:
    def test_default_false(self):
        from clean_room_agent.orchestrator.runner import _use_decomposed_scaffold
        assert _use_decomposed_scaffold({}) is False

    def test_explicit_true(self):
        from clean_room_agent.orchestrator.runner import _use_decomposed_scaffold
        config = {"orchestrator": {"decomposed_scaffold": True}}
        assert _use_decomposed_scaffold(config) is True

    def test_explicit_false(self):
        from clean_room_agent.orchestrator.runner import _use_decomposed_scaffold
        config = {"orchestrator": {"decomposed_scaffold": False}}
        assert _use_decomposed_scaffold(config) is False
