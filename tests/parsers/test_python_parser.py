from clean_room_agent.parsers.python_parser import PythonParser

SAMPLE_PYTHON = b'''
"""Module docstring."""

import os
from pathlib import Path

# TODO: refactor this
MAX_SIZE = 1024

class MyClass:
    """A sample class.

    Args:
        name: The name.
    """

    def __init__(self, name: str):
        """Initialize with name."""
        self.name = name

    def greet(self) -> str:
        """Return greeting."""
        # This returns a formatted string because we need i18n support
        return f"Hello, {self.name}"

def helper(x: int) -> int:
    """Helper function.

    Parameters
    ----------
    x : int
        The input.
    """
    return x + MAX_SIZE
'''


def make_result():
    parser = PythonParser()
    return parser.parse(SAMPLE_PYTHON, "sample.py")


class TestSymbolExtraction:
    def test_symbol_extraction(self):
        result = make_result()
        names = {s.name for s in result.symbols}
        assert "MyClass" in names
        assert "__init__" in names
        assert "greet" in names
        assert "helper" in names
        assert "MAX_SIZE" in names

    def test_symbol_kinds(self):
        result = make_result()
        by_name = {s.name: s for s in result.symbols}
        assert by_name["MyClass"].kind == "class"
        assert by_name["__init__"].kind == "method"
        assert by_name["greet"].kind == "method"
        assert by_name["helper"].kind == "function"
        assert by_name["MAX_SIZE"].kind == "variable"

    def test_parent_child(self):
        result = make_result()
        by_name = {s.name: s for s in result.symbols}
        assert by_name["__init__"].parent_name == "MyClass"
        assert by_name["greet"].parent_name == "MyClass"
        assert by_name["MyClass"].parent_name is None
        assert by_name["helper"].parent_name is None
        assert by_name["MAX_SIZE"].parent_name is None


class TestDocstringExtraction:
    def test_docstring_extraction(self):
        result = make_result()
        by_sym = {d.symbol_name: d for d in result.docstrings}
        assert None in by_sym  # module docstring
        assert "MyClass" in by_sym
        assert "A sample class." in by_sym["MyClass"].content
        assert "__init__" in by_sym
        assert "Initialize with name." in by_sym["__init__"].content
        assert "greet" in by_sym
        assert "Return greeting." in by_sym["greet"].content
        assert "helper" in by_sym
        assert "Helper function." in by_sym["helper"].content

    def test_module_docstring(self):
        result = make_result()
        module_docs = [d for d in result.docstrings if d.symbol_name is None]
        assert len(module_docs) == 1
        assert "Module docstring" in module_docs[0].content

    def test_google_format(self):
        result = make_result()
        by_sym = {d.symbol_name: d for d in result.docstrings}
        assert by_sym["MyClass"].format == "google"

    def test_numpy_format(self):
        result = make_result()
        by_sym = {d.symbol_name: d for d in result.docstrings}
        assert by_sym["helper"].format == "numpy"

    def test_plain_format(self):
        result = make_result()
        by_sym = {d.symbol_name: d for d in result.docstrings}
        # Short docstrings without special format markers -> plain
        assert by_sym["__init__"].format == "plain"
        assert by_sym["greet"].format == "plain"


class TestCommentClassification:
    def test_todo_comment(self):
        result = make_result()
        todo_comments = [c for c in result.comments if c.kind == "todo"]
        assert len(todo_comments) >= 1
        assert any("TODO" in c.content for c in todo_comments)

    def test_rationale_comment(self):
        result = make_result()
        rationale_comments = [c for c in result.comments if c.kind == "rationale"]
        assert len(rationale_comments) >= 1
        assert any("because" in c.content for c in rationale_comments)
        assert all(c.is_rationale for c in rationale_comments)

    def test_comment_enclosing_symbol(self):
        result = make_result()
        # The rationale comment is inside greet
        rationale = [c for c in result.comments if c.kind == "rationale"]
        assert len(rationale) >= 1
        assert rationale[0].symbol_name == "greet"

        # The TODO comment is at module level (between class and MAX_SIZE,
        # but before the class definition, so module-level)
        todo = [c for c in result.comments if c.kind == "todo"]
        assert len(todo) >= 1
        assert todo[0].symbol_name is None


class TestImportExtraction:
    def test_import_os(self):
        result = make_result()
        os_imports = [i for i in result.imports if i.module == "os"]
        assert len(os_imports) == 1
        assert os_imports[0].names == []
        assert os_imports[0].is_relative is False

    def test_import_from_pathlib(self):
        result = make_result()
        pathlib_imports = [i for i in result.imports if i.module == "pathlib"]
        assert len(pathlib_imports) == 1
        assert "Path" in pathlib_imports[0].names
        assert pathlib_imports[0].is_relative is False


class TestSymbolReferences:
    def test_helper_references_max_size(self):
        result = make_result()
        refs_from_helper = [
            r for r in result.references if r.caller_symbol == "helper"
        ]
        callee_names = {r.callee_symbol for r in refs_from_helper}
        assert "MAX_SIZE" in callee_names

    def test_reference_kind(self):
        result = make_result()
        max_size_refs = [
            r for r in result.references if r.callee_symbol == "MAX_SIZE"
        ]
        assert len(max_size_refs) >= 1
        assert all(r.reference_kind == "name" for r in max_size_refs)


class TestSignatures:
    def test_function_signature(self):
        result = make_result()
        by_name = {s.name: s for s in result.symbols}
        assert "def helper(x: int) -> int:" in by_name["helper"].signature

    def test_method_signature(self):
        result = make_result()
        by_name = {s.name: s for s in result.symbols}
        assert "def __init__(self, name: str):" in by_name["__init__"].signature
        assert "def greet(self) -> str:" in by_name["greet"].signature

    def test_class_signature(self):
        result = make_result()
        by_name = {s.name: s for s in result.symbols}
        assert "class MyClass:" in by_name["MyClass"].signature

    def test_variable_signature(self):
        result = make_result()
        by_name = {s.name: s for s in result.symbols}
        assert "MAX_SIZE = 1024" in by_name["MAX_SIZE"].signature


class TestDecoratedDefinitions:
    """T11: decorated symbols should include decorator lines in their range."""

    DECORATED_SOURCE = b'''
@dataclass
class Config:
    """Configuration."""
    name: str = "default"

@staticmethod
def standalone():
    pass

class Outer:
    @property
    def value(self):
        return 42

    @classmethod
    def create(cls):
        return cls()
'''

    def test_decorated_class_start_line(self):
        parser = PythonParser()
        result = parser.parse(self.DECORATED_SOURCE, "decorated.py")
        by_name = {s.name: s for s in result.symbols}
        config = by_name["Config"]
        # @dataclass is on line 2, class Config on line 3
        assert config.start_line == 2  # includes the decorator

    def test_decorated_function_start_line(self):
        parser = PythonParser()
        result = parser.parse(self.DECORATED_SOURCE, "decorated.py")
        by_name = {s.name: s for s in result.symbols}
        standalone = by_name["standalone"]
        # @staticmethod is on line 7, def standalone on line 8
        assert standalone.start_line == 7  # includes the decorator

    def test_decorated_method_start_line(self):
        parser = PythonParser()
        result = parser.parse(self.DECORATED_SOURCE, "decorated.py")
        by_name = {s.name: s for s in result.symbols}
        value = by_name["value"]
        # @property is on line 12, def value on line 13
        assert value.start_line == 12  # includes the decorator

    def test_undecorated_start_line_unchanged(self):
        parser = PythonParser()
        result = parser.parse(self.DECORATED_SOURCE, "decorated.py")
        by_name = {s.name: s for s in result.symbols}
        outer = by_name["Outer"]
        # class Outer: is on line 11, no decorator
        assert outer.start_line == 11


class TestMultiLineSignatures:
    """T12: multi-line signatures should include full parameter list."""

    MULTILINE_SOURCE = b'''
def simple(x: int) -> bool:
    return True

def complex_func(
    param1: str,
    param2: int,
    param3: list,
) -> bool:
    return True

class Base(
    Protocol,
):
    pass
'''

    def test_single_line_signature(self):
        parser = PythonParser()
        result = parser.parse(self.MULTILINE_SOURCE, "multi.py")
        by_name = {s.name: s for s in result.symbols}
        sig = by_name["simple"].signature
        assert "def simple(x: int) -> bool:" in sig

    def test_multiline_function_signature(self):
        parser = PythonParser()
        result = parser.parse(self.MULTILINE_SOURCE, "multi.py")
        by_name = {s.name: s for s in result.symbols}
        sig = by_name["complex_func"].signature
        # Should contain all parameters, not just first line
        assert "param1" in sig
        assert "param2" in sig
        assert "param3" in sig
        assert "-> bool:" in sig

    def test_multiline_class_signature(self):
        parser = PythonParser()
        result = parser.parse(self.MULTILINE_SOURCE, "multi.py")
        by_name = {s.name: s for s in result.symbols}
        sig = by_name["Base"].signature
        assert "Protocol" in sig


class TestA5AssignmentSignature:
    """A5: Multi-line module assignment uses extract_body_signature (first-line fallback)."""

    MULTILINE_ASSIGNMENT = b'''
X = {
    "a": 1,
    "b": 2,
}

Y = [
    1,
    2,
    3,
]
'''

    def test_multiline_dict_assignment_full_text_signature(self):
        """T72: Bodyless nodes use full node text — no split("\n")[0] truncation."""
        parser = PythonParser()
        result = parser.parse(self.MULTILINE_ASSIGNMENT, "assign.py")
        by_name = {s.name: s for s in result.symbols}
        assert "X" in by_name
        sig = by_name["X"].signature
        # T72: Full node text preserved for bodyless nodes (R4 compliance)
        assert "X = {" in sig
        assert '"a": 1' in sig

    def test_multiline_list_assignment_full_text_signature(self):
        """T72: Bodyless nodes use full node text — no split("\n")[0] truncation."""
        parser = PythonParser()
        result = parser.parse(self.MULTILINE_ASSIGNMENT, "assign.py")
        by_name = {s.name: s for s in result.symbols}
        assert "Y" in by_name
        sig = by_name["Y"].signature
        # T72: Full node text preserved for bodyless nodes (R4 compliance)
        assert "Y = [" in sig
        assert "1," in sig
