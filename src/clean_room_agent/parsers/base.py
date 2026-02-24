from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ExtractedSymbol:
    name: str
    kind: str  # "function", "class", "method", "interface", "type_alias", "enum", "variable"
    start_line: int
    end_line: int
    signature: str | None = None
    parent_name: str | None = None  # name of enclosing symbol (for nesting)


@dataclass
class ExtractedDocstring:
    content: str
    format: str | None = None  # "google", "numpy", "sphinx", "jsdoc", "plain"
    parsed_fields: str | None = None  # JSON of structured fields
    symbol_name: str | None = None  # associated symbol, None for module-level
    line: int | None = None  # 1-based line where docstring starts


@dataclass
class ExtractedComment:
    line: int
    content: str
    kind: str | None = None  # "todo", "fixme", "hack", "note", "bug_ref", "rationale", "general"
    is_rationale: bool = False
    symbol_name: str | None = None  # innermost enclosing symbol, None if module-level


@dataclass
class ExtractedImport:
    module: str  # what's being imported from
    names: list[str] = field(default_factory=list)  # specific names imported (empty for bare import)
    is_relative: bool = False
    level: int = 0  # for relative imports (number of dots)
    is_type_only: bool = False  # TS "import type"


@dataclass
class ExtractedReference:
    caller_symbol: str
    callee_symbol: str
    reference_kind: str  # "call", "attribute", "name"


@dataclass
class ParseResult:
    symbols: list[ExtractedSymbol] = field(default_factory=list)
    docstrings: list[ExtractedDocstring] = field(default_factory=list)
    comments: list[ExtractedComment] = field(default_factory=list)
    imports: list[ExtractedImport] = field(default_factory=list)
    references: list[ExtractedReference] = field(default_factory=list)


class LanguageParser(Protocol):
    def parse(self, source: bytes, file_path: str) -> ParseResult: ...
