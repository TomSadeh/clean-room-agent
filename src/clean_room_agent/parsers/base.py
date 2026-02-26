from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


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


# -- Shared parser utilities (T84) --

# Comment classification patterns — language-agnostic, applied to stripped content
_COMMENT_TAG_PATTERNS = {
    "todo": re.compile(r"\bTODO\b", re.IGNORECASE),
    "fixme": re.compile(r"\bFIXME\b", re.IGNORECASE),
    "hack": re.compile(r"\bHACK\b", re.IGNORECASE),
    "note": re.compile(r"\bNOTE\b", re.IGNORECASE),
}
_BUG_REF_RE = re.compile(r"(?:#\d+|GH-\d+|BUG-\d+)", re.IGNORECASE)
_RATIONALE_RE = re.compile(
    r"\b(?:because|reason\s*:|rationale\s*:|why\s*:|note\s*:)\b", re.IGNORECASE
)


def classify_comment_content(content: str) -> tuple[str, bool]:
    """Classify comment content into (kind, is_rationale).

    Takes the comment text with language-specific prefix already stripped
    (e.g., no leading '#' for Python, no '//' for JS/TS).
    """
    for tag, pattern in _COMMENT_TAG_PATTERNS.items():
        if pattern.search(content):
            return tag, False
    if _BUG_REF_RE.search(content):
        return "bug_ref", False
    if _RATIONALE_RE.search(content):
        return "rationale", True
    return "general", False


def find_enclosing_symbol_by_line(
    line: int, symbols: list[ExtractedSymbol],
) -> str | None:
    """Find the innermost symbol enclosing the given 1-based line number."""
    best: ExtractedSymbol | None = None
    for sym in symbols:
        if sym.start_line <= line <= sym.end_line:
            if best is None or (sym.end_line - sym.start_line) < (best.end_line - best.start_line):
                best = sym
    return best.name if best else None


def node_text(node, source: bytes) -> str:
    """Extract the text of a tree-sitter node from the source bytes."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def extract_body_signature(node, source: bytes) -> str:
    """Extract declaration header using AST body node position (R4/T12).

    Returns everything from node start to body start. Falls back to
    first line if no body field is found.
    """
    body = node.child_by_field_name("body")
    if body is not None:
        header_bytes = source[node.start_byte:body.start_byte]
        return header_bytes.decode("utf-8", errors="replace").rstrip()
    logger.debug("extract_body_signature: no body field for node type %s — first-line fallback", node.type)
    return node_text(node, source).split("\n")[0]
