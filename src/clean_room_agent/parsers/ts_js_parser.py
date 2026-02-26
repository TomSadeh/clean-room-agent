"""Tree-sitter parser for TypeScript and JavaScript."""

import json
import re

import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser

from clean_room_agent.parsers.base import (
    ExtractedComment,
    ExtractedDocstring,
    ExtractedImport,
    ExtractedSymbol,
    ParseResult,
    classify_comment_content,
    extract_body_signature,
    find_enclosing_symbol_by_line,
)

TS_LANGUAGE = Language(tsts.language_typescript())
TSX_LANGUAGE = Language(tsts.language_tsx())
JS_LANGUAGE = Language(tsjs.language())

# Comment classification now uses shared patterns from base.py (T84).
# JSDoc-specific pattern stays here.
_JSDOC_TAG = re.compile(r"@(\w+)\s*(.*)")


class TSJSParser:
    """Parser for TypeScript (.ts/.tsx) and JavaScript (.js/.jsx/.mjs/.cjs)."""

    def __init__(self, language: str):
        if language not in ("typescript", "javascript"):
            raise ValueError(f"Unsupported language: {language}")
        self._language = language

    def _get_parser(self, file_path: str) -> Parser:
        if self._language == "javascript":
            return Parser(JS_LANGUAGE)
        # TypeScript: choose TSX for .tsx files
        if file_path.endswith(".tsx"):
            return Parser(TSX_LANGUAGE)
        return Parser(TS_LANGUAGE)

    def parse(self, source: bytes, file_path: str) -> ParseResult:
        parser = self._get_parser(file_path)
        tree = parser.parse(source)
        root = tree.root_node

        symbols = self._extract_symbols(root, source)
        docstrings = self._extract_docstrings(root, source, symbols)
        comments = self._extract_comments(root, source, symbols)
        imports = self._extract_imports(root, source)

        return ParseResult(
            symbols=symbols,
            docstrings=docstrings,
            comments=comments,
            imports=imports,
            references=[],  # No symbol references for TS/JS MVP
        )

    def _extract_symbols(self, root, source: bytes) -> list[ExtractedSymbol]:
        symbols = []
        self._walk_symbols(root, source, None, symbols)
        return symbols

    def _walk_symbols(self, node, source: bytes, parent_name: str | None, out: list):
        for child in node.children:
            kind = self._classify_symbol(child, parent_name)
            if kind:
                name = self._symbol_name(child, source)
                if name:
                    sig = self._extract_signature(child, source)
                    sym = ExtractedSymbol(
                        name=name,
                        kind=kind,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        signature=sig,
                        parent_name=parent_name,
                    )
                    out.append(sym)
                    # Recurse into classes for methods
                    if kind == "class":
                        body = _find_child(child, "class_body")
                        if body:
                            self._walk_symbols(body, source, name, out)
                    continue
            # Check for exported declarations
            if child.type in ("export_statement", "export_default_declaration"):
                self._walk_symbols(child, source, parent_name, out)
                continue
            # Check for variable declarations with arrow functions at module level
            if child.type == "lexical_declaration" and parent_name is None:
                self._extract_variable_symbols(child, source, out)
                continue
            if child.type == "variable_declaration" and parent_name is None:
                self._extract_variable_symbols(child, source, out)
                continue

    def _classify_symbol(self, node, parent_name: str | None) -> str | None:
        t = node.type
        if t == "function_declaration":
            return "method" if parent_name else "function"
        if t == "class_declaration":
            return "class"
        if t == "method_definition":
            return "method"
        # TS-specific
        if t == "interface_declaration":
            return "interface"
        if t == "type_alias_declaration":
            return "type_alias"
        if t == "enum_declaration":
            return "enum"
        return None

    def _symbol_name(self, node, source: bytes) -> str | None:
        # Most declarations have a `name` child or `type_identifier`/`identifier`
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "property_identifier"):
                return child.text.decode("utf-8")
        return None

    def _extract_signature(self, node, source: bytes) -> str:
        """Extract the declaration header (R4/T12). Delegates to shared utility."""
        return extract_body_signature(node, source)

    def _extract_variable_symbols(self, node, source: bytes, out: list):
        """Extract arrow functions and other variable declarations."""
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = _find_child(child, "identifier")
                if not name_node:
                    continue
                value_node = _find_child(child, "arrow_function")
                kind = "function" if value_node else "variable"
                out.append(ExtractedSymbol(
                    name=name_node.text.decode("utf-8"),
                    kind=kind,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    signature=extract_body_signature(child, source),
                    parent_name=None,
                ))

    def _extract_docstrings(self, root, source: bytes, symbols: list[ExtractedSymbol]) -> list[ExtractedDocstring]:
        docstrings = []
        # Find all comment nodes that are JSDoc (start with /**)
        for node in _iter_tree(root):
            if node.type == "comment":
                text = node.text.decode("utf-8")
                if text.startswith("/**"):
                    content = self._clean_jsdoc(text)
                    # Find the symbol this JSDoc is attached to (next sibling)
                    sym_name = self._find_attached_symbol(node, symbols)
                    fmt = "jsdoc" if _JSDOC_TAG.search(content) else "plain"
                    parsed = self._parse_jsdoc_tags(content) if fmt == "jsdoc" else None
                    docstrings.append(ExtractedDocstring(
                        content=content,
                        format=fmt,
                        parsed_fields=parsed,
                        symbol_name=sym_name,
                        line=node.start_point[0] + 1,
                    ))
        return docstrings

    def _clean_jsdoc(self, text: str) -> str:
        """Strip JSDoc comment markers, preserving paragraph breaks."""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if line.startswith("/**"):
                line = line[3:].strip()
                if line.endswith("*/"):
                    line = line[:-2].strip()
            elif line.startswith("*/"):
                continue
            elif line.startswith("*"):
                line = line[1:].strip()
            cleaned.append(line)
        # Strip leading/trailing empty lines, keep internal ones
        while cleaned and not cleaned[0]:
            cleaned.pop(0)
        while cleaned and not cleaned[-1]:
            cleaned.pop()
        return "\n".join(cleaned)

    def _parse_jsdoc_tags(self, content: str) -> str:
        """Extract JSDoc tags into JSON string."""
        tags = {}
        for match in _JSDOC_TAG.finditer(content):
            tag, value = match.group(1), match.group(2).strip()
            if tag in tags:
                if isinstance(tags[tag], list):
                    tags[tag].append(value)
                else:
                    tags[tag] = [tags[tag], value]
            else:
                tags[tag] = value
        return json.dumps(tags)

    def _find_attached_symbol(self, comment_node, symbols: list[ExtractedSymbol]) -> str | None:
        """Find the symbol defined immediately after a JSDoc comment."""
        comment_end = comment_node.end_point[0] + 1
        for sym in symbols:
            if sym.start_line == comment_end + 1 or sym.start_line == comment_end:
                return sym.name
        return None

    def _extract_comments(self, root, source: bytes, symbols: list[ExtractedSymbol]) -> list[ExtractedComment]:
        comments = []
        for node in _iter_tree(root):
            if node.type == "comment":
                text = node.text.decode("utf-8")
                # Skip JSDoc (handled as docstrings)
                if text.startswith("/**"):
                    continue
                line = node.start_point[0] + 1
                # Strip comment markers
                if text.startswith("//"):
                    content = text[2:].strip()
                elif text.startswith("/*"):
                    content = text[2:-2].strip()
                else:
                    content = text.strip()
                kind, is_rationale = classify_comment_content(content)
                sym_name = _find_enclosing_symbol(line, symbols)
                comments.append(ExtractedComment(
                    line=line,
                    content=content,
                    kind=kind,
                    is_rationale=is_rationale,
                    symbol_name=sym_name,
                ))
        return comments

    def _extract_imports(self, root, source: bytes) -> list[ExtractedImport]:
        imports = []
        for node in _iter_tree(root):
            if node.type == "import_statement":
                imp = self._parse_es_import(node, source)
                if imp:
                    imports.append(imp)
            elif node.type in ("lexical_declaration", "variable_declaration"):
                # CommonJS: const x = require("...")
                imp = self._parse_commonjs_require(node, source)
                if imp:
                    imports.append(imp)
        return imports

    def _parse_es_import(self, node, source: bytes) -> ExtractedImport | None:
        """Parse ES module import statement."""
        module = None
        names = []
        is_type_only = False

        for child in node.children:
            if child.type == "string":
                module = child.text.decode("utf-8").strip("'\"")
            elif child.type == "import_clause":
                names.extend(_extract_import_clause_names(child))
            elif child.type == "type" or child.text == b"type":
                is_type_only = True

        if not module:
            return None

        is_relative = module.startswith(".")
        return ExtractedImport(
            module=module,
            names=names,
            is_relative=is_relative,
            level=0,
            is_type_only=is_type_only,
        )

    def _parse_commonjs_require(self, node, source: bytes) -> ExtractedImport | None:
        """Parse CommonJS require() call from lexical/variable declarations.

        R4: walks the tree-sitter AST instead of regex. Handles:
        - const x = require("module")
        - const { x, y } = require("module")
        - let/var variants
        """
        if self._language != "javascript":
            return None

        # Walk variable_declarators inside the declaration
        for child in node.children:
            if child.type != "variable_declarator":
                continue

            # Right side must be a call_expression calling "require"
            value = child.child_by_field_name("value")
            if value is None or value.type != "call_expression":
                continue
            func_node = value.child_by_field_name("function")
            if func_node is None or func_node.text != b"require":
                continue

            # Extract the module path from arguments
            args = value.child_by_field_name("arguments")
            if args is None:
                continue
            module = None
            for arg in args.children:
                if arg.type == "string":
                    module = arg.text.decode("utf-8").strip("'\"")
                    break
            if module is None:
                continue

            # Extract names from the left side (name node)
            lhs = child.child_by_field_name("name")
            names = _extract_require_names(lhs) if lhs is not None else []

            return ExtractedImport(
                module=module,
                names=names,
                is_relative=module.startswith("."),
                level=0,
                is_type_only=False,
            )
        return None


def _extract_import_clause_names(clause_node) -> list[str]:
    """Extract imported names from an ES import_clause node."""
    names = []
    for ic in clause_node.children:
        if ic.type == "identifier":
            names.append(ic.text.decode("utf-8"))
        elif ic.type == "named_imports":
            for spec in ic.children:
                if spec.type == "import_specifier":
                    name_node = _find_child(spec, "identifier")
                    if name_node:
                        names.append(name_node.text.decode("utf-8"))
        elif ic.type == "namespace_import":
            alias = _find_child(ic, "identifier")
            if alias:
                names.append(f"* as {alias.text.decode('utf-8')}")
    return names


def _extract_require_names(name_node) -> list[str]:
    """Extract bound names from the left side of a CommonJS require()."""
    if name_node.type == "identifier":
        return [name_node.text.decode("utf-8")]
    if name_node.type == "object_pattern":
        names = []
        for pat_child in name_node.children:
            if pat_child.type == "shorthand_property_identifier_pattern":
                names.append(pat_child.text.decode("utf-8"))
            elif pat_child.type == "pair_pattern":
                val = pat_child.child_by_field_name("value")
                if val and val.type == "identifier":
                    names.append(val.text.decode("utf-8"))
        return names
    return []


def _find_child(node, child_type: str):
    """Find the first child of a given type."""
    for child in node.children:
        if child.type == child_type:
            return child
    return None


def _iter_tree(node):
    """Iterate all nodes in the tree depth-first."""
    yield node
    for child in node.children:
        yield from _iter_tree(child)


def _classify_comment(content: str) -> str:
    """Classify a comment by its content. Delegates to shared utility (T84)."""
    kind, _ = classify_comment_content(content)
    return kind


def _find_enclosing_symbol(line: int, symbols: list[ExtractedSymbol]) -> str | None:
    """Find the innermost symbol enclosing the given line. Delegates to shared utility (T84)."""
    return find_enclosing_symbol_by_line(line, symbols)
