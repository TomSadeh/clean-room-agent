from __future__ import annotations

import re

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from clean_room_agent.parsers.base import (
    ExtractedComment,
    ExtractedDocstring,
    ExtractedImport,
    ExtractedReference,
    ExtractedSymbol,
    ParseResult,
    classify_comment_content,
    extract_body_signature,
    find_enclosing_symbol_by_line,
    node_text,
)

PY_LANGUAGE = Language(tspython.language())

# Docstring format detection patterns (Python-specific)
_GOOGLE_RE = re.compile(r"^\s*(Args|Returns|Raises|Yields|Attributes|Examples)\s*:", re.MULTILINE)
_NUMPY_RE = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)
_SPHINX_RE = re.compile(r":(param|type|returns|rtype|raises)\s")


def _classify_comment(text: str) -> tuple[str, bool]:
    """Return (kind, is_rationale) for a Python comment string.

    Strips the leading '#' before delegating to shared classification.
    """
    stripped = re.sub(r"^#\s*", "", text)
    return classify_comment_content(stripped)


def _detect_docstring_format(content: str) -> str:
    """Detect docstring format from its content."""
    if _GOOGLE_RE.search(content):
        return "google"
    if _NUMPY_RE.search(content):
        return "numpy"
    if _SPHINX_RE.search(content):
        return "sphinx"
    return "plain"


# Keep local alias for backward compat with _node_text(node, source) call pattern
_node_text = node_text


def _strip_docstring_quotes(text: str) -> str:
    """Strip surrounding triple-quotes (or single quotes) from a docstring literal."""
    for quote in ('"""', "'''", '"', "'"):
        if text.startswith(quote) and text.endswith(quote):
            return text[len(quote):-len(quote)]
    return text


def _find_enclosing_symbol(
    node, symbol_ranges: list[tuple[str, int, int]]
) -> str | None:
    """Find the innermost symbol that encloses the given node's start line."""
    line = node.start_point[0] + 1  # tree-sitter uses 0-based lines
    # Convert tuples to ExtractedSymbol for shared utility
    symbols = [
        ExtractedSymbol(name=name, kind="", start_line=start, end_line=end)
        for name, start, end in symbol_ranges
    ]
    return find_enclosing_symbol_by_line(line, symbols)


class PythonParser:
    """Tree-sitter based Python parser implementing the LanguageParser protocol."""

    def __init__(self) -> None:
        self._parser = Parser(PY_LANGUAGE)

    def parse(self, source: bytes, file_path: str) -> ParseResult:
        tree = self._parser.parse(source)
        root = tree.root_node

        symbols: list[ExtractedSymbol] = []
        docstrings: list[ExtractedDocstring] = []
        comments: list[ExtractedComment] = []
        imports: list[ExtractedImport] = []

        # First pass: extract symbols, docstrings, imports, comments
        self._extract_symbols(root, source, symbols, parent_name=None)
        self._extract_module_docstring(root, source, docstrings)
        self._extract_docstrings_from_symbols(root, source, docstrings)
        self._extract_imports(root, source, imports)

        # Build symbol ranges for enclosing-symbol lookup
        symbol_ranges = [(s.name, s.start_line, s.end_line) for s in symbols]
        self._extract_comments(root, source, comments, symbol_ranges)

        # Second pass: extract references within function/method bodies
        references = self._extract_references(root, source, symbols)

        return ParseResult(
            symbols=symbols,
            docstrings=docstrings,
            comments=comments,
            imports=imports,
            references=references,
        )

    def _extract_symbols(
        self,
        node,
        source: bytes,
        symbols: list[ExtractedSymbol],
        parent_name: str | None,
    ) -> None:
        """Recursively extract symbols from the tree."""
        for child in node.children:
            if child.type == "class_definition":
                self._handle_class(child, source, symbols, parent_name)
            elif child.type == "function_definition":
                self._handle_function(child, source, symbols, parent_name)
            elif child.type == "decorated_definition":
                # The actual definition is inside the decorated_definition.
                # Pass the decorated_definition node so start_line includes decorators.
                for sub in child.children:
                    if sub.type == "class_definition":
                        self._handle_class(sub, source, symbols, parent_name, decorated_node=child)
                    elif sub.type == "function_definition":
                        self._handle_function(sub, source, symbols, parent_name, decorated_node=child)
            elif child.type == "expression_statement" and parent_name is None:
                # Module-level assignments
                self._handle_module_assignment(child, source, symbols)

    def _handle_class(
        self,
        node,
        source: bytes,
        symbols: list[ExtractedSymbol],
        parent_name: str | None,
        decorated_node=None,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = _node_text(name_node, source)
        # T11: use decorated_definition start to include decorators in line range
        outer = decorated_node if decorated_node is not None else node
        start_line = outer.start_point[0] + 1
        end_line = outer.end_point[0] + 1

        # Build signature: the first line of the class definition (up to the colon)
        signature = self._extract_definition_line(node, source)

        symbols.append(ExtractedSymbol(
            name=name,
            kind="class",
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            parent_name=parent_name,
        ))

        # Recurse into the class body for methods and nested classes
        body = node.child_by_field_name("body")
        if body is not None:
            self._extract_symbols(body, source, symbols, parent_name=name)

    def _handle_function(
        self,
        node,
        source: bytes,
        symbols: list[ExtractedSymbol],
        parent_name: str | None,
        decorated_node=None,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = _node_text(name_node, source)
        # T11: use decorated_definition start to include decorators in line range
        outer = decorated_node if decorated_node is not None else node
        start_line = outer.start_point[0] + 1
        end_line = outer.end_point[0] + 1
        kind = "method" if parent_name is not None else "function"

        signature = self._extract_definition_line(node, source)

        symbols.append(ExtractedSymbol(
            name=name,
            kind=kind,
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            parent_name=parent_name,
        ))

    def _handle_module_assignment(
        self,
        node,
        source: bytes,
        symbols: list[ExtractedSymbol],
    ) -> None:
        """Handle module-level expression_statement that contains an assignment."""
        for child in node.children:
            if child.type == "assignment":
                left = child.child_by_field_name("left")
                if left is not None and left.type == "identifier":
                    name = _node_text(left, source)
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    signature = extract_body_signature(node, source)
                    symbols.append(ExtractedSymbol(
                        name=name,
                        kind="variable",
                        start_line=start_line,
                        end_line=end_line,
                        signature=signature,
                        parent_name=None,
                    ))

    def _extract_definition_line(self, node, source: bytes) -> str:
        """Extract the definition header (R4/T12). Delegates to shared utility."""
        return extract_body_signature(node, source)

    def _extract_module_docstring(
        self,
        root,
        source: bytes,
        docstrings: list[ExtractedDocstring],
    ) -> None:
        """Extract module-level docstring (first expression_statement with a string)."""
        for child in root.children:
            if child.type == "comment":
                continue
            if child.type == "expression_statement":
                string_node = self._find_string_child(child)
                if string_node is not None:
                    content = _strip_docstring_quotes(_node_text(string_node, source))
                    fmt = _detect_docstring_format(content)
                    docstrings.append(ExtractedDocstring(
                        content=content,
                        format=fmt,
                        symbol_name=None,
                        line=string_node.start_point[0] + 1,
                    ))
                return
            else:
                # If the first non-comment statement is not a string expression, no module docstring
                return

    def _extract_docstrings_from_symbols(
        self,
        root,
        source: bytes,
        docstrings: list[ExtractedDocstring],
    ) -> None:
        """Walk the tree and extract docstrings from function/class definitions."""
        self._walk_for_docstrings(root, source, docstrings)

    def _walk_for_docstrings(
        self,
        node,
        source: bytes,
        docstrings: list[ExtractedDocstring],
    ) -> None:
        """Recursively walk tree looking for function/class definitions with docstrings."""
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    continue
                name = _node_text(name_node, source)
                body = child.child_by_field_name("body")
                if body is not None:
                    ds = self._first_docstring_in_body(body, source)
                    if ds is not None:
                        ds_text, ds_line = ds
                        content = _strip_docstring_quotes(ds_text)
                        fmt = _detect_docstring_format(content)
                        docstrings.append(ExtractedDocstring(
                            content=content,
                            format=fmt,
                            symbol_name=name,
                            line=ds_line,
                        ))
                    # Recurse into body for nested definitions
                    self._walk_for_docstrings(body, source, docstrings)
            elif child.type == "decorated_definition":
                self._walk_for_docstrings(child, source, docstrings)
            elif child.type == "block":
                self._walk_for_docstrings(child, source, docstrings)

    def _first_docstring_in_body(self, body_node, source: bytes) -> tuple[str, int] | None:
        """Check if the first statement in a body is a string expression (docstring)."""
        for child in body_node.children:
            if child.type == "comment":
                continue
            if child.type == "expression_statement":
                string_node = self._find_string_child(child)
                if string_node is not None:
                    return _node_text(string_node, source), string_node.start_point[0] + 1
            return None
        return None

    def _find_string_child(self, expr_stmt_node) -> object | None:
        """Find a string or concatenated_string child of an expression_statement."""
        for child in expr_stmt_node.children:
            if child.type in ("string", "concatenated_string"):
                return child
        return None

    def _extract_imports(
        self,
        root,
        source: bytes,
        imports: list[ExtractedImport],
    ) -> None:
        """Extract import statements from the module level."""
        self._walk_for_imports(root, source, imports)

    def _walk_for_imports(self, node, source: bytes, imports: list[ExtractedImport]) -> None:
        """Recursively find import statements."""
        for child in node.children:
            if child.type == "import_statement":
                self._handle_import(child, source, imports)
            elif child.type == "import_from_statement":
                self._handle_import_from(child, source, imports)
            # Don't recurse into function/class bodies for imports -- only module-level
            # But do handle if-guarded imports at module level
            elif child.type in ("if_statement", "try_statement"):
                self._walk_for_imports(child, source, imports)
            elif child.type in ("block", "else_clause", "except_clause", "finally_clause"):
                self._walk_for_imports(child, source, imports)

    def _handle_import(
        self,
        node,
        source: bytes,
        imports: list[ExtractedImport],
    ) -> None:
        """Handle `import X` or `import X, Y` statements."""
        for child in node.children:
            if child.type == "dotted_name":
                module = _node_text(child, source)
                imports.append(ExtractedImport(
                    module=module,
                    names=[],
                    is_relative=False,
                    level=0,
                ))
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                if name_node is not None:
                    module = _node_text(name_node, source)
                    imports.append(ExtractedImport(
                        module=module,
                        names=[],
                        is_relative=False,
                        level=0,
                    ))

    def _handle_import_from(
        self,
        node,
        source: bytes,
        imports: list[ExtractedImport],
    ) -> None:
        """Handle `from X import Y` statements."""
        module_name = ""
        level = 0
        names: list[str] = []

        # Count leading dots for relative imports and find the module name
        found_from = False
        found_import = False
        for child in node.children:
            if child.type == "from":
                found_from = True
                continue
            if child.type == "import":
                found_import = True
                continue

            if found_from and not found_import:
                # Between 'from' and 'import': module path or relative dots
                if child.type == "relative_import":
                    # relative_import contains dots and optionally a dotted_name
                    for sub in child.children:
                        if sub.type == "import_prefix":
                            dot_text = _node_text(sub, source)
                            level = len(dot_text)
                        elif sub.type == "dotted_name":
                            module_name = _node_text(sub, source)
                elif child.type == "dotted_name":
                    module_name = _node_text(child, source)

            if found_import:
                # After 'import': the imported names
                if child.type == "dotted_name":
                    names.append(_node_text(child, source))
                elif child.type == "aliased_import":
                    name_node = child.child_by_field_name("name")
                    if name_node is not None:
                        names.append(_node_text(name_node, source))
                elif child.type == "import_list":
                    for item in child.children:
                        if item.type == "dotted_name":
                            names.append(_node_text(item, source))
                        elif item.type == "aliased_import":
                            name_node = item.child_by_field_name("name")
                            if name_node is not None:
                                names.append(_node_text(name_node, source))

        imports.append(ExtractedImport(
            module=module_name,
            names=names,
            is_relative=level > 0,
            level=level,
        ))

    def _extract_comments(
        self,
        root,
        source: bytes,
        comments: list[ExtractedComment],
        symbol_ranges: list[tuple[str, int, int]],
    ) -> None:
        """Walk the entire tree and collect all comment nodes."""
        self._walk_for_comments(root, source, comments, symbol_ranges)

    def _walk_for_comments(
        self,
        node,
        source: bytes,
        comments: list[ExtractedComment],
        symbol_ranges: list[tuple[str, int, int]],
    ) -> None:
        if node.type == "comment":
            text = _node_text(node, source)
            line = node.start_point[0] + 1
            kind, is_rationale = _classify_comment(text)
            enclosing = _find_enclosing_symbol(node, symbol_ranges)
            comments.append(ExtractedComment(
                line=line,
                content=text,
                kind=kind,
                is_rationale=is_rationale,
                symbol_name=enclosing,
            ))
        for child in node.children:
            self._walk_for_comments(child, source, comments, symbol_ranges)

    def _extract_references(
        self,
        root,
        source: bytes,
        symbols: list[ExtractedSymbol],
    ) -> list[ExtractedReference]:
        """Extract symbol references within function/method bodies."""
        references: list[ExtractedReference] = []

        # Build lookup of known symbol names for same-file matching
        symbol_names = {s.name for s in symbols}

        # Find all function/method definitions and scan their bodies
        self._walk_for_references(root, source, symbols, symbol_names, references)
        return references

    def _walk_for_references(
        self,
        node,
        source: bytes,
        symbols: list[ExtractedSymbol],
        symbol_names: set[str],
        references: list[ExtractedReference],
    ) -> None:
        """Recursively find function definitions and scan their bodies for references."""
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                if child.type == "function_definition":
                    name_node = child.child_by_field_name("name")
                    if name_node is None:
                        continue
                    caller_name = _node_text(name_node, source)
                    body = child.child_by_field_name("body")
                    if body is not None:
                        self._scan_body_for_references(
                            body, source, caller_name, symbol_names, references
                        )
                        # Continue recursing for nested functions
                        self._walk_for_references(
                            body, source, symbols, symbol_names, references
                        )
                else:
                    # For classes, recurse into body to find methods
                    body = child.child_by_field_name("body")
                    if body is not None:
                        self._walk_for_references(
                            body, source, symbols, symbol_names, references
                        )
            elif child.type == "decorated_definition":
                self._walk_for_references(child, source, symbols, symbol_names, references)

    def _scan_body_for_references(
        self,
        node,
        source: bytes,
        caller_name: str,
        symbol_names: set[str],
        references: list[ExtractedReference],
    ) -> None:
        """Scan a function body for identifier and attribute references to known symbols."""
        # Track already-seen (caller, callee, kind) to avoid duplicates
        seen: set[tuple[str, str, str]] = set()
        self._scan_node_for_refs(node, source, caller_name, symbol_names, references, seen)

    def _scan_node_for_refs(
        self,
        node,
        source: bytes,
        caller_name: str,
        symbol_names: set[str],
        references: list[ExtractedReference],
        seen: set[tuple[str, str, str]],
    ) -> None:
        """Recursively scan nodes for references."""
        if node.type == "identifier":
            name = _node_text(node, source)
            if name in symbol_names and name != caller_name:
                ref_kind = self._determine_reference_kind(node)
                key = (caller_name, name, ref_kind)
                if key not in seen:
                    seen.add(key)
                    references.append(ExtractedReference(
                        caller_symbol=caller_name,
                        callee_symbol=name,
                        reference_kind=ref_kind,
                    ))

        # Don't recurse into nested function/class definitions --
        # they get their own caller_name
        if node.type in ("function_definition", "class_definition"):
            return

        for child in node.children:
            self._scan_node_for_refs(
                child, source, caller_name, symbol_names, references, seen
            )

    def _determine_reference_kind(self, node) -> str:
        """Determine the kind of reference based on the parent node."""
        parent = node.parent
        if parent is None:
            return "name"
        if parent.type == "call":
            # The identifier is the function being called
            func_node = parent.child_by_field_name("function")
            if func_node is not None and func_node.id == node.id:
                return "call"
        if parent.type == "attribute":
            return "attribute"
        return "name"
