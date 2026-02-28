"""Structural linter: detect transparency anti-patterns.

This test file runs as part of pytest and fails the build when transparency
anti-patterns are found.  It enforces the CLAUDE.md traceability principle:
a human must be able to trace any output back through every decision that
produced it using only the logs.

Patterns detected:
    T-A. Discarded binary judgment omissions (run_binary_judgment 2nd return ignored)
    T-B. Post-judgment silent cascade drop (if not x: return without logging)
    T-C. LLM call records discarded without DB persistence (bare .flush())
    T-D. Curated DB UPDATE without audit trail (UPDATE in curated queries)

Each pattern has an allowlist for intentional exceptions.  Every allowlist
entry requires a justification comment.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Directories containing core logic (not CLI, not tests).
_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "clean_room_agent"
_CORE_DIRS = [
    _SRC_ROOT / "retrieval",
    _SRC_ROOT / "orchestrator",
    _SRC_ROOT / "llm",
    _SRC_ROOT / "execute",
]


def _collect_py_files(*dirs: Path) -> list[Path]:
    """Collect all .py files under the given directories."""
    files: list[Path] = []
    for d in dirs:
        if d.is_dir():
            files.extend(sorted(d.rglob("*.py")))
    return files


def _rel(path: Path) -> str:
    """Return a short relative path for reporting (always forward slashes)."""
    try:
        return str(path.relative_to(_SRC_ROOT.parent.parent)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# ---------------------------------------------------------------------------
# Allowlists — every entry must have a justification.
# Format: (relative_path_suffix, line_number, justification)
# ---------------------------------------------------------------------------

# T-A: run_binary_judgment() 2nd return value discarded via underscore naming.
DISCARDED_OMISSIONS_ALLOWLIST: list[tuple[str, int, str]] = [
    # -- precision_stage.py: R2 cascade; per-item parse failures, not actionable --
    ("retrieval/precision_stage.py", 198, "R2 cascade pass1: omissions are per-item LLM response parse failures"),
    ("retrieval/precision_stage.py", 222, "R2 cascade pass2: omissions are per-item LLM response parse failures"),
    ("retrieval/precision_stage.py", 247, "R2 cascade pass3: omissions are per-item LLM response parse failures"),
    # -- routing.py: binary routing; omitted stages are simply not selected --
    ("retrieval/routing.py", 49, "Binary routing: omitted stages are simply not selected"),
    # -- decomposed_scaffold.py: KB pattern selection; omissions reduce pattern count --
    ("execute/decomposed_scaffold.py", 424, "KB pattern selection: omissions reduce available patterns"),
]

# T-B: if not <name>: return without logger call in functions using run_binary_judgment.
SILENT_CASCADE_DROP_ALLOWLIST: list[tuple[str, int, str]] = [
    # All cascade drop sites now have logger calls (T2-3 fixed).
]

# T-C: bare .flush() at statement level (return value discarded).
BARE_FLUSH_ALLOWLIST: list[tuple[str, int, str]] = [
    # All flush sites now capture return value (T2-5 fixed).
]

# T-D: conn.execute("UPDATE ...") in curated DB queries.
CURATED_UPDATE_ALLOWLIST: list[tuple[str, int, str]] = [
    # -- queries.py: upsert functions; re-index overwrites are idempotent --
    ("db/queries.py", 13, "upsert_repo: re-index overwrites; raw DB has enrichment audit trail"),
    ("db/queries.py", 39, "upsert_file: re-index overwrites; file identity preserved by path"),
    ("db/queries.py", 146, "upsert_commit: idempotent commit metadata; immutable by nature"),
    ("db/queries.py", 239, "upsert_ref_source: KB re-index overwrites; content is deterministic"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_rbj_call(node: ast.AST) -> bool:
    """Check if an AST node is a call to run_binary_judgment."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "run_binary_judgment":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "run_binary_judgment":
        return True
    return False


def _walk_skip_nested_funcs(node: ast.AST):
    """Yield all descendants of node, skipping nested function bodies."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        yield child
        yield from _walk_skip_nested_funcs(child)


def _has_logger_call(nodes: list[ast.stmt]) -> bool:
    """Check if any node in the list (or descendants) is a logger.xxx() call."""
    for node in nodes:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "logger"
            ):
                return True
    return False


def _has_return(nodes: list[ast.stmt]) -> bool:
    """Check if any node in the list (or descendants) is a Return statement."""
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                return True
    return False


# ---------------------------------------------------------------------------
# T-A: Discarded binary judgment omissions
# ---------------------------------------------------------------------------
class _DiscardedOmissionsVisitor(ast.NodeVisitor):
    """Find run_binary_judgment() calls where 2nd return value is underscore-named."""

    def __init__(self, path: Path):
        self.path = path
        self.violations: list[tuple[Path, int, str]] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        # Pattern: x, _omitted = run_binary_judgment(...)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and len(node.targets[0].elts) >= 2
            and isinstance(node.targets[0].elts[1], ast.Name)
            and node.targets[0].elts[1].id.startswith("_")
            and _is_rbj_call(node.value)
        ):
            line = node.lineno
            var_name = node.targets[0].elts[1].id
            # Check allowlist
            for suffix, allowed_line, _justification in DISCARDED_OMISSIONS_ALLOWLIST:
                if _rel(self.path).endswith(suffix) and allowed_line == line:
                    break
            else:
                self.violations.append(
                    (self.path, line, f"run_binary_judgment() omissions discarded as '{var_name}'")
                )
        self.generic_visit(node)


def test_no_discarded_binary_omissions():
    """T-A: run_binary_judgment() 2nd return value must not be discarded."""
    violations: list[tuple[Path, int, str]] = []
    for path in _collect_py_files(*_CORE_DIRS):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        visitor = _DiscardedOmissionsVisitor(path)
        visitor.visit(tree)
        violations.extend(visitor.violations)

    if violations:
        msg = "run_binary_judgment() omissions discarded (2nd return value ignored):\n"
        for path, line, detail in violations:
            msg += f"  {_rel(path)}:{line}: {detail}\n"
        msg += "\nEither use the omitted keys, or add to DISCARDED_OMISSIONS_ALLOWLIST with justification."
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# T-B: Post-judgment silent cascade drop
# ---------------------------------------------------------------------------
class _SilentCascadeDropVisitor(ast.NodeVisitor):
    """Find 'if not x: return' without logger call in functions using run_binary_judgment."""

    def __init__(self, path: Path):
        self.path = path
        self.violations: list[tuple[Path, int, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Check if this function (excluding nested funcs) calls run_binary_judgment
        # and find the earliest call line to distinguish pre- vs post-judgment guards
        rbj_lines = [
            child.lineno
            for child in _walk_skip_nested_funcs(node)
            if _is_rbj_call(child)
        ]
        if rbj_lines:
            earliest_rbj = min(rbj_lines)
            # Only flag if-not-return patterns AFTER the first binary judgment call.
            # Pre-judgment guards (empty-input checks) are not cascade drops.
            for child in _walk_skip_nested_funcs(node):
                if (
                    isinstance(child, ast.If)
                    and child.lineno > earliest_rbj
                    and isinstance(child.test, ast.UnaryOp)
                    and isinstance(child.test.op, ast.Not)
                    and _has_return(child.body)
                    and not _has_logger_call(child.body)
                ):
                    line = child.lineno
                    # Check allowlist
                    for suffix, allowed_line, _justification in SILENT_CASCADE_DROP_ALLOWLIST:
                        if _rel(self.path).endswith(suffix) and allowed_line == line:
                            break
                    else:
                        self.violations.append(
                            (self.path, line, "'if not ...: return' without logger call after binary judgment")
                        )
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


def test_no_silent_cascade_drops():
    """T-B: Cascade drops in binary judgment functions must log the reason."""
    violations: list[tuple[Path, int, str]] = []
    for path in _collect_py_files(*_CORE_DIRS):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        visitor = _SilentCascadeDropVisitor(path)
        visitor.visit(tree)
        violations.extend(visitor.violations)

    if violations:
        msg = "Silent cascade drop in binary judgment function (if not x: return without logging):\n"
        for path, line, detail in violations:
            msg += f"  {_rel(path)}:{line}: {detail}\n"
        msg += "\nAdd a logger call before returning, or add to SILENT_CASCADE_DROP_ALLOWLIST with justification."
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# T-C: LLM call records discarded without DB persistence
# ---------------------------------------------------------------------------
class _BareFlushVisitor(ast.NodeVisitor):
    """Find .flush() calls where return value is not assigned."""

    def __init__(self, path: Path):
        self.path = path
        self.violations: list[tuple[Path, int, str]] = []

    def visit_Expr(self, node: ast.Expr) -> None:
        # Pattern: something.flush() as a bare statement (not assigned)
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "flush"
        ):
            line = node.lineno
            # Check allowlist
            for suffix, allowed_line, _justification in BARE_FLUSH_ALLOWLIST:
                if _rel(self.path).endswith(suffix) and allowed_line == line:
                    break
            else:
                self.violations.append(
                    (self.path, line, "bare .flush() — call records discarded without persistence")
                )
        self.generic_visit(node)


def test_no_bare_flush_calls():
    """T-C: .flush() return value must be captured for DB persistence."""
    violations: list[tuple[Path, int, str]] = []
    for path in _collect_py_files(*_CORE_DIRS):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        visitor = _BareFlushVisitor(path)
        visitor.visit(tree)
        violations.extend(visitor.violations)

    if violations:
        msg = "Bare .flush() call — LLM call records discarded without DB persistence:\n"
        for path, line, detail in violations:
            msg += f"  {_rel(path)}:{line}: {detail}\n"
        msg += "\nCapture the return value and persist to raw DB, or add to BARE_FLUSH_ALLOWLIST with justification."
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# T-D: Curated DB UPDATE without audit trail
# ---------------------------------------------------------------------------
class _CuratedUpdateVisitor(ast.NodeVisitor):
    """Find conn.execute('UPDATE ...') in curated DB query files."""

    def __init__(self, path: Path):
        self.path = path
        self.violations: list[tuple[Path, int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        # Pattern: conn.execute("UPDATE ...")
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "execute"
            and node.args
        ):
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                sql = first_arg.value.strip()
                if sql.upper().startswith("UPDATE"):
                    line = node.lineno
                    # Check allowlist
                    for suffix, allowed_line, _justification in CURATED_UPDATE_ALLOWLIST:
                        if _rel(self.path).endswith(suffix) and allowed_line == line:
                            break
                    else:
                        display = sql[:80] + "..." if len(sql) > 80 else sql
                        self.violations.append(
                            (self.path, line, f"UPDATE without audit trail: {display}")
                        )
        self.generic_visit(node)


def test_no_curated_update_without_audit():
    """T-D: Curated DB UPDATEs must have audit trail or allowlist justification."""
    # Only scan curated DB queries — raw_queries.py IS the audit trail.
    target = _SRC_ROOT / "db" / "queries.py"
    if not target.exists():
        return

    violations: list[tuple[Path, int, str]] = []
    try:
        source = target.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(target))
    except SyntaxError:
        return

    visitor = _CuratedUpdateVisitor(target)
    visitor.visit(tree)
    violations.extend(visitor.violations)

    if violations:
        msg = "Curated DB UPDATE without audit trail (old values silently overwritten):\n"
        for path, line, detail in violations:
            msg += f"  {_rel(path)}:{line}: {detail}\n"
        msg += "\nAdd audit trail (log old values to raw DB), or add to CURATED_UPDATE_ALLOWLIST with justification."
        pytest.fail(msg)
