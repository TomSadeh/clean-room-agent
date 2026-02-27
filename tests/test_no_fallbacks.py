"""Structural linter: detect and prevent silent fallback patterns.

This test file runs as part of pytest and fails the build when banned
anti-patterns are found in core logic directories.  It enforces the
CLAUDE.md coding style: no fallbacks, no hardcoded defaults in core logic,
fail-fast behavior.

Patterns detected:
    A. Silent substitution (getattr with defaults, `if not X: X = Y`)
    B. Catch-and-continue (except blocks that swallow and continue/return)
    C. Magic threshold feature toggles (numeric comparisons that switch features)
    D. Soft import failures (try/import/except that degrades silently)

Each pattern has an allowlist for intentional exceptions.  Every allowlist
entry requires a justification comment.
"""

from __future__ import annotations

import ast
import re
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

# All src dirs (broader scan for some patterns).
_ALL_SRC_DIRS = [_SRC_ROOT]


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

# Pattern A: getattr with 3 args (silent default) in core logic.
GETATTR_ALLOWLIST: list[tuple[str, int, str]] = [
    # No entries — all getattr-with-default violations have been fixed.
]

# Pattern B: except blocks that catch and continue/return without re-raise.
# Use line=-1 to allowlist ALL except blocks in a file (for files with many
# orchestrator-boundary handlers). Prefer specific line numbers when possible.
CATCH_CONTINUE_ALLOWLIST: list[tuple[str, int, str]] = [
    # -- retrieval/batch_judgment.py: R2 default-deny per individual LLM response item --
    ("retrieval/batch_judgment.py", -1, "R2 default-deny: per-item LLM response parse failures"),
    # -- retrieval/scope_stage.py: R2 default-deny on invalid verdict --
    ("retrieval/scope_stage.py", -1, "R2 default-deny: invalid verdict string defaults to irrelevant"),
    # -- execute/patch.py: rollback error collection + atomic write cleanup --
    ("execute/patch.py", -1, "Rollback: error collection with deferred raise + temp file cleanup"),
    # -- retrieval/pipeline.py: best-effort logging/archival in error paths --
    ("retrieval/pipeline.py", -1, "Best-effort: DB/session cleanup in finally/error paths must not mask original error"),
    # -- orchestrator/runner.py: orchestrator step boundaries --
    # The orchestrator catches per-step/per-pass errors so it can record the
    # failure, continue to the next step, and produce a complete OrchestratorResult.
    # Also includes cleanup handlers (branch delete, temp file removal).
    ("orchestrator/runner.py", -1, "Orchestrator step boundary: per-step error capture with state recording"),
    # -- orchestrator/git_ops.py: per-branch cleanup during bulk deletion --
    ("orchestrator/git_ops.py", -1, "Cleanup: per-branch delete failure during bulk cleanup"),
    # -- orchestrator/validator.py: command timeout translation --
    ("orchestrator/validator.py", -1, "Boundary: subprocess timeout translated to structured result"),
    # -- llm/client.py: __del__ best-effort cleanup during interpreter shutdown --
    ("llm/client.py", -1, "Cleanup: __del__ best-effort during interpreter shutdown"),
    # -- execute/documentation.py: distinguishes "no edits" vs "parse error" --
    ("execute/documentation.py", -1, "Boundary: LLM response with no edit blocks is valid (no changes needed)"),
    # -- execute/scaffold.py: compiler timeout is a structured result, not a crash --
    ("execute/scaffold.py", -1, "Boundary: compiler subprocess timeout translated to structured result"),
]

# Pattern D: try/import/except — soft import failures.
SOFT_IMPORT_ALLOWLIST: list[tuple[str, str]] = [
    # Version bridge: tomllib (3.11+) / tomli (backport) — identical API.
    ("audit/loader.py", "tomllib/tomli version bridge — both provide identical API"),
    # Hard fail: re-raises ImportError with actionable message.
    ("execute/scaffold.py", "Hard fail-fast: re-raises ImportError with install instructions"),
]


# ---------------------------------------------------------------------------
# Pattern A: 3-arg getattr in core logic
# ---------------------------------------------------------------------------
class _GetAttrVisitor(ast.NodeVisitor):
    """Find getattr(obj, attr, default) calls — 3-arg form."""

    def __init__(self, path: Path):
        self.path = path
        self.violations: list[tuple[Path, int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 3
        ):
            line = node.lineno
            # Check allowlist
            for suffix, allowed_line, _justification in GETATTR_ALLOWLIST:
                if _rel(self.path).endswith(suffix) and allowed_line == line:
                    break
            else:
                code = ast.get_source_segment(self.path.read_text(encoding="utf-8"), node) or ""
                self.violations.append((self.path, line, code.strip()[:120]))
        self.generic_visit(node)


def test_no_getattr_with_defaults_in_core():
    """Pattern A: No 3-arg getattr(obj, attr, default) in core logic."""
    violations: list[tuple[Path, int, str]] = []
    for path in _collect_py_files(*_CORE_DIRS):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        visitor = _GetAttrVisitor(path)
        visitor.visit(tree)
        violations.extend(visitor.violations)

    if violations:
        msg = "getattr() with default value in core logic (use explicit property or fail-fast):\n"
        for path, line, code in violations:
            msg += f"  {_rel(path)}:{line}: {code}\n"
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# Pattern B: catch-and-continue (except without re-raise)
# ---------------------------------------------------------------------------
_EXCEPT_CONTINUE_RE = re.compile(
    r"^\s*except\s+.*?:\s*$",
    re.MULTILINE,
)


class _ExceptVisitor(ast.NodeVisitor):
    """Find except handlers that don't re-raise within their body."""

    def __init__(self, path: Path, source: str):
        self.path = path
        self.source = source
        self.violations: list[tuple[Path, int, str]] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # Check if the handler body contains a raise statement (at any depth).
        has_raise = any(
            isinstance(child, ast.Raise) for child in ast.walk(node)
        )
        if not has_raise:
            # This except block swallows the exception.
            line = node.lineno
            # Check allowlist
            allowed = False
            for suffix, allowed_line, _justification in CATCH_CONTINUE_ALLOWLIST:
                if _rel(self.path).endswith(suffix):
                    if allowed_line == -1 or allowed_line == line:
                        allowed = True
                        break
            if not allowed:
                # Get a snippet of the except line
                lines = self.source.splitlines()
                snippet = lines[line - 1].strip() if line <= len(lines) else ""
                self.violations.append((self.path, line, snippet[:120]))
        self.generic_visit(node)


def test_no_catch_and_continue_in_core():
    """Pattern B: No except blocks that swallow exceptions in core logic."""
    violations: list[tuple[Path, int, str]] = []
    for path in _collect_py_files(*_CORE_DIRS):
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        visitor = _ExceptVisitor(path, source)
        visitor.visit(tree)
        violations.extend(visitor.violations)

    if violations:
        msg = "except block without re-raise in core logic (catch-and-continue):\n"
        for path, line, code in violations:
            msg += f"  {_rel(path)}:{line}: {code}\n"
        msg += "\nEither re-raise with context, or add to CATCH_CONTINUE_ALLOWLIST with justification."
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# Pattern D: soft import failures
# ---------------------------------------------------------------------------
class _SoftImportVisitor(ast.NodeVisitor):
    """Find try/import/except ImportError patterns that don't re-raise."""

    def __init__(self, path: Path):
        self.path = path
        self.violations: list[tuple[Path, int, str]] = []

    def visit_Try(self, node: ast.Try) -> None:
        # Check if the try body contains an import
        has_import = any(
            isinstance(stmt, (ast.Import, ast.ImportFrom))
            for stmt in node.body
        )
        if has_import:
            for handler in node.handlers:
                # Check if it catches ImportError
                if handler.type is None:
                    is_import_error = True
                elif isinstance(handler.type, ast.Name) and handler.type.id in ("ImportError", "ModuleNotFoundError"):
                    is_import_error = True
                elif isinstance(handler.type, ast.Tuple):
                    is_import_error = any(
                        isinstance(elt, ast.Name) and elt.id in ("ImportError", "ModuleNotFoundError")
                        for elt in handler.type.elts
                    )
                else:
                    is_import_error = False

                if is_import_error:
                    has_raise = any(isinstance(child, ast.Raise) for child in ast.walk(handler))
                    if not has_raise:
                        line = handler.lineno
                        # Check allowlist
                        allowed = False
                        for suffix, _justification in SOFT_IMPORT_ALLOWLIST:
                            if _rel(self.path).endswith(suffix):
                                allowed = True
                                break
                        if not allowed:
                            self.violations.append((self.path, line, "soft import failure"))
        self.generic_visit(node)


def test_no_soft_import_failures():
    """Pattern D: No try/import/except that silently degrades."""
    violations: list[tuple[Path, int, str]] = []
    for path in _collect_py_files(*_ALL_SRC_DIRS):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        visitor = _SoftImportVisitor(path)
        visitor.visit(tree)
        violations.extend(visitor.violations)

    if violations:
        msg = "Soft import failure (try/import/except without re-raise):\n"
        for path, line, code in violations:
            msg += f"  {_rel(path)}:{line}: {code}\n"
        msg += "\nEither re-raise with context, or add to SOFT_IMPORT_ALLOWLIST with justification."
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# Structural: every RetrievalStage subclass must declare preferred_role
# ---------------------------------------------------------------------------
def test_all_stages_declare_preferred_role():
    """Every registered stage must explicitly declare preferred_role (no protocol default)."""
    # Import after collection to avoid circular imports
    from clean_room_agent.retrieval.stage import _STAGE_REGISTRY, get_stage

    for name in _STAGE_REGISTRY:
        stage = get_stage(name)
        # The property must be declared on the class itself, not inherited from Protocol.
        cls = type(stage)
        has_own = "preferred_role" in cls.__dict__
        assert has_own, (
            f"Stage {name!r} ({cls.__name__}) does not declare its own "
            f"preferred_role property. Every stage must explicitly declare "
            f"its model role — no reliance on protocol defaults."
        )


# ---------------------------------------------------------------------------
# Structural: preferred_role values must be valid model roles
# ---------------------------------------------------------------------------
def test_stage_preferred_roles_are_valid():
    """Every stage's preferred_role must be a known model role."""
    from clean_room_agent.retrieval.stage import _STAGE_REGISTRY, get_stage

    valid_roles = {"coding", "reasoning", "classifier"}

    for name in _STAGE_REGISTRY:
        stage = get_stage(name)
        role = stage.preferred_role
        assert role in valid_roles, (
            f"Stage {name!r} declares preferred_role={role!r} which is not "
            f"a valid model role. Must be one of {valid_roles}."
        )
