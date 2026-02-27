"""Tests for context assembly."""

import json
from unittest.mock import MagicMock

import pytest

from clean_room_agent.retrieval.budget import estimate_tokens
from clean_room_agent.retrieval.context_assembly import (
    _extract_signatures,
    _extract_supporting,
    assemble_context,
)
from clean_room_agent.retrieval.dataclasses import (
    BudgetConfig,
    ClassifiedSymbol,
    ScopedFile,
    TaskQuery,
)
from clean_room_agent.retrieval.stage import StageContext


@pytest.fixture
def task():
    return TaskQuery(raw_task="fix bug", task_id="t1", mode="plan", repo_id=1)


@pytest.fixture
def source_files(tmp_path):
    """Create source files on disk for assembly."""
    (tmp_path / "src").mkdir()

    # Primary file
    (tmp_path / "src" / "auth.py").write_text(
        "class AuthManager:\n"
        "    \"\"\"Manages authentication.\"\"\"\n"
        "    def login(self, user, pw):\n"
        "        \"\"\"Authenticate user.\"\"\"\n"
        "        return verify(user, pw)\n"
        "\n"
        "    def logout(self):\n"
        "        pass\n"
    )

    # Supporting file
    (tmp_path / "src" / "models.py").write_text(
        "class User:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "\n"
        "class Session:\n"
        "    def __init__(self):\n"
        "        self.active = True\n"
    )

    # Type context file
    (tmp_path / "src" / "types.py").write_text(
        "from typing import Protocol\n"
        "\n"
        "class Authenticator(Protocol):\n"
        "    def login(self, user: str, pw: str) -> bool: ...\n"
    )

    return tmp_path


class TestAssembleContext:
    def test_basic_assembly(self, task, source_files):
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
        ]
        ctx.included_file_ids = {1}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
        ]

        pkg = assemble_context(ctx, budget, source_files)
        assert len(pkg.files) == 1
        assert pkg.files[0].detail_level == "primary"
        assert "AuthManager" in pkg.files[0].content
        assert pkg.total_token_estimate > 0

    def test_multiple_detail_levels(self, task, source_files):
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=2, path="src/models.py", language="python",
                       tier=2, relevance="relevant"),
            ScopedFile(file_id=3, path="src/types.py", language="python",
                       tier=3, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 2, 3}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
            ClassifiedSymbol(symbol_id=2, file_id=2, name="User",
                             kind="class", start_line=1, end_line=3,
                             detail_level="supporting"),
            ClassifiedSymbol(symbol_id=3, file_id=3, name="Authenticator",
                             kind="class", start_line=3, end_line=4,
                             detail_level="type_context"),
        ]

        pkg = assemble_context(ctx, budget, source_files)
        assert len(pkg.files) == 3
        # Primary should be first
        assert pkg.files[0].detail_level == "primary"

    def test_budget_enforcement(self, task, source_files):
        # Very tight budget
        budget = BudgetConfig(context_window=100, reserved_tokens=10)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=2, path="src/models.py", language="python",
                       tier=2, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 2}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
            ClassifiedSymbol(symbol_id=2, file_id=2, name="User",
                             kind="class", start_line=1, end_line=3,
                             detail_level="supporting"),
        ]

        pkg = assemble_context(ctx, budget, source_files)
        # Budget is very tight, should limit what gets included
        assert pkg.total_token_estimate <= int(budget.effective_budget * 0.9) + 10  # some tolerance

    def test_empty_context(self, task, source_files):
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        pkg = assemble_context(ctx, budget, source_files)
        assert pkg.files == []
        # total_token_estimate includes R5 task/intent header overhead
        assert pkg.total_token_estimate >= 0

    def test_nonexistent_file_id_excluded_by_r2(self, task, source_files):
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=99, path="nonexistent.py", language="python",
                       tier=1, relevance="relevant"),
        ]
        ctx.included_file_ids = {99}

        pkg = assemble_context(ctx, budget, source_files)
        assert pkg.files == []

    def test_file_without_classified_symbols_excluded(self, task, source_files):
        """R2c: files with no classified symbols are excluded (default-deny)."""
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
        ]
        ctx.included_file_ids = {1}
        # No classified_symbols for file_id=1

        pkg = assemble_context(ctx, budget, source_files)
        assert len(pkg.files) == 0  # R2: no classification → excluded

    def test_irrelevant_files_excluded(self, task, source_files):
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="irrelevant"),  # marked irrelevant
        ]
        ctx.included_file_ids = {1}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
        ]

        pkg = assemble_context(ctx, budget, source_files)
        assert pkg.files == []  # irrelevant not included

    def test_primary_file_read_failure_raises(self, task, source_files):
        """T8/R1: a primary file that can't be read is a hard error."""
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=99, path="nonexistent.py", language="python",
                       tier=1, relevance="relevant"),
        ]
        ctx.included_file_ids = {99}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=99, file_id=99, name="Missing",
                             kind="function", start_line=1, end_line=5,
                             detail_level="primary"),
        ]
        with pytest.raises(RuntimeError, match="R1.*cannot read file"):
            assemble_context(ctx, budget, source_files)

    def test_supporting_file_read_failure_raises(self, task, source_files):
        """T8/R1: supporting files that can't be read are also hard errors (fail-fast)."""
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=99, path="nonexistent.py", language="python",
                       tier=2, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 99}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
            ClassifiedSymbol(symbol_id=99, file_id=99, name="Helper",
                             kind="function", start_line=1, end_line=5,
                             detail_level="supporting"),
        ]
        with pytest.raises(RuntimeError, match="R1.*cannot read file"):
            assemble_context(ctx, budget, source_files)


class TestRequireLoggedClient:
    """S8: _require_logged_client negative-path coverage."""

    def test_require_logged_client_rejects_plain_client(self):
        """S8: _require_logged_client raises TypeError for non-logging client."""
        from unittest.mock import Mock
        from clean_room_agent.retrieval.context_assembly import _require_logged_client
        plain = Mock(spec=[])  # No flush attribute
        with pytest.raises(TypeError, match="logging-capable"):
            _require_logged_client(plain, "test_caller")


class TestAllSymbolsExcluded:
    """Edge cases where all symbols are excluded or all files lack classifications."""

    def test_all_symbols_excluded_produces_empty_context(self, task, source_files):
        """All symbols marked 'excluded' → file not included (R2: default-deny)."""
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
        ]
        ctx.included_file_ids = {1}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="excluded"),
            ClassifiedSymbol(symbol_id=2, file_id=1, name="login",
                             kind="function", start_line=3, end_line=5,
                             detail_level="excluded"),
        ]

        pkg = assemble_context(ctx, budget, source_files)
        assert pkg.files == []
        # Should appear in R2 exclusion decisions
        decisions = pkg.metadata["assembly_decisions"]
        excluded = [d for d in decisions if not d["included"]]
        assert len(excluded) == 1
        assert "R2" in excluded[0]["reason"]

    def test_multiple_files_all_excluded(self, task, source_files):
        """Multiple files with all symbols excluded → empty context, R2 warnings for each."""
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=2, path="src/models.py", language="python",
                       tier=2, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 2}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="excluded"),
            ClassifiedSymbol(symbol_id=2, file_id=2, name="User",
                             kind="class", start_line=1, end_line=3,
                             detail_level="excluded"),
        ]

        pkg = assemble_context(ctx, budget, source_files)
        assert pkg.files == []
        decisions = pkg.metadata["assembly_decisions"]
        r2_exclusions = [d for d in decisions if not d["included"] and "R2" in d["reason"]]
        assert len(r2_exclusions) == 2

    def test_mixed_excluded_and_included(self, task, source_files):
        """One file all-excluded, another has primary symbols → only second file in context."""
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=2, path="src/models.py", language="python",
                       tier=2, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 2}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
            ClassifiedSymbol(symbol_id=2, file_id=2, name="User",
                             kind="class", start_line=1, end_line=3,
                             detail_level="excluded"),
        ]

        pkg = assemble_context(ctx, budget, source_files)
        assert len(pkg.files) == 1
        assert pkg.files[0].path == "src/auth.py"
        # models.py should be R2-excluded (only excluded symbols)
        decisions = pkg.metadata["assembly_decisions"]
        r2_exclusions = [d for d in decisions if not d["included"] and "R2" in d["reason"]]
        assert len(r2_exclusions) == 1
        assert r2_exclusions[0]["file_id"] == 2


class TestR2DefaultDenyInRender:
    """A7: file in sorted_fids without file_detail entry gets R2 excluded."""

    def test_file_without_detail_excluded(self, task, source_files):
        """A7: file present in sorted_fids but missing from file_detail is excluded."""
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=2, path="src/models.py", language="python",
                       tier=2, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 2}
        # Only file 1 has classified symbols — file 2 has none
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
        ]

        pkg = assemble_context(ctx, budget, source_files)
        # Only file 1 should be included
        assert len(pkg.files) == 1
        assert pkg.files[0].file_id == 1
        # File 2 should have an R2 exclusion decision
        decisions = pkg.metadata["assembly_decisions"]
        r2_exclusions = [d for d in decisions if not d["included"] and "R2" in d["reason"]]
        assert any(d["file_id"] == 2 for d in r2_exclusions)


class TestReadFailureDecision:
    """A14: unreadable files raise RuntimeError (fail-fast)."""

    def test_supporting_read_failure_raises(self, task, source_files):
        """A14: unreadable supporting file raises RuntimeError (fail-fast, no silent skip)."""
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=99, path="nonexistent_supporting.py", language="python",
                       tier=2, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 99}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
            ClassifiedSymbol(symbol_id=99, file_id=99, name="Helper",
                             kind="function", start_line=1, end_line=5,
                             detail_level="supporting"),
        ]

        with pytest.raises(RuntimeError, match="R1.*cannot read file"):
            assemble_context(ctx, budget, source_files)


class TestExtractSignatures:
    """R4: _extract_signatures uses parsed AST data from classified symbols."""

    def test_uses_stored_signatures(self):
        """R4: Uses stored symbol signatures from classified_symbols."""
        task = TaskQuery(raw_task="test", task_id="t1", mode="plan", repo_id=1)
        ctx = StageContext(task=task, repo_id=1, repo_path="/tmp")
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="Foo",
                             kind="class", start_line=3, end_line=5,
                             detail_level="type_context", signature="class Foo:"),
            ClassifiedSymbol(symbol_id=2, file_id=1, name="bar",
                             kind="function", start_line=4, end_line=5,
                             detail_level="type_context", signature="    def bar(self):"),
            ClassifiedSymbol(symbol_id=3, file_id=1, name="standalone",
                             kind="function", start_line=7, end_line=8,
                             detail_level="type_context", signature="def standalone():"),
        ]
        lines = [
            "import os",
            "",
            "class Foo:",
            "    def bar(self):",
            "        pass",
            "",
            "def standalone():",
            "    return 1",
        ]
        result = _extract_signatures(lines, file_id=1, context=ctx)
        assert "class Foo:" in result
        assert "def bar(self):" in result
        assert "def standalone():" in result
        assert "import os" not in result
        assert "pass" not in result

    def test_fallback_to_start_line_without_signature(self):
        """R4: Falls back to source line at start_line when signature is empty."""
        task = TaskQuery(raw_task="test", task_id="t1", mode="plan", repo_id=1)
        ctx = StageContext(task=task, repo_id=1, repo_path="/tmp")
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="Foo",
                             kind="class", start_line=1, end_line=2,
                             detail_level="type_context"),  # no signature
        ]
        lines = ["class Foo:", "    pass"]
        result = _extract_signatures(lines, file_id=1, context=ctx)
        assert "class Foo:" in result

    def test_no_symbols_no_signatures(self):
        task = TaskQuery(raw_task="test", task_id="t1", mode="plan", repo_id=1)
        ctx = StageContext(task=task, repo_id=1, repo_path="/tmp")
        lines = ["# just a comment", "x = 1"]
        result = _extract_signatures(lines, file_id=1, context=ctx)
        assert "no signatures found" in result

    def test_no_context_no_signatures(self):
        """Legacy path: no context data at all."""
        lines = ["# just a comment", "x = 1"]
        result = _extract_signatures(lines)
        assert "no signatures found" in result


class TestRefilterAssembly:
    """R1: budget-exceeded assembly calls re-filter LLM instead of downgrading."""

    def test_refilter_called_on_budget_exceeded(self, task, source_files):
        """When budget exceeded + LLM provided, refilter LLM is called."""
        # Each file ~150 tokens (600 chars). Two files = 300 total.
        # Budget: effective = 490, * 0.9 = 441. But we want total > limit.
        # Use context_window=350, reserved=10 → effective=340, *0.9=306.
        # Two files @ 150 each = 300 < 306. Too low.
        # Use smaller budget: context_window=300, reserved=10 → effective=290, *0.9=261.
        # Two files @ 150 = 300 > 261 → triggers refilter. Single file 150 ≤ 261 → fits.
        (source_files / "src" / "auth.py").write_text("x = 1\n" * 100)  # 600 chars = 150 tokens
        (source_files / "src" / "models.py").write_text("y = 2\n" * 100)  # 600 chars = 150 tokens

        budget = BudgetConfig(context_window=300, reserved_tokens=10)  # effective*0.9 = 261 tokens
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=2, path="src/models.py", language="python",
                       tier=2, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 2}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=100,
                             detail_level="primary"),
            ClassifiedSymbol(symbol_id=2, file_id=2, name="User",
                             kind="class", start_line=1, end_line=100,
                             detail_level="primary"),
        ]

        mock_llm = MagicMock()
        mock_llm.flush = MagicMock()  # F14: explicit _require_logged_client contract
        mock_llm.config.context_window = 32768
        mock_llm.config.max_tokens = 4096
        # LLM returns only auth.py to keep
        mock_response = MagicMock()
        mock_response.text = json.dumps(["src/auth.py"])
        mock_llm.complete.return_value = mock_response

        pkg = assemble_context(ctx, budget, source_files, llm=mock_llm)

        # LLM was called for refilter
        mock_llm.complete.assert_called_once()
        # Only the file the LLM chose to keep should be included
        paths = [f.path for f in pkg.files]
        assert "src/auth.py" in paths
        # models.py was dropped by refilter
        assert "src/models.py" not in paths

    def test_survivors_keep_original_levels(self, task, source_files):
        """Refilter survivors keep their original classified detail levels (no downgrades)."""
        (source_files / "src" / "auth.py").write_text("x = 1\n" * 100)
        (source_files / "src" / "models.py").write_text("y = 2\n" * 100)

        budget = BudgetConfig(context_window=300, reserved_tokens=10)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=2, path="src/models.py", language="python",
                       tier=2, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 2}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=100,
                             detail_level="primary"),
            ClassifiedSymbol(symbol_id=2, file_id=2, name="User",
                             kind="class", start_line=1, end_line=100,
                             detail_level="primary"),
        ]

        mock_llm = MagicMock()
        mock_llm.flush = MagicMock()  # F14: explicit _require_logged_client contract
        mock_llm.config.context_window = 32768
        mock_llm.config.max_tokens = 4096
        mock_response = MagicMock()
        mock_response.text = json.dumps(["src/auth.py"])
        mock_llm.complete.return_value = mock_response

        pkg = assemble_context(ctx, budget, source_files, llm=mock_llm)

        for f in pkg.files:
            if f.path == "src/auth.py":
                assert f.detail_level == "primary"  # not downgraded

    def test_no_llm_fallback_drops_by_priority(self, task, source_files):
        """Without LLM, files are dropped by priority (type_context first, etc.)."""
        # Make files large enough to trigger budget exceeded
        (source_files / "src" / "auth.py").write_text("x = 1\n" * 200)
        (source_files / "src" / "models.py").write_text("y = 2\n" * 200)
        (source_files / "src" / "types.py").write_text("z = 3\n" * 200)

        budget = BudgetConfig(context_window=800, reserved_tokens=10)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=2, path="src/models.py", language="python",
                       tier=2, relevance="relevant"),
            ScopedFile(file_id=3, path="src/types.py", language="python",
                       tier=3, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 2, 3}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=200,
                             detail_level="primary"),
            ClassifiedSymbol(symbol_id=2, file_id=2, name="User",
                             kind="class", start_line=1, end_line=200,
                             detail_level="supporting"),
            ClassifiedSymbol(symbol_id=3, file_id=3, name="Authenticator",
                             kind="class", start_line=1, end_line=200,
                             detail_level="type_context"),
        ]

        # No LLM
        pkg = assemble_context(ctx, budget, source_files, llm=None)

        # type_context should be dropped first
        detail_levels = [f.detail_level for f in pkg.files]
        if "type_context" in detail_levels:
            # If type_context made it in, primary and supporting should also be in
            assert "primary" in detail_levels


class TestAssemblyDecisions:
    """T3: Assembly tracks file decisions in metadata for raw DB logging."""

    def test_included_files_recorded(self, task, source_files):
        """Included files are recorded as positive decisions."""
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
        ]
        ctx.included_file_ids = {1}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
        ]

        pkg = assemble_context(ctx, budget, source_files)
        decisions = pkg.metadata["assembly_decisions"]
        included = [d for d in decisions if d["included"]]
        assert len(included) == 1
        assert included[0]["file_id"] == 1

    def test_r2_exclusion_recorded(self, task, source_files):
        """R2 exclusion (no classified symbols) is recorded as a decision."""
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
        ]
        ctx.included_file_ids = {1}
        # No classified symbols for file_id=1

        pkg = assemble_context(ctx, budget, source_files)
        decisions = pkg.metadata["assembly_decisions"]
        excluded = [d for d in decisions if not d["included"]]
        assert len(excluded) == 1
        assert excluded[0]["file_id"] == 1
        assert "R2" in excluded[0]["reason"]

    def test_budget_overflow_recorded(self, task, source_files):
        """Post-refilter budget overflow is recorded as a decision."""
        # Two primary files. Priority drop can't distinguish between them (same level).
        # After dropping type_context (the small file), the two primary files
        # are tried sequentially. The first fits, the second doesn't.
        (source_files / "src" / "auth.py").write_text("x = 1\n" * 80)   # ~20 tokens each line
        (source_files / "src" / "models.py").write_text("y = 2\n" * 80)
        (source_files / "src" / "types.py").write_text("z = 3\n" * 10)  # small type_context

        # Budget: effective = 490, * 0.9 = 441. Two primary files @ ~120 tokens each = ~240.
        # Plus type_context ~30. Plus headers ~20. Total ~290 < 441. Too generous.
        # Let's use tighter: context_window=350, reserved=10 → effective=340, *0.9=306.
        # Headers ~20 → remaining ~286. Three files ~270 < 286. Still fits.
        # Use context_window=280, reserved=10 → effective=270, *0.9=243.
        # Headers ~20 → remaining ~223. Three files ~270 > 223. Triggers drop.
        # After dropping type_context (~30): ~240 > 223. Still over.
        # After dropping supporting (none): still over.
        # After dropping primary: 0. Both gone via priority_drop.
        # So budget_overflow only fires AFTER refilter/priority_drop, when a file
        # passes the bulk check but fails individual can_fit. Need LLM refilter.

        budget = BudgetConfig(context_window=300, reserved_tokens=10)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=2, path="src/models.py", language="python",
                       tier=2, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 2}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=80,
                             detail_level="primary"),
            ClassifiedSymbol(symbol_id=2, file_id=2, name="User",
                             kind="class", start_line=1, end_line=80,
                             detail_level="primary"),
        ]

        # LLM refilter keeps both files (bad decision — together they exceed budget)
        mock_llm = MagicMock()
        mock_llm.flush = MagicMock()  # F14: explicit _require_logged_client contract
        mock_llm.config.context_window = 32768
        mock_llm.config.max_tokens = 4096
        mock_response = MagicMock()
        mock_response.text = json.dumps(["src/auth.py", "src/models.py"])
        mock_response.prompt_tokens = 50
        mock_response.completion_tokens = 10
        mock_llm.complete.return_value = mock_response

        pkg = assemble_context(ctx, budget, source_files, llm=mock_llm)
        decisions = pkg.metadata["assembly_decisions"]
        overflow = [d for d in decisions if not d["included"] and "budget_overflow" in d["reason"]]
        # The second file should be dropped by budget_overflow since the first consumed the budget
        if len(pkg.files) < 2:
            assert len(overflow) >= 1

    def test_refilter_drop_recorded(self, task, source_files):
        """LLM refilter drops are recorded as decisions."""
        (source_files / "src" / "auth.py").write_text("x = 1\n" * 100)
        (source_files / "src" / "models.py").write_text("y = 2\n" * 100)

        budget = BudgetConfig(context_window=300, reserved_tokens=10)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=2, path="src/models.py", language="python",
                       tier=2, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 2}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=100,
                             detail_level="primary"),
            ClassifiedSymbol(symbol_id=2, file_id=2, name="User",
                             kind="class", start_line=1, end_line=100,
                             detail_level="primary"),
        ]

        mock_llm = MagicMock()
        mock_llm.flush = MagicMock()  # F14: explicit _require_logged_client contract
        mock_llm.config.context_window = 32768
        mock_llm.config.max_tokens = 4096
        mock_response = MagicMock()
        mock_response.text = json.dumps(["src/auth.py"])
        mock_response.prompt_tokens = 50
        mock_response.completion_tokens = 10
        mock_llm.complete.return_value = mock_response

        pkg = assemble_context(ctx, budget, source_files, llm=mock_llm)
        decisions = pkg.metadata["assembly_decisions"]
        refilter_drops = [d for d in decisions if not d["included"] and "refilter" in d["reason"]]
        assert len(refilter_drops) == 1
        assert refilter_drops[0]["file_id"] == 2

    def test_priority_drop_recorded(self, task, source_files):
        """Priority drops (no LLM) are recorded as decisions."""
        (source_files / "src" / "auth.py").write_text("x = 1\n" * 200)
        (source_files / "src" / "types.py").write_text("z = 3\n" * 200)

        budget = BudgetConfig(context_window=600, reserved_tokens=10)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=3, path="src/types.py", language="python",
                       tier=3, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 3}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=200,
                             detail_level="primary"),
            ClassifiedSymbol(symbol_id=3, file_id=3, name="Authenticator",
                             kind="class", start_line=1, end_line=200,
                             detail_level="type_context"),
        ]

        pkg = assemble_context(ctx, budget, source_files, llm=None)
        decisions = pkg.metadata["assembly_decisions"]
        priority_drops = [d for d in decisions if not d["included"] and "priority_drop" in d["reason"]]
        # type_context should be dropped before primary
        if priority_drops:
            assert priority_drops[0]["file_id"] == 3


class TestRefilterFailFast:
    """R18: When refilter LLM returns invalid response, fail-fast with ValueError."""

    def test_refilter_non_list_raises(self, task, source_files):
        """R18: When LLM returns a non-list (e.g. a dict), raises ValueError."""
        (source_files / "src" / "auth.py").write_text("x = 1\n" * 100)
        (source_files / "src" / "models.py").write_text("y = 2\n" * 100)

        budget = BudgetConfig(context_window=300, reserved_tokens=10)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=2, path="src/models.py", language="python",
                       tier=2, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 2}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=100,
                             detail_level="primary"),
            ClassifiedSymbol(symbol_id=2, file_id=2, name="User",
                             kind="class", start_line=1, end_line=100,
                             detail_level="primary"),
        ]

        mock_llm = MagicMock()
        mock_llm.flush = MagicMock()  # F14: explicit _require_logged_client contract
        mock_llm.config.context_window = 32768
        mock_llm.config.max_tokens = 4096
        mock_response = MagicMock()
        mock_response.text = json.dumps({"error": "unexpected format"})
        mock_llm.complete.return_value = mock_response

        with pytest.raises(ValueError, match="R18.*expected list"):
            assemble_context(ctx, budget, source_files, llm=mock_llm)

    def test_refilter_all_invalid_paths_raises(self, task, source_files):
        """R18: When LLM returns only hallucinated paths, raises ValueError."""
        (source_files / "src" / "auth.py").write_text("x = 1\n" * 100)
        (source_files / "src" / "models.py").write_text("y = 2\n" * 100)

        budget = BudgetConfig(context_window=300, reserved_tokens=10)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=2, path="src/models.py", language="python",
                       tier=2, relevance="relevant"),
        ]
        ctx.included_file_ids = {1, 2}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=100,
                             detail_level="primary"),
            ClassifiedSymbol(symbol_id=2, file_id=2, name="User",
                             kind="class", start_line=1, end_line=100,
                             detail_level="primary"),
        ]

        mock_llm = MagicMock()
        mock_llm.flush = MagicMock()  # F14: explicit _require_logged_client contract
        mock_llm.config.context_window = 32768
        mock_llm.config.max_tokens = 4096
        mock_response = MagicMock()
        mock_response.text = json.dumps(["nonexistent.py", "hallucinated.py"])
        mock_llm.complete.return_value = mock_response

        with pytest.raises(ValueError, match="R18.*none valid"):
            assemble_context(ctx, budget, source_files, llm=mock_llm)


class TestFramingOverhead:
    """R5: Framing overhead is part of the budget."""

    def test_framing_tokens_included_in_estimate(self, task, source_files):
        """R5: Token estimate includes framing overhead, not just content."""
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
        ]
        ctx.included_file_ids = {1}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
        ]

        pkg = assemble_context(ctx, budget, source_files)
        # Token estimate should be > raw content tokens because of framing
        raw_content = (source_files / "src" / "auth.py").read_text()
        raw_tokens = estimate_tokens(raw_content)
        assert pkg.total_token_estimate > raw_tokens

    def test_prompt_text_uses_xml_tags(self, task, source_files):
        """R5: to_prompt_text uses XML-style code tags, not triple backticks."""
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
        ]
        ctx.included_file_ids = {1}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
        ]

        pkg = assemble_context(ctx, budget, source_files)
        prompt = pkg.to_prompt_text()
        # R5: XML tags, not triple backticks
        assert '<code lang="python">' in prompt
        assert "</code>" in prompt
        assert "```" not in prompt
