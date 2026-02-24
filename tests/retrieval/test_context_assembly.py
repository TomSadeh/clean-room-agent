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

    def test_missing_file_skipped(self, task, source_files):
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
