"""Tests for metadata surfacing in scope, precision, assembly, and to_prompt_text (Feature 1)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from clean_room_agent.db.connection import get_connection
from clean_room_agent.db.queries import (
    insert_docstring,
    insert_symbol,
    upsert_file,
    upsert_file_metadata,
    upsert_repo,
)
from clean_room_agent.query.api import KnowledgeBase
from clean_room_agent.retrieval.budget import estimate_tokens
from clean_room_agent.retrieval.context_assembly import assemble_context
from clean_room_agent.retrieval.dataclasses import (
    BudgetConfig,
    ClassifiedSymbol,
    ContextPackage,
    FileContent,
    ScopedFile,
    TaskQuery,
)
from clean_room_agent.retrieval.precision_stage import classify_symbols
from clean_room_agent.retrieval.scope_stage import judge_scope
from clean_room_agent.retrieval.stage import StageContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm_with_response(text):
    """Create a mock LLM with standard config and a fixed response."""
    mock_llm = MagicMock()
    mock_llm.config.context_window = 32768
    mock_llm.config.max_tokens = 4096
    mock_response = MagicMock()
    mock_response.text = text
    mock_llm.complete.return_value = mock_response
    return mock_llm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scope_kb(tmp_repo):
    """KB with files, some having metadata, for scope judgment tests."""
    conn = get_connection("curated", repo_path=tmp_repo)
    repo_id = upsert_repo(conn, str(tmp_repo), None)

    fid1 = upsert_file(conn, repo_id, "src/auth.py", "python", "h1", 500)
    fid2 = upsert_file(conn, repo_id, "src/utils.py", "python", "h2", 300)
    fid3 = upsert_file(conn, repo_id, "src/config.py", "python", "h3", 200)

    # Only fid1 and fid2 get metadata
    upsert_file_metadata(
        conn, fid1,
        purpose="Authentication and authorization",
        domain="security",
        concepts="login,oauth,jwt",
    )
    upsert_file_metadata(
        conn, fid2,
        purpose="Shared utility functions",
        domain="core",
        concepts="parsing,validation",
    )
    # fid3 intentionally has no metadata

    conn.commit()
    kb = KnowledgeBase(conn)
    yield kb, repo_id, {"auth": fid1, "utils": fid2, "config": fid3}
    conn.close()


@pytest.fixture
def precision_kb(tmp_repo):
    """KB with files, symbols, and docstrings for precision tests."""
    conn = get_connection("curated", repo_path=tmp_repo)
    repo_id = upsert_repo(conn, str(tmp_repo), None)

    fid1 = upsert_file(conn, repo_id, "src/auth.py", "python", "h1", 500)

    sid1 = insert_symbol(conn, fid1, "login", "function", 10, 30, "def login(username, password)")
    sid2 = insert_symbol(conn, fid1, "verify_token", "function", 35, 50, "def verify_token(token)")

    # Docstrings associated with symbols
    insert_docstring(conn, fid1, "Authenticate user with credentials.", "google", symbol_id=sid1)
    insert_docstring(conn, fid1, "Verify a JWT token.\nReturns decoded payload.", "google", symbol_id=sid2)

    conn.commit()
    kb = KnowledgeBase(conn)
    yield kb, repo_id, {"auth": fid1}, {"login": sid1, "verify": sid2}
    conn.close()


@pytest.fixture
def assembly_source_files(tmp_path):
    """Create source files on disk for assembly tests."""
    (tmp_path / "src").mkdir()
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
    (tmp_path / "src" / "utils.py").write_text(
        "def parse(data):\n"
        "    return data.strip()\n"
        "\n"
        "def validate(schema, data):\n"
        "    return True\n"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Scope stage: metadata in candidate lines
# ---------------------------------------------------------------------------

class TestScopeCandidateLinesWithMetadata:
    def test_scope_candidate_lines_with_metadata(self, scope_kb):
        """When metadata exists, scope judgment candidate lines include
        purpose/domain/concepts."""
        kb, repo_id, fids = scope_kb

        # Non-seed candidates (tier 2) so they go through LLM judgment
        candidates = [
            ScopedFile(file_id=fids["auth"], path="src/auth.py",
                       language="python", tier=2, reason="imported"),
            ScopedFile(file_id=fids["utils"], path="src/utils.py",
                       language="python", tier=2, reason="co-changed"),
        ]
        task = TaskQuery(raw_task="fix auth", task_id="t1", mode="plan", repo_id=repo_id)

        # Mock LLM: capture the prompt to inspect candidate lines
        mock_llm = MagicMock()
        mock_llm.config.context_window = 32768
        mock_llm.config.max_tokens = 4096

        mock_response = MagicMock()
        mock_response.text = json.dumps([
            {"path": "src/auth.py", "verdict": "relevant", "reason": "auth related"},
            {"path": "src/utils.py", "verdict": "relevant", "reason": "utility"},
        ])
        mock_llm.complete.return_value = mock_response

        result = judge_scope(candidates, task, mock_llm, kb=kb)

        # Inspect the prompt sent to the LLM
        call_args = mock_llm.complete.call_args
        prompt_text = call_args[0][0]

        # Metadata for auth.py should appear in the prompt
        assert "purpose=Authentication and authorization" in prompt_text
        assert "domain=security" in prompt_text
        assert "concepts=login,oauth,jwt" in prompt_text

        # Metadata for utils.py should also appear
        assert "purpose=Shared utility functions" in prompt_text
        assert "domain=core" in prompt_text

    def test_scope_candidate_lines_without_metadata(self, scope_kb):
        """When no metadata exists, candidate lines are unchanged (no crash)."""
        kb, repo_id, fids = scope_kb

        # Only config.py (no metadata)
        candidates = [
            ScopedFile(file_id=fids["config"], path="src/config.py",
                       language="python", tier=2, reason="imported"),
        ]
        task = TaskQuery(raw_task="fix config", task_id="t1", mode="plan", repo_id=repo_id)

        mock_llm = MagicMock()
        mock_llm.config.context_window = 32768
        mock_llm.config.max_tokens = 4096

        mock_response = MagicMock()
        mock_response.text = json.dumps([
            {"path": "src/config.py", "verdict": "relevant", "reason": "needed"},
        ])
        mock_llm.complete.return_value = mock_response

        result = judge_scope(candidates, task, mock_llm, kb=kb)

        # Should not crash; config.py should be judged normally
        assert len(result) == 1
        assert result[0].relevance == "relevant"

        # Prompt should contain the basic line but no metadata brackets
        call_args = mock_llm.complete.call_args
        prompt_text = call_args[0][0]
        assert "src/config.py" in prompt_text
        # No metadata section for config.py
        assert "purpose=" not in prompt_text

    def test_scope_without_kb_no_metadata(self):
        """When kb is None, metadata is not fetched (no crash)."""
        candidates = [
            ScopedFile(file_id=1, path="some/file.py",
                       language="python", tier=2, reason="dep"),
        ]
        task = TaskQuery(raw_task="fix it", task_id="t1", mode="plan", repo_id=1)

        mock_llm = _mock_llm_with_response(json.dumps([
            {"path": "some/file.py", "verdict": "relevant", "reason": "ok"},
        ]))

        result = judge_scope(candidates, task, mock_llm, kb=None)
        assert len(result) == 1
        assert result[0].relevance == "relevant"


# ---------------------------------------------------------------------------
# Precision stage: docstrings in symbol lines
# ---------------------------------------------------------------------------

class TestPrecisionSymbolsWithDocstrings:
    def test_precision_symbols_with_docstrings(self, precision_kb):
        """When docstrings exist, classify_symbols includes doc summaries in symbol lines."""
        kb, repo_id, fids, sids = precision_kb

        candidates = [
            {
                "symbol_id": sids["login"], "file_id": fids["auth"],
                "file_path": "src/auth.py", "name": "login", "kind": "function",
                "start_line": 10, "end_line": 30,
                "signature": "def login(username, password)", "connections": [],
            },
            {
                "symbol_id": sids["verify"], "file_id": fids["auth"],
                "file_path": "src/auth.py", "name": "verify_token", "kind": "function",
                "start_line": 35, "end_line": 50,
                "signature": "def verify_token(token)", "connections": [],
            },
        ]
        task = TaskQuery(raw_task="fix login", task_id="t1", mode="plan", repo_id=repo_id)

        # Mock LLM: capture prompt
        mock_llm = MagicMock()
        mock_llm.config.context_window = 32768
        mock_llm.config.max_tokens = 4096

        mock_response = MagicMock()
        mock_response.text = json.dumps([
            {"name": "login", "file_path": "src/auth.py", "start_line": 10,
             "detail_level": "primary", "reason": "directly changed"},
            {"name": "verify_token", "file_path": "src/auth.py", "start_line": 35,
             "detail_level": "supporting", "reason": "called by login"},
        ])
        mock_llm.complete.return_value = mock_response

        result = classify_symbols(candidates, task, mock_llm, kb=kb)

        # The LLM should have been called with docstring summaries in the prompt
        call_args = mock_llm.complete.call_args
        prompt_text = call_args[0][0]

        # First line of login's docstring
        assert "Authenticate user with credentials." in prompt_text
        # First line of verify_token's docstring
        assert "Verify a JWT token." in prompt_text

        # Results should still be classified correctly
        assert len(result) == 2
        login_cs = next(cs for cs in result if cs.name == "login")
        verify_cs = next(cs for cs in result if cs.name == "verify_token")
        assert login_cs.detail_level == "primary"
        assert verify_cs.detail_level == "supporting"

    def test_precision_symbols_without_docstrings(self):
        """When no kb is provided, no docstrings are included (no crash)."""
        candidates = [
            {
                "symbol_id": 1, "file_id": 10, "file_path": "auth.py",
                "name": "login", "kind": "function", "start_line": 1, "end_line": 10,
                "signature": "def login()", "connections": [],
            },
        ]
        task = TaskQuery(raw_task="fix login", task_id="t1", mode="plan", repo_id=1)

        mock_llm = _mock_llm_with_response(json.dumps([
            {"name": "login", "file_path": "auth.py", "start_line": 1,
             "detail_level": "primary", "reason": "target"},
        ]))

        result = classify_symbols(candidates, task, mock_llm, kb=None)
        assert len(result) == 1
        assert result[0].detail_level == "primary"

        # Prompt should not contain "doc:" since no KB
        prompt_text = mock_llm.complete.call_args[0][0]
        assert "doc:" not in prompt_text


# ---------------------------------------------------------------------------
# Assembly: metadata_summary in FileContent and to_prompt_text
# ---------------------------------------------------------------------------

class TestToPromptTextRendersMetadata:
    def test_to_prompt_text_renders_metadata(self):
        """FileContent with metadata_summary renders it in to_prompt_text output."""
        task = TaskQuery(raw_task="fix bug", task_id="t1", mode="plan", repo_id=1)
        pkg = ContextPackage(
            task=task,
            files=[
                FileContent(
                    file_id=1,
                    path="src/auth.py",
                    language="python",
                    content="class AuthManager:\n    pass\n",
                    token_estimate=10,
                    detail_level="primary",
                    metadata_summary="Purpose: Authentication handler | Domain: security",
                ),
            ],
            total_token_estimate=20,
        )

        prompt = pkg.to_prompt_text()

        # Metadata summary should appear in the rendered output
        assert "Purpose: Authentication handler | Domain: security" in prompt
        # The file content should also be present
        assert "class AuthManager:" in prompt
        # Standard header
        assert "## src/auth.py [python] (primary)" in prompt

    def test_to_prompt_text_no_metadata(self):
        """FileContent without metadata_summary renders normally."""
        task = TaskQuery(raw_task="fix bug", task_id="t1", mode="plan", repo_id=1)
        pkg = ContextPackage(
            task=task,
            files=[
                FileContent(
                    file_id=1,
                    path="src/auth.py",
                    language="python",
                    content="class AuthManager:\n    pass\n",
                    token_estimate=10,
                    detail_level="primary",
                    metadata_summary="",
                ),
            ],
            total_token_estimate=20,
        )

        prompt = pkg.to_prompt_text()

        # Content should be rendered
        assert "class AuthManager:" in prompt
        assert "## src/auth.py [python] (primary)" in prompt
        # No metadata line between header and code tag
        # The header line should be immediately followed by the code tag
        lines = prompt.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("## src/auth.py"):
                # Next line should be the opening code tag, not a metadata line
                assert lines[i + 1].startswith("<code lang=")
                break

    def test_to_prompt_text_multiple_files_mixed_metadata(self):
        """Multiple files: one with metadata, one without."""
        task = TaskQuery(raw_task="fix bug", task_id="t1", mode="plan", repo_id=1)
        pkg = ContextPackage(
            task=task,
            files=[
                FileContent(
                    file_id=1,
                    path="src/auth.py",
                    language="python",
                    content="def login(): pass",
                    token_estimate=10,
                    detail_level="primary",
                    metadata_summary="Purpose: Auth | Domain: security",
                ),
                FileContent(
                    file_id=2,
                    path="src/config.py",
                    language="python",
                    content="DEBUG = True",
                    token_estimate=5,
                    detail_level="supporting",
                    metadata_summary="",
                ),
            ],
            total_token_estimate=25,
        )

        prompt = pkg.to_prompt_text()

        # auth.py has metadata
        assert "Purpose: Auth | Domain: security" in prompt
        # config.py does not
        assert "## src/config.py [python] (supporting)" in prompt
        # Both have content
        assert "def login(): pass" in prompt
        assert "DEBUG = True" in prompt


# ---------------------------------------------------------------------------
# Assembly: token estimate includes metadata overhead
# ---------------------------------------------------------------------------

class TestTokenEstimateIncludesMetadata:
    def test_token_estimate_includes_metadata(self, assembly_source_files):
        """Token estimate in assembly includes metadata overhead."""
        task = TaskQuery(raw_task="fix bug", task_id="t1", mode="plan", repo_id=1)

        # Set up a KB with metadata for auth.py
        conn = get_connection("curated", repo_path=assembly_source_files)
        repo_id = upsert_repo(conn, str(assembly_source_files), None)
        fid1 = upsert_file(conn, repo_id, "src/auth.py", "python", "h1", 500)
        upsert_file_metadata(
            conn, fid1,
            purpose="Authentication and authorization module",
            domain="security",
        )
        conn.commit()
        kb = KnowledgeBase(conn)

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=repo_id, repo_path=str(assembly_source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=fid1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
        ]
        ctx.included_file_ids = {fid1}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=fid1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
        ]

        # Assembly WITH metadata (via kb)
        pkg_with_meta = assemble_context(ctx, budget, assembly_source_files, kb=kb)

        # Assembly WITHOUT metadata (no kb)
        pkg_no_meta = assemble_context(ctx, budget, assembly_source_files, kb=None)

        # With metadata, the token estimate should be higher because the
        # metadata_summary string adds to the per-file token count
        assert pkg_with_meta.total_token_estimate > pkg_no_meta.total_token_estimate

        # The file should have a metadata_summary when KB is provided
        assert len(pkg_with_meta.files) == 1
        assert pkg_with_meta.files[0].metadata_summary != ""
        assert "Purpose:" in pkg_with_meta.files[0].metadata_summary
        assert "Domain:" in pkg_with_meta.files[0].metadata_summary

        # Without KB, no metadata_summary
        assert len(pkg_no_meta.files) == 1
        assert pkg_no_meta.files[0].metadata_summary == ""

        conn.close()

    def test_metadata_tokens_are_nontrivial(self, assembly_source_files):
        """Metadata token overhead is measurable (not zero)."""
        task = TaskQuery(raw_task="fix bug", task_id="t1", mode="plan", repo_id=1)

        conn = get_connection("curated", repo_path=assembly_source_files)
        repo_id = upsert_repo(conn, str(assembly_source_files), None)
        fid1 = upsert_file(conn, repo_id, "src/auth.py", "python", "h1", 500)
        upsert_file_metadata(
            conn, fid1,
            purpose="A very detailed purpose description for the authentication module",
            domain="enterprise-security",
        )
        conn.commit()
        kb = KnowledgeBase(conn)

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=repo_id, repo_path=str(assembly_source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=fid1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
        ]
        ctx.included_file_ids = {fid1}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=fid1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
        ]

        pkg = assemble_context(ctx, budget, assembly_source_files, kb=kb)

        # The metadata summary should contribute tokens
        meta_summary = pkg.files[0].metadata_summary
        meta_tokens = estimate_tokens(meta_summary)
        assert meta_tokens > 0

        conn.close()


# ---------------------------------------------------------------------------
# Assembly: metadata_summary populated from KB
# ---------------------------------------------------------------------------

class TestAssemblyMetadataSummary:
    def test_assembly_populates_metadata_summary(self, assembly_source_files):
        """assemble_context populates FileContent.metadata_summary from KB enrichment."""
        task = TaskQuery(raw_task="fix bug", task_id="t1", mode="plan", repo_id=1)

        conn = get_connection("curated", repo_path=assembly_source_files)
        repo_id = upsert_repo(conn, str(assembly_source_files), None)
        fid1 = upsert_file(conn, repo_id, "src/auth.py", "python", "h1", 500)
        fid2 = upsert_file(conn, repo_id, "src/utils.py", "python", "h2", 300)

        upsert_file_metadata(conn, fid1, purpose="Auth module", domain="security")
        # fid2 has no metadata
        conn.commit()
        kb = KnowledgeBase(conn)

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=repo_id, repo_path=str(assembly_source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=fid1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
            ScopedFile(file_id=fid2, path="src/utils.py", language="python",
                       tier=2, relevance="relevant"),
        ]
        ctx.included_file_ids = {fid1, fid2}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=fid1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
            ClassifiedSymbol(symbol_id=2, file_id=fid2, name="parse",
                             kind="function", start_line=1, end_line=2,
                             detail_level="supporting"),
        ]

        pkg = assemble_context(ctx, budget, assembly_source_files, kb=kb)

        # Find each file in the output
        auth_fc = next(fc for fc in pkg.files if fc.path == "src/auth.py")
        utils_fc = next(fc for fc in pkg.files if fc.path == "src/utils.py")

        # auth.py should have metadata_summary
        assert auth_fc.metadata_summary != ""
        assert "Purpose: Auth module" in auth_fc.metadata_summary
        assert "Domain: security" in auth_fc.metadata_summary

        # utils.py should have empty metadata_summary
        assert utils_fc.metadata_summary == ""

        conn.close()

    def test_assembly_metadata_only_purpose_no_domain(self, assembly_source_files):
        """Metadata summary with purpose but no domain includes only purpose."""
        task = TaskQuery(raw_task="fix bug", task_id="t1", mode="plan", repo_id=1)

        conn = get_connection("curated", repo_path=assembly_source_files)
        repo_id = upsert_repo(conn, str(assembly_source_files), None)
        fid1 = upsert_file(conn, repo_id, "src/auth.py", "python", "h1", 500)

        upsert_file_metadata(conn, fid1, purpose="Auth module")
        # No domain set
        conn.commit()
        kb = KnowledgeBase(conn)

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=repo_id, repo_path=str(assembly_source_files))
        ctx.scoped_files = [
            ScopedFile(file_id=fid1, path="src/auth.py", language="python",
                       tier=1, relevance="relevant"),
        ]
        ctx.included_file_ids = {fid1}
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=1, file_id=fid1, name="AuthManager",
                             kind="class", start_line=1, end_line=8,
                             detail_level="primary"),
        ]

        pkg = assemble_context(ctx, budget, assembly_source_files, kb=kb)

        auth_fc = next(fc for fc in pkg.files if fc.path == "src/auth.py")
        assert "Purpose: Auth module" in auth_fc.metadata_summary
        # No " | " separator since there is only purpose
        assert "Domain:" not in auth_fc.metadata_summary

        conn.close()

    def test_assembly_no_kb_no_metadata(self, assembly_source_files):
        """When no KB is provided, metadata_summary is empty for all files."""
        task = TaskQuery(raw_task="fix bug", task_id="t1", mode="plan", repo_id=1)

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        ctx = StageContext(task=task, repo_id=1, repo_path=str(assembly_source_files))
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

        pkg = assemble_context(ctx, budget, assembly_source_files, kb=None)

        assert len(pkg.files) == 1
        assert pkg.files[0].metadata_summary == ""
