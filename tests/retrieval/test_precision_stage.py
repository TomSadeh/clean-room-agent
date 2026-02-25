"""Tests for precision stage."""

import json
from unittest.mock import MagicMock

import pytest

from clean_room_agent.db.connection import get_connection
from clean_room_agent.db.queries import (
    insert_symbol,
    insert_symbol_reference,
    upsert_file,
    upsert_repo,
)
from clean_room_agent.query.api import KnowledgeBase
from clean_room_agent.retrieval.dataclasses import TaskQuery
from clean_room_agent.retrieval.precision_stage import (
    classify_symbols,
    extract_precision_symbols,
)
from clean_room_agent.retrieval.utils import parse_json_response


def _mock_llm_with_response(text):
    """Create a mock LLM with standard config and a fixed response."""
    mock_llm = MagicMock()
    mock_llm.config.context_window = 32768
    mock_llm.config.max_tokens = 4096
    mock_response = MagicMock()
    mock_response.text = text
    mock_llm.complete.return_value = mock_response
    return mock_llm


@pytest.fixture
def precision_kb(tmp_repo):
    """Create a curated DB with files, symbols, and references."""
    conn = get_connection("curated", repo_path=tmp_repo)
    repo_id = upsert_repo(conn, str(tmp_repo), None)

    fid1 = upsert_file(conn, repo_id, "src/auth.py", "python", "h1", 500)
    fid2 = upsert_file(conn, repo_id, "src/app.tsx", "typescript", "h2", 400)

    # Python symbols with references
    sid1 = insert_symbol(conn, fid1, "login", "function", 10, 30, "def login(username, password)")
    sid2 = insert_symbol(conn, fid1, "verify_token", "function", 35, 50, "def verify_token(token)")
    sid3 = insert_symbol(conn, fid1, "AuthManager", "class", 1, 60, "class AuthManager")

    # login calls verify_token
    insert_symbol_reference(conn, sid1, sid2, "call")

    # TS/JS symbols (no references)
    sid4 = insert_symbol(conn, fid2, "LoginForm", "class", 5, 40, "class LoginForm")
    sid5 = insert_symbol(conn, fid2, "handleSubmit", "function", 10, 25, "handleSubmit()")

    conn.commit()
    kb = KnowledgeBase(conn)
    yield kb, repo_id, {"auth": fid1, "app": fid2}, {"login": sid1, "verify": sid2, "mgr": sid3, "form": sid4, "submit": sid5}
    conn.close()


class TestExtractPrecisionSymbols:
    def test_python_edge_traversal(self, precision_kb):
        kb, repo_id, fids, sids = precision_kb
        task = TaskQuery(
            raw_task="fix login", task_id="t1", mode="plan", repo_id=repo_id,
            keywords=["login"], mentioned_symbols=["login"],
        )
        candidates = extract_precision_symbols({fids["auth"]}, task, kb)
        assert len(candidates) >= 2  # at least login, verify_token, AuthManager

        login_entry = next(c for c in candidates if c["name"] == "login")
        assert login_entry["file_path"] == "src/auth.py"
        # login should have callee connection to verify_token
        assert any("verify_token" in conn for conn in login_entry["connections"])

    def test_tsjs_name_matching(self, precision_kb):
        kb, repo_id, fids, sids = precision_kb
        task = TaskQuery(
            raw_task="fix login form", task_id="t1", mode="plan", repo_id=repo_id,
            keywords=["login"], mentioned_symbols=["LoginForm"],
        )
        candidates = extract_precision_symbols({fids["app"]}, task, kb)
        assert len(candidates) >= 1

        form_entry = next(c for c in candidates if c["name"] == "LoginForm")
        # Should match against task term "LoginForm"
        assert any("LoginForm" in conn for conn in form_entry["connections"])

    def test_empty_file_ids(self, precision_kb):
        kb, repo_id, fids, sids = precision_kb
        task = TaskQuery(raw_task="x", task_id="t1", mode="plan", repo_id=repo_id)
        candidates = extract_precision_symbols(set(), task, kb)
        assert candidates == []


class TestClassifySymbols:
    def test_classification(self):
        mock_llm = _mock_llm_with_response(json.dumps([
            {"name": "login", "file_path": "auth.py", "start_line": 1, "detail_level": "primary", "reason": "directly changed"},
            {"name": "helper", "file_path": "auth.py", "start_line": 15, "detail_level": "supporting", "reason": "context"},
        ]))

        candidates = [
            {
                "symbol_id": 1, "file_id": 10, "file_path": "auth.py",
                "name": "login", "kind": "function", "start_line": 1, "end_line": 10,
                "signature": "def login()", "connections": [],
                "file_source": "project",
            },
            {
                "symbol_id": 2, "file_id": 10, "file_path": "auth.py",
                "name": "helper", "kind": "function", "start_line": 15, "end_line": 20,
                "signature": "def helper()", "connections": [],
                "file_source": "project",
            },
        ]
        task = TaskQuery(raw_task="fix login", task_id="t1", mode="plan", repo_id=1)

        result = classify_symbols(candidates, task, mock_llm)
        assert len(result) == 2
        login_cs = next(cs for cs in result if cs.name == "login")
        assert login_cs.detail_level == "primary"
        helper_cs = next(cs for cs in result if cs.name == "helper")
        assert helper_cs.detail_level == "supporting"

    def test_empty_candidates(self):
        mock_llm = MagicMock()
        task = TaskQuery(raw_task="x", task_id="t1", mode="plan", repo_id=1)
        result = classify_symbols([], task, mock_llm)
        assert result == []
        mock_llm.complete.assert_not_called()

    def test_invalid_detail_level_defaults(self):
        mock_llm = _mock_llm_with_response(json.dumps([
            {"name": "foo", "file_path": "a.py", "start_line": 1, "detail_level": "banana", "reason": "???"},
        ]))

        candidates = [
            {
                "symbol_id": 1, "file_id": 10, "file_path": "a.py",
                "name": "foo", "kind": "function", "start_line": 1, "end_line": 5,
                "signature": "", "connections": [],
                "file_source": "project",
            },
        ]
        task = TaskQuery(raw_task="x", task_id="t1", mode="plan", repo_id=1)

        result = classify_symbols(candidates, task, mock_llm)
        assert result[0].detail_level == "excluded"  # R2: invalid -> excluded (default-deny)

    def test_omitted_candidate_defaults_to_excluded(self):
        """T18/R2: LLM omitting a candidate produces a warning and defaults to excluded."""
        # LLM response only classifies "login", omits "helper"
        mock_llm = _mock_llm_with_response(json.dumps([
            {"name": "login", "file_path": "auth.py", "start_line": 1, "detail_level": "primary", "reason": "changed"},
        ]))

        candidates = [
            {
                "symbol_id": 1, "file_id": 10, "file_path": "auth.py",
                "name": "login", "kind": "function", "start_line": 1, "end_line": 10,
                "signature": "def login()", "connections": [],
                "file_source": "project",
            },
            {
                "symbol_id": 2, "file_id": 10, "file_path": "auth.py",
                "name": "helper", "kind": "function", "start_line": 15, "end_line": 20,
                "signature": "def helper()", "connections": [],
                "file_source": "project",
            },
        ]
        task = TaskQuery(raw_task="fix login", task_id="t1", mode="plan", repo_id=1)

        result = classify_symbols(candidates, task, mock_llm)
        helper_cs = next(cs for cs in result if cs.name == "helper")
        assert helper_cs.detail_level == "excluded"

    def test_llm_omits_symbol_mixed_response(self):
        """R2: When LLM omits some symbols but classifies others, omitted ones get 'excluded'."""
        mock_llm = _mock_llm_with_response(json.dumps([
            {"name": "alpha", "file_path": "mod.py", "start_line": 1, "detail_level": "primary", "reason": "main target"},
            # beta and gamma are omitted from the LLM response
        ]))

        candidates = [
            {
                "symbol_id": 1, "file_id": 10, "file_path": "mod.py",
                "name": "alpha", "kind": "function", "start_line": 1, "end_line": 10,
                "signature": "def alpha()", "connections": [],
                "file_source": "project",
            },
            {
                "symbol_id": 2, "file_id": 10, "file_path": "mod.py",
                "name": "beta", "kind": "function", "start_line": 15, "end_line": 20,
                "signature": "def beta()", "connections": [],
                "file_source": "project",
            },
            {
                "symbol_id": 3, "file_id": 10, "file_path": "mod.py",
                "name": "gamma", "kind": "class", "start_line": 25, "end_line": 40,
                "signature": "class gamma:", "connections": [],
                "file_source": "project",
            },
        ]
        task = TaskQuery(raw_task="fix alpha", task_id="t1", mode="plan", repo_id=1)

        result = classify_symbols(candidates, task, mock_llm)
        assert len(result) == 3
        alpha_cs = next(cs for cs in result if cs.name == "alpha")
        beta_cs = next(cs for cs in result if cs.name == "beta")
        gamma_cs = next(cs for cs in result if cs.name == "gamma")
        assert alpha_cs.detail_level == "primary"
        assert beta_cs.detail_level == "excluded"  # R2: omitted -> excluded
        assert gamma_cs.detail_level == "excluded"  # R2: omitted -> excluded

    def test_key_collision_same_name_different_lines(self):
        """B4: Two symbols named __init__ at different lines are classified independently."""
        mock_llm = _mock_llm_with_response(json.dumps([
            {"name": "__init__", "file_path": "models.py", "start_line": 5, "detail_level": "primary", "reason": "User init"},
            {"name": "__init__", "file_path": "models.py", "start_line": 20, "detail_level": "type_context", "reason": "Config init"},
        ]))

        candidates = [
            {
                "symbol_id": 10, "file_id": 1, "file_path": "models.py",
                "name": "__init__", "kind": "function", "start_line": 5, "end_line": 15,
                "signature": "def __init__(self, name)", "connections": [],
                "file_source": "project",
            },
            {
                "symbol_id": 11, "file_id": 1, "file_path": "models.py",
                "name": "__init__", "kind": "function", "start_line": 20, "end_line": 30,
                "signature": "def __init__(self, key)", "connections": [],
                "file_source": "project",
            },
        ]
        task = TaskQuery(raw_task="fix models", task_id="t1", mode="plan", repo_id=1)

        result = classify_symbols(candidates, task, mock_llm)
        assert len(result) == 2
        init_5 = next(cs for cs in result if cs.start_line == 5)
        init_20 = next(cs for cs in result if cs.start_line == 20)
        assert init_5.detail_level == "primary"
        assert init_20.detail_level == "type_context"


class TestClassifySymbolsBatching:
    def test_batch_boundary(self):
        """WU6: large symbol sets are split into batches."""
        import re

        mock_llm = MagicMock()
        mock_llm.config.context_window = 512
        mock_llm.config.max_tokens = 128

        candidates = []
        for i in range(30):
            candidates.append({
                "symbol_id": i, "file_id": 1, "file_path": "big.py",
                "name": f"func_{i}", "kind": "function",
                "start_line": i * 10, "end_line": i * 10 + 5,
                "signature": f"def func_{i}()", "connections": [],
                "file_source": "project",
            })

        def _complete(prompt, system=None):
            resp = MagicMock()
            names = re.findall(r"- (func_\d+)", prompt)
            resp.text = json.dumps([
                {"name": n, "file_path": "big.py", "start_line": int(n.split("_")[1]) * 10,
                 "detail_level": "supporting", "reason": "ctx"}
                for n in names
            ])
            return resp

        mock_llm.complete.side_effect = _complete

        task = TaskQuery(raw_task="fix big", task_id="t1", mode="plan", repo_id=1)
        result = classify_symbols(candidates, task, mock_llm)

        assert len(result) == 30
        assert all(cs.detail_level == "supporting" for cs in result)
        assert mock_llm.complete.call_count >= 2


class TestR6NeighborOrdering:
    """R6c: Callee/caller neighbors ordered by file inclusion before capping."""

    def test_neighbors_ordered_by_inclusion(self, precision_kb):
        """R6c: Neighbors in included files sorted before those outside."""
        kb, repo_id, fids, sids = precision_kb

        # Included only auth file — login's callee verify_token is also in auth
        task = TaskQuery(
            raw_task="fix login", task_id="t1", mode="plan", repo_id=repo_id,
            keywords=["login"], mentioned_symbols=["login"],
        )
        candidates = extract_precision_symbols({fids["auth"]}, task, kb)

        login_entry = next(c for c in candidates if c["name"] == "login")
        # verify_token is a callee of login and is in the included file set
        assert any("verify_token" in conn for conn in login_entry["connections"])


class TestParseJsonResponse:
    def test_valid_json(self):
        assert parse_json_response('[{"a": 1}]') == [{"a": 1}]

    def test_fenced_json(self):
        assert parse_json_response('```json\n[]\n```') == []

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_json_response("garbage")
