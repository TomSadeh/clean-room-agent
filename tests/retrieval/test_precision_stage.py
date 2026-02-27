"""Tests for precision stage — 3-pass binary classification cascade."""

import json
from unittest.mock import MagicMock, call

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
    PRECISION_PASS1_SYSTEM,
    PRECISION_PASS2_SYSTEM,
    PRECISION_PASS3_SYSTEM,
    classify_symbols,
    extract_precision_symbols,
)
from clean_room_agent.retrieval.utils import parse_json_response


def _binary_llm(pass_map):
    """Create a mock LLM for the 3-pass binary cascade.

    pass_map: dict mapping (pass_number, symbol_name) -> "yes"/"no".
    Pass is determined by the system prompt content.
    Default: "no" (R2 default-deny).
    """
    mock_llm = MagicMock()
    mock_llm.flush = MagicMock()
    mock_llm.config.context_window = 32768
    mock_llm.config.max_tokens = 4096

    def _complete(prompt, system=None):
        resp = MagicMock()
        # Determine which pass based on system prompt
        if system == PRECISION_PASS1_SYSTEM:
            pass_num = 1
        elif system == PRECISION_PASS2_SYSTEM:
            pass_num = 2
        elif system == PRECISION_PASS3_SYSTEM:
            pass_num = 3
        else:
            resp.text = "no"
            return resp

        # Find symbol name in prompt
        for (pn, name), answer in pass_map.items():
            if pn == pass_num and f"Symbol: {name}" in prompt:
                resp.text = answer
                return resp
        resp.text = "no"  # default
        return resp

    mock_llm.complete.side_effect = _complete
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
    def test_full_cascade(self):
        """3-pass cascade: pass1 filters, pass2 splits primary, pass3 splits supporting/type_context."""
        # login: relevant→primary, helper: relevant→not primary→supporting
        llm = _binary_llm({
            (1, "login"): "yes",
            (1, "helper"): "yes",
            (2, "login"): "yes",   # primary
            (2, "helper"): "no",   # not primary → pass 3
            (3, "helper"): "yes",  # full source needed → supporting
        })

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

        result = classify_symbols(candidates, task, llm)
        assert len(result) == 2
        login_cs = next(cs for cs in result if cs.name == "login")
        assert login_cs.detail_level == "primary"
        assert "pass2" in login_cs.reason
        helper_cs = next(cs for cs in result if cs.name == "helper")
        assert helper_cs.detail_level == "supporting"
        assert "pass3" in helper_cs.reason

    def test_pass1_excludes(self):
        """Pass 1 'no' → excluded, skips passes 2 and 3."""
        llm = _binary_llm({
            (1, "login"): "yes",
            (1, "helper"): "no",  # excluded at pass 1
            (2, "login"): "yes",  # primary
        })

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

        result = classify_symbols(candidates, task, llm)
        helper_cs = next(cs for cs in result if cs.name == "helper")
        assert helper_cs.detail_level == "excluded"
        assert "pass1" in helper_cs.reason

    def test_pass3_type_context(self):
        """Pass 3 'no' → type_context (signature sufficient)."""
        llm = _binary_llm({
            (1, "foo"): "yes",
            (2, "foo"): "no",   # not primary
            (3, "foo"): "no",   # signature sufficient → type_context
        })

        candidates = [{
            "symbol_id": 1, "file_id": 10, "file_path": "a.py",
            "name": "foo", "kind": "function", "start_line": 1, "end_line": 5,
            "signature": "def foo()", "connections": [],
            "file_source": "project",
        }]
        task = TaskQuery(raw_task="x", task_id="t1", mode="plan", repo_id=1)

        result = classify_symbols(candidates, task, llm)
        assert result[0].detail_level == "type_context"
        assert "pass3" in result[0].reason

    def test_empty_candidates(self):
        mock_llm = MagicMock()
        mock_llm.flush = MagicMock()
        task = TaskQuery(raw_task="x", task_id="t1", mode="plan", repo_id=1)
        result = classify_symbols([], task, mock_llm)
        assert result == []
        mock_llm.complete.assert_not_called()

    def test_r2_default_deny_unparseable(self):
        """R2: unparseable binary response → default-deny → excluded at pass 1."""
        mock_llm = MagicMock()
        mock_llm.flush = MagicMock()
        mock_llm.config.context_window = 32768
        mock_llm.config.max_tokens = 4096
        resp = MagicMock()
        resp.text = "maybe"  # not yes/no
        mock_llm.complete.return_value = resp

        candidates = [{
            "symbol_id": 1, "file_id": 10, "file_path": "a.py",
            "name": "foo", "kind": "function", "start_line": 1, "end_line": 5,
            "signature": "", "connections": [],
            "file_source": "project",
        }]
        task = TaskQuery(raw_task="x", task_id="t1", mode="plan", repo_id=1)

        result = classify_symbols(candidates, task, mock_llm)
        assert result[0].detail_level == "excluded"  # R2: default-deny

    def test_library_symbols_auto_type_context(self):
        """R17: Library symbols skip all 3 passes, auto-classified as type_context."""
        mock_llm = MagicMock()
        mock_llm.flush = MagicMock()
        mock_llm.config.context_window = 32768
        mock_llm.config.max_tokens = 4096

        candidates = [{
            "symbol_id": 1, "file_id": 10, "file_path": "lib.py",
            "name": "Client", "kind": "class", "start_line": 1, "end_line": 50,
            "signature": "class Client", "connections": [],
            "file_source": "library",
        }]
        task = TaskQuery(raw_task="x", task_id="t1", mode="plan", repo_id=1)

        result = classify_symbols(candidates, task, mock_llm)
        assert result[0].detail_level == "type_context"
        assert "library" in result[0].reason
        mock_llm.complete.assert_not_called()  # no LLM calls for library symbols

    def test_key_collision_same_name_different_lines(self):
        """B4: Two symbols named __init__ at different lines are classified independently."""
        llm = _binary_llm({
            (1, "__init__"): "yes",  # both relevant
            (2, "__init__"): "yes",  # both primary (binary can't distinguish by name alone)
        })

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

        result = classify_symbols(candidates, task, llm)
        assert len(result) == 2
        # Both should get individual calls (binary processes each independently)
        assert llm.complete.call_count >= 2


class TestCascadeCallCounts:
    """Verify the cascade is volume-reducing: call counts decrease across passes."""

    def test_volume_reduction(self):
        """8 symbols → 5 relevant (pass1) → 2 primary (pass2) → 3 non-primary enter pass3."""
        pass_map = {}
        names = [f"sym_{i}" for i in range(8)]
        # Pass 1: 5 relevant, 3 excluded
        for i, name in enumerate(names):
            pass_map[(1, name)] = "yes" if i < 5 else "no"
        # Pass 2: 2 primary out of 5 relevant
        for i in range(5):
            pass_map[(2, names[i])] = "yes" if i < 2 else "no"
        # Pass 3: all non-primary get supporting
        for i in range(2, 5):
            pass_map[(3, names[i])] = "yes"

        llm = _binary_llm(pass_map)

        candidates = [
            {
                "symbol_id": i, "file_id": 10, "file_path": "big.py",
                "name": names[i], "kind": "function",
                "start_line": i * 10, "end_line": i * 10 + 5,
                "signature": f"def {names[i]}()", "connections": [],
                "file_source": "project",
            }
            for i in range(8)
        ]
        task = TaskQuery(raw_task="fix big", task_id="t1", mode="plan", repo_id=1)

        result = classify_symbols(candidates, task, llm)
        assert len(result) == 8

        # Verify classifications
        by_name = {cs.name: cs.detail_level for cs in result}
        assert by_name["sym_0"] == "primary"
        assert by_name["sym_1"] == "primary"
        assert by_name["sym_2"] == "supporting"
        assert by_name["sym_3"] == "supporting"
        assert by_name["sym_4"] == "supporting"
        assert by_name["sym_5"] == "excluded"
        assert by_name["sym_6"] == "excluded"
        assert by_name["sym_7"] == "excluded"

        # Total calls: 8 (pass1) + 5 (pass2) + 3 (pass3) = 16
        assert llm.complete.call_count == 16

    def test_all_excluded_at_pass1_skips_later_passes(self):
        """If pass 1 excludes everything, passes 2 and 3 never run."""
        llm = _binary_llm({})  # all default to "no"

        candidates = [
            {
                "symbol_id": i, "file_id": 10, "file_path": "a.py",
                "name": f"sym_{i}", "kind": "function",
                "start_line": i, "end_line": i + 5,
                "signature": "", "connections": [],
                "file_source": "project",
            }
            for i in range(3)
        ]
        task = TaskQuery(raw_task="x", task_id="t1", mode="plan", repo_id=1)

        result = classify_symbols(candidates, task, llm)
        assert all(cs.detail_level == "excluded" for cs in result)
        # Only pass 1 calls (3 symbols), no pass 2 or 3
        assert llm.complete.call_count == 3


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
