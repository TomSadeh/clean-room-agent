"""Tests for similarity stage."""

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
from clean_room_agent.retrieval.dataclasses import ClassifiedSymbol, TaskQuery
from clean_room_agent.retrieval.similarity_stage import (
    MAX_CANDIDATE_PAIRS,
    MAX_GROUP_SIZE,
    MIN_COMPOSITE_SCORE,
    _longest_common_subsequence_ratio,
    assign_groups,
    find_similar_pairs,
    judge_similarity,
)
from clean_room_agent.retrieval.stage import StageContext


def _mock_llm_with_response(text):
    """Create a mock LLM with standard config and a fixed response."""
    mock_llm = MagicMock()
    mock_llm.flush = MagicMock()  # F14: explicit _require_logged_client contract
    mock_llm.config.context_window = 32768
    mock_llm.config.max_tokens = 4096
    mock_response = MagicMock()
    mock_response.text = text
    mock_llm.complete.return_value = mock_response
    return mock_llm


def _make_symbol(symbol_id, file_id, name, kind="function", start=1, end=10, detail="primary"):
    return ClassifiedSymbol(
        symbol_id=symbol_id, file_id=file_id, name=name,
        kind=kind, start_line=start, end_line=end,
        detail_level=detail,
    )


# ---------- LCS ratio ----------

class TestLCSRatio:
    def test_identical_strings(self):
        assert _longest_common_subsequence_ratio("abc", "abc") == 1.0

    def test_empty_strings(self):
        assert _longest_common_subsequence_ratio("", "abc") == 0.0
        assert _longest_common_subsequence_ratio("abc", "") == 0.0

    def test_partial_overlap(self):
        ratio = _longest_common_subsequence_ratio("process_data", "process_file")
        assert 0.0 < ratio < 1.0

    def test_no_overlap(self):
        ratio = _longest_common_subsequence_ratio("abc", "xyz")
        assert ratio == 0.0


# ---------- find_similar_pairs ----------

@pytest.fixture
def sim_kb(tmp_path):
    """KB with similar functions for pairing tests."""
    conn = get_connection("curated", repo_path=tmp_path)
    repo_id = upsert_repo(conn, str(tmp_path), None)

    fid = upsert_file(conn, repo_id, "src/handlers.py", "python", "h1", 500)

    # Two similar functions with same callees
    sid1 = insert_symbol(conn, fid, "process_users", "function", 1, 20, "def process_users()")
    sid2 = insert_symbol(conn, fid, "process_items", "function", 25, 44, "def process_items()")
    # A helper they both call
    helper = insert_symbol(conn, fid, "validate", "function", 50, 55, "def validate()")
    insert_symbol_reference(conn, sid1, helper, "call")
    insert_symbol_reference(conn, sid2, helper, "call")

    # A very different function
    sid3 = insert_symbol(conn, fid, "XManager", "class", 60, 200, "class XManager")

    conn.commit()
    kb = KnowledgeBase(conn)
    yield kb, repo_id, fid, {"users": sid1, "items": sid2, "validate": helper, "manager": sid3}
    conn.close()


class TestFindSimilarPairs:
    def test_similar_functions_found(self, sim_kb):
        kb, repo_id, fid, sids = sim_kb
        symbols = [
            _make_symbol(sids["users"], fid, "process_users", start=1, end=20),
            _make_symbol(sids["items"], fid, "process_items", start=25, end=44),
        ]
        pairs = find_similar_pairs(symbols, kb, min_score=0.0)
        assert len(pairs) == 1
        assert pairs[0]["sym_a"].symbol_id == sids["users"]
        assert pairs[0]["sym_b"].symbol_id == sids["items"]

    def test_classes_filtered_out(self, sim_kb):
        kb, repo_id, fid, sids = sim_kb
        symbols = [
            _make_symbol(sids["manager"], fid, "XManager", kind="class", start=60, end=200),
            _make_symbol(sids["users"], fid, "process_users", start=1, end=20),
        ]
        pairs = find_similar_pairs(symbols, kb, min_score=0.0)
        # Only functions/methods, class is filtered out → no pairs possible with one function
        assert len(pairs) == 0

    def test_less_than_2_candidates_returns_empty(self, sim_kb):
        kb, repo_id, fid, sids = sim_kb
        symbols = [_make_symbol(sids["users"], fid, "process_users")]
        assert find_similar_pairs(symbols, kb) == []

    def test_threshold_filtering(self, sim_kb):
        kb, repo_id, fid, sids = sim_kb
        symbols = [
            _make_symbol(sids["users"], fid, "process_users", start=1, end=20),
            _make_symbol(sids["items"], fid, "process_items", start=25, end=44),
        ]
        # Very high threshold should filter everything
        pairs = find_similar_pairs(symbols, kb, min_score=0.99)
        assert len(pairs) == 0

    def test_max_pairs_cap(self, sim_kb):
        kb, repo_id, fid, sids = sim_kb
        symbols = [
            _make_symbol(sids["users"], fid, "process_users", start=1, end=20),
            _make_symbol(sids["items"], fid, "process_items", start=25, end=44),
            _make_symbol(sids["validate"], fid, "validate", start=50, end=55),
        ]
        pairs = find_similar_pairs(symbols, kb, max_pairs=1, min_score=0.0)
        assert len(pairs) <= 1

    def test_scoring_includes_signals(self, sim_kb):
        kb, repo_id, fid, sids = sim_kb
        symbols = [
            _make_symbol(sids["users"], fid, "process_users", start=1, end=20),
            _make_symbol(sids["items"], fid, "process_items", start=25, end=44),
        ]
        pairs = find_similar_pairs(symbols, kb, min_score=0.0)
        assert len(pairs) == 1
        signals = pairs[0]["signals"]
        assert "line_ratio" in signals
        assert "callee_jaccard" in signals
        assert "name_lcs" in signals
        assert "same_parent" in signals

    def test_pairs_sorted_by_score_desc(self, sim_kb):
        kb, repo_id, fid, sids = sim_kb
        symbols = [
            _make_symbol(sids["users"], fid, "process_users", start=1, end=20),
            _make_symbol(sids["items"], fid, "process_items", start=25, end=44),
            _make_symbol(sids["validate"], fid, "validate", start=50, end=55),
        ]
        pairs = find_similar_pairs(symbols, kb, min_score=0.0)
        scores = [p["score"] for p in pairs]
        assert scores == sorted(scores, reverse=True)


# ---------- judge_similarity ----------

class TestJudgeSimilarity:
    def test_confirmed_pairs_kept(self):
        sym_a = _make_symbol(1, 10, "process_users")
        sym_b = _make_symbol(2, 10, "process_items")
        pairs = [{
            "pair_id": 0, "sym_a": sym_a, "sym_b": sym_b, "score": 0.5,
            "signals": {"line_ratio": 0.9, "callee_jaccard": 0.5, "name_lcs": 0.7, "same_parent": False},
        }]
        task = TaskQuery(raw_task="dedup handlers", task_id="t1", mode="plan", repo_id=1)
        llm = _mock_llm_with_response(json.dumps([
            {"pair_id": 0, "keep": True, "group_label": "process_handlers", "reason": "similar pattern"},
        ]))
        confirmed = judge_similarity(pairs, task, llm)
        assert len(confirmed) == 1
        assert confirmed[0]["group_label"] == "process_handlers"

    def test_omitted_pairs_denied(self):
        sym_a = _make_symbol(1, 10, "process_users")
        sym_b = _make_symbol(2, 10, "process_items")
        pairs = [{
            "pair_id": 0, "sym_a": sym_a, "sym_b": sym_b, "score": 0.5,
            "signals": {"line_ratio": 0.9, "callee_jaccard": 0.5, "name_lcs": 0.7, "same_parent": False},
        }]
        task = TaskQuery(raw_task="dedup", task_id="t1", mode="plan", repo_id=1)
        # LLM returns empty array (omits the pair)
        llm = _mock_llm_with_response("[]")
        confirmed = judge_similarity(pairs, task, llm)
        assert len(confirmed) == 0

    def test_keep_false_denied(self):
        sym_a = _make_symbol(1, 10, "a")
        sym_b = _make_symbol(2, 10, "b")
        pairs = [{
            "pair_id": 0, "sym_a": sym_a, "sym_b": sym_b, "score": 0.5,
            "signals": {"line_ratio": 0.5, "callee_jaccard": 0.0, "name_lcs": 0.1, "same_parent": False},
        }]
        task = TaskQuery(raw_task="dedup", task_id="t1", mode="plan", repo_id=1)
        llm = _mock_llm_with_response(json.dumps([
            {"pair_id": 0, "keep": False, "group_label": "", "reason": "not similar"},
        ]))
        confirmed = judge_similarity(pairs, task, llm)
        assert len(confirmed) == 0

    def test_invalid_response_raises(self):
        sym_a = _make_symbol(1, 10, "a")
        sym_b = _make_symbol(2, 10, "b")
        pairs = [{
            "pair_id": 0, "sym_a": sym_a, "sym_b": sym_b, "score": 0.5,
            "signals": {"line_ratio": 0.5, "callee_jaccard": 0.0, "name_lcs": 0.1, "same_parent": False},
        }]
        task = TaskQuery(raw_task="dedup", task_id="t1", mode="plan", repo_id=1)
        llm = _mock_llm_with_response("not json at all")
        with pytest.raises(ValueError, match="unparseable JSON"):
            judge_similarity(pairs, task, llm)

    def test_empty_pairs_returns_empty(self):
        task = TaskQuery(raw_task="dedup", task_id="t1", mode="plan", repo_id=1)
        llm = _mock_llm_with_response("[]")
        assert judge_similarity([], task, llm) == []

    def test_batching(self):
        """When many pairs exceed batch size, multiple LLM calls are made."""
        sym_a = _make_symbol(1, 10, "a")
        sym_b = _make_symbol(2, 10, "b")
        pairs = [
            {
                "pair_id": i, "sym_a": sym_a, "sym_b": sym_b, "score": 0.5,
                "signals": {"line_ratio": 0.5, "callee_jaccard": 0.0, "name_lcs": 0.1, "same_parent": False},
            }
            for i in range(5)
        ]
        task = TaskQuery(raw_task="dedup", task_id="t1", mode="plan", repo_id=1)
        # LLM with very small context to force batching
        llm = MagicMock()
        llm.flush = MagicMock()  # F14: explicit _require_logged_client contract
        llm.config.context_window = 500
        llm.config.max_tokens = 200
        # Return empty but valid response each time
        response = MagicMock()
        response.text = "[]"
        llm.complete.return_value = response

        judge_similarity(pairs, task, llm)
        # Should have called complete at least once
        assert llm.complete.call_count >= 1


# ---------- assign_groups ----------

class TestAssignGroups:
    def test_empty_input(self):
        assert assign_groups([]) == {}

    def test_single_pair_forms_group(self):
        sym_a = _make_symbol(5, 10, "a")
        sym_b = _make_symbol(3, 10, "b")
        confirmed = [{"pair_id": 0, "sym_a": sym_a, "sym_b": sym_b, "group_label": "g", "reason": "r"}]
        groups = assign_groups(confirmed)
        assert groups[5] == "sim_group_3"
        assert groups[3] == "sim_group_3"

    def test_transitive_grouping(self):
        sym_a = _make_symbol(1, 10, "a")
        sym_b = _make_symbol(2, 10, "b")
        sym_c = _make_symbol(3, 10, "c")
        confirmed = [
            {"pair_id": 0, "sym_a": sym_a, "sym_b": sym_b, "group_label": "g", "reason": "r"},
            {"pair_id": 1, "sym_a": sym_b, "sym_b": sym_c, "group_label": "g", "reason": "r"},
        ]
        groups = assign_groups(confirmed)
        # All three should be in the same group
        assert groups[1] == groups[2] == groups[3]
        assert groups[1] == "sim_group_1"

    def test_disjoint_groups(self):
        sym_a = _make_symbol(1, 10, "a")
        sym_b = _make_symbol(2, 10, "b")
        sym_c = _make_symbol(10, 20, "c")
        sym_d = _make_symbol(11, 20, "d")
        confirmed = [
            {"pair_id": 0, "sym_a": sym_a, "sym_b": sym_b, "group_label": "g1", "reason": "r"},
            {"pair_id": 1, "sym_a": sym_c, "sym_b": sym_d, "group_label": "g2", "reason": "r"},
        ]
        groups = assign_groups(confirmed)
        assert groups[1] == groups[2]
        assert groups[10] == groups[11]
        assert groups[1] != groups[10]

    def test_max_group_size_cap(self):
        # Build a chain of 5 pairs creating a group of 6 symbols
        syms = [_make_symbol(i, 10, f"s{i}") for i in range(6)]
        confirmed = [
            {"pair_id": i, "sym_a": syms[i], "sym_b": syms[i + 1], "group_label": "g", "reason": "r"}
            for i in range(5)
        ]
        groups = assign_groups(confirmed, max_group_size=3)
        # Only 3 symbols should be in the group
        assigned = [sid for sid in range(6) if sid in groups]
        assert len(assigned) == 3


# ---------- Stage integration ----------

class TestSimilarityStageIntegration:
    def test_no_symbols_skips(self):
        """Stage returns context unchanged when no classified symbols."""
        from clean_room_agent.retrieval.similarity_stage import SimilarityStage

        task = TaskQuery(raw_task="test", task_id="t1", mode="plan", repo_id=1)
        context = StageContext(task=task, repo_id=1, repo_path="/tmp")
        llm = MagicMock()
        llm.flush = MagicMock()  # F14: explicit _require_logged_client contract
        kb = MagicMock()

        stage = SimilarityStage()
        result = stage.run(context, kb, task, llm)
        assert result is context
        # LLM should not have been called
        llm.complete.assert_not_called()

    def test_stage_registered(self):
        from clean_room_agent.retrieval.stage import get_stage_descriptions
        descs = get_stage_descriptions()
        assert "similarity" in descs


# ---------- Serialization round-trip ----------

class TestGroupIdSerialization:
    def test_group_id_survives_to_dict_from_dict(self):
        task = TaskQuery(raw_task="test", task_id="t1", mode="plan", repo_id=1)
        ctx = StageContext(task=task, repo_id=1, repo_path="/tmp")
        ctx.classified_symbols = [
            ClassifiedSymbol(
                symbol_id=1, file_id=10, name="fn", kind="function",
                start_line=1, end_line=10, detail_level="primary",
                group_id="sim_group_1",
            ),
        ]
        d = ctx.to_dict()
        assert d["classified_symbols"][0]["group_id"] == "sim_group_1"

        # Reconstruct
        restored = StageContext.from_dict(d, task)
        assert restored.classified_symbols[0].group_id == "sim_group_1"

    def test_backward_compat_missing_group_id(self):
        """Old serialized data without group_id works (defaults to None)."""
        task = TaskQuery(raw_task="test", task_id="t1", mode="plan", repo_id=1)
        data = {
            "repo_id": 1,
            "repo_path": "/tmp",
            "scoped_files": [],
            "included_file_ids": [],
            "classified_symbols": [{
                "symbol_id": 1, "file_id": 10, "name": "fn", "kind": "function",
                "start_line": 1, "end_line": 10, "detail_level": "primary",
                "reason": "", "signature": "",
            }],
            "tokens_used": 0,
            "stage_timings": {},
        }
        ctx = StageContext.from_dict(data, task)
        assert ctx.classified_symbols[0].group_id is None


# ---------- Assembly group integrity ----------

class TestAssemblyGroupIntegrity:
    def test_partial_group_drops_all(self):
        from clean_room_agent.retrieval.context_assembly import _enforce_group_integrity

        symbols = [
            ClassifiedSymbol(
                symbol_id=1, file_id=10, name="a", kind="function",
                start_line=1, end_line=10, detail_level="primary",
                group_id="sim_group_1",
            ),
            ClassifiedSymbol(
                symbol_id=2, file_id=20, name="b", kind="function",
                start_line=1, end_line=10, detail_level="primary",
                group_id="sim_group_1",
            ),
        ]
        # Only file 10 is included, file 20 is missing
        rendered = [{"file_id": 10, "path": "a.py", "tokens": 100}]
        filtered, drops = _enforce_group_integrity(rendered, symbols)
        # File 10 should be dropped since file 20 is missing
        assert len(filtered) == 0
        assert len(drops) == 1
        assert drops[0]["file_id"] == 10

    def test_complete_group_kept(self):
        from clean_room_agent.retrieval.context_assembly import _enforce_group_integrity

        symbols = [
            ClassifiedSymbol(
                symbol_id=1, file_id=10, name="a", kind="function",
                start_line=1, end_line=10, detail_level="primary",
                group_id="sim_group_1",
            ),
            ClassifiedSymbol(
                symbol_id=2, file_id=20, name="b", kind="function",
                start_line=1, end_line=10, detail_level="primary",
                group_id="sim_group_1",
            ),
        ]
        rendered = [
            {"file_id": 10, "path": "a.py", "tokens": 100},
            {"file_id": 20, "path": "b.py", "tokens": 100},
        ]
        filtered, drops = _enforce_group_integrity(rendered, symbols)
        assert len(filtered) == 2
        assert len(drops) == 0

    def test_no_groups_passes_through(self):
        from clean_room_agent.retrieval.context_assembly import _enforce_group_integrity

        symbols = [
            ClassifiedSymbol(
                symbol_id=1, file_id=10, name="a", kind="function",
                start_line=1, end_line=10, detail_level="primary",
            ),
        ]
        rendered = [{"file_id": 10, "path": "a.py", "tokens": 100}]
        filtered, drops = _enforce_group_integrity(rendered, symbols)
        assert len(filtered) == 1
        assert len(drops) == 0
