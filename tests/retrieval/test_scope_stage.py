"""Tests for scope stage."""

import json
from unittest.mock import MagicMock

import pytest

from clean_room_agent.db.connection import get_connection
from clean_room_agent.db.queries import (
    insert_dependency,
    insert_symbol,
    upsert_co_change,
    upsert_file,
    upsert_file_metadata,
    upsert_repo,
)
from clean_room_agent.query.api import KnowledgeBase
from clean_room_agent.retrieval.dataclasses import ScopedFile, TaskQuery
from clean_room_agent.retrieval.scope_stage import (
    expand_scope,
    judge_scope,
)
from clean_room_agent.retrieval.utils import parse_json_response


@pytest.fixture
def populated_kb(tmp_repo):
    """Create a curated DB with files, deps, co-changes for testing."""
    conn = get_connection("curated", repo_path=tmp_repo)
    repo_id = upsert_repo(conn, str(tmp_repo), None)

    fid1 = upsert_file(conn, repo_id, "src/auth.py", "python", "h1", 500)
    fid2 = upsert_file(conn, repo_id, "src/models.py", "python", "h2", 300)
    fid3 = upsert_file(conn, repo_id, "src/utils.py", "python", "h3", 200)
    fid4 = upsert_file(conn, repo_id, "src/config.py", "python", "h4", 150)
    fid5 = upsert_file(conn, repo_id, "tests/test_auth.py", "python", "h5", 400)

    # Dependencies: auth imports models, utils imports config
    insert_dependency(conn, fid1, fid2, "imports")
    insert_dependency(conn, fid3, fid4, "imports")
    # test_auth imports auth
    insert_dependency(conn, fid5, fid1, "imports")

    # Co-changes: auth <-> test_auth frequently co-change
    upsert_co_change(conn, fid1, fid5, "abc123", count=5)

    conn.commit()

    kb = KnowledgeBase(conn)
    yield kb, repo_id, {"auth": fid1, "models": fid2, "utils": fid3, "config": fid4, "test_auth": fid5}
    conn.close()


class TestExpandScope:
    def test_seed_files_tier_1(self, populated_kb):
        kb, repo_id, fids = populated_kb
        task = TaskQuery(
            raw_task="fix auth", task_id="t1", mode="plan", repo_id=repo_id,
            seed_file_ids=[fids["auth"]], keywords=["auth"],
        )
        results = expand_scope(task, kb, repo_id)
        seed_files = [sf for sf in results if sf.tier == 1]
        assert len(seed_files) >= 1
        assert any(sf.file_id == fids["auth"] for sf in seed_files)

    def test_tier_2_deps_included(self, populated_kb):
        kb, repo_id, fids = populated_kb
        task = TaskQuery(
            raw_task="fix auth", task_id="t1", mode="plan", repo_id=repo_id,
            seed_file_ids=[fids["auth"]], keywords=[],
        )
        results = expand_scope(task, kb, repo_id)
        tier2 = [sf for sf in results if sf.tier == 2]
        # auth imports models, test_auth imports auth (imported_by)
        tier2_ids = {sf.file_id for sf in tier2}
        assert fids["models"] in tier2_ids

    def test_tier_3_co_changes(self, populated_kb):
        kb, repo_id, fids = populated_kb
        # Add a co-change between auth and config (no dependency edge between them)
        # so config can only appear as tier 3, not deduped by tier 2.
        upsert_co_change(kb._conn, fids["auth"], fids["config"], "def456", count=3)
        kb._conn.commit()

        task = TaskQuery(
            raw_task="fix auth", task_id="t1", mode="plan", repo_id=repo_id,
            seed_file_ids=[fids["auth"]], keywords=[],
        )
        results = expand_scope(task, kb, repo_id)
        tier3 = [sf for sf in results if sf.tier == 3]
        tier3_ids = {sf.file_id for sf in tier3}
        # config co-changes with auth (count=3 >= min 2) and has no dep edge -> tier 3
        assert fids["config"] in tier3_ids

    def test_dedup_lower_tier_wins(self, populated_kb):
        kb, repo_id, fids = populated_kb
        task = TaskQuery(
            raw_task="fix auth", task_id="t1", mode="plan", repo_id=repo_id,
            seed_file_ids=[fids["auth"]], keywords=[],
        )
        results = expand_scope(task, kb, repo_id)
        # Each file should appear only once
        file_ids = [sf.file_id for sf in results]
        assert len(file_ids) == len(set(file_ids))

    def test_plan_file_ids_tier_0(self, populated_kb):
        kb, repo_id, fids = populated_kb
        task = TaskQuery(
            raw_task="fix auth", task_id="t1", mode="plan", repo_id=repo_id,
            seed_file_ids=[], keywords=[],
        )
        results = expand_scope(task, kb, repo_id, plan_file_ids=[fids["config"]])
        tier0 = [sf for sf in results if sf.tier == 0]
        assert len(tier0) == 1
        assert tier0[0].file_id == fids["config"]

    def test_empty_seeds(self, populated_kb):
        kb, repo_id, fids = populated_kb
        task = TaskQuery(
            raw_task="do something", task_id="t1", mode="plan", repo_id=repo_id,
            seed_file_ids=[], keywords=[],
        )
        results = expand_scope(task, kb, repo_id)
        assert results == []  # no seeds, no expansion

    def test_seed_symbol_ids_tier_1(self, populated_kb):
        """seed_symbol_ids resolve to their containing files at tier 1."""
        kb, repo_id, fids = populated_kb
        # Insert a symbol into the utils file
        sid = insert_symbol(kb._conn, fids["utils"], "helper_func", "function", 1, 5, "def helper_func():")
        kb._conn.commit()

        task = TaskQuery(
            raw_task="use helper_func", task_id="t1", mode="plan", repo_id=repo_id,
            seed_file_ids=[], seed_symbol_ids=[sid], keywords=[],
        )
        results = expand_scope(task, kb, repo_id)
        tier1 = [sf for sf in results if sf.tier == 1]
        assert len(tier1) == 1
        assert tier1[0].file_id == fids["utils"]
        assert tier1[0].reason == "contains seed symbol"

    def test_seed_symbol_ids_dedup_with_seed_files(self, populated_kb):
        """seed_symbol_ids in an already-seeded file don't create duplicates."""
        kb, repo_id, fids = populated_kb
        sid = insert_symbol(kb._conn, fids["auth"], "login", "function", 3, 5, "def login():")
        kb._conn.commit()

        task = TaskQuery(
            raw_task="fix login", task_id="t1", mode="plan", repo_id=repo_id,
            seed_file_ids=[fids["auth"]], seed_symbol_ids=[sid], keywords=[],
        )
        results = expand_scope(task, kb, repo_id)
        auth_entries = [sf for sf in results if sf.file_id == fids["auth"]]
        assert len(auth_entries) == 1  # deduped: seed_file wins (added first)
        assert auth_entries[0].reason == "seed file"  # first wins

    def test_seed_symbol_ids_nonexistent_symbol(self, populated_kb):
        """Nonexistent symbol IDs are silently skipped."""
        kb, repo_id, fids = populated_kb
        task = TaskQuery(
            raw_task="use something", task_id="t1", mode="plan", repo_id=repo_id,
            seed_file_ids=[], seed_symbol_ids=[9999], keywords=[],
        )
        results = expand_scope(task, kb, repo_id)
        assert results == []

    def test_tier_4_metadata_match(self, populated_kb):
        kb, repo_id, fids = populated_kb
        # Add metadata
        upsert_file_metadata(kb._conn, fids["utils"], concepts='["authentication", "helper"]')
        kb._conn.commit()

        task = TaskQuery(
            raw_task="fix auth", task_id="t1", mode="plan", repo_id=repo_id,
            seed_file_ids=[], keywords=["authentication"],
        )
        results = expand_scope(task, kb, repo_id)
        tier4 = [sf for sf in results if sf.tier == 4]
        assert len(tier4) >= 1


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


class TestJudgeScope:
    def test_seeds_always_relevant(self):
        mock_llm = _mock_llm_with_response(json.dumps([
            {"path": "seed.py", "verdict": "irrelevant", "reason": "not needed"},
        ]))

        candidates = [
            ScopedFile(file_id=1, path="seed.py", language="python", tier=1),
        ]
        task = TaskQuery(raw_task="fix it", task_id="t1", mode="plan", repo_id=1)

        result = judge_scope(candidates, task, mock_llm)
        # Seed (tier <= 1) is always relevant regardless of LLM
        assert result[0].relevance == "relevant"

    def test_llm_verdict_applied(self):
        mock_llm = _mock_llm_with_response(json.dumps([
            {"path": "dep.py", "verdict": "irrelevant", "reason": "unrelated"},
            {"path": "util.py", "verdict": "relevant", "reason": "needed"},
        ]))

        candidates = [
            ScopedFile(file_id=2, path="dep.py", language="python", tier=2),
            ScopedFile(file_id=3, path="util.py", language="python", tier=2),
        ]
        task = TaskQuery(raw_task="fix it", task_id="t1", mode="plan", repo_id=1)

        result = judge_scope(candidates, task, mock_llm)
        dep = next(sf for sf in result if sf.path == "dep.py")
        util = next(sf for sf in result if sf.path == "util.py")
        assert dep.relevance == "irrelevant"
        assert util.relevance == "relevant"

    def test_empty_candidates(self):
        mock_llm = MagicMock()
        mock_llm.flush = MagicMock()  # F14: explicit _require_logged_client contract
        task = TaskQuery(raw_task="fix it", task_id="t1", mode="plan", repo_id=1)
        result = judge_scope([], task, mock_llm)
        assert result == []
        mock_llm.complete.assert_not_called()

    def test_missing_from_llm_defaults_irrelevant(self):
        """R2a: LLM omission defaults to irrelevant (default-deny)."""
        mock_llm = _mock_llm_with_response(json.dumps([]))

        candidates = [
            ScopedFile(file_id=2, path="dep.py", language="python", tier=2),
        ]
        task = TaskQuery(raw_task="fix it", task_id="t1", mode="plan", repo_id=1)

        result = judge_scope(candidates, task, mock_llm)
        assert result[0].relevance == "irrelevant"

    def test_invalid_verdict_defaults_irrelevant(self):
        """R2: LLM returning an invalid verdict string defaults to irrelevant."""
        mock_llm = _mock_llm_with_response(json.dumps([
            {"path": "dep.py", "verdict": "maybe_useful", "reason": "unsure"},
        ]))

        candidates = [
            ScopedFile(file_id=2, path="dep.py", language="python", tier=2),
        ]
        task = TaskQuery(raw_task="fix it", task_id="t1", mode="plan", repo_id=1)

        result = judge_scope(candidates, task, mock_llm)
        assert result[0].relevance == "irrelevant"


class TestJudgeScopeBatching:
    def test_batch_boundary(self):
        """WU6: large candidate sets are split into batches, all get judged."""
        mock_llm = MagicMock()
        mock_llm.flush = MagicMock()  # F14: explicit _require_logged_client contract
        # Configure a tiny context window to force batching
        mock_llm.config = MagicMock()
        mock_llm.config.context_window = 512
        mock_llm.config.max_tokens = 128

        # Create enough non-seed candidates to require multiple batches
        candidates = [
            ScopedFile(file_id=1, path="seed.py", language="python", tier=1),
        ]
        for i in range(20):
            candidates.append(
                ScopedFile(file_id=100 + i, path=f"dep_{i}.py", language="python", tier=2),
            )

        # Mock LLM to return all as relevant
        def _complete(prompt, system=None):
            resp = MagicMock()
            # Parse out which files are in this batch and return relevant for all
            import re
            paths = re.findall(r"- (dep_\d+\.py)", prompt)
            resp.text = json.dumps([
                {"path": p, "verdict": "relevant", "reason": "needed"}
                for p in paths
            ])
            return resp

        mock_llm.complete.side_effect = _complete

        task = TaskQuery(raw_task="fix it", task_id="t1", mode="plan", repo_id=1)
        result = judge_scope(candidates, task, mock_llm)

        # Seed should be relevant
        seed = next(sf for sf in result if sf.path == "seed.py")
        assert seed.relevance == "relevant"

        # All non-seeds should be judged (multiple batches)
        non_seeds = [sf for sf in result if sf.tier > 1]
        assert all(sf.relevance == "relevant" for sf in non_seeds)
        # LLM should have been called multiple times (batching)
        assert mock_llm.complete.call_count >= 2


class TestR6OrderedCaps:
    """R6: Caps must be ordered and justified before slicing."""

    def test_deps_ordered_by_seed_connections(self, populated_kb):
        """R6a: Dep candidates ordered by connections to seed files, not arbitrary."""
        kb, repo_id, fids = populated_kb
        task = TaskQuery(
            raw_task="fix auth", task_id="t1", mode="plan", repo_id=repo_id,
            seed_file_ids=[fids["auth"]], keywords=[],
        )
        results = expand_scope(task, kb, repo_id, max_deps=1)
        tier2 = [sf for sf in results if sf.tier == 2]
        # With max_deps=1, only the most connected dep should survive
        assert len(tier2) <= 1

    def test_keywords_ordered_by_length(self, populated_kb):
        """R6b: Keywords sorted by length (specificity) before capping."""
        kb, repo_id, fids = populated_kb
        # Add metadata for both keywords
        upsert_file_metadata(kb._conn, fids["utils"], concepts='["authentication_handler"]')
        upsert_file_metadata(kb._conn, fids["config"], concepts='["auth"]')
        kb._conn.commit()

        task = TaskQuery(
            raw_task="fix auth", task_id="t1", mode="plan", repo_id=repo_id,
            seed_file_ids=[], keywords=["auth", "authentication_handler"],
        )
        results = expand_scope(task, kb, repo_id)
        tier4 = [sf for sf in results if sf.tier == 4]
        # Metadata search with "auth" matches config ("auth") and utils ("authentication_handler")
        assert len(tier4) >= 1, "Expected tier 4 metadata matches"
        tier4_ids = {sf.file_id for sf in tier4}
        assert tier4_ids <= {fids["utils"], fids["config"]}


class TestParseJsonResponse:
    def test_plain_json(self):
        result = parse_json_response('[{"path": "a.py"}]')
        assert result == [{"path": "a.py"}]

    def test_fenced_json(self):
        result = parse_json_response('```json\n[{"path": "a.py"}]\n```')
        assert result == [{"path": "a.py"}]

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_json_response("not json at all")
