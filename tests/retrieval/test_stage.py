"""Tests for stage protocol, context, and registry."""

import json

import pytest

from clean_room_agent.retrieval.dataclasses import (
    ClassifiedSymbol,
    ScopedFile,
    TaskQuery,
)
from clean_room_agent.retrieval.stage import (
    StageContext,
    _STAGE_REGISTRY,
    get_stage,
    register_stage,
)


@pytest.fixture
def sample_task():
    return TaskQuery(
        raw_task="fix the bug", task_id="t1", mode="plan", repo_id=1,
        seed_file_ids=[10], keywords=["bug"],
    )


class TestStageContext:
    def test_initial_state(self, sample_task):
        ctx = StageContext(task=sample_task, repo_id=1, repo_path="/repo")
        assert ctx.scoped_files == []
        assert ctx.included_file_ids == set()
        assert ctx.classified_symbols == []
        assert ctx.tokens_used == 0

    def test_get_relevant_file_ids(self, sample_task):
        ctx = StageContext(task=sample_task, repo_id=1, repo_path="/repo")
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="a.py", language="python", tier=1, relevance="relevant"),
            ScopedFile(file_id=2, path="b.py", language="python", tier=2, relevance="irrelevant"),
            ScopedFile(file_id=3, path="c.py", language="python", tier=3, relevance="relevant"),
        ]
        assert ctx.get_relevant_file_ids() == {1, 3}

    def test_mutation(self, sample_task):
        ctx = StageContext(task=sample_task, repo_id=1, repo_path="/repo")
        ctx.scoped_files.append(
            ScopedFile(file_id=1, path="a.py", language="python", tier=1)
        )
        ctx.included_file_ids.add(1)
        ctx.tokens_used = 100
        ctx.stage_timings["scope"] = 500
        assert len(ctx.scoped_files) == 1
        assert 1 in ctx.included_file_ids

    def test_to_dict(self, sample_task):
        ctx = StageContext(task=sample_task, repo_id=1, repo_path="/repo")
        ctx.scoped_files = [
            ScopedFile(file_id=1, path="a.py", language="python", tier=1, relevance="relevant"),
        ]
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=10, file_id=1, name="foo", kind="function",
                             start_line=1, end_line=5, detail_level="primary"),
        ]
        ctx.included_file_ids = {1}
        ctx.tokens_used = 200

        d = ctx.to_dict()
        assert d["repo_id"] == 1
        assert d["repo_path"] == "/repo"
        assert len(d["scoped_files"]) == 1
        assert d["scoped_files"][0]["path"] == "a.py"
        assert len(d["classified_symbols"]) == 1
        assert d["classified_symbols"][0]["name"] == "foo"
        assert d["included_file_ids"] == [1]
        assert d["tokens_used"] == 200

    def test_from_dict_round_trip(self, sample_task):
        ctx = StageContext(task=sample_task, repo_id=1, repo_path="/repo")
        ctx.scoped_files = [
            ScopedFile(file_id=5, path="x.py", language="python", tier=2,
                       relevance="relevant", reason="dep"),
        ]
        ctx.classified_symbols = [
            ClassifiedSymbol(symbol_id=20, file_id=5, name="bar", kind="class",
                             start_line=10, end_line=30, detail_level="supporting",
                             reason="context"),
        ]
        ctx.included_file_ids = {5}
        ctx.tokens_used = 500
        ctx.stage_timings = {"scope": 100}

        d = ctx.to_dict()
        restored = StageContext.from_dict(d, sample_task)
        assert restored.repo_id == 1
        assert len(restored.scoped_files) == 1
        assert restored.scoped_files[0].path == "x.py"
        assert len(restored.classified_symbols) == 1
        assert restored.classified_symbols[0].name == "bar"
        assert restored.included_file_ids == {5}
        assert restored.tokens_used == 500

    def test_to_json(self, sample_task):
        ctx = StageContext(task=sample_task, repo_id=1, repo_path="/repo")
        j = ctx.to_json()
        data = json.loads(j)
        assert data["repo_id"] == 1


class TestStageRegistry:
    def test_register_and_get(self):
        # Clean up after test
        original = dict(_STAGE_REGISTRY)

        @register_stage("test_stage_123")
        class TestStage123:
            @property
            def name(self):
                return "test_stage_123"

            def run(self, context, kb, task, llm):
                return context

        stage = get_stage("test_stage_123")
        assert stage.name == "test_stage_123"

        # Cleanup
        _STAGE_REGISTRY.clear()
        _STAGE_REGISTRY.update(original)

    def test_unknown_stage_raises(self):
        with pytest.raises(ValueError, match="Unknown stage"):
            get_stage("nonexistent_stage_xyz")

    def test_builtin_stages_registered(self):
        # Import the modules to trigger registration
        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        scope = get_stage("scope")
        assert scope.name == "scope"

        precision = get_stage("precision")
        assert precision.name == "precision"

    def test_stages_registered_via_package_import(self):
        """B1: importing retrieval package auto-registers stages."""
        import clean_room_agent.retrieval  # noqa: F401

        scope = get_stage("scope")
        assert scope.name == "scope"
        precision = get_stage("precision")
        assert precision.name == "precision"


class TestFromDictMissingKeys:
    def test_missing_scoped_files_raises(self, sample_task):
        data = {
            "repo_id": 1,
            "repo_path": "/repo",
            # missing scoped_files
            "included_file_ids": [],
            "classified_symbols": [],
            "tokens_used": 0,
            "stage_timings": {},
        }
        with pytest.raises(KeyError):
            StageContext.from_dict(data, sample_task)

    def test_missing_tokens_used_raises(self, sample_task):
        data = {
            "repo_id": 1,
            "repo_path": "/repo",
            "scoped_files": [],
            "included_file_ids": [],
            "classified_symbols": [],
            # missing tokens_used
            "stage_timings": {},
        }
        with pytest.raises(KeyError):
            StageContext.from_dict(data, sample_task)
