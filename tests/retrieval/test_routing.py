"""Tests for stage routing — per-stage binary LLM selection."""

from unittest.mock import MagicMock

import pytest

from clean_room_agent.retrieval.dataclasses import TaskQuery
from clean_room_agent.retrieval.routing import (
    ROUTING_BINARY_SYSTEM,
    route_stages,
)


@pytest.fixture
def sample_task():
    return TaskQuery(
        raw_task="Fix the authentication bug in login.py",
        task_id="t1",
        mode="plan",
        repo_id=1,
        intent_summary="Fix authentication bypass in login handler",
        task_type="bug_fix",
        seed_file_ids=[10, 20],
        seed_symbol_ids=[100],
        mentioned_files=["src/login.py"],
        keywords=["auth", "login"],
    )


@pytest.fixture
def available_stages():
    return {
        "scope": "Finds related files using imports, co-change, and metadata.",
        "precision": "Classifies symbols by relevance level.",
    }


def _binary_routing_llm(stage_verdicts):
    """Create mock LLM that returns yes/no per stage name.

    stage_verdicts: dict mapping stage_name -> "yes"/"no".
    """
    llm = MagicMock()
    llm.flush = MagicMock()
    llm.config.context_window = 32768
    llm.config.max_tokens = 4096

    def _complete(prompt, system=None, *, sub_stage=None):
        resp = MagicMock()
        for name, answer in stage_verdicts.items():
            if f"Stage: {name}" in prompt:
                resp.text = answer
                return resp
        resp.text = "no"  # default
        return resp

    llm.complete.side_effect = _complete
    return llm


class TestRouteStages:
    def test_selects_both_stages(self, sample_task, available_stages):
        llm = _binary_routing_llm({"scope": "yes", "precision": "yes"})
        selected, reasoning = route_stages(sample_task, available_stages, llm)
        assert selected == ["scope", "precision"]
        assert reasoning == ""  # reasoning dropped in binary decomposition
        # One call per stage
        assert llm.complete.call_count == 2
        # Verify system prompt is ROUTING_BINARY_SYSTEM
        for c in llm.complete.call_args_list:
            assert c.kwargs["system"] == ROUTING_BINARY_SYSTEM

    def test_selects_no_stages(self, sample_task, available_stages):
        llm = _binary_routing_llm({"scope": "no", "precision": "no"})
        selected, reasoning = route_stages(sample_task, available_stages, llm)
        assert selected == []

    def test_selects_subset(self, sample_task, available_stages):
        llm = _binary_routing_llm({"scope": "yes", "precision": "no"})
        selected, reasoning = route_stages(sample_task, available_stages, llm)
        assert selected == ["scope"]

    def test_empty_available_stages(self, sample_task):
        llm = MagicMock()
        llm.flush = MagicMock()
        selected, reasoning = route_stages(sample_task, {}, llm)
        assert selected == []
        llm.complete.assert_not_called()

    def test_r2_default_deny_unparseable(self, sample_task, available_stages):
        """R2: unparseable response → stage not selected (default-deny)."""
        llm = MagicMock()
        llm.flush = MagicMock()
        llm.config.context_window = 32768
        llm.config.max_tokens = 4096
        resp = MagicMock()
        resp.text = "maybe"  # not yes/no
        llm.complete.return_value = resp

        selected, reasoning = route_stages(sample_task, available_stages, llm)
        # Unparseable defaults to False → neither stage selected
        assert selected == []

    def test_requires_logged_client(self, sample_task, available_stages):
        """Routing requires a LoggedLLMClient (flush capability)."""
        llm = MagicMock(spec=[])  # no flush attribute
        with pytest.raises(TypeError, match="logging-capable"):
            route_stages(sample_task, available_stages, llm)

    def test_prompt_contains_task_context(self, sample_task, available_stages):
        """Each binary call includes task intent summary."""
        llm = _binary_routing_llm({"scope": "yes", "precision": "no"})
        route_stages(sample_task, available_stages, llm)
        # Check first call's prompt contains task info
        first_prompt = llm.complete.call_args_list[0].args[0]
        assert "Fix authentication bypass" in first_prompt
        assert "bug_fix" in first_prompt
