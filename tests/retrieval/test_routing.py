"""Tests for stage routing module."""

import json
from unittest.mock import MagicMock

import pytest

from clean_room_agent.retrieval.dataclasses import TaskQuery
from clean_room_agent.retrieval.routing import (
    ROUTING_SYSTEM,
    build_routing_prompt,
    parse_routing_response,
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


class TestBuildRoutingPrompt:
    def test_contains_intent_summary(self, sample_task, available_stages):
        prompt = build_routing_prompt(sample_task, available_stages)
        assert "Fix authentication bypass" in prompt

    def test_contains_task_type(self, sample_task, available_stages):
        prompt = build_routing_prompt(sample_task, available_stages)
        assert "Type: bug_fix" in prompt

    def test_contains_seed_counts(self, sample_task, available_stages):
        prompt = build_routing_prompt(sample_task, available_stages)
        assert "Seed files: 2" in prompt
        assert "Seed symbols: 1" in prompt

    def test_contains_explicit_file_count(self, sample_task, available_stages):
        prompt = build_routing_prompt(sample_task, available_stages)
        assert "Explicit file paths: 1" in prompt

    def test_contains_stage_descriptions(self, sample_task, available_stages):
        prompt = build_routing_prompt(sample_task, available_stages)
        assert "- scope:" in prompt
        assert "- precision:" in prompt
        assert "Finds related files" in prompt
        assert "Classifies symbols" in prompt

    def test_empty_stages(self, sample_task):
        prompt = build_routing_prompt(sample_task, {})
        assert "Available stages:" in prompt
        # No stage lines after the header
        lines = prompt.split("\n")
        stage_header_idx = lines.index("Available stages:")
        assert stage_header_idx == len(lines) - 1


class TestParseRoutingResponse:
    def test_valid_response_both_stages(self):
        text = json.dumps({
            "stages": ["scope", "precision"],
            "reasoning": "Bug fix needs full context.",
        })
        stages, reasoning = parse_routing_response(text)
        assert stages == ["scope", "precision"]
        assert reasoning == "Bug fix needs full context."

    def test_valid_empty_stages(self):
        text = json.dumps({
            "stages": [],
            "reasoning": "Simple single-file edit.",
        })
        stages, reasoning = parse_routing_response(text)
        assert stages == []
        assert reasoning == "Simple single-file edit."

    def test_valid_no_reasoning(self):
        text = json.dumps({"stages": ["scope"]})
        stages, reasoning = parse_routing_response(text)
        assert stages == ["scope"]
        assert reasoning == ""

    def test_markdown_fenced_json(self):
        text = '```json\n{"stages": ["scope"], "reasoning": "test"}\n```'
        stages, reasoning = parse_routing_response(text)
        assert stages == ["scope"]

    def test_not_a_dict_raises(self):
        text = json.dumps(["scope", "precision"])
        with pytest.raises(ValueError, match="must be a JSON object"):
            parse_routing_response(text)

    def test_missing_stages_key_raises(self):
        text = json.dumps({"reasoning": "oops"})
        with pytest.raises(ValueError, match="missing 'stages' key"):
            parse_routing_response(text)

    def test_stages_not_a_list_raises(self):
        text = json.dumps({"stages": "scope"})
        with pytest.raises(ValueError, match="must be a list"):
            parse_routing_response(text)

    def test_non_string_stage_raises(self):
        text = json.dumps({"stages": [123]})
        with pytest.raises(ValueError, match="must be a string"):
            parse_routing_response(text)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_routing_response("not json at all")


class TestRouteStages:
    def _make_mock_llm(self, response_text):
        llm = MagicMock()
        response = MagicMock()
        response.text = response_text
        llm.complete.return_value = response
        # R3 budget validation needs real ints
        llm.config.context_window = 32768
        llm.config.max_tokens = 4096
        return llm

    def test_selects_both_stages(self, sample_task, available_stages):
        llm = self._make_mock_llm(json.dumps({
            "stages": ["scope", "precision"],
            "reasoning": "Need full context for bug fix.",
        }))
        selected, reasoning = route_stages(sample_task, available_stages, llm)
        assert selected == ["scope", "precision"]
        assert "bug fix" in reasoning.lower()
        llm.complete.assert_called_once()
        # Verify system prompt is ROUTING_SYSTEM
        _, kwargs = llm.complete.call_args
        assert kwargs["system"] == ROUTING_SYSTEM

    def test_selects_no_stages(self, sample_task, available_stages):
        llm = self._make_mock_llm(json.dumps({
            "stages": [],
            "reasoning": "Targeted single-file edit.",
        }))
        selected, reasoning = route_stages(sample_task, available_stages, llm)
        assert selected == []

    def test_selects_subset(self, sample_task, available_stages):
        llm = self._make_mock_llm(json.dumps({
            "stages": ["scope"],
            "reasoning": "Only need file expansion.",
        }))
        selected, reasoning = route_stages(sample_task, available_stages, llm)
        assert selected == ["scope"]

    def test_unknown_stage_raises(self, sample_task, available_stages):
        llm = self._make_mock_llm(json.dumps({
            "stages": ["scope", "hallucinated_stage"],
            "reasoning": "oops",
        }))
        with pytest.raises(ValueError, match="unknown stages.*hallucinated_stage"):
            route_stages(sample_task, available_stages, llm)

    def test_unparseable_response_raises(self, sample_task, available_stages):
        llm = self._make_mock_llm("I don't know what to do")
        with pytest.raises(ValueError):
            route_stages(sample_task, available_stages, llm)
