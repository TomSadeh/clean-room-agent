"""Tests for library guard in precision stage."""

import json
from unittest.mock import MagicMock

from clean_room_agent.retrieval.dataclasses import TaskQuery
from clean_room_agent.retrieval.precision_stage import classify_symbols


def _mock_llm_with_response(text):
    """Create a mock LLM with standard config and a fixed response."""
    mock_llm = MagicMock()
    mock_llm.config.context_window = 32768
    mock_llm.config.max_tokens = 4096
    mock_response = MagicMock()
    mock_response.text = text
    mock_llm.complete.return_value = mock_response
    return mock_llm


class TestLibrarySymbolDowngrade:
    def test_library_symbol_downgraded_from_primary(self):
        """A ClassifiedSymbol with file_source='library' and detail_level='primary'
        gets downgraded to 'type_context'."""
        # LLM classifies the library symbol as "primary"
        mock_llm = _mock_llm_with_response(json.dumps([
            {
                "name": "Widget",
                "file_path": "requests/models.py",
                "start_line": 10,
                "detail_level": "primary",
                "reason": "directly used",
            },
            {
                "name": "handle_request",
                "file_path": "src/app.py",
                "start_line": 1,
                "detail_level": "primary",
                "reason": "changed function",
            },
        ]))

        candidates = [
            {
                "symbol_id": 1, "file_id": 100,
                "file_path": "requests/models.py",
                "name": "Widget", "kind": "class",
                "start_line": 10, "end_line": 50,
                "signature": "class Widget:",
                "connections": [],
                "file_source": "library",
            },
            {
                "symbol_id": 2, "file_id": 200,
                "file_path": "src/app.py",
                "name": "handle_request", "kind": "function",
                "start_line": 1, "end_line": 20,
                "signature": "def handle_request()",
                "connections": [],
                "file_source": "project",
            },
        ]

        task = TaskQuery(
            raw_task="fix request handling",
            task_id="t1", mode="plan", repo_id=1,
        )

        result = classify_symbols(candidates, task, mock_llm)

        assert len(result) == 2

        # Library symbol should be downgraded from primary to type_context
        widget_cs = next(cs for cs in result if cs.name == "Widget")
        assert widget_cs.detail_level == "type_context"
        assert widget_cs.file_source == "library"
        assert "downgraded from primary" in widget_cs.reason

        # Project symbol should remain primary
        handler_cs = next(cs for cs in result if cs.name == "handle_request")
        assert handler_cs.detail_level == "primary"
        assert handler_cs.file_source == "project"

    def test_library_symbol_supporting_not_downgraded(self):
        """A library symbol classified as 'supporting' is not downgraded."""
        mock_llm = _mock_llm_with_response(json.dumps([
            {
                "name": "Config",
                "file_path": "flask/config.py",
                "start_line": 5,
                "detail_level": "supporting",
                "reason": "context for handler",
            },
        ]))

        candidates = [
            {
                "symbol_id": 3, "file_id": 300,
                "file_path": "flask/config.py",
                "name": "Config", "kind": "class",
                "start_line": 5, "end_line": 40,
                "signature": "class Config:",
                "connections": [],
                "file_source": "library",
            },
        ]

        task = TaskQuery(
            raw_task="update config handling",
            task_id="t2", mode="plan", repo_id=1,
        )

        result = classify_symbols(candidates, task, mock_llm)

        assert len(result) == 1
        config_cs = result[0]
        assert config_cs.detail_level == "supporting"
        assert config_cs.file_source == "library"
        # Should NOT have the downgrade notice
        assert "downgraded from primary" not in config_cs.reason
