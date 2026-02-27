"""Tests for library guard in precision stage."""

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


class TestLibrarySymbolPreFilter:
    def test_library_symbol_auto_classified_as_type_context(self):
        """R17: Library symbols are auto-classified as type_context without LLM."""
        # Return "yes" for all binary calls → handle_request: pass1 yes, pass2 yes → primary
        mock_llm = _mock_llm_with_response("yes")

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

        # Library symbol auto-classified as type_context (R17)
        widget_cs = next(cs for cs in result if cs.name == "Widget")
        assert widget_cs.detail_level == "type_context"
        assert widget_cs.file_source == "library"
        assert "library symbol" in widget_cs.reason

        # Project symbol classified by LLM as primary
        handler_cs = next(cs for cs in result if cs.name == "handle_request")
        assert handler_cs.detail_level == "primary"
        assert handler_cs.file_source == "project"

        # LLM prompts should only contain project symbol, not library
        for c in mock_llm.complete.call_args_list:
            prompt = c.args[0]
            assert "handle_request" in prompt
            assert "Widget" not in prompt

    def test_library_only_candidates_skip_llm(self):
        """R17: When all candidates are library, LLM is not called."""
        mock_llm = MagicMock()
        mock_llm.config.context_window = 32768
        mock_llm.config.max_tokens = 4096

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
        assert config_cs.detail_level == "type_context"
        assert config_cs.file_source == "library"
        # LLM should NOT have been called
        mock_llm.complete.assert_not_called()
