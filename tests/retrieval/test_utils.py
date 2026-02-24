"""Tests for retrieval/utils.py — parse_json_response edge cases."""

import pytest

from clean_room_agent.retrieval.utils import parse_json_response


class TestParseJsonResponse:
    """Comprehensive tests for LLM JSON response parsing."""

    def test_plain_json_array(self):
        result = parse_json_response('[{"path": "a.py"}]')
        assert result == [{"path": "a.py"}]

    def test_plain_json_object(self):
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_fenced_json(self):
        result = parse_json_response('```json\n[{"path": "a.py"}]\n```')
        assert result == [{"path": "a.py"}]

    def test_fenced_no_language_tag(self):
        result = parse_json_response('```\n[1, 2, 3]\n```')
        assert result == [1, 2, 3]

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_json_response("not json at all")

    def test_empty_array(self):
        assert parse_json_response("[]") == []

    def test_empty_object(self):
        assert parse_json_response("{}") == {}

    def test_whitespace_stripped(self):
        result = parse_json_response("  \n  [1, 2]  \n  ")
        assert result == [1, 2]

    def test_fenced_with_trailing_whitespace(self):
        result = parse_json_response('```json\n{"a": 1}\n```\n\n')
        assert result == {"a": 1}

    def test_fenced_with_empty_trailing_lines(self):
        """Trailing empty lines after closing fence are stripped."""
        result = parse_json_response('```json\n[1]\n```\n\n\n')
        assert result == [1]

    def test_no_closing_fence(self):
        """Incomplete fencing: opening ``` but no closing ```.
        Should still attempt to parse the content after stripping the opening line."""
        result = parse_json_response('```json\n[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_context_in_error_message(self):
        """Custom context label appears in error message."""
        with pytest.raises(ValueError, match="Failed to parse scope judgment"):
            parse_json_response("bad", context="scope judgment")

    def test_nested_string_with_backticks(self):
        """JSON containing backtick characters inside string values."""
        raw = '{"code": "use `foo` here"}'
        result = parse_json_response(raw)
        assert result == {"code": "use `foo` here"}

    def test_raw_text_in_error_includes_content(self):
        """Error message includes the raw text for debugging."""
        with pytest.raises(ValueError, match="Raw: broken json"):
            parse_json_response("broken json")
