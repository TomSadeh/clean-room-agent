"""Tests for llm/enrichment.py."""

from clean_room_agent.llm.enrichment import _parse_enrichment_response


class TestParseEnrichmentResponse:
    def test_valid_json(self):
        text = '{"purpose": "utils", "module": "core", "domain": "general"}'
        result = _parse_enrichment_response(text)
        assert result["purpose"] == "utils"

    def test_strips_markdown_fencing(self):
        text = '```json\n{"purpose": "utils"}\n```'
        result = _parse_enrichment_response(text)
        assert result["purpose"] == "utils"

    def test_invalid_json_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Failed to parse"):
            _parse_enrichment_response("not json at all")
