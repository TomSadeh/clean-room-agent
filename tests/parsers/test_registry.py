"""Tests for parsers/registry.py."""

import pytest

from clean_room_agent.parsers.registry import get_parser
from clean_room_agent.parsers.python_parser import PythonParser
from clean_room_agent.parsers.ts_js_parser import TSJSParser


class TestGetParser:
    def test_python(self):
        parser = get_parser("python")
        assert isinstance(parser, PythonParser)

    def test_typescript(self):
        parser = get_parser("typescript")
        assert isinstance(parser, TSJSParser)

    def test_javascript(self):
        parser = get_parser("javascript")
        assert isinstance(parser, TSJSParser)

    def test_caches_instances(self):
        p1 = get_parser("python")
        p2 = get_parser("python")
        assert p1 is p2

    def test_unknown_language_raises(self):
        with pytest.raises(ValueError, match="No parser for language"):
            get_parser("rust")
