"""Language -> parser dispatch registry."""

from clean_room_agent.parsers.base import LanguageParser
from clean_room_agent.parsers.python_parser import PythonParser
from clean_room_agent.parsers.ts_js_parser import TSJSParser

_PARSERS: dict[str, LanguageParser] = {}


def get_parser(language: str) -> LanguageParser:
    """Get or create a parser for the given language."""
    if language not in _PARSERS:
        if language == "python":
            _PARSERS[language] = PythonParser()
        elif language in ("typescript", "javascript"):
            _PARSERS[language] = TSJSParser(language)
        else:
            raise ValueError(f"No parser for language: {language}")
    return _PARSERS[language]
