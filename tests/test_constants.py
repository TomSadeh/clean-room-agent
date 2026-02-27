"""Tests for centralized file extension / language constants (A17)."""

from clean_room_agent.constants import (
    KNOWN_EXTENSIONS,
    LANGUAGE_MAP,
    TS_JS_RESOLVE_EXTENSIONS,
    language_from_extension,
)


class TestLanguageMap:
    """LANGUAGE_MAP must cover all supported languages."""

    def test_python_extension(self):
        assert LANGUAGE_MAP[".py"] == "python"

    def test_typescript_extensions(self):
        assert LANGUAGE_MAP[".ts"] == "typescript"
        assert LANGUAGE_MAP[".tsx"] == "typescript"

    def test_javascript_extensions(self):
        assert LANGUAGE_MAP[".js"] == "javascript"
        assert LANGUAGE_MAP[".jsx"] == "javascript"
        assert LANGUAGE_MAP[".mjs"] == "javascript"
        assert LANGUAGE_MAP[".cjs"] == "javascript"

    def test_c_extensions(self):
        assert LANGUAGE_MAP[".c"] == "c"
        assert LANGUAGE_MAP[".h"] == "c"

    def test_all_extensions_present(self):
        assert set(LANGUAGE_MAP) == {
            ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".c", ".h",
        }


class TestKnownExtensions:
    """KNOWN_EXTENSIONS is derived from LANGUAGE_MAP keys."""

    def test_is_frozenset(self):
        assert isinstance(KNOWN_EXTENSIONS, frozenset)

    def test_matches_language_map_keys(self):
        assert KNOWN_EXTENSIONS == frozenset(LANGUAGE_MAP)


class TestTsJsResolveExtensions:
    """TS_JS_RESOLVE_EXTENSIONS includes all TS/JS extensions for import resolution."""

    def test_is_tuple(self):
        assert isinstance(TS_JS_RESOLVE_EXTENSIONS, tuple)

    def test_includes_mjs_cjs(self):
        assert ".mjs" in TS_JS_RESOLVE_EXTENSIONS
        assert ".cjs" in TS_JS_RESOLVE_EXTENSIONS

    def test_includes_core_extensions(self):
        for ext in (".ts", ".tsx", ".js", ".jsx"):
            assert ext in TS_JS_RESOLVE_EXTENSIONS


class TestLanguageFromExtension:
    """language_from_extension maps file paths to language names."""

    def test_python_file(self):
        assert language_from_extension("src/foo.py") == "python"

    def test_typescript_file(self):
        assert language_from_extension("src/foo.ts") == "typescript"
        assert language_from_extension("src/foo.tsx") == "typescript"

    def test_javascript_file(self):
        assert language_from_extension("src/foo.js") == "javascript"
        assert language_from_extension("src/foo.jsx") == "javascript"

    def test_mjs_cjs(self):
        assert language_from_extension("lib/utils.mjs") == "javascript"
        assert language_from_extension("lib/utils.cjs") == "javascript"

    def test_unsupported_returns_none(self):
        assert language_from_extension("README.md") is None
        assert language_from_extension("data.json") is None
        assert language_from_extension("Makefile") is None
