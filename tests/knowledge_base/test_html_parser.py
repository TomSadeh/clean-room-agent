"""Tests for HTML parsers (Crafting Interpreters and cppreference)."""

from pathlib import Path

import pytest

from clean_room_agent.knowledge_base.html_parser import (
    _CraftingInterpretersParser,
    _CpprefParser,
    parse_crafting_interpreters,
    parse_cppreference,
)

_CI_DIR = (
    Path(__file__).resolve().parents[2]
    / "knowledge_base" / "c_references" / "crafting_interpreters"
)
_CPPREF_DIR = (
    Path(__file__).resolve().parents[2]
    / "knowledge_base" / "c_references" / "cppreference" / "reference" / "en" / "c"
)


class TestCraftingInterpretersParserUnit:
    """Unit tests for Crafting Interpreters HTML parser."""

    def test_extracts_chapter_title(self):
        html = (
            '<article class="chapter">'
            '<div class="number">14</div>'
            '<h1>Chunks of Bytecode</h1>'
            '<p>Some intro text.</p>'
            '</article>'
        )
        parser = _CraftingInterpretersParser()
        parser.feed(html)
        assert parser.chapter_title == "Chunks of Bytecode"
        assert parser.chapter_number == "14"

    def test_extracts_sections(self):
        html = (
            '<article class="chapter">'
            '<h1>Test Chapter</h1>'
            '<p>Intro.</p>'
            '<h2>First Section</h2>'
            '<p>Section content.</p>'
            '<h2>Second Section</h2>'
            '<p>More content.</p>'
            '</article>'
        )
        parser = _CraftingInterpretersParser()
        parser.feed(html)
        assert len(parser.sections) == 3  # h1 + 2 h2
        assert parser.sections[1]["title"] == "First Section"

    def test_skips_nav_content(self):
        html = (
            '<nav class="wide"><p>Nav content</p></nav>'
            '<article class="chapter">'
            '<h1>Real Chapter</h1>'
            '<p>Real content.</p>'
            '</article>'
        )
        parser = _CraftingInterpretersParser()
        parser.feed(html)
        assert parser.chapter_title == "Real Chapter"
        assert len(parser.sections) == 1
        assert "Nav content" not in parser.sections[0].get("content", "")

    def test_extracts_code_blocks(self):
        html = (
            '<article class="chapter">'
            '<h1>Code Test</h1>'
            '<p>Before code.</p>'
            '<div class="codehilite"><pre>int x = 42;</pre></div>'
            '<p>After code.</p>'
            '</article>'
        )
        parser = _CraftingInterpretersParser()
        parser.feed(html)
        content = parser.sections[0].get("content", "")
        assert "int x = 42" in content

    def test_skips_aside_content(self):
        html = (
            '<article class="chapter">'
            '<h1>Test</h1>'
            '<p>Main content.</p>'
            '<aside><p>Side note.</p></aside>'
            '<p>More main.</p>'
            '</article>'
        )
        parser = _CraftingInterpretersParser()
        parser.feed(html)
        content = parser.sections[0].get("content", "")
        assert "Side note" not in content
        assert "Main content" in content


class TestCpprefParserUnit:
    """Unit tests for cppreference HTML parser."""

    def test_extracts_function_name(self):
        html = (
            '<h1 id="firstHeading">malloc</h1>'
            '<div id="mw-content-text">'
            '<p>Allocates memory.</p>'
            '</div>'
        )
        parser = _CpprefParser()
        parser.feed(html)
        assert parser.title == "malloc"
        content = "".join(parser.content_parts)
        assert "Allocates memory" in content

    def test_extracts_code_blocks(self):
        html = (
            '<h1 id="firstHeading">test</h1>'
            '<div id="mw-content-text">'
            '<div class="c source-c"><pre>void *malloc(size_t size);</pre></div>'
            '</div>'
        )
        parser = _CpprefParser()
        parser.feed(html)
        content = "".join(parser.content_parts)
        assert "void *malloc" in content

    def test_skips_navbar(self):
        html = (
            '<h1 id="firstHeading">func</h1>'
            '<div id="mw-content-text">'
            '<div class="t-navbar">Nav stuff</div>'
            '<p>Real content.</p>'
            '</div>'
        )
        parser = _CpprefParser()
        parser.feed(html)
        content = "".join(parser.content_parts)
        assert "Nav stuff" not in content
        assert "Real content" in content


@pytest.mark.skipif(not _CI_DIR.exists(), reason="Crafting Interpreters source not available")
class TestCraftingInterpretersReal:
    """Integration tests against real Crafting Interpreters files."""

    @pytest.fixture(scope="class")
    def sections(self):
        return parse_crafting_interpreters(_CI_DIR)

    def test_has_chapters(self, sections):
        chapters = [s for s in sections if s.section_type == "chapter"]
        assert len(chapters) == 17

    def test_has_sections(self, sections):
        secs = [s for s in sections if s.section_type == "section"]
        assert len(secs) > 100

    def test_chapter_titles_present(self, sections):
        titles = {s.title for s in sections if s.section_type == "chapter"}
        assert "Chunks of Bytecode" in titles
        assert "A Virtual Machine" in titles


@pytest.mark.skipif(not _CPPREF_DIR.exists(), reason="cppreference source not available")
class TestCppreferenceReal:
    """Integration tests against real cppreference files."""

    @pytest.fixture(scope="class")
    def sections(self):
        return parse_cppreference(_CPPREF_DIR)

    def test_has_pages(self, sections):
        assert len(sections) > 500

    def test_malloc_present(self, sections):
        malloc = [s for s in sections if s.title == "malloc"]
        assert len(malloc) == 1
        assert "memory/malloc" in malloc[0].section_path

    def test_sections_have_content(self, sections):
        for s in sections[:50]:  # Check first 50
            assert len(s.content) > 0, f"Empty content for {s.section_path}"
