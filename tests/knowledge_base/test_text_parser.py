"""Tests for K&R text parser."""

from pathlib import Path

import pytest

from clean_room_agent.knowledge_base.text_parser import parse_knr2, _clean_content

# Path to real K&R data (skip if not available)
_KNR2_PATH = Path(__file__).resolve().parents[2] / "knowledge_base" / "c_references" / "knr2" / "knr2_clean.txt"


class TestCleanContent:
    """Unit tests for _clean_content helper."""

    def test_strips_form_feeds(self):
        result = _clean_content("hello\x0cworld")
        assert "\x0c" not in result
        assert "hello" in result
        assert "world" in result

    def test_strips_page_headers(self):
        text = "content before\n42 A TUTORIAL INTRODUCTION\nCHAPTER 1\ncontent after"
        result = _clean_content(text)
        assert "A TUTORIAL INTRODUCTION" not in result
        assert "CHAPTER 1" not in result
        assert "content before" in result
        assert "content after" in result

    def test_strips_bare_page_numbers(self):
        text = "some text\n42\nmore text"
        result = _clean_content(text)
        assert "\n42\n" not in result
        assert "some text" in result

    def test_collapses_blank_lines(self):
        text = "line1\n\n\n\nline2"
        result = _clean_content(text)
        assert "line1\n\nline2" == result


class TestParseKnr2Synthetic:
    """Tests against synthetic K&R-like text."""

    def test_parses_chapter_heading(self):
        text = "\n" * 100 + "\x0cPreface\nSome preface text.\n\n"
        text += "\x0cchapter 1: A Tutorial Introduction\nChapter body text.\n"
        text += "1.1 Getting Started\nSection body.\n"
        sections = parse_knr2(text)
        assert len(sections) >= 2
        ch = [s for s in sections if s.section_type == "chapter"]
        assert len(ch) == 1
        assert ch[0].title == "A Tutorial Introduction"
        assert ch[0].section_path == "ch1"

    def test_parses_section_heading(self):
        text = "\n" * 100 + "\x0cPreface\nPreface text.\n\n"
        text += "\x0cchapter 1: Intro\nIntro text.\n"
        text += "1.1 Getting Started\nSection content here.\n"
        text += "1.2 Variables\nMore content.\n"
        sections = parse_knr2(text)
        secs = [s for s in sections if s.section_type == "section"]
        assert len(secs) == 2
        assert secs[0].title == "Getting Started"
        assert secs[0].section_path == "ch1/1.1"
        assert secs[0].parent_section_path == "ch1"

    def test_parses_appendix_heading(self):
        text = "\n" * 100 + "\x0cPreface\nText.\n\n"
        text += "\x0cappendix A: Reference Manual\nAppendix content.\n"
        sections = parse_knr2(text)
        app = [s for s in sections if s.section_type == "appendix"]
        assert len(app) == 1
        assert app[0].title == "Reference Manual"
        assert app[0].section_path == "appA"


@pytest.mark.skipif(not _KNR2_PATH.exists(), reason="K&R source not available")
class TestParseKnr2Real:
    """Integration tests against real K&R text file."""

    @pytest.fixture(scope="class")
    def sections(self):
        text = _KNR2_PATH.read_text(encoding="utf-8", errors="replace")
        return parse_knr2(text)

    def test_has_8_chapters(self, sections):
        chapters = [s for s in sections if s.section_type == "chapter"]
        assert len(chapters) == 8

    def test_has_3_appendices(self, sections):
        appendices = [s for s in sections if s.section_type == "appendix"]
        assert len(appendices) == 3

    def test_chapter_titles(self, sections):
        chapters = {s.section_path: s.title for s in sections if s.section_type == "chapter"}
        assert chapters["ch1"] == "A Tutorial Introduction"
        assert chapters["ch5"] == "Pointers and Arrays"
        assert chapters["ch6"] == "Structures"
        assert chapters["ch8"] == "The UNIX System Interface"

    def test_sections_have_content(self, sections):
        for s in sections:
            assert len(s.content) > 0, f"Empty content for {s.section_path}"

    def test_section_parent_paths(self, sections):
        for s in sections:
            if s.section_type == "section":
                assert s.parent_section_path is not None, f"No parent for {s.section_path}"

    def test_total_sections_reasonable(self, sections):
        # Should have 100+ sections (chapters + appendices + their subsections)
        assert len(sections) > 100
