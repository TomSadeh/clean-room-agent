"""Tests for markdown parser (CERT C and PDF-converted)."""

from pathlib import Path

import pytest

from clean_room_agent.knowledge_base.markdown_parser import (
    parse_cert_c,
    parse_pdf_markdown,
)

_CERT_C_PATH = (
    Path(__file__).resolve().parents[2]
    / "knowledge_base" / "c_references" / "cert_c" / "cert_c_rules_index.md"
)


class TestParseCertCSynthetic:
    """Unit tests with synthetic CERT C markdown."""

    def test_parses_category_and_rules(self):
        text = (
            "# Header\n\n"
            "## MEM - Memory Management\n"
            "- MEM30-C: Do not access freed memory\n"
            "- MEM31-C: Free dynamically allocated memory when no longer needed\n"
        )
        sections = parse_cert_c(text)
        categories = [s for s in sections if s.section_type == "section"]
        rules = [s for s in sections if s.section_type == "rule"]
        assert len(categories) == 1
        assert categories[0].title == "MEM - Memory Management"
        assert categories[0].section_path == "mem"
        assert len(rules) == 2
        assert rules[0].section_path == "mem/MEM30-C"
        assert rules[0].parent_section_path == "mem"

    def test_multiple_categories(self):
        text = (
            "## PRE - Preprocessor\n"
            "- PRE30-C: Rule one\n\n"
            "## DCL - Declarations\n"
            "- DCL30-C: Rule two\n"
        )
        sections = parse_cert_c(text)
        categories = [s for s in sections if s.section_type == "section"]
        assert len(categories) == 2
        assert categories[0].section_path == "pre"
        assert categories[1].section_path == "dcl"


@pytest.mark.skipif(not _CERT_C_PATH.exists(), reason="CERT C source not available")
class TestParseCertCReal:
    """Integration tests against real CERT C rules file."""

    @pytest.fixture(scope="class")
    def sections(self):
        text = _CERT_C_PATH.read_text(encoding="utf-8")
        return parse_cert_c(text)

    def test_has_categories(self, sections):
        categories = [s for s in sections if s.section_type == "section"]
        assert len(categories) >= 10

    def test_has_rules(self, sections):
        rules = [s for s in sections if s.section_type == "rule"]
        assert len(rules) >= 100

    def test_rule_ids_follow_pattern(self, sections):
        rules = [s for s in sections if s.section_type == "rule"]
        for r in rules:
            # Rule IDs should be like "mem/MEM30-C"
            parts = r.section_path.split("/")
            assert len(parts) == 2
            assert parts[1].endswith("-C")


class TestParsePdfMarkdown:
    """Unit tests for PDF-converted markdown parser."""

    def test_parses_headings(self):
        text = (
            "# Chapter One\n\nIntro text.\n\n"
            "## Section A\n\nSection A content.\n\n"
            "## Section B\n\nSection B content.\n"
        )
        sections = parse_pdf_markdown(text, source_file="test.pdf")
        assert len(sections) == 3
        assert sections[0].section_type == "chapter"
        assert sections[0].title == "Chapter One"
        assert sections[1].section_type == "section"
        assert sections[1].parent_section_path == "chapter_one"

    def test_nested_headings(self):
        text = (
            "# Top\n\nTop content.\n\n"
            "## Mid\n\nMid content.\n\n"
            "### Deep\n\nDeep content.\n"
        )
        sections = parse_pdf_markdown(text)
        assert len(sections) == 3
        deep = sections[2]
        assert deep.title == "Deep"
        assert "top" in deep.section_path

    def test_code_blocks_preserved(self):
        text = "# Code Example\n\n```c\nint x = 42;\n```\n"
        sections = parse_pdf_markdown(text)
        assert len(sections) == 1
        assert "int x = 42" in sections[0].content
