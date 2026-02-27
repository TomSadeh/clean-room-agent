"""HTML parser for Crafting Interpreters chapters and cppreference pages."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from clean_room_agent.knowledge_base.models import RefSection, title_to_path


# ============================================================
# Crafting Interpreters parser
# ============================================================

class _CraftingInterpretersParser(HTMLParser):
    """Extract chapter content from Crafting Interpreters HTML pages."""

    def __init__(self):
        super().__init__()
        self.chapter_number: str = ""
        self.chapter_title: str = ""
        self.sections: list[dict] = []  # {title, id, content, level}
        self._current_section: dict | None = None
        self._in_article = False
        self._in_heading: int = 0  # 0=no, 1/2/3=h level
        self._in_code = False
        self._nav_depth = 0
        self._aside_depth = 0
        self._in_number_div = False
        self._text_buf: list[str] = []
        self._heading_buf: list[str] = []

    @property
    def _skipping(self) -> bool:
        return self._nav_depth > 0 or self._aside_depth > 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        attr_dict = dict(attrs)

        if tag == "nav":
            self._nav_depth += 1
            return
        if tag == "aside":
            self._aside_depth += 1
            return

        if self._skipping:
            return

        if tag == "article" and "chapter" in attr_dict.get("class", ""):
            self._in_article = True
            return

        if not self._in_article:
            return

        # Chapter number div
        if tag == "div" and attr_dict.get("class") == "number":
            self._in_number_div = True
            return

        # Headings
        if tag in ("h1", "h2", "h3"):
            level = int(tag[1])
            self._in_heading = level
            self._heading_buf = []
            return

        # Code blocks
        if tag == "div" and "codehilite" in attr_dict.get("class", ""):
            self._in_code = True
            self._text_buf.append("\n```\n")
            return

        if tag == "pre" and self._in_code:
            return

        if tag == "p":
            self._text_buf.append("\n")

        if tag == "br":
            self._text_buf.append("\n")

    def handle_endtag(self, tag: str):
        if tag == "nav" and self._nav_depth > 0:
            self._nav_depth -= 1
            return
        if tag == "aside" and self._aside_depth > 0:
            self._aside_depth -= 1
            return

        if self._skipping:
            return

        if not self._in_article:
            return

        if tag == "div" and self._in_number_div:
            self._in_number_div = False
            return

        if tag in ("h1", "h2", "h3") and self._in_heading:
            title = "".join(self._heading_buf).strip()
            level = self._in_heading
            self._in_heading = 0

            if level == 1:
                self.chapter_title = title
                self._flush_section()
                self._current_section = {
                    "title": title,
                    "level": 1,
                    "id": "",
                }
                self._text_buf = []
            else:
                self._flush_section()
                self._current_section = {
                    "title": title,
                    "level": level,
                    "id": "",
                }
                self._text_buf = []
            return

        if tag == "pre" and self._in_code:
            self._text_buf.append("\n```\n")
            self._in_code = False
            return

        if tag == "article":
            self._flush_section()
            self._in_article = False

    def handle_data(self, data: str):
        if self._skipping:
            return

        if self._in_number_div:
            self.chapter_number = data.strip()
            return

        if self._in_heading:
            self._heading_buf.append(data)
            return

        if self._in_article and self._current_section is not None:
            self._text_buf.append(data)

    def handle_entityref(self, name: str):
        char = {
            "rsquo": "'", "lsquo": "'", "rdquo": '"', "ldquo": '"',
            "mdash": "—", "ndash": "–", "amp": "&", "lt": "<",
            "gt": ">", "nbsp": " ", "rarr": "→", "larr": "←",
        }.get(name, f"&{name};")
        if self._skipping:
            return
        if self._in_heading:
            self._heading_buf.append(char)
        elif self._in_article and self._current_section is not None:
            self._text_buf.append(char)

    def handle_charref(self, name: str):
        try:
            char = chr(int(name, 16) if name.startswith("x") else int(name))
        except (ValueError, OverflowError):
            char = f"&#{name};"
        if self._skipping:
            return
        if self._in_heading:
            self._heading_buf.append(char)
        elif self._in_article and self._current_section is not None:
            self._text_buf.append(char)

    def _flush_section(self):
        if self._current_section is not None:
            content = "".join(self._text_buf).strip()
            if content:
                self._current_section["content"] = content
                self.sections.append(self._current_section)
            self._current_section = None
            self._text_buf = []


def parse_crafting_interpreters(html_dir: Path) -> list[RefSection]:
    """Parse all Crafting Interpreters HTML files in a directory."""
    sections: list[RefSection] = []
    ordering = 0

    # Sort HTML files by chapter number from filename
    html_files = sorted(html_dir.glob("*.html"))

    for html_file in html_files:
        try:
            text = html_file.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise RuntimeError(
                f"Failed to read KB source file {html_file}: {e}"
            ) from e

        parser = _CraftingInterpretersParser()
        parser.feed(text)

        if not parser.chapter_title:
            continue

        ch_num = parser.chapter_number or str(ordering)
        ch_path = f"ch{ch_num}"
        filename = html_file.name

        # Create chapter-level section
        intro_content = ""
        sub_sections: list[dict] = []

        for sec in parser.sections:
            if sec["level"] == 1:
                intro_content = sec.get("content", "")
            else:
                sub_sections.append(sec)

        if intro_content or sub_sections:
            sections.append(RefSection(
                title=parser.chapter_title,
                section_path=ch_path,
                content=intro_content,
                section_type="chapter",
                ordering=ordering,
                source_file=filename,
            ))
            ordering += 1

        for i, sec in enumerate(sub_sections):
            sec_path = title_to_path(sec["title"])
            sections.append(RefSection(
                title=sec["title"],
                section_path=f"{ch_path}/{sec_path}",
                content=sec.get("content", ""),
                section_type="section",
                ordering=ordering,
                parent_section_path=ch_path,
                source_file=filename,
            ))
            ordering += 1

    return sections


# ============================================================
# cppreference parser
# ============================================================

class _CpprefParser(HTMLParser):
    """Extract function/type documentation from cppreference HTML pages."""

    def __init__(self):
        super().__init__()
        self.title: str = ""
        self.content_parts: list[str] = []
        self._in_first_heading = False
        self._in_content = False
        self._in_code = False
        self._in_nav = False
        self._depth = 0
        self._heading_level: int = 0
        self._heading_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        attr_dict = dict(attrs)

        # Skip navbar content
        if attr_dict.get("class", "").startswith("t-navbar"):
            self._in_nav = True
            self._depth = 1
            return
        if self._in_nav:
            self._depth += 1
            return

        # First heading
        if tag == "h1" and attr_dict.get("id") == "firstHeading":
            self._in_first_heading = True
            return

        # Content area
        if tag == "div" and attr_dict.get("id") == "mw-content-text":
            self._in_content = True
            return

        if not self._in_content:
            return

        # Section headings
        if tag in ("h2", "h3", "h4"):
            self._heading_level = int(tag[1])
            self._heading_buf = []
            return

        # Code blocks
        cls = attr_dict.get("class", "")
        if tag == "div" and "source-c" in cls:
            self._in_code = True
            self.content_parts.append("\n```c\n")
            return

        if tag == "p":
            self.content_parts.append("\n")

        if tag == "br":
            self.content_parts.append("\n")

        if tag == "li":
            self.content_parts.append("\n- ")

        if tag == "td":
            self.content_parts.append(" | ")

        if tag == "tr":
            self.content_parts.append("\n")

    def handle_endtag(self, tag: str):
        if self._in_nav:
            self._depth -= 1
            if self._depth <= 0:
                self._in_nav = False
            return

        if tag == "h1" and self._in_first_heading:
            self._in_first_heading = False
            return

        if not self._in_content:
            return

        if tag in ("h2", "h3", "h4") and self._heading_level:
            heading = "".join(self._heading_buf).strip()
            if heading and heading != "[edit]":
                prefix = "#" * self._heading_level
                self.content_parts.append(f"\n{prefix} {heading}\n")
            self._heading_level = 0
            return

        if tag == "pre" and self._in_code:
            self.content_parts.append("\n```\n")
            self._in_code = False

    def handle_data(self, data: str):
        if self._in_nav:
            return

        if self._in_first_heading:
            self.title = data.strip()
            return

        if self._heading_level:
            self._heading_buf.append(data)
            return

        if self._in_content:
            self.content_parts.append(data)

    def handle_entityref(self, name: str):
        char = {
            "amp": "&", "lt": "<", "gt": ">", "nbsp": " ",
            "rsquo": "'", "lsquo": "'", "rdquo": '"', "ldquo": '"',
        }.get(name, f"&{name};")
        if self._in_first_heading:
            self.title += char
        elif self._in_content:
            self.content_parts.append(char)

    def handle_charref(self, name: str):
        try:
            char = chr(int(name, 16) if name.startswith("x") else int(name))
        except (ValueError, OverflowError):
            char = f"&#{name};"
        if self._in_first_heading:
            self.title += char
        elif self._in_content:
            self.content_parts.append(char)


def parse_cppreference(ref_dir: Path) -> list[RefSection]:
    """Parse cppreference C reference HTML pages.

    Scans the reference/en/c/ subtree for HTML files.
    Each file becomes a single RefSection (function/type documentation).
    """
    sections: list[RefSection] = []
    ordering = 0

    c_dir = ref_dir
    if not c_dir.is_dir():
        return sections

    html_files = sorted(c_dir.rglob("*.html"))

    for html_file in html_files:
        try:
            text = html_file.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise RuntimeError(
                f"Failed to read KB source file {html_file}: {e}"
            ) from e

        parser = _CpprefParser()
        parser.feed(text)

        if not parser.title:
            continue

        content = "".join(parser.content_parts).strip()
        if not content:
            continue

        # Build section path from relative path
        rel = html_file.relative_to(c_dir)
        # Remove .html extension for path
        section_path = str(rel.with_suffix("")).replace("\\", "/")

        # Determine parent from directory
        parent = str(rel.parent).replace("\\", "/")
        parent_path = parent if parent != "." else None

        sections.append(RefSection(
            title=parser.title,
            section_path=section_path,
            content=content,
            section_type="function_ref",
            ordering=ordering,
            parent_section_path=parent_path,
            source_file=html_file.name,
        ))
        ordering += 1

    return sections
