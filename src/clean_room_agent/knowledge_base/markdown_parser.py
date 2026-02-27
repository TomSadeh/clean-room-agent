"""Markdown parser — handles CERT C rules index and PDF-converted markdown."""

from __future__ import annotations

import re

from clean_room_agent.knowledge_base.models import RefSection


# CERT C rule pattern: "- RULEnn-C: description"
_CERT_RULE_RE = re.compile(r"^-\s+(\w+\d+-C)\s*:\s*(.*)")

# Markdown heading pattern
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


def parse_cert_c(text: str) -> list[RefSection]:
    """Parse CERT C rules index markdown into sections.

    Structure: ## Category headers → rule entries (- RULEnn-C: description)
    Each category becomes a section, each rule becomes a child section.
    """
    lines = text.split("\n")
    sections: list[RefSection] = []
    current_category: str | None = None
    current_category_path: str | None = None
    category_lines: list[str] = []
    category_start: int = 0
    ordering = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Category heading: ## PRE - Preprocessor
        m = _HEADING_RE.match(stripped)
        if m and m.group(1) == "##":
            # Flush previous category
            if current_category and category_lines:
                sections.append(RefSection(
                    title=current_category,
                    section_path=current_category_path,
                    content="\n".join(category_lines).strip(),
                    section_type="section",
                    ordering=ordering,
                    source_file="cert_c_rules_index.md",
                    start_line=category_start + 1,
                    end_line=i,
                ))
                ordering += 1

            title = m.group(2).strip()
            # Extract short code: "PRE - Preprocessor" → "PRE"
            parts = title.split(" - ", 1)
            code = parts[0].strip() if parts else title
            current_category = title
            current_category_path = code.lower()
            category_lines = [stripped]
            category_start = i
            continue

        # Rule entry: - PRE30-C: Do not create...
        rule_m = _CERT_RULE_RE.match(stripped)
        if rule_m and current_category_path:
            rule_id = rule_m.group(1)
            description = rule_m.group(2).strip()

            sections.append(RefSection(
                title=f"{rule_id}: {description}",
                section_path=f"{current_category_path}/{rule_id}",
                content=f"{rule_id}: {description}",
                section_type="rule",
                ordering=ordering,
                parent_section_path=current_category_path,
                source_file="cert_c_rules_index.md",
                start_line=i + 1,
                end_line=i + 1,
            ))
            ordering += 1

        if current_category:
            category_lines.append(stripped)

    # Flush last category
    if current_category and category_lines:
        sections.append(RefSection(
            title=current_category,
            section_path=current_category_path,
            content="\n".join(category_lines).strip(),
            section_type="section",
            ordering=ordering,
            source_file="cert_c_rules_index.md",
            start_line=category_start + 1,
            end_line=len(lines),
        ))

    return sections


def parse_pdf_markdown(text: str, source_file: str = "") -> list[RefSection]:
    """Parse markdown converted from PDF (via opendataloader-pdf).

    Splits on # / ## / ### headings. Code blocks (triple backticks) are
    included in the content of their enclosing section.
    """
    lines = text.split("\n")
    sections: list[RefSection] = []
    heading_stack: list[tuple[int, str, str]] = []  # (level, title, path)
    current_lines: list[str] = []
    current_heading: dict | None = None
    ordering = 0

    def _flush():
        nonlocal ordering
        if current_heading and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                parent_path = current_heading.get("parent_path")
                sections.append(RefSection(
                    title=current_heading["title"],
                    section_path=current_heading["path"],
                    content=content,
                    section_type=current_heading["type"],
                    ordering=ordering,
                    parent_section_path=parent_path,
                    source_file=source_file,
                    start_line=current_heading["start_line"],
                    end_line=current_heading["end_line"],
                ))
                ordering += 1

    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line.strip())
        if m:
            # Flush previous section
            if current_heading:
                current_heading["end_line"] = i
            _flush()
            current_lines = []

            level = len(m.group(1))
            title = m.group(2).strip()
            path = _title_to_path(title)

            # Pop stack to find parent
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            parent_path = heading_stack[-1][2] if heading_stack else None
            full_path = f"{parent_path}/{path}" if parent_path else path

            heading_stack.append((level, title, full_path))

            section_type = "chapter" if level == 1 else "section"

            current_heading = {
                "title": title,
                "path": full_path,
                "type": section_type,
                "parent_path": parent_path,
                "start_line": i + 1,
                "end_line": len(lines),
            }
        else:
            current_lines.append(line)

    # Flush last section
    if current_heading:
        current_heading["end_line"] = len(lines)
    _flush()

    return sections


def _title_to_path(title: str) -> str:
    """Convert a heading title to a URL-safe path segment."""
    # Lowercase, replace spaces/special chars with underscores
    path = re.sub(r"[^a-z0-9]+", "_", title.lower())
    # Strip leading/trailing underscores, collapse runs
    path = re.sub(r"_+", "_", path).strip("_")
    return path[:60]  # Limit length
