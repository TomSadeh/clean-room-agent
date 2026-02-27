"""K&R text file parser — splits knr2_clean.txt into chapters and sections."""

from __future__ import annotations

import re

from clean_room_agent.knowledge_base.models import RefSection


# Page header patterns to strip (OCR artifacts from the PDF scan)
_PAGE_HEADER_RE = re.compile(
    r"^(?:"
    r"\d+\s+[A-Z ]+$"               # "34 A TUTORIAL INTRODUCTION"
    r"|[A-Z ]+\s+\d+$"              # "SECTION 1.2  9"
    r"|CHAPTER\s+\d+$"              # "CHAPTER 1"
    r"|SECTION\s+\d+\.\d+$"         # "SECTION 1.2"
    r"|[A-Z][A-Z ]{3,}\d*$"         # "THE C PROGRAMMING LANGUAGE 3" or all-caps page headers
    r")"
)

# Body chapter heading: requires form feed prefix (page boundary)
# Matches "chapter N: Title", "chapter N. Title", "CHAPTERS: StfUCtUfeS"
_CHAPTER_RE = re.compile(
    r"^chapter\s*(\w+)\s*[:.]?\s*(.*)",
    re.IGNORECASE,
)

# Appendix heading: "appendix A: Title" or "Appendix A. Title"
_APPENDIX_RE = re.compile(
    r"^appendix\s+([A-C])\s*[:.]?\s*(.*)",
    re.IGNORECASE,
)

# Section heading in body text: "N.N Title" or "AN.N Title" (appendix subsections)
_SECTION_RE = re.compile(
    r"^([AB]?\d+\.\d+(?:\.\d+)?)\s+([A-Z].*)")

# TOC: find the start of body text (Preface)
_PREFACE_RE = re.compile(r"^\x0c?Preface\s*$")

# OCR correction: "CHAPTERS:" (no space) = "CHAPTER 6:", "chapter s:" (with space) = "chapter 8:"
_CHAPTERS_NO_SPACE_RE = re.compile(r"^chapters\s*:", re.IGNORECASE)

# Chapter number OCR corrections (applied after regex match)
_CHAPTER_NUMBER_FIXES: dict[str, str] = {
    "s": "8",        # "chapter s:" → chapter 8
    "S": "8",
}

# Known chapter titles (for OCR-garbled headings where title is lost)
_CHAPTER_TITLES: dict[str, str] = {
    "6": "Structures",
    "8": "The UNIX System Interface",
}


def parse_knr2(text: str) -> list[RefSection]:
    """Parse K&R 2nd edition text into structured sections.

    Returns a flat list of RefSections with parent_section_path set
    for sections within chapters/appendices.
    """
    lines = text.split("\n")

    # Skip the TOC — find where body text starts
    body_start = _find_body_start(lines)

    # First pass: identify all heading positions
    headings = _find_headings(lines, body_start)

    # Deduplicate: keep only the first occurrence of each section_path
    # (later ones are typically inline references that slipped through)
    seen: set[str] = set()
    deduped: list[dict] = []
    for h in headings:
        if h["section_path"] not in seen:
            seen.add(h["section_path"])
            deduped.append(h)
    headings = deduped

    # Second pass: extract content between headings
    sections = _extract_sections(lines, headings)

    return sections


def _find_body_start(lines: list[str]) -> int:
    """Find where the body text starts (after TOC, at Preface)."""
    for i, line in enumerate(lines):
        if _PREFACE_RE.match(line.lstrip("\x0c").strip()) and i > 50:
            return i
    return 350


def _find_headings(
    lines: list[str],
    body_start: int,
) -> list[dict]:
    """Identify all chapter, appendix, and section headings in the body text."""
    headings: list[dict] = []

    for i in range(body_start, len(lines)):
        raw_line = lines[i]
        has_formfeed = "\x0c" in raw_line
        stripped = raw_line.lstrip("\x0c").strip()
        if not stripped:
            continue

        # Previous line blank = potential heading boundary
        prev_blank = i > 0 and not lines[i - 1].strip()

        # Chapter/appendix headings must be at page boundaries (form feed)
        # or preceded by a blank line after an exercise
        if has_formfeed or prev_blank:
            # Special case: "CHAPTERS:" (no space) is OCR garble of "CHAPTER 6:"
            if _CHAPTERS_NO_SPACE_RE.match(stripped):
                title = _CHAPTER_TITLES.get("6", "Structures")
                headings.append({
                    "type": "chapter",
                    "number": "6",
                    "title": title,
                    "line": i,
                    "section_path": "ch6",
                })
                continue

            # Check for chapter heading
            m = _CHAPTER_RE.match(stripped)
            if m and _is_chapter_heading(m, stripped):
                num_raw = m.group(1)
                title = m.group(2).strip()
                num = _CHAPTER_NUMBER_FIXES.get(num_raw, num_raw)
                if not num.isdigit():
                    pass  # Skip
                else:
                    # Use known title if OCR garbled the title
                    if not title or len(title) < 3 or not title[0].isupper():
                        title = _CHAPTER_TITLES.get(num, title)
                    if title and len(title) >= 3:
                        headings.append({
                            "type": "chapter",
                            "number": num,
                            "title": title,
                            "line": i,
                            "section_path": f"ch{num}",
                        })
                        continue

            # Check for appendix heading
            m = _APPENDIX_RE.match(stripped)
            if m:
                letter = m.group(1).upper()
                title = m.group(2).strip()
                if title and len(title) >= 3 and title[0].isupper():
                    headings.append({
                        "type": "appendix",
                        "number": letter,
                        "title": title,
                        "line": i,
                        "section_path": f"app{letter}",
                    })
                    continue

        # Section headings don't require form feed — they appear inline
        m = _SECTION_RE.match(stripped)
        if m:
            sec_num = m.group(1)
            title = m.group(2).strip()

            # Must start with uppercase letter and be a reasonable title
            if not title or len(title) < 2:
                continue
            # Skip TOC entries (have page numbers at end)
            if re.search(r'\d+\s*$', title) and i < body_start + 200:
                continue

            # Determine parent from section numbering
            if sec_num.startswith("A"):
                parent_path = "appA"
            elif sec_num.startswith("B"):
                parent_path = "appB"
            else:
                parts = sec_num.split(".")
                parent_path = f"ch{parts[0]}"

            headings.append({
                "type": "section",
                "number": sec_num,
                "title": title,
                "line": i,
                "section_path": f"{parent_path}/{sec_num}",
                "parent_path": parent_path,
            })

    return headings


def _is_chapter_heading(m: re.Match, stripped: str) -> bool:
    """Determine if a chapter regex match is a real heading vs. inline reference."""
    title = m.group(2).strip()

    # No title at all → likely inline reference like "Chapter 2."
    if not title:
        return False

    # Title starts with lowercase or is a sentence continuation
    if title and title[0].islower():
        # Could be OCR garble (e.g., "StfUCtUfeS") — check if it's short enough
        if len(title) > 30:
            return False

    # Contains prose markers → inline reference
    # These must be specific enough not to match legitimate titles
    prose_markers = [
        "describes", "was concerned", "its first", "discusses",
        "copes with", "to accept", "is printf",
        "2 through 6", "2 and 3", "will necessarily",
    ]
    title_lower = title.lower()
    if any(marker in title_lower for marker in prose_markers):
        return False

    # Title is too long to be a real chapter title (>80 chars = prose)
    if len(title) > 80:
        return False

    # Has page number at end (TOC entry)
    if re.search(r'\s+\d+\s*$', title):
        return False

    return True


def _extract_sections(
    lines: list[str],
    headings: list[dict],
) -> list[RefSection]:
    """Extract content between consecutive headings."""
    sections: list[RefSection] = []

    for idx, heading in enumerate(headings):
        start_line = heading["line"]

        # Content runs from heading line to next heading (or end of file)
        if idx + 1 < len(headings):
            end_line = headings[idx + 1]["line"]
        else:
            end_line = len(lines)

        # Extract and clean content
        raw_content = "\n".join(lines[start_line:end_line])
        content = _clean_content(raw_content)

        if not content.strip():
            continue

        section_type = heading["type"]
        parent_path = heading.get("parent_path")

        sections.append(RefSection(
            title=heading["title"],
            section_path=heading["section_path"],
            content=content,
            section_type=section_type,
            ordering=len(sections),
            parent_section_path=parent_path,
            source_file="knr2_clean.txt",
            start_line=start_line + 1,  # 1-indexed
            end_line=end_line,
        ))

    return sections


def _clean_content(text: str) -> str:
    """Remove page headers, form feeds, and other noise from section content."""
    cleaned_lines: list[str] = []

    for line in text.split("\n"):
        # Strip form feeds
        line = line.replace("\x0c", "")

        stripped = line.strip()

        # Skip page headers
        if _PAGE_HEADER_RE.match(stripped):
            continue

        # Skip bare page numbers (just digits on a line)
        if stripped.isdigit():
            continue

        cleaned_lines.append(line)

    # Collapse multiple consecutive blank lines into one
    result: list[str] = []
    prev_blank = False
    for line in cleaned_lines:
        if not line.strip():
            if not prev_blank:
                result.append("")
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False

    return "\n".join(result).strip()
