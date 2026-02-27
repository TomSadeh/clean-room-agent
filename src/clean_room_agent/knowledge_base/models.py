"""Data models for knowledge base reference sections."""

from __future__ import annotations

import dataclasses
import re


@dataclasses.dataclass
class RefSource:
    """A reference source (book, guide, API reference)."""

    name: str
    source_type: str  # book, guide, api_reference, coding_standard
    path: str
    format: str  # text, pdf, html, markdown


@dataclasses.dataclass
class RefSection:
    """A parsed section from a reference source."""

    title: str
    section_path: str  # e.g. "ch1/1.1_getting_started"
    content: str
    section_type: str  # chapter, section, appendix, rule, function_ref
    ordering: int
    parent_section_path: str | None = None
    source_file: str | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclasses.dataclass
class RefSectionMeta:
    """Deterministic metadata extracted from a section."""

    domain: str | None = None
    concepts: str | None = None  # comma-separated
    c_standard: str | None = None  # e.g. "C89", "C99", "C11"
    header: str | None = None  # e.g. "<stdlib.h>"
    related_functions: str | None = None  # comma-separated


def title_to_path(title: str) -> str:
    """Convert a heading title to a URL-safe path segment."""
    # Lowercase, replace spaces/special chars with underscores
    path = re.sub(r"[^a-z0-9]+", "_", title.lower())
    # Strip leading/trailing underscores, collapse runs
    path = re.sub(r"_+", "_", path).strip("_")
    return path[:60]  # Limit length
