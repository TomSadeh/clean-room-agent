"""Tests for deterministic metadata extraction."""

from clean_room_agent.knowledge_base.metadata import extract_metadata


class TestExtractMetadata:
    """Tests for extract_metadata()."""

    def test_detects_pointer_domain(self):
        meta = extract_metadata("Pointers and Addresses", "pointer dereference * indirection")
        assert meta.domain == "pointers"

    def test_detects_memory_domain(self):
        meta = extract_metadata("Memory Management", "malloc free calloc realloc heap")
        assert meta.domain == "memory_management"

    def test_detects_io_domain(self):
        meta = extract_metadata("File Access", "fopen fclose fread fwrite stream")
        assert meta.domain == "io"

    def test_detects_concurrency_domain(self):
        meta = extract_metadata("Threads", "mutex lock deadlock thread race condition")
        assert meta.domain == "concurrency"

    def test_extracts_concepts(self):
        meta = extract_metadata("Pointer Arrays", "pointer address array dereference")
        assert meta.concepts is not None
        concepts = meta.concepts.split(",")
        assert "pointer" in concepts
        assert "address" in concepts

    def test_extracts_headers(self):
        meta = extract_metadata("IO", "Use <stdio.h> and <stdlib.h> for I/O")
        assert meta.header is not None
        assert "<stdio.h>" in meta.header
        assert "<stdlib.h>" in meta.header

    def test_extracts_functions(self):
        meta = extract_metadata("Memory", "Call malloc(size) then free(ptr) when done.")
        assert meta.related_functions is not None
        funcs = meta.related_functions.split(",")
        assert "malloc" in funcs
        assert "free" in funcs

    def test_detects_c_standard(self):
        meta = extract_metadata("C11 Features", "The C11 standard introduced _Atomic and _Generic")
        assert meta.c_standard is not None
        assert "C11" in meta.c_standard

    def test_no_domain_for_generic_text(self):
        meta = extract_metadata("Preface", "This book is about programming.")
        # Should still return something (or None), not crash
        assert isinstance(meta.domain, (str, type(None)))

    def test_empty_content(self):
        meta = extract_metadata("", "")
        assert meta.domain is None or isinstance(meta.domain, str)
