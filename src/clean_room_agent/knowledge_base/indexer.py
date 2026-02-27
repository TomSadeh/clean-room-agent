"""Knowledge base indexer — orchestrates parsing and DB insertion."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import time
from pathlib import Path

from clean_room_agent.db.connection import get_connection
from clean_room_agent.db.queries import (
    delete_bridge_files_for_source,
    delete_ref_sections_for_source,
    insert_ref_section,
    insert_ref_section_metadata,
    upsert_file,
    upsert_file_metadata,
    upsert_ref_source,
    upsert_repo,
)
from clean_room_agent.knowledge_base.metadata import extract_metadata
from clean_room_agent.knowledge_base.models import RefSection

logger = logging.getLogger(__name__)


# Source registry: maps source name → parser config
SOURCE_REGISTRY: dict[str, dict] = {
    "knr2": {
        "format": "text",
        "type": "book",
        "file": "knr2_clean.txt",
    },
    "cert_c": {
        "format": "markdown",
        "type": "coding_standard",
        "file": "cert_c_rules_index.md",
    },
    "crafting_interpreters": {
        "format": "html",
        "type": "book",
        "multi_file": True,
    },
    "cppreference": {
        "format": "html",
        "type": "api_reference",
        "subdir": "reference/en/c",
    },
    "modern_c": {
        "format": "pdf",
        "type": "book",
        "file": "modernC.pdf",
    },
    "beej_c": {
        "format": "pdf",
        "type": "guide",
        "file": "bgc.pdf",
    },
    "beej_network": {
        "format": "pdf",
        "type": "guide",
        "file": "bgnet.pdf",
    },
    "ostep": {
        "format": "pdf",
        "type": "book",
        "multi_file": True,
    },
}


@dataclasses.dataclass
class KBIndexResult:
    """Result of indexing the knowledge base."""

    sources_indexed: int = 0
    sections_total: int = 0
    bridge_files_created: int = 0
    errors: list[str] = dataclasses.field(default_factory=list)
    duration_ms: int = 0


def index_knowledge_base(
    kb_path: Path,
    repo_path: Path,
    sources: list[str] | None = None,
    continue_on_error: bool = False,
) -> KBIndexResult:
    """Index knowledge base reference sources into curated DB.

    Args:
        kb_path: Path to knowledge_base/c_references/ directory.
        repo_path: Repository root (for DB connection).
        sources: Optional list of source names to index. Defaults to all.
        continue_on_error: If False (default), re-raise parse/insert errors with
            context. If True, log with traceback and continue.
    """
    start = time.monotonic()
    result = KBIndexResult()

    source_names = sources or list(SOURCE_REGISTRY.keys())

    for name in source_names:
        if name not in SOURCE_REGISTRY:
            result.errors.append(f"Unknown source: {name}")
            continue

        config = SOURCE_REGISTRY[name]
        source_dir = kb_path / name

        if not source_dir.exists():
            result.errors.append(f"Source directory not found: {source_dir}")
            continue

        try:
            sections = _parse_source(name, config, source_dir)
        except Exception as e:
            if not continue_on_error:
                raise RuntimeError(f"Failed to parse source {name}: {e}") from e
            logger.exception("Failed to parse source %s", name)
            result.errors.append(f"Parse error for {name}: {e}")
            continue

        if not sections:
            logger.info("No sections parsed for source %s", name)
            continue

        try:
            n_sections, n_bridge = _insert_source(
                name, config, sections, kb_path, repo_path,
            )
            result.sources_indexed += 1
            result.sections_total += n_sections
            result.bridge_files_created += n_bridge
            logger.info(
                "Indexed %s: %d sections, %d bridge files",
                name, n_sections, n_bridge,
            )
        except Exception as e:
            if not continue_on_error:
                raise RuntimeError(f"Failed to insert source {name}: {e}") from e
            logger.exception("Failed to insert source %s", name)
            result.errors.append(f"Insert error for {name}: {e}")

    result.duration_ms = int((time.monotonic() - start) * 1000)
    return result


def _parse_source(
    name: str,
    config: dict,
    source_dir: Path,
) -> list[RefSection]:
    """Parse a source using the appropriate parser."""
    fmt = config["format"]

    if fmt == "text":
        from clean_room_agent.knowledge_base.text_parser import parse_knr2

        file_path = source_dir / config["file"]
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return parse_knr2(text)

    if fmt == "markdown":
        from clean_room_agent.knowledge_base.markdown_parser import parse_cert_c

        file_path = source_dir / config["file"]
        text = file_path.read_text(encoding="utf-8")
        return parse_cert_c(text)

    if fmt == "html":
        if name == "crafting_interpreters":
            from clean_room_agent.knowledge_base.html_parser import (
                parse_crafting_interpreters,
            )
            return parse_crafting_interpreters(source_dir)

        if name == "cppreference":
            from clean_room_agent.knowledge_base.html_parser import (
                parse_cppreference,
            )
            subdir = config.get("subdir", "")
            ref_dir = source_dir / subdir if subdir else source_dir
            return parse_cppreference(ref_dir)

    if fmt == "pdf":
        return _parse_pdf_source(name, config, source_dir)

    raise ValueError(f"Unknown format {fmt!r} for source {name!r}")


def _parse_pdf_source(
    name: str,
    config: dict,
    source_dir: Path,
) -> list[RefSection]:
    """Parse PDF sources by converting to markdown first.

    Requires opendataloader-pdf to be installed.
    """
    import subprocess
    import sys
    import tempfile

    from clean_room_agent.knowledge_base.markdown_parser import parse_pdf_markdown

    all_sections: list[RefSection] = []

    if config.get("multi_file"):
        pdf_files = sorted(source_dir.glob("*.pdf"))
    else:
        pdf_files = [source_dir / config["file"]]

    for pdf_file in pdf_files:
        if not pdf_file.exists():
            logger.warning("PDF file not found: %s", pdf_file)
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                subprocess.run(
                    [
                        sys.executable, "-m", "opendataloader_pdf",
                        str(pdf_file),
                        "--format", "markdown",
                        "--output-dir", tmpdir,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
            except FileNotFoundError:
                raise RuntimeError(
                    "opendataloader-pdf not installed. "
                    "Install it to index PDF sources: pip install opendataloader-pdf"
                )
            except subprocess.CalledProcessError as e:
                logger.warning(
                    "opendataloader-pdf failed for %s: %s",
                    pdf_file.name, e.stderr.decode(errors="replace")[:500],
                )
                continue

            # Find the output markdown file
            md_files = list(Path(tmpdir).glob("*.md"))
            if not md_files:
                logger.warning("No markdown output from opendataloader-pdf for %s", pdf_file.name)
                continue

            md_text = md_files[0].read_text(encoding="utf-8", errors="replace")
            sections = parse_pdf_markdown(md_text, source_file=pdf_file.name)
            all_sections.extend(sections)

    return all_sections


def _insert_source(
    name: str,
    config: dict,
    sections: list[RefSection],
    kb_path: Path,
    repo_path: Path,
) -> tuple[int, int]:
    """Insert parsed sections into curated DB. Returns (section_count, bridge_count)."""
    conn = get_connection("curated", repo_path=repo_path)
    try:
        # Upsert the reference source
        source_id = upsert_ref_source(
            conn,
            name=name,
            source_type=config["type"],
            path=str(kb_path / name),
            format_=config["format"],
        )

        # Delete existing sections for this source (clean re-index)
        delete_ref_sections_for_source(conn, source_id)

        # Also delete existing bridge files for this source
        delete_bridge_files_for_source(conn, name)

        # Get or create repo record for bridge files
        repo_id = upsert_repo(conn, str(repo_path), None)

        # Build parent section_path → section_id mapping
        section_id_map: dict[str, int] = {}
        n_bridge = 0

        for section in sections:
            content_hash = hashlib.sha256(section.content.encode()).hexdigest()[:16]
            parent_id = section_id_map.get(section.parent_section_path)

            section_id = insert_ref_section(
                conn,
                source_id=source_id,
                title=section.title,
                section_path=section.section_path,
                content=section.content,
                content_hash=content_hash,
                size_bytes=len(section.content.encode()),
                section_type=section.section_type,
                ordering=section.ordering,
                parent_section_id=parent_id,
                source_file=section.source_file,
                start_line=section.start_line,
                end_line=section.end_line,
            )
            section_id_map[section.section_path] = section_id

            # Extract and insert metadata
            meta = extract_metadata(section.title, section.content)
            insert_ref_section_metadata(
                conn,
                section_id=section_id,
                domain=meta.domain,
                concepts=meta.concepts,
                c_standard=meta.c_standard,
                header=meta.header,
                related_functions=meta.related_functions,
            )

            # Create bridge file in files table
            virtual_path = f"kb/{name}/{section.section_path}"
            file_id = upsert_file(
                conn,
                repo_id=repo_id,
                path=virtual_path,
                language="text",
                content_hash=content_hash,
                size_bytes=len(section.content.encode()),
                file_source="knowledge_base",
            )

            # Create bridge file_metadata
            upsert_file_metadata(
                conn,
                file_id=file_id,
                purpose=section.title,
                module=f"kb/{name}",
                domain=meta.domain,
                concepts=meta.concepts,
            )
            n_bridge += 1

        conn.commit()
        return len(sections), n_bridge
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
