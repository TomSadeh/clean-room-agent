"""Per-file LLM enrichment pipeline."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from clean_room_agent.db.connection import get_connection
from clean_room_agent.db.queries import upsert_file_metadata
from clean_room_agent.db.raw_queries import insert_enrichment_output
from clean_room_agent.llm.client import LLMClient
from clean_room_agent.llm.router import ModelRouter

logger = logging.getLogger(__name__)

ENRICHMENT_SYSTEM = (
    "You are a code analysis assistant. Analyze the given source file and produce a JSON object "
    "with exactly these keys: purpose, module, domain, concepts, public_api_surface, complexity_notes. "
    "- purpose: one-sentence description of what this file does\n"
    "- module: the logical module/subsystem this file belongs to\n"
    "- domain: the domain area (e.g. 'authentication', 'database', 'ui', 'testing')\n"
    "- concepts: JSON array of key concepts/patterns used\n"
    "- public_api_surface: JSON array of public function/class names\n"
    "- complexity_notes: brief notes on complexity or tricky parts\n"
    "Respond with ONLY the JSON object, no markdown fencing."
)


@dataclass
class EnrichmentResult:
    files_enriched: int
    files_skipped: int
    files_promoted: int


def enrich_repository(
    repo_path: Path, models_config: dict, promote: bool = False
) -> EnrichmentResult:
    """Run LLM enrichment on all indexed files.

    Args:
        repo_path: Root of the target repository.
        models_config: The [models] config dict (caller validates via require_models_config).
        promote: If True, copy enrichment data to curated DB.
    """
    router = ModelRouter(models_config)
    model_config = router.resolve("reasoning")
    client = None
    curated_conn = None
    raw_conn = None

    try:
        client = LLMClient(model_config)
        curated_conn = get_connection("curated", repo_path=repo_path, read_only=not promote)
        raw_conn = get_connection("raw", repo_path=repo_path)
        # Scope to the repo at this path (multi-repo safe)
        repo_row = curated_conn.execute(
            "SELECT id FROM repos WHERE path = ?", (str(repo_path),)
        ).fetchone()
        if not repo_row:
            raise RuntimeError(
                f"No indexed repo found at {repo_path}. Run 'cra index' first."
            )
        repo_id = repo_row["id"]
        files = curated_conn.execute(
            "SELECT * FROM files WHERE repo_id = ?", (repo_id,)
        ).fetchall()
        enriched = 0
        skipped = 0
        promoted = 0

        for file_row in files:
            file_id = file_row["id"]
            file_path = file_row["path"]

            # Skip already enriched files.
            # NOTE: file_id is from curated DB; not stable across curated DB rebuilds.
            # A full fix requires storing file_path in enrichment_outputs schema.
            existing = raw_conn.execute(
                "SELECT id FROM enrichment_outputs WHERE file_id = ?", (file_id,)
            ).fetchone()
            if existing:
                skipped += 1
                continue

            # Build enrichment prompt
            prompt = _build_prompt(file_path, file_id, curated_conn, repo_path)

            try:
                response = client.complete(prompt, system=ENRICHMENT_SYSTEM)
                parsed = _parse_enrichment_response(response.text)
            except Exception as e:
                logger.error("Enrichment failed for %s: %s", file_path, e)
                raise

            # Write to raw DB
            insert_enrichment_output(
                raw_conn,
                file_id=file_id,
                model=model_config.model,
                raw_prompt=prompt,
                raw_response=response.text,
                purpose=parsed.get("purpose"),
                module=parsed.get("module"),
                domain=parsed.get("domain"),
                concepts=json.dumps(parsed.get("concepts", [])),
                public_api_surface=json.dumps(parsed.get("public_api_surface", [])),
                complexity_notes=parsed.get("complexity_notes"),
                promoted=promote,
            )
            raw_conn.commit()
            enriched += 1

            # Optionally promote to curated
            if promote:
                upsert_file_metadata(
                    curated_conn,
                    file_id=file_id,
                    purpose=parsed.get("purpose"),
                    module=parsed.get("module"),
                    domain=parsed.get("domain"),
                    concepts=json.dumps(parsed.get("concepts", [])),
                    public_api_surface=json.dumps(parsed.get("public_api_surface", [])),
                    complexity_notes=parsed.get("complexity_notes"),
                )
                curated_conn.commit()
                promoted += 1

            logger.info("Enriched %s", file_path)

        return EnrichmentResult(
            files_enriched=enriched,
            files_skipped=skipped,
            files_promoted=promoted,
        )
    finally:
        if client is not None:
            client.close()
        if curated_conn is not None:
            curated_conn.close()
        if raw_conn is not None:
            raw_conn.close()


def _build_prompt(
    file_path: str,
    file_id: int,
    curated_conn,
    repo_path: Path,
) -> str:
    """Build the enrichment prompt for a file."""
    parts = [f"File: {file_path}\n"]

    # Add symbol summary
    symbols = curated_conn.execute(
        "SELECT name, kind, signature FROM symbols WHERE file_id = ?", (file_id,)
    ).fetchall()
    if symbols:
        parts.append("Symbols:")
        for s in symbols:
            sig = f" — {s['signature']}" if s["signature"] else ""
            parts.append(f"  {s['kind']} {s['name']}{sig}")
        parts.append("")

    # Add docstring summary
    docstrings = curated_conn.execute(
        "SELECT content FROM docstrings WHERE file_id = ? ORDER BY id LIMIT 3", (file_id,)
    ).fetchall()
    if docstrings:
        parts.append("Docstrings:")
        for d in docstrings:
            preview = d["content"][:200]
            parts.append(f"  {preview}")
        parts.append("")

    # Add source preview (first 100 lines)
    abs_path = repo_path / file_path
    if abs_path.exists():
        source = abs_path.read_text(encoding="utf-8", errors="replace")
        lines = source.split("\n")[:100]
        parts.append("Source preview (first 100 lines):")
        parts.append("\n".join(lines))

    return "\n".join(parts)


def _parse_enrichment_response(text: str) -> dict:
    """Parse JSON from the LLM response."""
    text = text.strip()
    # Strip markdown code fencing if present (only opening and closing fences)
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening fence (first line)
        lines = lines[1:]
        # Remove closing fence (last non-empty line)
        while lines and lines[-1].strip() in ("```", ""):
            if lines[-1].strip() == "```":
                lines.pop()
                break
            lines.pop()
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse enrichment JSON: {e}\nRaw response: {text}") from e
