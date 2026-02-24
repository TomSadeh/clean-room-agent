"""Preflight checks before pipeline execution."""

import logging
from pathlib import Path

from clean_room_agent.config import require_models_config
from clean_room_agent.db.connection import get_connection

logger = logging.getLogger(__name__)

CODE_MODES = ("plan", "implement")
TRAINING_MODES = ("train_plan", "curate_data")


def run_preflight_checks(config: dict | None, repo_path: Path, mode: str) -> None:
    """Run all preflight checks. Raises on failure.

    Checks:
    1. Model config exists
    2. Curated DB exists + files table populated (code modes)
    3. Raw DB exists + task_runs populated (training modes)
    4. Enrichment status (info log if 0)
    """
    # 1. Model config
    require_models_config(config)

    # 2. Curated DB (code modes)
    if mode in CODE_MODES:
        try:
            conn = get_connection("curated", repo_path=repo_path, read_only=True)
        except FileNotFoundError:
            raise RuntimeError(
                f"Curated DB not found at {repo_path / '.clean_room' / 'curated.sqlite'}. "
                "Run 'cra index' first."
            )
        try:
            count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            if count == 0:
                raise RuntimeError(
                    "Curated DB has no indexed files. Run 'cra index' first."
                )

            # 4. Enrichment status (informational)
            meta_count = conn.execute("SELECT COUNT(*) FROM file_metadata").fetchone()[0]
            if meta_count == 0:
                logger.info(
                    "No enrichment data found. Tier 4 (metadata) retrieval will be skipped. "
                    "Run 'cra enrich --promote' to enable metadata-based retrieval."
                )
        finally:
            conn.close()

    # 3. Raw DB (training modes)
    if mode in TRAINING_MODES:
        try:
            conn = get_connection("raw", repo_path=repo_path, read_only=True)
        except FileNotFoundError:
            raise RuntimeError(
                f"Raw DB not found at {repo_path / '.clean_room' / 'raw.sqlite'}. "
                "Need logged task runs for training modes."
            )
        try:
            count = conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0]
            if count == 0:
                raise RuntimeError(
                    "Raw DB has no task runs. Need logged activity for training modes."
                )
        finally:
            conn.close()
