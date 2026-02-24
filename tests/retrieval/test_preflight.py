"""Tests for preflight checks."""

import pytest

from clean_room_agent.db.connection import get_connection
from clean_room_agent.db.queries import upsert_file, upsert_file_metadata, upsert_repo
from clean_room_agent.db.raw_queries import insert_task_run
from clean_room_agent.retrieval.preflight import run_preflight_checks


def _make_config():
    return {
        "models": {
            "provider": "ollama",
            "coding": "qwen2.5-coder:3b",
            "reasoning": "qwen3:4b",
            "base_url": "http://localhost:11434",
        }
    }


class TestPreflightModelConfig:
    def test_missing_config_raises(self, tmp_repo):
        with pytest.raises(RuntimeError, match="No config file found"):
            run_preflight_checks(None, tmp_repo, "plan")

    def test_missing_models_section(self, tmp_repo):
        with pytest.raises(RuntimeError, match="Missing \\[models\\]"):
            run_preflight_checks({}, tmp_repo, "plan")


class TestPreflightCuratedDB:
    def test_missing_curated_db_raises(self, tmp_repo):
        with pytest.raises(RuntimeError, match="Curated DB not found"):
            run_preflight_checks(_make_config(), tmp_repo, "plan")

    def test_empty_curated_db_raises(self, tmp_repo):
        # Create curated DB with schema but no files
        conn = get_connection("curated", repo_path=tmp_repo)
        conn.close()
        with pytest.raises(RuntimeError, match="no indexed files"):
            run_preflight_checks(_make_config(), tmp_repo, "plan")

    def test_populated_curated_db_passes(self, tmp_repo):
        conn = get_connection("curated", repo_path=tmp_repo)
        repo_id = upsert_repo(conn, str(tmp_repo), None)
        upsert_file(conn, repo_id, "test.py", "python", "abc123", 100)
        conn.commit()
        conn.close()
        # Should not raise
        run_preflight_checks(_make_config(), tmp_repo, "plan")

    def test_implement_mode_checks_curated(self, tmp_repo):
        with pytest.raises(RuntimeError, match="Curated DB not found"):
            run_preflight_checks(_make_config(), tmp_repo, "implement")


class TestPreflightRawDB:
    def test_training_mode_missing_raw_raises(self, tmp_repo):
        # Create curated DB (not needed for training modes, but models config is)
        with pytest.raises(RuntimeError, match="Raw DB not found"):
            run_preflight_checks(_make_config(), tmp_repo, "train_plan")

    def test_training_mode_empty_task_runs_raises(self, tmp_repo):
        conn = get_connection("raw", repo_path=tmp_repo)
        conn.close()
        with pytest.raises(RuntimeError, match="no task runs"):
            run_preflight_checks(_make_config(), tmp_repo, "train_plan")

    def test_training_mode_with_task_runs_passes(self, tmp_repo):
        conn = get_connection("raw", repo_path=tmp_repo)
        insert_task_run(conn, "t1", "/repo", "plan", "model", 32768, 4096, "scope")
        conn.commit()
        conn.close()
        run_preflight_checks(_make_config(), tmp_repo, "train_plan")


class TestPreflightEnrichment:
    def test_no_enrichment_logs_info(self, tmp_repo, caplog):
        import logging
        conn = get_connection("curated", repo_path=tmp_repo)
        repo_id = upsert_repo(conn, str(tmp_repo), None)
        upsert_file(conn, repo_id, "test.py", "python", "abc", 100)
        conn.commit()
        conn.close()

        with caplog.at_level(logging.INFO):
            run_preflight_checks(_make_config(), tmp_repo, "plan")
        assert "No enrichment data found" in caplog.text

    def test_with_enrichment_no_warning(self, tmp_repo, caplog):
        import logging
        conn = get_connection("curated", repo_path=tmp_repo)
        repo_id = upsert_repo(conn, str(tmp_repo), None)
        fid = upsert_file(conn, repo_id, "test.py", "python", "abc", 100)
        upsert_file_metadata(conn, fid, purpose="test file")
        conn.commit()
        conn.close()

        with caplog.at_level(logging.INFO):
            run_preflight_checks(_make_config(), tmp_repo, "plan")
        assert "No enrichment data found" not in caplog.text


class TestPreflightNonCodeModes:
    def test_curate_data_mode(self, tmp_repo):
        conn = get_connection("raw", repo_path=tmp_repo)
        insert_task_run(conn, "t1", "/repo", "plan", "model", 32768, 4096, "scope")
        conn.commit()
        conn.close()
        # curate_data is training mode, should pass with task_runs
        run_preflight_checks(_make_config(), tmp_repo, "curate_data")
