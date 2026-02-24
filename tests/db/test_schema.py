"""Tests for db/schema.py."""

from clean_room_agent.db.connection import get_connection


def _table_names(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r["name"] for r in rows]


class TestCuratedSchema:
    def test_tables_created(self, curated_conn):
        tables = _table_names(curated_conn)
        expected = [
            "adapter_metadata", "co_changes", "commits", "dependencies",
            "docstrings", "file_commits", "file_metadata", "files",
            "inline_comments", "repos", "symbol_references", "symbols",
        ]
        for t in expected:
            assert t in tables, f"Missing table: {t}"

    def test_idempotent(self, curated_conn):
        from clean_room_agent.db.schema import create_curated_schema
        # Running again should not error
        create_curated_schema(curated_conn)


class TestRawSchema:
    def test_tables_created(self, raw_conn):
        tables = _table_names(raw_conn)
        expected = [
            "adapter_registry", "enrichment_outputs", "index_runs",
            "orchestrator_passes", "orchestrator_runs",
            "retrieval_decisions", "retrieval_llm_calls", "run_attempts",
            "session_archives", "task_runs", "training_datasets",
            "training_plans", "validation_results",
        ]
        for t in expected:
            assert t in tables, f"Missing table: {t}"


class TestSessionSchema:
    def test_kv_table(self, session_conn):
        tables = _table_names(session_conn)
        assert "kv" in tables
