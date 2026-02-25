"""Tests for Batch 0 audit fixes (A1-A5).

A1: orchestrator_passes.task_run_id nullable (P0)
A2: retrieval_llm_calls.thinking column (P1)
A3: orchestrator_runs.error_message column (P1)
A4: Error messages stored on orchestrator failure (P1)
A5: Synthetic task_run for documentation pass (P1)
"""

import sqlite3

import pytest

from clean_room_agent.db.raw_queries import (
    insert_orchestrator_pass,
    insert_orchestrator_run,
    insert_retrieval_llm_call,
    insert_task_run,
    update_orchestrator_run,
)
from clean_room_agent.db.schema import create_raw_schema


class TestA1NullableTaskRunId:
    """A1 (P0): orchestrator_passes.task_run_id accepts None."""

    def test_insert_pass_with_null_task_run_id(self, raw_conn):
        orch_id = insert_orchestrator_run(
            raw_conn, "a1-test", "/repo", "task desc",
        )
        raw_conn.commit()

        pass_id = insert_orchestrator_pass(
            raw_conn, orch_id, None, "documentation", 0,
            part_id="part-1",
        )
        raw_conn.commit()
        assert pass_id > 0

        row = raw_conn.execute(
            "SELECT * FROM orchestrator_passes WHERE id = ?", (pass_id,)
        ).fetchone()
        assert row["task_run_id"] is None
        assert row["pass_type"] == "documentation"
        assert row["part_id"] == "part-1"

    def test_insert_pass_with_valid_task_run_id_still_works(self, raw_conn):
        orch_id = insert_orchestrator_run(
            raw_conn, "a1-test2", "/repo", "task desc",
        )
        tr_id = insert_task_run(
            raw_conn, "a1-test2:meta_plan", "/repo", "plan",
            "model", 32768, 4096, "scope",
        )
        raw_conn.commit()

        pass_id = insert_orchestrator_pass(
            raw_conn, orch_id, tr_id, "meta_plan", 0,
        )
        raw_conn.commit()

        row = raw_conn.execute(
            "SELECT * FROM orchestrator_passes WHERE id = ?", (pass_id,)
        ).fetchone()
        assert row["task_run_id"] == tr_id


class TestA2ThinkingColumn:
    """A2 (P1): retrieval_llm_calls.thinking column round-trips."""

    def test_thinking_stored_and_retrieved(self, raw_conn):
        thinking_text = "Let me analyze this step by step..."
        call_id = insert_retrieval_llm_call(
            raw_conn, "a2-test", "task_analysis", "model",
            "prompt text", "response text", 100, 50, 500,
            system_prompt="system",
            thinking=thinking_text,
        )
        raw_conn.commit()

        row = raw_conn.execute(
            "SELECT * FROM retrieval_llm_calls WHERE id = ?", (call_id,)
        ).fetchone()
        assert row["thinking"] == thinking_text

    def test_thinking_null_by_default(self, raw_conn):
        call_id = insert_retrieval_llm_call(
            raw_conn, "a2-test2", "task_analysis", "model",
            "prompt", "response", 100, 50, 500,
        )
        raw_conn.commit()

        row = raw_conn.execute(
            "SELECT * FROM retrieval_llm_calls WHERE id = ?", (call_id,)
        ).fetchone()
        assert row["thinking"] is None

    def test_schema_migration_idempotent(self, raw_conn):
        """Running create_raw_schema twice doesn't raise."""
        create_raw_schema(raw_conn)
        # Should not raise — migration catches OperationalError


class TestA3ErrorMessageColumn:
    """A3 (P1): orchestrator_runs.error_message column round-trips."""

    def test_error_message_stored(self, raw_conn):
        run_id = insert_orchestrator_run(
            raw_conn, "a3-test", "/repo", "task desc",
        )
        raw_conn.commit()

        update_orchestrator_run(
            raw_conn, run_id,
            status="failed",
            error_message="ValueError: missing config key",
        )
        raw_conn.commit()

        row = raw_conn.execute(
            "SELECT * FROM orchestrator_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["error_message"] == "ValueError: missing config key"
        assert row["status"] == "failed"

    def test_error_message_null_on_success(self, raw_conn):
        run_id = insert_orchestrator_run(
            raw_conn, "a3-test2", "/repo", "task desc",
        )
        raw_conn.commit()

        update_orchestrator_run(raw_conn, run_id, status="complete")
        raw_conn.commit()

        row = raw_conn.execute(
            "SELECT * FROM orchestrator_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["error_message"] is None


class TestA4ErrorMessageInFinalizer:
    """A4 (P1): _finalize_orchestrator_run passes error_message."""

    def test_finalize_with_error(self, raw_conn):
        from clean_room_agent.orchestrator.runner import _finalize_orchestrator_run

        run_id = insert_orchestrator_run(
            raw_conn, "a4-test", "/repo", "task desc",
        )
        raw_conn.commit()

        _finalize_orchestrator_run(
            raw_conn, run_id, "failed",
            error_message="RuntimeError: something broke",
        )

        row = raw_conn.execute(
            "SELECT * FROM orchestrator_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["status"] == "failed"
        assert row["error_message"] == "RuntimeError: something broke"
        assert row["completed_at"] is not None

    def test_finalize_without_error(self, raw_conn):
        from clean_room_agent.orchestrator.runner import _finalize_orchestrator_run

        run_id = insert_orchestrator_run(
            raw_conn, "a4-test2", "/repo", "task desc",
        )
        raw_conn.commit()

        _finalize_orchestrator_run(raw_conn, run_id, "complete")

        row = raw_conn.execute(
            "SELECT * FROM orchestrator_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["status"] == "complete"
        assert row["error_message"] is None


class TestA5SyntheticDocTaskRun:
    """A5 (P1): Documentation pass creates a synthetic task_run."""

    def test_doc_pass_creates_task_run_and_links_pass(self, raw_conn):
        """Verify the pattern: insert_task_run -> insert_orchestrator_pass with task_run_id."""
        orch_id = insert_orchestrator_run(
            raw_conn, "a5-test", "/repo", "task desc",
        )
        raw_conn.commit()

        # Simulate what runner.py now does for doc pass
        doc_task_run_id = insert_task_run(
            raw_conn, "a5-test:doc:part-1", str("/repo"), "documentation",
            "qwen3:4b", 32768, 4096, "",
        )
        pass_id = insert_orchestrator_pass(
            raw_conn, orch_id, doc_task_run_id, "documentation", 5,
            part_id="part-1",
        )
        raw_conn.commit()

        # Verify the chain: orchestrator_pass -> task_run
        op = raw_conn.execute(
            "SELECT * FROM orchestrator_passes WHERE id = ?", (pass_id,)
        ).fetchone()
        assert op["task_run_id"] == doc_task_run_id
        assert op["pass_type"] == "documentation"

        tr = raw_conn.execute(
            "SELECT * FROM task_runs WHERE id = ?", (doc_task_run_id,)
        ).fetchone()
        assert tr["task_id"] == "a5-test:doc:part-1"
        assert tr["mode"] == "documentation"


class TestFlushSitesPassThinking:
    """Verify that _flush_llm_calls passes thinking to insert_retrieval_llm_call."""

    def test_flush_with_thinking(self, raw_conn):
        from clean_room_agent.orchestrator.runner import _flush_llm_calls
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.flush.return_value = [{
            "prompt": "test prompt",
            "system": "test system",
            "response": "test response",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "elapsed_ms": 500,
            "thinking": "Let me think about this...",
        }]

        _flush_llm_calls(mock_llm, raw_conn, "flush-test", "test_call", "test_stage")

        row = raw_conn.execute(
            "SELECT * FROM retrieval_llm_calls WHERE task_id = 'flush-test'"
        ).fetchone()
        assert row["thinking"] == "Let me think about this..."

    def test_flush_without_thinking(self, raw_conn):
        from clean_room_agent.orchestrator.runner import _flush_llm_calls
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.flush.return_value = [{
            "prompt": "test prompt",
            "system": "test system",
            "response": "test response",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "elapsed_ms": 500,
        }]

        _flush_llm_calls(mock_llm, raw_conn, "flush-test2", "test_call", "test_stage")

        row = raw_conn.execute(
            "SELECT * FROM retrieval_llm_calls WHERE task_id = 'flush-test2'"
        ).fetchone()
        assert row["thinking"] is None
