"""Tests for Phase 2 raw DB query additions."""

import pytest

from clean_room_agent.db.raw_queries import (
    insert_enrichment_output,
    insert_retrieval_decision,
    insert_retrieval_llm_call,
    insert_session_archive,
    insert_task_run,
    update_task_run,
)


class TestInsertRetrievalLlmCall:
    def test_insert_basic(self, raw_conn):
        call_id = insert_retrieval_llm_call(
            raw_conn, "task-001", "task_analysis", "qwen3:4b",
            "prompt text", "response text", 100, 50, 1500,
            stage_name="task_analysis",
        )
        raw_conn.commit()
        assert call_id > 0

        row = raw_conn.execute(
            "SELECT * FROM retrieval_llm_calls WHERE id = ?", (call_id,)
        ).fetchone()
        assert row["task_id"] == "task-001"
        assert row["call_type"] == "task_analysis"
        assert row["model"] == "qwen3:4b"
        assert row["prompt"] == "prompt text"
        assert row["response"] == "response text"
        assert row["prompt_tokens"] == 100
        assert row["completion_tokens"] == 50
        assert row["latency_ms"] == 1500
        assert row["stage_name"] == "task_analysis"

    def test_insert_null_tokens(self, raw_conn):
        call_id = insert_retrieval_llm_call(
            raw_conn, "task-002", "scope", "qwen3:4b",
            "prompt", "response", None, None, 500,
            stage_name="scope",
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM retrieval_llm_calls WHERE id = ?", (call_id,)
        ).fetchone()
        assert row["prompt_tokens"] is None
        assert row["completion_tokens"] is None

    def test_insert_no_stage_name(self, raw_conn):
        call_id = insert_retrieval_llm_call(
            raw_conn, "task-003", "enrichment", "qwen3:4b",
            "p", "r", 10, 5, 100,
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM retrieval_llm_calls WHERE id = ?", (call_id,)
        ).fetchone()
        assert row["stage_name"] is None

    def test_insert_with_system_prompt(self, raw_conn):
        """T1: system_prompt is persisted to raw DB for full traceability."""
        system = "You are a retrieval judge. Classify files as relevant or irrelevant."
        call_id = insert_retrieval_llm_call(
            raw_conn, "task-004", "scope", "qwen3:4b",
            "classify these files", "relevant", 80, 10, 300,
            stage_name="scope",
            system_prompt=system,
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM retrieval_llm_calls WHERE id = ?", (call_id,)
        ).fetchone()
        assert row["system_prompt"] == system

    def test_insert_null_system_prompt(self, raw_conn):
        """T1: system_prompt defaults to NULL when not provided."""
        call_id = insert_retrieval_llm_call(
            raw_conn, "task-005", "task_analysis", "qwen3:4b",
            "prompt", "response", None, None, 100,
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM retrieval_llm_calls WHERE id = ?", (call_id,)
        ).fetchone()
        assert row["system_prompt"] is None


class TestInsertEnrichmentOutput:
    def test_insert_with_system_prompt(self, raw_conn):
        """T1: enrichment system_prompt is persisted to raw DB."""
        system = "You are a code analysis assistant."
        out_id = insert_enrichment_output(
            raw_conn,
            file_id=1, model="qwen3:4b",
            raw_prompt="File: main.py\n...",
            raw_response='{"purpose": "entry point"}',
            purpose="entry point",
            system_prompt=system,
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM enrichment_outputs WHERE id = ?", (out_id,)
        ).fetchone()
        assert row["system_prompt"] == system

    def test_insert_null_system_prompt(self, raw_conn):
        """T1: enrichment system_prompt defaults to NULL for backwards compat."""
        out_id = insert_enrichment_output(
            raw_conn,
            file_id=2, model="qwen3:4b",
            raw_prompt="prompt", raw_response="response",
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM enrichment_outputs WHERE id = ?", (out_id,)
        ).fetchone()
        assert row["system_prompt"] is None

    def test_insert_with_task_id(self, raw_conn):
        """T14: enrichment outputs can be linked to a task_id."""
        out_id = insert_enrichment_output(
            raw_conn,
            file_id=3, model="qwen3:4b",
            raw_prompt="prompt", raw_response="response",
            task_id="enrich-run-001",
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM enrichment_outputs WHERE id = ?", (out_id,)
        ).fetchone()
        assert row["task_id"] == "enrich-run-001"

    def test_insert_null_task_id(self, raw_conn):
        """T14: task_id is nullable for standalone enrichment runs."""
        out_id = insert_enrichment_output(
            raw_conn,
            file_id=4, model="qwen3:4b",
            raw_prompt="prompt", raw_response="response",
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM enrichment_outputs WHERE id = ?", (out_id,)
        ).fetchone()
        assert row["task_id"] is None


class TestInsertRetrievalDecision:
    def test_insert_included(self, raw_conn):
        dec_id = insert_retrieval_decision(
            raw_conn, "task-001", "scope", 42,
            included=True, tier="1", reason="seed file",
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM retrieval_decisions WHERE id = ?", (dec_id,)
        ).fetchone()
        assert row["task_id"] == "task-001"
        assert row["stage"] == "scope"
        assert row["file_id"] == 42
        assert row["included"] == 1
        assert row["tier"] == "1"
        assert row["reason"] == "seed file"

    def test_insert_excluded(self, raw_conn):
        dec_id = insert_retrieval_decision(
            raw_conn, "task-001", "scope", 99,
            included=False, reason="LLM judged irrelevant",
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM retrieval_decisions WHERE id = ?", (dec_id,)
        ).fetchone()
        assert row["included"] == 0

    def test_insert_symbol_decision(self, raw_conn):
        """T16: symbol-level decisions include symbol_id and detail_level."""
        dec_id = insert_retrieval_decision(
            raw_conn, "task-001", "precision", 42,
            included=True, reason="directly modified",
            symbol_id=17, detail_level="primary",
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM retrieval_decisions WHERE id = ?", (dec_id,)
        ).fetchone()
        assert row["symbol_id"] == 17
        assert row["detail_level"] == "primary"

    def test_insert_file_decision_null_symbol(self, raw_conn):
        """T16: file-level decisions have NULL symbol_id."""
        dec_id = insert_retrieval_decision(
            raw_conn, "task-001", "scope", 42,
            included=True, tier="1",
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM retrieval_decisions WHERE id = ?", (dec_id,)
        ).fetchone()
        assert row["symbol_id"] is None
        assert row["detail_level"] is None


class TestInsertTaskRun:
    def test_insert_basic(self, raw_conn):
        run_id = insert_task_run(
            raw_conn, "task-001", "/repo", "plan",
            "qwen2.5-coder:3b", 32768, 4096, "scope,precision",
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM task_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["task_id"] == "task-001"
        assert row["repo_path"] == "/repo"
        assert row["mode"] == "plan"
        assert row["execute_model"] == "qwen2.5-coder:3b"
        assert row["context_window"] == 32768
        assert row["reserved_tokens"] == 4096
        assert row["stages"] == "scope,precision"
        assert row["plan_artifact"] is None
        assert row["success"] is None

    def test_insert_with_plan_artifact(self, raw_conn):
        run_id = insert_task_run(
            raw_conn, "task-002", "/repo", "implement",
            "qwen2.5-coder:3b", 32768, 4096, "scope,precision",
            plan_artifact="/path/to/plan.json",
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM task_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["plan_artifact"] == "/path/to/plan.json"

    def test_unique_task_id(self, raw_conn):
        insert_task_run(
            raw_conn, "task-unique", "/repo", "plan",
            "model", 32768, 4096, "scope",
        )
        raw_conn.commit()
        with pytest.raises(Exception):
            insert_task_run(
                raw_conn, "task-unique", "/repo", "plan",
                "model", 32768, 4096, "scope",
            )


class TestUpdateTaskRun:
    def test_update_success(self, raw_conn):
        run_id = insert_task_run(
            raw_conn, "task-upd", "/repo", "plan",
            "model", 32768, 4096, "scope",
        )
        raw_conn.commit()

        update_task_run(raw_conn, run_id, success=True, total_tokens=5000, total_latency_ms=3000)
        raw_conn.commit()

        row = raw_conn.execute(
            "SELECT * FROM task_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["success"] == 1
        assert row["total_tokens"] == 5000
        assert row["total_latency_ms"] == 3000

    def test_update_failure(self, raw_conn):
        run_id = insert_task_run(
            raw_conn, "task-fail", "/repo", "plan",
            "model", 32768, 4096, "scope",
        )
        raw_conn.commit()

        update_task_run(raw_conn, run_id, success=False, total_latency_ms=500)
        raw_conn.commit()

        row = raw_conn.execute(
            "SELECT * FROM task_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["success"] == 0
        assert row["total_tokens"] is None

    def test_update_nonexistent_raises(self, raw_conn):
        """B3: updating a nonexistent task_run should raise."""
        with pytest.raises(RuntimeError, match="No task_run"):
            update_task_run(raw_conn, 99999, success=True)


class TestInsertSessionArchive:
    """T13: session_archives stores serialized session DB blobs."""

    def test_insert_basic(self, raw_conn):
        blob = b"SQLite format 3\x00" + b"\x00" * 100
        archive_id = insert_session_archive(raw_conn, "task-001", blob)
        raw_conn.commit()
        assert archive_id > 0

        row = raw_conn.execute(
            "SELECT * FROM session_archives WHERE id = ?", (archive_id,)
        ).fetchone()
        assert row["task_id"] == "task-001"
        assert row["session_blob"] == blob
        assert row["archived_at"] is not None

    def test_multiple_archives_same_task(self, raw_conn):
        """Multiple refinement runs can archive the same task_id."""
        id1 = insert_session_archive(raw_conn, "task-002", b"blob1")
        id2 = insert_session_archive(raw_conn, "task-002", b"blob2")
        raw_conn.commit()
        assert id1 != id2
        rows = raw_conn.execute(
            "SELECT * FROM session_archives WHERE task_id = ? ORDER BY id",
            ("task-002",),
        ).fetchall()
        assert len(rows) == 2
