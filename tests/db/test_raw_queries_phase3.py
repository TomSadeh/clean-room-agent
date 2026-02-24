"""Tests for Phase 3 raw DB query additions."""

import pytest

from clean_room_agent.db.raw_queries import (
    insert_orchestrator_pass,
    insert_orchestrator_run,
    insert_run_attempt,
    insert_task_run,
    insert_validation_result,
    update_orchestrator_run,
)


class TestInsertRunAttempt:
    def _make_task_run(self, raw_conn):
        run_id = insert_task_run(
            raw_conn, "task-ra", "/repo", "implement",
            "qwen2.5-coder:3b", 32768, 4096, "scope,precision",
        )
        raw_conn.commit()
        return run_id

    def test_insert_basic(self, raw_conn):
        run_id = self._make_task_run(raw_conn)
        attempt_id = insert_run_attempt(
            raw_conn, run_id, 1, 500, 200, 3000,
            "<edit>...</edit>", True,
        )
        raw_conn.commit()
        assert attempt_id > 0

        row = raw_conn.execute(
            "SELECT * FROM run_attempts WHERE id = ?", (attempt_id,)
        ).fetchone()
        assert row["task_run_id"] == run_id
        assert row["attempt"] == 1
        assert row["prompt_tokens"] == 500
        assert row["completion_tokens"] == 200
        assert row["latency_ms"] == 3000
        assert row["raw_response"] == "<edit>...</edit>"
        assert row["patch_applied"] == 1
        assert row["timestamp"] is not None

    def test_insert_patch_not_applied(self, raw_conn):
        run_id = self._make_task_run(raw_conn)
        attempt_id = insert_run_attempt(
            raw_conn, run_id, 1, 500, 200, 3000,
            "malformed response", False,
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM run_attempts WHERE id = ?", (attempt_id,)
        ).fetchone()
        assert row["patch_applied"] == 0

    def test_insert_null_tokens(self, raw_conn):
        run_id = self._make_task_run(raw_conn)
        attempt_id = insert_run_attempt(
            raw_conn, run_id, 1, None, None, 1000,
            "response", True,
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM run_attempts WHERE id = ?", (attempt_id,)
        ).fetchone()
        assert row["prompt_tokens"] is None
        assert row["completion_tokens"] is None

    def test_multiple_attempts_same_run(self, raw_conn):
        run_id = self._make_task_run(raw_conn)
        id1 = insert_run_attempt(raw_conn, run_id, 1, 100, 50, 500, "r1", False)
        id2 = insert_run_attempt(raw_conn, run_id, 2, 200, 100, 600, "r2", True)
        raw_conn.commit()
        assert id1 != id2
        rows = raw_conn.execute(
            "SELECT * FROM run_attempts WHERE task_run_id = ? ORDER BY attempt",
            (run_id,),
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["attempt"] == 1
        assert rows[1]["attempt"] == 2

    def test_fk_constraint(self, raw_conn):
        raw_conn.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(Exception):
            insert_run_attempt(
                raw_conn, 99999, 1, 100, 50, 500, "response", True,
            )


class TestInsertValidationResult:
    def _make_attempt(self, raw_conn):
        run_id = insert_task_run(
            raw_conn, "task-vr", "/repo", "implement",
            "model", 32768, 4096, "scope",
        )
        attempt_id = insert_run_attempt(
            raw_conn, run_id, 1, 100, 50, 500, "response", True,
        )
        raw_conn.commit()
        return attempt_id

    def test_insert_success(self, raw_conn):
        attempt_id = self._make_attempt(raw_conn)
        vr_id = insert_validation_result(
            raw_conn, attempt_id, True,
            test_output="3 passed",
        )
        raw_conn.commit()
        assert vr_id > 0

        row = raw_conn.execute(
            "SELECT * FROM validation_results WHERE id = ?", (vr_id,)
        ).fetchone()
        assert row["attempt_id"] == attempt_id
        assert row["success"] == 1
        assert row["test_output"] == "3 passed"
        assert row["lint_output"] is None
        assert row["type_check_output"] is None
        assert row["failing_tests"] is None

    def test_insert_failure_with_details(self, raw_conn):
        attempt_id = self._make_attempt(raw_conn)
        vr_id = insert_validation_result(
            raw_conn, attempt_id, False,
            test_output="1 failed, 2 passed",
            lint_output="E501 line too long",
            type_check_output="error: incompatible type",
            failing_tests="test_foo::test_bar",
        )
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM validation_results WHERE id = ?", (vr_id,)
        ).fetchone()
        assert row["success"] == 0
        assert row["test_output"] == "1 failed, 2 passed"
        assert row["lint_output"] == "E501 line too long"
        assert row["type_check_output"] == "error: incompatible type"
        assert row["failing_tests"] == "test_foo::test_bar"

    def test_insert_all_null_outputs(self, raw_conn):
        attempt_id = self._make_attempt(raw_conn)
        vr_id = insert_validation_result(raw_conn, attempt_id, True)
        raw_conn.commit()
        row = raw_conn.execute(
            "SELECT * FROM validation_results WHERE id = ?", (vr_id,)
        ).fetchone()
        assert row["test_output"] is None

    def test_fk_chain(self, raw_conn):
        """Full FK chain: task_run -> run_attempt -> validation_result."""
        run_id = insert_task_run(
            raw_conn, "task-chain", "/repo", "implement",
            "model", 32768, 4096, "scope",
        )
        attempt_id = insert_run_attempt(
            raw_conn, run_id, 1, 100, 50, 500, "response", True,
        )
        vr_id = insert_validation_result(raw_conn, attempt_id, True, test_output="ok")
        raw_conn.commit()

        # Verify chain
        vr = raw_conn.execute("SELECT * FROM validation_results WHERE id = ?", (vr_id,)).fetchone()
        att = raw_conn.execute("SELECT * FROM run_attempts WHERE id = ?", (vr["attempt_id"],)).fetchone()
        tr = raw_conn.execute("SELECT * FROM task_runs WHERE id = ?", (att["task_run_id"],)).fetchone()
        assert tr["task_id"] == "task-chain"


class TestInsertOrchestratorRun:
    def test_insert_basic(self, raw_conn):
        run_id = insert_orchestrator_run(
            raw_conn, "orch-001", "/repo", "Add input validation",
        )
        raw_conn.commit()
        assert run_id > 0

        row = raw_conn.execute(
            "SELECT * FROM orchestrator_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["task_id"] == "orch-001"
        assert row["repo_path"] == "/repo"
        assert row["task_description"] == "Add input validation"
        assert row["status"] == "running"
        assert row["total_parts"] is None
        assert row["total_steps"] is None
        assert row["parts_completed"] == 0
        assert row["steps_completed"] == 0
        assert row["completed_at"] is None
        assert row["timestamp"] is not None


class TestUpdateOrchestratorRun:
    def _make_run(self, raw_conn, task_id="orch-upd"):
        run_id = insert_orchestrator_run(
            raw_conn, task_id, "/repo", "task desc",
        )
        raw_conn.commit()
        return run_id

    def test_update_totals(self, raw_conn):
        run_id = self._make_run(raw_conn)
        update_orchestrator_run(raw_conn, run_id, total_parts=3, total_steps=9)
        raw_conn.commit()

        row = raw_conn.execute(
            "SELECT * FROM orchestrator_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["total_parts"] == 3
        assert row["total_steps"] == 9

    def test_update_progress(self, raw_conn):
        run_id = self._make_run(raw_conn)
        update_orchestrator_run(raw_conn, run_id, parts_completed=2, steps_completed=6)
        raw_conn.commit()

        row = raw_conn.execute(
            "SELECT * FROM orchestrator_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["parts_completed"] == 2
        assert row["steps_completed"] == 6

    def test_update_status(self, raw_conn):
        run_id = self._make_run(raw_conn)
        update_orchestrator_run(raw_conn, run_id, status="complete", completed_at="2025-01-01T00:00:00Z")
        raw_conn.commit()

        row = raw_conn.execute(
            "SELECT * FROM orchestrator_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["status"] == "complete"
        assert row["completed_at"] == "2025-01-01T00:00:00Z"

    def test_update_nonexistent_raises(self, raw_conn):
        with pytest.raises(RuntimeError, match="No orchestrator_run"):
            update_orchestrator_run(raw_conn, 99999, status="failed")

    def test_update_no_fields_raises(self, raw_conn):
        run_id = self._make_run(raw_conn)
        with pytest.raises(ValueError, match="no fields"):
            update_orchestrator_run(raw_conn, run_id)


class TestInsertOrchestratorPass:
    def _make_prereqs(self, raw_conn, task_id="orch-pass"):
        orch_id = insert_orchestrator_run(
            raw_conn, task_id, "/repo", "task desc",
        )
        tr_id = insert_task_run(
            raw_conn, f"{task_id}:meta_plan", "/repo", "plan",
            "model", 32768, 4096, "scope,precision",
        )
        raw_conn.commit()
        return orch_id, tr_id

    def test_insert_meta_plan_pass(self, raw_conn):
        orch_id, tr_id = self._make_prereqs(raw_conn)
        pass_id = insert_orchestrator_pass(
            raw_conn, orch_id, tr_id, "meta_plan", 0,
        )
        raw_conn.commit()
        assert pass_id > 0

        row = raw_conn.execute(
            "SELECT * FROM orchestrator_passes WHERE id = ?", (pass_id,)
        ).fetchone()
        assert row["orchestrator_run_id"] == orch_id
        assert row["task_run_id"] == tr_id
        assert row["pass_type"] == "meta_plan"
        assert row["sequence_order"] == 0
        assert row["part_id"] is None
        assert row["step_id"] is None
        assert row["timestamp"] is not None

    def test_insert_step_implement_pass(self, raw_conn):
        orch_id, tr_id = self._make_prereqs(raw_conn, "orch-pass2")
        pass_id = insert_orchestrator_pass(
            raw_conn, orch_id, tr_id, "step_implement", 3,
            part_id="part-1", step_id="step-2",
        )
        raw_conn.commit()

        row = raw_conn.execute(
            "SELECT * FROM orchestrator_passes WHERE id = ?", (pass_id,)
        ).fetchone()
        assert row["pass_type"] == "step_implement"
        assert row["part_id"] == "part-1"
        assert row["step_id"] == "step-2"
        assert row["sequence_order"] == 3

    def test_multiple_passes_same_run(self, raw_conn):
        orch_id, tr_id = self._make_prereqs(raw_conn, "orch-pass3")
        # Add a second task_run for part_plan pass
        tr_id2 = insert_task_run(
            raw_conn, "orch-pass3:part_plan:p1", "/repo", "plan",
            "model", 32768, 4096, "scope,precision",
        )
        raw_conn.commit()

        id1 = insert_orchestrator_pass(raw_conn, orch_id, tr_id, "meta_plan", 0)
        id2 = insert_orchestrator_pass(
            raw_conn, orch_id, tr_id2, "part_plan", 1, part_id="p1",
        )
        raw_conn.commit()

        assert id1 != id2
        rows = raw_conn.execute(
            "SELECT * FROM orchestrator_passes WHERE orchestrator_run_id = ? ORDER BY sequence_order",
            (orch_id,),
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["pass_type"] == "meta_plan"
        assert rows[1]["pass_type"] == "part_plan"

    def test_traceability_chain(self, raw_conn):
        """Full chain: orchestrator_run -> orchestrator_pass -> task_run -> run_attempt -> validation."""
        orch_id = insert_orchestrator_run(raw_conn, "trace-001", "/repo", "task")
        tr_id = insert_task_run(
            raw_conn, "trace-001:impl:p1:s1", "/repo", "implement",
            "model", 32768, 4096, "scope",
        )
        pass_id = insert_orchestrator_pass(
            raw_conn, orch_id, tr_id, "step_implement", 2,
            part_id="p1", step_id="s1",
        )
        attempt_id = insert_run_attempt(
            raw_conn, tr_id, 1, 500, 200, 3000, "<edit>...</edit>", True,
        )
        vr_id = insert_validation_result(raw_conn, attempt_id, True, test_output="ok")
        raw_conn.commit()

        # Walk the chain
        vr = raw_conn.execute("SELECT * FROM validation_results WHERE id = ?", (vr_id,)).fetchone()
        att = raw_conn.execute("SELECT * FROM run_attempts WHERE id = ?", (vr["attempt_id"],)).fetchone()
        tr = raw_conn.execute("SELECT * FROM task_runs WHERE id = ?", (att["task_run_id"],)).fetchone()
        op = raw_conn.execute(
            "SELECT * FROM orchestrator_passes WHERE task_run_id = ?", (tr["id"],)
        ).fetchone()
        orch = raw_conn.execute(
            "SELECT * FROM orchestrator_runs WHERE id = ?", (op["orchestrator_run_id"],)
        ).fetchone()
        assert orch["task_id"] == "trace-001"
