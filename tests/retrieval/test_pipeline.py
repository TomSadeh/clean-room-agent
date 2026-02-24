"""Tests for pipeline runner."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clean_room_agent.db.connection import get_connection
from clean_room_agent.db.queries import (
    insert_symbol,
    upsert_file,
    upsert_repo,
)
from clean_room_agent.db.session_helpers import get_state, set_state
from clean_room_agent.retrieval.dataclasses import BudgetConfig, ContextPackage, RefinementRequest
from clean_room_agent.retrieval.pipeline import run_pipeline, _resume_task_from_session


def _make_config():
    return {
        "models": {
            "provider": "ollama",
            "coding": "qwen2.5-coder:3b",
            "reasoning": "qwen3:4b",
            "base_url": "http://localhost:11434",
            "context_window": 32768,
        },
    }


@pytest.fixture
def pipeline_repo(tmp_path):
    """Set up a repo with curated DB, source files, and config."""
    # Create source files
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        "def main():\n"
        "    print('hello')\n"
        "\n"
        "def helper():\n"
        "    return 42\n"
    )
    (src / "utils.py").write_text(
        "def format_output(data):\n"
        "    return str(data)\n"
    )

    # Populate curated DB
    conn = get_connection("curated", repo_path=tmp_path)
    repo_id = upsert_repo(conn, str(tmp_path), None)
    fid1 = upsert_file(conn, repo_id, "src/main.py", "python", "h1", 100)
    fid2 = upsert_file(conn, repo_id, "src/utils.py", "python", "h2", 50)
    insert_symbol(conn, fid1, "main", "function", 1, 2, "def main()")
    insert_symbol(conn, fid1, "helper", "function", 4, 5, "def helper()")
    insert_symbol(conn, fid2, "format_output", "function", 1, 2, "def format_output(data)")
    conn.commit()
    conn.close()

    # Ensure raw DB exists
    raw_conn = get_connection("raw", repo_path=tmp_path)
    raw_conn.close()

    return tmp_path, repo_id, fid1, fid2


def _mock_llm_complete(prompt, system=None):
    """Mock LLM that returns different responses based on system prompt."""
    response = MagicMock()
    response.latency_ms = 100
    response.prompt_tokens = 50
    response.completion_tokens = 20

    if system and "task analyzer" in system.lower():
        response.text = "Fix the main function in main.py"
    elif system and "stage router" in system.lower():
        response.text = json.dumps({
            "stages": ["scope", "precision"],
            "reasoning": "Bug fix needs full context.",
        })
    elif system and "retrieval judge" in system.lower():
        response.text = json.dumps([
            {"path": "src/main.py", "verdict": "relevant", "reason": "directly mentioned"},
            {"path": "src/utils.py", "verdict": "relevant", "reason": "related utility"},
        ])
    elif system and "precision analyst" in system.lower():
        response.text = json.dumps([
            {"name": "main", "file_path": "src/main.py", "start_line": 1, "detail_level": "primary", "reason": "target"},
            {"name": "helper", "file_path": "src/main.py", "start_line": 4, "detail_level": "supporting", "reason": "context"},
            {"name": "format_output", "file_path": "src/utils.py", "start_line": 1, "detail_level": "type_context", "reason": "utility"},
        ])
    elif system and "budget optimizer" in system.lower():
        response.text = "Generic response"
    else:
        response.text = "Generic response"
    return response


def _make_mock_llm_instance():
    """Create a mock LoggedLLMClient instance with config and call recording."""
    mock_instance = MagicMock()
    mock_instance.calls = []
    mock_instance.config.context_window = 32768
    mock_instance.config.max_tokens = 4096
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=False)

    def _recording_complete(prompt, system=None):
        response = _mock_llm_complete(prompt, system)
        mock_instance.calls.append({
            "prompt": prompt,
            "system": system,
            "response": response.text,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "elapsed_ms": 100,
        })
        return response

    def _flush():
        calls = list(mock_instance.calls)
        mock_instance.calls.clear()
        return calls

    mock_instance.complete.side_effect = _recording_complete
    mock_instance.flush.side_effect = _flush
    return mock_instance


class TestRunPipeline:
    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_end_to_end(self, mock_llm_class, pipeline_repo):
        tmp_path, repo_id, fid1, fid2 = pipeline_repo

        mock_llm_class.return_value = _make_mock_llm_instance()

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        config = _make_config()

        package = run_pipeline(
            raw_task="Fix the main function in src/main.py",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id="test-pipeline-001",
            config=config,
        )

        assert package is not None
        assert package.total_token_estimate > 0
        assert len(package.files) >= 1
        assert package.task.task_id == "test-pipeline-001"

    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_session_archived_and_deleted(self, mock_llm_class, pipeline_repo):
        """T13: session DB is archived to raw DB and file is deleted after pipeline."""
        tmp_path, repo_id, fid1, fid2 = pipeline_repo

        mock_llm_class.return_value = _make_mock_llm_instance()

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        task_id = "test-session-001"

        run_pipeline(
            raw_task="Fix the bug",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id=task_id,
            config=_make_config(),
        )

        # Session file should be deleted
        session_file = tmp_path / ".clean_room" / "sessions" / f"session_{task_id}.sqlite"
        assert not session_file.exists(), "Session file should be deleted after archival"

        # Session should be archived in raw DB
        raw_conn = get_connection("raw", repo_path=tmp_path)
        try:
            archive = raw_conn.execute(
                "SELECT * FROM session_archives WHERE task_id = ?", (task_id,)
            ).fetchone()
            assert archive is not None
            assert len(archive["session_blob"]) > 0
        finally:
            raw_conn.close()

    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_raw_db_logging(self, mock_llm_class, pipeline_repo):
        tmp_path, repo_id, fid1, fid2 = pipeline_repo

        mock_llm_class.return_value = _make_mock_llm_instance()

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        task_id = "test-rawlog-001"

        run_pipeline(
            raw_task="Fix the bug",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id=task_id,
            config=_make_config(),
        )

        # Verify raw DB has task_run
        raw_conn = get_connection("raw", repo_path=tmp_path)
        try:
            task_run = raw_conn.execute(
                "SELECT * FROM task_runs WHERE task_id = ?", (task_id,)
            ).fetchone()
            assert task_run is not None
            assert task_run["success"] == 1
            assert task_run["mode"] == "plan"

            # Check LLM calls logged
            llm_calls = raw_conn.execute(
                "SELECT * FROM retrieval_llm_calls WHERE task_id = ?", (task_id,)
            ).fetchall()
            assert len(llm_calls) >= 1  # at least task_analysis
        finally:
            raw_conn.close()

    def test_unknown_stage_raises(self, pipeline_repo):
        tmp_path, repo_id, fid1, fid2 = pipeline_repo
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)

        with pytest.raises(ValueError, match="Unknown stage"):
            run_pipeline(
                raw_task="task",
                repo_path=tmp_path,
                stage_names=["nonexistent"],
                budget=budget,
                mode="plan",
                task_id="test-error-001",
                config=_make_config(),
            )

    def test_missing_config_raises(self, pipeline_repo):
        tmp_path, repo_id, fid1, fid2 = pipeline_repo
        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)

        with pytest.raises(RuntimeError, match="No config file found"):
            run_pipeline(
                raw_task="task",
                repo_path=tmp_path,
                stage_names=["scope"],
                budget=budget,
                mode="plan",
                task_id="test-noconfig-001",
                config=None,
            )

    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_with_plan_artifact(self, mock_llm_class, pipeline_repo):
        tmp_path, repo_id, fid1, fid2 = pipeline_repo

        mock_llm_class.return_value = _make_mock_llm_instance()

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        # Create plan artifact
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps({
            "affected_files": ["src/main.py"],
        }))

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)

        package = run_pipeline(
            raw_task="Fix it",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id="test-plan-001",
            config=_make_config(),
            plan_artifact_path=plan_path,
        )

        assert package is not None
        assert len(package.files) >= 1

    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_environment_brief_threaded(self, mock_llm_class, pipeline_repo):
        """Environment brief flows through to ContextPackage and task analysis prompt."""
        tmp_path, repo_id, fid1, fid2 = pipeline_repo

        mock_llm_class.return_value = _make_mock_llm_instance()

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        config = _make_config()
        config["testing"] = {"test_command": "pytest tests/"}

        package = run_pipeline(
            raw_task="Fix the main function in src/main.py",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id="test-env-brief-001",
            config=config,
        )

        # Environment brief should be set on the package
        assert package.environment_brief != ""
        assert "<environment>" in package.environment_brief
        assert "pytest" in package.environment_brief

        # to_prompt_text should include the brief
        prompt_text = package.to_prompt_text()
        assert "<environment>" in prompt_text

    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_error_handler_propagates_original(self, mock_llm_class, pipeline_repo):
        """B2: if the pipeline errors, the original exception propagates even if DB logging fails."""
        tmp_path, repo_id, fid1, fid2 = pipeline_repo

        # First context manager: task analysis succeeds
        # Second context manager: scope stage's LLM call raises
        call_count = 0
        def _side_effect(prompt, system=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return _mock_llm_complete(prompt, system)
            raise RuntimeError("LLM exploded")

        mock_instance = _make_mock_llm_instance()
        mock_instance.complete.side_effect = _side_effect
        mock_llm_class.return_value = mock_instance

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)

        # The scope stage calls llm.complete() which raises on call #2.
        # The error handler in pipeline should propagate the original RuntimeError.
        with pytest.raises(RuntimeError, match="LLM exploded"):
            run_pipeline(
                raw_task="Fix the main function in src/main.py",
                repo_path=tmp_path,
                stage_names=["scope", "precision"],
                budget=budget,
                mode="plan",
                task_id="test-error-002",
                config=_make_config(),
            )


class TestPipelineTraceability:
    """T1/T2/T3: Verify full traceability of LLM calls and decisions in raw DB."""

    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_system_prompt_logged(self, mock_llm_class, pipeline_repo):
        """T1: system prompts are persisted to raw DB retrieval_llm_calls."""
        tmp_path, repo_id, fid1, fid2 = pipeline_repo
        mock_llm_class.return_value = _make_mock_llm_instance()

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        task_id = "test-sysprompt-001"

        run_pipeline(
            raw_task="Fix the main function in src/main.py",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id=task_id,
            config=_make_config(),
        )

        raw_conn = get_connection("raw", repo_path=tmp_path)
        try:
            calls = raw_conn.execute(
                "SELECT * FROM retrieval_llm_calls WHERE task_id = ?", (task_id,)
            ).fetchall()
            assert len(calls) >= 1
            # Every call should have a system_prompt recorded (not NULL)
            for call in calls:
                assert call["system_prompt"] is not None, (
                    f"Call {call['call_type']} has NULL system_prompt"
                )
        finally:
            raw_conn.close()

    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_token_counts_logged(self, mock_llm_class, pipeline_repo):
        """T6: prompt_tokens and completion_tokens are captured from LLM responses."""
        tmp_path, repo_id, fid1, fid2 = pipeline_repo
        mock_llm_class.return_value = _make_mock_llm_instance()

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        task_id = "test-tokens-001"

        run_pipeline(
            raw_task="Fix the main function in src/main.py",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id=task_id,
            config=_make_config(),
        )

        raw_conn = get_connection("raw", repo_path=tmp_path)
        try:
            calls = raw_conn.execute(
                "SELECT * FROM retrieval_llm_calls WHERE task_id = ?", (task_id,)
            ).fetchall()
            assert len(calls) >= 1
            # Mock returns prompt_tokens=50, completion_tokens=20
            for call in calls:
                assert call["prompt_tokens"] == 50, (
                    f"Call {call['call_type']} has wrong prompt_tokens: {call['prompt_tokens']}"
                )
                assert call["completion_tokens"] == 20, (
                    f"Call {call['call_type']} has wrong completion_tokens: {call['completion_tokens']}"
                )
        finally:
            raw_conn.close()

    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_assembly_decisions_logged(self, mock_llm_class, pipeline_repo):
        """T3: assembly file decisions are logged to raw DB retrieval_decisions."""
        tmp_path, repo_id, fid1, fid2 = pipeline_repo
        mock_llm_class.return_value = _make_mock_llm_instance()

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        task_id = "test-assembly-dec-001"

        run_pipeline(
            raw_task="Fix the main function in src/main.py",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id=task_id,
            config=_make_config(),
        )

        raw_conn = get_connection("raw", repo_path=tmp_path)
        try:
            decisions = raw_conn.execute(
                "SELECT * FROM retrieval_decisions WHERE task_id = ? AND stage = 'assembly'",
                (task_id,),
            ).fetchall()
            # Should have at least the included files from assembly
            assert len(decisions) >= 1
            # At least one should be included=True
            included = [d for d in decisions if d["included"]]
            assert len(included) >= 1
        finally:
            raw_conn.close()


class TestRefinementPipeline:
    """Test pipeline re-entry via refinement_request."""

    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_refinement_restores_session_and_merges(self, mock_llm_class, pipeline_repo):
        """Full refinement flow: run pipeline, then re-enter with refinement."""
        tmp_path, repo_id, fid1, fid2 = pipeline_repo

        mock_llm_class.return_value = _make_mock_llm_instance()

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)

        # Step 1: Normal pipeline run
        source_task_id = "test-refine-source"
        package1 = run_pipeline(
            raw_task="Fix the main function in src/main.py",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id=source_task_id,
            config=_make_config(),
        )
        assert package1 is not None

        # Session file should be deleted after run
        from clean_room_agent.db.connection import _db_path
        session_file = _db_path(tmp_path, "session", source_task_id)
        assert not session_file.exists()

        # Step 2: Refinement re-entry with new task_id
        refinement = RefinementRequest(
            reason="need more context",
            source_task_id=source_task_id,
            missing_files=["src/utils.py"],
            error_patterns=["NameError: helper not defined"],
        )

        refinement_task_id = "test-refine-pass2"
        package2 = run_pipeline(
            raw_task="Fix the main function in src/main.py",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id=refinement_task_id,
            config=_make_config(),
            refinement_request=refinement,
        )
        assert package2 is not None
        assert package2.task.task_id == refinement_task_id
        # Merged error patterns from refinement
        assert "NameError: helper not defined" in package2.task.error_patterns
        # Merged missing_files into mentioned_files
        assert "src/utils.py" in package2.task.mentioned_files

    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_refinement_no_archive_raises(self, mock_llm_class, pipeline_repo):
        """Refinement without a prior session archive should fail fast."""
        tmp_path, repo_id, fid1, fid2 = pipeline_repo

        mock_llm_class.return_value = _make_mock_llm_instance()

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        refinement = RefinementRequest(
            reason="need more",
            source_task_id="nonexistent-task",
        )

        with pytest.raises(RuntimeError, match="no session archive found"):
            run_pipeline(
                raw_task="Fix it",
                repo_path=tmp_path,
                stage_names=["scope", "precision"],
                budget=budget,
                mode="plan",
                task_id="test-refine-noarchive",
                config=_make_config(),
                refinement_request=refinement,
            )

    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_refinement_logged_to_raw_db(self, mock_llm_class, pipeline_repo):
        """Refinement merge decision is logged as a retrieval_llm_call."""
        tmp_path, repo_id, fid1, fid2 = pipeline_repo

        mock_llm_class.return_value = _make_mock_llm_instance()

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)

        # Normal run first
        run_pipeline(
            raw_task="Fix it",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id="test-reflog-source",
            config=_make_config(),
        )

        # Refinement run
        refinement = RefinementRequest(
            reason="missing context",
            source_task_id="test-reflog-source",
            missing_files=["extra.py"],
        )
        run_pipeline(
            raw_task="Fix it",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id="test-reflog-pass2",
            config=_make_config(),
            refinement_request=refinement,
        )

        # Check raw DB for refinement_merge call
        raw_conn = get_connection("raw", repo_path=tmp_path)
        try:
            calls = raw_conn.execute(
                "SELECT * FROM retrieval_llm_calls WHERE task_id = ? AND call_type = 'refinement_merge'",
                ("test-reflog-pass2",),
            ).fetchall()
            assert len(calls) == 1
            assert calls[0]["stage_name"] == "task_analysis"
        finally:
            raw_conn.close()


class TestRestoreSessionFromArchive:
    def test_restore_from_archive(self, pipeline_repo):
        """Session blob is correctly written to disk from archive."""
        tmp_path, repo_id, fid1, fid2 = pipeline_repo
        from clean_room_agent.db.connection import _db_path
        from clean_room_agent.retrieval.pipeline import _restore_session_from_archive

        # Create a session, write some state, archive it
        session_conn = get_connection("session", repo_path=tmp_path, task_id="src-task")
        set_state(session_conn, "task_query", {"raw_task": "fix it", "intent_summary": "fix"})
        session_conn.close()

        # Archive the session
        session_file = _db_path(tmp_path, "session", "src-task")
        session_blob = session_file.read_bytes()
        raw_conn = get_connection("raw", repo_path=tmp_path)
        from clean_room_agent.db.raw_queries import insert_session_archive
        insert_session_archive(raw_conn, "src-task", session_blob)
        raw_conn.commit()

        # Delete the original
        session_file.unlink()
        assert not session_file.exists()

        # Restore into a new task_id
        _restore_session_from_archive(raw_conn, tmp_path, "new-task", "src-task")
        raw_conn.close()

        # Verify the restored session has the data
        new_session_file = _db_path(tmp_path, "session", "new-task")
        assert new_session_file.exists()
        new_conn = get_connection("session", repo_path=tmp_path, task_id="new-task")
        data = get_state(new_conn, "task_query")
        assert data is not None
        assert data["raw_task"] == "fix it"
        new_conn.close()

    def test_no_archive_raises(self, pipeline_repo):
        tmp_path, repo_id, fid1, fid2 = pipeline_repo
        from clean_room_agent.retrieval.pipeline import _restore_session_from_archive

        raw_conn = get_connection("raw", repo_path=tmp_path)
        with pytest.raises(RuntimeError, match="no session archive found"):
            _restore_session_from_archive(raw_conn, tmp_path, "new-task", "nonexistent")
        raw_conn.close()


class TestResumeTaskFromSession:
    def test_error_patterns_merged(self, tmp_repo):
        """B7: error_patterns from session and refinement should be merged, not replaced."""
        session_conn = get_connection("session", repo_path=tmp_repo, task_id="merge-test")
        set_state(session_conn, "task_query", {
            "raw_task": "fix bug",
            "intent_summary": "Fix the auth bug",
            "task_type": "bug_fix",
            "mentioned_files": ["auth.py"],
            "mentioned_symbols": [],
            "keywords": ["auth"],
            "error_patterns": ["TypeError: cannot read", "KeyError: missing"],
            "seed_file_ids": [],
            "seed_symbol_ids": [],
        })

        refinement = RefinementRequest(
            reason="more errors found",
            source_task_id="merge-test",
            error_patterns=["ValueError: invalid input"],
        )

        result = _resume_task_from_session(
            session_conn, refinement, "fix bug", "merge-test", "plan", 1,
        )
        # Should contain both original and new error patterns
        assert "TypeError: cannot read" in result.error_patterns
        assert "KeyError: missing" in result.error_patterns
        assert "ValueError: invalid input" in result.error_patterns
        session_conn.close()


class TestPipelineRouting:
    """Tests for LLM-based stage routing in the pipeline."""

    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_routing_logged_to_raw_db(self, mock_llm_class, pipeline_repo):
        """Routing LLM call is logged to raw DB with stage_name='stage_routing'."""
        tmp_path, repo_id, fid1, fid2 = pipeline_repo

        mock_llm_class.return_value = _make_mock_llm_instance()

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)
        task_id = "test-routing-log-001"

        run_pipeline(
            raw_task="Fix the main function in src/main.py",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id=task_id,
            config=_make_config(),
        )

        raw_conn = get_connection("raw", repo_path=tmp_path)
        try:
            calls = raw_conn.execute(
                "SELECT * FROM retrieval_llm_calls WHERE task_id = ? AND call_type = 'stage_routing'",
                (task_id,),
            ).fetchall()
            assert len(calls) == 1
            assert calls[0]["stage_name"] == "stage_routing"
        finally:
            raw_conn.close()

    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_routing_skips_stages(self, mock_llm_class, pipeline_repo):
        """When routing selects no stages, stage loop is skipped and assembly uses seeds."""
        tmp_path, repo_id, fid1, fid2 = pipeline_repo

        call_count = 0

        def _routing_skips_all(prompt, system=None):
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            response.latency_ms = 100
            response.prompt_tokens = 50
            response.completion_tokens = 20

            if system and "task analyzer" in system.lower():
                response.text = "Fix the main function in main.py"
            elif system and "stage router" in system.lower():
                response.text = json.dumps({
                    "stages": [],
                    "reasoning": "Simple targeted edit, no expansion needed.",
                })
            else:
                response.text = "Generic response"
            return response

        mock_instance = _make_mock_llm_instance()
        mock_instance.complete.side_effect = _routing_skips_all
        mock_llm_class.return_value = mock_instance

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)

        package = run_pipeline(
            raw_task="Fix the main function in src/main.py",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id="test-routing-skip-001",
            config=_make_config(),
        )

        assert package is not None
        # No scope/precision calls should have been made
        # Only task_analysis + routing + assembly calls
        for c in mock_instance.calls:
            sys = c.get("system", "") or ""
            assert "retrieval judge" not in sys.lower(), "Scope stage should not have run"
            assert "precision analyst" not in sys.lower(), "Precision stage should not have run"

    @patch("clean_room_agent.retrieval.pipeline.LoggedLLMClient")
    def test_routing_selects_subset(self, mock_llm_class, pipeline_repo):
        """When routing selects only scope, precision stage is skipped."""
        tmp_path, repo_id, fid1, fid2 = pipeline_repo

        def _routing_scope_only(prompt, system=None):
            response = MagicMock()
            response.latency_ms = 100
            response.prompt_tokens = 50
            response.completion_tokens = 20

            if system and "task analyzer" in system.lower():
                response.text = "Fix the main function in main.py"
            elif system and "stage router" in system.lower():
                response.text = json.dumps({
                    "stages": ["scope"],
                    "reasoning": "Need file expansion but not symbol classification.",
                })
            elif system and "retrieval judge" in system.lower():
                response.text = json.dumps([
                    {"path": "src/main.py", "verdict": "relevant", "reason": "mentioned"},
                ])
            else:
                response.text = "Generic response"
            return response

        mock_instance = _make_mock_llm_instance()
        mock_instance.complete.side_effect = _routing_scope_only
        mock_llm_class.return_value = mock_instance

        import clean_room_agent.retrieval.scope_stage  # noqa: F401
        import clean_room_agent.retrieval.precision_stage  # noqa: F401

        budget = BudgetConfig(context_window=32768, reserved_tokens=4096)

        package = run_pipeline(
            raw_task="Fix the main function in src/main.py",
            repo_path=tmp_path,
            stage_names=["scope", "precision"],
            budget=budget,
            mode="plan",
            task_id="test-routing-subset-001",
            config=_make_config(),
        )

        assert package is not None
        # Precision should not have run
        for c in mock_instance.calls:
            sys = c.get("system", "") or ""
            assert "precision analyst" not in sys.lower(), "Precision stage should not have run"
