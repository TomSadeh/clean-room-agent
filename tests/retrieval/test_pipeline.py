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
from clean_room_agent.retrieval.dataclasses import BudgetConfig, RefinementRequest
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
    else:
        response.text = "Generic response"
    return response


def _make_mock_llm_instance():
    """Create a mock LLM instance with config attributes for batching."""
    mock_instance = MagicMock()
    mock_instance.complete.side_effect = _mock_llm_complete
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=False)
    mock_instance.config.context_window = 32768
    mock_instance.config.max_tokens = 4096
    return mock_instance


class TestRunPipeline:
    @patch("clean_room_agent.retrieval.pipeline.LLMClient")
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

    @patch("clean_room_agent.retrieval.pipeline.LLMClient")
    def test_session_state_saved(self, mock_llm_class, pipeline_repo):
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

        # Verify session DB has expected keys
        session_conn = get_connection("session", repo_path=tmp_path, task_id=task_id)
        try:
            task_query = get_state(session_conn, "task_query")
            assert task_query is not None
            final_ctx = get_state(session_conn, "final_context")
            assert final_ctx is not None
            progress = get_state(session_conn, "stage_progress")
            assert progress is not None
        finally:
            session_conn.close()

    @patch("clean_room_agent.retrieval.pipeline.LLMClient")
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

    @patch("clean_room_agent.retrieval.pipeline.LLMClient")
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

    @patch("clean_room_agent.retrieval.pipeline.LLMClient")
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
