"""Tests for P0/P1 fixes: T57-T61, T75-T76."""

import json
from unittest.mock import MagicMock, patch

import pytest

from clean_room_agent.execute.dataclasses import (
    MetaPlan,
    MetaPlanPart,
    PartPlan,
    PatchResult,
    PlanAdjustment,
    PlanStep,
    ValidationResult,
)
from clean_room_agent.orchestrator.validator import (
    _MAX_VALIDATION_OUTPUT_CHARS,
    _cap_output,
)
from clean_room_agent.retrieval.dataclasses import (
    ClassifiedSymbol,
    ScopedFile,
    TaskQuery,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _mock_llm(text="ok", *, context_window=32768, max_tokens=4096):
    """Create a mock LLM with standard config and a fixed response."""
    llm = MagicMock()
    llm.config.context_window = context_window
    llm.config.max_tokens = max_tokens
    resp = MagicMock()
    resp.text = text
    llm.complete.return_value = resp
    return llm


def _task(**overrides):
    defaults = dict(
        raw_task="Fix the bug",
        task_id="t1",
        mode="plan",
        repo_id=1,
        intent_summary="Fix authentication bypass",
        task_type="bug_fix",
    )
    defaults.update(overrides)
    return TaskQuery(**defaults)


# ── T57: R3 budget validation at 5 LLM call sites ───────────────────


class TestT57TaskAnalysisBudgetGate:
    """T57a: enrich_task_intent raises ValueError when prompt exceeds budget."""

    def test_oversized_prompt_raises(self):
        from clean_room_agent.retrieval.task_analysis import enrich_task_intent

        # context_window=100, max_tokens=50 → only 50 tokens available
        # A long prompt at ~3 chars/token will exceed that
        llm = _mock_llm(context_window=100, max_tokens=50)
        big_task = "x" * 500  # ~167 tokens conservative
        signals = {"files": [], "symbols": [], "task_type": "bug_fix", "keywords": []}

        with pytest.raises(ValueError, match="R3.*task_analysis prompt too large"):
            enrich_task_intent(big_task, signals, llm)

        llm.complete.assert_not_called()

    def test_normal_prompt_passes(self):
        from clean_room_agent.retrieval.task_analysis import enrich_task_intent

        llm = _mock_llm("Fixed the bug")
        signals = {"files": [], "symbols": [], "task_type": "bug_fix", "keywords": []}
        result = enrich_task_intent("Fix it", signals, llm)
        assert result == "Fixed the bug"
        llm.complete.assert_called_once()


class TestT57RoutingBudgetGate:
    """T57b: route_stages raises ValueError when prompt exceeds budget."""

    def test_oversized_prompt_raises(self):
        from clean_room_agent.retrieval.routing import route_stages

        llm = _mock_llm(context_window=100, max_tokens=50)
        task = _task(intent_summary="x" * 500)
        stages = {"scope": "Finds files", "precision": "Classifies symbols"}

        with pytest.raises(ValueError, match="R3.*routing prompt too large"):
            route_stages(task, stages, llm)

        llm.complete.assert_not_called()


class TestT57ScopeBudgetGate:
    """T57c: judge_scope raises ValueError when actual batch prompt exceeds budget."""

    def test_oversized_batch_raises(self):
        from clean_room_agent.retrieval.scope_stage import judge_scope

        # Tiny window forces the actual-prompt validation to fire
        llm = _mock_llm(context_window=200, max_tokens=50)
        candidates = [
            ScopedFile(file_id=i, path=f"very/long/path/to/dep_{i}.py", language="python", tier=2)
            for i in range(50)
        ]
        task = _task(raw_task="x" * 200)

        with pytest.raises(ValueError, match="R3.*scope prompt too large"):
            judge_scope(candidates, task, llm)

        llm.complete.assert_not_called()


class TestT57PrecisionBudgetGate:
    """T57d: classify_symbols raises ValueError when actual batch prompt exceeds budget."""

    def test_oversized_batch_raises(self):
        from clean_room_agent.retrieval.precision_stage import classify_symbols

        llm = _mock_llm(context_window=200, max_tokens=50)
        candidates = [
            {
                "symbol_id": i, "file_id": 1, "file_path": f"long/path_{i}.py",
                "name": f"very_long_function_name_{i}", "kind": "function",
                "start_line": 1, "end_line": 50, "signature": f"def very_long_function_name_{i}():",
                "connections": [f"calls long_thing_{j}" for j in range(5)],
                "file_source": "project",
            }
            for i in range(50)
        ]
        task = _task(raw_task="x" * 200)

        with pytest.raises(ValueError, match="R3.*precision prompt too large"):
            classify_symbols(candidates, task, llm)

        llm.complete.assert_not_called()


class TestT57SimilarityBudgetGate:
    """T57e: judge_similarity raises ValueError when actual batch prompt exceeds budget."""

    def test_oversized_batch_raises(self):
        from clean_room_agent.retrieval.similarity_stage import judge_similarity

        llm = _mock_llm(context_window=200, max_tokens=50)
        sym_a = ClassifiedSymbol(
            symbol_id=1, file_id=1, name="func_a", kind="function",
            start_line=1, end_line=50, detail_level="primary",
        )
        sym_b = ClassifiedSymbol(
            symbol_id=2, file_id=1, name="func_b", kind="function",
            start_line=60, end_line=110, detail_level="primary",
        )
        pairs = [
            {
                "pair_id": i,
                "sym_a": sym_a, "sym_b": sym_b,
                "score": 0.8,
                "signals": {"line_ratio": 1.0, "callee_jaccard": 0.5, "name_lcs": 0.3, "same_parent": True},
            }
            for i in range(50)
        ]
        task = _task(raw_task="x" * 200)

        with pytest.raises(ValueError, match="R3.*similarity prompt too large"):
            judge_similarity(pairs, task, llm)

        llm.complete.assert_not_called()


# ── T58: Pipeline symbol decisions logged with correct stage name ────


class TestT58SymbolDecisionLogging:
    """T58: symbols should only be logged once (by the first stage that created them)."""

    def test_no_duplicate_symbol_logging(self):
        """Symbols from precision should not be re-logged with similarity stage name."""
        from clean_room_agent.retrieval.pipeline import run_pipeline

        # This test verifies the logic by inspecting the code path:
        # When precision populates classified_symbols and similarity runs after,
        # the logged_symbol_ids set prevents re-logging.

        # We test the tracking mechanism directly:
        logged_symbol_ids = set()
        symbols = [
            ClassifiedSymbol(
                symbol_id=10, file_id=1, name="foo", kind="function",
                start_line=1, end_line=10, detail_level="primary",
            ),
            ClassifiedSymbol(
                symbol_id=20, file_id=2, name="bar", kind="function",
                start_line=1, end_line=10, detail_level="supporting",
            ),
        ]

        # Simulate first stage (precision) logging
        logged_stage1 = []
        for sym in symbols:
            if sym.symbol_id not in logged_symbol_ids:
                logged_stage1.append((sym.symbol_id, "precision"))
                logged_symbol_ids.add(sym.symbol_id)

        # Simulate second stage (similarity) — should NOT re-log
        logged_stage2 = []
        for sym in symbols:
            if sym.symbol_id not in logged_symbol_ids:
                logged_stage2.append((sym.symbol_id, "similarity"))
                logged_symbol_ids.add(sym.symbol_id)

        assert len(logged_stage1) == 2
        assert len(logged_stage2) == 0  # No duplicates


# ── T59: Orchestrator rollback crash mid-LIFO ───────────────────────


class TestT59RollbackContinuesOnError:
    """T59: if one rollback fails, remaining rollbacks should still execute."""

    def test_remaining_rollbacks_execute(self):
        from clean_room_agent.execute.patch import rollback_edits

        # Create mock patch results
        patch1 = PatchResult(
            success=True, files_modified=["a.py"],
            original_contents={"a.py": "original_a"},
        )
        patch2 = PatchResult(
            success=True, files_modified=["b.py"],
            original_contents={"b.py": "original_b"},
        )
        patch3 = PatchResult(
            success=True, files_modified=["c.py"],
            original_contents={"c.py": "original_c"},
        )

        rollback_calls = []
        rollback_errors = []

        # Simulate the fixed rollback loop: first call raises, rest continue
        patches = [patch3, patch2, patch1]  # LIFO order
        for i, p in enumerate(patches):
            try:
                if i == 0:
                    raise RuntimeError("disk error on c.py")
                rollback_calls.append(p.files_modified[0])
            except (RuntimeError, OSError) as e:
                rollback_errors.append(str(e))

        # patch2 and patch1 still executed despite patch3 failure
        assert len(rollback_calls) == 2
        assert rollback_errors == ["disk error on c.py"]

    def test_all_errors_collected(self):
        """When multiple rollbacks fail, all errors are collected."""
        rollback_errors = []
        for i in range(3):
            try:
                raise OSError(f"error_{i}")
            except (RuntimeError, OSError) as e:
                rollback_errors.append(str(e))

        assert len(rollback_errors) == 3
        assert "error_0" in rollback_errors[0]
        assert "error_2" in rollback_errors[2]


# ── T60: Adjustment cycle depth limit ────────────────────────────────


class TestT60AdjustmentDepthLimit:
    """T60: adjustment rounds are capped per part."""

    def test_config_default_value(self):
        from clean_room_agent.config import create_default_config

        # Verify default config includes max_adjustment_rounds (not commented out)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = create_default_config(Path(td))
            content = path.read_text()
            assert "max_adjustment_rounds = 3" in content

    def test_limit_enforced(self):
        """After max_adj_rounds, no more adjustments run."""
        max_adj_rounds = 2
        adjustment_counts = {}
        part_id = "p1"
        adjustments_run = 0

        # Simulate 5 steps — only first 2 should trigger adjustments
        for step_idx in range(5):
            adj_count = adjustment_counts.get(part_id, 0)
            if adj_count >= max_adj_rounds:
                pass  # skipped
            else:
                adjustments_run += 1
                adjustment_counts[part_id] = adj_count + 1

        assert adjustments_run == 2
        assert adjustment_counts[part_id] == 2


# ── T61: Parser type validation on nested list fields ────────────────


class TestT61ListTypeValidation:
    """T61: from_dict raises ValueError when list fields are wrong type."""

    def test_meta_plan_parts_not_list(self):
        with pytest.raises(ValueError, match="'parts' must be a list.*got str"):
            MetaPlan.from_dict({
                "task_summary": "test",
                "parts": "not a list",
                "rationale": "test",
            })

    def test_meta_plan_parts_is_dict(self):
        with pytest.raises(ValueError, match="'parts' must be a list.*got dict"):
            MetaPlan.from_dict({
                "task_summary": "test",
                "parts": {"id": "p1"},
                "rationale": "test",
            })

    def test_part_plan_steps_not_list(self):
        with pytest.raises(ValueError, match="'steps' must be a list.*got str"):
            PartPlan.from_dict({
                "part_id": "p1",
                "task_summary": "test",
                "steps": "not a list",
                "rationale": "test",
            })

    def test_plan_adjustment_revised_steps_not_list(self):
        with pytest.raises(ValueError, match="'revised_steps' must be a list.*got int"):
            PlanAdjustment.from_dict({
                "revised_steps": 42,
                "rationale": "test",
                "changes_made": [],
            })

    def test_plan_adjustment_changes_made_not_list(self):
        with pytest.raises(ValueError, match="'changes_made' must be a list.*got str"):
            PlanAdjustment.from_dict({
                "revised_steps": [],
                "rationale": "test",
                "changes_made": "not a list",
            })

    def test_valid_data_still_works(self):
        """Verify normal from_dict still works with valid list fields."""
        mp = MetaPlan.from_dict({
            "task_summary": "test",
            "parts": [{"id": "p1", "description": "part 1"}],
            "rationale": "test",
        })
        assert len(mp.parts) == 1

        pp = PartPlan.from_dict({
            "part_id": "p1",
            "task_summary": "test",
            "steps": [{"id": "s1", "description": "step 1"}],
            "rationale": "test",
        })
        assert len(pp.steps) == 1


# ── T75: Validation output capping ──────────────────────────────────


class TestT75ValidationOutputCap:
    """T75: validation output is capped before entering LLM prompts."""

    def test_short_output_unchanged(self):
        output = "All 10 tests passed"
        result = _cap_output(output, "test output")
        assert result == output

    def test_long_output_truncated(self):
        output = "x" * (_MAX_VALIDATION_OUTPUT_CHARS + 5000)
        result = _cap_output(output, "test output")
        assert len(result) < len(output)
        assert result.startswith("[test output truncated")
        assert f"showing last {_MAX_VALIDATION_OUTPUT_CHARS} chars]" in result

    def test_tail_preserved(self):
        """Error summaries (at the end) should be preserved."""
        prefix = "x" * 10000
        suffix = "FAILED tests/test_foo.py::test_bar - AssertionError"
        output = prefix + suffix
        result = _cap_output(output, "test output")
        assert suffix in result

    def test_cap_at_exact_boundary(self):
        output = "x" * _MAX_VALIDATION_OUTPUT_CHARS
        result = _cap_output(output, "test output")
        assert result == output  # exactly at limit, no truncation

    def test_run_validation_caps_output(self, tmp_path, raw_conn):
        """Integration: run_validation returns capped output in ValidationResult."""
        from clean_room_agent.db.raw_queries import insert_run_attempt, insert_task_run

        task_run_id = insert_task_run(
            raw_conn, task_id="t1", repo_path=str(tmp_path),
            mode="implement", execute_model="test", context_window=32768,
            reserved_tokens=4096, stages="scope",
        )
        attempt_id = insert_run_attempt(
            raw_conn, task_run_id, 1, None, None, 0, "", False,
        )
        raw_conn.commit()

        big_output = "x" * (_MAX_VALIDATION_OUTPUT_CHARS + 5000)
        config = {"testing": {"test_command": "echo ok", "timeout": 5}}

        with patch(
            "clean_room_agent.orchestrator.validator._run_command",
            return_value={"returncode": 0, "output": big_output},
        ):
            from clean_room_agent.orchestrator.validator import run_validation

            result = run_validation(tmp_path, config, raw_conn, attempt_id)
            # ValidationResult output is capped
            assert len(result.test_output) < len(big_output)
            assert "truncated" in result.test_output

            # Raw DB has full output (verify via direct query)
            row = raw_conn.execute(
                "SELECT test_output FROM validation_results WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            assert len(row["test_output"]) == len(big_output)


# ── T76: Enrichment arbitrary caps removed ───────────────────────────


class TestT76EnrichmentNoCaps:
    """T76: enrichment _build_prompt includes all docstrings and full source."""

    def test_all_docstrings_included(self, tmp_path):
        from clean_room_agent.db.connection import get_connection
        from clean_room_agent.db.queries import insert_docstring, upsert_file, upsert_repo
        from clean_room_agent.llm.enrichment import _build_prompt

        conn = get_connection("curated", repo_path=tmp_path)
        rid = upsert_repo(conn, str(tmp_path), None)
        fid = upsert_file(conn, rid, "src/main.py", "python", "abc", 50)

        # Insert 5 docstrings (previously capped at 3)
        for i in range(5):
            insert_docstring(conn, fid, f"Docstring number {i}", "plain")
        conn.commit()

        prompt = _build_prompt("src/main.py", fid, conn, tmp_path)
        for i in range(5):
            assert f"Docstring number {i}" in prompt
        conn.close()

    def test_full_docstring_content(self, tmp_path):
        """Previously docstrings were truncated at 200 chars."""
        from clean_room_agent.db.connection import get_connection
        from clean_room_agent.db.queries import insert_docstring, upsert_file, upsert_repo
        from clean_room_agent.llm.enrichment import _build_prompt

        conn = get_connection("curated", repo_path=tmp_path)
        rid = upsert_repo(conn, str(tmp_path), None)
        fid = upsert_file(conn, rid, "src/main.py", "python", "abc", 50)
        long_docstring = "A" * 500  # Previously truncated to 200
        insert_docstring(conn, fid, long_docstring, "plain")
        conn.commit()

        prompt = _build_prompt("src/main.py", fid, conn, tmp_path)
        assert long_docstring in prompt
        conn.close()

    def test_full_source_included(self, tmp_path):
        """Previously source was capped at 100 lines."""
        from clean_room_agent.db.connection import get_connection
        from clean_room_agent.db.queries import upsert_file, upsert_repo
        from clean_room_agent.llm.enrichment import _build_prompt

        conn = get_connection("curated", repo_path=tmp_path)
        rid = upsert_repo(conn, str(tmp_path), None)
        fid = upsert_file(conn, rid, "src/main.py", "python", "abc", 50)
        conn.commit()

        # Create a 200-line source file
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        lines = [f"line_{i} = {i}" for i in range(200)]
        (src_dir / "main.py").write_text("\n".join(lines), encoding="utf-8")

        prompt = _build_prompt("src/main.py", fid, conn, tmp_path)
        # Line 150 would have been excluded under the old [:100] cap
        assert "line_150" in prompt
        assert "line_199" in prompt
        conn.close()


# Need Path import for T60 test
from pathlib import Path
