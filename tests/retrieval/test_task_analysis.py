"""Tests for task analysis module."""

from unittest.mock import MagicMock

import pytest

from clean_room_agent.retrieval.task_analysis import (
    analyze_task,
    enrich_task_intent,
    extract_task_signals,
    resolve_seeds,
)


class TestExtractTaskSignals:
    def test_file_paths(self):
        signals = extract_task_signals("Fix the bug in src/auth/login.py and src/config.ts")
        assert "src/auth/login.py" in signals["files"]
        assert "src/config.ts" in signals["files"]

    def test_no_files(self):
        signals = extract_task_signals("Improve performance")
        assert signals["files"] == []

    def test_camel_case_symbols(self):
        signals = extract_task_signals("The UserManager class needs a fix")
        assert "UserManager" in signals["symbols"]

    def test_snake_case_symbols(self):
        signals = extract_task_signals("Fix the get_user_by_id function")
        assert "get_user_by_id" in signals["symbols"]

    def test_dotted_names(self):
        signals = extract_task_signals("Fix Foo.bar_method to handle nulls")
        assert "Foo.bar_method" in signals["symbols"]

    def test_error_patterns(self):
        signals = extract_task_signals(
            "Getting: TypeError: cannot read property\n"
            "Traceback at line 42"
        )
        assert len(signals["error_patterns"]) >= 1

    def test_keywords(self):
        signals = extract_task_signals("Refactor the authentication module")
        assert "refactor" in signals["keywords"] or "authentication" in signals["keywords"]

    def test_task_type_bug_fix(self):
        signals = extract_task_signals("Fix the login bug")
        assert signals["task_type"] == "bug_fix"

    def test_task_type_feature(self):
        signals = extract_task_signals("Add a new logout button")
        assert signals["task_type"] == "feature"

    def test_task_type_refactor(self):
        signals = extract_task_signals("Refactor the database layer")
        assert signals["task_type"] == "refactor"

    def test_task_type_test(self):
        signals = extract_task_signals("Write pytest coverage for auth module")
        assert signals["task_type"] == "test"

    def test_task_type_docs(self):
        signals = extract_task_signals("Update the README documentation")
        assert signals["task_type"] == "docs"

    def test_task_type_unknown(self):
        signals = extract_task_signals("Investigate the performance characteristics")
        assert signals["task_type"] == "unknown"

    def test_jsx_extension(self):
        signals = extract_task_signals("Fix src/components/App.jsx rendering")
        assert "src/components/App.jsx" in signals["files"]

    def test_tsx_extension(self):
        signals = extract_task_signals("Update components/Button.tsx styles")
        assert "components/Button.tsx" in signals["files"]

    def test_multiple_symbols(self):
        signals = extract_task_signals(
            "UserManager.get_user calls DatabaseClient.fetch_data"
        )
        symbol_names = signals["symbols"]
        assert "UserManager" in symbol_names
        assert "DatabaseClient" in symbol_names


class TestResolveSeeds:
    def test_resolve_file_path(self):
        mock_kb = MagicMock()
        mock_file = MagicMock()
        mock_file.id = 42
        mock_kb.get_file_by_path.return_value = mock_file
        mock_kb.search_symbols_by_name.return_value = []

        signals = {"files": ["src/auth.py"], "symbols": []}
        file_ids, symbol_ids = resolve_seeds(signals, mock_kb, repo_id=1)
        assert file_ids == [42]
        mock_kb.get_file_by_path.assert_called_with(1, "src/auth.py")

    def test_resolve_symbol(self):
        mock_kb = MagicMock()
        mock_kb.get_file_by_path.return_value = None
        mock_sym = MagicMock()
        mock_sym.id = 99
        mock_kb.search_symbols_by_name.return_value = [mock_sym]

        signals = {"files": [], "symbols": ["MyClass"]}
        file_ids, symbol_ids = resolve_seeds(signals, mock_kb, repo_id=1)
        assert symbol_ids == [99]

    def test_unresolved_file(self):
        mock_kb = MagicMock()
        mock_kb.get_file_by_path.return_value = None
        mock_kb.search_symbols_by_name.return_value = []

        signals = {"files": ["nonexistent.py"], "symbols": []}
        file_ids, symbol_ids = resolve_seeds(signals, mock_kb, repo_id=1)
        assert file_ids == []

    def test_unresolved_symbol(self):
        mock_kb = MagicMock()
        mock_kb.get_file_by_path.return_value = None
        mock_kb.search_symbols_by_name.return_value = []

        signals = {"files": [], "symbols": ["NoSuchThing"]}
        file_ids, symbol_ids = resolve_seeds(signals, mock_kb, repo_id=1)
        assert symbol_ids == []

    def test_like_matches_ordered_by_name_length(self):
        """T9/R6: non-exact LIKE matches must be ordered by specificity before cap."""
        mock_kb = MagicMock()
        mock_kb.get_file_by_path.return_value = None

        # Return matches in non-length order (simulating SQLite rowid order)
        # Note: MagicMock(name=...) is special, must set .name after creation
        sym_long = MagicMock()
        sym_long.id = 1
        sym_long.name = "verify_login_status"
        sym_med = MagicMock()
        sym_med.id = 2
        sym_med.name = "login_handler"
        sym_short = MagicMock()
        sym_short.id = 3
        sym_short.name = "do_login"
        mock_kb.search_symbols_by_name.return_value = [sym_long, sym_med, sym_short]

        signals = {"files": [], "symbols": ["login"]}
        _, symbol_ids = resolve_seeds(signals, mock_kb, repo_id=1)
        # Shortest name first (most specific match)
        assert symbol_ids == [3, 2, 1]

    def test_exact_matches_preferred_over_like(self):
        """R6: exact name matches take priority over LIKE wildcards."""
        mock_kb = MagicMock()
        mock_kb.get_file_by_path.return_value = None

        sym_exact = MagicMock()
        sym_exact.id = 1
        sym_exact.name = "login"
        sym_like = MagicMock()
        sym_like.id = 2
        sym_like.name = "login_handler"
        mock_kb.search_symbols_by_name.return_value = [sym_like, sym_exact]

        signals = {"files": [], "symbols": ["login"]}
        _, symbol_ids = resolve_seeds(signals, mock_kb, repo_id=1)
        # Only exact match selected
        assert symbol_ids == [1]


class TestEnrichTaskIntent:
    def test_returns_text(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Fix the authentication bug in login handler"
        mock_llm.complete.return_value = mock_response

        signals = {"files": ["auth.py"], "symbols": ["login"], "task_type": "bug_fix", "keywords": ["fix"]}
        result = enrich_task_intent("fix login bug", signals, mock_llm)
        assert result == "Fix the authentication bug in login handler"
        mock_llm.complete.assert_called_once()

    def test_includes_repo_file_tree(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Intent summary"
        mock_llm.complete.return_value = mock_response

        signals = {"files": [], "symbols": [], "task_type": "feature", "keywords": []}
        enrich_task_intent(
            "add auth", signals, mock_llm,
            repo_file_tree="src/\n  main.py\n  utils.py",
        )
        prompt_arg = mock_llm.complete.call_args[0][0]
        assert "<repo_structure>" in prompt_arg
        assert "main.py" in prompt_arg
        assert "</repo_structure>" in prompt_arg

    def test_includes_environment_brief(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Intent summary"
        mock_llm.complete.return_value = mock_response

        signals = {"files": [], "symbols": [], "task_type": "feature", "keywords": []}
        enrich_task_intent(
            "add auth", signals, mock_llm,
            environment_brief="<environment>\nOS: Linux\n</environment>",
        )
        prompt_arg = mock_llm.complete.call_args[0][0]
        assert "<environment>" in prompt_arg
        assert "OS: Linux" in prompt_arg

    def test_no_tree_no_brief_backward_compat(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Intent"
        mock_llm.complete.return_value = mock_response

        signals = {"files": [], "symbols": [], "task_type": "unknown", "keywords": []}
        enrich_task_intent("do thing", signals, mock_llm)
        prompt_arg = mock_llm.complete.call_args[0][0]
        assert "<repo_structure>" not in prompt_arg
        assert "<environment>" not in prompt_arg


class TestAnalyzeTask:
    def test_full_pipeline(self):
        mock_kb = MagicMock()
        mock_file = MagicMock()
        mock_file.id = 10
        mock_kb.get_file_by_path.return_value = mock_file
        mock_kb.search_symbols_by_name.return_value = []

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Intent summary"
        mock_llm.complete.return_value = mock_response

        tq = analyze_task(
            "Fix bug in src/auth.py", "task-001", "plan",
            mock_kb, repo_id=1, llm=mock_llm,
        )
        assert tq.raw_task == "Fix bug in src/auth.py"
        assert tq.task_id == "task-001"
        assert tq.mode == "plan"
        assert tq.repo_id == 1
        assert tq.task_type == "bug_fix"
        assert "src/auth.py" in tq.mentioned_files
        assert tq.intent_summary == "Intent summary"
        assert 10 in tq.seed_file_ids
