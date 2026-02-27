"""Tests for environment.py — EnvironmentBrief, CODING_STYLES, builders."""

from unittest.mock import MagicMock

import pytest

from clean_room_agent.environment import (
    CODING_STYLES,
    EnvironmentBrief,
    build_environment_brief,
    build_repo_file_tree,
)


class TestCodingStyles:
    def test_all_keys_present(self):
        assert "development" in CODING_STYLES
        assert "maintenance" in CODING_STYLES
        assert "prototyping" in CODING_STYLES

    def test_values_are_nonempty_strings(self):
        for key, val in CODING_STYLES.items():
            assert isinstance(val, str)
            assert len(val) > 10, f"{key} style text is too short"


class TestEnvironmentBrief:
    def test_basic_rendering(self):
        brief = EnvironmentBrief(
            os_name="Windows",
            languages={"python": 42, "typescript": 10},
            test_framework="pytest",
            coding_style="development",
            file_count=52,
            runtime_version="Python 3.13.3",
        )
        text = brief.to_prompt_text()
        assert "<environment>" in text
        assert "</environment>" in text
        assert "OS: Windows" in text
        assert "python (42 files)" in text
        assert "typescript (10 files)" in text
        assert "Test framework: pytest" in text
        assert "Runtime: Python 3.13.3" in text
        assert "Files indexed: 52" in text
        assert "Development mode" in text

    def test_empty_languages(self):
        brief = EnvironmentBrief(os_name="Linux")
        text = brief.to_prompt_text()
        assert "Languages:" not in text
        assert "OS: Linux" in text

    def test_no_test_framework(self):
        brief = EnvironmentBrief(os_name="Darwin", test_framework="")
        text = brief.to_prompt_text()
        assert "Test framework:" not in text

    def test_no_runtime_version(self):
        brief = EnvironmentBrief(os_name="Linux", runtime_version=None)
        text = brief.to_prompt_text()
        assert "Runtime:" not in text

    def test_maintenance_style(self):
        brief = EnvironmentBrief(os_name="Linux", coding_style="maintenance")
        text = brief.to_prompt_text()
        assert "Maintenance mode" in text

    def test_prototyping_style(self):
        brief = EnvironmentBrief(os_name="Linux", coding_style="prototyping")
        text = brief.to_prompt_text()
        assert "Prototyping mode" in text

    def test_unknown_style_raises(self):
        """Invalid coding_style is a bug — no silent fallback to development."""
        brief = EnvironmentBrief(os_name="Linux", coding_style="nonexistent")
        with pytest.raises(KeyError, match="nonexistent"):
            brief.to_prompt_text()

    def test_languages_sorted_by_count_descending(self):
        brief = EnvironmentBrief(
            os_name="Linux",
            languages={"python": 5, "javascript": 50, "typescript": 20},
        )
        text = brief.to_prompt_text()
        js_pos = text.index("javascript")
        ts_pos = text.index("typescript")
        py_pos = text.index("python")
        assert js_pos < ts_pos < py_pos


class TestBuildEnvironmentBrief:
    def _make_mock_kb(self, language_counts=None, file_count=10):
        kb = MagicMock()
        overview = MagicMock()
        overview.language_counts = language_counts or {"python": 10}
        overview.file_count = file_count
        kb.get_repo_overview.return_value = overview
        return kb

    def test_basic_build(self):
        kb = self._make_mock_kb({"python": 30, "javascript": 5}, file_count=35)
        config = {"testing": {"test_command": "pytest tests/"}, "environment": {"coding_style": "development"}}
        brief = build_environment_brief(config, kb, repo_id=1)

        assert brief.languages == {"python": 30, "javascript": 5}
        assert brief.test_framework == "pytest"
        assert brief.file_count == 35
        assert brief.coding_style == "development"
        assert brief.runtime_version is not None  # python is in language_counts
        assert "Python" in brief.runtime_version

    def test_custom_coding_style(self):
        kb = self._make_mock_kb()
        config = {"testing": {"test_command": ""}, "environment": {"coding_style": "maintenance"}}
        brief = build_environment_brief(config, kb, repo_id=1)
        assert brief.coding_style == "maintenance"

    def test_invalid_coding_style_raises(self):
        kb = self._make_mock_kb()
        config = {"testing": {"test_command": ""}, "environment": {"coding_style": "yolo"}}
        with pytest.raises(ValueError, match="Unknown coding_style"):
            build_environment_brief(config, kb, repo_id=1)

    def test_no_environment_section_defaults(self):
        kb = self._make_mock_kb()
        config = {"testing": {"test_command": ""}, "environment": {"coding_style": "development"}}
        brief = build_environment_brief(config, kb, repo_id=1)
        assert brief.coding_style == "development"

    def test_no_testing_section_empty_framework(self):
        kb = self._make_mock_kb()
        config = {"testing": {"test_command": ""}, "environment": {"coding_style": "development"}}
        brief = build_environment_brief(config, kb, repo_id=1)
        assert brief.test_framework == ""

    def test_no_python_no_runtime(self):
        kb = self._make_mock_kb({"typescript": 20}, file_count=20)
        config = {"testing": {"test_command": ""}, "environment": {"coding_style": "development"}}
        brief = build_environment_brief(config, kb, repo_id=1)
        assert brief.runtime_version is None


class TestBuildRepoFileTree:
    def _make_mock_kb(self, paths):
        kb = MagicMock()
        files = []
        for p in paths:
            f = MagicMock()
            f.path = p
            files.append(f)
        kb.get_files.return_value = files
        return kb

    def test_empty_repo(self):
        kb = self._make_mock_kb([])
        result = build_repo_file_tree(kb, repo_id=1)
        assert result == "(empty repository)"

    def test_flat_files(self):
        kb = self._make_mock_kb(["a.py", "b.py", "c.py"])
        result = build_repo_file_tree(kb, repo_id=1)
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "a.py"
        assert lines[1] == "b.py"
        assert lines[2] == "c.py"

    def test_nested_files(self):
        kb = self._make_mock_kb(["src/main.py", "src/utils.py", "tests/test_main.py"])
        result = build_repo_file_tree(kb, repo_id=1)
        # src/main.py has 2 parts -> 1 level indent (2 spaces)
        assert "  main.py" in result
        assert "  utils.py" in result
        assert "  test_main.py" in result

    def test_sorted_output(self):
        kb = self._make_mock_kb(["z.py", "a.py", "m.py"])
        result = build_repo_file_tree(kb, repo_id=1)
        lines = result.split("\n")
        assert lines[0] == "a.py"
        assert lines[1] == "m.py"
        assert lines[2] == "z.py"

    def test_deep_nesting(self):
        kb = self._make_mock_kb(["a/b/c/d.py"])
        result = build_repo_file_tree(kb, repo_id=1)
        # 4 parts = 3 levels of indent (6 spaces)
        assert "      d.py" in result
