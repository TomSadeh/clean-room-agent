"""Tests for audit reference task TOML loader."""

import pytest
from pathlib import Path

from clean_room_agent.audit.loader import load_reference_task, load_all_reference_tasks


@pytest.fixture
def tasks_dir(tmp_path):
    """Create a temporary directory with reference task TOML files."""
    d = tmp_path / "reference_tasks"
    d.mkdir()
    return d


def _write_task(tasks_dir, filename, content):
    """Helper to write a TOML file."""
    (tasks_dir / filename).write_text(content, encoding="utf-8")


VALID_TASK = """\
[task]
id = "RT-001"
description = "Fix a localized bug"
task_type = "bug_fix"

[context_requirements]
must_contain_files = ["src/foo.py"]
should_contain_files = ["src/bar.py"]
must_not_contain = ["src/baz/*"]
must_contain_information = ["Full source of foo()"]
budget_range = [30, 70]

[routing_notes]
reasoning = "Needs scope + precision"
"""

VALID_TASK_2 = """\
[task]
id = "RT-002"
description = "Add a feature"
task_type = "feature"

[context_requirements]
must_contain_files = ["src/feature.py"]
should_contain_files = []
must_not_contain = []
must_contain_information = []
budget_range = [20, 80]

[routing_notes]
reasoning = ""
"""

MINIMAL_TASK = """\
[task]
id = "RT-003"
description = "Minimal task"
task_type = "refactor"

[context_requirements]
must_contain_files = ["src/a.py"]
should_contain_files = []
must_not_contain = []
must_contain_information = []
budget_range = [20, 80]

[routing_notes]
reasoning = ""
"""


class TestLoadReferenceTask:
    def test_valid_full(self, tasks_dir):
        _write_task(tasks_dir, "rt_001.toml", VALID_TASK)
        rt = load_reference_task(tasks_dir / "rt_001.toml")
        assert rt.id == "RT-001"
        assert rt.description == "Fix a localized bug"
        assert rt.task_type == "bug_fix"
        assert rt.must_contain_files == ["src/foo.py"]
        assert rt.should_contain_files == ["src/bar.py"]
        assert rt.must_not_contain == ["src/baz/*"]
        assert rt.must_contain_information == ["Full source of foo()"]
        assert rt.budget_range == (30, 70)
        assert rt.routing_reasoning == "Needs scope + precision"

    def test_minimal(self, tasks_dir):
        _write_task(tasks_dir, "rt_003.toml", MINIMAL_TASK)
        rt = load_reference_task(tasks_dir / "rt_003.toml")
        assert rt.id == "RT-003"
        assert rt.budget_range == (20, 80)
        assert rt.should_contain_files == []
        assert rt.must_not_contain == []

    def test_missing_task_section(self, tasks_dir):
        _write_task(tasks_dir, "bad.toml", "[context_requirements]\nmust_contain_files = ['a.py']")
        with pytest.raises(ValueError, match="missing \\[task\\] section"):
            load_reference_task(tasks_dir / "bad.toml")

    def test_missing_context_requirements(self, tasks_dir):
        _write_task(tasks_dir, "bad.toml", '[task]\nid = "RT-X"\ndescription = "x"\ntask_type = "bug_fix"')
        with pytest.raises(ValueError, match="missing \\[context_requirements\\] section"):
            load_reference_task(tasks_dir / "bad.toml")

    def test_missing_task_id(self, tasks_dir):
        content = """\
[task]
description = "No id"
task_type = "bug_fix"

[context_requirements]
must_contain_files = ["a.py"]
should_contain_files = []
must_not_contain = []
must_contain_information = []
budget_range = [20, 80]

[routing_notes]
reasoning = ""
"""
        _write_task(tasks_dir, "bad.toml", content)
        with pytest.raises(ValueError, match="missing task.id"):
            load_reference_task(tasks_dir / "bad.toml")

    def test_missing_description(self, tasks_dir):
        content = """\
[task]
id = "RT-X"
task_type = "bug_fix"

[context_requirements]
must_contain_files = ["a.py"]
should_contain_files = []
must_not_contain = []
must_contain_information = []
budget_range = [20, 80]

[routing_notes]
reasoning = ""
"""
        _write_task(tasks_dir, "bad.toml", content)
        with pytest.raises(ValueError, match="missing task.description"):
            load_reference_task(tasks_dir / "bad.toml")

    def test_missing_task_type(self, tasks_dir):
        content = """\
[task]
id = "RT-X"
description = "No type"

[context_requirements]
must_contain_files = ["a.py"]
should_contain_files = []
must_not_contain = []
must_contain_information = []
budget_range = [20, 80]

[routing_notes]
reasoning = ""
"""
        _write_task(tasks_dir, "bad.toml", content)
        with pytest.raises(ValueError, match="missing task.task_type"):
            load_reference_task(tasks_dir / "bad.toml")

    def test_invalid_budget_range_not_list(self, tasks_dir):
        content = """\
[task]
id = "RT-X"
description = "Bad budget"
task_type = "bug_fix"

[context_requirements]
must_contain_files = ["a.py"]
should_contain_files = []
must_not_contain = []
must_contain_information = []
budget_range = 50

[routing_notes]
reasoning = ""
"""
        _write_task(tasks_dir, "bad.toml", content)
        with pytest.raises(ValueError, match="budget_range must be a two-element array"):
            load_reference_task(tasks_dir / "bad.toml")

    def test_invalid_budget_range_three_elements(self, tasks_dir):
        content = """\
[task]
id = "RT-X"
description = "Bad budget"
task_type = "bug_fix"

[context_requirements]
must_contain_files = ["a.py"]
should_contain_files = []
must_not_contain = []
must_contain_information = []
budget_range = [20, 50, 80]

[routing_notes]
reasoning = ""
"""
        _write_task(tasks_dir, "bad.toml", content)
        with pytest.raises(ValueError, match="budget_range must be a two-element array"):
            load_reference_task(tasks_dir / "bad.toml")


class TestLoadAllReferenceTasks:
    def test_load_multiple(self, tasks_dir):
        _write_task(tasks_dir, "rt_001.toml", VALID_TASK)
        _write_task(tasks_dir, "rt_002.toml", VALID_TASK_2)
        tasks = load_all_reference_tasks(tasks_dir=tasks_dir)
        assert len(tasks) == 2
        assert tasks[0].id == "RT-001"
        assert tasks[1].id == "RT-002"

    def test_sorted_by_id(self, tasks_dir):
        _write_task(tasks_dir, "z_task.toml", VALID_TASK_2)
        _write_task(tasks_dir, "a_task.toml", VALID_TASK)
        tasks = load_all_reference_tasks(tasks_dir=tasks_dir)
        assert tasks[0].id == "RT-001"
        assert tasks[1].id == "RT-002"

    def test_empty_dir_raises(self, tasks_dir):
        with pytest.raises(FileNotFoundError, match="No .toml files"):
            load_all_reference_tasks(tasks_dir=tasks_dir)

    def test_nonexistent_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_all_reference_tasks(tasks_dir=tmp_path / "nonexistent")

    def test_duplicate_ids_raises(self, tasks_dir):
        _write_task(tasks_dir, "rt_001a.toml", VALID_TASK)
        _write_task(tasks_dir, "rt_001b.toml", VALID_TASK)  # same ID
        with pytest.raises(ValueError, match="Duplicate reference task IDs"):
            load_all_reference_tasks(tasks_dir=tasks_dir)

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Either tasks_dir or repo_path"):
            load_all_reference_tasks()

    def test_repo_path_resolution(self, tmp_path):
        d = tmp_path / "protocols" / "retrieval_audit" / "reference_tasks"
        d.mkdir(parents=True)
        _write_task(d, "rt_001.toml", VALID_TASK)
        tasks = load_all_reference_tasks(repo_path=tmp_path)
        assert len(tasks) == 1
