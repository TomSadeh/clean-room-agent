"""Tests for Phase 3 patch application."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from clean_room_agent.execute.dataclasses import PatchEdit, PatchResult
from clean_room_agent.execute.patch import (
    _check_path_traversal,
    apply_edits,
    rollback_edits,
    validate_edits,
)


@pytest.fixture
def repo_with_files(tmp_path):
    """Create a repo with sample files."""
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def bar():\n    return 1\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    return tmp_path


class TestValidateEdits:
    def test_valid_single_edit(self, repo_with_files):
        edits = [PatchEdit(file_path="a.py", search="    pass", replacement="    return 42")]
        errors = validate_edits(edits, repo_with_files)
        assert errors == []

    def test_file_not_found(self, repo_with_files):
        edits = [PatchEdit(file_path="nonexistent.py", search="x", replacement="y")]
        errors = validate_edits(edits, repo_with_files)
        assert len(errors) == 1
        assert "does not exist" in errors[0]

    def test_search_not_found(self, repo_with_files):
        edits = [PatchEdit(file_path="a.py", search="not in file", replacement="new")]
        errors = validate_edits(edits, repo_with_files)
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_duplicate_search_string(self, tmp_path):
        (tmp_path / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
        edits = [PatchEdit(file_path="dup.py", search="x = 1", replacement="x = 2")]
        errors = validate_edits(edits, tmp_path)
        assert len(errors) == 1
        assert "2 times" in errors[0]

    def test_sequential_simulation(self, repo_with_files):
        """Two edits to same file: second sees result of first."""
        edits = [
            PatchEdit(file_path="a.py", search="    pass", replacement="    return 42"),
            PatchEdit(file_path="a.py", search="    return 42", replacement="    return 99"),
        ]
        errors = validate_edits(edits, repo_with_files)
        assert errors == []

    def test_sequential_simulation_failure(self, repo_with_files):
        """Second edit depends on first but uses wrong search."""
        edits = [
            PatchEdit(file_path="a.py", search="    pass", replacement="    return 42"),
            PatchEdit(file_path="a.py", search="    pass", replacement="    return 99"),  # pass no longer exists
        ]
        errors = validate_edits(edits, repo_with_files)
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_subdirectory_file(self, repo_with_files):
        edits = [PatchEdit(file_path="sub/c.py", search="x = 1", replacement="x = 10")]
        errors = validate_edits(edits, repo_with_files)
        assert errors == []


class TestApplyEdits:
    def test_single_edit(self, repo_with_files):
        edits = [PatchEdit(file_path="a.py", search="    pass", replacement="    return 42")]
        result = apply_edits(edits, repo_with_files)
        assert result.success is True
        assert "a.py" in result.files_modified
        content = (repo_with_files / "a.py").read_text(encoding="utf-8")
        assert "return 42" in content
        assert "pass" not in content

    def test_multiple_files(self, repo_with_files):
        edits = [
            PatchEdit(file_path="a.py", search="    pass", replacement="    return 1"),
            PatchEdit(file_path="b.py", search="    return 1", replacement="    return 2"),
        ]
        result = apply_edits(edits, repo_with_files)
        assert result.success is True
        assert set(result.files_modified) == {"a.py", "b.py"}
        assert "return 1" in (repo_with_files / "a.py").read_text()
        assert "return 2" in (repo_with_files / "b.py").read_text()

    def test_multiple_edits_same_file(self, repo_with_files):
        edits = [
            PatchEdit(file_path="sub/c.py", search="x = 1", replacement="x = 10"),
            PatchEdit(file_path="sub/c.py", search="y = 2", replacement="y = 20"),
        ]
        result = apply_edits(edits, repo_with_files)
        assert result.success is True
        content = (repo_with_files / "sub" / "c.py").read_text()
        assert "x = 10" in content
        assert "y = 20" in content

    def test_deletion(self, repo_with_files):
        edits = [PatchEdit(file_path="a.py", search="    pass\n", replacement="")]
        result = apply_edits(edits, repo_with_files)
        assert result.success is True
        content = (repo_with_files / "a.py").read_text()
        assert "pass" not in content

    def test_validation_failure_returns_error(self, repo_with_files):
        edits = [PatchEdit(file_path="nonexistent.py", search="x", replacement="y")]
        result = apply_edits(edits, repo_with_files)
        assert result.success is False
        assert "does not exist" in result.error_info

    def test_saves_originals(self, repo_with_files):
        original = (repo_with_files / "a.py").read_text()
        edits = [PatchEdit(file_path="a.py", search="    pass", replacement="    return 42")]
        result = apply_edits(edits, repo_with_files)
        assert result.original_contents["a.py"] == original

    def test_sequential_edits_same_file(self, repo_with_files):
        """Edits applied in order: second edit searches in result of first."""
        edits = [
            PatchEdit(file_path="a.py", search="    pass", replacement="    x = 1"),
            PatchEdit(file_path="a.py", search="    x = 1", replacement="    x = 2"),
        ]
        result = apply_edits(edits, repo_with_files)
        assert result.success is True
        content = (repo_with_files / "a.py").read_text()
        assert "x = 2" in content
        assert "x = 1" not in content


class TestRollbackEdits:
    def test_rollback(self, repo_with_files):
        original = (repo_with_files / "a.py").read_text()
        edits = [PatchEdit(file_path="a.py", search="    pass", replacement="    return 42")]
        result = apply_edits(edits, repo_with_files)
        assert result.success is True
        assert "return 42" in (repo_with_files / "a.py").read_text()

        rollback_edits(result, repo_with_files)
        assert (repo_with_files / "a.py").read_text() == original

    def test_rollback_multiple_files(self, repo_with_files):
        orig_a = (repo_with_files / "a.py").read_text()
        orig_b = (repo_with_files / "b.py").read_text()
        edits = [
            PatchEdit(file_path="a.py", search="    pass", replacement="    return 1"),
            PatchEdit(file_path="b.py", search="    return 1", replacement="    return 2"),
        ]
        result = apply_edits(edits, repo_with_files)
        rollback_edits(result, repo_with_files)
        assert (repo_with_files / "a.py").read_text() == orig_a
        assert (repo_with_files / "b.py").read_text() == orig_b


class TestPathTraversal:
    """T43: Path traversal validation in _check_path_traversal and apply_edits."""

    def test_parent_traversal_raises(self, tmp_path):
        """Parent traversal (../../etc/passwd) raises ValueError."""
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Path traversal detected"):
            _check_path_traversal("../../etc/passwd", tmp_path)

    def test_absolute_path_escaping_repo_raises(self, tmp_path):
        """Absolute path outside repo raises ValueError."""
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        if sys.platform == "win32":
            escape_path = "C:/Windows/System32/evil.py"
        else:
            escape_path = "/etc/passwd"
        with pytest.raises(ValueError, match="Path traversal detected"):
            _check_path_traversal(escape_path, tmp_path)

    def test_valid_subdirectory_path_allowed(self, tmp_path):
        """Subdirectory paths within repo do not raise."""
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file.py").write_text("x = 1\n", encoding="utf-8")
        # Should not raise
        _check_path_traversal("sub/file.py", tmp_path)

    def test_apply_edits_rejects_traversal(self, tmp_path):
        """apply_edits raises ValueError for path traversal (fail-fast)."""
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        edits = [PatchEdit(file_path="../../etc/passwd", search="root", replacement="hacked")]
        with pytest.raises(ValueError, match="Path traversal detected"):
            apply_edits(edits, tmp_path)

    def test_validate_edits_rejects_traversal(self, tmp_path):
        """validate_edits raises ValueError on traversal."""
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        edits = [PatchEdit(file_path="../escape.py", search="x", replacement="y")]
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_edits(edits, tmp_path)

    def test_dot_segments_within_repo_ok(self, tmp_path):
        """Paths with dot segments that still resolve within repo are allowed."""
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file.py").write_text("x = 1\n", encoding="utf-8")
        # sub/../sub/file.py resolves to sub/file.py, still under repo
        _check_path_traversal("sub/../sub/file.py", tmp_path)


class TestRollbackReRaises:
    """T34: rollback_edits raises RuntimeError when a file can't be restored."""

    def test_rollback_raises_on_unwritable_path(self, tmp_path):
        """rollback_edits raises RuntimeError when the target path doesn't exist."""
        # Create a PatchResult with original_contents pointing to a non-existent directory
        fake_result = PatchResult(
            success=True,
            files_modified=["nonexistent_dir/a.py"],
            original_contents={"nonexistent_dir/a.py": "original content"},
        )
        with pytest.raises(RuntimeError, match="Rollback failed for 1 file"):
            rollback_edits(fake_result, tmp_path)

    def test_rollback_attempts_all_files_before_raising(self, tmp_path):
        """rollback_edits attempts all files, then raises with all errors."""
        fake_result = PatchResult(
            success=True,
            files_modified=["missing1/a.py", "missing2/b.py"],
            original_contents={
                "missing1/a.py": "content a",
                "missing2/b.py": "content b",
            },
        )
        with pytest.raises(RuntimeError, match="Rollback failed for 2 file") as exc_info:
            rollback_edits(fake_result, tmp_path)
        # Both files mentioned in the error
        assert "missing1/a.py" in str(exc_info.value)
        assert "missing2/b.py" in str(exc_info.value)

    def test_rollback_partial_failure_raises(self, tmp_path):
        """If one file rollbacks fine but another fails, still raises."""
        # Create one valid file and one invalid path
        (tmp_path / "a.py").write_text("modified", encoding="utf-8")
        fake_result = PatchResult(
            success=True,
            files_modified=["a.py", "nonexistent_dir/b.py"],
            original_contents={
                "a.py": "original a",
                "nonexistent_dir/b.py": "original b",
            },
        )
        with pytest.raises(RuntimeError, match="Rollback failed for 1 file"):
            rollback_edits(fake_result, tmp_path)
        # a.py should have been restored even though b.py failed
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "original a"


class TestApplyEditsTOCTOU:
    """T42: apply_edits reads files once and validates+applies atomically."""

    def test_single_read_per_file(self, repo_with_files):
        """apply_edits reads each file exactly once (not once for validate, once for apply)."""
        edits = [
            PatchEdit(file_path="a.py", search="    pass", replacement="    return 42"),
        ]
        # Track how many times read_text is called on a.py
        original_read = Path.read_text
        read_counts = {"a.py": 0}

        def counting_read(self, *args, **kwargs):
            if self.name == "a.py":
                read_counts["a.py"] += 1
            return original_read(self, *args, **kwargs)

        with patch.object(Path, "read_text", counting_read):
            result = apply_edits(edits, repo_with_files)

        assert result.success is True
        # File should only be read once (not twice as in old validate+apply pattern)
        assert read_counts["a.py"] == 1

    def test_multi_edit_same_file_single_read(self, repo_with_files):
        """Multiple edits to the same file still read it only once."""
        edits = [
            PatchEdit(file_path="sub/c.py", search="x = 1", replacement="x = 10"),
            PatchEdit(file_path="sub/c.py", search="y = 2", replacement="y = 20"),
        ]
        original_read = Path.read_text
        read_counts = {"c.py": 0}

        def counting_read(self, *args, **kwargs):
            if self.name == "c.py":
                read_counts["c.py"] += 1
            return original_read(self, *args, **kwargs)

        with patch.object(Path, "read_text", counting_read):
            result = apply_edits(edits, repo_with_files)

        assert result.success is True
        assert read_counts["c.py"] == 1

    def test_apply_validates_internally_and_returns_error(self, repo_with_files):
        """apply_edits does its own validation (not relying on separate validate_edits call)."""
        # Mix of valid and invalid edits: apply_edits should catch the bad one
        edits = [
            PatchEdit(file_path="a.py", search="    pass", replacement="    return 42"),
            PatchEdit(file_path="a.py", search="nonexistent_string", replacement="new"),
        ]
        result = apply_edits(edits, repo_with_files)
        assert result.success is False
        assert "not found" in result.error_info
        # Original file should be unchanged since apply_edits validates before writing
        content = (repo_with_files / "a.py").read_text(encoding="utf-8")
        assert "pass" in content

    def test_apply_edits_original_saved_before_write(self, repo_with_files):
        """apply_edits saves original content from its single read, before any writes."""
        original_content = (repo_with_files / "a.py").read_text(encoding="utf-8")
        edits = [PatchEdit(file_path="a.py", search="    pass", replacement="    return 42")]
        result = apply_edits(edits, repo_with_files)
        assert result.success is True
        assert result.original_contents["a.py"] == original_content


class TestAtomicWriteLineEndings:
    """T54: _atomic_write uses binary mode to preserve \\n line endings."""

    def test_preserves_lf_endings(self, tmp_path):
        """_atomic_write does not convert \\n to \\r\\n on any platform."""
        from clean_room_agent.execute.patch import _atomic_write

        content = "line1\nline2\nline3\n"
        target = tmp_path / "test.py"
        target.write_bytes(b"original\n")

        _atomic_write(target, content)

        raw_bytes = target.read_bytes()
        assert b"\r\n" not in raw_bytes
        assert raw_bytes == b"line1\nline2\nline3\n"
