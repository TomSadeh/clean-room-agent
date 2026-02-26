"""Tests for indexer/library_scanner.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clean_room_agent.indexer.library_scanner import (
    LibraryFileInfo,
    LibrarySource,
    resolve_library_sources,
    scan_library,
)


def _write(path, content=""):
    """Helper: write a file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestScanLibrary:
    def test_scan_library_python_only(self, tmp_path):
        """scan_library returns only .py files, skipping non-Python files."""
        lib_dir = tmp_path / "mylib"
        _write(lib_dir / "module.py", "def foo(): pass")
        _write(lib_dir / "data.txt", "not python")
        _write(lib_dir / "config.json", "{}")
        _write(lib_dir / "readme.md", "# Readme")

        lib = LibrarySource(package_name="mylib", package_path=lib_dir)
        result = scan_library(lib)

        assert len(result) == 1
        assert result[0].relative_path == "mylib/module.py"

    def test_scan_library_skip_dirs(self, tmp_path):
        """scan_library skips tests/, __pycache__/, and examples/ directories."""
        lib_dir = tmp_path / "mylib"
        _write(lib_dir / "core.py", "x = 1")
        _write(lib_dir / "tests" / "test_core.py", "def test(): pass")
        _write(lib_dir / "test" / "test_alt.py", "def test2(): pass")
        _write(lib_dir / "__pycache__" / "core.cpython-311.py", "compiled")
        _write(lib_dir / "examples" / "demo.py", "print('demo')")
        _write(lib_dir / "docs" / "conf.py", "doc config")
        _write(lib_dir / "_vendor" / "vendored.py", "vendored code")

        lib = LibrarySource(package_name="mylib", package_path=lib_dir)
        result = scan_library(lib)

        assert len(result) == 1
        assert result[0].relative_path == "mylib/core.py"

    def test_scan_library_size_limit(self, tmp_path):
        """scan_library skips files larger than max_file_size."""
        lib_dir = tmp_path / "mylib"
        _write(lib_dir / "small.py", "x = 1")
        big_file = lib_dir / "huge.py"
        big_file.parent.mkdir(parents=True, exist_ok=True)
        big_file.write_bytes(b"x" * 2000)

        lib = LibrarySource(package_name="mylib", package_path=lib_dir)
        result = scan_library(lib, max_file_size=1000)

        assert len(result) == 1
        assert result[0].relative_path == "mylib/small.py"

    def test_scan_library_relative_paths(self, tmp_path):
        """Paths are prefixed with the package name."""
        lib_dir = tmp_path / "numpy"
        _write(lib_dir / "core" / "numeric.py", "def add(): pass")
        _write(lib_dir / "__init__.py", "")

        lib = LibrarySource(package_name="numpy", package_path=lib_dir)
        result = scan_library(lib)

        paths = {r.relative_path for r in result}
        assert "numpy/core/numeric.py" in paths
        assert "numpy/__init__.py" in paths
        # All paths should start with the package name
        for r in result:
            assert r.relative_path.startswith("numpy/")


class TestResolveLibrarySources:
    def test_resolve_explicit_paths(self, tmp_path):
        """Config with library_paths returns those paths as sources."""
        lib_a = tmp_path / "lib_a"
        lib_a.mkdir()
        lib_b = tmp_path / "lib_b"
        lib_b.mkdir()

        config = {
            "library_paths": [
                {"name": "alpha", "path": str(lib_a)},
                {"name": "beta", "path": str(lib_b)},
            ]
        }
        result = resolve_library_sources(tmp_path, config)

        assert len(result) == 2
        names = {s.package_name for s in result}
        assert names == {"alpha", "beta"}
        assert result[0].package_path == lib_a
        assert result[1].package_path == lib_b

    def test_resolve_auto_mocked(self, tmp_path):
        """Auto-resolution finds packages in site-packages via importlib.util.find_spec (2-P2-2)."""
        # Create a project file that imports 'requests'
        _write(tmp_path / "app.py", "import requests\n")

        mock_spec = MagicMock()
        mock_spec.origin = "/usr/lib/python3.11/site-packages/requests/__init__.py"

        with patch("clean_room_agent.indexer.library_scanner.importlib.util.find_spec") as mock_find:
            mock_find.return_value = mock_spec
            result = resolve_library_sources(tmp_path, {"library_sources": ["auto"]})

        # find_spec was called for 'requests'
        request_calls = [c for c in mock_find.call_args_list if c[0][0] == "requests"]
        assert len(request_calls) == 1
        # The result should include a LibrarySource for 'requests'
        names = {s.package_name for s in result}
        assert "requests" in names
        # The package_path should point to the package directory (parent of __init__.py)
        requests_source = [s for s in result if s.package_name == "requests"][0]
        assert str(requests_source.package_path).endswith("requests")

    def test_resolve_explicit_string_paths(self, tmp_path):
        """Config with plain string library_paths uses dirname as package name."""
        lib_dir = tmp_path / "mypackage"
        lib_dir.mkdir()

        config = {"library_paths": [str(lib_dir)]}
        result = resolve_library_sources(tmp_path, config)

        assert len(result) == 1
        assert result[0].package_name == "mypackage"
        assert result[0].package_path == lib_dir

    def test_resolve_nonexistent_path_skipped(self, tmp_path):
        """Nonexistent library_paths are skipped with a warning."""
        config = {"library_paths": [str(tmp_path / "does_not_exist")]}
        result = resolve_library_sources(tmp_path, config)
        assert len(result) == 0


class TestLibraryConfigValidation:
    """A13: library_paths entries with missing/empty name or path raise ValueError."""

    def test_missing_name_raises(self, tmp_path):
        config = {"library_paths": [{"path": str(tmp_path)}]}
        with pytest.raises(ValueError, match="missing or empty 'name'"):
            resolve_library_sources(tmp_path, config)

    def test_empty_name_raises(self, tmp_path):
        config = {"library_paths": [{"name": "", "path": str(tmp_path)}]}
        with pytest.raises(ValueError, match="missing or empty 'name'"):
            resolve_library_sources(tmp_path, config)

    def test_missing_path_raises(self, tmp_path):
        config = {"library_paths": [{"name": "mylib"}]}
        with pytest.raises(ValueError, match="missing or empty 'path'"):
            resolve_library_sources(tmp_path, config)

    def test_empty_path_raises(self, tmp_path):
        config = {"library_paths": [{"name": "mylib", "path": ""}]}
        with pytest.raises(ValueError, match="missing or empty 'path'"):
            resolve_library_sources(tmp_path, config)
