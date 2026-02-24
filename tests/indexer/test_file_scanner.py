"""Tests for indexer/file_scanner.py."""

import hashlib

import pytest

from clean_room_agent.indexer.file_scanner import FileInfo, scan_repo


def _write(path, content=""):
    """Helper: write a file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestLanguageDetection:
    def test_python(self, tmp_path):
        _write(tmp_path / "main.py", "x = 1")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].language == "python"

    def test_typescript_ts(self, tmp_path):
        _write(tmp_path / "app.ts", "const x = 1;")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].language == "typescript"

    def test_typescript_tsx(self, tmp_path):
        _write(tmp_path / "component.tsx", "export default () => <div/>;")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].language == "typescript"

    def test_javascript_js(self, tmp_path):
        _write(tmp_path / "index.js", "module.exports = {};")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].language == "javascript"

    def test_javascript_jsx(self, tmp_path):
        _write(tmp_path / "app.jsx", "export default () => <div/>;")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].language == "javascript"

    def test_javascript_mjs(self, tmp_path):
        _write(tmp_path / "lib.mjs", "export const x = 1;")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].language == "javascript"

    def test_javascript_cjs(self, tmp_path):
        _write(tmp_path / "lib.cjs", "module.exports = {};")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].language == "javascript"


class TestUnknownExtensions:
    def test_txt_skipped(self, tmp_path):
        _write(tmp_path / "readme.txt", "hello")
        assert scan_repo(tmp_path) == []

    def test_md_skipped(self, tmp_path):
        _write(tmp_path / "notes.md", "# notes")
        assert scan_repo(tmp_path) == []

    def test_json_skipped(self, tmp_path):
        _write(tmp_path / "data.json", "{}")
        assert scan_repo(tmp_path) == []

    def test_no_extension_skipped(self, tmp_path):
        _write(tmp_path / "Makefile", "all:")
        assert scan_repo(tmp_path) == []


class TestDirectorySkipping:
    def test_git_dir_skipped(self, tmp_path):
        _write(tmp_path / ".git" / "config.py", "x = 1")
        _write(tmp_path / "real.py", "y = 2")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].path == "real.py"

    def test_node_modules_skipped(self, tmp_path):
        _write(tmp_path / "node_modules" / "pkg" / "index.js", "x")
        _write(tmp_path / "app.js", "y")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].path == "app.js"

    def test_pycache_skipped(self, tmp_path):
        _write(tmp_path / "__pycache__" / "mod.py", "x")
        _write(tmp_path / "mod.py", "y")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].path == "mod.py"

    def test_venv_skipped(self, tmp_path):
        _write(tmp_path / ".venv" / "lib" / "site.py", "x")
        _write(tmp_path / "venv" / "lib" / "site.py", "x")
        _write(tmp_path / "app.py", "y")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].path == "app.py"

    def test_clean_room_skipped(self, tmp_path):
        _write(tmp_path / ".clean_room" / "internal.py", "x")
        _write(tmp_path / "app.py", "y")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].path == "app.py"

    def test_egg_info_skipped(self, tmp_path):
        _write(tmp_path / "mylib.egg-info" / "PKG-INFO.py", "x")
        _write(tmp_path / "app.py", "y")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].path == "app.py"


class TestFileSkipping:
    def test_pyc_skipped(self, tmp_path):
        (tmp_path / "mod.pyc").write_bytes(b"\x00")
        _write(tmp_path / "mod.py", "x = 1")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].path == "mod.py"

    def test_min_js_skipped(self, tmp_path):
        _write(tmp_path / "bundle.min.js", "minified")
        _write(tmp_path / "app.js", "real")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].path == "app.js"

    def test_source_map_skipped(self, tmp_path):
        _write(tmp_path / "bundle.js.map", "{}")
        _write(tmp_path / "app.js", "real")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].path == "app.js"


class TestOversizeFile:
    def test_file_over_1mb_skipped(self, tmp_path):
        big = tmp_path / "huge.py"
        big.write_bytes(b"x" * (1_048_576 + 1))
        _write(tmp_path / "small.py", "y = 1")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].path == "small.py"

    def test_file_exactly_1mb_included(self, tmp_path):
        exact = tmp_path / "exact.py"
        exact.write_bytes(b"x" * 1_048_576)
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].path == "exact.py"


class TestContentHash:
    def test_hash_matches_sha256(self, tmp_path):
        content = "def hello():\n    return 'world'\n"
        _write(tmp_path / "hello.py", content)
        result = scan_repo(tmp_path)
        assert len(result) == 1
        # Hash is computed from raw bytes on disk (may include \r\n on Windows)
        raw_bytes = (tmp_path / "hello.py").read_bytes()
        expected = hashlib.sha256(raw_bytes).hexdigest()
        assert result[0].content_hash == expected

    def test_empty_file_hash(self, tmp_path):
        _write(tmp_path / "empty.py", "")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        expected = hashlib.sha256(b"").hexdigest()
        assert result[0].content_hash == expected


class TestGitignore:
    def test_gitignore_patterns_respected(self, tmp_path):
        _write(tmp_path / ".gitignore", "ignored_dir/\nignored.py\n")
        _write(tmp_path / "ignored_dir" / "secret.py", "x")
        _write(tmp_path / "ignored.py", "x")
        _write(tmp_path / "kept.py", "y")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].path == "kept.py"

    def test_gitignore_glob_pattern(self, tmp_path):
        _write(tmp_path / ".gitignore", "*.generated.py\n")
        _write(tmp_path / "schema.generated.py", "x")
        _write(tmp_path / "schema.py", "y")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].path == "schema.py"

    def test_no_gitignore_is_fine(self, tmp_path):
        _write(tmp_path / "app.py", "x = 1")
        result = scan_repo(tmp_path)
        assert len(result) == 1


class TestFileInfoFields:
    def test_relative_path_uses_forward_slashes(self, tmp_path):
        _write(tmp_path / "src" / "lib" / "mod.py", "x")
        result = scan_repo(tmp_path)
        assert len(result) == 1
        assert result[0].path == "src/lib/mod.py"

    def test_abs_path_is_absolute(self, tmp_path):
        _write(tmp_path / "app.py", "x")
        result = scan_repo(tmp_path)
        assert result[0].abs_path.is_absolute()

    def test_size_bytes_correct(self, tmp_path):
        content = "hello world"
        _write(tmp_path / "app.py", content)
        result = scan_repo(tmp_path)
        assert result[0].size_bytes == len(content.encode("utf-8"))


class TestSortOrder:
    def test_results_sorted_by_relative_path(self, tmp_path):
        _write(tmp_path / "z.py", "z")
        _write(tmp_path / "a.py", "a")
        _write(tmp_path / "m" / "b.py", "b")
        result = scan_repo(tmp_path)
        paths = [fi.path for fi in result]
        assert paths == sorted(paths)
        assert paths == ["a.py", "m/b.py", "z.py"]
