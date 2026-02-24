"""Tests for extractors/dependencies.py."""

from pathlib import Path

from clean_room_agent.extractors.dependencies import resolve_dependencies
from clean_room_agent.parsers.base import ExtractedImport


class TestPythonDeps:
    def test_absolute_import(self):
        imp = ExtractedImport(module="foo.bar", names=["baz"])
        file_index = {"foo/bar.py", "foo/bar/baz.py"}
        deps = resolve_dependencies([imp], "main.py", "python", file_index, Path("/repo"))
        targets = {d.target_path for d in deps}
        assert "foo/bar.py" in targets

    def test_relative_import(self):
        imp = ExtractedImport(module="utils", names=["helper"], is_relative=True, level=1)
        file_index = {"pkg/utils.py", "pkg/main.py"}
        deps = resolve_dependencies([imp], "pkg/main.py", "python", file_index, Path("/repo"))
        targets = {d.target_path for d in deps}
        assert "pkg/utils.py" in targets

    def test_init_resolution(self):
        imp = ExtractedImport(module="pkg", names=[])
        file_index = {"pkg/__init__.py"}
        deps = resolve_dependencies([imp], "main.py", "python", file_index, Path("/repo"))
        targets = {d.target_path for d in deps}
        assert "pkg/__init__.py" in targets

    def test_external_ignored(self):
        imp = ExtractedImport(module="requests", names=["get"])
        file_index = {"main.py"}
        deps = resolve_dependencies([imp], "main.py", "python", file_index, Path("/repo"))
        assert deps == []

    def test_kind_defaults_to_import(self):
        imp = ExtractedImport(module="foo", names=[])
        file_index = {"foo.py"}
        deps = resolve_dependencies([imp], "main.py", "python", file_index, Path("/repo"))
        assert deps[0].kind == "import"


class TestTSJSDeps:
    def test_relative_import(self, tmp_path):
        imp = ExtractedImport(module="./utils", names=["helper"], is_relative=True)
        file_index = {"src/utils.ts", "src/main.ts"}
        deps = resolve_dependencies([imp], "src/main.ts", "typescript", file_index, tmp_path)
        targets = {d.target_path for d in deps}
        assert "src/utils.ts" in targets

    def test_index_file_resolution(self, tmp_path):
        imp = ExtractedImport(module="./components", names=[], is_relative=True)
        file_index = {"src/components/index.ts", "src/main.ts"}
        deps = resolve_dependencies([imp], "src/main.ts", "typescript", file_index, tmp_path)
        targets = {d.target_path for d in deps}
        assert "src/components/index.ts" in targets

    def test_external_package_ignored(self, tmp_path):
        imp = ExtractedImport(module="react", names=["useState"], is_relative=False)
        file_index = {"src/main.tsx"}
        deps = resolve_dependencies([imp], "src/main.tsx", "typescript", file_index, tmp_path)
        assert deps == []

    def test_type_import_kind(self, tmp_path):
        imp = ExtractedImport(
            module="./types", names=["Config"], is_relative=True, is_type_only=True,
        )
        file_index = {"src/types.ts", "src/main.ts"}
        deps = resolve_dependencies([imp], "src/main.ts", "typescript", file_index, tmp_path)
        assert deps[0].kind == "type_ref"

    def test_tsx_extension(self, tmp_path):
        imp = ExtractedImport(module="./App", names=[], is_relative=True)
        file_index = {"src/App.tsx", "src/index.ts"}
        deps = resolve_dependencies([imp], "src/index.ts", "typescript", file_index, tmp_path)
        targets = {d.target_path for d in deps}
        assert "src/App.tsx" in targets

    def test_explicit_extension_import(self, tmp_path):
        imp = ExtractedImport(module="./utils.ts", names=[], is_relative=True)
        file_index = {"src/utils.ts", "src/main.ts"}
        deps = resolve_dependencies([imp], "src/main.ts", "typescript", file_index, tmp_path)
        targets = {d.target_path for d in deps}
        assert "src/utils.ts" in targets
