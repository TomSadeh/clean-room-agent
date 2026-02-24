"""Tests for query/api.py."""

import pytest

from clean_room_agent.db import queries
from clean_room_agent.query.api import KnowledgeBase


@pytest.fixture
def kb(curated_conn):
    """Populated KnowledgeBase fixture."""
    # Set up test data
    rid = queries.upsert_repo(curated_conn, "/test", "https://github.com/test/repo")
    f1 = queries.upsert_file(curated_conn, rid, "src/main.py", "python", "aaa", 100)
    f2 = queries.upsert_file(curated_conn, rid, "src/utils.py", "python", "bbb", 200)
    f3 = queries.upsert_file(curated_conn, rid, "src/app.ts", "typescript", "ccc", 300)

    s1 = queries.insert_symbol(curated_conn, f1, "main", "function", 1, 10, "def main():")
    s2 = queries.insert_symbol(curated_conn, f1, "MyClass", "class", 12, 30, "class MyClass:")
    s3 = queries.insert_symbol(curated_conn, f2, "helper", "function", 1, 5, "def helper():")
    s4 = queries.insert_symbol(curated_conn, f1, "__init__", "method", 13, 15, "def __init__(self):", s2)

    queries.insert_docstring(curated_conn, f1, "Module docstring", "plain")
    queries.insert_docstring(curated_conn, f1, "Main function doc", "google", symbol_id=s1)

    queries.insert_inline_comment(curated_conn, f1, 5, "# TODO: fix this", "todo", False)
    queries.insert_inline_comment(curated_conn, f1, 8, "# because we need thread safety", "rationale", True, s1)

    queries.insert_dependency(curated_conn, f1, f2, "import")
    queries.insert_dependency(curated_conn, f3, f2, "import")

    queries.insert_symbol_reference(curated_conn, s1, s3, "call")

    c1 = queries.insert_commit(curated_conn, rid, "abc123", "2024-01-01T00:00:00Z", "author", "fix bug")
    queries.insert_file_commit(curated_conn, f1, c1)

    queries.upsert_co_change(curated_conn, f1, f2, "abc123")
    queries.upsert_co_change(curated_conn, f1, f2, "def456")

    queries.upsert_file_metadata(
        curated_conn, f1, purpose="entry point", module="core",
        domain="application", concepts='["cli"]', public_api_surface='["main"]',
    )

    curated_conn.commit()
    return KnowledgeBase(curated_conn), rid, f1, f2, f3, s1, s2, s3


class TestFileQueries:
    def test_get_files(self, kb):
        api, rid, *_ = kb
        files = api.get_files(rid)
        assert len(files) == 3

    def test_get_files_by_language(self, kb):
        api, rid, *_ = kb
        py_files = api.get_files(rid, language="python")
        assert len(py_files) == 2

    def test_get_file_by_path(self, kb):
        api, rid, *_ = kb
        f = api.get_file_by_path(rid, "src/main.py")
        assert f is not None
        assert f.path == "src/main.py"

    def test_get_file_by_path_not_found(self, kb):
        api, rid, *_ = kb
        f = api.get_file_by_path(rid, "nonexistent.py")
        assert f is None

    def test_search_files_by_metadata(self, kb):
        api, rid, *_ = kb
        files = api.search_files_by_metadata(rid, domain="application")
        assert len(files) == 1
        assert files[0].path == "src/main.py"


class TestSymbolQueries:
    def test_get_symbols_for_file(self, kb):
        api, rid, f1, *_ = kb
        symbols = api.get_symbols_for_file(f1)
        assert len(symbols) >= 3

    def test_get_symbols_by_kind(self, kb):
        api, rid, f1, *_ = kb
        funcs = api.get_symbols_for_file(f1, kind="function")
        assert all(s.kind == "function" for s in funcs)

    def test_search_symbols_by_name(self, kb):
        api, rid, *_ = kb
        symbols = api.search_symbols_by_name(rid, "main")
        assert any(s.name == "main" for s in symbols)

    def test_get_symbol_neighbors_callees(self, kb):
        api, rid, f1, f2, f3, s1, s2, s3 = kb
        callees = api.get_symbol_neighbors(s1, "callees")
        assert any(s.name == "helper" for s in callees)

    def test_get_symbol_neighbors_callers(self, kb):
        api, rid, f1, f2, f3, s1, s2, s3 = kb
        callers = api.get_symbol_neighbors(s3, "callers")
        assert any(s.name == "main" for s in callers)


class TestDependencyQueries:
    def test_get_dependencies_imports(self, kb):
        api, rid, f1, *_ = kb
        deps = api.get_dependencies(f1, "imports")
        assert len(deps) == 1
        assert deps[0].target_file_id != f1

    def test_get_dependencies_imported_by(self, kb):
        api, rid, f1, f2, *_ = kb
        deps = api.get_dependencies(f2, "imported_by")
        assert len(deps) == 2  # imported by f1 and f3

    def test_get_dependency_subgraph(self, kb):
        api, rid, f1, *_ = kb
        deps = api.get_dependency_subgraph([f1], depth=1)
        assert len(deps) >= 1


class TestCoChangeQueries:
    def test_get_co_change_neighbors(self, kb):
        api, rid, f1, *_ = kb
        co = api.get_co_change_neighbors(f1, min_count=2)
        assert len(co) == 1
        assert co[0].count == 2


class TestDocstringCommentQueries:
    def test_get_docstrings(self, kb):
        api, rid, f1, *_ = kb
        docs = api.get_docstrings_for_file(f1)
        assert len(docs) == 2

    def test_get_rationale_comments(self, kb):
        api, rid, f1, *_ = kb
        comments = api.get_rationale_comments(f1)
        assert len(comments) == 1
        assert "thread safety" in comments[0].content


class TestCommitQueries:
    def test_get_recent_commits(self, kb):
        api, rid, f1, *_ = kb
        commits = api.get_recent_commits_for_file(f1)
        assert len(commits) == 1
        assert commits[0].hash == "abc123"


class TestCompositeQueries:
    def test_get_file_context(self, kb):
        api, rid, f1, *_ = kb
        ctx = api.get_file_context(f1)
        assert ctx is not None
        assert ctx.file.path == "src/main.py"
        assert len(ctx.symbols) >= 3
        assert len(ctx.docstrings) == 2
        assert len(ctx.rationale_comments) == 1
        assert len(ctx.dependencies) == 1
        assert len(ctx.recent_commits) == 1

    def test_get_repo_overview(self, kb):
        api, rid, *_ = kb
        overview = api.get_repo_overview(rid)
        assert overview.file_count == 3
        assert overview.language_counts["python"] == 2
        assert overview.language_counts["typescript"] == 1

    def test_get_adapter_for_stage_none(self, kb):
        api, *_ = kb
        adapter = api.get_adapter_for_stage("scope")
        assert adapter is None
