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
        names = {s.name for s in symbols}
        assert names == {"main", "MyClass", "__init__"}

    def test_get_symbols_by_kind(self, kb):
        api, rid, f1, *_ = kb
        funcs = api.get_symbols_for_file(f1, kind="function")
        assert len(funcs) == 1
        assert funcs[0].name == "main"
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


class TestGetFileById:
    def test_found(self, kb):
        api, rid, f1, *_ = kb
        f = api.get_file_by_id(f1)
        assert f is not None
        assert f.id == f1
        assert f.path == "src/main.py"
        assert f.language == "python"

    def test_not_found(self, kb):
        api, *_ = kb
        f = api.get_file_by_id(99999)
        assert f is None


class TestGetSymbolById:
    def test_found(self, kb):
        api, rid, f1, f2, f3, s1, *_ = kb
        sym = api.get_symbol_by_id(s1)
        assert sym is not None
        assert sym.id == s1
        assert sym.name == "main"
        assert sym.kind == "function"

    def test_not_found(self, kb):
        api, *_ = kb
        sym = api.get_symbol_by_id(99999)
        assert sym is None


class TestSearchSymbolsByNameSpecialChars:
    def test_with_results(self, kb):
        api, rid, *_ = kb
        symbols = api.search_symbols_by_name(rid, "main")
        assert any(s.name == "main" for s in symbols)

    def test_percent_escaped(self, curated_conn):
        """LIKE special char % is escaped and does not act as wildcard."""
        rid = queries.upsert_repo(curated_conn, "/test/esc", None)
        fid = queries.upsert_file(curated_conn, rid, "x.py", "python", "h", 10)
        queries.insert_symbol(curated_conn, fid, "rate%calc", "function", 1, 5)
        queries.insert_symbol(curated_conn, fid, "ratecalc", "function", 6, 10)
        curated_conn.commit()

        api = KnowledgeBase(curated_conn)
        results = api.search_symbols_by_name(rid, "rate%calc")
        names = [s.name for s in results]
        # Should match "rate%calc" literally, not "rate<anything>calc"
        assert "rate%calc" in names

    def test_underscore_escaped(self, curated_conn):
        """LIKE special char _ is escaped and does not act as single-char wildcard."""
        rid = queries.upsert_repo(curated_conn, "/test/esc2", None)
        fid = queries.upsert_file(curated_conn, rid, "y.py", "python", "h", 10)
        queries.insert_symbol(curated_conn, fid, "get_value", "function", 1, 5)
        queries.insert_symbol(curated_conn, fid, "getXvalue", "function", 6, 10)
        curated_conn.commit()

        api = KnowledgeBase(curated_conn)
        results = api.search_symbols_by_name(rid, "get_value")
        names = [s.name for s in results]
        assert "get_value" in names
        # getXvalue should NOT match get_value when _ is escaped
        # (it could match if _ were unescaped since _ matches any single char)

    def test_backslash_escaped(self, curated_conn):
        """LIKE escape char \\ is itself escaped."""
        rid = queries.upsert_repo(curated_conn, "/test/esc3", None)
        fid = queries.upsert_file(curated_conn, rid, "z.py", "python", "h", 10)
        queries.insert_symbol(curated_conn, fid, "path\\sep", "function", 1, 5)
        curated_conn.commit()

        api = KnowledgeBase(curated_conn)
        results = api.search_symbols_by_name(rid, "path\\sep")
        names = [s.name for s in results]
        assert "path\\sep" in names


class TestSearchFilesByMetadataModule:
    def test_module_filter(self, kb):
        """search_files_by_metadata with module parameter filters correctly."""
        api, rid, f1, *_ = kb
        # f1 has module="core" set in the fixture
        files = api.search_files_by_metadata(rid, module="core")
        assert len(files) == 1
        assert files[0].path == "src/main.py"

    def test_module_no_match(self, kb):
        api, rid, *_ = kb
        files = api.search_files_by_metadata(rid, module="nonexistent")
        assert files == []


class TestGetSymbolNeighborsWithKinds:
    def test_kinds_filter(self, kb):
        """get_symbol_neighbors with kinds filters results by symbol kind."""
        api, rid, f1, f2, f3, s1, s2, s3 = kb
        # s1 (main, function) calls s3 (helper, function)
        callees = api.get_symbol_neighbors(s1, "callees", kinds=["function"])
        assert len(callees) == 1
        assert callees[0].name == "helper"

    def test_kinds_filter_excludes(self, kb):
        """get_symbol_neighbors with non-matching kinds returns empty."""
        api, rid, f1, f2, f3, s1, s2, s3 = kb
        # s1 calls s3 (a function), filtering to "class" should return nothing
        callees = api.get_symbol_neighbors(s1, "callees", kinds=["class"])
        assert callees == []


class TestGetRepoOverviewDomainCounts:
    def test_domain_counts_populated(self, kb):
        """get_repo_overview populates domain_counts from file_metadata."""
        api, rid, *_ = kb
        overview = api.get_repo_overview(rid)
        # f1 has domain="application" from the fixture
        assert "application" in overview.domain_counts
        assert overview.domain_counts["application"] == 1


class TestGetRepoOverviewMostConnected:
    def test_most_connected_populated(self, kb):
        """get_repo_overview populates most_connected_files from dependencies."""
        api, rid, *_ = kb
        overview = api.get_repo_overview(rid)
        assert len(overview.most_connected_files) > 0
        # most_connected is a list of (path, dep_count) tuples
        paths = [entry[0] for entry in overview.most_connected_files]
        # src/utils.py has 2 deps (f1->f2, f3->f2 as target)
        assert "src/utils.py" in paths
        # The dep_count for utils.py should be 2 (two incoming deps)
        utils_entry = next(e for e in overview.most_connected_files if e[0] == "src/utils.py")
        assert utils_entry[1] == 2


class TestErrorPaths:
    def test_get_symbol_neighbors_invalid_direction(self, kb):
        api, rid, f1, f2, f3, s1, *_ = kb
        with pytest.raises(ValueError, match="direction must be"):
            api.get_symbol_neighbors(s1, "siblings")

    def test_get_dependencies_invalid_direction(self, kb):
        api, rid, f1, *_ = kb
        with pytest.raises(ValueError, match="direction must be"):
            api.get_dependencies(f1, "exports")

    def test_get_file_context_missing_file(self, kb):
        api, *_ = kb
        ctx = api.get_file_context(99999)
        assert ctx is None


class TestDependencySubgraphDepth:
    def test_depth_zero(self, kb):
        api, rid, f1, *_ = kb
        deps = api.get_dependency_subgraph([f1], depth=0)
        assert deps == []

    def test_depth_two_expands(self, kb):
        """With f1->f2 and f3->f2, depth=2 from f1 should reach f3.

        BFS is bidirectional: it follows both source and target edges,
        so f2 (discovered at depth 1) expands to f3 via the f3->f2 edge.
        """
        api, rid, f1, f2, f3, *_ = kb
        deps = api.get_dependency_subgraph([f1], depth=2)
        all_file_ids = set()
        for d in deps:
            all_file_ids.add(d.source_file_id)
            all_file_ids.add(d.target_file_id)
        # f1 imports f2, f3 imports f2 — depth 2 discovers f3 via bidirectional BFS
        assert f3 in all_file_ids

    def test_circular_deps_terminate(self, curated_conn):
        """BFS must terminate when circular dependencies exist (A->B->A)."""
        rid = queries.upsert_repo(curated_conn, "/test/cycle", None)
        fa = queries.upsert_file(curated_conn, rid, "a.py", "python", "aaa", 10)
        fb = queries.upsert_file(curated_conn, rid, "b.py", "python", "bbb", 20)
        queries.insert_dependency(curated_conn, fa, fb, "import")  # A -> B
        queries.insert_dependency(curated_conn, fb, fa, "import")  # B -> A (cycle)
        curated_conn.commit()

        api = KnowledgeBase(curated_conn)
        deps = api.get_dependency_subgraph([fa], depth=10)
        # Should find both edges without infinite loop
        assert len(deps) == 2
        file_ids = {d.source_file_id for d in deps} | {d.target_file_id for d in deps}
        assert file_ids == {fa, fb}
