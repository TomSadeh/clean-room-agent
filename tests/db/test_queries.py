"""Tests for db/queries.py and db/raw_queries.py."""

from clean_room_agent.db import queries, raw_queries


class TestCuratedQueries:
    def test_upsert_repo(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", "https://github.com/x/y")
        curated_conn.commit()
        assert rid is not None
        # Second call updates
        rid2 = queries.upsert_repo(curated_conn, "/tmp/repo", "https://github.com/x/z")
        curated_conn.commit()
        assert rid2 == rid

    def test_upsert_file(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        fid = queries.upsert_file(curated_conn, rid, "src/main.py", "python", "abc123", 100)
        curated_conn.commit()
        assert fid is not None
        # Upsert same path
        fid2 = queries.upsert_file(curated_conn, rid, "src/main.py", "python", "def456", 200)
        curated_conn.commit()
        assert fid2 == fid

    def test_insert_symbol(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        fid = queries.upsert_file(curated_conn, rid, "main.py", "python", "abc", 50)
        sid = queries.insert_symbol(curated_conn, fid, "main", "function", 1, 10, "def main():")
        curated_conn.commit()
        assert sid is not None

    def test_insert_docstring(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        fid = queries.upsert_file(curated_conn, rid, "main.py", "python", "abc", 50)
        did = queries.insert_docstring(curated_conn, fid, "Module docstring", "plain")
        curated_conn.commit()
        assert did is not None

    def test_insert_inline_comment(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        fid = queries.upsert_file(curated_conn, rid, "main.py", "python", "abc", 50)
        cid = queries.insert_inline_comment(curated_conn, fid, 5, "# TODO: fix this", "todo")
        curated_conn.commit()
        assert cid is not None

    def test_insert_dependency(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        f1 = queries.upsert_file(curated_conn, rid, "a.py", "python", "aaa", 10)
        f2 = queries.upsert_file(curated_conn, rid, "b.py", "python", "bbb", 20)
        did = queries.insert_dependency(curated_conn, f1, f2, "import")
        curated_conn.commit()
        assert did is not None

    def test_upsert_co_change(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        f1 = queries.upsert_file(curated_conn, rid, "a.py", "python", "aaa", 10)
        f2 = queries.upsert_file(curated_conn, rid, "b.py", "python", "bbb", 20)
        queries.upsert_co_change(curated_conn, f1, f2, "abc")
        queries.upsert_co_change(curated_conn, f2, f1, "def")  # reversed order
        curated_conn.commit()
        row = curated_conn.execute(
            "SELECT count FROM co_changes WHERE file_a_id = ? AND file_b_id = ?",
            (min(f1, f2), max(f1, f2)),
        ).fetchone()
        assert row["count"] == 2

    def test_insert_symbol_reference(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        fid = queries.upsert_file(curated_conn, rid, "main.py", "python", "abc", 50)
        caller = queries.insert_symbol(curated_conn, fid, "main", "function", 1, 10)
        callee = queries.insert_symbol(curated_conn, fid, "helper", "function", 12, 20)
        ref_id = queries.insert_symbol_reference(curated_conn, caller, callee, "call")
        curated_conn.commit()
        assert ref_id is not None
        row = curated_conn.execute(
            "SELECT * FROM symbol_references WHERE id = ?", (ref_id,)
        ).fetchone()
        assert row["caller_symbol_id"] == caller
        assert row["callee_symbol_id"] == callee
        assert row["reference_kind"] == "call"

    def test_insert_file_commit(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        fid = queries.upsert_file(curated_conn, rid, "main.py", "python", "abc", 50)
        cid = queries.insert_commit(curated_conn, rid, "aaa111", "2024-01-01T00:00:00Z")
        queries.insert_file_commit(curated_conn, fid, cid)
        curated_conn.commit()
        row = curated_conn.execute(
            "SELECT * FROM file_commits WHERE file_id = ? AND commit_id = ?",
            (fid, cid),
        ).fetchone()
        assert row is not None

    def test_insert_file_commit_ignores_duplicate(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        fid = queries.upsert_file(curated_conn, rid, "main.py", "python", "abc", 50)
        cid = queries.insert_commit(curated_conn, rid, "aaa111", "2024-01-01T00:00:00Z")
        queries.insert_file_commit(curated_conn, fid, cid)
        queries.insert_file_commit(curated_conn, fid, cid)  # no error
        curated_conn.commit()
        count = curated_conn.execute("SELECT COUNT(*) FROM file_commits").fetchone()[0]
        assert count == 1

    def test_upsert_file_metadata_insert_and_update(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        fid = queries.upsert_file(curated_conn, rid, "main.py", "python", "abc", 50)
        queries.upsert_file_metadata(curated_conn, fid, purpose="entry point", domain="cli")
        curated_conn.commit()
        row = curated_conn.execute(
            "SELECT * FROM file_metadata WHERE file_id = ?", (fid,)
        ).fetchone()
        assert row["purpose"] == "entry point"
        assert row["domain"] == "cli"

        # Update with new values
        queries.upsert_file_metadata(curated_conn, fid, purpose="main entry", module="core")
        curated_conn.commit()
        row = curated_conn.execute(
            "SELECT * FROM file_metadata WHERE file_id = ?", (fid,)
        ).fetchone()
        assert row["purpose"] == "main entry"
        assert row["module"] == "core"
        # Only one row, not two
        count = curated_conn.execute("SELECT COUNT(*) FROM file_metadata").fetchone()[0]
        assert count == 1

    def test_upsert_co_change_explicit_count(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        f1 = queries.upsert_file(curated_conn, rid, "a.py", "python", "aaa", 10)
        f2 = queries.upsert_file(curated_conn, rid, "b.py", "python", "bbb", 20)
        queries.upsert_co_change(curated_conn, f1, f2, "abc", count=5)
        curated_conn.commit()
        row = curated_conn.execute(
            "SELECT count FROM co_changes WHERE file_a_id = ? AND file_b_id = ?",
            (min(f1, f2), max(f1, f2)),
        ).fetchone()
        assert row["count"] == 5

        # Overwrite with explicit count
        queries.upsert_co_change(curated_conn, f1, f2, "def", count=10)
        curated_conn.commit()
        row = curated_conn.execute(
            "SELECT count, last_commit_hash FROM co_changes WHERE file_a_id = ? AND file_b_id = ?",
            (min(f1, f2), max(f1, f2)),
        ).fetchone()
        assert row["count"] == 10
        assert row["last_commit_hash"] == "def"

    def test_clear_file_children_preserves_file_row(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        fid = queries.upsert_file(curated_conn, rid, "main.py", "python", "abc", 50)
        queries.insert_symbol(curated_conn, fid, "foo", "function", 1, 5)
        queries.insert_docstring(curated_conn, fid, "doc", "plain")
        queries.insert_inline_comment(curated_conn, fid, 3, "# note", "general")
        queries.upsert_file_metadata(curated_conn, fid, purpose="test")
        curated_conn.commit()

        queries.clear_file_children(curated_conn, fid)
        curated_conn.commit()

        # File row still exists
        assert curated_conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1
        # All children gone
        assert curated_conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == 0
        assert curated_conn.execute("SELECT COUNT(*) FROM docstrings").fetchone()[0] == 0
        assert curated_conn.execute("SELECT COUNT(*) FROM inline_comments").fetchone()[0] == 0
        assert curated_conn.execute("SELECT COUNT(*) FROM file_metadata").fetchone()[0] == 0

    def test_clear_file_children_keeps_incoming_deps(self, curated_conn):
        """When re-indexing file B, deps A->B from other files must survive."""
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        fa = queries.upsert_file(curated_conn, rid, "a.py", "python", "aaa", 10)
        fb = queries.upsert_file(curated_conn, rid, "b.py", "python", "bbb", 20)
        queries.insert_dependency(curated_conn, fa, fb, "import")  # A -> B
        queries.insert_dependency(curated_conn, fb, fa, "import")  # B -> A
        curated_conn.commit()

        queries.clear_file_children(curated_conn, fb)
        curated_conn.commit()

        # B's outgoing dep (B->A) is deleted
        assert curated_conn.execute(
            "SELECT COUNT(*) FROM dependencies WHERE source_file_id = ?", (fb,)
        ).fetchone()[0] == 0
        # A's dep pointing to B is preserved
        assert curated_conn.execute(
            "SELECT COUNT(*) FROM dependencies WHERE source_file_id = ? AND target_file_id = ?",
            (fa, fb),
        ).fetchone()[0] == 1

    def test_clear_file_children_cleans_symbol_references(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        fa = queries.upsert_file(curated_conn, rid, "a.py", "python", "aaa", 10)
        fb = queries.upsert_file(curated_conn, rid, "b.py", "python", "bbb", 20)
        sa = queries.insert_symbol(curated_conn, fa, "func_a", "function", 1, 5)
        sb = queries.insert_symbol(curated_conn, fb, "func_b", "function", 1, 5)
        queries.insert_symbol_reference(curated_conn, sa, sb, "call")  # A calls B
        queries.insert_symbol_reference(curated_conn, sb, sa, "call")  # B calls A
        curated_conn.commit()

        queries.clear_file_children(curated_conn, fb)
        curated_conn.commit()

        # Both refs are gone (B's symbol is involved in both)
        assert curated_conn.execute(
            "SELECT COUNT(*) FROM symbol_references"
        ).fetchone()[0] == 0
        # A's symbol still exists, B's symbol is deleted
        assert curated_conn.execute(
            "SELECT COUNT(*) FROM symbols WHERE file_id = ?", (fa,)
        ).fetchone()[0] == 1
        assert curated_conn.execute(
            "SELECT COUNT(*) FROM symbols WHERE file_id = ?", (fb,)
        ).fetchone()[0] == 0

    def test_delete_file_data(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        fid = queries.upsert_file(curated_conn, rid, "main.py", "python", "abc", 50)
        queries.insert_symbol(curated_conn, fid, "foo", "function", 1, 5)
        queries.insert_docstring(curated_conn, fid, "doc", "plain")
        queries.insert_inline_comment(curated_conn, fid, 3, "# note", "general")
        curated_conn.commit()

        queries.delete_file_data(curated_conn, fid)
        curated_conn.commit()

        assert curated_conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
        assert curated_conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == 0

    def test_delete_file_data_cleans_incoming_deps(self, curated_conn):
        """delete_file_data should remove incoming deps from other files."""
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        fa = queries.upsert_file(curated_conn, rid, "a.py", "python", "aaa", 10)
        fb = queries.upsert_file(curated_conn, rid, "b.py", "python", "bbb", 20)
        queries.insert_dependency(curated_conn, fa, fb, "import")  # A -> B
        curated_conn.commit()

        queries.delete_file_data(curated_conn, fb)
        curated_conn.commit()

        # B's file row is gone
        assert curated_conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1
        # The A -> B dep is also gone
        assert curated_conn.execute("SELECT COUNT(*) FROM dependencies").fetchone()[0] == 0

    def test_insert_commit_deduplicates_by_repo_and_hash(self, curated_conn):
        rid = queries.upsert_repo(curated_conn, "/tmp/repo", None)
        cid1 = queries.insert_commit(
            curated_conn, rid, "abc123", "2024-01-01T00:00:00Z", "alice", "first"
        )
        cid2 = queries.insert_commit(
            curated_conn, rid, "abc123", "2024-01-02T00:00:00Z", "bob", "updated"
        )
        curated_conn.commit()

        assert cid2 == cid1
        row = curated_conn.execute(
            "SELECT author, message, timestamp FROM commits WHERE id = ?", (cid1,)
        ).fetchone()
        assert row["author"] == "bob"
        assert row["message"] == "updated"


class TestRawQueries:
    def test_insert_index_run(self, raw_conn):
        rid = raw_queries.insert_index_run(raw_conn, "/tmp/repo", 100, 5, 1234, "success")
        raw_conn.commit()
        assert rid is not None

    def test_insert_enrichment_output(self, raw_conn):
        eid = raw_queries.insert_enrichment_output(
            raw_conn,
            file_id=1,
            model="qwen3:4b",
            raw_prompt="Analyze this file",
            raw_response='{"purpose": "utils"}',
            purpose="utils",
        )
        raw_conn.commit()
        assert eid is not None
