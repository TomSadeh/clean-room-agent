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
