"""Tests for session DB key-value helpers."""

import pytest

from clean_room_agent.db.session_helpers import (
    delete_state,
    get_state,
    list_keys,
    set_state,
)


class TestSetAndGetState:
    def test_set_and_get_string(self, session_conn):
        set_state(session_conn, "test_key", "hello")
        assert get_state(session_conn, "test_key") == "hello"

    def test_set_and_get_dict(self, session_conn):
        data = {"name": "test", "count": 42}
        set_state(session_conn, "my_dict", data)
        assert get_state(session_conn, "my_dict") == data

    def test_set_and_get_list(self, session_conn):
        data = [1, 2, 3, "four"]
        set_state(session_conn, "my_list", data)
        assert get_state(session_conn, "my_list") == data

    def test_set_and_get_int(self, session_conn):
        set_state(session_conn, "counter", 99)
        assert get_state(session_conn, "counter") == 99

    def test_set_and_get_none_value(self, session_conn):
        set_state(session_conn, "nullable", None)
        assert get_state(session_conn, "nullable") is None

    def test_get_nonexistent_returns_none(self, session_conn):
        assert get_state(session_conn, "does_not_exist") is None

    def test_upsert_overwrites(self, session_conn):
        set_state(session_conn, "key", "first")
        set_state(session_conn, "key", "second")
        assert get_state(session_conn, "key") == "second"

    def test_json_round_trip_nested(self, session_conn):
        data = {"files": [{"id": 1, "path": "a.py"}], "meta": {"budget": 32768}}
        set_state(session_conn, "complex", data)
        assert get_state(session_conn, "complex") == data


class TestDeleteState:
    def test_delete_existing(self, session_conn):
        set_state(session_conn, "to_delete", "val")
        delete_state(session_conn, "to_delete")
        assert get_state(session_conn, "to_delete") is None

    def test_delete_nonexistent_no_error(self, session_conn):
        delete_state(session_conn, "nope")  # should not raise, but logs warning


class TestListKeys:
    def test_list_all(self, session_conn):
        set_state(session_conn, "a", 1)
        set_state(session_conn, "b", 2)
        set_state(session_conn, "c", 3)
        keys = list_keys(session_conn)
        assert set(keys) == {"a", "b", "c"}

    def test_list_empty(self, session_conn):
        assert list_keys(session_conn) == []

    def test_list_with_prefix(self, session_conn):
        set_state(session_conn, "stage_scope", 1)
        set_state(session_conn, "stage_precision", 2)
        set_state(session_conn, "task_query", 3)
        keys = list_keys(session_conn, prefix="stage_")
        assert set(keys) == {"stage_scope", "stage_precision"}

    def test_prefix_no_match(self, session_conn):
        set_state(session_conn, "alpha", 1)
        keys = list_keys(session_conn, prefix="beta_")
        assert keys == []

    def test_prefix_special_chars(self, session_conn):
        set_state(session_conn, "test%key", 1)
        set_state(session_conn, "test_key", 2)
        set_state(session_conn, "testXkey", 3)
        # prefix "test%" should match only "test%key", not "testXkey" or "test_key"
        keys = list_keys(session_conn, prefix="test%")
        assert keys == ["test%key"]
