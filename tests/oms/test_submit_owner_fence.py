"""Isolated contract tests for the PostgreSQL submit-owner protocol."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

import pytest

from ai_trading.oms.intent_store import IntentStore


@dataclass
class _LockState:
    owner_backend_pid: int | None = None


class _Connection:
    def __init__(self, state: _LockState, backend_pid: int) -> None:
        self.state = state
        self.backend_pid = backend_pid
        self.closed = False

    def scalar(self, statement: Any, _params: Any = None) -> int | bool:
        query = str(statement)
        if "pg_backend_pid()" in query and "pg_locks" not in query:
            return self.backend_pid
        if "pg_try_advisory_lock" in query:
            if self.state.owner_backend_pid is not None:
                return False
            self.state.owner_backend_pid = self.backend_pid
            return True
        if "pg_locks" in query:
            return self.state.owner_backend_pid == self.backend_pid
        if "pg_advisory_unlock" in query:
            held = self.state.owner_backend_pid == self.backend_pid
            if held:
                self.state.owner_backend_pid = None
            return held
        raise AssertionError(f"unexpected owner query: {query}")

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _Engine:
    def __init__(self, state: _LockState, backend_pid: int) -> None:
        self.state = state
        self.backend_pid = backend_pid
        self.connections: list[_Connection] = []

    def connect(self) -> _Connection:
        connection = _Connection(self.state, self.backend_pid)
        self.connections.append(connection)
        return connection


def _store(state: _LockState, backend_pid: int) -> tuple[IntentStore, _Engine]:
    store = IntentStore.__new__(IntentStore)
    engine = _Engine(state, backend_pid)
    store._database_url = "postgresql+psycopg://isolated.invalid/oms"
    store._engine = engine  # type: ignore[assignment]
    store._lock = threading.RLock()
    store._submit_owner_connection = None
    store._submit_owner_pid = None
    store._submit_owner_backend_pid = None
    store._submit_owner_required = False
    return store, engine


def test_only_one_candidate_owns_submit_lock_until_release() -> None:
    state = _LockState()
    first, first_engine = _store(state, 101)
    second, second_engine = _store(state, 202)

    assert first.acquire_submit_owner() is True
    first.assert_submit_owner()
    assert second.acquire_submit_owner() is False
    assert second_engine.connections[0].closed is True
    with pytest.raises(RuntimeError, match="OMS_SUBMIT_OWNER_UNAVAILABLE"):
        second.assert_submit_owner()

    first.release_submit_owner()
    assert first_engine.connections[0].closed is True
    assert second.acquire_submit_owner() is True
    second.assert_submit_owner()
    second.release_submit_owner()


def test_lost_backend_or_forked_process_blocks_owner_assertion(monkeypatch) -> None:
    state = _LockState()
    store, _engine = _store(state, 303)
    assert store.acquire_submit_owner() is True
    state.owner_backend_pid = None
    with pytest.raises(RuntimeError, match="OMS_SUBMIT_OWNER_LOST"):
        store.assert_submit_owner()
    state.owner_backend_pid = 303
    current_pid = os.getpid()
    monkeypatch.setattr("ai_trading.oms.intent_store.os.getpid", lambda: current_pid + 1)
    with pytest.raises(RuntimeError, match="OMS_SUBMIT_OWNER_UNAVAILABLE"):
        store.assert_submit_owner()
