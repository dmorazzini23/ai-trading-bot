"""Isolated submit-owner session tests; no PostgreSQL service or broker is used."""

from __future__ import annotations

import os
import multiprocessing
import select
import threading
from typing import Any

import pytest

from ai_trading.execution import engine as execution_engine
from ai_trading.oms.intent_store import IntentStore


class _AdvisoryConnection:
    def __init__(self, owner_state: Any, backend_pid: int) -> None:
        self.owner_state = owner_state
        self.backend_pid = backend_pid
        self.closed = False

    def scalar(self, statement: Any, _params: Any = None) -> int | bool:
        if self.closed:
            raise OSError("simulated database session loss")
        sql = str(statement)
        if "pg_locks" in sql:
            return self.owner_state.get("owner") == self.backend_pid
        if "pg_backend_pid" in sql:
            return self.backend_pid
        if "pg_try_advisory_lock" in sql:
            if self.owner_state.get("owner") not in (None, self.backend_pid):
                return False
            self.owner_state["owner"] = self.backend_pid
            return True
        if "pg_advisory_unlock" in sql:
            held = self.owner_state.get("owner") == self.backend_pid
            if held:
                self.owner_state["owner"] = None
            return held
        raise AssertionError(sql)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def invalidate(self) -> None:
        self.close()
        if self.owner_state.get("owner") == self.backend_pid:
            self.owner_state["owner"] = None


class _AdvisoryEngine:
    def __init__(self, owner_state: Any, backend_pid: int) -> None:
        self.owner_state = owner_state
        self.backend_pid = backend_pid

    def connect(self) -> _AdvisoryConnection:
        return _AdvisoryConnection(self.owner_state, self.backend_pid)


def _store(owner_state: Any, backend_pid: int) -> IntentStore:
    store = IntentStore.__new__(IntentStore)
    store._database_url = "postgresql+psycopg://isolated-test"
    store._engine = _AdvisoryEngine(owner_state, backend_pid)
    store._lock = threading.RLock()
    store._submit_owner_connection = None
    store._submit_owner_pid = None
    store._submit_owner_backend_pid = None
    store._submit_owner_required = False
    return store


class _SharedOwnerState:
    def __init__(self, owner_value: Any) -> None:
        self.owner_value = owner_value

    def get(self, _key: str) -> int | None:
        owner = int(self.owner_value.value)
        return owner if owner >= 0 else None

    def __setitem__(self, _key: str, value: int | None) -> None:
        self.owner_value.value = -1 if value is None else int(value)


def _hold_owner_in_process(owner_state: Any, ready_write: int, release_read: int) -> None:
    store = _store(owner_state, os.getpid())
    if not store.acquire_submit_owner():
        os._exit(41)
    os.write(ready_write, b"1")
    if os.read(release_read, 1) != b"1":
        os._exit(42)
    store.release_submit_owner()


def test_second_owner_is_refused_until_first_releases() -> None:
    shared_state: dict[str, Any] = {"owner": None}
    first = _store(shared_state, 1001)
    second = _store(shared_state, 1002)

    assert first.acquire_submit_owner() is True
    assert second.acquire_submit_owner() is False
    first.assert_submit_owner()
    with pytest.raises(RuntimeError, match="OMS_SUBMIT_OWNER_UNAVAILABLE"):
        second.assert_submit_owner()

    first.release_submit_owner()
    assert second.acquire_submit_owner() is True
    second.release_submit_owner()
    assert shared_state["owner"] is None


def test_separate_process_cannot_take_submit_ownership_until_release() -> None:
    context = multiprocessing.get_context("fork")
    shared_state = _SharedOwnerState(multiprocessing.RawValue("q", -1))
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    process = context.Process(
        target=_hold_owner_in_process,
        args=(shared_state, ready_write, release_read),
    )
    process.start()
    os.close(ready_write)
    os.close(release_read)
    try:
        readable, _, _ = select.select([ready_read], [], [], 10)
        assert readable and os.read(ready_read, 1) == b"1"
        standby = _store(shared_state, os.getpid())
        assert standby.acquire_submit_owner() is False
        os.write(release_write, b"1")
        process.join(timeout=10)
        assert process.exitcode == 0
        assert standby.acquire_submit_owner() is True
        standby.release_submit_owner()
    finally:
        os.close(ready_read)
        os.close(release_write)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_lost_or_forked_owner_blocks_submit_claim(monkeypatch) -> None:
    shared_state: dict[str, Any] = {"owner": None}
    store = _store(shared_state, 1001)
    assert store.acquire_submit_owner() is True
    connection = store._submit_owner_connection

    connection.backend_pid = 1002
    with pytest.raises(RuntimeError, match="OMS_SUBMIT_OWNER_LOST"):
        store.claim_for_submit("intent-1")
    connection.backend_pid = 1001
    shared_state["owner"] = None
    with pytest.raises(RuntimeError, match="OMS_SUBMIT_OWNER_LOST"):
        store.claim_for_submit("intent-1")
    shared_state["owner"] = 1001
    monkeypatch.setattr(os, "getpid", lambda: 2002)
    with pytest.raises(RuntimeError, match="OMS_SUBMIT_OWNER_UNAVAILABLE"):
        store.claim_for_submit("intent-1")


def test_database_session_loss_blocks_new_submit_claim() -> None:
    shared_state: dict[str, Any] = {"owner": None}
    store = _store(shared_state, 1001)
    assert store.acquire_submit_owner() is True
    connection = store._submit_owner_connection
    connection.closed = True

    with pytest.raises(RuntimeError, match="OMS_SUBMIT_OWNER_UNAVAILABLE"):
        store.claim_for_submit("intent-1")


@pytest.mark.parametrize("lease_available", [True, False])
@pytest.mark.parametrize("mode", ["live", "prod"])
def test_live_order_manager_requires_exclusive_database_owner(
    monkeypatch, lease_available: bool, mode: str
) -> None:
    class _Store:
        database_url = "postgresql+psycopg://localhost/test"
        path = "database-url"
        closed = False
        attempts = 0

        def acquire_submit_owner(self) -> bool:
            self.attempts += 1
            return lease_available

        def close(self) -> None:
            self.closed = True

    store = _Store()
    monkeypatch.setattr(execution_engine, "IntentStore", lambda **_kwargs: store)
    monkeypatch.setattr(execution_engine, "_env_bool", lambda _name, _default: True)
    monkeypatch.setattr(
        execution_engine,
        "get_env",
        lambda name, default="": {
            "EXECUTION_MODE": mode,
            "DATABASE_URL": store.database_url,
        }.get(name, default),
    )
    manager = execution_engine.OrderManager.__new__(execution_engine.OrderManager)
    manager._test_mode = False
    manager._intent_store = None

    if lease_available:
        manager._init_intent_store()
        assert manager._intent_store is store
        assert store.closed is False
    else:
        with pytest.raises(RuntimeError, match="OMS_INTENT_STORE_INIT_FAILED"):
            manager._init_intent_store()
        assert store.closed is True
    assert store.attempts == 1
