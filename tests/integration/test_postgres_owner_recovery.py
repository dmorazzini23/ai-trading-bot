"""Opt-in real PostgreSQL process-boundary owner drill, with no schema writes.

Use a disposable server, never the live OMS: this acquires the actual owner
lock, kills its owning process, and verifies takeover by a second connection.
"""
from __future__ import annotations

import multiprocessing
import threading
import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

from ai_trading.config.management import get_env
from ai_trading.oms.intent_store import IntentStore


def _owner(url: str) -> IntentStore:
    store = IntentStore.__new__(IntentStore)
    store._database_url = url
    store._engine = create_engine(url, pool_pre_ping=True)
    store._lock = threading.RLock()
    store._submit_owner_connection = None
    store._submit_owner_pid = None
    store._submit_owner_backend_pid = None
    store._submit_owner_required = False
    return store


def _hold_owner(url: str, pipe) -> None:
    store = _owner(url)
    try:
        pipe.send(store.acquire_submit_owner())
        store.assert_submit_owner()
        pipe.recv()
    finally:
        store.release_submit_owner()
        store._engine.dispose()
        pipe.close()


def test_actual_postgres_owner_crash_and_takeover() -> None:
    url = str(get_env("AI_TRADING_TEST_OWNER_DATABASE_URL", "") or "")
    if not url:
        pytest.skip("isolated PostgreSQL URL required for actual two-process owner drill")
    if not url.startswith("postgresql+psycopg://"):
        pytest.fail("test owner URL must use postgresql+psycopg")
    parsed = make_url(url)
    if parsed.host not in {"127.0.0.1", "localhost", "::1"} or parsed.database != "ai_trading_owner_test":
        pytest.fail("owner drill requires a local disposable ai_trading_owner_test database")
    if url == get_env("DATABASE_URL", ""):
        pytest.fail("owner drill must not use the configured runtime database")
    context = multiprocessing.get_context("spawn")
    parent_pipe, child_pipe = context.Pipe()
    process = context.Process(target=_hold_owner, args=(url, child_pipe))
    contender = _owner(url)
    process.start()
    child_pipe.close()
    try:
        assert parent_pipe.poll(15), "owner process did not report acquisition"
        assert parent_pipe.recv() is True
        assert contender.acquire_submit_owner() is False
        with pytest.raises(RuntimeError, match="OMS_SUBMIT_OWNER_UNAVAILABLE"):
            contender.assert_submit_owner()
        process.kill()
        process.join(timeout=5)
        assert not process.is_alive()
        deadline = time.monotonic() + 10
        while not contender.acquire_submit_owner():
            assert time.monotonic() < deadline, "crashed backend still owns submit fence"
            time.sleep(0.1)
        contender.assert_submit_owner()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        contender.release_submit_owner()
        contender._engine.dispose()
        parent_pipe.close()
