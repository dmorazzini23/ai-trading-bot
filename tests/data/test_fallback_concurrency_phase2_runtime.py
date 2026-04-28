from __future__ import annotations

import asyncio
import logging
from collections import UserList, deque
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any

from ai_trading.data.fallback import concurrency


@dataclass(frozen=True)
class _FrozenHolder:
    lock: asyncio.Lock
    values: tuple[asyncio.Semaphore, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class _SlotsDataclassHolder:
    semaphore: asyncio.Semaphore


class _SlotsHolder:
    __slots__ = ("lock",)

    def __init__(self, lock: asyncio.Lock) -> None:
        self.lock = lock


class _ProjectObject:
    __module__ = "ai_trading.tests"

    def __init__(self, lock: asyncio.Lock) -> None:
        self.lock = lock


class _CustomSet(set):
    pass


def test_pooling_state_helpers_normalize_and_cache(monkeypatch) -> None:
    concurrency._POOLING_LIMIT_STATE = None
    concurrency._LOCAL_POOLING_VERSION = 0

    assert concurrency._normalise_pooling_state(None) is None
    assert concurrency._normalise_pooling_state((0, "3")) == (1, 3)
    assert concurrency._normalise_pooling_state(SimpleNamespace(limit=4, version=7)) == (4, 7)
    assert concurrency._normalise_pooling_state(SimpleNamespace(limit="4", version=7)) is None

    assert concurrency._next_local_pooling_version() == 1
    concurrency._LOCAL_POOLING_VERSION = -3
    assert concurrency._next_local_pooling_version() == 1

    concurrency._record_pooling_snapshot(5, 9)
    assert concurrency._POOLING_LIMIT_STATE == (5, 9)
    assert concurrency._get_effective_host_limit() == 5

    concurrency._invalidate_pooling_snapshot()
    monkeypatch.setattr(concurrency, "_pooling_reload_host_limit", lambda: (2, 10))
    monkeypatch.setattr(concurrency, "_pooling_get_limit_snapshot", lambda: (_ for _ in ()).throw(RuntimeError("unused")))
    assert concurrency._get_effective_host_limit() == 2
    assert concurrency._POOLING_LIMIT_STATE == (2, 10)

    concurrency._invalidate_pooling_snapshot()
    monkeypatch.setattr(concurrency, "_pooling_reload_host_limit", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(concurrency, "_pooling_get_limit_snapshot", lambda: (3, 11))
    assert concurrency._get_effective_host_limit() == 3

    concurrency._invalidate_pooling_snapshot()
    monkeypatch.setattr(concurrency, "_pooling_get_limit_snapshot", lambda: None)
    monkeypatch.setattr(concurrency, "_pooling_host_limit", lambda: "6")
    assert concurrency._get_effective_host_limit() == 6

    concurrency._invalidate_pooling_snapshot()
    monkeypatch.setattr(concurrency, "_pooling_host_limit", lambda: (_ for _ in ()).throw(ValueError("bad")))
    assert concurrency._get_effective_host_limit() is None


def test_host_limit_semaphore_refreshes_version_and_rejects_foreign_loop(monkeypatch) -> None:
    async def scenario() -> None:
        current_loop = asyncio.get_running_loop()
        stale = asyncio.Semaphore(5)
        setattr(stale, "_ai_trading_host_limit", 5)
        setattr(stale, "_ai_trading_host_limit_version", 1)
        concurrency._POOLING_LIMIT_STATE = (2, 2)

        refreshed = asyncio.Semaphore(2)
        setattr(refreshed, "_ai_trading_host_limit", 2)
        setattr(refreshed, "_ai_trading_host_limit_version", 2)

        monkeypatch.setattr(concurrency, "_running_under_pytest_worker", lambda: False)
        monkeypatch.setattr(concurrency, "_pooling_get_host_semaphore", lambda: stale)
        monkeypatch.setattr(concurrency, "_pooling_refresh_host_semaphore", lambda loop=None: refreshed)

        assert concurrency._get_host_limit_semaphore() is refreshed
        assert concurrency._POOLING_LIMIT_STATE == (2, 2)

        foreign = asyncio.Semaphore(1)
        setattr(foreign, "_bound_loop", object())
        monkeypatch.setattr(concurrency, "_pooling_get_host_semaphore", lambda: foreign)
        monkeypatch.setattr(concurrency, "_pooling_refresh_host_semaphore", lambda loop=None: foreign)
        assert concurrency._get_host_limit_semaphore() is None

        good = asyncio.Semaphore(1)
        setattr(good, "_bound_loop", current_loop)
        monkeypatch.setattr(concurrency, "_pooling_get_host_semaphore", lambda: good)
        assert concurrency._get_host_limit_semaphore() is good

    asyncio.run(scenario())


def test_scan_recreates_asyncio_primitives_inside_supported_containers() -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(3)
        bounded = asyncio.BoundedSemaphore(4)
        ns = SimpleNamespace(lock=lock)
        frozen = _FrozenHolder(lock=asyncio.Lock(), values=(semaphore,))
        slots_dataclass = _SlotsDataclassHolder(asyncio.Semaphore(2))
        slots_holder = _SlotsHolder(asyncio.Lock())
        project_object = _ProjectObject(asyncio.Lock())
        custom_sequence = UserList([asyncio.Lock()])
        custom_set = _CustomSet({asyncio.Lock()})
        original_slots_lock = slots_holder.lock
        original_project_lock = project_object.lock
        original_sequence_lock = custom_sequence[0]
        original_set_lock = next(iter(custom_set))
        original_bounded = bounded
        payload: dict[Any, Any] = {
            "list": [asyncio.Lock()],
            "tuple": (asyncio.Lock(),),
            "set": {asyncio.Lock()},
            "frozenset": frozenset({asyncio.Lock()}),
            "deque": deque([asyncio.Lock()]),
            "namespace": ns,
            "frozen": frozen,
            "slots_dataclass": slots_dataclass,
            "slots_holder": slots_holder,
            "project_object": project_object,
            "custom_sequence": custom_sequence,
            "custom_set": custom_set,
            "bounded": bounded,
        }

        scanned = concurrency._scan(payload, set(), loop)

        assert scanned is payload
        assert payload["namespace"].lock is not lock
        assert payload["frozen"] is not frozen
        assert payload["frozen"].lock is not frozen.lock
        assert payload["frozen"].values[0] is not semaphore
        assert payload["slots_dataclass"].semaphore is not slots_dataclass.semaphore
        assert payload["slots_holder"].lock is not original_slots_lock
        assert payload["project_object"].lock is not original_project_lock
        assert payload["custom_sequence"][0] is not original_sequence_lock
        assert next(iter(payload["custom_set"])) is not original_set_lock
        assert payload["bounded"] is not original_bounded
        assert isinstance(payload["bounded"], asyncio.BoundedSemaphore)

        proxy = MappingProxyType({"lock": asyncio.Lock()})
        rebuilt_proxy = concurrency._scan(proxy, set(), loop)
        assert isinstance(rebuilt_proxy, MappingProxyType)
        assert rebuilt_proxy["lock"] is not proxy["lock"]

    asyncio.run(scenario())


def test_closure_rebinding_replaces_nested_lock_for_worker() -> None:
    async def scenario() -> None:
        holder = SimpleNamespace(lock=asyncio.Lock())
        original = holder.lock

        async def worker(_symbol: str) -> str:
            async with holder.lock:
                return "ok"

        concurrency._rebind_worker_closure(worker, asyncio.get_running_loop())

        assert holder.lock is not original
        assert await worker("AAPL") == "ok"

    asyncio.run(scenario())


def test_non_pytest_scheduler_records_success_failure_and_host_permits(monkeypatch, caplog) -> None:
    async def scenario() -> None:
        concurrency.reset_tracking_state()
        concurrency.reset_peak_simultaneous_workers()
        host_semaphore = asyncio.Semaphore(1)
        setattr(host_semaphore, "_ai_trading_host_limit", 1)

        monkeypatch.setattr(concurrency, "_running_under_pytest_worker", lambda: False)
        monkeypatch.setattr(concurrency, "_get_effective_host_limit", lambda: 2)
        monkeypatch.setattr(concurrency, "_get_host_limit_semaphore", lambda: host_semaphore)

        active = 0
        peak = 0

        async def worker(symbol: str) -> str:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0)
                if symbol == "FAIL":
                    raise RuntimeError("boom")
                return symbol.lower()
            finally:
                active -= 1

        with caplog.at_level(logging.WARNING):
            results, succeeded, failed = await concurrency.run_with_concurrency(
                ["OK1", "FAIL", "OK2"], worker, max_concurrency=5
            )

        assert results == {"OK1": "ok1", "FAIL": None, "OK2": "ok2"}
        assert succeeded == {"OK1", "OK2"}
        assert failed == {"FAIL"}
        assert any(
            record.message == "FALLBACK_WORKER_FAILED"
            and getattr(record, "symbol", None) == "FAIL"
            and getattr(record, "error_type", None) == "RuntimeError"
            for record in caplog.records
        )
        assert peak == 1
        assert concurrency._HOST_PERMITS_HELD == 0
        assert concurrency.LAST_RUN_PEAK_SIMULTANEOUS_WORKERS == 1

    asyncio.run(scenario())


def test_run_with_concurrency_preserves_cancellation(monkeypatch) -> None:
    async def scenario() -> None:
        concurrency.reset_tracking_state()
        monkeypatch.setattr(concurrency, "_running_under_pytest_worker", lambda: False)
        monkeypatch.setattr(concurrency, "_get_effective_host_limit", lambda: None)
        monkeypatch.setattr(concurrency, "_get_host_limit_semaphore", lambda: None)

        async def worker(_symbol: str) -> str:
            raise asyncio.CancelledError()

        try:
            await concurrency.run_with_concurrency(["STOP"], worker, max_concurrency=1)
        except asyncio.CancelledError:
            pass
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("worker cancellation must propagate")

        assert "STOP" not in concurrency.FAILED_SYMBOLS

    asyncio.run(scenario())


def test_run_with_concurrency_timeout_does_not_rerun_worker(monkeypatch) -> None:
    async def scenario() -> None:
        concurrency.reset_tracking_state()
        monkeypatch.setattr(concurrency, "_running_under_pytest_worker", lambda: False)
        monkeypatch.setattr(concurrency, "_get_effective_host_limit", lambda: None)
        monkeypatch.setattr(concurrency, "_get_host_limit_semaphore", lambda: None)
        calls: list[str] = []

        async def worker(symbol: str) -> str:
            calls.append(symbol)
            if calls.count(symbol) > 1:
                return "rerun"
            await asyncio.sleep(10)
            return "slow"

        results, succeeded, failed = await concurrency.run_with_concurrency(
            ["AAPL"],
            worker,
            max_concurrency=1,
            timeout_s=0.001,
        )

        assert calls == ["AAPL"]
        assert results == {"AAPL": None}
        assert succeeded == set()
        assert failed == {"AAPL"}

    asyncio.run(scenario())


def test_peak_and_permit_helpers_are_defensive(monkeypatch) -> None:
    peaks: list[int] = []
    pooling_records: list[int] = []
    monkeypatch.setattr(concurrency, "_http_host_limit", SimpleNamespace(record_peak=peaks.append))
    monkeypatch.setattr(concurrency, "_pooling_record_concurrency", pooling_records.append)

    concurrency.PEAK_SIMULTANEOUS_WORKERS = "bad"  # type: ignore[assignment]
    concurrency._update_peak_counters(4)
    concurrency._update_peak_counters(2)

    assert concurrency.PEAK_SIMULTANEOUS_WORKERS == 4
    assert concurrency.LAST_RUN_PEAK_SIMULTANEOUS_WORKERS == 2
    assert peaks == [4]
    assert pooling_records == [4]

    concurrency._HOST_PERMITS_HELD = 0
    concurrency._release_host_permit()
    assert concurrency._HOST_PERMITS_HELD == 0
    concurrency._increment_host_permits()
    concurrency._increment_host_permits()
    concurrency._release_host_permit()
    assert concurrency._HOST_PERMITS_HELD == 1
