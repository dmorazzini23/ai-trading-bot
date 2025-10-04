import asyncio
from types import SimpleNamespace

import pytest

from ai_trading.data.fallback import concurrency


def test_worker_closure_preserves_foreign_host_semaphore() -> None:
    concurrency.reset_tracking_state()
    concurrency.reset_peak_simultaneous_workers()

    host_limit = 2
    host_semaphore = asyncio.Semaphore(host_limit)
    setattr(host_semaphore, "_loop", object())
    setattr(host_semaphore, "_ai_trading_host_limit", host_limit)
    setattr(host_semaphore, "_ai_trading_host_limit_version", 1)

    holder = SimpleNamespace(semaphore=host_semaphore)

    symbols = ["ASYNC_A", "ASYNC_B"]

    async def orchestrate() -> tuple[dict[str, str | None], set[str], set[str]]:
        async def worker(symbol: str) -> str:
            assert holder.semaphore is host_semaphore
            await asyncio.sleep(0)
            return symbol

        return await concurrency.run_with_concurrency(
            symbols,
            worker,
            max_concurrency=host_limit,
        )

    results, succeeded, failed = asyncio.run(orchestrate())

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed
    assert holder.semaphore is host_semaphore


def test_foreign_host_semaphore_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    concurrency.reset_tracking_state()
    concurrency.reset_peak_simultaneous_workers()

    host_limit = 2
    foreign_loop = object()
    host_semaphore = asyncio.Semaphore(host_limit)
    setattr(host_semaphore, "_loop", foreign_loop)
    setattr(host_semaphore, "_ai_trading_host_limit", host_limit)
    setattr(host_semaphore, "_ai_trading_host_limit_version", 1)

    monkeypatch.setattr(concurrency, "_get_effective_host_limit", lambda: host_limit)
    monkeypatch.setattr(concurrency, "_get_host_limit_semaphore", lambda: host_semaphore)

    tracker_lock = asyncio.Lock()
    running = 0
    peak = 0

    symbols = [f"ASYNC_SEM{i}" for i in range(4)]

    async def orchestrate() -> tuple[dict[str, str | None], set[str], set[str]]:
        async def worker(symbol: str) -> str:
            nonlocal running, peak
            async with tracker_lock:
                running += 1
                peak = max(peak, running)
            try:
                await asyncio.sleep(0)
                return symbol
            finally:
                async with tracker_lock:
                    running -= 1

        return await concurrency.run_with_concurrency(
            symbols,
            worker,
            max_concurrency=5,
        )

    results, succeeded, failed = asyncio.run(orchestrate())

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed
    assert peak == host_limit
    assert host_semaphore._value == host_limit
    assert concurrency.LAST_RUN_PEAK_SIMULTANEOUS_WORKERS == host_limit
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS >= host_limit


def test_peak_counter_remains_monotonic_without_reset() -> None:
    concurrency.reset_tracking_state()
    concurrency.reset_peak_simultaneous_workers()

    tracker_lock = asyncio.Lock()
    running = 0
    peak = 0

    symbols = [f"ASYNC_MONO{i}" for i in range(6)]

    async def run_once(limit: int) -> tuple[dict[str, str | None], set[str], set[str]]:
        async def worker(symbol: str) -> str:
            nonlocal running, peak
            async with tracker_lock:
                running += 1
                peak = max(peak, running)
            try:
                await asyncio.sleep(0.01)
                return symbol
            finally:
                async with tracker_lock:
                    running -= 1

        return await concurrency.run_with_concurrency(
            symbols,
            worker,
            max_concurrency=limit,
        )

    results, succeeded, failed = asyncio.run(run_once(3))

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed
    assert peak == 3
    assert concurrency.LAST_RUN_PEAK_SIMULTANEOUS_WORKERS == 3
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS >= 3

    peak = 0
    running = 0

    results, succeeded, failed = asyncio.run(run_once(1))

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed
    assert peak == 1
    assert concurrency.LAST_RUN_PEAK_SIMULTANEOUS_WORKERS == 1
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS >= 3
