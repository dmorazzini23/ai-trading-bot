import asyncio
from dataclasses import dataclass

from ai_trading.data.fallback import concurrency


def test_run_with_concurrency_returns_results():
    async def worker(sym: str) -> str:
        await asyncio.sleep(0)
        return sym

    symbols = ["AAPL", "MSFT", "GOOG"]
    results, succeeded, failed = asyncio.run(
        concurrency.run_with_concurrency(symbols, worker, max_concurrency=2)
    )

    assert results and set(results) == set(symbols)
    assert succeeded == set(symbols)
    assert not failed


def test_run_with_concurrency_respects_limit():
    @dataclass
    class InnerLockHolder:
        lock: asyncio.Lock

    @dataclass
    class Wrapper:
        holder: InnerLockHolder

    def build_lock_in_fresh_loop() -> asyncio.Lock:
        loop = asyncio.new_event_loop()
        try:
            async def _factory() -> asyncio.Lock:
                return asyncio.Lock()

            return loop.run_until_complete(_factory())
        finally:
            loop.close()

    foreign_lock = build_lock_in_fresh_loop()
    wrapped = Wrapper(holder=InnerLockHolder(lock=foreign_lock))

    max_seen = 0
    current = 0
    tracker_lock = asyncio.Lock()

    async def worker(sym: str) -> str:
        nonlocal max_seen, current
        async with tracker_lock:
            current += 1
            if current > max_seen:
                max_seen = current
        try:
            async with wrapped.holder.lock:
                await asyncio.sleep(0.01)
            return sym
        finally:
            async with tracker_lock:
                current -= 1

    symbols = [f"SYM{i}" for i in range(5)]

    async def run_with_timeout():
        return await asyncio.wait_for(
            concurrency.run_with_concurrency(symbols, worker, max_concurrency=2),
            timeout=1,
        )

    results, succeeded, failed = asyncio.run(run_with_timeout())

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed
    assert max_seen <= 2
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS <= 2


def test_run_with_concurrency_rebinds_foreign_loop_lock():
    def build_lock_in_fresh_loop() -> asyncio.Lock:
        loop = asyncio.new_event_loop()
        try:
            async def _factory() -> asyncio.Lock:
                return asyncio.Lock()

            return loop.run_until_complete(_factory())
        finally:
            loop.close()

    external_lock = build_lock_in_fresh_loop()

    async def worker(sym: str) -> str:
        async with external_lock:
            await asyncio.sleep(0)
            return sym

    symbols = ["AAPL", "MSFT"]
    results, succeeded, failed = asyncio.run(
        concurrency.run_with_concurrency(symbols, worker, max_concurrency=2)
    )

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed


def test_run_with_concurrency_handles_blocking_and_failures():
    max_seen = 0
    current = 0
    lock = asyncio.Lock()
    release_blocker = asyncio.Event()

    async def worker(sym: str) -> str:
        nonlocal max_seen, current
        async with lock:
            current += 1
            if current > max_seen:
                max_seen = current
        try:
            if sym == "BLOCK":
                await release_blocker.wait()
            elif sym == "FAIL":
                await asyncio.sleep(0)
                raise RuntimeError("boom")
            else:
                await asyncio.sleep(0.01)
            return sym
        finally:
            async with lock:
                current -= 1

    symbols = ["BLOCK", "OK1", "FAIL", "OK2"]

    async def run_and_release():
        task = asyncio.create_task(
            concurrency.run_with_concurrency(symbols, worker, max_concurrency=2)
        )
        await asyncio.sleep(0.05)
        release_blocker.set()
        return await asyncio.wait_for(task, 1)

    results, succeeded, failed = asyncio.run(run_and_release())

    assert max_seen <= 2
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS <= 2
    assert results["FAIL"] is None
    assert "FAIL" in failed
    assert "BLOCK" in succeeded
