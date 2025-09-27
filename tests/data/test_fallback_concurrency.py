import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

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


def test_run_with_concurrency_peak_counter_respects_limit():
    tracker_lock = asyncio.Lock()
    running = 0
    max_seen = 0

    async def worker(sym: str) -> str:
        nonlocal running, max_seen
        async with tracker_lock:
            running += 1
            if running > max_seen:
                max_seen = running
        try:
            await asyncio.sleep(0.01)
            return sym
        finally:
            async with tracker_lock:
                running -= 1

    symbols = [f"PEAK{i}" for i in range(6)]

    results, succeeded, failed = asyncio.run(
        concurrency.run_with_concurrency(symbols, worker, max_concurrency=3)
    )

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed
    assert max_seen <= 3
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS <= 3


def test_run_with_concurrency_rebinds_nested_dataclass_lock():
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

    wrapped = Wrapper(holder=InnerLockHolder(lock=build_lock_in_fresh_loop()))

    async def worker(sym: str) -> str:
        async with wrapped.holder.lock:
            await asyncio.sleep(0)
            return sym

    symbols = ["A", "B", "C"]

    async def run_with_timeout():
        return await asyncio.wait_for(
            concurrency.run_with_concurrency(symbols, worker, max_concurrency=2),
            timeout=1,
        )

    results, succeeded, failed = asyncio.run(run_with_timeout())

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed


def test_run_with_concurrency_rebinds_frozen_nested_dataclass_lock_from_namespace():
    @dataclass(frozen=True)
    class FrozenInnerHolder:
        lock: asyncio.Lock

    @dataclass
    class Wrapper:
        namespace: SimpleNamespace

    def build_lock_in_fresh_loop() -> asyncio.Lock:
        loop = asyncio.new_event_loop()
        try:
            async def _factory() -> asyncio.Lock:
                return asyncio.Lock()

            return loop.run_until_complete(_factory())
        finally:
            loop.close()

    original_lock = build_lock_in_fresh_loop()
    wrapped = Wrapper(namespace=SimpleNamespace(holder=FrozenInnerHolder(lock=original_lock)))

    async def worker(sym: str) -> str:
        async with wrapped.namespace.holder.lock:
            await asyncio.sleep(0)
            return sym

    symbols = ["A", "B", "C"]

    results, succeeded, failed = asyncio.run(
        concurrency.run_with_concurrency(symbols, worker, max_concurrency=2)
    )

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed
    assert wrapped.namespace.holder.lock is not original_lock


def test_run_with_concurrency_rebinds_lock_inside_frozenset_slots_dataclass():
    @dataclass(frozen=True, slots=True, eq=False)
    class FrozenSlotsHolder:
        lock: asyncio.Lock

    def build_lock_in_fresh_loop() -> asyncio.Lock:
        loop = asyncio.new_event_loop()
        try:
            async def _factory() -> asyncio.Lock:
                return asyncio.Lock()

            return loop.run_until_complete(_factory())
        finally:
            loop.close()

    original_lock = build_lock_in_fresh_loop()
    namespace = SimpleNamespace(payload=frozenset({FrozenSlotsHolder(lock=original_lock)}))
    async def worker(sym: str) -> str:
        holder = next(iter(namespace.payload))
        async with holder.lock:
            await asyncio.sleep(0)
            return sym

    symbols = ["A", "B", "C"]
    results, succeeded, failed = asyncio.run(
        concurrency.run_with_concurrency(symbols, worker, max_concurrency=2)
    )

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed

    holder_after = next(iter(namespace.payload))
    assert holder_after.lock is not original_lock


def test_run_with_concurrency_rebinds_lock_inside_tuple_structure():
    @dataclass
    class Holder:
        lock: asyncio.Lock

    def build_lock_in_fresh_loop() -> asyncio.Lock:
        loop = asyncio.new_event_loop()
        try:
            async def _factory() -> asyncio.Lock:
                return asyncio.Lock()

            return loop.run_until_complete(_factory())
        finally:
            loop.close()

    original_lock = build_lock_in_fresh_loop()
    namespace = SimpleNamespace(payload=(Holder(lock=original_lock),))

    async def worker(sym: str) -> str:
        holder = namespace.payload[0]
        async with holder.lock:
            await asyncio.sleep(0)
            return sym

    symbols = ["T1", "T2"]
    results, succeeded, failed = asyncio.run(
        concurrency.run_with_concurrency(symbols, worker, max_concurrency=2)
    )

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed

    holder_after = namespace.payload[0]
    assert holder_after.lock is not original_lock


def test_run_with_concurrency_rebinds_lock_inside_frozenset_of_tuple():
    @dataclass(frozen=True)
    class FrozenHolder:
        lock: asyncio.Lock

    def build_lock_in_fresh_loop() -> asyncio.Lock:
        loop = asyncio.new_event_loop()
        try:
            async def _factory() -> asyncio.Lock:
                return asyncio.Lock()

            return loop.run_until_complete(_factory())
        finally:
            loop.close()

    original_lock = build_lock_in_fresh_loop()
    namespace = SimpleNamespace(
        payload=frozenset({(FrozenHolder(lock=original_lock), "marker")})
    )

    async def worker(sym: str) -> str:
        holder_tuple = next(iter(namespace.payload))
        holder = holder_tuple[0]
        async with holder.lock:
            await asyncio.sleep(0)
            return sym

    symbols = ["F1", "F2"]
    results, succeeded, failed = asyncio.run(
        concurrency.run_with_concurrency(symbols, worker, max_concurrency=2)
    )

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed

    holder_tuple_after = next(iter(namespace.payload))
    holder_after = holder_tuple_after[0]
    assert holder_after.lock is not original_lock


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
