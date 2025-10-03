import asyncio
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

import pytest

from ai_trading.data.fallback import concurrency
from ai_trading.data import fallback_concurrency as legacy_concurrency


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


def test_run_with_concurrency_respects_limit_minimal():
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

    symbols = [f"MIN{i}" for i in range(5)]

    results, succeeded, failed = asyncio.run(
        concurrency.run_with_concurrency(symbols, worker, max_concurrency=2)
    )

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed
    assert max_seen == 2
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS == 2


def test_run_with_concurrency_peak_counter_respects_limit_minimal():
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

    concurrency.reset_peak_simultaneous_workers()

    symbols = [f"PEAK_SIMPLE{i}" for i in range(4)]

    results, succeeded, failed = asyncio.run(
        concurrency.run_with_concurrency(symbols, worker, max_concurrency=3)
    )

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed
    assert max_seen == 3
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS == max_seen


def test_run_with_concurrency_peak_counter_is_monotonic_across_runs():
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

    concurrency.PEAK_SIMULTANEOUS_WORKERS = 5

    symbols = [f"PEAK_MONO{i}" for i in range(4)]

    results, succeeded, failed = asyncio.run(
        concurrency.run_with_concurrency(symbols, worker, max_concurrency=2)
    )

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed
    assert max_seen == 2
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS == max_seen


def test_run_with_concurrency_peak_matches_requested_limits(monkeypatch):
    symbols = [f"REQ{i}" for i in range(6)]

    def run_scenario(requested_limit: int, host_limit: int | None = None) -> int:
        concurrency.reset_tracking_state()

        if host_limit is not None:
            semaphore = asyncio.Semaphore(host_limit)
            monkeypatch.setattr(concurrency, "_get_effective_host_limit", lambda: host_limit)
            monkeypatch.setattr(concurrency, "_get_host_limit_semaphore", lambda: semaphore)

        async def orchestrate() -> int:
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

            results, succeeded, failed = await concurrency.run_with_concurrency(
                symbols,
                worker,
                max_concurrency=requested_limit,
            )

            assert results == {symbol: symbol for symbol in symbols}
            assert succeeded == set(symbols)
            assert not failed

            return max_seen

        return asyncio.run(orchestrate())

    sequential_limit = 3
    sequential_max = run_scenario(sequential_limit)
    assert sequential_max == sequential_limit
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS == sequential_limit

    host_limit = 2
    host_max = run_scenario(5, host_limit=host_limit)
    assert host_max == host_limit
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS == host_limit


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
    assert max_seen == 2
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS == 2


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
    assert max_seen == 3
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS == 3


@pytest.mark.parametrize("module", (concurrency, legacy_concurrency))
def test_run_with_concurrency_respects_host_limit(module):
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

    symbols = [f"HOST{i}" for i in range(6)]

    module.reset_tracking_state()
    module.reset_peak_simultaneous_workers()

    original_host_limit = module._get_effective_host_limit

    def _fake_host_limit() -> int:
        return 2

    module._get_effective_host_limit = _fake_host_limit
    try:
        results, succeeded, failed = asyncio.run(
            module.run_with_concurrency(symbols, worker, max_concurrency=5)
        )
    finally:
        module._get_effective_host_limit = original_host_limit

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed
    assert max_seen == 2
    assert module.PEAK_SIMULTANEOUS_WORKERS == 2


@pytest.mark.parametrize("module", (concurrency, legacy_concurrency))
def test_run_with_concurrency_host_limit_floors_to_one(module):
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
            await asyncio.sleep(0)
            return sym
        finally:
            async with tracker_lock:
                running -= 1

    symbols = [f"HOSTF{i}" for i in range(4)]

    module.reset_tracking_state()
    module.reset_peak_simultaneous_workers()

    original_host_limit = module._get_effective_host_limit

    def _fake_host_limit() -> int:
        return 0

    module._get_effective_host_limit = _fake_host_limit
    try:
        results, succeeded, failed = asyncio.run(
            module.run_with_concurrency(symbols, worker, max_concurrency=3)
        )
    finally:
        module._get_effective_host_limit = original_host_limit

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed
    assert max_seen == 1
    assert module.PEAK_SIMULTANEOUS_WORKERS == 1


def test_run_with_concurrency_waiter_cancellation_does_not_overshoot_limit():
    class StrictSemaphore(asyncio.Semaphore):
        def __init__(self, value: int):
            super().__init__(value)
            self._initial_value = value

        def release(self) -> None:
            super().release()
            if self._value > self._initial_value:
                raise AssertionError(
                    f"Semaphore value exceeded initial limit: {self._value} > {self._initial_value}"
                )

    original_semaphore = concurrency.asyncio.Semaphore
    concurrency.asyncio.Semaphore = StrictSemaphore

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
            await asyncio.sleep(0.1)
            return sym
        finally:
            async with tracker_lock:
                running -= 1

    symbols = [f"WAIT{i}" for i in range(6)]

    try:
        results, succeeded, failed = asyncio.run(
            concurrency.run_with_concurrency(symbols, worker, max_concurrency=2, timeout_s=0.05)
        )
    finally:
        concurrency.asyncio.Semaphore = original_semaphore

    assert any(value is None for value in results.values())
    assert failed
    assert max_seen == 2
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS == max_seen


def test_run_with_concurrency_host_semaphore_cancellation_does_not_over_release():
    class StrictSemaphore(asyncio.Semaphore):
        def __init__(self, value: int):
            super().__init__(value)
            self._initial_value = value
            self._held = 0

        async def acquire(self) -> bool:
            result = await super().acquire()
            self._held += 1
            return result

        def release(self) -> None:
            if self._held <= 0:
                raise AssertionError("release called without a matching acquire")
            self._held -= 1
            super().release()
            if self._value > self._initial_value:
                raise AssertionError(
                    f"Semaphore value exceeded initial limit: {self._value} > {self._initial_value}"
                )

    host_semaphore = StrictSemaphore(1)

    original_host_limit = concurrency._get_effective_host_limit
    original_get_host_semaphore = concurrency._get_host_limit_semaphore

    def _fake_host_limit() -> int:
        return 1

    def _fake_get_host_semaphore() -> asyncio.Semaphore:
        return host_semaphore

    concurrency._get_effective_host_limit = _fake_host_limit
    concurrency._get_host_limit_semaphore = _fake_get_host_semaphore

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
            await asyncio.sleep(0.1)
            return sym
        finally:
            async with tracker_lock:
                running -= 1

    symbols = [f"HOST_WAIT{i}" for i in range(4)]

    try:
        results, succeeded, failed = asyncio.run(
            concurrency.run_with_concurrency(symbols, worker, max_concurrency=3, timeout_s=0.05)
        )
    finally:
        concurrency._get_effective_host_limit = original_host_limit
        concurrency._get_host_limit_semaphore = original_get_host_semaphore

    assert any(value is None for value in results.values())
    assert failed
    assert max_seen == 1
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS == max_seen
    assert host_semaphore._held == 0


def test_run_with_concurrency_back_to_back_host_limit_runs_reset_peak(monkeypatch):
    class TrackingSemaphore(asyncio.Semaphore):
        def __init__(self, value: int):
            super().__init__(value)
            self._initial_value = value
            self._held = 0

        async def acquire(self) -> bool:  # type: ignore[override]
            await super().acquire()
            self._held += 1
            return True

        def release(self) -> None:  # type: ignore[override]
            if self._held <= 0:
                raise AssertionError("release called without a matching acquire")
            self._held -= 1
            super().release()
            if self._value > self._initial_value:
                raise AssertionError(
                    f"Semaphore value exceeded initial limit: {self._value} > {self._initial_value}"
                )

    symbols = [f"SEQ{i}" for i in range(6)]

    async def run_once(limit: int, *, timeout: float | None = None) -> tuple[int, TrackingSemaphore]:
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
                await asyncio.sleep(0.1)
                return sym
            finally:
                async with tracker_lock:
                    running -= 1

        semaphore = TrackingSemaphore(limit)

        monkeypatch.setattr(concurrency, "_get_effective_host_limit", lambda: limit)
        monkeypatch.setattr(concurrency, "_get_host_limit_semaphore", lambda: semaphore)

        await concurrency.run_with_concurrency(
            symbols,
            worker,
            max_concurrency=5,
            timeout_s=timeout,
        )

        return max_seen, semaphore

    async def orchestrate() -> None:
        max_seen_timeout, timeout_semaphore = await run_once(1, timeout=0.05)
        assert max_seen_timeout == 1
        assert concurrency.PEAK_SIMULTANEOUS_WORKERS == max_seen_timeout
        assert timeout_semaphore._held == 0

        max_seen_success, success_semaphore = await run_once(3)
        assert max_seen_success == 3
        assert concurrency.PEAK_SIMULTANEOUS_WORKERS == max_seen_success
        assert success_semaphore._held == 0

    asyncio.run(orchestrate())


def test_run_with_concurrency_smaller_run_observes_lower_peak_during_execution():
    concurrency.reset_peak_simultaneous_workers()

    symbols_high = ["HIGH1", "HIGH2", "HIGH3"]

    async def high_worker(sym: str) -> str:
        await asyncio.sleep(0)
        return sym

    asyncio.run(
        concurrency.run_with_concurrency(symbols_high, high_worker, max_concurrency=3)
    )

    assert concurrency.PEAK_SIMULTANEOUS_WORKERS == 3

    observed_peaks: list[int] = []
    first_started = asyncio.Event()
    release_gate = asyncio.Event()

    async def low_worker(sym: str) -> str:
        observed_peaks.append(concurrency.PEAK_SIMULTANEOUS_WORKERS)
        if not first_started.is_set():
            first_started.set()
        await release_gate.wait()
        return sym

    async def run_low() -> None:
        task = asyncio.create_task(
            concurrency.run_with_concurrency(["LOW1", "LOW2"], low_worker, max_concurrency=1)
        )
        await asyncio.wait_for(first_started.wait(), timeout=1)
        release_gate.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(run_low())

    assert observed_peaks
    assert len(observed_peaks) == 2
    assert max(observed_peaks) == 1
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS == 1


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


def test_run_with_concurrency_rebinds_lock_and_semaphore_inside_mapping_proxy():
    @dataclass(frozen=True, eq=False)
    class FrozenHolder:
        lock: asyncio.Lock
        semaphore: asyncio.Semaphore

    def build_holder_in_fresh_loop() -> FrozenHolder:
        loop = asyncio.new_event_loop()
        try:
            async def _factory() -> FrozenHolder:
                return FrozenHolder(lock=asyncio.Lock(), semaphore=asyncio.Semaphore(1))

            return loop.run_until_complete(_factory())
        finally:
            loop.close()

    original_holder = build_holder_in_fresh_loop()
    original_proxy = MappingProxyType({"holder": original_holder})
    namespace = SimpleNamespace(payload=(original_proxy,))

    async def worker(sym: str) -> str:
        mapping = namespace.payload[0]
        holder = mapping["holder"]
        async with holder.lock:
            async with holder.semaphore:
                await asyncio.sleep(0)
                return sym

    symbols = ["MP1", "MP2"]
    results, succeeded, failed = asyncio.run(
        concurrency.run_with_concurrency(symbols, worker, max_concurrency=2)
    )

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed

    mapping_after = namespace.payload[0]
    assert mapping_after is not original_proxy
    holder_after = mapping_after["holder"]
    assert holder_after.lock is not original_holder.lock
    assert holder_after.semaphore is not original_holder.semaphore


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


def test_run_with_concurrency_rebinds_lock_inside_deque_nested_dataclass():
    @dataclass
    class Inner:
        lock: asyncio.Lock

    @dataclass
    class Wrapper:
        holders: deque[Inner]

    def build_lock_in_fresh_loop() -> asyncio.Lock:
        loop = asyncio.new_event_loop()
        try:
            async def _factory() -> asyncio.Lock:
                return asyncio.Lock()

            return loop.run_until_complete(_factory())
        finally:
            loop.close()

    original_lock = build_lock_in_fresh_loop()
    wrapped = Wrapper(holders=deque([Inner(lock=original_lock)]))

    async def worker(sym: str) -> str:
        holder = wrapped.holders[0]
        async with holder.lock:
            await asyncio.sleep(0)
            return sym

    symbols = ["DQ1", "DQ2"]
    results, succeeded, failed = asyncio.run(
        concurrency.run_with_concurrency(symbols, worker, max_concurrency=2)
    )

    assert results == {symbol: symbol for symbol in symbols}
    assert succeeded == set(symbols)
    assert not failed

    holder_after = wrapped.holders[0]
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
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS == max_seen
    assert results["FAIL"] is None
    assert "FAIL" in failed
    assert "BLOCK" in succeeded


def test_legacy_shim_reuses_concurrency_module():
    assert legacy_concurrency is concurrency
    assert legacy_concurrency.run_with_concurrency is concurrency.run_with_concurrency
    assert legacy_concurrency.__doc__ == concurrency.__doc__


