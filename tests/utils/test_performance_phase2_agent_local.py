from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.utils import performance as perf


def test_performance_cache_expiry_eviction_and_decorator_key_paths() -> None:
    cache = perf.PerformanceCache(max_size=2, ttl_seconds=10)
    cache.set("old", "stale")
    cache._cache["old"]["timestamp"] = datetime.now(UTC) - timedelta(seconds=20)  # noqa: SLF001
    assert cache.get("old") is None

    cache.set("first", 1)
    cache.set("second", 2)
    cache._cache["first"]["last_accessed"] = datetime(1970, 1, 1, tzinfo=UTC)  # noqa: SLF001
    cache.set("third", 3)

    assert cache.get("first") is None
    assert cache.get("second") == 2
    assert cache.stats()["size"] == 2
    assert str(perf.BenchmarkResult("op", 12.345, 1000.0, 4.5, 2)) == "op: 12.35ms, 1000 ops/sec, 4.5MB"

    calls = {"count": 0}

    @perf.cached_operation(cache_ttl=30)
    def add(value: int, *, labels: dict[str, str]) -> int:
        calls["count"] += 1
        return value + len(labels)

    assert add(3, labels={"b": "2", "a": "1"}) == 5
    assert add(3, labels={"a": "1", "b": "2"}) == 5
    assert calls["count"] == 1

    custom_calls = {"count": 0}

    @perf.cached_operation(cache_key_func=lambda value: f"custom:{value}")
    def double(value: int) -> int:
        custom_calls["count"] += 1
        return value * 2

    assert double(4) == 8
    assert double(4) == 8
    assert custom_calls["count"] == 1


def test_parallel_processor_handles_sequential_parallel_and_indicator_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = perf.ParallelProcessor(max_workers=1)
    assert processor.parallel_apply(lambda chunk, scale: chunk * scale, [1, 2], 3) == [3, 6]

    class FakeFuture:
        def __init__(self, value: Any = None, error: Exception | None = None) -> None:
            self.value = value
            self.error = error

        def result(self) -> Any:
            if self.error is not None:
                raise self.error
            return self.value

    class FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, *_exc: Any) -> None:
            return None

        def submit(self, func: Any, chunk: Any, *args: Any, **kwargs: Any) -> FakeFuture:
            if chunk == "bad":
                return FakeFuture(error=ValueError("boom"))
            return FakeFuture(func(chunk, *args, **kwargs))

    submitted: list[FakeFuture] = []

    def fake_as_completed(futures: dict[FakeFuture, int]) -> list[FakeFuture]:
        submitted.extend(futures)
        return list(futures)

    monkeypatch.setattr(perf, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(perf, "as_completed", fake_as_completed)

    parallel = perf.ParallelProcessor(max_workers=2)
    assert parallel.parallel_apply(lambda chunk: chunk.upper(), ["a", "bad", "c"]) == ["A", None, "C"]
    assert len(submitted) == 3

    frame = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    indicators = [
        {"name": "ret", "function": lambda data: data["close"].pct_change()},
        {"name": "wide", "function": lambda data: pd.DataFrame({"x": data["close"], "y": data["close"] * 2})},
        {"name": "bad", "function": lambda _data: (_ for _ in ()).throw(ValueError("bad"))},
    ]
    result = parallel.parallel_indicators(frame, indicators)

    assert {"ret", "wide_x", "wide_y"} <= set(result.columns)
    assert "bad" not in result.columns


def test_vectorized_operations_and_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    series = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0])
    zscore = perf.VectorizedOperations.rolling_zscore(series, window=3)
    corr_short = perf.VectorizedOperations.rolling_correlation(series, series.shift(10), window=3)
    returns = perf.VectorizedOperations.fast_returns(series, periods=2)
    indicators = perf.VectorizedOperations.batch_technical_indicators(
        pd.DataFrame({"close": series}),
        [2, 99],
    )

    assert zscore.notna().any()
    assert corr_short.empty
    assert returns.iloc[:2].isna().all()
    assert returns.iloc[2] == pytest.approx(3.0)
    assert {"sma_2", "ema_2", "vol_2", "mom_2", "bb_upper_2", "bb_lower_2"} <= set(indicators.columns)
    assert not any(column.endswith("_99") for column in indicators.columns)

    monkeypatch.setattr(perf.time, "perf_counter", iter([10.0, 10.5]).__next__)
    monkeypatch.setattr(perf.mp, "cpu_count", lambda: 6)
    monkeypatch.setattr(perf, "load_pandas", lambda: pd)
    monkeypatch.setitem(__import__("sys").modules, "psutil", SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=20 * 1024 * 1024))))

    benchmark = perf.benchmark_operation("series", lambda: pd.Series([1, 2, 3]))

    assert benchmark.operation == "series"
    assert benchmark.duration_ms == pytest.approx(500.0)
    assert benchmark.throughput_ops_per_sec == pytest.approx(6.0)
    assert benchmark.cpu_cores_used == 6


def test_global_helpers_return_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(perf, "_global_processor", None)
    monkeypatch.setattr(perf, "_global_cache", None)

    assert perf.get_parallel_processor() is perf.get_parallel_processor()
    assert perf.get_performance_cache() is perf.get_performance_cache()
