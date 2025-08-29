import threading
import time

import ai_trading.core.bot_engine as be
import pytest
pd = pytest.importorskip("pandas")
def _mk_df():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "open": [1,2,3], "high":[1,2,3], "low":[1,2,3], "close":[1,2,3], "volume":[100,200,300]
        }
    )

def test_daily_fallback_parallel(monkeypatch):
    # Force batch to return empty so we hit fallback path for all.
    calls = {"single": []}
    monkeypatch.setattr(be, "get_bars_batch", lambda *a, **k: {})
    def fake_single(sym, *a, **k):
        calls["single"].append(sym)
        return _mk_df()
    monkeypatch.setattr(be, "get_bars", fake_single)
    out = be._fetch_universe_bars(["A","B","C","D"], "1D", "2024-01-01", "2024-02-01", None)
    assert set(out.keys()) == {"A","B","C","D"}
    # We can't assert true parallelism, but we can ensure all fallbacks executed.
    assert set(calls["single"]) == {"A","B","C","D"}

def test_intraday_fallback_parallel(monkeypatch):
    calls = {"single": []}
    monkeypatch.setattr(be, "get_bars_batch", lambda *a, **k: {})
    def fake_single(sym, *a, **k):
        calls["single"].append(sym)
        return _mk_df()
    monkeypatch.setattr(be, "get_minute_df", fake_single)
    out = be._fetch_intraday_bars_chunked(["X","Y","Z"], "2024-01-01 09:30", "2024-01-01 10:30", None)
    assert set(out.keys()) == {"X","Y","Z"}
    assert set(calls["single"]) == {"X","Y","Z"}

def test_parallel_execution_timing(monkeypatch):
    """Test that parallel execution provides performance benefit."""
    call_times = []

    monkeypatch.setattr(be, "get_bars_batch", lambda *a, **k: {})

    def slow_single(sym, *a, **k):
        # Simulate slow API call
        time.sleep(0.1)
        call_times.append((sym, time.time()))
        return _mk_df()

    monkeypatch.setattr(be, "get_bars", slow_single)

    # Test with 4 symbols that should run in parallel
    start_time = time.time()
    out = be._fetch_universe_bars(["A","B","C","D"], "1D", "2024-01-01", "2024-02-01", None)
    end_time = time.time()

    # Should complete faster than sequential (0.4s) due to parallelism
    # Allow some overhead but should be significantly faster than sequential
    assert end_time - start_time < 0.3, f"Parallel execution took {end_time - start_time:.2f}s, expected < 0.3s"
    assert len(out) == 4

    # Verify calls happened in a parallel timeframe (not perfectly sequential)
    call_times.sort(key=lambda x: x[1])
    first_call = call_times[0][1]
    last_call = call_times[-1][1]
    time_span = last_call - first_call

    # Should be less than sequential time but account for overlap
    assert time_span < 0.25, f"Call time span {time_span:.2f}s suggests sequential execution"

def test_bounded_concurrency_respects_limit(monkeypatch):
    """Test that the worker limit is respected."""
    active_workers = []
    max_concurrent = 0

    # Mock settings to use only 2 workers
    def mock_get_settings():
        class Settings:
            batch_fallback_workers = 2

        return Settings()

    monkeypatch.setattr(be, "get_settings", mock_get_settings)
    monkeypatch.setattr(be, "get_bars_batch", lambda *a, **k: {})

    def track_concurrent(sym, *a, **k):
        thread_id = threading.current_thread().ident
        active_workers.append(thread_id)

        nonlocal max_concurrent
        current_count = len(set(active_workers))
        max_concurrent = max(max_concurrent, current_count)

        time.sleep(0.1)  # Simulate work
        return _mk_df()

    monkeypatch.setattr(be, "get_bars", track_concurrent)

    # Test with 6 symbols but limit to 2 workers
    out = be._fetch_universe_bars(["A","B","C","D","E","F"], "1D", "2024-01-01", "2024-02-01", None)

    assert len(out) == 6
    # Should never exceed our worker limit
    assert max_concurrent <= 2, f"Max concurrent workers {max_concurrent} exceeded limit of 2"


def test_intraday_bounded_concurrency_respects_limit(monkeypatch):
    """Intraday fallback should honor worker limits and return all results."""

    active_workers: list[int] = []
    max_concurrent = 0

    def mock_get_settings():
        class Settings:
            batch_fallback_workers = 2

        return Settings()

    monkeypatch.setattr(be, "get_settings", mock_get_settings)
    monkeypatch.setattr(be, "get_bars_batch", lambda *a, **k: {})

    def track_concurrent(sym, *a, **k):
        thread_id = threading.current_thread().ident
        active_workers.append(thread_id)

        nonlocal max_concurrent
        current_count = len(set(active_workers))
        max_concurrent = max(max_concurrent, current_count)

        time.sleep(0.1)
        return _mk_df()

    monkeypatch.setattr(be, "get_minute_df", track_concurrent)

    out = be._fetch_intraday_bars_chunked(
        ["A", "B", "C", "D", "E", "F"],
        "2024-01-01 09:30",
        "2024-01-01 10:30",
        None,
    )

    assert len(out) == 6
    assert max_concurrent <= 2, f"Max concurrent workers {max_concurrent} exceeded limit of 2"
