import time

import pytest

pd = pytest.importorskip("pandas")
from ai_trading.market import cache as mcache


def test_mem_cache_ttl_basic(tmp_path):
    df = pd.DataFrame({"timestamp":[1], "open":[1], "high":[1], "low":[1], "close":[1], "volume":[1]})
    mcache.put_mem("AAPL", "1D", "2024-01-01", "2024-01-31", df)
    got = mcache.get_mem("AAPL","1D","2024-01-01","2024-01-31", ttl=60)
    assert got is not None and list(got.columns)==list(df.columns)
    got2 = mcache.get_mem("AAPL","1D","2024-01-01","2024-01-31", ttl=0)
    assert got2 is None

def test_disk_cache_basic(tmp_path):
    """Test disk cache functionality"""
    cache_dir = str(tmp_path / "cache")
    df = pd.DataFrame({"timestamp":[1], "open":[2], "high":[3], "low":[1], "close":[2.5], "volume":[1000]})

    # Put data in disk cache
    mcache.put_disk(cache_dir, "TSLA", "1h", "2024-01-01", "2024-01-02", df)

    # Retrieve from disk cache
    retrieved = mcache.get_disk(cache_dir, "TSLA", "1h", "2024-01-01", "2024-01-02")
    assert retrieved is not None
    assert list(retrieved.columns) == list(df.columns)
    assert len(retrieved) == len(df)

    p = mcache.disk_path(cache_dir, "TSLA", "1h", "2024-01-01", "2024-01-02")
    csv_p = p.with_suffix(".csv")
    assert p.exists() or csv_p.exists()


def test_disk_cache_parquet_import_error_fallback(tmp_path, monkeypatch):
    """If the parquet engine is missing, the cache should fall back to CSV."""
    cache_dir = str(tmp_path / "cache")
    df = pd.DataFrame({"timestamp": [1], "open": [2]})

    def raise_import_error(*_, **__):
        raise ImportError("no parquet")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", raise_import_error)
    monkeypatch.setattr(pd, "read_parquet", raise_import_error)

    mcache.put_disk(cache_dir, "MSFT", "1h", "2024-01-01", "2024-01-02", df)
    retrieved = mcache.get_disk(cache_dir, "MSFT", "1h", "2024-01-01", "2024-01-02")
    assert retrieved is not None

    p = mcache.disk_path(cache_dir, "MSFT", "1h", "2024-01-01", "2024-01-02")
    assert not p.exists()
    assert p.with_suffix(".csv").exists()

def test_cache_key_generation():
    """Test cache key generation with special characters"""
    key1 = mcache._key("SPY", "1D", "2024-01-01", "2024-01-31")
    key2 = mcache._key("BTC/USD", "1h", "2024:01:01T10:30:00", "2024:01:01T11:30:00")

    assert key1 == "SPY|1D|2024-01-01|2024-01-31"
    assert key2 == "BTC/USD|1h|2024:01:01T10:30:00|2024:01:01T11:30:00"

def test_memory_cache_thread_safety():
    """Test that memory cache handles concurrent access safely"""
    import threading

    df = pd.DataFrame({"close": [100, 101, 102]})
    results = []

    def worker(symbol_suffix):
        # Put and get data concurrently
        symbol = f"TEST{symbol_suffix}"
        mcache.put_mem(symbol, "1M", "2024-01-01", "2024-01-01", df)
        time.sleep(0.01)  # Small delay to encourage race conditions
        retrieved = mcache.get_mem(symbol, "1M", "2024-01-01", "2024-01-01", ttl=60)
        results.append(retrieved is not None)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All operations should succeed
    assert all(results), "Some cache operations failed under concurrent access"

def test_settings_integration():
    """Test that cache settings can be loaded and work as expected"""
    from ai_trading.config.settings import get_settings

    settings = get_settings()

    # Test default values
    assert isinstance(settings.data_cache_enable, bool)
    assert isinstance(settings.data_cache_ttl_seconds, int)
    assert isinstance(settings.data_cache_dir, str)
    assert isinstance(settings.data_cache_disk_enable, bool)

    # Test that defaults are reasonable
    assert settings.data_cache_ttl_seconds > 0
    assert len(settings.data_cache_dir) > 0


def test_get_or_load_caches_memory_and_disk(tmp_path):
    with mcache._lock:
        mcache._mem.clear()

    cache_dir = tmp_path / "cache"
    df = pd.DataFrame({"close": [1.0], "volume": [10]})

    calls: list[int] = []

    def loader():
        calls.append(1)
        return df

    result1 = mcache.get_or_load(
        key=("AAPL", "1Min", "2024-01-01T00:00:00", "2024-01-01T01:00:00"),
        loader=loader,
        ttl=60,
        cache_dir=str(cache_dir),
        disk_enabled=True,
    )
    result2 = mcache.get_or_load(
        key=("AAPL", "1Min", "2024-01-01T00:00:00", "2024-01-01T01:00:00"),
        loader=loader,
        ttl=60,
        cache_dir=str(cache_dir),
        disk_enabled=True,
    )

    assert calls == [1]
    pd.testing.assert_frame_equal(result1, df)
    pd.testing.assert_frame_equal(result2, df)

    disk_path = mcache.disk_path(
        str(cache_dir), "AAPL", "1Min", "2024-01-01T00:00:00", "2024-01-01T01:00:00"
    )
    assert disk_path.exists() or disk_path.with_suffix(".csv").exists()


def test_get_or_load_disk_hit_without_loader(tmp_path):
    with mcache._lock:
        mcache._mem.clear()

    cache_dir = tmp_path / "cache"
    df = pd.DataFrame({"close": [2.0], "volume": [20]})

    def loader():
        return df

    mcache.get_or_load(
        key=("TSLA", "1Min", "2024-01-02T00:00:00", "2024-01-02T01:00:00"),
        loader=loader,
        ttl=60,
        cache_dir=str(cache_dir),
        disk_enabled=True,
    )

    with mcache._lock:
        mcache._mem.clear()

    def fail_loader():
        raise AssertionError("loader should not run when disk cache hits")

    restored = mcache.get_or_load(
        key=("TSLA", "1Min", "2024-01-02T00:00:00", "2024-01-02T01:00:00"),
        loader=fail_loader,
        ttl=60,
        cache_dir=str(cache_dir),
        disk_enabled=True,
    )

    pd.testing.assert_frame_equal(restored, df)
