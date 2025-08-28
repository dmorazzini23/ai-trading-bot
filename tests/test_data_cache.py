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
