import pytest
from datetime import UTC, datetime, timedelta

pd = pytest.importorskip("pandas")

from ai_trading.data import daily_cache
from ai_trading.data.fetch import DataFetchError


def _stub_daily_df():
    index = pd.date_range(start="2024-01-01", periods=2, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [100, 100],
        },
        index=index,
    )


def test_cache_used_on_network_failure(monkeypatch):
    daily_cache._CACHE.clear()
    daily_cache._CACHE_TS.clear()
    df = _stub_daily_df()
    calls = {"count": 0}

    def fake_fetch(symbol, start, end, feed=None, adjustment=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return df
        raise DataFetchError("boom")

    monkeypatch.setattr(daily_cache, "_fetch_daily_df", fake_fetch)

    assert daily_cache.get_daily_df("AAPL") is df

    with pytest.warns(UserWarning):
        out = daily_cache.get_daily_df("AAPL")

    assert out is df
    assert calls["count"] == 2


def test_cache_not_used_when_schema_invalid(monkeypatch):
    daily_cache._CACHE.clear()
    daily_cache._CACHE_TS.clear()
    daily_cache._CACHE[("AAPL", None, None, None, None)] = pd.DataFrame({"close": [1.0]})
    daily_cache._CACHE_TS[("AAPL", None, None, None, None)] = datetime.now(UTC)

    monkeypatch.setattr(
        daily_cache,
        "_fetch_daily_df",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DataFetchError("boom")),
    )

    with pytest.raises(DataFetchError):
        daily_cache.get_daily_df("AAPL")


def test_cache_not_used_when_stale(monkeypatch):
    daily_cache._CACHE.clear()
    daily_cache._CACHE_TS.clear()
    key = ("AAPL", None, None, None, None)
    daily_cache._CACHE[key] = _stub_daily_df()
    daily_cache._CACHE_TS[key] = datetime.now(UTC) - timedelta(days=2)

    monkeypatch.setattr(
        daily_cache,
        "_fetch_daily_df",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DataFetchError("boom")),
    )

    with pytest.raises(DataFetchError):
        daily_cache.get_daily_df("AAPL")


def test_unexpected_cache_fetch_error_is_not_swallowed(monkeypatch):
    daily_cache._CACHE.clear()
    daily_cache._CACHE_TS.clear()
    daily_cache._CACHE[("AAPL", None, None, None, None)] = _stub_daily_df()
    daily_cache._CACHE_TS[("AAPL", None, None, None, None)] = datetime.now(UTC)

    monkeypatch.setattr(
        daily_cache,
        "_fetch_daily_df",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("bug")),
    )

    with pytest.raises(AssertionError):
        daily_cache.get_daily_df("AAPL")
