from datetime import UTC, datetime

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import bars as bars_mod
from ai_trading.data.bars import StockBarsRequest, TimeFrame


class _HTTPBoom(RuntimeError):
    status_code = 503


def test_http_get_bars_returns_sentinel(monkeypatch):
    """Wrapper should emit a sentinel when the underlying call raises."""

    def fail_get_bars(*_args, **_kwargs):
        raise _HTTPBoom("service unavailable")

    monkeypatch.setattr(bars_mod, "_raw_http_get_bars", fail_get_bars)

    start = datetime(2024, 1, 2, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)
    result = bars_mod.http_get_bars("SPY", "1Min", start, end, feed="iex")

    assert isinstance(result, bars_mod.BarsFetchFailed)
    assert result.symbol == "SPY"
    assert result.feed == "iex"
    assert result.status == 503


def test_safe_get_stock_bars_handles_sentinel(monkeypatch):
    """``safe_get_stock_bars`` should degrade to an empty frame on sentinel."""

    class DummyClient:
        pass

    req = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Day,
        start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 1, 3, tzinfo=UTC),
        feed="iex",
    )

    def fail_client_fetch(*_args, **_kwargs):
        raise RuntimeError("client failure")

    def fail_http_fetch(*_args, **_kwargs):
        raise _HTTPBoom("service unavailable")

    monkeypatch.setattr(bars_mod, "_client_fetch_stock_bars", fail_client_fetch)
    monkeypatch.setattr(bars_mod, "_raw_http_get_bars", fail_http_fetch)
    monkeypatch.setattr(bars_mod, "get_minute_df", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(bars_mod.time, "sleep", lambda *_: None)

    frame = bars_mod.safe_get_stock_bars(DummyClient(), req, symbol="SPY", context="TEST")

    assert isinstance(frame, pd.DataFrame)
    assert frame.empty


def test_safe_get_stock_bars_minute_path_uses_minute_fetch_without_error(monkeypatch, caplog):
    class DummyClient:
        pass

    req = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Minute,
        start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 1, 2, 0, 2, tzinfo=UTC),
        feed="iex",
    )

    timestamps = pd.date_range(start=req.start, periods=2, freq="1min", tz=UTC)
    minute_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100, 110],
        }
    )

    monkeypatch.setattr(
        bars_mod,
        "_client_fetch_stock_bars",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("client path should not run for 1Min")),
    )
    monkeypatch.setattr(bars_mod, "get_minute_df", lambda *_args, **_kwargs: minute_df.copy())

    with caplog.at_level("ERROR", logger=bars_mod._log.name):
        frame = bars_mod.safe_get_stock_bars(DummyClient(), req, symbol="SPY", context="TEST")

    assert isinstance(frame, pd.DataFrame)
    assert not frame.empty
    assert not any(record.message == "ALPACA_BARS_FETCH_FAILED" for record in caplog.records)
