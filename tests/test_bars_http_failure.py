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

    monkeypatch.setenv("ENABLE_HTTP_FALLBACK", "force")

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

