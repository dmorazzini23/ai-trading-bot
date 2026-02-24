import types
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.core import bot_engine as bot
from ai_trading.data import bars
from ai_trading.data.bars import StockBarsRequest, TimeFrame


def test_screen_universe_apierror_skips_symbol(monkeypatch):
    monkeypatch.setattr(bot, "_SCREEN_CACHE", {})
    monkeypatch.setattr(bot, "_LAST_SCREEN_FETCH", {})
    monkeypatch.setattr(bot.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(bot, "is_market_open", lambda: True)
    monkeypatch.setattr(bot, "is_valid_ohlcv", lambda df: df is not None and not df.empty)
    monkeypatch.setattr(bot, "_validate_market_data_quality", lambda df, s: {"valid": True})

    rows = bot.ATR_LENGTH + 5
    idx = pd.date_range("2024-01-01", periods=rows, tz="UTC")
    base = pd.DataFrame(
        {
            "open": [1.0] * rows,
            "high": [1.0] * rows,
            "low": [1.0] * rows,
            "close": [1.0] * rows,
            "volume": [200_000] * rows,
            "trade_count": [10] * rows,
            "vwap": [1.0] * rows,
        },
        index=idx,
    )

    def fake_client_fetch_stock_bars(client, request):
        sym = request.symbol_or_symbols[0]
        if sym == "BAD":
            raise bars.APIError("boom")
        return types.SimpleNamespace(df=base)

    monkeypatch.setattr(bars, "_client_fetch_stock_bars", fake_client_fetch_stock_bars)

    class DummyFetcher:
        def get_daily_df(self, runtime, sym):
            req = StockBarsRequest(
                symbol_or_symbols=[sym],
                timeframe=TimeFrame.Day,
                start=idx[0],
                end=idx[-1],
                feed="iex",
            )
            return bars.safe_get_stock_bars(object(), req, sym, "DAILY")

    runtime = types.SimpleNamespace(data_fetcher=DummyFetcher())

    monkeypatch.setattr(
        bot,
        "ta",
        types.SimpleNamespace(
            atr=lambda h, l, c, length=bot.ATR_LENGTH: pd.Series([1.0] * len(h))
        ),
    )

    result = bot.screen_universe(["AAA", "BAD", "CCC"], runtime)

    assert set(result) == {"AAA", "CCC"}
    assert "BAD" not in result
