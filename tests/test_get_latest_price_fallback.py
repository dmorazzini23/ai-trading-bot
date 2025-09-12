"""Tests for :func:`ai_trading.core.bot_engine.get_latest_price` fallbacks."""

from __future__ import annotations

import pandas as pd

from ai_trading.core import bot_engine
import ai_trading.data.fetch as data_fetcher


def _df(price: float) -> pd.DataFrame:
    return pd.DataFrame({"close": [price]})


def test_get_latest_price_uses_yahoo_when_alpaca_none(monkeypatch):
    """Alpaca returning ``None`` should trigger Yahoo fallback."""

    monkeypatch.setattr(
        bot_engine,
        "_alpaca_symbols",
        lambda: (lambda *_a, **_k: {"ap": None}, None),
    )

    called: dict[str, bool] = {"yahoo": False}

    def fake_yahoo(symbol, start, end, interval):  # noqa: ARG001
        called["yahoo"] = True
        return _df(101.0)

    monkeypatch.setattr(data_fetcher, "_backup_get_bars", fake_yahoo)
    monkeypatch.setattr(bot_engine, "get_latest_close", lambda df: float(df["close"].iloc[-1]))

    price = bot_engine.get_latest_price("AAPL")

    assert called["yahoo"]
    assert price == 101.0


def test_get_latest_price_uses_latest_close_when_providers_fail(monkeypatch):
    """If Alpaca and Yahoo fail, fall back to ``get_latest_close`` from bars."""

    monkeypatch.setattr(
        bot_engine,
        "_alpaca_symbols",
        lambda: (lambda *_a, **_k: {"ap": None}, None),
    )
    monkeypatch.setattr(data_fetcher, "_backup_get_bars", lambda *a, **k: (_ for _ in ()).throw(RuntimeError))

    monkeypatch.setattr(bot_engine, "get_bars_df", lambda symbol: _df(55.0))
    monkeypatch.setattr(bot_engine, "get_latest_close", lambda df: float(df["close"].iloc[-1]) if not df.empty else None)

    price = bot_engine.get_latest_price("AAPL")

    assert price == 55.0

