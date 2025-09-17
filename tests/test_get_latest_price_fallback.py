"""Tests for :func:`ai_trading.core.bot_engine.get_latest_price` fallbacks."""

from __future__ import annotations

import pandas as pd

import ai_trading.alpaca_api as alpaca_api
from ai_trading.alpaca_api import AlpacaAuthenticationError, is_alpaca_service_available
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


def test_get_latest_price_handles_auth_failure(monkeypatch):
    monkeypatch.setattr(alpaca_api, "_ALPACA_SERVICE_AVAILABLE", True)

    def raise_auth(*_a, **_k):
        monkeypatch.setattr(alpaca_api, "_ALPACA_SERVICE_AVAILABLE", False)
        raise AlpacaAuthenticationError("Unauthorized")

    def fail_backup(*_a, **_k):  # pragma: no cover - defensive guard
        raise AssertionError("Backup provider should not be queried on auth failure")

    monkeypatch.setattr(bot_engine, "_alpaca_symbols", lambda: (raise_auth, None))
    monkeypatch.setattr(data_fetcher, "_backup_get_bars", fail_backup)
    monkeypatch.setattr(bot_engine, "get_bars_df", fail_backup)

    price = bot_engine.get_latest_price("AAPL")

    assert price is None
    assert bot_engine._PRICE_SOURCE["AAPL"] == "alpaca_auth_failed"
    assert not is_alpaca_service_available()

