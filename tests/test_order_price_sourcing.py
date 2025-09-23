from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from ai_trading.core import bot_engine


class DummyDataClient:
    def __init__(self, quotes: dict[str | None, Any]) -> None:
        self.quotes = quotes
        self.calls: list[str | None] = []

    def get_stock_latest_quote(self, req: Any) -> Any:
        feed = getattr(req, "feed", None)
        self.calls.append(feed)
        return self.quotes.get(feed)


def _ctx_with_quotes(quotes: dict[str | None, Any]) -> SimpleNamespace:
    return SimpleNamespace(data_client=DummyDataClient(quotes))


def test_price_source_prefers_nbbo(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("SLIPPAGE_BPS", "2")
    quotes = {
        None: SimpleNamespace(bid_price=100.0, ask_price=100.5),
    }
    ctx = _ctx_with_quotes(quotes)
    price, source = bot_engine._resolve_limit_price(ctx, "AAPL", "buy", None, None)
    assert source == "broker_nbbo"
    assert price == pytest.approx(100.27005, rel=1e-6)
    assert any("fallback_chain=[\"primary_mid\",\"backup_mid\",\"last_close\"]" in rec.message for rec in caplog.records)


def test_price_source_primary_mid_when_nbbo_missing(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("SLIPPAGE_BPS", "2")
    monkeypatch.setattr(bot_engine, "DATA_FEED_INTRADAY", "iex", raising=False)
    quotes = {
        None: SimpleNamespace(bid_price=0.0, ask_price=0.0),
        "iex": SimpleNamespace(bid_price=104.0, ask_price=105.0),
    }
    ctx = _ctx_with_quotes(quotes)
    price, source = bot_engine._resolve_limit_price(ctx, "MSFT", "sell", None, None)
    assert source == "primary_mid"
    assert price == pytest.approx(104.4791, rel=1e-6)
    assert any("fallback_chain=[\"backup_mid\",\"last_close\"]" in rec.message for rec in caplog.records)


def test_price_source_backup_mid(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("SLIPPAGE_BPS", "2")
    monkeypatch.setattr(bot_engine, "DATA_FEED_INTRADAY", "iex", raising=False)
    monkeypatch.setattr(bot_engine.data_fetcher_module, "_sip_configured", lambda: True, raising=False)
    quotes = {
        None: SimpleNamespace(bid_price=0.0, ask_price=0.0),
        "iex": SimpleNamespace(bid_price=0.0, ask_price=0.0),
        "sip": SimpleNamespace(bid_price=200.0, ask_price=201.0),
    }
    ctx = _ctx_with_quotes(quotes)
    price, source = bot_engine._resolve_limit_price(ctx, "TSLA", "buy", None, None)
    assert source == "backup_mid"
    assert price == pytest.approx(200.0402, rel=1e-6)
    assert any("fallback_chain=[\"last_close\"]" in rec.message for rec in caplog.records)


def test_price_source_last_close_fallback(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("SLIPPAGE_BPS", "2")
    monkeypatch.setattr(bot_engine, "DATA_FEED_INTRADAY", "iex", raising=False)
    monkeypatch.setattr(bot_engine.data_fetcher_module, "_sip_configured", lambda: False, raising=False)
    quotes = {
        None: None,
        "iex": None,
    }
    ctx = _ctx_with_quotes(quotes)
    minute_df = pd.DataFrame()
    price, source = bot_engine._resolve_limit_price(ctx, "NFLX", "buy", minute_df, 50.0)
    assert source == "last_close"
    assert price == pytest.approx(50.01, rel=1e-6)
    assert any("fallback_chain=[]" in rec.message for rec in caplog.records)
