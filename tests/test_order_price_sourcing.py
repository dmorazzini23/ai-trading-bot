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
        self.requests: list[Any] = []

    def get_stock_latest_quote(self, req: Any) -> Any:
        feed = getattr(req, "feed", None)
        self.calls.append(feed)
        self.requests.append(req)
        return self.quotes.get(feed)


def _ctx_with_quotes(quotes: dict[str | None, Any]) -> SimpleNamespace:
    return SimpleNamespace(data_client=DummyDataClient(quotes))


@pytest.fixture(autouse=True)
def stub_stock_quote_request(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, str | None]]:
    calls: list[tuple[Any, str | None]] = []

    class StubStockLatestQuoteRequest(SimpleNamespace):
        def __init__(
            self,
            *,
            symbol_or_symbols: Any,
            feed: str | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                symbol_or_symbols=symbol_or_symbols,
                feed=feed,
                params=dict(kwargs),
            )
            symbols: Any
            if isinstance(symbol_or_symbols, (list, tuple)):
                symbols = tuple(symbol_or_symbols)
            else:
                symbols = symbol_or_symbols
            calls.append((symbols, feed))

    def _unexpected_ensure_call() -> None:
        raise AssertionError("_ensure_alpaca_classes should not run when stubbed")

    monkeypatch.setattr(bot_engine, "StockLatestQuoteRequest", StubStockLatestQuoteRequest, raising=False)
    monkeypatch.setattr(bot_engine, "_ensure_alpaca_classes", _unexpected_ensure_call)
    monkeypatch.setattr(bot_engine, "_ALPACA_IMPORT_ERROR", None, raising=False)
    return calls


def test_price_source_prefers_nbbo(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("PRICE_SLIPPAGE_BPS", "2")
    quotes = {
        None: SimpleNamespace(bid_price=100.0, ask_price=100.5),
    }
    ctx = _ctx_with_quotes(quotes)
    price, source = bot_engine._resolve_limit_price(ctx, "AAPL", "buy", None, None)
    assert source == "broker_nbbo"
    assert price == pytest.approx(100.2708, rel=1e-6)
    messages = [rec.message for rec in caplog.records if rec.message.startswith("ORDER_PRICE_SOURCE")]
    assert messages and "reason=ok" in messages[-1]


def test_price_source_primary_mid_when_nbbo_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("PRICE_SLIPPAGE_BPS", "2")
    monkeypatch.setattr(bot_engine, "DATA_FEED_INTRADAY", "iex", raising=False)
    quotes = {
        None: SimpleNamespace(bid_price=0.0, ask_price=0.0),
        "iex": SimpleNamespace(bid_price=104.0, ask_price=105.0),
    }
    ctx = _ctx_with_quotes(quotes)
    price, source = bot_engine._resolve_limit_price(ctx, "MSFT", "sell", None, None)
    assert source == "primary_mid"
    assert price == pytest.approx(104.4783, rel=1e-6)
    messages = [rec.message for rec in caplog.records if rec.message.startswith("ORDER_PRICE_SOURCE")]
    assert messages and "reason=broker_nbbo:no_bid_ask" in messages[-1]


def test_price_source_backup_mid(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("PRICE_SLIPPAGE_BPS", "2")
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
    assert price == pytest.approx(200.5409, rel=1e-6)
    messages = [rec.message for rec in caplog.records if rec.message.startswith("ORDER_PRICE_SOURCE")]
    assert messages and "primary_mid:iex:no_bid_ask" in messages[-1]


def test_price_source_last_close_fallback(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("PRICE_SLIPPAGE_BPS", "2")
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
    assert price == pytest.approx(50.0108, rel=1e-6)
    messages = [rec.message for rec in caplog.records if rec.message.startswith("ORDER_PRICE_SOURCE")]
    assert messages and "reason=broker_nbbo:unavailable;primary_mid:iex:unavailable" in messages[-1]


def test_price_source_prefers_nbbo_with_stubbed_request(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stub_stock_quote_request: list[tuple[Any, str | None]],
) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("PRICE_SLIPPAGE_BPS", "2")
    quotes = {
        None: SimpleNamespace(bid_price=102.0, ask_price=102.5),
    }
    ctx = _ctx_with_quotes(quotes)
    price, source = bot_engine._resolve_limit_price(ctx, "NVDA", "buy", None, None)
    assert source == "broker_nbbo"
    assert price == pytest.approx(102.2712, rel=1e-6)
    assert stub_stock_quote_request == [(("NVDA",), None)]


def test_price_source_primary_mid_with_stubbed_request(
    monkeypatch: pytest.MonkeyPatch,
    stub_stock_quote_request: list[tuple[Any, str | None]],
) -> None:
    monkeypatch.setenv("PRICE_SLIPPAGE_BPS", "2")
    monkeypatch.setattr(bot_engine, "DATA_FEED_INTRADAY", "iex", raising=False)
    quotes = {
        None: SimpleNamespace(bid_price=0.0, ask_price=0.0),
        "iex": SimpleNamespace(bid_price=108.0, ask_price=109.0),
    }
    ctx = _ctx_with_quotes(quotes)
    price, source = bot_engine._resolve_limit_price(ctx, "META", "sell", None, None)
    assert source == "primary_mid"
    assert price == pytest.approx(108.4775, rel=1e-6)
    assert stub_stock_quote_request == [(("META",), None), (("META",), "iex")]


def test_price_source_backup_mid_with_stubbed_request(
    monkeypatch: pytest.MonkeyPatch,
    stub_stock_quote_request: list[tuple[Any, str | None]],
) -> None:
    monkeypatch.setenv("PRICE_SLIPPAGE_BPS", "2")
    monkeypatch.setattr(bot_engine, "DATA_FEED_INTRADAY", "iex", raising=False)
    monkeypatch.setattr(bot_engine.data_fetcher_module, "_sip_configured", lambda: True, raising=False)
    quotes = {
        None: SimpleNamespace(bid_price=0.0, ask_price=0.0),
        "iex": SimpleNamespace(bid_price=0.0, ask_price=0.0),
        "sip": SimpleNamespace(bid_price=210.0, ask_price=211.0),
    }
    ctx = _ctx_with_quotes(quotes)
    price, source = bot_engine._resolve_limit_price(ctx, "AMZN", "buy", None, None)
    assert source == "backup_mid"
    assert price == pytest.approx(210.5429, rel=1e-6)
    assert stub_stock_quote_request == [
        (("AMZN",), None),
        (("AMZN",), "iex"),
        (("AMZN",), "sip"),
    ]
