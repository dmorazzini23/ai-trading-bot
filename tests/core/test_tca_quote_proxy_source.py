from __future__ import annotations

from types import SimpleNamespace

from ai_trading.core import bot_engine


def test_resolve_quote_proxy_source_prefers_order_fields(monkeypatch) -> None:
    order = SimpleNamespace(order=SimpleNamespace(price_source="alpaca_ask"))
    monkeypatch.setattr(bot_engine, "get_price_source", lambda _symbol: "last_trade")

    resolved = bot_engine._resolve_quote_proxy_source(  # type: ignore[attr-defined]
        order,
        symbol="AAPL",
        default_source="last_trade",
    )

    assert resolved == "alpaca_ask"


def test_resolve_quote_proxy_source_uses_runtime_symbol_source(monkeypatch) -> None:
    monkeypatch.setattr(bot_engine, "get_price_source", lambda _symbol: "alpaca_bid")

    resolved = bot_engine._resolve_quote_proxy_source(  # type: ignore[attr-defined]
        order=SimpleNamespace(),
        symbol="AAPL",
        default_source="last_trade",
    )

    assert resolved == "alpaca_bid"


def test_resolve_quote_proxy_source_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setattr(bot_engine, "get_price_source", lambda _symbol: "unknown")

    resolved = bot_engine._resolve_quote_proxy_source(  # type: ignore[attr-defined]
        order=SimpleNamespace(),
        symbol="AAPL",
        default_source="last_trade",
    )

    assert resolved == "last_trade"
