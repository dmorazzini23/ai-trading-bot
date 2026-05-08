from __future__ import annotations

import types

from ai_trading.core import bot_engine


def test_resolve_order_quote_basis_prefers_nbbo_for_buy(monkeypatch) -> None:
    monkeypatch.setattr(
        bot_engine,
        "_fetch_quote",
        lambda *_a, **_k: types.SimpleNamespace(bid_price=100.0, ask_price=100.2),
    )
    monkeypatch.setattr(bot_engine, "get_price_source", lambda _symbol: "last_trade")

    source, bid, ask, mid, arrival, quote_ts = bot_engine._resolve_order_quote_basis(  # type: ignore[attr-defined]
        types.SimpleNamespace(),
        symbol="AAPL",
        side="buy",
        fallback_price=99.8,
    )

    assert source == "broker_nbbo"
    assert bid == 100.0
    assert ask == 100.2
    assert mid == 100.1
    assert arrival == 100.2
    assert quote_ts is None


def test_resolve_order_quote_basis_prefers_nbbo_for_sell(monkeypatch) -> None:
    monkeypatch.setattr(
        bot_engine,
        "_fetch_quote",
        lambda *_a, **_k: types.SimpleNamespace(bid_price=55.1, ask_price=55.3),
    )
    monkeypatch.setattr(bot_engine, "get_price_source", lambda _symbol: "alpaca_last")

    source, bid, ask, mid, arrival, quote_ts = bot_engine._resolve_order_quote_basis(  # type: ignore[attr-defined]
        types.SimpleNamespace(),
        symbol="MSFT",
        side="sell",
        fallback_price=55.0,
    )

    assert source == "broker_nbbo"
    assert bid == 55.1
    assert ask == 55.3
    assert mid == 55.2
    assert arrival == 55.1
    assert quote_ts is None


def test_resolve_order_quote_basis_accepts_dict_bp_ap(monkeypatch) -> None:
    monkeypatch.setattr(
        bot_engine,
        "_fetch_quote",
        lambda *_a, **_k: {"bp": "10.00", "ap": "10.10", "t": "2026-04-24T15:00:00Z"},
    )
    monkeypatch.setattr(bot_engine, "get_price_source", lambda _symbol: "alpaca_last")

    source, bid, ask, mid, arrival, quote_ts = bot_engine._resolve_order_quote_basis(  # type: ignore[attr-defined]
        types.SimpleNamespace(),
        symbol="AAPL",
        side="buy",
        fallback_price=9.9,
    )

    assert source == "broker_nbbo"
    assert bid == 10.0
    assert ask == 10.1
    assert mid == 10.05
    assert arrival == 10.1
    assert quote_ts is not None


def test_resolve_order_quote_basis_falls_back_to_price_source(monkeypatch) -> None:
    monkeypatch.setattr(bot_engine, "_fetch_quote", lambda *_a, **_k: None)
    monkeypatch.setattr(bot_engine, "get_price_source", lambda _symbol: "last_trade")

    source, bid, ask, mid, arrival, quote_ts = bot_engine._resolve_order_quote_basis(  # type: ignore[attr-defined]
        types.SimpleNamespace(),
        symbol="TSLA",
        side="buy",
        fallback_price=222.5,
    )

    assert source == "last_trade"
    assert bid is None
    assert ask is None
    assert mid is None
    assert arrival == 222.5
    assert quote_ts is None
