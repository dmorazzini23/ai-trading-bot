from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.core import bot_engine
from ai_trading.oms.pretrade import OrderIntent, SlidingWindowRateLimiter, validate_pretrade


def test_pretrade_uses_non_prefixed_env_limits(monkeypatch) -> None:
    monkeypatch.setenv("MAX_ORDER_DOLLARS", "1000")
    monkeypatch.setenv("MAX_ORDER_SHARES", "5")
    monkeypatch.setenv("PRICE_COLLAR_PCT", "0.05")
    monkeypatch.delenv("AI_TRADING_MAX_ORDER_DOLLARS", raising=False)
    monkeypatch.delenv("AI_TRADING_MAX_ORDER_SHARES", raising=False)
    monkeypatch.delenv("AI_TRADING_PRICE_COLLAR_PCT", raising=False)

    cfg = SimpleNamespace(max_order_dollars=None, max_order_shares=None, price_collar_pct=None)
    intent = OrderIntent(
        symbol="AAPL",
        side="buy",
        qty=10,
        notional=2500.0,
        limit_price=120.0,
        bar_ts=datetime.now(UTC),
        client_order_id="cid-env-limits",
        last_price=100.0,
        mid=100.0,
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)
    allowed, reason, _details = validate_pretrade(intent, cfg=cfg, ledger=None, rate_limiter=limiter)
    assert allowed is False
    assert reason == "ORDER_SIZE_BLOCK"


def test_pretrade_rate_limiter_uses_non_prefixed_env_keys(monkeypatch) -> None:
    monkeypatch.setenv("MAX_ORDERS_PER_MINUTE_GLOBAL", "21")
    monkeypatch.setenv("MAX_ORDERS_PER_MINUTE_PER_SYMBOL", "4")
    monkeypatch.setenv("MAX_CANCELS_PER_MINUTE_GLOBAL", "19")
    monkeypatch.delenv("AI_TRADING_ORDERS_PER_MIN_GLOBAL", raising=False)
    monkeypatch.delenv("AI_TRADING_ORDERS_PER_MIN_SYMBOL", raising=False)
    monkeypatch.delenv("AI_TRADING_CANCELS_PER_MIN", raising=False)

    state = bot_engine.BotState()
    limiter = bot_engine._pretrade_rate_limiter(state)
    assert limiter.global_orders_per_min == 21
    assert limiter.per_symbol_orders_per_min == 4
    assert limiter.cancels_per_min == 19
