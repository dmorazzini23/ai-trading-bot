from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ai_trading.config.runtime import TradingConfig
from ai_trading.oms.pretrade import OrderIntent, SlidingWindowRateLimiter, safe_validate_pretrade
from ai_trading.runtime.paper_sampling import (
    evaluate_paper_sampling_order,
    reserve_paper_sampling_order,
)


def _cfg(**updates):
    values = {
        "paper_sampling_enabled": True,
        "paper_sampling_allowed_symbols": ("AAPL", "AMZN"),
        "paper_sampling_max_trades_per_day": 1,
        "paper_sampling_max_notional_per_order": 250.0,
        "execution_mode": "paper",
        "paper": True,
        "alpaca_base_url": "https://paper-api.alpaca.markets",
        "launch_profile": "paper_trade",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_config_rejects_paper_sampling_outside_paper_mode() -> None:
    with pytest.raises(ValueError, match="PAPER_SAMPLING_ENABLED requires EXECUTION_MODE=paper"):
        TradingConfig.from_env(
            {
                "APP_ENV": "prod",
                "EXECUTION_MODE": "live",
                "ALPACA_TRADING_BASE_URL": "https://api.alpaca.markets",
                "AI_TRADING_PAPER_SAMPLING_ENABLED": "1",
                "MAX_DRAWDOWN_THRESHOLD": "0.2",
            }
        )


def test_config_rejects_paper_sampling_for_live_canary_profile() -> None:
    with pytest.raises(ValueError, match="non-live launch profile"):
        TradingConfig.from_env(
            {
                "APP_ENV": "test",
                "EXECUTION_MODE": "paper",
                "ALPACA_TRADING_BASE_URL": "https://paper-api.alpaca.markets",
                "AI_TRADING_LAUNCH_PROFILE": "live_canary",
                "AI_TRADING_PAPER_SAMPLING_ENABLED": "1",
                "MAX_DRAWDOWN_THRESHOLD": "0.2",
            }
        )


def test_paper_sampling_symbol_short_size_and_daily_caps(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    cfg = _cfg()

    short_decision = evaluate_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="sell_short",
        qty=1,
        price=100.0,
    )
    assert short_decision.allowed is False
    assert short_decision.reason == "PAPER_SAMPLING_SHORT_BLOCK"

    symbol_decision = evaluate_paper_sampling_order(
        cfg,
        symbol="MSFT",
        side="buy",
        qty=1,
        price=100.0,
    )
    assert symbol_decision.allowed is False
    assert symbol_decision.reason == "PAPER_SAMPLING_SYMBOL_BLOCK"

    size_decision = evaluate_paper_sampling_order(
        cfg,
        symbol="AMZN",
        side="buy",
        qty=10,
        price=310.0,
    )
    assert size_decision.allowed is False
    assert size_decision.reason == "PAPER_SAMPLING_MAX_NOTIONAL_BLOCK"
    assert size_decision.qty == 0

    capped_decision = evaluate_paper_sampling_order(
        cfg,
        symbol="AMZN",
        side="buy",
        qty=10,
        price=100.0,
    )
    assert capped_decision.allowed is True
    assert capped_decision.qty == 2

    now = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)
    first = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        now=now,
    )
    second = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        now=now,
    )

    assert first.allowed is True
    assert second.allowed is False
    assert second.reason == "PAPER_SAMPLING_DAILY_CAP_BLOCK"


def test_paper_sampling_reduce_orders_do_not_consume_daily_cap(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    cfg = _cfg()
    now = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)

    first = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        now=now,
    )
    closing_sell = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="sell",
        qty=1,
        price=100.0,
        now=now,
        consumes_daily_slot=False,
    )
    next_entry = reserve_paper_sampling_order(
        cfg,
        symbol="AMZN",
        side="buy",
        qty=1,
        price=100.0,
        now=now,
    )

    assert first.allowed is True
    assert closing_sell.allowed is True
    assert closing_sell.details["consumes_daily_slot"] is False
    assert next_entry.allowed is False
    assert next_entry.reason == "PAPER_SAMPLING_DAILY_CAP_BLOCK"
    assert next_entry.details["count"] == 1


def test_paper_sampling_does_not_bypass_oms_order_size_block() -> None:
    cfg = _cfg(max_order_dollars=50.0)
    sampling = evaluate_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
    )
    assert sampling.allowed is True

    intent = OrderIntent(
        symbol="AAPL",
        side="buy",
        qty=sampling.qty,
        notional=100.0,
        limit_price=100.0,
        bar_ts=datetime(2026, 5, 8, 15, 0, tzinfo=UTC),
        client_order_id="paper-sampling-test",
        last_price=100.0,
        mid=100.0,
    )
    allowed, reason, _details = safe_validate_pretrade(
        intent,
        cfg=cfg,
        ledger=None,
        rate_limiter=SlidingWindowRateLimiter(),
    )

    assert allowed is False
    assert reason == "ORDER_SIZE_BLOCK"
