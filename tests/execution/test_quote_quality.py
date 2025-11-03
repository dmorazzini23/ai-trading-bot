from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from tests.test_orders import StubExecutionEngine


def test_skip_trade_when_quote_degraded(monkeypatch, caplog):
    engine = StubExecutionEngine()

    # Prevent the execution path from attempting to submit to Alpaca.
    monkeypatch.setattr(
        "ai_trading.execution.live_trading._require_bid_ask_quotes",
        lambda: False,
    )
    monkeypatch.setattr(
        "ai_trading.execution.live_trading.get_trading_config",
        lambda: SimpleNamespace(
            nbbo_required_for_limit=False,
            execution_require_realtime_nbbo=False,
            execution_market_on_degraded=False,
            degraded_feed_mode="block",
            degraded_feed_limit_widen_bps=0,
        ),
    )

    caplog.set_level(logging.WARNING)

    result = engine.execute_order(
        "AAPL",
        "buy",
        qty=5,
        order_type="limit",
        annotations={"using_fallback_price": True},
    )

    assert result is None
    record = next(rec for rec in caplog.records if rec.msg == "QUOTE_QUALITY_BLOCKED")
    assert record.symbol == "AAPL"
