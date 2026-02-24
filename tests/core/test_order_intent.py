import logging
import types

import pytest

from ai_trading.core import bot_engine
from ai_trading.core.bot_engine import BotState


class DummyExecutionEngine:
    def __init__(self, pending=None):
        self._pending = pending or []

    def get_pending_orders(self):
        return list(self._pending)


class DummyOrderInfo:
    def __init__(self, symbol: str, side: str):
        self.symbol = symbol
        self.side = side


def _make_context(**overrides):
    ctx = types.SimpleNamespace(
        allow_short_selling=True,
        position_map={},
        execution_engine=DummyExecutionEngine(),
        portfolio_weights={},
    )
    ctx.__dict__.update(overrides)
    return ctx


def test_short_signal_intent_allows_sell_short():
    ctx = _make_context()
    state = BotState()

    decision = bot_engine._resolve_order_intent(  # type: ignore[attr-defined]
        ctx,
        state,
        symbol="AAPL",
        signal_side="sell_short",
        target_weight=0.1,
        intended_side="sell_short",
    )

    assert decision.allowed
    assert decision.order_side == "sell_short"
    assert set(decision.expected_sides) == {"sell", "sell_short"}


def test_buy_then_sell_conflict_same_cycle():
    ctx = _make_context(allow_short_selling=False)
    state = BotState()

    first = bot_engine._resolve_order_intent(  # type: ignore[attr-defined]
        ctx,
        state,
        symbol="AAPL",
        signal_side="buy",
        target_weight=0.2,
    )
    assert first.allowed
    assert first.order_side == "buy"

    conflict = bot_engine._resolve_order_intent(  # type: ignore[attr-defined]
        ctx,
        state,
        symbol="AAPL",
        signal_side="sell",
        target_weight=0.2,
    )

    assert not conflict.allowed
    assert conflict.reason == "cycle_conflict"


def test_same_side_duplicate_blocked_same_cycle():
    ctx = _make_context()
    state = BotState()

    first = bot_engine._resolve_order_intent(  # type: ignore[attr-defined]
        ctx,
        state,
        symbol="AAPL",
        signal_side="buy",
        target_weight=0.2,
    )
    assert first.allowed
    assert first.order_side == "buy"

    duplicate = bot_engine._resolve_order_intent(  # type: ignore[attr-defined]
        ctx,
        state,
        symbol="AAPL",
        signal_side="buy",
        target_weight=0.2,
    )

    assert not duplicate.allowed
    assert duplicate.reason == "cycle_duplicate"
    assert duplicate.details["duplicate_side"] == "buy"


def test_conflict_logs_structured_event(caplog):
    conflict_engine = DummyExecutionEngine(
        pending=[DummyOrderInfo("AAPL", "buy")]
    )
    ctx = _make_context(execution_engine=conflict_engine)
    state = BotState()

    decision = bot_engine._resolve_order_intent(  # type: ignore[attr-defined]
        ctx,
        state,
        symbol="AAPL",
        signal_side="sell",
        target_weight=0.0,
    )
    assert not decision.allowed
    assert decision.reason == "open_order_conflict"

    with caplog.at_level("ERROR"):
        if not decision:
            bot_engine.logger.error("ORDER_INTENT_BLOCKED", extra=decision.details)

    assert any(rec.message == "ORDER_INTENT_BLOCKED" for rec in caplog.records)
    error_record = next(rec for rec in caplog.records if rec.message == "ORDER_INTENT_BLOCKED")
    assert getattr(error_record, "intended_order_side") == "sell"
    assert getattr(error_record, "conflict_side") == ("buy",)


def test_cycle_duplicate_logs_info_level(caplog):
    ctx = _make_context()
    state = BotState()

    assert bot_engine._resolve_order_intent(  # type: ignore[attr-defined]
        ctx,
        state,
        symbol="AAPL",
        signal_side="buy",
        target_weight=0.2,
    )
    decision = bot_engine._resolve_order_intent(  # type: ignore[attr-defined]
        ctx,
        state,
        symbol="AAPL",
        signal_side="buy",
        target_weight=0.2,
    )
    assert not decision

    caplog.set_level(logging.INFO)
    bot_engine._log_order_intent_blocked(decision)  # type: ignore[attr-defined]

    record = next(rec for rec in caplog.records if rec.message == "ORDER_INTENT_BLOCKED")
    assert record.levelno == logging.INFO
    assert getattr(record, "duplicate_side") == "buy"
