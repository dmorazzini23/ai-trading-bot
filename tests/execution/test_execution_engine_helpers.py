from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai_trading.core.enums import OrderSide, OrderStatus, OrderType
from ai_trading.execution import engine as eng
from ai_trading.math.money import Money
from ai_trading.monitoring.order_health_monitor import OrderInfo


def test_order_validation_helpers_and_stale_cleanup(monkeypatch):
    assert eng._ensure_positive_qty(cast(Any, "2.5")) == 2.5
    assert eng._ensure_valid_price(None) is None
    assert eng._ensure_valid_price(cast(Any, "101.25")) == 101.25
    assert eng._normalize_order_side("buy") is OrderSide.BUY
    assert eng._normalize_order_side("unknown") is None
    assert eng._as_bool("YES")
    assert not eng._as_bool("no")

    with pytest.raises(ValueError, match="qty_none"):
        eng._ensure_positive_qty(None)
    with pytest.raises(ValueError, match="invalid_qty"):
        eng._ensure_positive_qty(0)
    with pytest.raises(ValueError, match="invalid_price"):
        eng._ensure_valid_price(float("nan"))

    monkeypatch.setattr(eng, "_active_orders", {}, raising=False)
    eng._active_orders["old"] = OrderInfo("old", "AAPL", "buy", 1, 10.0, "new")
    eng._active_orders["new"] = OrderInfo("new", "MSFT", "sell", 1, 98.0, "new")

    assert eng._cleanup_stale_orders(now=100.0, max_age_s=50) == 1
    assert set(eng._active_orders) == {"new"}


def test_safe_counter_increment_and_deterministic_jitter(monkeypatch):
    class Counter:
        def __init__(self):
            self.values: list[float] = []

        def inc(self, amount):
            self.values.append(float(amount))

    counter = Counter()
    eng._safe_counter_inc(counter, "orders", amount=2.5, extra={"source": "test"})
    assert counter.values == [2.5]

    class BrokenCounter:
        def inc(self, _amount):
            raise ValueError("metric backend down")

    eng._safe_counter_inc(BrokenCounter(), "orders")
    monkeypatch.setattr(eng, "hash", lambda _value: 75, raising=False)
    try:
        assert eng._deterministic_fill_jitter_ratio("AAPL", "buy") == pytest.approx(0.0025)
    finally:
        monkeypatch.delattr(eng, "hash", raising=False)
    assert eng._deterministic_fill_jitter_ratio("AAPL", "buy") == eng._deterministic_fill_jitter_ratio(
        "AAPL",
        "buy",
    )


def test_order_fill_cancel_and_dict_representation():
    order = eng.Order(
        "AAPL",
        OrderSide.BUY,
        4,
        order_type=OrderType.LIMIT,
        price=Money("100.00"),
        expected_price=Money("99.50"),
        client_order_id="cid-1",
    )

    assert order.remaining_quantity == 4
    assert order.notional_value == Money("400.00")
    order.add_fill(1, Money("100.00"))
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.is_partially_filled
    order.add_fill(3, Money("101.00"))
    assert order.status is OrderStatus.FILLED
    assert order.is_filled
    assert not order.cancel("too late")

    payload = order.to_dict()
    assert payload["side"] == "buy"
    assert payload["order_type"] == "limit"
    assert payload["status"] == "filled"
    assert payload["client_order_id"] == "cid-1"
    assert payload["fill_percentage"] == 100.0
    assert payload["executed_at"] is not None

    pending = eng.Order("MSFT", OrderSide.SELL, 2)
    assert pending.cancel("manual")
    assert pending.status is OrderStatus.CANCELED
    assert "manual" in pending.notes


def test_execution_result_normalizes_side_status_quantities_and_weight():
    order = eng.Order("AAPL", OrderSide.BUY, 10)
    result = eng.ExecutionResult(order, "partially_filled", cast(Any, "4"), cast(Any, "10"), 0.5)

    assert str(result) == order.id
    assert result.side == "buy"
    assert result.symbol == "AAPL"
    assert result.status is OrderStatus.PARTIALLY_FILLED
    assert result.has_fill
    assert result.fill_ratio == 0.4
    assert result.filled_weight == pytest.approx(0.2)

    setattr(cast(Any, order), "side", "sell_short")
    assert result.side == "sell"
    setattr(cast(Any, order), "side", "cover")
    assert result.side == "buy"
    setattr(cast(Any, order), "side", "nonsense")
    assert result.side is None

    empty = eng.ExecutionResult(None, "not-a-status", None, cast(Any, "bad"), cast(Any, "bad"))
    assert empty.status is None
    assert empty.side is None
    assert empty.symbol is None
    assert empty.fill_ratio == 0.0
    assert empty.filled_weight is None


def test_execution_algorithm_slices_and_quantity_helpers():
    engine = eng.ExecutionEngine()

    assert engine._normalize_execution_algorithm("is") is eng.ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL
    assert engine._normalize_execution_algorithm("pov") is eng.ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL
    assert engine._normalize_execution_algorithm("") is None
    assert engine._normalize_execution_urgency("urgent") == 0.95
    assert engine._normalize_execution_urgency("bad") == 0.5
    assert engine._normalize_execution_urgency(1.5) == 1.0
    assert engine._allocate_weighted_quantities(7, [1, 2, 0, "bad"]) == [2, 5]
    assert engine._allocate_weighted_quantities(5, [0, -1]) == [5]
    assert engine._default_vwap_profile(0) == [1.0]
    assert len(engine._default_vwap_profile(10)) == 10

    twap_slices, twap_meta = engine._build_algorithmic_slices(
        algorithm=eng.ExecutionAlgorithm.TWAP,
        total_quantity=7,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        kwargs={"duration_minutes": "10", "twap_slices": "3"},
    )
    vwap_slices, vwap_meta = engine._build_algorithmic_slices(
        algorithm=eng.ExecutionAlgorithm.VWAP,
        total_quantity=7,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        kwargs={"duration_minutes": 16, "volume_profile": [1, "bad", 3], "limit_price": "100.5"},
    )
    fallback_slices, fallback_meta = engine._build_algorithmic_slices(
        algorithm=eng.ExecutionAlgorithm.MARKET,
        total_quantity=5,
        side=OrderSide.SELL,
        order_type=cast(Any, "unknown"),
        kwargs={"duration_minutes": "bad"},
    )

    assert sum(slice_def["qty"] for slice_def in twap_slices) == 7
    assert twap_meta["benchmark_style"] == "time"
    assert [slice_def["qty"] for slice_def in vwap_slices] == [2, 5]
    assert all(slice_def["order_type"] == "limit" for slice_def in vwap_slices)
    assert all(slice_def["limit_price"] == 100.5 for slice_def in vwap_slices)
    assert vwap_meta["volume_profile_name"] == "custom"
    assert fallback_slices == [{"qty": 5, "order_type": "market"}]
    assert fallback_meta["benchmark_style"] == "arrival"


def test_broker_snapshot_positions_and_parent_scope_summaries(monkeypatch):
    engine = eng.ExecutionEngine()
    monkeypatch.setattr(engine, "_emit_runtime_snapshots_from_broker_sync", lambda **_kwargs: None)

    snapshot = engine._update_broker_snapshot(
        open_orders=[
            {"symbol": "aapl", "side": "buy", "qty": "2"},
            SimpleNamespace(symbol="AAPL", side="sell_short", remaining_qty="1.5"),
            {"symbol": "MSFT", "side": "bad", "qty": 10},
            {"symbol": "", "side": "buy", "qty": 1},
        ],
        positions=[
            {"symbol": "AAPL", "qty": "4", "side": "long"},
            SimpleNamespace(symbol="MSFT", quantity="3.5", side="short"),
            {"symbol": "BAD", "qty": "oops"},
        ],
    )

    assert snapshot.open_buy_by_symbol == {"AAPL": 2.0}
    assert snapshot.open_sell_by_symbol == {"AAPL": 1.5}
    assert engine.open_order_totals("aapl") == (2.0, 1.5)
    assert engine.position_ledger == {}
    assert engine._position_tracker == {"AAPL": 4.0, "MSFT": -3.5}

    engine._append_parent_execution_summary_history(
        {
            "symbol": "aapl",
            "strategy_id": "mean",
            "session_id": "regular",
            "requested_quantity": 10,
            "submitted_quantity": 8,
            "failed_slices": 1,
            "retry_count": 2,
            "cancel_replace_count": 1,
            "success_ratio": 0.5,
        }
    )
    engine._append_parent_execution_summary_history(
        {
            "symbol": "AAPL",
            "strategy_id": "mean",
            "session_id": "regular",
            "requested_quantity": 5,
            "submitted_quantity": 5,
            "success_ratio": 1.0,
        }
    )
    engine._append_parent_execution_summary_history(
        {"symbol": "MSFT", "requested_quantity": 0, "submitted_quantity": 0}
    )

    summaries = engine.summarize_parent_execution_scopes()
    assert summaries[0]["symbol"] == "AAPL"
    assert summaries[0]["parent_orders"] == 2
    assert summaries[0]["requested_quantity"] == 15
    assert summaries[0]["submitted_quantity"] == 13
    assert summaries[0]["avg_success_ratio"] == pytest.approx(0.75)
    assert summaries[0]["avg_fill_ratio"] == pytest.approx(((8 / 10) + (5 / 5)) / 2)
    assert summaries[1]["symbol"] == "MSFT"
