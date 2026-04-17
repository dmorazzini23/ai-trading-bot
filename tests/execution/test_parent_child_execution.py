from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.execution.engine import ExecutionAlgorithm
from ai_trading.execution.engine import ExecutionEngine
from ai_trading.core.enums import OrderSide, OrderType


def test_execute_sliced_retries_with_limit_replace(monkeypatch) -> None:
    engine = ExecutionEngine()
    calls: list[dict[str, Any]] = []
    attempts = {"count": 0}

    def _fake_execute_order(
        symbol: str,
        side: Any,
        quantity: int,
        order_type: Any,
        *,
        asset_class: str | None = None,
        **kwargs: Any,
    ) -> Any:
        attempts["count"] += 1
        calls.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "asset_class": asset_class,
                "kwargs": dict(kwargs),
            }
        )
        if attempts["count"] == 1:
            return None
        return SimpleNamespace(
            filled_quantity=quantity,
            order=SimpleNamespace(average_fill_price=101.0, expected_price=100.0),
        )

    monkeypatch.setattr(engine, "execute_order", _fake_execute_order)
    monkeypatch.setattr("ai_trading.execution.engine.time.sleep", lambda _seconds: None)

    results = engine.execute_sliced(
        [{"qty": 10, "limit_price": 100.0}],
        symbol="AAPL",
        side="buy",
        quantity=10,
        order_type="limit",
        strategy_id="mean_reversion_v2",
        session_id="regular_hours",
        slice_retry_attempts=2,
        slice_limit_replace_step_bps=10.0,
        slice_allow_cancel_replace=True,
        slice_retry_backoff_seconds=0.0,
        slice_interval_seconds=0.0,
    )

    assert len(results) == 1
    assert results[0] is not None
    assert len(calls) == 2
    first_call = calls[0]["kwargs"]
    second_call = calls[1]["kwargs"]
    assert first_call["slice_attempt"] == 1
    assert second_call["slice_attempt"] == 2
    assert first_call["slice_sequence"] == second_call["slice_sequence"] == 1
    assert first_call["slice_count"] == second_call["slice_count"] == 1
    assert first_call["parent_order_id"] == second_call["parent_order_id"]
    assert float(second_call["limit_price"]) > float(first_call["limit_price"])

    summary = engine.last_parent_execution_summary
    assert summary is not None
    assert summary["retry_count"] == 1
    assert summary["cancel_replace_count"] == 1
    assert summary["submitted_slices"] == 1
    assert summary["failed_slices"] == 0
    assert summary["strategy_id"] == "mean_reversion_v2"
    assert summary["session_id"] == "regular_hours"
    assert summary["arrival_slippage_sample_count"] == 1


def test_execute_sliced_applies_participation_cap_split(monkeypatch) -> None:
    engine = ExecutionEngine()
    quantities: list[int] = []

    def _fake_execute_order(
        symbol: str,
        side: Any,
        quantity: int,
        order_type: Any,
        *,
        asset_class: str | None = None,
        **kwargs: Any,
    ) -> Any:
        quantities.append(int(quantity))
        return SimpleNamespace(filled_quantity=quantity)

    monkeypatch.setattr(engine, "execute_order", _fake_execute_order)
    monkeypatch.setattr("ai_trading.execution.engine.time.sleep", lambda _seconds: None)

    results = engine.execute_sliced(
        [6, 4],
        symbol="MSFT",
        side="sell",
        quantity=10,
        max_participation_rate=0.2,
        avg_daily_volume=15,
        slice_retry_attempts=1,
        slice_interval_seconds=0.0,
    )

    assert len(results) == 4
    assert quantities == [3, 3, 3, 1]
    summary = engine.last_parent_execution_summary
    assert summary is not None
    assert summary["child_slice_target"] == 4
    assert summary["participation_cap_qty"] == 3
    assert summary["submitted_quantity"] == 10


def test_execute_sliced_persists_parent_summary_event(monkeypatch) -> None:
    engine = ExecutionEngine()

    class _EventStore:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def append_oms_event_payload(self, **kwargs: Any) -> bool:
            self.calls.append(dict(kwargs))
            return True

    store = _EventStore()
    monkeypatch.setattr(engine, "_resolve_execution_audit_store", lambda: store)
    monkeypatch.setattr("ai_trading.execution.engine.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        engine,
        "execute_order",
        lambda *_args, **_kwargs: SimpleNamespace(filled_quantity=5),
    )

    engine.execute_sliced(
        [5],
        symbol="NVDA",
        side="buy",
        quantity=5,
        strategy_id="momentum_breakout",
        session_id="pre_market",
        slice_retry_attempts=1,
        slice_interval_seconds=0.0,
    )

    assert len(store.calls) == 1
    payload = store.calls[0]["payload"]
    assert store.calls[0]["event_type"] == "RECONCILE_UPDATE"
    assert payload["record_type"] == "parent_execution_summary"
    assert payload["symbol"] == "NVDA"
    assert payload["strategy_id"] == "momentum_breakout"
    assert payload["session_id"] == "pre_market"

    summary = engine.last_parent_execution_summary
    assert summary is not None
    assert summary["persisted_to_event_store"] is True


def test_execute_order_routes_vwap_through_parent_child(monkeypatch) -> None:
    engine = ExecutionEngine()
    captured: dict[str, Any] = {}

    def _fake_execute_sliced(slices: Any, **kwargs: Any) -> list[Any]:
        captured["slices"] = list(slices)
        captured["kwargs"] = dict(kwargs)
        return [
            SimpleNamespace(
                filled_quantity=4,
                order=SimpleNamespace(average_fill_price=100.4, expected_price=100.0),
            ),
            SimpleNamespace(
                filled_quantity=6,
                order=SimpleNamespace(average_fill_price=100.3, expected_price=100.0),
            ),
        ]

    monkeypatch.setattr(engine, "execute_sliced", _fake_execute_sliced)

    result = engine.execute_order(
        "AAPL",
        OrderSide.BUY,
        10,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        execution_algorithm=ExecutionAlgorithm.VWAP,
        vwap_slices=2,
        vwap_duration_min=20,
    )

    assert result is not None
    assert getattr(result, "filled_quantity", 0) == pytest.approx(10.0)
    assert captured["kwargs"]["parent_order_id"]
    assert captured["kwargs"]["execution_algorithm"] == ExecutionAlgorithm.VWAP.value
    assert captured["kwargs"]["schedule_style"] == ExecutionAlgorithm.VWAP.value
    assert captured["kwargs"]["benchmark_style"] == "vwap"
    assert captured["kwargs"]["child_order_execution"] is True
    slices = captured["slices"]
    assert len(slices) == 2
    assert sum(int(slice_def["qty"]) for slice_def in slices) == 10
    assert all(str(slice_def["order_type"]) == "limit" for slice_def in slices)


def test_execute_order_routes_implementation_shortfall_through_parent_child(monkeypatch) -> None:
    engine = ExecutionEngine()
    captured: dict[str, Any] = {}

    def _fake_execute_sliced(slices: Any, **kwargs: Any) -> list[Any]:
        captured["slices"] = list(slices)
        captured["kwargs"] = dict(kwargs)
        return [
            SimpleNamespace(
                filled_quantity=int(slice_def["qty"]),
                order=SimpleNamespace(average_fill_price=101.0, expected_price=100.0),
            )
            for slice_def in captured["slices"]
        ]

    monkeypatch.setattr(engine, "execute_sliced", _fake_execute_sliced)

    result = engine.execute_order(
        "MSFT",
        OrderSide.BUY,
        12,
        order_type=OrderType.LIMIT,
        limit_price=101.0,
        execution_algorithm=ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL,
        urgency_level="urgent",
        is_duration_min=18,
    )

    assert result is not None
    assert getattr(result, "filled_quantity", 0) == pytest.approx(12.0)
    assert captured["kwargs"]["execution_algorithm"] == (
        ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL.value
    )
    assert captured["kwargs"]["schedule_style"] == (
        ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL.value
    )
    assert captured["kwargs"]["benchmark_style"] == "implementation_shortfall"
    assert captured["kwargs"]["urgency_score"] > 0.75
    slices = captured["slices"]
    assert len(slices) >= 3
    assert sum(int(slice_def["qty"]) for slice_def in slices) == 12
    assert str(slices[0]["order_type"]) == "market"
