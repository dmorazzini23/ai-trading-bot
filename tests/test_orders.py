from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai_trading.execution import live_trading as lt
from ai_trading.execution.live_trading import (
    CapacityCheck,
    ExecutionEngine,
    NonRetryableBrokerError,
)


class _StubClient:
    def __init__(self) -> None:
        self._poll_sequence = [
            SimpleNamespace(id="order-1", status="accepted"),
            SimpleNamespace(id="order-1", status="filled"),
        ]
        self._poll_index = 0

    def get_order_by_id(self, order_id: str):
        idx = min(self._poll_index, len(self._poll_sequence) - 1)
        self._poll_index += 1
        return self._poll_sequence[idx]

    def get_orders(self, status: str = "open"):
        return []

    def get_all_positions(self):
        return []


class StubExecutionEngine(ExecutionEngine):
    def __init__(self) -> None:
        super().__init__(ctx=None)
        self.is_initialized = True
        self.trading_client = _StubClient()

    def _pre_execution_checks(self) -> bool:
        return True

    def _pre_execution_order_checks(self, order=None) -> bool:
        return True

    def _supports_asset_class(self) -> bool:
        return False

    def _broker_lock_suppressed(self, **kwargs) -> bool:
        return False

    def submit_market_order(self, symbol: str, side: str, quantity: int, **kwargs):
        return {
            "id": "order-1",
            "status": "accepted",
            "qty": quantity,
            "symbol": symbol,
            "side": side,
        }


@pytest.fixture(autouse=True)
def _patch_capacity(monkeypatch):
    monkeypatch.setattr(
        "ai_trading.execution.live_trading._call_preflight_capacity",
        lambda *args, **kwargs: CapacityCheck(True, int(args[3]), None),
    )


def test_execute_order_logs_submission_and_broker_state(caplog):
    engine = StubExecutionEngine()
    caplog.set_level(logging.INFO)

    result = engine.execute_order("AAPL", "buy", qty=5, order_type="market")

    assert result is not None

    submitted = next(record for record in caplog.records if record.msg == "ORDER_SUBMITTED")
    assert submitted.symbol == "AAPL"
    assert submitted.side == "buy"
    assert submitted.qty == 5
    assert submitted.order_id == "order-1"
    broker = next(record for record in caplog.records if record.msg == "BROKER_STATE_AFTER_SUBMIT")
    assert broker.open_orders == 0
    assert broker.positions == 0
    assert broker.final_status.lower() in {"accepted", "filled"}


def test_execute_order_preserves_sell_short_semantics_to_live_route(monkeypatch):
    engine = StubExecutionEngine()
    submitted: list[tuple[str, str, int]] = []

    def _capture_market(symbol: str, side: str, quantity: int, **_kwargs):
        submitted.append((symbol, side, quantity))
        return {
            "id": "order-short",
            "status": "accepted",
            "qty": quantity,
            "symbol": symbol,
            "side": side,
        }

    monkeypatch.setattr(engine, "submit_market_order", _capture_market)

    result = engine.execute_order("AAPL", "sell_short", qty=3, order_type="market")

    assert result is not None
    assert submitted == [("AAPL", "sell_short", 3)]


def test_execute_order_validation_failure_logs_context(caplog):
    engine = StubExecutionEngine()
    setattr(engine, "_position_tracker", {"AAPL": 4})
    caplog.set_level(logging.ERROR)

    with pytest.raises(ValueError):
        engine.execute_order("AAPL", cast(Any, "invalid"), qty=5, order_type="market")

    failure = next(record for record in caplog.records if record.msg == "ORDER_VALIDATION_FAILED")
    assert failure.symbol == "AAPL"
    assert failure.reason == "invalid_side"
    assert failure.position_qty_before == 4

    caplog.clear()
    with pytest.raises(ValueError):
        engine.execute_order("AAPL", "buy", qty=0, order_type="market")

    qty_failure = next(record for record in caplog.records if record.msg == "ORDER_VALIDATION_FAILED")
    assert qty_failure.reason == "invalid_qty"


class _Request:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class _LimitRequest(_Request):
    def __init__(self, *, limit_price: float, **kwargs: Any) -> None:
        super().__init__(limit_price=limit_price, **kwargs)


class _SubmitClient:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def submit_order(self, *, order_data: Any) -> Any:
        self.calls.append(order_data)
        return SimpleNamespace(
            id="order-1",
            status="accepted",
            symbol=getattr(order_data, "symbol", None),
            side=getattr(order_data, "side", None),
            qty=getattr(order_data, "qty", None),
            client_order_id=getattr(order_data, "client_order_id", None),
        )


def _broker_submit_engine(monkeypatch) -> tuple[ExecutionEngine, _SubmitClient]:
    engine = ExecutionEngine.__new__(ExecutionEngine)
    client = _SubmitClient()
    engine.trading_client = client
    monkeypatch.setattr(
        engine,
        "_should_suppress_duplicate_client_order_id",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(engine, "_record_client_order_id_submission", lambda _client_id: None)
    monkeypatch.setattr(engine, "_lookup_order_by_client_order_id", lambda *_a, **_k: None)
    monkeypatch.setattr(engine, "_position_quantity", lambda _symbol: 0)
    monkeypatch.setattr(engine, "open_order_totals", lambda _symbol: (0.0, 0.0))
    monkeypatch.setattr(lt, "OrderSide", SimpleNamespace(BUY="buy", SELL="sell"))
    monkeypatch.setattr(lt, "TimeInForce", SimpleNamespace(DAY="day"))
    monkeypatch.setattr(
        lt,
        "PositionIntent",
        SimpleNamespace(
            BUY_TO_CLOSE="buy_to_close",
            SELL_TO_CLOSE="sell_to_close",
            SELL_TO_OPEN="sell_to_open",
        ),
    )
    monkeypatch.setattr(lt, "MarketOrderRequest", _Request)
    monkeypatch.setattr(lt, "LimitOrderRequest", _LimitRequest)
    return engine, client


def test_submit_order_to_alpaca_maps_explicit_short_to_sell_to_open(monkeypatch):
    engine, client = _broker_submit_engine(monkeypatch)

    engine._submit_order_to_alpaca(
        {
            "symbol": "AAPL",
            "side": "sell_short",
            "quantity": 5,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": "cid-short",
        }
    )

    req = client.calls[-1]
    assert req.side == "sell"
    assert req.position_intent == "sell_to_open"
    assert req.qty == 5


def test_effective_closing_position_preserves_fractional_quantities() -> None:
    engine = SimpleNamespace(_position_quantity=lambda _symbol: 0.5)

    assert lt._effective_closing_position(
        engine,
        symbol="AAPL",
        side="sell",
        quantity=0.25,
        closing_position=False,
    )


def test_effective_closing_position_treats_epsilon_position_as_flat() -> None:
    engine = SimpleNamespace(_position_quantity=lambda _symbol: 0.0000000001)

    assert not lt._effective_closing_position(
        engine,
        symbol="AAPL",
        side="sell",
        quantity=0.25,
        closing_position=False,
    )


def test_submit_order_to_alpaca_rejects_sell_short_reduce_only(monkeypatch):
    engine, client = _broker_submit_engine(monkeypatch)

    with pytest.raises(NonRetryableBrokerError, match="sell_short_cannot_close_position"):
        engine._submit_order_to_alpaca(
            {
                "symbol": "AAPL",
                "side": "sell_short",
                "quantity": 5,
                "type": "market",
                "time_in_force": "day",
                "client_order_id": "cid-short-close",
                "reduce_only": True,
            }
        )

    assert client.calls == []


def test_submit_order_to_alpaca_clips_cover_and_uses_buy_to_close(monkeypatch):
    engine, client = _broker_submit_engine(monkeypatch)
    monkeypatch.setattr(engine, "_position_quantity", lambda _symbol: -4)

    engine._submit_order_to_alpaca(
        {
            "symbol": "AAPL",
            "side": "cover",
            "quantity": 10,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": "cid-cover",
            "reduce_only": True,
            "closing_position": True,
        }
    )

    req = client.calls[-1]
    assert req.side == "buy"
    assert req.position_intent == "buy_to_close"
    assert req.qty == 4


def test_submit_order_to_alpaca_clips_reduce_only_sell_and_uses_sell_to_close(monkeypatch):
    engine, client = _broker_submit_engine(monkeypatch)
    monkeypatch.setattr(engine, "_position_quantity", lambda _symbol: 6)

    engine._submit_order_to_alpaca(
        {
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 10,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": "cid-sell-close",
            "reduce_only": True,
            "closing_position": True,
        }
    )

    req = client.calls[-1]
    assert req.side == "sell"
    assert req.position_intent == "sell_to_close"
    assert req.qty == 6


def test_execute_order_rejects_standalone_protective_order_type_before_limit_validation():
    engine = StubExecutionEngine()

    with pytest.raises(ValueError, match="unsupported standalone order type: stop"):
        engine.execute_order("AAPL", "sell", qty=1, order_type=cast(Any, "stop"), stop_price=99.0)
