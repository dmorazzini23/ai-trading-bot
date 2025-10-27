from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ai_trading.execution.live_trading import ExecutionEngine, CapacityCheck


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


def test_execute_order_validation_failure_logs_context(caplog):
    engine = StubExecutionEngine()
    engine._position_tracker = {"AAPL": 4}
    caplog.set_level(logging.ERROR)

    with pytest.raises(ValueError):
        engine.execute_order("AAPL", "invalid", qty=5, order_type="market")

    failure = next(record for record in caplog.records if record.msg == "ORDER_VALIDATION_FAILED")
    assert failure.symbol == "AAPL"
    assert failure.reason == "invalid_side"
    assert failure.position_qty_before == 4

    caplog.clear()
    with pytest.raises(ValueError):
        engine.execute_order("AAPL", "buy", qty=0, order_type="market")

    qty_failure = next(record for record in caplog.records if record.msg == "ORDER_VALIDATION_FAILED")
    assert qty_failure.reason == "invalid_qty"

