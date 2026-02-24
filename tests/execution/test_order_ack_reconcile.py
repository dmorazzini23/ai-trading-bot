from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from tests.test_orders import StubExecutionEngine


class _AckStubClient:
    def __init__(self) -> None:
        self._polls = 0

    def submit_order(self, *args, **kwargs):
        self._polls = 0
        return {"id": "order-1", "status": "accepted", "filled_qty": "0"}

    def get_order_by_id(self, order_id: str):
        self._polls += 1
        if self._polls >= 2:
            return {"id": order_id, "status": "filled", "filled_qty": "5"}
        return {"id": order_id, "status": "accepted", "filled_qty": "0"}

    def get_order_by_client_order_id(self, client_order_id: str):
        return self.get_order_by_id(client_order_id)

    def get_orders(self, status: str = "open"):
        return []

    def list_orders(self, status: str = "open"):
        return []

    def get_all_positions(self):
        if self._polls < 2:
            return []
        return [SimpleNamespace(symbol="AAPL", qty="5")]

    def list_positions(self):
        return self.get_all_positions()

    def get_account(self):
        if self._polls < 2:
            return SimpleNamespace(cash="10000", buying_power="10000")
        return SimpleNamespace(cash="9750", buying_power="9750")


class _TimeoutStubClient(_AckStubClient):
    def submit_order(self, *args, **kwargs):
        self._polls = 0
        return {"id": "order-1", "filled_qty": "0"}

    def get_order_by_id(self, order_id: str):
        self._polls += 1
        return {"id": order_id, "filled_qty": "0"}

    def get_account(self):
        return SimpleNamespace(cash="10000", buying_power="10000")


class _MismatchStubClient(_AckStubClient):
    def get_all_positions(self):
        return []

    def list_positions(self):
        return []

    def get_account(self):
        return SimpleNamespace(cash="10000", buying_power="10000")


class _NoFillStubClient(_AckStubClient):
    def submit_order(self, *args, **kwargs):
        self._polls = 0
        return {"id": "order-1", "status": "accepted", "filled_qty": "0"}

    def get_order_by_id(self, order_id: str):
        self._polls += 1
        return {"id": order_id, "status": "accepted", "filled_qty": "0"}

    def get_order_by_client_order_id(self, client_order_id: str):
        return self.get_order_by_id(client_order_id)

    def get_all_positions(self):
        return []

    def list_positions(self):
        return []

    def get_account(self):
        return SimpleNamespace(cash="10000", buying_power="10000")


class _AcceptedNoPollClient(_AckStubClient):
    def submit_order(self, *args, **kwargs):
        self._polls = 0
        return {"id": "order-1", "status": "accepted", "filled_qty": "0"}

    def get_order_by_id(self, order_id: str):
        raise ConnectionError("status endpoint unavailable")

    def get_order_by_client_order_id(self, client_order_id: str):
        raise ConnectionError("status endpoint unavailable")


class _AckFirstProbeClient(_AckStubClient):
    def __init__(self) -> None:
        super().__init__()
        self.poll_calls = 0

    def get_order_by_id(self, order_id: str):
        self.poll_calls += 1
        return super().get_order_by_id(order_id)

    def get_order_by_client_order_id(self, client_order_id: str):
        self.poll_calls += 1
        return super().get_order_by_client_order_id(client_order_id)


class _EnumStatusClient(_AckStubClient):
    def submit_order(self, *args, **kwargs):
        self._polls = 0
        return {"id": "order-1", "status": "OrderStatus.PENDING_NEW", "filled_qty": "0"}

    def get_order_by_id(self, order_id: str):
        self._polls += 1
        if self._polls >= 2:
            return {"id": order_id, "status": "OrderStatus.FILLED", "filled_qty": "5"}
        return {"id": order_id, "status": "OrderStatus.PENDING_NEW", "filled_qty": "0"}

    def get_order_by_client_order_id(self, client_order_id: str):
        return self.get_order_by_id(client_order_id)


@pytest.fixture(autouse=True)
def _fast_ack(monkeypatch):
    monkeypatch.setattr("ai_trading.execution.live_trading._ACK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("ai_trading.execution.live_trading.time.sleep", lambda *_: None)


def _build_engine(client) -> StubExecutionEngine:
    engine = StubExecutionEngine()
    engine.trading_client = client
    return engine


def _prime_engine(engine: StubExecutionEngine, monkeypatch):
    monkeypatch.setattr(engine, "_refresh_settings", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine, "_ensure_initialized", lambda *args, **kwargs: True)
    monkeypatch.setattr(engine, "_pre_execution_checks", lambda *args, **kwargs: True)
    monkeypatch.setattr(engine, "_pre_execution_order_checks", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "ai_trading.execution.live_trading.get_trading_config",
        lambda: SimpleNamespace(
            nbbo_required_for_limit=False,
            execution_require_realtime_nbbo=False,
            execution_market_on_degraded=False,
            degraded_feed_mode="widen",
            degraded_feed_limit_widen_bps=0,
            min_quote_freshness_ms=1500,
        ),
    )
    monkeypatch.setattr(
        "ai_trading.execution.live_trading.provider_monitor.is_disabled",
        lambda *_a, **_k: False,
    )

    def _fake_submit_limit_order(symbol: str, side: str, quantity: int, limit_price: float, **kwargs):
        return engine.trading_client.submit_order(symbol, side, quantity, limit_price=limit_price, **kwargs)

    monkeypatch.setattr(engine, "submit_limit_order", _fake_submit_limit_order)


def test_order_submit_ack_and_reconcile_success(caplog, monkeypatch):
    engine = _build_engine(_AckStubClient())
    _prime_engine(engine, monkeypatch)
    caplog.set_level(logging.INFO)

    result = engine.execute_order("AAPL", "buy", qty=5, order_type="limit", limit_price=100.0)

    assert result is not None
    assert getattr(result, "reconciled", False)
    msgs = {record.msg for record in caplog.records}
    assert "ORDER_ACK_RECEIVED" in msgs
    assert "ORDER_FILL_CONFIRMED" in msgs
    assert "BROKER_RECONCILE_SUMMARY" in msgs
    assert "BROKER_RECONCILE_MISMATCH" not in msgs


def test_ack_first_short_circuits_polling(monkeypatch, caplog):
    client = _AckFirstProbeClient()
    engine = _build_engine(client)
    _prime_engine(engine, monkeypatch)
    monkeypatch.setenv("AI_TRADING_ACK_FIRST_RECONCILE_ENABLED", "1")
    caplog.set_level(logging.INFO)

    result = engine.execute_order("AAPL", "buy", qty=5, order_type="limit", limit_price=100.0)

    assert result is not None
    assert client.poll_calls == 0
    msgs = {record.msg for record in caplog.records}
    assert "ORDER_ACK_FIRST_SHORT_CIRCUIT" in msgs


def test_order_submit_ack_timeout(monkeypatch, caplog):
    engine = _build_engine(_TimeoutStubClient())
    _prime_engine(engine, monkeypatch)
    caplog.set_level(logging.WARNING)

    result = engine.execute_order("AAPL", "buy", qty=5, order_type="limit", limit_price=100.0)

    assert result is not None
    assert getattr(result, "reconciled", True) is False
    assert result.status == "submitted"
    msgs = {record.message for record in caplog.records}
    assert "ORDER_PENDING_NO_TERMINAL" in msgs
    assert "ORDER_ACK_TIMEOUT" in msgs


def test_broker_reconcile_mismatch(monkeypatch, caplog):
    engine = _build_engine(_MismatchStubClient())
    _prime_engine(engine, monkeypatch)
    caplog.set_level(logging.WARNING)

    result = engine.execute_order("AAPL", "buy", qty=5, order_type="limit", limit_price=100.0)

    assert result is not None
    assert getattr(result, "reconciled", True) is False
    assert any(record.msg == "BROKER_RECONCILE_MISMATCH" for record in caplog.records)


def test_ack_timeout_pending_no_cancel(monkeypatch, caplog):
    engine = _build_engine(_NoFillStubClient())
    _prime_engine(engine, monkeypatch)
    cancel_calls: list[str] = []
    monkeypatch.setattr(
        engine,
        "_cancel_order_alpaca",
        lambda order_id: cancel_calls.append(order_id) or True,
    )
    caplog.set_level(logging.INFO)

    result = engine.execute_order("AAPL", "buy", qty=5, order_type="limit", limit_price=100.0)

    assert result is not None
    assert getattr(result, "reconciled", True) is False
    assert result.status == "accepted"
    assert getattr(result, "ack_timed_out", True) is False
    assert cancel_calls == []
    msgs = {record.message for record in caplog.records}
    assert "ORDER_PENDING_NO_TERMINAL" not in msgs
    assert "ORDER_PENDING_CANCELLED" not in msgs
    assert "ORDER_ACK_TIMEOUT" not in msgs


def test_ack_timeout_does_not_block_followup_orders(monkeypatch, caplog):
    engine = _build_engine(_TimeoutStubClient())
    _prime_engine(engine, monkeypatch)
    status_updates: list[dict] = []
    monkeypatch.setattr(
        "ai_trading.execution.live_trading.runtime_state.update_broker_status",
        lambda **kwargs: status_updates.append(dict(kwargs)),
    )
    caplog.set_level(logging.INFO)

    result = engine.execute_order("AAPL", "buy", qty=5, order_type="limit", limit_price=100.0)

    assert result is not None
    assert any(update.get("last_error") == "order_ack_timeout" and update.get("connected") for update in status_updates)
    assert any(update.get("last_error") == "order_pending_no_terminal" for update in status_updates)

    caplog.clear()
    engine.trading_client = _AckStubClient()
    submissions: list[str | None] = []
    original_submit = engine.trading_client.submit_order

    def _record_submit(*args, **kwargs):
        symbol = kwargs.get("symbol")
        if not symbol and args:
            symbol = args[0]
        submissions.append(symbol)
        return original_submit(*args, **kwargs)

    monkeypatch.setattr(engine.trading_client, "submit_order", _record_submit)

    followup = engine.execute_order("MSFT", "buy", qty=3, order_type="limit", limit_price=200.0)

    assert followup is not None
    assert "MSFT" in submissions


def test_ack_timeout_logs_initial_status(monkeypatch, caplog):
    engine = _build_engine(_AcceptedNoPollClient())
    _prime_engine(engine, monkeypatch)
    caplog.set_level(logging.WARNING)

    result = engine.execute_order("AAPL", "buy", qty=5, order_type="limit", limit_price=100.0)

    assert result is not None
    assert getattr(result, "reconciled", True) is False
    assert getattr(result, "ack_timed_out", True) is False
    pending = [record for record in caplog.records if record.msg == "ORDER_PENDING_NO_TERMINAL"]
    timeout = [record for record in caplog.records if record.msg == "ORDER_ACK_TIMEOUT"]
    assert not pending
    assert not timeout


def test_submitted_status_counts_as_ack(monkeypatch, caplog):
    class _SubmittedClient(_AckStubClient):
        def submit_order(self, *args, **kwargs):
            self._polls = 0
            return {"id": "order-xyz", "status": "submitted", "filled_qty": "0"}

        def get_order_by_id(self, order_id: str):
            raise ConnectionError("polling disabled")

        def get_order_by_client_order_id(self, client_order_id: str):
            raise ConnectionError("polling disabled")

    engine = _build_engine(_SubmittedClient())
    _prime_engine(engine, monkeypatch)
    caplog.set_level(logging.INFO)

    result = engine.execute_order("AAPL", "buy", qty=2, order_type="limit", limit_price=100.0)

    assert result is not None
    assert getattr(result, "ack_timed_out", True) is False
    msgs = {record.msg for record in caplog.records}
    assert "ORDER_ACK_TIMEOUT" not in msgs
    assert "ORDER_PENDING_NO_TERMINAL" not in msgs
    assert "ORDER_ACK_RECEIVED" in msgs


def test_order_status_enum_tokens_normalized(monkeypatch, caplog):
    engine = _build_engine(_EnumStatusClient())
    _prime_engine(engine, monkeypatch)
    caplog.set_level(logging.INFO)

    result = engine.execute_order("AAPL", "buy", qty=5, order_type="limit", limit_price=100.0)

    assert result is not None
    assert getattr(result, "reconciled", False)
    msgs = {record.msg for record in caplog.records}
    assert "ORDER_SUBMITTED" in msgs
    assert "ORDER_FILL_CONFIRMED" in msgs
    assert "ORDER_PENDING_NO_TERMINAL" not in msgs
