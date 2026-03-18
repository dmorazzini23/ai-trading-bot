from __future__ import annotations

import json
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

    def _cancel_order(order_id: str) -> bool:
        cancel_calls.append(order_id)
        return True

    monkeypatch.setattr(
        engine,
        "_cancel_order_alpaca",
        _cancel_order,
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


def test_execute_order_persists_runtime_order_and_fill_events(monkeypatch, tmp_path):
    class _FillPriceClient(_AckStubClient):
        def get_order_by_id(self, order_id: str):
            self._polls += 1
            if self._polls >= 2:
                return {
                    "id": order_id,
                    "status": "filled",
                    "filled_qty": "5",
                    "filled_avg_price": "100.25",
                }
            return {"id": order_id, "status": "accepted", "filled_qty": "0"}

    monkeypatch.setenv("AI_TRADING_RUNTIME_EXEC_EVENT_PERSIST_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_ORDER_EVENTS_PATH", "runtime/order_events.jsonl")
    monkeypatch.setenv("AI_TRADING_FILL_EVENTS_PATH", "runtime/fill_events.jsonl")

    engine = _build_engine(_FillPriceClient())
    _prime_engine(engine, monkeypatch)

    result = engine.execute_order("AAPL", "buy", qty=5, order_type="limit", limit_price=100.0)

    assert result is not None
    order_events_path = tmp_path / "runtime" / "order_events.jsonl"
    fill_events_path = tmp_path / "runtime" / "fill_events.jsonl"
    assert order_events_path.exists()
    assert fill_events_path.exists()

    order_rows = [
        json.loads(line)
        for line in order_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fill_rows = []
    if fill_events_path.exists():
        fill_rows = [
            json.loads(line)
            for line in fill_events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    assert any(row.get("event") == "status_transition" for row in order_rows)
    assert any(row.get("event") == "final_state" and row.get("status") == "filled" for row in order_rows)
    assert any(row.get("event") == "fill_recorded" and row.get("symbol") == "AAPL" for row in fill_rows)


def test_synchronize_broker_state_reconciles_pending_fill_events(monkeypatch, tmp_path):
    class _BrokerTerminalFillClient(_AckStubClient):
        def get_orders(self, status: str = "open"):
            return []

        def get_order_by_id(self, order_id: str):
            return {
                "id": order_id,
                "client_order_id": "client-1",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "5",
                "status": "filled",
                "filled_qty": "5",
                "filled_avg_price": "100.75",
                "filled_at": "2026-03-11T19:55:48+00:00",
            }

    monkeypatch.setenv("AI_TRADING_RUNTIME_EXEC_EVENT_PERSIST_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_ORDER_EVENTS_PATH", "runtime/order_events.jsonl")
    monkeypatch.setenv("AI_TRADING_FILL_EVENTS_PATH", "runtime/fill_events.jsonl")

    engine = _build_engine(_BrokerTerminalFillClient())
    _prime_engine(engine, monkeypatch)
    engine._pending_orders = {
        "order-reconcile": {
            "status": "pending_new",
            "symbol": "AAPL",
            "side": "buy",
            "qty": 5,
            "order_type": "limit",
            "client_order_id": "client-1",
        }
    }

    snapshot = engine.synchronize_broker_state()

    assert snapshot is not None
    order_events_path = tmp_path / "runtime" / "order_events.jsonl"
    fill_events_path = tmp_path / "runtime" / "fill_events.jsonl"
    order_rows = [
        json.loads(line)
        for line in order_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fill_rows = []
    if fill_events_path.exists():
        fill_rows = [
            json.loads(line)
            for line in fill_events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    assert any(
        row.get("event") == "status_transition"
        and row.get("source") == "broker_reconcile"
        and row.get("status") == "filled"
        for row in order_rows
    )
    assert any(
        row.get("event") == "final_state"
        and row.get("source") == "broker_reconcile"
        and row.get("status") == "filled"
        for row in order_rows
    )
    assert any(
        row.get("event") == "fill_recorded"
        and row.get("symbol") == "AAPL"
        and row.get("order_id") == "order-reconcile"
        for row in fill_rows
    )
    assert engine._pending_orders == {}


def test_synchronize_broker_state_reconciles_pending_cancel_without_fill(monkeypatch, tmp_path):
    class _BrokerTerminalCancelClient(_AckStubClient):
        def get_orders(self, status: str = "open"):
            return []

        def get_order_by_id(self, order_id: str):
            return {
                "id": order_id,
                "client_order_id": "client-2",
                "symbol": "MSFT",
                "side": "buy",
                "qty": "3",
                "status": "canceled",
                "filled_qty": "0",
            }

    monkeypatch.setenv("AI_TRADING_RUNTIME_EXEC_EVENT_PERSIST_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_ORDER_EVENTS_PATH", "runtime/order_events.jsonl")
    monkeypatch.setenv("AI_TRADING_FILL_EVENTS_PATH", "runtime/fill_events.jsonl")

    engine = _build_engine(_BrokerTerminalCancelClient())
    _prime_engine(engine, monkeypatch)
    engine._pending_orders = {
        "order-cancel": {
            "status": "pending_new",
            "symbol": "MSFT",
            "side": "buy",
            "qty": 3,
            "order_type": "limit",
            "client_order_id": "client-2",
        }
    }

    snapshot = engine.synchronize_broker_state()

    assert snapshot is not None
    order_events_path = tmp_path / "runtime" / "order_events.jsonl"
    fill_events_path = tmp_path / "runtime" / "fill_events.jsonl"
    order_rows = [
        json.loads(line)
        for line in order_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fill_rows = []
    if fill_events_path.exists():
        fill_rows = [
            json.loads(line)
            for line in fill_events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    assert any(
        row.get("event") == "final_state"
        and row.get("source") == "broker_reconcile"
        and row.get("status") == "canceled"
        for row in order_rows
    )
    assert not fill_rows
    assert engine._pending_orders == {}


def test_synchronize_broker_state_bootstraps_pending_from_order_events(monkeypatch, tmp_path):
    class _BrokerHydrateClient(_AckStubClient):
        def get_orders(self, status: str = "open"):
            return []

        def get_order_by_id(self, order_id: str):
            return {
                "id": order_id,
                "client_order_id": "client-hydrate",
                "symbol": "IBM",
                "side": "buy",
                "qty": "2",
                "status": "filled",
                "filled_qty": "2",
                "filled_avg_price": "145.10",
                "filled_at": "2026-03-11T19:40:16+00:00",
            }

    monkeypatch.setenv("AI_TRADING_RUNTIME_EXEC_EVENT_PERSIST_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_ORDER_EVENTS_PATH", "runtime/order_events.jsonl")
    monkeypatch.setenv("AI_TRADING_FILL_EVENTS_PATH", "runtime/fill_events.jsonl")

    order_events_path = tmp_path / "runtime" / "order_events.jsonl"
    order_events_path.parent.mkdir(parents=True, exist_ok=True)
    order_events_path.write_text(
        json.dumps(
            {
                "ts": "2026-03-11T19:40:16.466514+00:00",
                "event": "final_state",
                "order_id": "order-hydrate",
                "client_order_id": "client-hydrate",
                "symbol": "IBM",
                "side": "buy",
                "status": "pending_new",
                "order_type": "limit",
                "qty": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    engine = _build_engine(_BrokerHydrateClient())
    _prime_engine(engine, monkeypatch)
    engine._pending_orders = {}

    snapshot = engine.synchronize_broker_state()

    assert snapshot is not None
    order_rows = [
        json.loads(line)
        for line in order_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fill_events_path = tmp_path / "runtime" / "fill_events.jsonl"
    fill_rows = [
        json.loads(line)
        for line in fill_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        row.get("event") == "final_state"
        and row.get("source") == "broker_reconcile"
        and row.get("order_id") == "order-hydrate"
        and row.get("status") == "filled"
        for row in order_rows
    )
    assert any(
        row.get("event") == "fill_recorded"
        and row.get("order_id") == "order-hydrate"
        and row.get("symbol") == "IBM"
        for row in fill_rows
    )
    assert engine._pending_orders == {}


def test_pending_bootstrap_ignores_older_pending_when_latest_terminal(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_TRADING_RUNTIME_EXEC_EVENT_PERSIST_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_ORDER_EVENTS_PATH", "runtime/order_events.jsonl")
    monkeypatch.setenv("AI_TRADING_FILL_EVENTS_PATH", "runtime/fill_events.jsonl")
    monkeypatch.setenv("AI_TRADING_PENDING_RECONCILE_ARTIFACT_SCAN_LINES", "2000")
    monkeypatch.setenv("AI_TRADING_PENDING_RECONCILE_ARTIFACT_MAX_CANDIDATES", "50")

    order_events_path = tmp_path / "runtime" / "order_events.jsonl"
    order_events_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "ts": "2026-03-11T18:35:25.493191+00:00",
            "event": "status_transition",
            "source": "initial",
            "order_id": "order-terminal",
            "client_order_id": "client-terminal",
            "symbol": "IBM",
            "side": "buy",
            "status": "pending_new",
            "order_type": "limit",
            "quantity": 2,
        },
        {
            "ts": "2026-03-12T01:32:45.606000+00:00",
            "event": "final_state",
            "source": "broker_reconcile",
            "order_id": "order-terminal",
            "client_order_id": "client-terminal",
            "symbol": "IBM",
            "side": "buy",
            "status": "filled",
            "order_type": "limit",
            "quantity": 2,
            "filled_qty": 2,
        },
        {
            "ts": "2026-03-11T18:39:54.923866+00:00",
            "event": "status_transition",
            "source": "initial",
            "order_id": "order-pending",
            "client_order_id": "client-pending",
            "symbol": "MSFT",
            "side": "buy",
            "status": "pending_new",
            "order_type": "limit",
            "quantity": 3,
        },
    ]
    with order_events_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    engine = _build_engine(_AckStubClient())
    _prime_engine(engine, monkeypatch)
    engine._pending_orders = {}

    candidates = engine._load_pending_candidates_from_runtime_order_events()
    assert "order-pending" in candidates
    assert "order-terminal" not in candidates


def test_synchronize_broker_state_reconciles_pending_with_get_order_lookup(monkeypatch, tmp_path):
    class _BrokerGetOrderClient:
        def get_orders(self, status: str = "open"):
            return []

        def get_order(self, order_id: str):
            return {
                "id": order_id,
                "client_order_id": "client-get-order",
                "symbol": "CSCO",
                "side": "buy",
                "qty": "3",
                "status": "filled",
                "filled_qty": "3",
                "filled_avg_price": "51.25",
                "limit_price": "51.00",
                "filled_at": "2026-03-11T20:10:00+00:00",
            }

        def get_all_positions(self):
            return []

        def get_account(self):
            return SimpleNamespace(cash="10000", buying_power="10000")

    monkeypatch.setenv("AI_TRADING_RUNTIME_EXEC_EVENT_PERSIST_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_ORDER_EVENTS_PATH", "runtime/order_events.jsonl")
    monkeypatch.setenv("AI_TRADING_FILL_EVENTS_PATH", "runtime/fill_events.jsonl")

    engine = _build_engine(_BrokerGetOrderClient())
    _prime_engine(engine, monkeypatch)
    engine._pending_orders = {
        "order-get-order": {
            "status": "pending_new",
            "symbol": "CSCO",
            "side": "buy",
            "qty": 3,
            "order_type": "limit",
            "client_order_id": "client-get-order",
        }
    }

    snapshot = engine.synchronize_broker_state()

    assert snapshot is not None
    order_events_path = tmp_path / "runtime" / "order_events.jsonl"
    fill_events_path = tmp_path / "runtime" / "fill_events.jsonl"
    order_rows = [
        json.loads(line)
        for line in order_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fill_rows = [
        json.loads(line)
        for line in fill_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        row.get("event") == "final_state"
        and row.get("source") == "broker_reconcile"
        and row.get("order_id") == "order-get-order"
        and row.get("status") == "filled"
        for row in order_rows
    )
    assert any(
        row.get("event") == "fill_recorded"
        and row.get("order_id") == "order-get-order"
        and row.get("symbol") == "CSCO"
        for row in fill_rows
    )
    fill_row = next(
        row
        for row in fill_rows
        if row.get("event") == "fill_recorded" and row.get("order_id") == "order-get-order"
    )
    assert float(fill_row["fill_price"]) == pytest.approx(51.25)
    assert float(fill_row["fill_qty"]) == pytest.approx(3.0)
    assert float(fill_row["expected_price"]) == pytest.approx(51.0)
    assert float(fill_row["slippage_bps"]) == pytest.approx(49.0196078431, rel=1e-6)
    assert engine._pending_orders == {}


def test_synchronize_broker_state_terminalizes_missing_lookup(monkeypatch, tmp_path):
    class _MissingLookupClient:
        def get_orders(self, status: str = "open"):
            return []

        def get_order_by_id(self, order_id: str):
            raise RuntimeError("order not found")

        def get_all_positions(self):
            return []

        def get_account(self):
            return SimpleNamespace(cash="10000", buying_power="10000")

    monkeypatch.setenv("AI_TRADING_RUNTIME_EXEC_EVENT_PERSIST_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_ORDER_EVENTS_PATH", "runtime/order_events.jsonl")
    monkeypatch.setenv("AI_TRADING_FILL_EVENTS_PATH", "runtime/fill_events.jsonl")

    engine = _build_engine(_MissingLookupClient())
    _prime_engine(engine, monkeypatch)
    engine._pending_orders = {
        "order-missing": {
            "status": "pending_new",
            "symbol": "CSCO",
            "side": "buy",
            "qty": 3,
            "order_type": "limit",
            "client_order_id": "client-missing",
        }
    }

    snapshot = engine.synchronize_broker_state()

    assert snapshot is not None
    order_events_path = tmp_path / "runtime" / "order_events.jsonl"
    order_rows = [
        json.loads(line)
        for line in order_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        row.get("event") == "final_state"
        and row.get("source") == "broker_reconcile"
        and row.get("order_id") == "order-missing"
        and row.get("status") == "canceled"
        for row in order_rows
    )
    assert engine._pending_orders == {}


def test_synchronize_broker_state_initializes_client_before_reconcile(monkeypatch, tmp_path):
    class _InitClient:
        def get_orders(self, status: str = "open"):
            return []

        def get_order_by_id(self, order_id: str):
            return {
                "id": order_id,
                "client_order_id": "client-init",
                "symbol": "MSFT",
                "side": "buy",
                "qty": "2",
                "status": "filled",
                "filled_qty": "2",
                "filled_avg_price": "417.50",
                "limit_price": "417.00",
                "filled_at": "2026-03-12T00:59:00+00:00",
            }

        def get_all_positions(self):
            return []

        def get_account(self):
            return SimpleNamespace(cash="10000", buying_power="10000")

    monkeypatch.setenv("AI_TRADING_RUNTIME_EXEC_EVENT_PERSIST_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_ORDER_EVENTS_PATH", "runtime/order_events.jsonl")
    monkeypatch.setenv("AI_TRADING_FILL_EVENTS_PATH", "runtime/fill_events.jsonl")

    engine = _build_engine(_AckStubClient())
    _prime_engine(engine, monkeypatch)
    engine.trading_client = None
    engine.is_initialized = False
    client = _InitClient()
    init_calls = {"count": 0}

    def _ensure_initialized():
        init_calls["count"] += 1
        engine.trading_client = client
        engine.is_initialized = True
        return True

    monkeypatch.setattr(engine, "_ensure_initialized", _ensure_initialized)
    engine._pending_orders = {
        "order-init": {
            "status": "pending_new",
            "symbol": "MSFT",
            "side": "buy",
            "qty": 2,
            "order_type": "limit",
            "client_order_id": "client-init",
        }
    }

    snapshot = engine.synchronize_broker_state()

    assert snapshot is not None
    assert init_calls["count"] >= 1
    order_events_path = tmp_path / "runtime" / "order_events.jsonl"
    order_rows = [
        json.loads(line)
        for line in order_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        row.get("event") == "final_state"
        and row.get("source") == "broker_reconcile"
        and row.get("order_id") == "order-init"
        and row.get("status") == "filled"
        for row in order_rows
    )
    fill_events_path = tmp_path / "runtime" / "fill_events.jsonl"
    fill_rows = [
        json.loads(line)
        for line in fill_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fill_row = next(
        row
        for row in fill_rows
        if row.get("event") == "fill_recorded" and row.get("order_id") == "order-init"
    )
    assert float(fill_row["fill_price"]) == pytest.approx(417.5)
    assert float(fill_row["fill_qty"]) == pytest.approx(2.0)
    assert float(fill_row["expected_price"]) == pytest.approx(417.0)
    assert float(fill_row["slippage_bps"]) == pytest.approx(11.9904076739, rel=1e-6)
    assert engine._pending_orders == {}
