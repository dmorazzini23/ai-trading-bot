from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_trading.execution import live_trading as lt
from ai_trading.health_payload import build_control_plane_snapshot


class _Adapter:
    provider = "paper"

    @staticmethod
    def submit_order(order_data):
        return {
            "id": "paper-1",
            "status": "accepted",
            "client_order_id": order_data.get("client_order_id"),
        }


class _FailingAdapter:
    def submit_order(self, _order_data):
        raise TimeoutError("adapter submit timeout")


class _SuccessAdapter:
    def __init__(self, provider: str) -> None:
        self.provider = provider

    def submit_order(self, order_data):
        return {
            "id": f"{self.provider}-1",
            "status": "accepted",
            "client_order_id": order_data.get("client_order_id"),
        }


def _engine_stub() -> lt.ExecutionEngine:
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine.stats = {}
    return engine


def test_attempt_failover_submit_success(monkeypatch, tmp_path: Path) -> None:
    engine = _engine_stub()
    playbook_path = tmp_path / "broker_playbook.jsonl"
    monkeypatch.setenv("AI_TRADING_BROKER_FAILOVER_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_BROKER_FAILOVER_PROVIDER", "paper")
    monkeypatch.setenv("AI_TRADING_BROKER_RESILIENCE_PLAYBOOK_PATH", str(playbook_path))
    monkeypatch.setattr(lt, "build_broker_adapter", lambda **_kwargs: _Adapter())

    response = engine._attempt_failover_submit(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 2,
            "type": "limit",
            "limit_price": 190.0,
            "client_order_id": "cid-1",
        },
        primary_error=TimeoutError("primary broker timeout"),
    )

    assert response is not None
    assert response["failover"] is True
    assert response["provider"] == "paper"
    assert engine.stats["failover_submits"] == 1
    assert playbook_path.exists()
    lines = playbook_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["action"] == "failover_submit_success"


def test_attempt_failover_submit_disabled(monkeypatch) -> None:
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_BROKER_FAILOVER_ENABLED", "0")
    response = engine._attempt_failover_submit(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1,
            "type": "limit",
            "limit_price": 190.0,
            "client_order_id": "cid-2",
        },
        primary_error=TimeoutError("primary broker timeout"),
    )
    assert response is None


def test_attempt_failover_submit_provider_chain_and_cooldown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    engine = _engine_stub()
    playbook_path = tmp_path / "broker_playbook.jsonl"
    monkeypatch.setenv("AI_TRADING_BROKER_FAILOVER_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_BROKER_FAILOVER_PROVIDERS", "paper,tradier")
    monkeypatch.setenv("AI_TRADING_BROKER_RESILIENCE_PLAYBOOK_PATH", str(playbook_path))
    monkeypatch.setenv("AI_TRADING_BROKER_FAILOVER_PROVIDER_COOLDOWN_SEC", "600")
    monkeypatch.setattr(lt, "monotonic_time", lambda: 1234.0)

    def _build_adapter(*, provider, **_kwargs):
        if provider == "paper":
            return _FailingAdapter()
        if provider == "tradier":
            return _SuccessAdapter("tradier")
        return None

    monkeypatch.setattr(lt, "build_broker_adapter", _build_adapter)

    response = engine._attempt_failover_submit(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 2,
            "type": "limit",
            "limit_price": 190.0,
            "client_order_id": "cid-chain",
        },
        primary_error=TimeoutError("primary broker timeout"),
    )

    assert response is not None
    assert response["provider"] == "tradier"
    assert response["provider_attempt_index"] == 2
    assert response["providers_considered"] == 2
    assert response["post_submit_reconcile"]["attempted"] is True
    assert engine.stats["failover_attempts"] == 2
    assert engine.stats["failover_failures"] == 1
    assert engine.stats["failover_submits"] == 1
    cooldown_map = getattr(engine, "_broker_failover_provider_cooldown_until", {})
    assert "paper" in cooldown_map
    assert "tradier" not in cooldown_map
    assert playbook_path.exists()
    lines = playbook_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["action"] == "failover_submit_success"
    assert payload["provider"] == "tradier"


def test_attempt_failover_submit_skips_provider_on_cooldown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    engine = _engine_stub()
    playbook_path = tmp_path / "broker_playbook.jsonl"
    engine._broker_failover_provider_cooldown_until = {"paper": 1500.0}
    monkeypatch.setenv("AI_TRADING_BROKER_FAILOVER_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_BROKER_FAILOVER_PROVIDERS", "paper")
    monkeypatch.setenv("AI_TRADING_BROKER_RESILIENCE_PLAYBOOK_PATH", str(playbook_path))
    monkeypatch.setattr(lt, "monotonic_time", lambda: 1000.0)
    monkeypatch.setattr(
        lt,
        "build_broker_adapter",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("adapter should not be built")),
    )

    response = engine._attempt_failover_submit(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1,
            "type": "limit",
            "limit_price": 190.0,
            "client_order_id": "cid-cooldown",
        },
        primary_error=TimeoutError("primary broker timeout"),
    )

    assert response is None
    assert engine.stats["failover_provider_cooldown_skips"] == 1
    lines = playbook_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["action"] == "failover_exhausted"


def test_failover_reconcile_and_control_plane_visibility_chain(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("sqlalchemy")
    from ai_trading.oms.event_store import EventStore
    from ai_trading.tca.event_analytics import summarize_oms_event_tca

    engine = _engine_stub()
    playbook_path = tmp_path / "broker_playbook.jsonl"
    monkeypatch.setenv("AI_TRADING_BROKER_FAILOVER_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_BROKER_FAILOVER_PROVIDER", "paper")
    monkeypatch.setenv("AI_TRADING_BROKER_RESILIENCE_PLAYBOOK_PATH", str(playbook_path))
    monkeypatch.setattr(lt, "build_broker_adapter", lambda **_kwargs: _SuccessAdapter("paper"))

    reconcile_calls = {"sync": 0, "intent": 0, "artifact": 0}

    def _sync_state() -> SimpleNamespace:
        reconcile_calls["sync"] += 1
        return SimpleNamespace(open_orders=("ord-1",), positions=("pos-1",))

    def _reconcile_intents(*, open_orders=()) -> None:
        reconcile_calls["intent"] += 1
        assert open_orders == ("ord-1",)

    def _reconcile_artifacts(*, open_orders=()) -> None:
        reconcile_calls["artifact"] += 1
        assert open_orders == ("ord-1",)

    monkeypatch.setattr(engine, "synchronize_broker_state", _sync_state)
    monkeypatch.setattr(engine, "_reconcile_durable_intents", _reconcile_intents)
    monkeypatch.setattr(
        engine,
        "_reconcile_pending_order_runtime_artifacts",
        _reconcile_artifacts,
    )

    response = engine._attempt_failover_submit(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 2,
            "type": "limit",
            "limit_price": 190.0,
            "client_order_id": "cid-chain-full",
        },
        primary_error=TimeoutError("primary broker timeout"),
    )
    assert response is not None
    assert response["failover"] is True
    assert response["provider"] == "paper"
    post_reconcile = response["post_submit_reconcile"]
    assert post_reconcile["attempted"] is True
    assert post_reconcile["reason"] == "reconciled"
    assert reconcile_calls["sync"] == 1
    assert reconcile_calls["intent"] == 1
    assert reconcile_calls["artifact"] == 1

    db_path = tmp_path / "oms_tca_chain.db"
    store = EventStore(url=f"sqlite:///{db_path}")
    store.append_oms_event_payload(
        event_type="SUBMIT_ACK",
        event_source="chain_test",
        idempotency_key="ack-chain-1",
        intent_id="intent-ack-1",
        payload={
            "symbol": "AAPL",
            "strategy_id": "mean_reversion_v2",
            "session_id": "regular_hours",
        },
    )
    store.append_oms_event_payload(
        event_type="SUBMIT_REJECT",
        event_source="chain_test",
        idempotency_key="reject-chain-1",
        intent_id="intent-reject-1",
        payload={
            "symbol": "AAPL",
            "strategy_id": "mean_reversion_v2",
            "session_id": "regular_hours",
            "error": "insufficient_buying_power",
        },
    )
    store.append_oms_event_payload(
        event_type="ORDER_CANCELED",
        event_source="chain_test",
        idempotency_key="cancel-chain-1",
        intent_id="intent-cancel-1",
        payload={
            "symbol": "AAPL",
            "strategy_id": "mean_reversion_v2",
            "session_id": "regular_hours",
            "reason": "operator_cancel",
        },
    )
    store.append_oms_event_payload(
        event_type="ORDER_FILLED",
        event_source="chain_test",
        idempotency_key="fill-chain-1",
        intent_id="intent-fill-1",
        payload={
            "symbol": "AAPL",
            "strategy_id": "mean_reversion_v2",
            "session_id": "regular_hours",
            "fill_qty": 2,
            "fill_price": 100.8,
            "expected_price": 100.0,
        },
    )
    store.append_oms_event_payload(
        event_type="RECONCILE_UPDATE",
        event_source="chain_test",
        idempotency_key="parent-chain-1",
        intent_id="parent-1",
        payload={
            "record_type": "parent_execution_summary",
            "symbol": "AAPL",
            "strategy_id": "mean_reversion_v2",
            "session_id": "regular_hours",
            "requested_quantity": 10,
            "submitted_quantity": 9,
            "failed_slices": 1,
            "retry_count": 2,
            "cancel_replace_count": 1,
            "success_ratio": 0.9,
            "arrival_slippage_bps_mean": 11.0,
            "arrival_slippage_sample_count": 4,
        },
    )
    store.close()

    oms_event_tca = summarize_oms_event_tca(
        database_url=f"sqlite:///{db_path}",
        intent_store_path=str(db_path),
        limit=2000,
    )
    report_payload = {
        "oms_event_tca": {
            **oms_event_tca,
            "enabled": True,
            "available": True,
        },
    }
    latest_path = tmp_path / "runtime" / "runtime_performance_report_latest.json"
    engine._runtime_perf_report_last_persist_mono = 0.0
    monkeypatch.setattr(engine, "_runtime_performance_report_latest_path", lambda: latest_path)
    monkeypatch.setattr(engine, "_runtime_performance_report_min_interval_s", lambda: 0.0)
    engine._persist_runtime_performance_report_snapshot(
        report=report_payload,
        gate_passed=True,
        context={
            "failed_checks": [],
            "reason": "integration_chain_ok",
            "thresholds": {},
            "observed": {"event_tca_parent_execution_consistent": True},
        },
    )
    assert latest_path.exists()
    monkeypatch.setenv("AI_TRADING_RUNTIME_PERF_REPORT_LATEST_PATH", str(latest_path))

    snapshot = build_control_plane_snapshot()
    execution_quality = snapshot["execution_quality"]
    reject_reasons = execution_quality["submit_reject_reasons_top"]
    cancel_reasons = execution_quality["cancel_reasons_top"]
    decomposition = execution_quality["realized_slippage_decomposition"]
    outcomes_by_scope = execution_quality["event_outcomes_by_scope"]
    parent_scope = execution_quality["parent_execution_kpis_by_scope"]

    assert reject_reasons and reject_reasons[0]["reason"] == "insufficient_buying_power"
    assert cancel_reasons and cancel_reasons[0]["reason"] == "operator_cancel"
    assert decomposition["sample_count"] == 1
    assert decomposition["adverse_sample_count"] == 1
    assert outcomes_by_scope and outcomes_by_scope[0]["symbol"] == "AAPL"
    assert outcomes_by_scope[0]["strategy_id"] == "mean_reversion_v2"
    assert parent_scope and parent_scope[0]["session_id"] == "regular_hours"
