from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pytest

from ai_trading.oms.event_store import EventStore
from ai_trading.oms.intent_store import IntentStore
from ai_trading.oms.invariants import evaluate_oms_lifecycle_parity_invariants
from ai_trading.oms.simulated_lifecycle import SimulatedLifecycleDriver


pytest.importorskip("sqlalchemy")


_TERMINAL_SCENARIOS: list[_TerminalScenario] = [
    {"name": "filled", "terminal_status": "FILLED", "fill_qty": 3.0, "fill_price": 101.25},
    {"name": "canceled", "terminal_status": "CANCELED", "fill_qty": 0.0, "fill_price": None},
    {"name": "rejected", "terminal_status": "REJECTED", "fill_qty": 0.0, "fill_price": None},
    {"name": "expired", "terminal_status": "EXPIRED", "fill_qty": 0.0, "fill_price": None},
]


class _TerminalScenario(TypedDict):
    name: str
    terminal_status: str
    fill_qty: float
    fill_price: float | None


def _event_type_stream(store: EventStore, intent_id: str) -> list[str]:
    rows = store.list_oms_events(intent_id=intent_id, limit=5000)
    return [str(row.get("event_type") or "").strip().upper() for row in rows]


def _run_live_scenario(
    *,
    db_path: Path,
    scenario_name: str,
    terminal_status: str,
    fill_qty: float,
    fill_price: float | None,
) -> str:
    store = IntentStore(path=str(db_path))
    intent_id = f"live-intent-{scenario_name}"
    record, created = store.create_intent(
        intent_id=intent_id,
        idempotency_key=f"live-intent-key-{scenario_name}",
        symbol="AAPL",
        side="buy",
        quantity=3.0,
        status="PENDING_SUBMIT",
    )
    assert created is True
    assert store.claim_for_submit(record.intent_id) is True
    store.mark_submitted(record.intent_id, f"live-broker-{scenario_name}")
    if fill_qty > 0.0:
        store.record_fill(
            record.intent_id,
            fill_qty=float(fill_qty),
            fill_price=fill_price,
        )
    store.close_intent(record.intent_id, final_status=terminal_status)
    store.close()
    return intent_id


def _run_simulated_scenario(
    *,
    db_path: Path,
    scenario_name: str,
    terminal_status: str,
    fill_qty: float,
    fill_price: float | None,
) -> str:
    lifecycle = SimulatedLifecycleDriver(
        enabled=True,
        source="backtest_engine",
        database_url=f"sqlite:///{db_path}",
        intent_store_path=str(db_path),
    )
    ref = lifecycle.open_submitted_intent(
        intent_id=f"bt-intent-{scenario_name}",
        idempotency_key=f"bt-intent-key-{scenario_name}",
        symbol="AAPL",
        side="buy",
        quantity=3.0,
        decision_ts="2025-01-01T00:00:00+00:00",
        broker_order_id=f"bt-broker-{scenario_name}",
        strategy_id="backtest_engine",
        metadata={"simulation": True},
    )
    assert ref is not None
    assert lifecycle.record_fill_and_close_intent(
        intent_id=ref.intent_id,
        fill_qty=float(fill_qty),
        fill_price=fill_price,
        fee=0.0,
        fill_ts="2025-01-01T00:00:01+00:00",
        terminal_status=terminal_status,
    )
    lifecycle.close()
    return str(ref.intent_id)


@pytest.mark.parametrize("scenario", _TERMINAL_SCENARIOS)
def test_simulated_lifecycle_stream_matches_live_intent_store_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scenario: _TerminalScenario,
) -> None:
    db_path = tmp_path / "oms_backtest_parity_stream.db"
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    scenario_name = str(scenario["name"])
    terminal_status = str(scenario["terminal_status"])
    fill_qty = float(scenario["fill_qty"])
    fill_price = scenario["fill_price"]
    live_intent_id = _run_live_scenario(
        db_path=db_path,
        scenario_name=scenario_name,
        terminal_status=terminal_status,
        fill_qty=fill_qty,
        fill_price=fill_price,
    )
    bt_intent_id = _run_simulated_scenario(
        db_path=db_path,
        scenario_name=scenario_name,
        terminal_status=terminal_status,
        fill_qty=fill_qty,
        fill_price=fill_price,
    )

    event_store = EventStore(url=f"sqlite:///{db_path}")
    live_stream = _event_type_stream(event_store, live_intent_id)
    backtest_stream = _event_type_stream(event_store, bt_intent_id)
    event_store.close()

    assert live_stream
    assert live_stream == backtest_stream


def test_lifecycle_parity_invariants_pass_for_live_and_simulated_intents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "oms_backtest_parity_invariants.db"
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    for scenario in _TERMINAL_SCENARIOS:
        scenario_name = str(scenario["name"])
        terminal_status = str(scenario["terminal_status"])
        fill_qty = float(scenario["fill_qty"])
        fill_price = scenario["fill_price"]
        _run_live_scenario(
            db_path=db_path,
            scenario_name=f"live-{scenario_name}",
            terminal_status=terminal_status,
            fill_qty=fill_qty,
            fill_price=fill_price,
        )
        _run_simulated_scenario(
            db_path=db_path,
            scenario_name=f"sim-{scenario_name}",
            terminal_status=terminal_status,
            fill_qty=fill_qty,
            fill_price=fill_price,
        )

    summary = evaluate_oms_lifecycle_parity_invariants(
        database_url=f"sqlite:///{db_path}",
        intent_store_path=str(db_path),
    )
    assert summary["available"] is True
    assert summary["ok"] is True
    assert int(summary["total_violations"]) == 0
