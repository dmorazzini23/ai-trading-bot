from __future__ import annotations

from pathlib import Path

import pytest

from ai_trading.oms.intent_store import IntentStore
from ai_trading.oms.invariants import evaluate_oms_reconciliation_invariants


pytest.importorskip("sqlalchemy")


def test_oms_invariants_pass_when_lifecycle_events_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "oms_invariants_ok.db"
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    store = IntentStore(path=str(db_path))
    record, created = store.create_intent(
        intent_id="intent-ok-1",
        idempotency_key="intent-ok-key-1",
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        status="PENDING_SUBMIT",
    )
    assert created is True
    assert store.claim_for_submit(record.intent_id) is True
    store.mark_submitted(record.intent_id, "broker-ok-1")
    store.record_fill(record.intent_id, fill_qty=1.0, fill_price=100.25)
    store.close_intent(record.intent_id, final_status="FILLED")
    store.close()

    summary = evaluate_oms_reconciliation_invariants(
        database_url=f"sqlite:///{db_path}",
        intent_store_path=str(db_path),
    )
    assert summary["available"] is True
    assert summary["ok"] is True
    assert summary["total_violations"] == 0


def test_oms_invariants_fail_when_events_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "oms_invariants_missing.db"
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    store = IntentStore(path=str(db_path))
    _record, created = store.create_intent(
        intent_id="intent-missing-1",
        idempotency_key="intent-missing-key-1",
        symbol="MSFT",
        side="sell",
        quantity=2.0,
        status="PENDING_SUBMIT",
    )
    assert created is True
    store.close()

    summary = evaluate_oms_reconciliation_invariants(
        database_url=f"sqlite:///{db_path}",
        intent_store_path=str(db_path),
    )
    assert summary["available"] is True
    assert summary["ok"] is False
    violations = summary["violations"]
    assert int(violations["missing_intent_created"]) >= 1
