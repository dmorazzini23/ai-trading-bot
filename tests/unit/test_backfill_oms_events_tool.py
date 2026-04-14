from __future__ import annotations

from pathlib import Path

import pytest

from ai_trading.oms.event_store import EventStore
from ai_trading.oms.intent_store import IntentStore
from ai_trading.tools.backfill_oms_events import backfill_oms_events


pytest.importorskip("sqlalchemy")


def test_backfill_oms_events_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "oms_backfill.db"
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    store = IntentStore(path=str(db_path))
    pending, created_pending = store.create_intent(
        intent_id="intent-pending-1",
        idempotency_key="pending-key-1",
        symbol="MSFT",
        side="buy",
        quantity=3.0,
        status="PENDING_SUBMIT",
    )
    terminal, created_terminal = store.create_intent(
        intent_id="intent-terminal-1",
        idempotency_key="terminal-key-1",
        symbol="NVDA",
        side="sell",
        quantity=2.0,
        status="PENDING_SUBMIT",
    )
    assert created_pending is True
    assert created_terminal is True
    assert store.claim_for_submit(terminal.intent_id) is True
    store.mark_submitted(terminal.intent_id, "broker-terminal-1")
    store.record_fill(terminal.intent_id, fill_qty=2.0, fill_price=910.25, fee=0.2)
    store.close_intent(terminal.intent_id, final_status="FILLED")
    store.close()

    dry_run = backfill_oms_events(
        database_url=f"sqlite:///{db_path}",
        intent_store_path=str(db_path),
        dry_run=True,
    )
    assert dry_run["ok"] is True
    assert dry_run["dry_run"] is True
    assert dry_run["generated_events"] > 0
    assert dry_run["inserted_events"] == 0

    first_run = backfill_oms_events(
        database_url=f"sqlite:///{db_path}",
        intent_store_path=str(db_path),
        dry_run=False,
    )
    assert first_run["ok"] is True
    assert first_run["inserted_events"] > 0

    second_run = backfill_oms_events(
        database_url=f"sqlite:///{db_path}",
        intent_store_path=str(db_path),
        dry_run=False,
    )
    assert second_run["ok"] is True
    assert second_run["inserted_events"] == 0
    assert second_run["duplicate_events"] == second_run["generated_events"]

    event_store = EventStore(url=f"sqlite:///{db_path}")
    events = event_store.list_oms_events()
    event_store.close()
    assert len(events) == first_run["generated_events"]
