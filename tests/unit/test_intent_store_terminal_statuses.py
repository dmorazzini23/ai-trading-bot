from __future__ import annotations

import pytest

from ai_trading.oms.intent_store import IntentStore
from ai_trading.oms.statuses import TERMINAL_INTENT_STATUSES

pytest.importorskip("sqlalchemy")


def test_terminal_statuses_include_failed_and_expired() -> None:
    assert "FAILED" in TERMINAL_INTENT_STATUSES
    assert "EXPIRED" in TERMINAL_INTENT_STATUSES


def test_get_open_intents_excludes_failed_and_expired(tmp_path) -> None:
    store = IntentStore(path=str(tmp_path / "oms_terminal_statuses.db"))

    failed_intent, created_failed = store.create_intent(
        intent_id="intent-failed",
        idempotency_key="terminal-failed",
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        status="PENDING_SUBMIT",
    )
    expired_intent, created_expired = store.create_intent(
        intent_id="intent-expired",
        idempotency_key="terminal-expired",
        symbol="MSFT",
        side="sell",
        quantity=2.0,
        status="PENDING_SUBMIT",
    )

    assert created_failed is True
    assert created_expired is True

    store.close_intent(failed_intent.intent_id, final_status="FAILED")
    store.close_intent(expired_intent.intent_id, final_status="EXPIRED")

    open_intent_ids = {record.intent_id for record in store.get_open_intents()}
    assert failed_intent.intent_id not in open_intent_ids
    assert expired_intent.intent_id not in open_intent_ids


def test_close_intent_rejects_non_terminal_status(tmp_path) -> None:
    store = IntentStore(path=str(tmp_path / "oms_non_terminal_guard.db"))
    record, created = store.create_intent(
        intent_id="intent-open",
        idempotency_key="intent-open-key",
        symbol="NVDA",
        side="buy",
        quantity=1.0,
        status="PENDING_SUBMIT",
    )
    assert created is True

    with pytest.raises(ValueError, match="terminal status"):
        store.close_intent(record.intent_id, final_status="SUBMITTED")
