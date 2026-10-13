from types import SimpleNamespace
from datetime import timedelta

import pytest

from ai_trading.execution.engine import ExecutionEngine
from ai_trading.execution import engine as engine_module
from ai_trading.oms.intent_store import IntentStore


@pytest.mark.parametrize("status", ["pending_new", "accepted", "partially_filled", "filled", "canceled"])
def test_known_intent_reconciliation_uses_native_lookup_without_account_scan(tmp_path, status):
    store = IntentStore(path=str(tmp_path/"intents.db"))
    manager = ExecutionEngine().order_manager
    manager.configure_intent_store(store)
    intent, _ = store.create_intent(intent_id="known", idempotency_key="key", symbol="AMZN", side="buy", quantity=1, status="SUBMITTING")
    def unexpected_scan(**_):
        pytest.fail("known-intent reconciliation must not scan unrelated historical orders")
    calls = []
    def lookup(client_id):
        calls.append(client_id)
        return SimpleNamespace(id="broker-order", client_order_id=client_id, status=status,
                               symbol="AMZN", side="buy", qty=1, filled_qty=1 if status=="filled" else 0)
    summary = manager.reconcile_open_intents(list_orders_fn=unexpected_scan,
        get_order_by_id_fn=lambda _:None, get_order_by_client_order_id_fn=lookup)
    assert summary["intents_checked"] == 1 and summary["errors"] == 0
    assert calls == [intent.intent_id]
    after = store.get_intent(intent.intent_id)
    assert after is not None
    assert after.status == {"filled":"FILLED", "canceled":"CANCELED"}.get(status, "SUBMITTED")
    assert after.broker_order_id == "broker-order" or status in {"filled", "canceled"}


@pytest.mark.parametrize("age_days", [0, 2])
def test_failed_exact_lookup_preserves_ambiguous_claim_and_never_resubmits(tmp_path, monkeypatch, age_days):
    store = IntentStore(path=str(tmp_path/"intents.db"))
    manager = ExecutionEngine().order_manager
    manager.configure_intent_store(store)
    store.create_intent(intent_id="unknown", idempotency_key="key", symbol="AMZN", side="buy", quantity=1, status="SUBMITTING")
    now = engine_module.safe_utcnow()
    monkeypatch.setattr(engine_module, "safe_utcnow", lambda:now+timedelta(days=age_days))
    def lookup(_):
        raise TimeoutError("broker unavailable")
    summary = manager.reconcile_open_intents(list_orders_fn=lambda **_:pytest.fail("unrelated scan"),
        get_order_by_id_fn=lookup, get_order_by_client_order_id_fn=lookup)
    assert summary["errors"] > 0
    assert store.get_intent("unknown").status == "SUBMITTING"
    assert summary["marked_failed"] == 0
