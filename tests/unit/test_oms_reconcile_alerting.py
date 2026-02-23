from __future__ import annotations

import logging
from types import SimpleNamespace

from ai_trading.execution.engine import OrderManager


class _Store:
    def __init__(self, intents):
        self._intents = intents

    def get_open_intents(self):
        return list(self._intents)


def test_reconcile_alerts_when_open_intents_exceed_threshold(monkeypatch, caplog) -> None:
    intents = [
        SimpleNamespace(
            intent_id="intent-1",
            status="PENDING_SUBMIT",
            broker_order_id=None,
            updated_at=None,
        ),
        SimpleNamespace(
            intent_id="intent-2",
            status="PENDING_SUBMIT",
            broker_order_id=None,
            updated_at=None,
        ),
        SimpleNamespace(
            intent_id="intent-3",
            status="PENDING_SUBMIT",
            broker_order_id=None,
            updated_at=None,
        ),
    ]
    manager = OrderManager.__new__(OrderManager)
    manager._intent_store = _Store(intents)

    monkeypatch.setenv("AI_TRADING_OMS_OPEN_INTENT_ALERT_THRESHOLD", "2")
    caplog.set_level(logging.WARNING, logger="ai_trading.execution.engine")

    summary = manager.reconcile_open_intents(broker_orders=[])

    assert summary["intents_checked"] == 3
    assert summary["pending_submit"] == 3
    alerts = [
        record
        for record in caplog.records
        if record.msg == "OMS_OPEN_INTENT_THRESHOLD_EXCEEDED"
    ]
    assert alerts
    alert = alerts[0]
    assert getattr(alert, "open_intents", None) == 3
    assert getattr(alert, "threshold", None) == 2
