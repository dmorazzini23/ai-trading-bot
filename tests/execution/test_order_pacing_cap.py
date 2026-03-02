from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging
from types import SimpleNamespace

from ai_trading.execution import live_trading as lt


def _engine_stub() -> lt.ExecutionEngine:
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine.stats = {}
    engine._cycle_submitted_orders = 0
    engine._cycle_new_orders_submitted = 0
    engine._cycle_maintenance_actions = 0
    engine._cycle_order_outcomes = []
    engine._recent_order_intents = {}
    engine._pending_new_actions_this_cycle = 0
    engine._pending_new_policy_last_cycle_index = None
    engine._skip_last_logged_at = {}
    engine._skip_detail_last_logged_at = {}
    engine._engine_cycle_index = 1
    engine.marketable_limit_slippage_bps = 10
    engine._capacity_broker = lambda client: client
    engine._open_order_qty_index = {}
    engine._pending_orders = {}
    engine._broker_sync = None
    engine._last_submit_outcome = {}
    return engine


def test_pending_new_policy_counts_maintenance_not_new_orders(monkeypatch) -> None:
    engine = _engine_stub()
    now_dt = datetime.now(UTC)
    stale_order = SimpleNamespace(
        id="ord-maint",
        symbol="AAPL",
        side="buy",
        qty="1",
        status="pending_new",
        created_at=now_dt - timedelta(seconds=120),
    )
    engine.trading_client = SimpleNamespace(list_orders=lambda status="open": [stale_order])
    canceled: list[str] = []
    engine._cancel_order_alpaca = lambda order_id: canceled.append(str(order_id))
    engine._replace_limit_order_with_marketable = lambda **_: None

    monkeypatch.setenv("AI_TRADING_PENDING_NEW_POLICY", "cancel")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_TIMEOUT_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_MAX_ACTIONS_PER_CYCLE", "2")

    engine._apply_pending_new_timeout_policy()

    assert canceled == ["ord-maint"]
    assert engine._cycle_maintenance_actions == 1
    assert engine._cycle_new_orders_submitted == 0
    assert engine._cycle_submitted_orders == 0


def test_order_pacing_cap_skip_detail_includes_cap_context(monkeypatch, caplog) -> None:
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_ORDER_SKIP_LOG_TTL_SEC", "0")
    monkeypatch.setenv("AI_TRADING_ORDER_SKIP_DETAIL_LOG_TTL_SEC", "0")

    caplog.set_level(logging.INFO, logger=lt.logger.name)
    engine._skip_submit(
        symbol="AAPL",
        side="buy",
        reason="order_pacing_cap",
        detail="max_new_orders_per_cycle reached source=configured",
        context={
            "cap_type": "new_orders",
            "limit": 2,
            "used": 2,
            "headroom": 0,
        },
    )

    details = [
        rec
        for rec in caplog.records
        if str(rec.message).startswith("ORDER_SUBMIT_SKIPPED_DETAIL")
    ]
    assert details
    latest = details[-1]
    assert getattr(latest, "reason", None) == "order_pacing_cap"
    assert getattr(latest, "context", {}).get("cap_type") == "new_orders"
    assert getattr(latest, "context", {}).get("limit") == 2
    assert getattr(latest, "context", {}).get("used") == 2
    assert getattr(latest, "context", {}).get("headroom") == 0
