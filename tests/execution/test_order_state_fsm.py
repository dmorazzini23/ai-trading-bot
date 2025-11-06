import logging
from collections import defaultdict
from types import SimpleNamespace

import pytest

from ai_trading.execution import live_trading


class SequencedClient:
    def __init__(self, statuses: list[str]) -> None:
        self._statuses = list(statuses)

    def get_order_by_client_order_id(self, client_order_id: str) -> SimpleNamespace:
        status = self._statuses.pop(0) if self._statuses else "rejected"
        return SimpleNamespace(
            status=status,
            id="order-1",
            client_order_id=client_order_id,
            symbol="AAPL",
            qty=5,
            filled_qty=0,
        )


class FSMEngine(live_trading.LiveTradingExecutionEngine):
    def __init__(self, statuses: list[str]) -> None:
        super().__init__(ctx=None)
        self.is_initialized = True
        self.shadow_mode = False
        self.stats = defaultdict(float)
        self._cycle_account = {
            "pattern_day_trader": False,
            "daytrade_limit": 3,
            "daytrade_count": 0,
            "shorting_enabled": True,
            "margin_enabled": True,
        }
        self.trading_client = SequencedClient(statuses)
        self._pending_order_kwargs = {}

    def _refresh_settings(self) -> None:
        return None

    def _ensure_initialized(self) -> bool:
        return True

    def _pre_execution_order_checks(self, _order: dict | None) -> bool:
        return True

    def _pre_execution_checks(self) -> bool:
        return True

    def _execute_with_retry(self, submit_fn, order: dict):
        return {
            "status": "accepted",
            "id": "order-1",
            "client_order_id": order.get("client_order_id"),
            "symbol": order.get("symbol"),
            "qty": order.get("quantity"),
        }

    def _get_account_snapshot(self):
        return dict(self._cycle_account)


@pytest.fixture(autouse=True)
def _patch_fsm_guards(monkeypatch) -> None:
    monkeypatch.setattr(live_trading, "_safe_mode_guard", lambda *a, **k: False)
    monkeypatch.setattr(live_trading, "_require_bid_ask_quotes", lambda: False)
    monkeypatch.setattr(live_trading, "quote_fresh_enough", lambda *a, **k: True)
    monkeypatch.setattr(live_trading, "guard_shadow_active", lambda: False)
    monkeypatch.setattr(live_trading, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(
        live_trading,
        "_call_preflight_capacity",
        lambda *a, **k: live_trading.CapacityCheck(True, int(a[3]), None),
    )
    monkeypatch.setattr(
        live_trading.provider_monitor,
        "is_disabled",
        lambda *_a, **_k: False,
    )
    config = type(
        "Cfg",
        (),
        {
            "min_quote_freshness_ms": 1500,
            "degraded_feed_mode": "widen",
            "degraded_feed_limit_widen_bps": 0,
            "execution_require_realtime_nbbo": False,
            "execution_market_on_degraded": True,
        },
    )()
    monkeypatch.setattr(live_trading, "get_trading_config", lambda: config)


def test_ack_pending_cancel_reject_sequence(monkeypatch, caplog) -> None:
    engine = FSMEngine(["accepted", "pending_cancel", "rejected"])
    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_: False)
    caplog.set_level(logging.INFO, logger="ai_trading.execution.live_trading")

    result = engine.execute_order("AAPL", "buy", 5, order_type="market")
    assert result is not None

    submitted_logs = [rec for rec in caplog.records if rec.msg == "ORDER_SUBMITTED"]
    assert len(submitted_logs) == 1
    submitted = submitted_logs[0]
    assert submitted.event_seq == 1
    assert submitted.prev_status is None
    assert submitted.new_status == "accepted"

    rejected_logs = [rec for rec in caplog.records if rec.msg == "ORDER_REJECTED"]
    assert rejected_logs, "ORDER_REJECTED log missing"
    rejected = rejected_logs[-1]
    assert rejected.event_seq > submitted.event_seq
    assert rejected.prev_status == "pending_cancel"
    assert rejected.new_status == "rejected"
