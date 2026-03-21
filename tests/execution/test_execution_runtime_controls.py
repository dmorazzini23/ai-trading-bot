from __future__ import annotations

from collections import deque
import json
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.execution import live_trading as lt


def _engine_stub() -> Any:
    engine: Any = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine.stats = {}
    engine._cycle_submitted_orders = 0
    engine._cycle_new_orders_submitted = 0
    engine._cycle_maintenance_actions = 0
    engine._cycle_order_outcomes = []
    engine._recent_order_intents = {}
    engine._pending_new_actions_this_cycle = 0
    engine._pending_new_policy_last_cycle_index = None
    engine._pending_new_replace_last_mono = {}
    engine._pending_new_replacements_this_cycle = 0
    engine.marketable_limit_slippage_bps = 10
    engine._capacity_broker = lambda client: client
    engine._open_order_qty_index = {}
    engine._position_tracker = {}
    engine._position_tracker_last_sync_mono = 0.0
    engine._pending_orders = {}
    engine._cancel_ratio_adaptive_new_orders_cap = None
    engine._cancel_ratio_adaptive_context = {}
    engine._pacing_relax_new_orders_cap = None
    engine._broker_sync = None
    engine._last_submit_outcome = {}
    engine._last_order_ack_timeout_mono = 0.0
    engine._last_order_ack_timeout_order_id = None
    engine._last_order_ack_timeout_client_order_id = None
    engine._symbol_loss_streak = {}
    engine._symbol_loss_cooldown_until = {}
    engine._symbol_reentry_cooldown_until = {}
    engine._opening_provider_ready_since_mono = 0.0
    engine._symbol_slippage_budget_cache_until_mono = 0.0
    engine._symbol_slippage_budget_cache = {}
    engine._markout_feedback_bps = deque(maxlen=512)
    engine._slippage_feedback_bps = deque(maxlen=512)
    engine._markout_feedback_last_context = {
        "sample_count": 0,
        "mean_bps": 0.0,
        "toxic": False,
        "threshold_bps": -4.0,
        "min_samples": 12,
    }
    engine._tca_fill_backfill_last_run_mono = 0.0
    engine._tca_fill_backfill_offset = 0
    engine._tca_fill_backfill_bootstrapped = False
    return engine


def test_pending_new_timeout_policy_cancel(monkeypatch, caplog):
    engine = _engine_stub()
    now_dt = datetime.now(UTC)
    stale_order = SimpleNamespace(
        id="ord-1",
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

    caplog.set_level(logging.WARNING)
    applied = engine._apply_pending_new_timeout_policy()

    assert canceled == ["ord-1"]
    assert applied is True
    assert engine._pending_new_actions_this_cycle == 1
    assert engine._cycle_maintenance_actions == 1
    assert engine._cycle_new_orders_submitted == 0
    assert any(record.message == "PENDING_NEW_TIMEOUT_ACTION" for record in caplog.records)


def test_pending_new_timeout_policy_idempotent_per_cycle(monkeypatch):
    engine = _engine_stub()
    now_dt = datetime.now(UTC)
    stale_order = SimpleNamespace(
        id="ord-1",
        symbol="AAPL",
        side="buy",
        qty="1",
        status="pending_new",
        created_at=now_dt - timedelta(seconds=120),
    )
    engine._engine_cycle_index = 9
    engine.trading_client = SimpleNamespace(list_orders=lambda status="open": [stale_order])
    canceled: list[str] = []
    engine._cancel_order_alpaca = lambda order_id: canceled.append(str(order_id))
    engine._replace_limit_order_with_marketable = lambda **_: None

    monkeypatch.setenv("AI_TRADING_PENDING_NEW_POLICY", "cancel")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_TIMEOUT_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_MAX_ACTIONS_PER_CYCLE", "2")

    first_applied = engine._apply_pending_new_timeout_policy()
    second_applied = engine._apply_pending_new_timeout_policy()

    assert canceled == ["ord-1"]
    assert first_applied is True
    assert second_applied is False

    engine._reset_pending_new_policy_state_for_tests()
    third_applied = engine._apply_pending_new_timeout_policy()

    assert canceled == ["ord-1", "ord-1"]
    assert third_applied is True


def test_pending_new_timeout_policy_ladder_replaces_then_cancels(monkeypatch):
    engine = _engine_stub()
    now_dt = datetime.now(UTC)
    stale_order = SimpleNamespace(
        id="ord-ladder-1",
        symbol="AAPL",
        side="buy",
        qty="2",
        status="pending_new",
        type="limit",
        limit_price="100.0",
        client_order_id="cid-ladder-1",
        created_at=now_dt - timedelta(seconds=120),
    )
    engine.trading_client = SimpleNamespace(list_orders=lambda status="open": [stale_order])
    canceled: list[str] = []
    replaced: list[dict[str, Any]] = []
    engine._cancel_order_alpaca = lambda order_id: canceled.append(str(order_id))

    def _replace(**kwargs):
        replaced.append(dict(kwargs))
        return {"id": "ord-repl-1"}

    engine._replace_limit_order_with_marketable = _replace
    engine._engine_cycle_index = 1

    monkeypatch.setenv("AI_TRADING_PENDING_NEW_POLICY", "ladder")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_TIMEOUT_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_HARD_TIMEOUT_SEC", "600")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_MAX_ACTIONS_PER_CYCLE", "2")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_LADDER_MAX_REPLACEMENTS", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_LADDER_WIDEN_STEP_BPS", "5")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_LADDER_CANCEL_AFTER_MAX_REPLACEMENTS", "1")

    first_applied = engine._apply_pending_new_timeout_policy()
    engine._engine_cycle_index = 2
    second_applied = engine._apply_pending_new_timeout_policy()

    assert first_applied is True
    assert second_applied is True
    assert len(replaced) == 1
    assert canceled == ["ord-ladder-1"]


def test_pending_new_timeout_policy_hard_timeout_forces_cancel(monkeypatch):
    engine = _engine_stub()
    now_dt = datetime.now(UTC)
    stale_order = SimpleNamespace(
        id="ord-hard-1",
        symbol="AAPL",
        side="buy",
        qty="1",
        status="pending_new",
        type="limit",
        limit_price="101.0",
        client_order_id="cid-hard-1",
        created_at=now_dt - timedelta(seconds=180),
    )
    engine.trading_client = SimpleNamespace(list_orders=lambda status="open": [stale_order])
    canceled: list[str] = []
    replaced: list[dict[str, Any]] = []
    engine._cancel_order_alpaca = lambda order_id: canceled.append(str(order_id))
    engine._replace_limit_order_with_marketable = lambda **kwargs: replaced.append(dict(kwargs))
    engine._engine_cycle_index = 1

    monkeypatch.setenv("AI_TRADING_PENDING_NEW_POLICY", "ladder")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_TIMEOUT_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_HARD_TIMEOUT_SEC", "60")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_MAX_ACTIONS_PER_CYCLE", "2")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_LADDER_MAX_REPLACEMENTS", "3")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_LADDER_CANCEL_AFTER_MAX_REPLACEMENTS", "1")

    applied = engine._apply_pending_new_timeout_policy()

    assert applied is True
    assert canceled == ["ord-hard-1"]
    assert replaced == []


def test_pending_new_timeout_policy_replace_churn_guard_blocks_rapid_replace(monkeypatch):
    engine = _engine_stub()
    now_dt = datetime.now(UTC)
    stale_order = SimpleNamespace(
        id="ord-guard-1",
        symbol="AAPL",
        side="buy",
        qty="2",
        status="pending_new",
        type="limit",
        limit_price="100.0",
        client_order_id="cid-guard-1",
        created_at=now_dt - timedelta(seconds=120),
    )
    engine.trading_client = SimpleNamespace(list_orders=lambda status="open": [stale_order])
    replaced: list[dict[str, Any]] = []
    canceled: list[str] = []
    monotonic_clock = {"value": 100.0}
    monkeypatch.setattr(lt, "monotonic_time", lambda: monotonic_clock["value"])

    def _replace(**kwargs):
        replaced.append(dict(kwargs))
        return {"id": f"ord-repl-{len(replaced)}"}

    engine._replace_limit_order_with_marketable = _replace
    engine._cancel_order_alpaca = lambda order_id: canceled.append(str(order_id))

    monkeypatch.setenv("AI_TRADING_PENDING_NEW_POLICY", "ladder")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_TIMEOUT_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_HARD_TIMEOUT_SEC", "600")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_MAX_ACTIONS_PER_CYCLE", "2")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_LADDER_MAX_REPLACEMENTS", "2")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_REPLACE_MAX_PER_CYCLE", "2")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_REPLACE_MIN_INTERVAL_SEC", "120")

    engine._engine_cycle_index = 1
    first_applied = engine._apply_pending_new_timeout_policy()
    engine._engine_cycle_index = 2
    monotonic_clock["value"] = 130.0
    second_applied = engine._apply_pending_new_timeout_policy()

    assert first_applied is True
    assert second_applied is False
    assert len(replaced) == 1
    assert canceled == []


def test_pending_new_timeout_policy_applies_dynamic_replace_controls(monkeypatch):
    engine = _engine_stub()
    now_dt = datetime.now(UTC)
    stale_order = SimpleNamespace(
        id="ord-dyn-1",
        symbol="AAPL",
        side="buy",
        qty="2",
        status="pending_new",
        type="limit",
        limit_price="100.0",
        client_order_id="cid-dyn-1",
        created_at=now_dt - timedelta(seconds=120),
    )
    engine.trading_client = SimpleNamespace(list_orders=lambda status="open": [stale_order])
    replaced: list[dict[str, Any]] = []
    engine._cancel_order_alpaca = lambda *_args, **_kwargs: None
    engine._replace_limit_order_with_marketable = (
        lambda **kwargs: replaced.append(dict(kwargs)) or {"id": "ord-dyn-repl"}
    )
    monkeypatch.setattr(
        engine,
        "_pending_new_dynamic_controls",
        lambda **_: (
            2,
            0.0,
            2,
            {"enabled": True, "stress": 0.75},
        ),
    )

    monkeypatch.setenv("AI_TRADING_PENDING_NEW_POLICY", "ladder")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_TIMEOUT_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_HARD_TIMEOUT_SEC", "600")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_MAX_ACTIONS_PER_CYCLE", "2")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_LADDER_MAX_REPLACEMENTS", "2")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_LADDER_WIDEN_STEP_BPS", "5")

    engine._engine_cycle_index = 3
    applied = engine._apply_pending_new_timeout_policy()

    assert applied is True
    assert replaced
    assert replaced[0]["slippage_bps"] == 2


def test_pending_new_timeout_policy_defaults_to_cancel(monkeypatch):
    engine = _engine_stub()
    now_dt = datetime.now(UTC)
    stale_order = SimpleNamespace(
        id="ord-default",
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

    monkeypatch.delenv("AI_TRADING_PENDING_NEW_POLICY", raising=False)
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_TIMEOUT_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_MAX_ACTIONS_PER_CYCLE", "2")

    applied = engine._apply_pending_new_timeout_policy()

    assert applied is True
    assert canceled == ["ord-default"]


def test_pending_new_timeout_policy_returns_false_without_stale_orders(monkeypatch):
    engine = _engine_stub()
    now_dt = datetime.now(UTC)
    fresh_order = SimpleNamespace(
        id="ord-fresh",
        symbol="AAPL",
        side="buy",
        qty="1",
        status="pending_new",
        created_at=now_dt - timedelta(seconds=5),
    )
    engine.trading_client = SimpleNamespace(list_orders=lambda status="open": [fresh_order])
    canceled: list[str] = []
    engine._cancel_order_alpaca = lambda order_id: canceled.append(str(order_id))
    engine._replace_limit_order_with_marketable = lambda **_: None

    monkeypatch.setenv("AI_TRADING_PENDING_NEW_POLICY", "cancel")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_TIMEOUT_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_MAX_ACTIONS_PER_CYCLE", "2")

    applied = engine._apply_pending_new_timeout_policy()

    assert applied is False
    assert canceled == []


def test_retry_capacity_precheck_with_fresh_account_enables_submit(monkeypatch):
    engine = _engine_stub()
    original_account = SimpleNamespace(tag="orig")
    refreshed_account = SimpleNamespace(tag="fresh")
    initial = lt.CapacityCheck(
        can_submit=False,
        suggested_qty=2,
        reason="insufficient_buying_power",
    )
    retry = lt.CapacityCheck(
        can_submit=True,
        suggested_qty=2,
        reason=None,
    )
    monkeypatch.delenv("AI_TRADING_CAPACITY_REFRESH_RETRY_ENABLED", raising=False)
    engine._refresh_cycle_account = lambda: refreshed_account
    engine._account_with_cycle_capacity_reservation = (
        lambda account, *, side, closing_position: account
    )
    monkeypatch.setattr(lt, "_call_preflight_capacity", lambda *args, **kwargs: retry)

    resolved, resolved_account = engine._retry_capacity_precheck_with_fresh_account(
        capacity=initial,
        symbol="AAPL",
        side="buy",
        price_hint=123.45,
        quantity=2,
        broker=object(),
        account_snapshot=original_account,
        closing_position=False,
    )

    assert resolved.can_submit is True
    assert resolved.reason is None
    assert resolved_account is refreshed_account


def test_retry_capacity_precheck_with_fresh_account_skips_non_buying_power_reason(monkeypatch):
    engine = _engine_stub()
    account = SimpleNamespace(tag="orig")
    initial = lt.CapacityCheck(
        can_submit=False,
        suggested_qty=1,
        reason="rate_throttle_block",
    )
    engine._refresh_cycle_account = lambda: pytest.fail("unexpected refresh call")

    resolved, resolved_account = engine._retry_capacity_precheck_with_fresh_account(
        capacity=initial,
        symbol="AAPL",
        side="buy",
        price_hint=123.45,
        quantity=1,
        broker=object(),
        account_snapshot=account,
        closing_position=False,
    )

    assert resolved is initial
    assert resolved_account is account


def test_duplicate_intent_window(monkeypatch):
    engine = _engine_stub()
    clock = {"value": 100.0}
    monkeypatch.setenv("AI_TRADING_INTENT_BLOCK_WHEN_OPEN_ORDER", "0")
    monkeypatch.setenv("AI_TRADING_DUPLICATE_INTENT_WINDOW_SEC", "60")
    monkeypatch.setattr(lt, "monotonic_time", lambda: clock["value"])

    engine._record_order_intent("AAPL", "buy")

    clock["value"] = 130.0
    assert engine._should_suppress_duplicate_intent("AAPL", "buy") is True

    clock["value"] = 200.0
    assert engine._should_suppress_duplicate_intent("AAPL", "buy") is False


def test_duplicate_intent_suppressed_when_open_order_present(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_INTENT_BLOCK_WHEN_OPEN_ORDER", "1")
    monkeypatch.setenv("AI_TRADING_DUPLICATE_INTENT_WINDOW_SEC", "0")
    engine._open_order_qty_index = {"AAPL": (2.0, 0.0)}

    assert engine._should_suppress_duplicate_intent("AAPL", "buy") is True


def test_symbol_reentry_cooldown_blocks_same_side_then_expires(monkeypatch):
    engine = _engine_stub()
    clock = {"value": 100.0}
    monkeypatch.setattr(lt, "monotonic_time", lambda: clock["value"])
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_REENTRY_COOLDOWN_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_REENTRY_COOLDOWN_SEC", "120")

    engine._arm_symbol_reentry_cooldown_from_fill(symbol="AAPL", side="buy")
    allowed_before, _ = engine._symbol_reentry_cooldown_allows_opening(symbol="AAPL", side="sell")
    allowed_during, context_during = engine._symbol_reentry_cooldown_allows_opening(symbol="AAPL", side="buy")

    assert allowed_before is True
    assert allowed_during is False
    assert context_during["reason"] == "symbol_reentry_cooldown"

    clock["value"] = 260.0
    allowed_after, context_after = engine._symbol_reentry_cooldown_allows_opening(symbol="AAPL", side="buy")
    assert allowed_after is True
    assert context_after["reason"] == "cooldown_inactive"


def test_cycle_intent_reservation_dedupes_symbol_side():
    engine = _engine_stub()

    assert engine._reserve_cycle_intent("AAPL", "buy") is True
    assert engine._reserve_cycle_intent("AAPL", "buy") is False
    assert engine._reserve_cycle_intent("AAPL", "sell") is True


def test_order_pacing_cap_warning_cooldown(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_CAP_LOG_COOLDOWN_SEC", "300")

    assert engine._should_emit_order_pacing_cap_log(now_ts=100.0) is True
    assert engine._should_emit_order_pacing_cap_log(now_ts=200.0) is False
    assert engine._should_emit_order_pacing_cap_log(now_ts=401.0) is True


def test_order_pacing_cap_warning_cooldown_zero_disables(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_CAP_LOG_COOLDOWN_SEC", "0")

    assert engine._should_emit_order_pacing_cap_log(now_ts=100.0) is True
    assert engine._should_emit_order_pacing_cap_log(now_ts=101.0) is True


def test_order_pacing_cap_log_level_warmup(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_WARMUP_MODE", "1")
    assert engine._order_pacing_cap_log_level() == "info"


def test_order_pacing_cap_log_level_runtime(monkeypatch):
    engine = _engine_stub()
    monkeypatch.delenv("AI_TRADING_WARMUP_MODE", raising=False)
    assert engine._order_pacing_cap_log_level() == "warning"


def test_execution_phase_gate_blocks_open_orders(monkeypatch):
    engine = _engine_stub()
    engine.ctx = SimpleNamespace(state={"service_phase": "reconcile"})
    monkeypatch.setenv("AI_TRADING_EXECUTION_PHASE_GATE_ENABLED", "1")

    allowed, detail = engine._execution_phase_allows_submits(closing_position=False)

    assert allowed is False
    assert detail == "phase=reconcile"


def test_execution_phase_gate_allows_closing_orders(monkeypatch):
    engine = _engine_stub()
    engine.ctx = SimpleNamespace(state={"service_phase": "reconcile"})
    monkeypatch.setenv("AI_TRADING_EXECUTION_PHASE_GATE_ENABLED", "1")

    allowed, detail = engine._execution_phase_allows_submits(closing_position=True)

    assert allowed is True
    assert detail is None


def test_opening_provider_guard_blocks_openings_when_provider_degraded(monkeypatch):
    engine = _engine_stub()
    engine.ctx = SimpleNamespace(state={"service_phase": "active"})
    monkeypatch.setenv("AI_TRADING_EXECUTION_PHASE_GATE_ENABLED", "1")
    monkeypatch.setattr(engine, "_opening_provider_guard_enabled", lambda: True)
    monkeypatch.setattr(
        engine,
        "_opening_provider_guard_elapsed_seconds",
        lambda: 120.0,
    )
    monkeypatch.setattr(
        lt.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "status": "degraded",
            "active": "yahoo",
            "using_backup": True,
            "safe_mode": False,
        },
    )
    monkeypatch.setattr(lt.provider_monitor, "is_disabled", lambda _provider: False)
    monkeypatch.setattr(lt, "is_safe_mode_active", lambda: False)

    allowed, detail = engine._execution_phase_allows_submits(closing_position=False)

    assert allowed is False
    assert detail is not None
    assert "opening_provider_guard" in detail


def test_opening_provider_guard_allows_openings_when_provider_healthy(monkeypatch):
    engine = _engine_stub()
    engine.ctx = SimpleNamespace(state={"service_phase": "active"})
    monkeypatch.setenv("AI_TRADING_EXECUTION_PHASE_GATE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_OPENING_PROVIDER_READY_MIN_SEC", "0")
    monkeypatch.setattr(engine, "_opening_provider_guard_enabled", lambda: True)
    monkeypatch.setattr(
        engine,
        "_opening_provider_guard_elapsed_seconds",
        lambda: 120.0,
    )
    monkeypatch.setattr(
        lt.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "status": "healthy",
            "active": "alpaca",
            "using_backup": False,
            "safe_mode": False,
        },
    )
    monkeypatch.setattr(lt.provider_monitor, "is_disabled", lambda _provider: False)
    monkeypatch.setattr(lt, "is_safe_mode_active", lambda: False)

    allowed, detail = engine._execution_phase_allows_submits(closing_position=False)

    assert allowed is True
    assert detail is None


def test_opening_provider_guard_requires_readiness_warmup(monkeypatch):
    engine = _engine_stub()
    engine.ctx = SimpleNamespace(state={"service_phase": "active"})
    monotonic_clock = {"value": 100.0}
    monkeypatch.setattr(lt, "monotonic_time", lambda: monotonic_clock["value"])
    monkeypatch.setenv("AI_TRADING_EXECUTION_PHASE_GATE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_OPENING_PROVIDER_READY_MIN_SEC", "30")
    monkeypatch.setattr(engine, "_opening_provider_guard_enabled", lambda: True)
    monkeypatch.setattr(engine, "_opening_provider_guard_elapsed_seconds", lambda: 120.0)
    monkeypatch.setattr(
        lt.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "status": "healthy",
            "active": "alpaca",
            "using_backup": False,
            "safe_mode": False,
            "quote_fresh_ms": 300.0,
            "updated": datetime.now(UTC).isoformat(),
        },
    )
    monkeypatch.setattr(lt.provider_monitor, "is_disabled", lambda _provider: False)
    monkeypatch.setattr(lt, "is_safe_mode_active", lambda: False)

    allowed_first, detail_first = engine._execution_phase_allows_submits(closing_position=False)
    monotonic_clock["value"] = 140.0
    allowed_second, detail_second = engine._execution_phase_allows_submits(closing_position=False)

    assert allowed_first is False
    assert detail_first is not None and "provider_readiness_warmup" in detail_first
    assert allowed_second is True
    assert detail_second is None


def test_pre_execution_order_checks_blocks_openings_when_exposure_overloaded(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_BLOCK_OPENINGS", "1")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_BP_MIN_RATIO", "0.03")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_MAX_GROSS_TO_EQUITY", "1.0")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_MAX_NET_TO_EQUITY", "0.35")
    monkeypatch.setattr(
        engine,
        "_enforce_opposite_side_policy",
        lambda *_args, **_kwargs: (True, None),
    )
    monkeypatch.setattr(
        engine,
        "_evaluate_pdt_preflight",
        lambda *_args, **_kwargs: (False, None, {}),
    )
    order = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 5,
        "client_order_id": "cid-1",
        "closing_position": False,
        "account_snapshot": {
            "equity": 100000,
            "buying_power": 500,
            "long_market_value": 150000,
            "short_market_value": 80000,
        },
    }

    allowed = engine._pre_execution_order_checks(order)

    assert allowed is False
    assert engine.stats["capacity_skips"] == 1
    assert engine.stats["skipped_orders"] == 1


def test_pre_execution_order_checks_blocks_openings_for_symbol_reentry_cooldown(monkeypatch):
    engine = _engine_stub()
    clock = {"value": 100.0}
    monkeypatch.setattr(lt, "monotonic_time", lambda: clock["value"])
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_REENTRY_COOLDOWN_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_REENTRY_COOLDOWN_SEC", "300")
    monkeypatch.setenv("AI_TRADING_EXECUTION_OPENING_MIN_NOTIONAL", "0")
    monkeypatch.setattr(
        engine,
        "_enforce_opposite_side_policy",
        lambda *_args, **_kwargs: (True, None),
    )
    monkeypatch.setattr(
        engine,
        "_evaluate_pdt_preflight",
        lambda *_args, **_kwargs: (False, None, {}),
    )
    monkeypatch.setattr(
        engine,
        "_runtime_gonogo_openings_allowed",
        lambda: (True, {"enabled": False}),
    )
    monkeypatch.setattr(
        engine,
        "_resolve_exposure_normalization_settings",
        lambda: {"block_openings": False},
    )
    monkeypatch.setattr(engine, "_exposure_normalization_context", lambda *_args, **_kwargs: {})
    engine._arm_symbol_reentry_cooldown_from_fill(symbol="AAPL", side="buy")

    order = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 1,
        "price_hint": "200.0",
        "client_order_id": "cid-1",
        "closing_position": False,
        "account_snapshot": {},
    }

    allowed = engine._pre_execution_order_checks(order)

    assert allowed is False
    assert engine.stats["capacity_skips"] == 1
    assert engine.stats["skipped_orders"] == 1


def test_pre_execution_order_checks_blocks_openings_for_symbol_slippage_budget(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_REENTRY_COOLDOWN_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_EXECUTION_OPENING_MIN_NOTIONAL", "0")
    monkeypatch.setattr(
        engine,
        "_enforce_opposite_side_policy",
        lambda *_args, **_kwargs: (True, None),
    )
    monkeypatch.setattr(
        engine,
        "_evaluate_pdt_preflight",
        lambda *_args, **_kwargs: (False, None, {}),
    )
    monkeypatch.setattr(
        engine,
        "_runtime_gonogo_openings_allowed",
        lambda: (True, {"enabled": False}),
    )
    monkeypatch.setattr(
        engine,
        "_resolve_exposure_normalization_settings",
        lambda: {"block_openings": False},
    )
    monkeypatch.setattr(engine, "_exposure_normalization_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        engine,
        "_symbol_intraday_slippage_budget_allows_opening",
        lambda **_: (
            False,
            {
                "enabled": True,
                "reason": "symbol_intraday_slippage_drag_breach",
                "symbol": "AAPL",
            },
        ),
    )

    order = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 1,
        "price_hint": "200.0",
        "client_order_id": "cid-1",
        "closing_position": False,
        "account_snapshot": {},
    }

    allowed = engine._pre_execution_order_checks(order)

    assert allowed is False
    assert engine.stats["capacity_skips"] == 1
    assert engine.stats["skipped_orders"] == 1


def test_pre_execution_order_checks_blocks_openings_below_min_notional(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_REENTRY_COOLDOWN_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_EXECUTION_OPENING_MIN_NOTIONAL", "250")
    monkeypatch.setattr(
        engine,
        "_enforce_opposite_side_policy",
        lambda *_args, **_kwargs: (True, None),
    )
    monkeypatch.setattr(
        engine,
        "_evaluate_pdt_preflight",
        lambda *_args, **_kwargs: (False, None, {}),
    )
    monkeypatch.setattr(
        engine,
        "_runtime_gonogo_openings_allowed",
        lambda: (True, {"enabled": False}),
    )
    monkeypatch.setattr(
        engine,
        "_resolve_exposure_normalization_settings",
        lambda: {"block_openings": False},
    )
    monkeypatch.setattr(engine, "_exposure_normalization_context", lambda *_args, **_kwargs: {})

    order = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 1,
        "price_hint": "120.0",
        "client_order_id": "cid-1",
        "closing_position": False,
        "account_snapshot": {},
    }

    allowed = engine._pre_execution_order_checks(order)

    assert allowed is False
    assert engine.stats["capacity_skips"] == 1
    assert engine.stats["skipped_orders"] == 1


def test_resolve_exposure_normalization_settings_uses_runtime_env(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_BLOCK_OPENINGS", "0")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_BP_MIN_RATIO", "0.05")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_MAX_GROSS_TO_EQUITY", "3.0")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_MAX_NET_TO_EQUITY", "2.2")

    settings = engine._resolve_exposure_normalization_settings()

    assert settings["block_openings"] is False
    assert settings["bp_min_ratio"] == pytest.approx(0.05)
    assert settings["max_gross_to_equity"] == pytest.approx(3.0)
    assert settings["max_net_to_equity"] == pytest.approx(2.2)


def test_prioritize_losing_short_reduction_targets_largest_losses(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_SHORTS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_MAX_ACTIONS", "2")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_FRACTION", "0.5")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_COOLDOWN_SEC", "0")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_BP_MIN_RATIO", "0.03")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_MAX_GROSS_TO_EQUITY", "1.0")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_MAX_NET_TO_EQUITY", "0.35")

    submitted: list[tuple[str, int]] = []

    def _submit_cover_order(symbol: str, qty: int) -> bool:
        submitted.append((symbol, qty))
        return True

    monkeypatch.setattr(engine, "_submit_cover_order", _submit_cover_order)

    actions = engine._prioritize_losing_short_reduction(
        positions=[
            {"symbol": "AAA", "side": "short", "qty": 10, "current_price": 50.0, "unrealized_intraday_pl": -100.0},
            {"symbol": "BBB", "side": "short", "qty": 8, "current_price": 80.0, "unrealized_intraday_pl": -50.0},
            {"symbol": "CCC", "side": "short", "qty": 12, "current_price": 40.0, "unrealized_intraday_pl": -200.0},
            {"symbol": "DDD", "side": "short", "qty": 5, "current_price": 25.0, "unrealized_intraday_pl": 20.0},
        ],
        account_snapshot={
            "equity": 100000,
            "buying_power": 1000,
            "long_market_value": 90000,
            "short_market_value": 140000,
        },
    )

    assert actions == 2
    assert submitted[0][0] == "CCC"
    assert submitted[1][0] == "AAA"


def test_prioritize_short_reduction_enabled_by_default(monkeypatch):
    engine = _engine_stub()
    monkeypatch.delenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_SHORTS_ENABLED", raising=False)
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_MAX_ACTIONS", "1")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_FRACTION", "0.5")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_COOLDOWN_SEC", "0")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_MAX_GROSS_TO_EQUITY", "1.0")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_MAX_NET_TO_EQUITY", "0.35")

    submitted: list[tuple[str, int]] = []

    def _submit_cover_order(symbol: str, qty: int) -> bool:
        submitted.append((symbol, qty))
        return True

    monkeypatch.setattr(engine, "_submit_cover_order", _submit_cover_order)

    actions = engine._prioritize_losing_short_reduction(
        positions=[
            {"symbol": "AAA", "side": "short", "qty": 10, "current_price": 50.0, "unrealized_intraday_pl": -100.0},
        ],
        account_snapshot={
            "equity": 100000,
            "buying_power": 1000,
            "long_market_value": 90000,
            "short_market_value": 140000,
        },
    )

    assert actions == 1
    assert submitted == [("AAA", 5)]


def test_prioritize_short_reduction_handles_positionside_short(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_SHORTS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_SHORTS_REQUIRE_LOSS", "1")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_MAX_ACTIONS", "1")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_COOLDOWN_SEC", "0")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_MIN_NOTIONAL", "0")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_MAX_GROSS_TO_EQUITY", "1.0")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_MAX_NET_TO_EQUITY", "0.35")

    submitted: list[tuple[str, int]] = []

    def _submit_cover_order(symbol: str, qty: int) -> bool:
        submitted.append((symbol, qty))
        return True

    monkeypatch.setattr(engine, "_submit_cover_order", _submit_cover_order)

    actions = engine._prioritize_losing_short_reduction(
        positions=[
            {
                "symbol": "ABC",
                "side": "PositionSide.SHORT",
                "qty": 10,
                "current_price": 20.0,
                "unrealized_intraday_pl": -25.0,
            }
        ],
        account_snapshot={
            "equity": 100000,
            "buying_power": 1000,
            "long_market_value": 90000,
            "short_market_value": 140000,
        },
    )

    assert actions == 1
    assert submitted and submitted[0][0] == "ABC"


def test_prioritize_short_reduction_can_ignore_loss_filter(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_SHORTS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_SHORTS_REQUIRE_LOSS", "0")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_MAX_ACTIONS", "2")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_FRACTION", "0.5")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_REDUCE_COOLDOWN_SEC", "0")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_MAX_GROSS_TO_EQUITY", "1.0")
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_MAX_NET_TO_EQUITY", "0.35")

    submitted: list[tuple[str, int]] = []

    def _submit_cover_order(symbol: str, qty: int) -> bool:
        submitted.append((symbol, qty))
        return True

    monkeypatch.setattr(engine, "_submit_cover_order", _submit_cover_order)

    actions = engine._prioritize_losing_short_reduction(
        positions=[
            {"symbol": "AAA", "side": "short", "qty": 10, "current_price": 50.0, "unrealized_intraday_pl": 25.0},
            {"symbol": "BBB", "side": "short", "qty": 8, "current_price": 80.0, "unrealized_intraday_pl": 10.0},
        ],
        account_snapshot={
            "equity": 100000,
            "buying_power": 1000,
            "long_market_value": 90000,
            "short_market_value": 140000,
        },
    )

    assert actions == 2
    assert submitted[0][0] == "BBB"
    assert submitted[1][0] == "AAA"


def _write_runtime_gonogo_artifacts(
    *,
    root: Path,
    trade_rows: list[dict[str, Any]],
    gate_summary: dict[str, Any],
) -> None:
    runtime_dir = root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "trade_history.json").write_text(
        json.dumps(trade_rows),
        encoding="utf-8",
    )
    (runtime_dir / "gate_effectiveness_summary.json").write_text(
        json.dumps(gate_summary),
        encoding="utf-8",
    )


def test_runtime_gonogo_precheck_blocks_openings_when_gate_fails(monkeypatch, tmp_path):
    engine = _engine_stub()
    _write_runtime_gonogo_artifacts(
        root=tmp_path,
        trade_rows=[{"symbol": "AAPL", "side": "buy", "pnl": -10.0}],
        gate_summary={
            "total_records": 10,
            "total_accepted_records": 7,
            "total_rejected_records": 3,
            "total_expected_net_edge_bps": 0.0,
        },
    )
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_TRADE_HISTORY_PATH", "runtime/trade_history.json")
    monkeypatch.setenv("AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH", "runtime/trade_history.json")
    monkeypatch.setenv(
        "AI_TRADING_RUNTIME_PERF_GATE_SUMMARY_PATH",
        "runtime/gate_effectiveness_summary.json",
    )
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_BLOCK_OPENINGS", "0")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_BLOCK_OPENINGS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_CACHE_TTL_SEC", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_CLOSED_TRADES", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_PROFIT_FACTOR", "1.1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_WIN_RATE", "0.5")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_NET_PNL", "0.0")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_ACCEPTANCE_RATE", "0.05")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_REQUIRE_PNL_AVAILABLE", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_REQUIRE_GATE_VALID", "1")
    monkeypatch.setattr(
        engine,
        "_enforce_opposite_side_policy",
        lambda *_args, **_kwargs: (True, None),
    )
    monkeypatch.setattr(
        engine,
        "_evaluate_pdt_preflight",
        lambda *_args, **_kwargs: (False, None, {}),
    )

    allowed = engine._pre_execution_order_checks(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 5,
            "client_order_id": "cid-gonogo-fail",
            "closing_position": False,
            "account_snapshot": {
                "equity": 100000,
                "buying_power": 20000,
                "long_market_value": 20000,
                "short_market_value": 10000,
            },
        }
    )

    assert allowed is False
    assert engine.stats["capacity_skips"] == 1
    assert engine.stats["skipped_orders"] == 1


def test_runtime_gonogo_monitor_only_for_paper_mode_by_default(monkeypatch, tmp_path):
    engine = _engine_stub()
    engine.execution_mode = "paper"
    _write_runtime_gonogo_artifacts(
        root=tmp_path,
        trade_rows=[{"symbol": "AAPL", "side": "buy", "pnl": -10.0}],
        gate_summary={
            "total_records": 10,
            "total_accepted_records": 7,
            "total_rejected_records": 3,
            "total_expected_net_edge_bps": 0.0,
        },
    )
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_TRADE_HISTORY_PATH", "runtime/trade_history.json")
    monkeypatch.setenv("AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH", "runtime/trade_history.json")
    monkeypatch.setenv(
        "AI_TRADING_RUNTIME_PERF_GATE_SUMMARY_PATH",
        "runtime/gate_effectiveness_summary.json",
    )
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_BLOCK_OPENINGS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_CACHE_TTL_SEC", "1")
    monkeypatch.delenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_ENFORCE_IN_PAPER", raising=False)

    allowed, context = engine._runtime_gonogo_openings_allowed()

    assert allowed is True
    assert context["enabled"] is True
    assert context["enforced"] is False
    assert context["reason"] == "paper_mode_monitor_only"
    assert context["execution_mode"] == "paper"


def test_runtime_gonogo_can_enforce_in_paper_when_enabled(monkeypatch, tmp_path):
    engine = _engine_stub()
    engine.execution_mode = "paper"
    _write_runtime_gonogo_artifacts(
        root=tmp_path,
        trade_rows=[{"symbol": "AAPL", "side": "buy", "pnl": -10.0}],
        gate_summary={
            "total_records": 10,
            "total_accepted_records": 7,
            "total_rejected_records": 3,
            "total_expected_net_edge_bps": 0.0,
        },
    )
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_TRADE_HISTORY_PATH", "runtime/trade_history.json")
    monkeypatch.setenv("AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH", "runtime/trade_history.json")
    monkeypatch.setenv(
        "AI_TRADING_RUNTIME_PERF_GATE_SUMMARY_PATH",
        "runtime/gate_effectiveness_summary.json",
    )
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_BLOCK_OPENINGS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_CACHE_TTL_SEC", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_CLOSED_TRADES", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_PROFIT_FACTOR", "1.1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_WIN_RATE", "0.5")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_NET_PNL", "0.0")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_ACCEPTANCE_RATE", "0.05")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_REQUIRE_PNL_AVAILABLE", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_REQUIRE_GATE_VALID", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_ENFORCE_IN_PAPER", "1")

    allowed, context = engine._runtime_gonogo_openings_allowed()

    assert allowed is False
    assert context["enabled"] is True
    assert context["gate_passed"] is False
    assert "profit_factor" in context["failed_checks"]


def test_runtime_gonogo_precheck_allows_openings_when_gate_passes(monkeypatch, tmp_path):
    engine = _engine_stub()
    _write_runtime_gonogo_artifacts(
        root=tmp_path,
        trade_rows=[
            {"symbol": "AAPL", "side": "buy", "pnl": 20.0},
            {"symbol": "MSFT", "side": "sell", "pnl": -10.0},
        ],
        gate_summary={
            "total_records": 20,
            "total_accepted_records": 10,
            "total_rejected_records": 10,
            "total_expected_net_edge_bps": 10.0,
        },
    )
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_TRADE_HISTORY_PATH", "runtime/trade_history.json")
    monkeypatch.setenv("AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH", "runtime/trade_history.json")
    monkeypatch.setenv(
        "AI_TRADING_RUNTIME_PERF_GATE_SUMMARY_PATH",
        "runtime/gate_effectiveness_summary.json",
    )
    monkeypatch.setenv("AI_TRADING_EXPOSURE_NORMALIZE_BLOCK_OPENINGS", "0")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_BLOCK_OPENINGS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_CACHE_TTL_SEC", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_CLOSED_TRADES", "2")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_PROFIT_FACTOR", "1.1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_WIN_RATE", "0.5")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_NET_PNL", "0.0")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_ACCEPTANCE_RATE", "0.05")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_REQUIRE_PNL_AVAILABLE", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_REQUIRE_GATE_VALID", "1")
    monkeypatch.setattr(
        engine,
        "_enforce_opposite_side_policy",
        lambda *_args, **_kwargs: (True, None),
    )
    monkeypatch.setattr(
        engine,
        "_evaluate_pdt_preflight",
        lambda *_args, **_kwargs: (False, None, {}),
    )

    allowed = engine._pre_execution_order_checks(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 5,
            "client_order_id": "cid-gonogo-pass",
            "closing_position": False,
            "account_snapshot": {
                "equity": 100000,
                "buying_power": 20000,
                "long_market_value": 20000,
                "short_market_value": 10000,
            },
        }
    )

    assert allowed is True
    assert engine.stats.get("capacity_skips", 0) == 0
    assert engine.stats.get("skipped_orders", 0) == 0


def test_runtime_gonogo_threshold_lock_prevents_intraday_loosening(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_LOCK_THRESHOLDS_INTRADAY", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_LOCK_TZ", "UTC")

    first, first_context = engine._locked_runtime_gonogo_thresholds(
        {
            "min_closed_trades": 50,
            "min_profit_factor": 0.90,
            "min_win_rate": 0.50,
            "min_net_pnl": -300.0,
            "min_acceptance_rate": 0.015,
            "min_expected_net_edge_bps": -50.0,
            "min_used_days": 3,
            "lookback_days": 5,
            "trade_fill_source": "live",
            "require_pnl_available": True,
            "require_gate_valid": True,
        }
    )
    assert first["min_profit_factor"] == pytest.approx(0.90)
    assert first_context["enabled"] is True

    relaxed, _ = engine._locked_runtime_gonogo_thresholds(
        {
            "min_closed_trades": 10,
            "min_profit_factor": 0.70,
            "min_win_rate": 0.45,
            "min_net_pnl": -1100.0,
            "min_acceptance_rate": 0.01,
            "min_expected_net_edge_bps": -100.0,
            "min_used_days": 1,
            "lookback_days": 1,
            "trade_fill_source": "all",
            "require_pnl_available": False,
            "require_gate_valid": False,
        }
    )

    assert relaxed["min_closed_trades"] == 50
    assert relaxed["min_profit_factor"] == pytest.approx(0.90)
    assert relaxed["min_win_rate"] == pytest.approx(0.50)
    assert relaxed["min_net_pnl"] == pytest.approx(-300.0)
    assert relaxed["min_acceptance_rate"] == pytest.approx(0.015)
    assert relaxed["min_expected_net_edge_bps"] == pytest.approx(-50.0)
    assert relaxed["min_used_days"] == 3
    assert relaxed["lookback_days"] == 5
    assert relaxed["trade_fill_source"] == "live"
    assert relaxed["require_pnl_available"] is True
    assert relaxed["require_gate_valid"] is True

    tightened, _ = engine._locked_runtime_gonogo_thresholds(
        {
            "min_closed_trades": 80,
            "min_profit_factor": 1.05,
            "min_win_rate": 0.52,
            "min_net_pnl": -200.0,
            "min_acceptance_rate": 0.02,
            "min_expected_net_edge_bps": -10.0,
            "min_used_days": 4,
            "lookback_days": 7,
            "trade_fill_source": "live",
            "require_pnl_available": True,
            "require_gate_valid": True,
        }
    )
    assert tightened["min_profit_factor"] == pytest.approx(1.05)
    assert tightened["min_net_pnl"] == pytest.approx(-200.0)
    assert tightened["min_closed_trades"] == 80
    assert tightened["lookback_days"] == 7


def test_runtime_gonogo_hourly_guard_blocks_current_weakest_hour(monkeypatch, tmp_path):
    engine = _engine_stub()
    gate_log_path = tmp_path / "runtime" / "gate_effectiveness.jsonl"
    gate_log_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    rows = [
        {
            "ts": now.isoformat(),
            "records_total": 120,
            "accepted_records": 12,
            "total_expected_net_edge_bps": -240.0,
        },
        {
            "ts": (now - timedelta(hours=1)).isoformat(),
            "records_total": 120,
            "accepted_records": 96,
            "total_expected_net_edge_bps": 120.0,
        },
    ]
    gate_log_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_HOURLY_BLOCK_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_HOURLY_BLOCK_TZ", "UTC")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_HOURLY_BLOCK_LOOKBACK_DAYS", "2")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_HOURLY_BLOCK_MIN_RECORDS", "20")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_HOURLY_BLOCK_MIN_ACCEPTANCE_RATE", "0.20")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_HOURLY_BLOCK_MIN_EDGE_BPS_PER_RECORD", "0.0")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_HOURLY_BLOCK_ONLY_WEAKEST", "1")

    allowed, context = engine._runtime_gonogo_hourly_guard_allows_openings(
        gate_log_path=gate_log_path,
        thresholds={"min_acceptance_rate": 0.015},
    )

    assert allowed is False
    assert context["enabled"] is True
    assert "hourly_acceptance_rate" in context["failed_checks"]
    assert "hourly_expected_edge_bps_per_record" in context["failed_checks"]


def test_runtime_gonogo_after_close_tighten_overrides(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_RUNTIME_GONOGO_AFTER_CLOSE_TIGHTEN_ENABLED",
        "1",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_RUNTIME_GONOGO_AFTER_CLOSE_MIN_PROFIT_FACTOR",
        "0.95",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_RUNTIME_GONOGO_AFTER_CLOSE_MIN_NET_PNL",
        "-200",
    )
    monkeypatch.setattr(lt, "_market_is_open_now", lambda *_args, **_kwargs: False)

    tightened, context = engine._apply_after_close_runtime_gonogo_overrides(
        {
            "min_profit_factor": 0.73,
            "min_net_pnl": -1100.0,
            "min_win_rate": 0.50,
            "min_acceptance_rate": 0.015,
        }
    )

    assert context["enabled"] is True
    assert context["reason"] == "market_closed"
    assert tightened["min_profit_factor"] == pytest.approx(0.95)
    assert tightened["min_net_pnl"] == pytest.approx(-200.0)


def test_runtime_gonogo_intraday_pnl_kill_switch_blocks(monkeypatch):
    engine = _engine_stub()
    today = datetime.now(UTC).date().isoformat()
    monkeypatch.setenv("AI_TRADING_EXECUTION_INTRADAY_PNL_KILL_SWITCH_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_INTRADAY_PNL_KILL_SWITCH_MAX_LOSS", "-50")
    monkeypatch.setenv("AI_TRADING_EXECUTION_INTRADAY_PNL_KILL_SWITCH_TZ", "UTC")
    allowed, context = engine._runtime_intraday_pnl_kill_switch_allows_openings(
        report={
            "trade_history": {
                "daily_trade_stats": [
                    {
                        "date": today,
                        "net_pnl": -100.0,
                    }
                ]
            }
        },
        thresholds={"trade_fill_source": "all"},
    )

    assert allowed is False
    assert context["reason"] == "intraday_loss_breach"


def test_runtime_gonogo_intraday_slippage_kill_switch_blocks(monkeypatch):
    engine = _engine_stub()
    today = datetime.now(UTC).date().isoformat()
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_INTRADAY_SLIPPAGE_KILL_SWITCH_ENABLED",
        "1",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_INTRADAY_SLIPPAGE_KILL_SWITCH_MAX_DRAG",
        "25",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_INTRADAY_SLIPPAGE_KILL_SWITCH_MIN_TRADES",
        "5",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_INTRADAY_SLIPPAGE_KILL_SWITCH_TZ",
        "UTC",
    )
    allowed, context = engine._runtime_intraday_slippage_kill_switch_allows_openings(
        report={
            "trade_history": {
                "daily_trade_stats": [
                    {
                        "date": today,
                        "trades": 10,
                        "slippage_cost": 40.0,
                    }
                ]
            }
        },
        thresholds={"trade_fill_source": "all"},
    )

    assert allowed is False
    assert context["reason"] == "intraday_slippage_drag_breach"


def test_symbol_intraday_slippage_budget_blocks_symbol_when_drag_breaches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = _engine_stub()
    today = datetime.now(UTC).date().isoformat()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    fill_events_path = runtime_dir / "fill_events.jsonl"
    fill_events_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "fill_recorded",
                        "symbol": "AAPL",
                        "side": "buy",
                        "fill_price": 200.0,
                        "fill_qty": 10.0,
                        "slippage_bps": 100.0,
                        "entry_time": f"{today}T14:30:00+00:00",
                    }
                ),
                json.dumps(
                    {
                        "event": "fill_recorded",
                        "symbol": "AAPL",
                        "side": "buy",
                        "fill_price": 210.0,
                        "fill_qty": 10.0,
                        "slippage_bps": 100.0,
                        "entry_time": f"{today}T15:00:00+00:00",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(
        "AI_TRADING_FILL_EVENTS_PATH",
        "runtime/fill_events.jsonl",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_SYMBOL_INTRADAY_SLIPPAGE_BUDGET_ENABLED",
        "1",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_SYMBOL_INTRADAY_SLIPPAGE_BUDGET_MAX_DRAG",
        "30",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_SYMBOL_INTRADAY_SLIPPAGE_BUDGET_MIN_FILLS",
        "2",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_SYMBOL_INTRADAY_SLIPPAGE_BUDGET_TZ",
        "UTC",
    )

    allowed, context = engine._symbol_intraday_slippage_budget_allows_opening(
        symbol="AAPL"
    )

    assert allowed is False
    assert context["reason"] == "symbol_intraday_slippage_drag_breach"
    assert context["symbol_fills"] == 2
    assert context["today_symbol_slippage_drag"] > 30.0


def test_symbol_loss_cooldown_triggers_and_expires(monkeypatch):
    engine = _engine_stub()
    clock = {"value": 100.0}
    monkeypatch.setattr(lt, "monotonic_time", lambda: clock["value"])
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_LOSS_COOLDOWN_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_LOSS_COOLDOWN_TRIGGER_STREAK", "2")
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_LOSS_COOLDOWN_MIN_SLIPPAGE_BPS", "2.0")
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_LOSS_COOLDOWN_MINUTES", "1")

    engine._update_symbol_loss_cooldown_from_fill(symbol="AAPL", slippage_bps=2.5)
    allowed_before, _ = engine._symbol_loss_cooldown_allows_opening(symbol="AAPL")
    assert allowed_before is True

    engine._update_symbol_loss_cooldown_from_fill(symbol="AAPL", slippage_bps=3.0)
    allowed_during, context_during = engine._symbol_loss_cooldown_allows_opening(symbol="AAPL")
    assert allowed_during is False
    assert context_during["reason"] == "symbol_loss_cooldown"

    clock["value"] = 170.0
    allowed_after, context_after = engine._symbol_loss_cooldown_allows_opening(symbol="AAPL")
    assert allowed_after is True
    assert context_after["reason"] == "cooldown_inactive"


def test_symbol_loss_cooldown_triggers_from_realized_loss(monkeypatch):
    engine = _engine_stub()
    clock = {"value": 200.0}
    monkeypatch.setattr(lt, "monotonic_time", lambda: clock["value"])
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_LOSS_COOLDOWN_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_LOSS_COOLDOWN_TRIGGER_STREAK", "5")
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_LOSS_COOLDOWN_REALIZED_TRIGGER_STREAK", "2")
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_LOSS_COOLDOWN_MIN_SLIPPAGE_BPS", "10.0")
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_LOSS_COOLDOWN_MIN_REALIZED_PNL", "-25.0")
    monkeypatch.setenv("AI_TRADING_EXECUTION_SYMBOL_LOSS_COOLDOWN_MINUTES", "1")

    engine._update_symbol_loss_cooldown_from_fill(
        symbol="AAPL",
        slippage_bps=0.5,
        realized_pnl=-30.0,
    )
    allowed_before, _ = engine._symbol_loss_cooldown_allows_opening(symbol="AAPL")
    assert allowed_before is True

    engine._update_symbol_loss_cooldown_from_fill(
        symbol="AAPL",
        slippage_bps=0.5,
        realized_pnl=-40.0,
    )
    allowed_after, context_after = engine._symbol_loss_cooldown_allows_opening(symbol="AAPL")
    assert allowed_after is False
    assert context_after["reason"] == "symbol_loss_cooldown"
    assert context_after["cooldown_reason"] == "realized_loss_streak"


def test_order_ack_timeout_recovery_clears_state(monkeypatch):
    engine = _engine_stub()
    engine._last_order_ack_timeout_mono = 10.0
    engine._last_order_ack_timeout_order_id = "oid-1"
    engine._last_order_ack_timeout_client_order_id = "cid-1"
    clock = {"value": 100.0}
    updates: list[dict[str, Any]] = []
    monkeypatch.setattr(lt, "monotonic_time", lambda: clock["value"])
    monkeypatch.setattr(
        lt.runtime_state,
        "update_broker_status",
        lambda **kwargs: updates.append(dict(kwargs)),
    )
    monkeypatch.setenv("AI_TRADING_ORDER_ACK_RECOVERY_SEC", "30")

    recovered = engine._maybe_recover_order_ack_timeout(open_orders_count=0)

    assert recovered is True
    assert engine._last_order_ack_timeout_mono == 0.0
    assert engine._last_order_ack_timeout_order_id is None
    assert engine._last_order_ack_timeout_client_order_id is None
    assert updates
    assert updates[-1]["last_error"] is None


def test_resolve_order_submit_cap_uses_bootstrap_defaults(monkeypatch):
    engine = _engine_stub()
    engine.ctx = SimpleNamespace(state={"service_phase": "bootstrap"})
    engine._engine_started_mono = 100.0
    engine._engine_cycle_index = 1
    monkeypatch.setattr(lt, "monotonic_time", lambda: 110.0)
    monkeypatch.delenv("EXECUTION_MAX_NEW_ORDERS_PER_CYCLE", raising=False)
    monkeypatch.delenv("AI_TRADING_MAX_NEW_ORDERS_PER_CYCLE", raising=False)
    monkeypatch.setenv("AI_TRADING_BOOTSTRAP_ORDER_CAP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_BOOTSTRAP_MAX_NEW_ORDERS_PER_CYCLE", "2")
    monkeypatch.setenv("AI_TRADING_BOOTSTRAP_CAP_CYCLES", "3")
    monkeypatch.setenv("AI_TRADING_BOOTSTRAP_CAP_SECONDS", "180")

    cap, source = engine._resolve_order_submit_cap()

    assert cap == 2
    assert source == "bootstrap"


def test_resolve_order_submit_cap_combines_configured_and_bootstrap(monkeypatch):
    engine = _engine_stub()
    engine.ctx = SimpleNamespace(state={"service_phase": "bootstrap"})
    engine._engine_started_mono = 100.0
    engine._engine_cycle_index = 1
    monkeypatch.setattr(lt, "monotonic_time", lambda: 110.0)
    monkeypatch.setenv("AI_TRADING_MAX_NEW_ORDERS_PER_CYCLE", "1")
    monkeypatch.setenv("AI_TRADING_BOOTSTRAP_ORDER_CAP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_BOOTSTRAP_MAX_NEW_ORDERS_PER_CYCLE", "3")

    cap, source = engine._resolve_order_submit_cap()

    assert cap == 1
    assert source == "configured+bootstrap"


def test_resolve_order_submit_cap_includes_adaptive(monkeypatch):
    engine = _engine_stub()
    engine.ctx = SimpleNamespace(state={"service_phase": "bootstrap"})
    engine._engine_started_mono = 100.0
    engine._engine_cycle_index = 1
    engine._adaptive_new_orders_cap = 2
    monkeypatch.setattr(lt, "monotonic_time", lambda: 110.0)
    monkeypatch.setenv("AI_TRADING_MAX_NEW_ORDERS_PER_CYCLE", "5")
    monkeypatch.setenv("AI_TRADING_BOOTSTRAP_ORDER_CAP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_BOOTSTRAP_MAX_NEW_ORDERS_PER_CYCLE", "4")

    cap, source = engine._resolve_order_submit_cap()

    assert cap == 2
    assert source == "configured+bootstrap+adaptive"


def test_resolve_order_submit_cap_includes_pending_backlog(monkeypatch):
    engine = _engine_stub()
    engine.ctx = SimpleNamespace(state={"service_phase": "runtime"})
    engine._pending_orders = {"AAPL": {"status": "pending_new"}}
    monkeypatch.setenv("AI_TRADING_MAX_NEW_ORDERS_PER_CYCLE", "5")
    monkeypatch.setenv("AI_TRADING_BOOTSTRAP_ORDER_CAP_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_CAP_THRESHOLD", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_CAP_VALUE", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_ADAPTIVE_CAP_ENABLED", "0")

    cap, source = engine._resolve_order_submit_cap()

    assert cap == 1
    assert source == "configured+pending_backlog"


def test_resolve_order_submit_cap_includes_cancel_ratio(monkeypatch):
    engine = _engine_stub()
    engine.ctx = SimpleNamespace(state={"service_phase": "runtime"})
    engine._cancel_ratio_adaptive_new_orders_cap = 2
    monkeypatch.setenv("AI_TRADING_MAX_NEW_ORDERS_PER_CYCLE", "5")
    monkeypatch.setenv("AI_TRADING_BOOTSTRAP_ORDER_CAP_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_CAP_THRESHOLD", "100")

    cap, source = engine._resolve_order_submit_cap()

    assert cap == 2
    assert source == "configured+cancel_ratio"


def test_resolve_order_submit_cap_applies_pacing_relax_floor(monkeypatch):
    engine = _engine_stub()
    engine.ctx = SimpleNamespace(state={"service_phase": "runtime"})
    engine._pacing_relax_new_orders_cap = 4
    monkeypatch.setenv("AI_TRADING_MAX_NEW_ORDERS_PER_CYCLE", "2")
    monkeypatch.setenv("AI_TRADING_BOOTSTRAP_ORDER_CAP_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_CAP_THRESHOLD", "100")

    cap, source = engine._resolve_order_submit_cap()

    assert cap == 4
    assert source == "configured+pacing_relax"


def test_cancel_ratio_adaptive_cap_triggers_and_recovers(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_CANCEL_RATIO_ADAPTIVE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_CANCEL_RATIO_TRIGGER", "0.65")
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_CANCEL_RATIO_CLEAR", "0.40")
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_CANCEL_RATIO_SCALE", "0.50")
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_CANCEL_RATIO_MIN_CAP", "2")
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_CANCEL_RATIO_MIN_SUBMITTED", "8")

    engine._update_cancel_ratio_adaptive_cap(cancel_ratio=0.8, submitted=10)

    assert engine._cancel_ratio_adaptive_new_orders_cap == 5
    assert engine._cancel_ratio_adaptive_context.get("state") == "triggered"

    engine._update_cancel_ratio_adaptive_cap(cancel_ratio=0.3, submitted=10)

    assert engine._cancel_ratio_adaptive_new_orders_cap is None
    assert engine._cancel_ratio_adaptive_context.get("state") == "recovered"


def test_cancel_ratio_adaptive_cap_triggers_pacing_relax_when_quality_good(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_CANCEL_RATIO_ADAPTIVE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_CANCEL_RATIO_TRIGGER", "0.65")
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_CANCEL_RATIO_MIN_SUBMITTED", "8")
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_RELAX_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_RELAX_TRIGGER_PCT", "10")
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_RELAX_CLEAR_PCT", "4")
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_RELAX_MIN_SUBMITTED", "8")
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_RELAX_MAX_CANCEL_RATIO", "0.45")
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_RELAX_MAX_REJECT_RATE_PCT", "3.0")
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_RELAX_CAP", "4")

    engine._update_cancel_ratio_adaptive_cap(
        cancel_ratio=0.10,
        submitted=12,
        pacing_cap_hit_rate_pct=25.0,
        reject_rate_pct=0.1,
    )
    assert engine._cancel_ratio_adaptive_new_orders_cap is None
    assert engine._pacing_relax_new_orders_cap == 4
    assert engine._cancel_ratio_adaptive_context.get("pacing_relax_state") == "triggered"

    engine._update_cancel_ratio_adaptive_cap(
        cancel_ratio=0.10,
        submitted=12,
        pacing_cap_hit_rate_pct=2.0,
        reject_rate_pct=0.1,
    )
    assert engine._pacing_relax_new_orders_cap is None
    assert engine._cancel_ratio_adaptive_context.get("pacing_relax_state") == "recovered"


def test_pending_backlog_cap_supports_adaptive_scaling(monkeypatch):
    engine = _engine_stub()
    now_dt = datetime.now(UTC)
    engine._pending_orders = {
        f"ord-{idx}": {
            "status": "pending_new",
            "updated_at": (now_dt - timedelta(seconds=40 + idx)).isoformat(),
        }
        for idx in range(6)
    }
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_CAP_THRESHOLD", "3")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_CAP_VALUE", "2")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_ADAPTIVE_CAP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_ADAPTIVE_MIN_CAP", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_ADAPTIVE_MAX_CAP", "4")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_ADAPTIVE_STEP_ORDERS", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_ADAPTIVE_AGE_SOFT_SEC", "600")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_ADAPTIVE_AGE_HARD_SEC", "1200")

    cap = engine._pending_backlog_order_cap()

    assert cap == 1
    context = getattr(engine, "_pending_backlog_last_context", {})
    assert context.get("adaptive_enabled") is True
    assert context.get("adaptive_cap") == 1
    assert context.get("selected_cap") == 1


def test_pending_backlog_cap_ignores_stale_local_pending(monkeypatch):
    engine = _engine_stub()
    stale_ts = (datetime.now(UTC) - timedelta(seconds=900)).isoformat()
    engine._pending_orders = {"ord-1": {"status": "pending_new", "updated_at": stale_ts}}

    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_CAP_THRESHOLD", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_CAP_VALUE", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_LOCAL_STALE_SEC", "60")

    cap = engine._pending_backlog_order_cap()

    assert cap is None
    context = getattr(engine, "_pending_backlog_last_context", {})
    assert context.get("stale_ignored_count") == 1
    assert context.get("local_pending_count") == 0


def test_pending_backlog_local_stale_default_respects_sweep_threshold(monkeypatch):
    engine = _engine_stub()
    monkeypatch.delenv("AI_TRADING_PENDING_BACKLOG_LOCAL_STALE_SEC", raising=False)
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_SEC", "240")

    stale_seconds = engine._pending_backlog_local_stale_seconds()

    assert stale_seconds == 240.0


def test_pending_backlog_hard_block_by_count(monkeypatch):
    engine = _engine_stub()
    engine._pending_orders = {
        "ord-1": {"status": "pending_new"},
        "ord-2": {"status": "pending_new"},
    }
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_CAP_THRESHOLD", "10")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_HARD_BLOCK_COUNT", "2")

    cap = engine._pending_backlog_order_cap()

    assert cap == 0
    context = getattr(engine, "_pending_backlog_last_context", {})
    assert context.get("hard_block_triggered") is True
    assert "count" in tuple(context.get("hard_block_reasons", ()))


def test_pending_backlog_hard_block_by_age(monkeypatch):
    engine = _engine_stub()
    stale_ts = (datetime.now(UTC) - timedelta(seconds=600)).isoformat()
    engine._pending_orders = {"ord-1": {"status": "pending_new", "updated_at": stale_ts}}
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_CAP_THRESHOLD", "10")
    monkeypatch.setenv("AI_TRADING_PENDING_BACKLOG_HARD_BLOCK_AGE_SEC", "300")

    cap = engine._pending_backlog_order_cap()

    assert cap == 0
    context = getattr(engine, "_pending_backlog_last_context", {})
    assert context.get("hard_block_triggered") is True
    assert "age" in tuple(context.get("hard_block_reasons", ()))


def test_ensure_initialized_uses_existing_client_validation(monkeypatch):
    engine = _engine_stub()
    engine.trading_client = object()
    engine.is_initialized = False
    init_calls: list[bool] = []
    monkeypatch.setattr(engine, "_validate_connection", lambda: True)

    def _initialize() -> bool:
        init_calls.append(True)
        return False

    monkeypatch.setattr(engine, "initialize", _initialize)

    assert engine._ensure_initialized() is True
    assert engine.is_initialized is True
    assert init_calls == []


def test_resolve_midpoint_offset_bps_honors_annotation_and_clamp(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_MIDPOINT_LIMIT_MAX_OFFSET_BPS", "12")
    monkeypatch.setenv("AI_TRADING_MIDPOINT_LIMIT_MIN_OFFSET_BPS", "2")
    monkeypatch.setenv("AI_TRADING_MIDPOINT_LIMIT_HARD_CAP_BPS", "20")

    resolved = engine._resolve_midpoint_offset_bps(
        symbol="AAPL",
        annotations={"execution_aggressiveness_bps": 15},
        metadata=None,
    )
    clamped = engine._resolve_midpoint_offset_bps(
        symbol="AAPL",
        annotations={"execution_aggressiveness_bps": 50},
        metadata=None,
    )

    assert resolved == 15
    assert clamped == 20


def test_resolve_midpoint_offset_bps_applies_symbol_adaptive_ewma(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_MIDPOINT_LIMIT_MAX_OFFSET_BPS", "12")
    monkeypatch.setenv("AI_TRADING_MIDPOINT_LIMIT_MIN_OFFSET_BPS", "2")
    monkeypatch.setenv("AI_TRADING_MIDPOINT_LIMIT_HARD_CAP_BPS", "25")
    monkeypatch.setenv("AI_TRADING_ADAPTIVE_LIMIT_OFFSET_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ADAPTIVE_LIMIT_OFFSET_WEIGHT", "1.0")
    monkeypatch.setattr("ai_trading.execution.slippage_log.get_ewma_cost_bps", lambda _symbol, default=2.0: 18.0)

    resolved = engine._resolve_midpoint_offset_bps(
        symbol="AAPL",
        annotations=None,
        metadata=None,
    )

    assert resolved == 18.0


def test_warmup_data_only_mode_defaults_to_block_orders(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_WARMUP_MODE", "1")
    monkeypatch.delenv("AI_TRADING_WARMUP_ALLOW_ORDERS", raising=False)
    assert engine._warmup_data_only_mode_active() is True


def test_warmup_data_only_mode_can_allow_orders(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_WARMUP_MODE", "1")
    monkeypatch.setenv("AI_TRADING_WARMUP_ALLOW_ORDERS", "1")
    assert engine._warmup_data_only_mode_active() is False


def test_warmup_data_only_mode_does_not_block_after_active_phase(monkeypatch):
    engine = _engine_stub()
    engine.ctx = SimpleNamespace(state={"service_phase": "active"})
    monkeypatch.setenv("AI_TRADING_WARMUP_MODE", "1")
    monkeypatch.delenv("AI_TRADING_WARMUP_ALLOW_ORDERS", raising=False)

    assert engine._warmup_data_only_mode_active() is False


def test_execute_order_skips_submit_during_data_only_warmup(monkeypatch, caplog):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_WARMUP_MODE", "1")
    monkeypatch.delenv("AI_TRADING_WARMUP_ALLOW_ORDERS", raising=False)

    caplog.set_level(logging.INFO)
    result = engine.execute_order("AAPL", "buy", 1, order_type="market")

    assert result is None
    assert any(
        record.message == "ORDER_SUBMIT_SKIPPED"
        and getattr(record, "reason", None) == "warmup_data_only"
        for record in caplog.records
    )


def test_skip_submit_records_execution_quality_event(monkeypatch):
    engine = _engine_stub()
    captured: list[dict[str, Any]] = []

    monkeypatch.setattr(engine, "_runtime_exec_event_persistence_enabled", lambda: True)
    monkeypatch.setattr(
        engine,
        "_append_runtime_jsonl",
        lambda **kwargs: captured.append(dict(kwargs)),
    )

    engine._skip_submit(
        symbol="AAPL",
        side="buy",
        reason="unit_skip_reason",
        order_type="limit",
        detail="unit detail",
        context={"gate": "test"},
    )

    quality_rows = [
        row
        for row in captured
        if row.get("env_key") == "AI_TRADING_EXEC_QUALITY_EVENTS_PATH"
    ]
    assert quality_rows
    payload = quality_rows[-1]["payload"]
    assert payload["event"] == "submit_skipped"
    assert payload["status"] == "skipped"
    assert payload["reason"] == "unit_skip_reason"
    assert payload["symbol"] == "AAPL"
    assert payload["side"] == "buy"


def test_submit_failure_records_execution_quality_event(monkeypatch):
    engine = _engine_stub()
    captured: list[dict[str, Any]] = []

    monkeypatch.setattr(engine, "_runtime_exec_event_persistence_enabled", lambda: True)
    monkeypatch.setattr(
        engine,
        "_append_runtime_jsonl",
        lambda **kwargs: captured.append(dict(kwargs)),
    )

    engine._record_submit_failure(
        symbol="MSFT",
        side="sell",
        reason="unit_failure_reason",
        order_type="market",
        status_code=503,
        detail="upstream unavailable",
    )

    quality_rows = [
        row
        for row in captured
        if row.get("env_key") == "AI_TRADING_EXEC_QUALITY_EVENTS_PATH"
    ]
    assert quality_rows
    payload = quality_rows[-1]["payload"]
    assert payload["event"] == "submit_failed"
    assert payload["status"] == "failed"
    assert payload["reason"] == "unit_failure_reason"
    assert payload["symbol"] == "MSFT"
    assert payload["side"] == "sell"
    assert payload["status_code"] == 503


def test_execution_kpi_snapshot_and_alerts(monkeypatch, caplog):
    engine = _engine_stub()
    now_dt = datetime.now(UTC)
    engine._cycle_order_outcomes = [
        {"status": "pending_new", "duration_s": 12.0, "ack_timed_out": True},
        {"status": "canceled", "duration_s": 20.0, "ack_timed_out": False},
    ]
    engine._list_open_orders_snapshot = lambda: [
        SimpleNamespace(status="pending_new", created_at=now_dt - timedelta(seconds=180))
    ]

    monkeypatch.setenv("AI_TRADING_EXEC_KPI_ALERTS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_KPI_MIN_FILL_RATIO", "0.90")
    monkeypatch.setenv("AI_TRADING_KPI_MAX_CANCEL_RATIO", "0.10")
    monkeypatch.setenv("AI_TRADING_KPI_MAX_CANCEL_NEW_RATIO", "0.10")
    monkeypatch.setenv("AI_TRADING_KPI_MAX_MEDIAN_PENDING_SEC", "5")
    monkeypatch.setenv("AI_TRADING_KPI_PENDING_AGE_ALERT_SEC", "60")

    emitted: list[str] = []
    import ai_trading.monitoring.alerts as alerts_mod

    def _emit_runtime_alert(event: str, **kwargs):
        del kwargs
        emitted.append(event)

    monkeypatch.setattr(alerts_mod, "emit_runtime_alert", _emit_runtime_alert)

    caplog.set_level(logging.INFO)
    engine._emit_cycle_execution_kpis()

    assert any(record.message == "EXECUTION_KPI_SNAPSHOT" for record in caplog.records)
    assert "ALERT_EXEC_KPI_LOW_FILL_RATIO" in emitted
    assert "ALERT_EXEC_KPI_HIGH_CANCEL_RATIO" in emitted
    assert "ALERT_EXEC_KPI_HIGH_CANCEL_NEW_RATIO" in emitted
    assert "ALERT_EXEC_KPI_MEDIAN_PENDING_HIGH" in emitted
    assert "ALERT_EXEC_KPI_OPEN_PENDING_AGED" in emitted


def test_execution_kpi_snapshot_includes_lockout_and_skip_reasons(monkeypatch, caplog):
    engine = _engine_stub()
    engine._cycle_order_outcomes = [
        {"status": "skipped", "reason": "broker_lock", "duration_s": 0.1, "ack_timed_out": False},
        {"status": "failed", "reason": "submit_exception", "duration_s": 0.2, "ack_timed_out": False},
        {"status": "filled", "duration_s": 0.3, "ack_timed_out": False},
    ]
    engine._list_open_orders_snapshot = lambda: []
    engine._broker_lock_reason = "unauthorized"
    engine._broker_locked_until = 160.0
    monkeypatch.setattr(lt, "monotonic_time", lambda: 100.0)

    caplog.set_level(logging.INFO)
    engine._emit_cycle_execution_kpis()

    record = next(rec for rec in caplog.records if rec.message == "EXECUTION_KPI_SNAPSHOT")
    assert record.submitted == 1
    assert record.failed == 1
    assert record.skipped == 1
    assert record.skip_reason_counts == {"broker_lock": 1}
    assert record.broker_lock_active is True
    assert record.broker_lock_reason == "unauthorized"
    assert record.broker_lock_ttl_s == pytest.approx(60.0, abs=0.1)


def test_execution_kpi_snapshot_records_slo_metrics(monkeypatch):
    engine = _engine_stub()
    engine._cycle_order_outcomes = [
        {
            "status": "filled",
            "duration_s": 0.3,
            "ack_timed_out": False,
            "execution_drift_bps": 7.5,
            "realized_slippage_bps": 6.25,
        },
        {"status": "failed", "duration_s": 0.2, "ack_timed_out": False},
    ]
    engine._list_open_orders_snapshot = lambda: []

    import ai_trading.monitoring.slo as slo_mod

    reject_rate_samples: list[float] = []
    drift_samples: list[float] = []
    slippage_samples: list[float] = []
    pacing_cap_hit_rate_samples: list[float] = []
    monkeypatch.setattr(
        slo_mod,
        "record_order_reject_rate",
        lambda value: reject_rate_samples.append(float(value)),
    )
    monkeypatch.setattr(
        slo_mod,
        "record_execution_drift",
        lambda value: drift_samples.append(float(value)),
    )
    monkeypatch.setattr(
        slo_mod,
        "record_realized_slippage",
        lambda value: slippage_samples.append(float(value)),
    )
    monkeypatch.setattr(
        slo_mod,
        "record_order_pacing_cap_hit_rate",
        lambda value: pacing_cap_hit_rate_samples.append(float(value)),
    )

    engine._emit_cycle_execution_kpis()

    assert reject_rate_samples == pytest.approx([50.0])
    assert drift_samples == pytest.approx([7.5])
    assert slippage_samples == pytest.approx([6.25])
    assert pacing_cap_hit_rate_samples == pytest.approx([0.0])


def test_update_markout_feedback_tracks_only_live_sources(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setattr(
        engine,
        "_runtime_exec_event_persistence_enabled",
        lambda: False,
    )

    engine._update_markout_feedback(
        symbol="AAPL",
        side="buy",
        status="filled",
        realized_net_edge_bps=-5.0,
        realized_slippage_bps=4.0,
        fill_source="reconcile_backfill",
    )
    assert len(engine._markout_feedback_bps) == 0
    assert len(engine._slippage_feedback_bps) == 1

    engine._update_markout_feedback(
        symbol="AAPL",
        side="buy",
        status="filled",
        realized_net_edge_bps=-6.0,
        realized_slippage_bps=5.0,
        fill_source="live",
    )
    assert len(engine._markout_feedback_bps) == 1
    assert engine._markout_feedback_last_context["sample_count"] == 1


def test_execution_kpi_snapshot_includes_markout_feedback_fields(caplog):
    engine = _engine_stub()
    engine._cycle_order_outcomes = [
        {"status": "filled", "duration_s": 0.3, "ack_timed_out": False},
    ]
    engine._list_open_orders_snapshot = lambda: []
    engine._markout_feedback_last_context = {
        "sample_count": 8,
        "mean_bps": -3.25,
        "toxic": True,
        "threshold_bps": -4.0,
        "min_samples": 6,
    }
    engine._slippage_feedback_bps = deque([1.0, 3.0, 5.0, 7.0], maxlen=512)

    caplog.set_level(logging.INFO)
    engine._emit_cycle_execution_kpis()

    record = next(rec for rec in caplog.records if rec.message == "EXECUTION_KPI_SNAPSHOT")
    assert record.markout_sample_count == 8
    assert record.markout_toxic is True
    assert record.markout_mean_bps == pytest.approx(-3.25)
    assert record.slippage_volatility_bps > 0.0


def test_resolve_smart_order_route_applies_ioc_recommendation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine_stub()

    class _Router:
        @staticmethod
        def create_order_request(**_kwargs: Any) -> dict[str, Any]:
            return {
                "type": "ioc",
                "time_in_force": "IOC",
                "limit_price": 101.25,
            }

    monkeypatch.setattr(lt, "get_smart_router", lambda: _Router())
    context = engine._resolve_smart_order_route(
        symbol="AAPL",
        side="buy",
        quantity=5,
        order_type="limit",
        limit_price=100.9,
        bid=100.8,
        ask=101.0,
        quote_age_ms=300.0,
        degrade_active=False,
        markout_context={"toxic": False},
        manual_limit_requested=False,
    )

    assert context["enabled"] is True
    assert context["applied"] is True
    assert context["resolved_order_type"] == "limit"
    assert context["resolved_time_in_force"] == "ioc"


def test_estimate_passive_fill_probability_penalizes_toxic_markout() -> None:
    engine = _engine_stub()
    baseline_prob, _ = engine._estimate_passive_fill_probability(
        side="buy",
        bid=100.0,
        ask=100.2,
        quote_age_ms=400.0,
        spread_bps_hint=None,
        degrade_active=False,
        gap_ratio=0.01,
        markout_context={"toxic": False, "mean_bps": 1.0},
    )
    toxic_prob, context = engine._estimate_passive_fill_probability(
        side="buy",
        bid=100.0,
        ask=100.2,
        quote_age_ms=400.0,
        spread_bps_hint=None,
        degrade_active=False,
        gap_ratio=0.01,
        markout_context={"toxic": True, "mean_bps": -12.0},
    )

    assert toxic_prob < baseline_prob
    components = context["components"]
    assert components["markout_toxicity"] > 0.0
