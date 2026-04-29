from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from ai_trading.core.run_all_trades_prelude import prepare_run_all_trades_cycle


def _base_state() -> Any:
    return SimpleNamespace(
        run_manifest_written=True,
        running=False,
        last_run_at=None,
        trade_cooldowns={},
        startup_cleanup_done=True,
        halt_trading=False,
        halt_reason=None,
        execution_metrics=None,
    )


def _base_runtime() -> Any:
    return SimpleNamespace(api=object(), execution_engine=None)


def _base_kwargs() -> dict[str, Any]:
    now = datetime(2026, 4, 19, 14, 0, tzinfo=UTC)
    logger = SimpleNamespace(
        warning=lambda *a, **k: None,
        info=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    return {
        "risk_engine": SimpleNamespace(wait_for_exposure_update=lambda timeout: None),
        "logger": logger,
        "loop_id": "loop-1",
        "run_interval_seconds": 60.0,
        "ensure_execution_engine_func": lambda runtime: None,
        "enforce_dependency_preflight_func": lambda runtime: None,
        "resolve_trading_config_func": lambda runtime: SimpleNamespace(
            rth_only=True,
            allow_extended=False,
            post_submit_broker_sync=True,
        ),
        "active_effective_policy_func": lambda state, cfg: SimpleNamespace(
            trading_mode="paper",
            objective=SimpleNamespace(objective_name="growth"),
        ),
        "policy_config_error_type": RuntimeError,
        "write_run_manifest_func": lambda *a, **k: None,
        "restore_exit_policy_state_func": lambda state: False,
        "ensure_exit_policy_state_func": lambda state: {},
        "get_trade_cooldown_min_func": lambda: 15.0,
        "is_market_open_func": lambda: True,
        "log_market_closed_func": lambda message: None,
        "record_broker_sync_metrics_func": lambda state, snapshot: None,
        "monotonic_time_func": lambda: 123.0,
        "persist_effective_policy_snapshot_func": lambda state, policy, loop_id: None,
        "execution_metrics_factory": lambda: {"metrics": True},
        "ensure_alpaca_attached_func": lambda runtime: None,
        "validate_trading_api_func": lambda api: True,
        "startup_cancel_mode_func": lambda: "off",
        "list_open_orders_func": lambda api: [],
        "startup_cancel_decision_func": lambda orders, mode: (False, {"mode": mode}),
        "cancel_open_orders_subset_func": lambda *a, **k: SimpleNamespace(total_open=0, cancelled=0, failed=0),
        "select_startup_stale_orders_func": lambda orders: [],
        "cancel_all_open_orders_oms_func": lambda runtime: SimpleNamespace(total_open=0, cancelled=0, failed=0),
        "arm_pending_cleanup_warmup_func": lambda *a, **k: None,
        "provider_monitor": SimpleNamespace(
            is_safe_mode_active=lambda: False,
            safe_mode_cycle_marker=lambda: (None, None),
        ),
        "safe_mode_blocks_trading_func": lambda: False,
        "safe_mode_reason_func": lambda: None,
        "cancel_all_open_orders_func": lambda runtime: None,
        "reset_cycle_cache_func": lambda: None,
        "get_strategies_func": lambda: ["s1"],
        "log_loop_heartbeat_func": lambda loop_id, loop_start: None,
        "emit_test_capture_func": lambda message, level: None,
        "common_exceptions": (RuntimeError, ValueError),
        "_now": now,
    }


def test_prepare_run_all_trades_cycle_blocks_recent_run(monkeypatch) -> None:
    state = _base_state()
    runtime = _base_runtime()
    kwargs = _base_kwargs()
    now = kwargs.pop("_now")
    state.last_run_at = now - timedelta(seconds=10)
    result = prepare_run_all_trades_cycle(
        state=state,
        runtime=runtime,
        utc_now_func=lambda: now,
        **kwargs,
    )

    assert result.ready is False
    assert state.running is False


def test_prepare_run_all_trades_cycle_blocks_market_closed_and_syncs(monkeypatch) -> None:
    state = _base_state()
    runtime = _base_runtime()
    synced: list[str] = []
    kwargs = _base_kwargs()
    now = kwargs.pop("_now")
    runtime.execution_engine = SimpleNamespace(synchronize_broker_state=lambda: {"ok": True})
    kwargs["is_market_open_func"] = lambda: False
    kwargs["record_broker_sync_metrics_func"] = lambda state, snapshot: synced.append("synced")
    result = prepare_run_all_trades_cycle(
        state=state,
        runtime=runtime,
        utc_now_func=lambda: now,
        **kwargs,
    )

    assert result.ready is False
    assert synced == ["synced"]


def test_prepare_run_all_trades_cycle_initializes_cycle_state(monkeypatch) -> None:
    state = _base_state()
    runtime = _base_runtime()
    kwargs = _base_kwargs()
    now = kwargs.pop("_now")
    state.minute_feed_cache = {"AAPL": 1}
    state.cycle_order_intents = {"x": 1}
    state.cycle_submit_compaction = {"y"}
    result = prepare_run_all_trades_cycle(
        state=state,
        runtime=runtime,
        utc_now_func=lambda: now,
        **kwargs,
    )

    assert result.ready is True
    assert result.now == now
    assert result.loop_start == 123.0
    assert state.running is True
    assert state.last_run_at == now
    assert state.minute_feed_cache == {}
    assert state.cycle_order_intents == {}
    assert state.cycle_submit_compaction == set()
    assert state._strategies_loaded is True
    assert runtime.strategies == ["s1"]


def test_prepare_run_all_trades_cycle_rolls_back_active_state_on_late_failure(
    monkeypatch,
) -> None:
    state = _base_state()
    runtime = _base_runtime()
    kwargs = _base_kwargs()
    now = kwargs.pop("_now")
    previous_last_run_at = now - timedelta(minutes=10)
    state.last_run_at = previous_last_run_at
    state.minute_feed_cache = {"AAPL": 1}
    state.cycle_order_intents = {"x": 1}
    state.cycle_submit_compaction = {"y"}
    kwargs["get_strategies_func"] = lambda: (_ for _ in ()).throw(
        RuntimeError("strategy load failed")
    )

    try:
        prepare_run_all_trades_cycle(
            state=state,
            runtime=runtime,
            utc_now_func=lambda: now,
            **kwargs,
        )
    except RuntimeError as exc:
        assert str(exc) == "strategy load failed"
    else:
        raise AssertionError("expected strategy load failure")

    assert state.running is False
    assert state.last_run_at == previous_last_run_at
    assert state._strategies_loaded is False
