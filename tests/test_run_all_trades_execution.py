from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai_trading.core import bot_engine
from ai_trading.core import run_all_trades_execution as execution_module
from ai_trading.core.run_all_trades_execution import execute_run_all_trades_cycle


def _clear_parity_cache() -> None:
    execution_module._REPLAY_LIVE_PARITY_GATE_CACHE["updated_mono"] = 0.0
    execution_module._REPLAY_LIVE_PARITY_GATE_CACHE["gate"] = None


def test_execute_run_all_trades_cycle_blocks_on_required_replay_live_parity_gate(
    monkeypatch,
) -> None:
    _clear_parity_cache()
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE", "1")

    updates: list[dict[str, object]] = []
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "update_service_status",
        lambda **kwargs: updates.append(kwargs),
    )
    monkeypatch.setattr(
        bot_engine,
        "list_open_orders",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not list orders")),
    )
    monkeypatch.setattr(
        "ai_trading.tools.runtime_performance_report.summarize_oms_lifecycle_parity",
        lambda: {"enabled": True, "available": True, "ok": True, "total_violations": 0},
    )
    monkeypatch.setattr(
        "ai_trading.governance.replay_live_parity.summarize_replay_live_parity_gate",
        lambda **_k: {
            "enabled": True,
            "available": True,
            "ok": False,
            "status": "fail",
            "reason": "replay_violations",
            "failed_checks": ["replay_violations"],
        },
    )

    state = SimpleNamespace()
    runtime = SimpleNamespace()

    execute_run_all_trades_cycle(
        state=state,
        runtime=runtime,
        cfg_runtime=SimpleNamespace(post_submit_broker_sync=False),
        loop_id="loop-1",
        loop_start=0.0,
        api=object(),
        restore_last_run_timestamp=lambda: None,
    )

    assert state.replay_live_parity_gate["ok"] is False
    assert runtime.replay_live_parity_gate["reason"] == "replay_violations"
    assert updates == [
        {"status": "degraded", "reason": "replay_live_parity_gate_failed"}
    ]


def test_execute_run_all_trades_cycle_requires_replay_live_parity_by_default_outside_pytest(
    monkeypatch,
) -> None:
    _clear_parity_cache()
    updates: list[dict[str, object]] = []

    def _fake_get_env(key, default=None, cast=None, **_kwargs):
        if key == "PYTEST_CURRENT_TEST":
            return ""
        if key == "PYTEST_RUNNING":
            return False
        return default

    monkeypatch.setattr(
        "ai_trading.core.run_all_trades_execution.get_env",
        _fake_get_env,
    )
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "update_service_status",
        lambda **kwargs: updates.append(kwargs),
    )
    monkeypatch.setattr(
        bot_engine,
        "list_open_orders",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not list orders")),
    )
    monkeypatch.setattr(
        "ai_trading.tools.runtime_performance_report.summarize_oms_lifecycle_parity",
        lambda: {"enabled": True, "available": True, "ok": True, "total_violations": 0},
    )
    monkeypatch.setattr(
        "ai_trading.governance.replay_live_parity.summarize_replay_live_parity_gate",
        lambda **_k: {
            "enabled": True,
            "available": True,
            "ok": False,
            "status": "fail",
            "reason": "replay_violations",
            "failed_checks": ["replay_violations"],
        },
    )

    state = SimpleNamespace()
    runtime = SimpleNamespace()

    execute_run_all_trades_cycle(
        state=state,
        runtime=runtime,
        cfg_runtime=SimpleNamespace(post_submit_broker_sync=False),
        loop_id="loop-2",
        loop_start=0.0,
        api=object(),
        restore_last_run_timestamp=lambda: None,
    )

    assert state.replay_live_parity_gate["ok"] is False
    assert runtime.replay_live_parity_gate["reason"] == "replay_violations"
    assert updates == [
        {"status": "degraded", "reason": "replay_live_parity_gate_failed"}
    ]


def test_replay_live_parity_gate_degrades_when_oms_summary_unavailable(monkeypatch) -> None:
    _clear_parity_cache()
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE", "1")

    updates: list[dict[str, object]] = []
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "update_service_status",
        lambda **kwargs: updates.append(kwargs),
    )
    monkeypatch.setattr(
        "ai_trading.tools.runtime_performance_report.summarize_oms_lifecycle_parity",
        lambda: (_ for _ in ()).throw(TimeoutError("pool exhausted")),
    )

    state = SimpleNamespace()
    runtime = SimpleNamespace()

    execution_module._evaluate_replay_live_parity_gate(
        state=state,
        runtime=runtime,
        be=bot_engine,
    )

    assert state.replay_live_parity_gate["ok"] is False
    assert state.replay_live_parity_gate["available"] is False
    assert state.replay_live_parity_gate["reason"] == "replay_live_parity_gate_error"
    assert updates == [
        {"status": "degraded", "reason": "replay_live_parity_gate_failed"}
    ]


def test_replay_live_parity_gate_unavailable_result_is_not_cached(monkeypatch) -> None:
    _clear_parity_cache()
    assert (
        execution_module._cacheable_replay_live_parity_gate(
            {
                "enabled": True,
                "available": False,
                "ok": False,
                "status": "fail",
                "reason": "replay_live_parity_gate_error",
            }
        )
        is False
    )


def test_replay_live_parity_gate_cache_still_marks_required_failure_degraded(
    monkeypatch,
) -> None:
    _clear_parity_cache()
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE", "1")
    monkeypatch.setenv("AI_TRADING_REPLAY_LIVE_PARITY_GATE_CYCLE_TTL_SEC", "300")
    execution_module._REPLAY_LIVE_PARITY_GATE_CACHE["updated_mono"] = (
        execution_module.time.monotonic()
    )
    execution_module._REPLAY_LIVE_PARITY_GATE_CACHE["gate"] = {
        "enabled": True,
        "available": True,
        "ok": False,
        "status": "fail",
        "reason": "replay_violations",
        "failed_checks": ["replay_violations"],
    }

    updates: list[dict[str, object]] = []
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "update_service_status",
        lambda **kwargs: updates.append(kwargs),
    )
    monkeypatch.setattr(
        "ai_trading.tools.runtime_performance_report.summarize_oms_lifecycle_parity",
        lambda: (_ for _ in ()).throw(AssertionError("cache should be used")),
    )

    state = SimpleNamespace()
    runtime = SimpleNamespace()

    gate = execution_module._evaluate_replay_live_parity_gate(
        state=state,
        runtime=runtime,
        be=bot_engine,
    )

    assert gate["reason"] == "replay_violations"
    assert "cache_age_s" in gate
    assert updates == [
        {"status": "degraded", "reason": "replay_live_parity_gate_failed"}
    ]


def test_execute_run_all_trades_cycle_accepts_native_get_orders(monkeypatch) -> None:
    _clear_parity_cache()
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE", "0")
    listed: list[object] = []
    handled: list[list[object]] = []

    monkeypatch.setattr(
        bot_engine,
        "list_open_orders",
        lambda api: listed.append(api) or [SimpleNamespace(id="ord-1")],
    )
    monkeypatch.setattr(
        bot_engine,
        "_handle_pending_orders",
        lambda open_orders, _runtime: handled.append(list(open_orders)) or True,
    )
    monkeypatch.setattr(bot_engine, "_pending_orders_block_scope", lambda: "global")

    class NativeOrdersOnly:
        def get_orders(self, **_kwargs):
            return []

    api = NativeOrdersOnly()

    execute_run_all_trades_cycle(
        state=SimpleNamespace(),
        runtime=SimpleNamespace(),
        cfg_runtime=SimpleNamespace(post_submit_broker_sync=False),
        loop_id="loop-get-orders",
        loop_start=0.0,
        api=api,
        restore_last_run_timestamp=lambda: None,
    )

    assert listed == [api]
    assert handled == [[SimpleNamespace(id="ord-1")]]


def test_replay_live_parity_gate_disabled_when_not_required(monkeypatch) -> None:
    _clear_parity_cache()
    monkeypatch.setenv("AI_TRADING_REPLAY_LIVE_PARITY_GATE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE", "0")

    gate = execution_module._evaluate_replay_live_parity_gate(
        state=SimpleNamespace(),
        runtime=SimpleNamespace(),
        be=bot_engine,
    )

    assert gate == {
        "enabled": False,
        "available": False,
        "ok": True,
        "status": "disabled",
    }


@pytest.mark.parametrize(
    ("raw_ttl", "expected"),
    [
        ("-1", 0.0),
        ("1200", 900.0),
        ("45.5", 45.5),
    ],
)
def test_replay_live_parity_gate_ttl_is_clamped(
    monkeypatch,
    raw_ttl: str,
    expected: float,
) -> None:
    monkeypatch.setenv("AI_TRADING_REPLAY_LIVE_PARITY_GATE_CYCLE_TTL_SEC", raw_ttl)

    assert execution_module._replay_live_parity_gate_cache_ttl_seconds() == expected


def test_execute_cycle_missing_list_orders_degrades_to_pending_skip(monkeypatch) -> None:
    _clear_parity_cache()
    monkeypatch.setenv("AI_TRADING_REPLAY_LIVE_PARITY_GATE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE", "0")
    monkeypatch.setattr(bot_engine, "_decision_log_runtime_path", lambda: "runtime/test-decisions.jsonl")
    monkeypatch.setattr(bot_engine, "_handle_pending_orders", lambda orders, runtime: True)
    monkeypatch.setattr(bot_engine, "_pending_orders_block_scope", lambda: "account")

    state = SimpleNamespace()

    execute_run_all_trades_cycle(
        state=state,
        runtime=SimpleNamespace(),
        cfg_runtime=SimpleNamespace(post_submit_broker_sync=False),
        loop_id="loop-missing-orders",
        loop_start=0.0,
        api=object(),
        restore_last_run_timestamp=lambda: None,
    )

    assert state._warned_missing_list_orders is True


def test_execute_cycle_open_order_failure_records_dependency_breaker(monkeypatch) -> None:
    _clear_parity_cache()
    monkeypatch.setenv("AI_TRADING_REPLAY_LIVE_PARITY_GATE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE", "0")
    monkeypatch.setattr(bot_engine, "_decision_log_runtime_path", lambda: "runtime/test-decisions.jsonl")
    monkeypatch.setattr(
        bot_engine,
        "list_open_orders",
        lambda _api: (_ for _ in ()).throw(TimeoutError("broker slow")),
    )
    monkeypatch.setattr(bot_engine, "_handle_pending_orders", lambda orders, runtime: True)
    monkeypatch.setattr(bot_engine, "_pending_orders_block_scope", lambda: "account")

    breaker_failures: list[tuple[str, object]] = []
    handled_errors: list[object] = []
    breaker_info = {"dependency": "broker_open_orders", "kind": "timeout"}
    monkeypatch.setattr(
        bot_engine,
        "classify_exception",
        lambda exc, dependency: breaker_info,
    )
    monkeypatch.setattr(
        bot_engine,
        "_dependency_breakers",
        lambda _state: SimpleNamespace(
            record_failure=lambda dependency, info: breaker_failures.append((dependency, info))
        ),
    )
    monkeypatch.setattr(
        bot_engine,
        "_handle_error",
        lambda info, **_kwargs: handled_errors.append(info),
    )

    class _Api:
        @staticmethod
        def list_orders() -> list[object]:
            return []

    execute_run_all_trades_cycle(
        state=SimpleNamespace(),
        runtime=SimpleNamespace(),
        cfg_runtime=SimpleNamespace(post_submit_broker_sync=False),
        loop_id="loop-open-order-failure",
        loop_start=0.0,
        api=_Api(),
        restore_last_run_timestamp=lambda: None,
    )

    assert breaker_failures == [("broker_open_orders", breaker_info)]
    assert handled_errors == [breaker_info]


def test_execute_cycle_logs_symbol_scoped_pending_block_then_runs_netting(
    monkeypatch,
) -> None:
    _clear_parity_cache()
    monkeypatch.setenv("AI_TRADING_REPLAY_LIVE_PARITY_GATE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE", "0")
    monkeypatch.setattr(bot_engine, "_decision_log_runtime_path", lambda: "runtime/test-decisions.jsonl")
    monkeypatch.setattr(bot_engine, "list_open_orders", lambda _api: [])
    monkeypatch.setattr(bot_engine, "_handle_pending_orders", lambda orders, runtime: True)
    monkeypatch.setattr(bot_engine, "_pending_orders_block_scope", lambda: "symbol")
    monkeypatch.setattr(bot_engine, "_resolve_runtime_info_log_ttl_seconds", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(bot_engine, "_should_emit_runtime_info_log", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(bot_engine, "_netting_pipeline_enabled", lambda _runtime: True)
    monkeypatch.setattr(bot_engine, "_run_netting_cycle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_engine, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_data_provider_state",
        lambda: {"status": "healthy", "using_backup": False},
    )
    data_provider_updates: list[dict[str, object]] = []
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "update_data_provider_state",
        lambda **kwargs: data_provider_updates.append(kwargs),
    )
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(
        bot_engine,
        "log_throttled_event",
        lambda _logger, _event, **kwargs: emitted.append(kwargs),
    )

    runtime = SimpleNamespace()
    setattr(runtime, bot_engine._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR, {"msft", "aapl"})

    execute_run_all_trades_cycle(
        state=SimpleNamespace(),
        runtime=runtime,
        cfg_runtime=SimpleNamespace(post_submit_broker_sync=False),
        loop_id="loop-symbol-pending",
        loop_start=0.0,
        api=SimpleNamespace(list_orders=lambda: []),
        restore_last_run_timestamp=lambda: None,
    )

    assert emitted
    assert emitted[0]["message"] == "PENDING_ORDERS_SYMBOL_BLOCK_ACTIVE"
    extra = cast(dict[str, Any], emitted[0]["extra"])
    assert extra["blocked_symbols_count"] == 2
    assert extra["blocked_symbols"] == ["AAPL", "MSFT"]
    assert data_provider_updates[0]["status"] == "healthy"
