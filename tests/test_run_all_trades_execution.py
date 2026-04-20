from __future__ import annotations

from types import SimpleNamespace

from ai_trading.core import bot_engine
from ai_trading.core.run_all_trades_execution import execute_run_all_trades_cycle


def test_execute_run_all_trades_cycle_blocks_on_required_replay_live_parity_gate(
    monkeypatch,
) -> None:
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
