from __future__ import annotations

from ai_trading.telemetry import runtime_state


def test_update_broker_status_clears_stale_last_error_on_recovery() -> None:
    runtime_state.reset_all_states()

    runtime_state.update_broker_status(
        connected=False,
        status="degraded",
        last_error="timeout",
    )
    runtime_state.update_broker_status(
        connected=True,
        status="reachable",
    )

    snapshot = runtime_state.observe_broker_status()
    assert snapshot["connected"] is True
    assert snapshot["status"] == "reachable"
    assert snapshot["last_error"] is None


def test_observe_data_provider_state_returns_deep_copy() -> None:
    runtime_state.reset_all_states()
    runtime_state.update_data_provider_state(
        active="alpaca",
        timeframe="1Min",
        using_backup=True,
    )

    snapshot = runtime_state.observe_data_provider_state()
    snapshot["timeframes"]["1Min"] = False

    latest = runtime_state.observe_data_provider_state()
    assert latest["timeframes"]["1Min"] is True
