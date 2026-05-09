from __future__ import annotations

from ai_trading.telemetry import runtime_state


def test_provider_state_clears_backup_when_primary_recovers() -> None:
    runtime_state.reset_all_states()
    runtime_state.update_data_provider_state(
        primary="alpaca",
        active="yahoo",
        backup="yahoo",
        using_backup=True,
        status="degraded",
    )

    runtime_state.update_data_provider_state(
        primary="alpaca",
        active="alpaca",
        using_backup=False,
        status="healthy",
    )

    snapshot = runtime_state.observe_data_provider_state()
    assert snapshot["active"] == "alpaca"
    assert snapshot["using_backup"] is False
    assert snapshot["backup"] is None
