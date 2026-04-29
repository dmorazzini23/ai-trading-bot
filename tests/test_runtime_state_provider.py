from ai_trading.telemetry import runtime_state


def test_data_provider_healthy_primary_recovery_clears_failure_fields() -> None:
    runtime_state.reset_data_provider_state()
    runtime_state.update_data_provider_state(
        status="failed",
        using_backup=True,
        reason="timeout",
        reason_detail="request timeout",
        consecutive_failures=3,
        last_error_at="2026-04-29T12:00:00+00:00",
        http_code=504,
        cooldown_sec=30.0,
        data_status="degraded",
        safe_mode=True,
    )

    runtime_state.update_data_provider_state(
        status="healthy",
        using_backup=False,
        primary="alpaca",
        active="alpaca",
    )

    snapshot = runtime_state.observe_data_provider_state()
    assert snapshot["status"] == "healthy"
    assert snapshot["using_backup"] is False
    assert snapshot["reason"] is None
    assert snapshot["reason_code"] is None
    assert snapshot["reason_detail"] is None
    assert snapshot["last_error_at"] is None
    assert snapshot["http_code"] is None
    assert snapshot["cooldown_sec"] is None
    assert snapshot["data_status"] is None
    assert snapshot["safe_mode"] is False
    assert snapshot["consecutive_failures"] == 0
