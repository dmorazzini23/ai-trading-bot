from datetime import UTC, datetime, timedelta
import logging

from ai_trading.data import provider_monitor


def test_safe_mode_auto_clears(monkeypatch, tmp_path, caplog):
    halt_path = tmp_path / "halt.flag"
    monkeypatch.setenv("AI_TRADING_HALT_FLAG_PATH", str(halt_path))
    monkeypatch.setenv("AI_TRADING_SAFE_MODE_HEALTH_PASSES", "2")
    monkeypatch.setattr(provider_monitor, "_resolve_halt_flag_path", lambda: str(halt_path))
    monkeypatch.setattr(provider_monitor, "_SAFE_MODE_RECOVERY_TARGET", 2, raising=False)
    provider_monitor.provider_monitor.reset()
    caplog.set_level(logging.INFO, logger="ai_trading.data.provider_monitor")

    provider_monitor._trigger_provider_safe_mode("minute_gap", count=3, metadata={"gap_ratio": 0.5})
    assert halt_path.exists()

    past = datetime.now(UTC) - timedelta(seconds=1)
    provider_monitor.provider_monitor.disabled_until["alpaca"] = past
    provider_monitor.provider_monitor.disabled_until["alpaca_sip"] = past
    provider_monitor.provider_monitor._last_switchover_provider = "alpaca"

    for _ in range(2):
        provider_monitor.provider_monitor.record_health_pass(
            True,
            provider="alpaca",
            gap_ratio=0.001,
            quote_timestamp_present=True,
        )

    assert provider_monitor.is_safe_mode_active() is False
    assert not halt_path.exists()
    assert any(rec.msg == "PROVIDER_SAFE_MODE_CLEARED" for rec in caplog.records)
    provider_monitor.provider_monitor.reset()
