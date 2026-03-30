from __future__ import annotations

from datetime import datetime, timezone

from ai_trading.training import after_hours


def test_run_after_hours_training_uses_non_overriding_reload(monkeypatch):
    reload_calls: list[dict[str, object]] = []

    def _fake_reload_env(path=None, override=True):
        reload_calls.append({"path": path, "override": override})
        return path

    monkeypatch.setattr(after_hours, "reload_env", _fake_reload_env)
    monkeypatch.setattr(after_hours, "refresh_default_feed", lambda: None)
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_CATCHUP_ENABLED", "0")

    # 15:00 UTC -> 11:00 America/New_York (before market close).
    result = after_hours.run_after_hours_training(
        now=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc)
    )

    assert reload_calls
    assert reload_calls[-1]["override"] is False
    assert result.get("status") == "skipped"
    assert result.get("reason") == "before_market_close"
