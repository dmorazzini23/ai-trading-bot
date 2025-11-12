import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from ai_trading.core import bot_engine
from ai_trading.data import fetch
from ai_trading.data.fetch import notify_primary_provider_safe_mode


@pytest.fixture(autouse=True)
def _reset_safe_mode_state():
    original_state = dict(fetch._SAFE_MODE_CYCLE_STATE)
    original_logged = set(fetch._SAFE_MODE_LOGGED)
    try:
        fetch._SAFE_MODE_CYCLE_STATE.update({"cycle_id": None, "reason": None, "version": 0})
        fetch._SAFE_MODE_LOGGED.clear()
        yield
    finally:
        fetch._SAFE_MODE_CYCLE_STATE.clear()
        fetch._SAFE_MODE_CYCLE_STATE.update(original_state)
        fetch._SAFE_MODE_LOGGED.clear()
        fetch._SAFE_MODE_LOGGED.update(original_logged)


def test_safe_mode_cycle_skips_primary(monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO, logger="ai_trading.data.fetch")

    notify_primary_provider_safe_mode(reason="minute_gap")

    monkeypatch.setattr(fetch, "_ensure_pandas", lambda: pd)
    monkeypatch.setattr(fetch, "_last_complete_minute", lambda _pd: datetime.now(UTC))
    monkeypatch.setattr(fetch, "_used_fallback", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "get_fallback_metadata", lambda *a, **k: {})
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "_ensure_override_state_current", lambda: None)
    monkeypatch.setattr(fetch, "_resolve_backup_provider", lambda: ("yahoo", "yahoo"))
    monkeypatch.setattr(fetch, "_http_fallback_permitted", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "record_attempt", lambda *a, **k: 1)
    monkeypatch.setattr(fetch, "_record_provider_failure_event", lambda *a, **k: None)
    monkeypatch.setattr(fetch, "_record_provider_success_event", lambda *a, **k: None)
    monkeypatch.setattr(fetch, "_mark_fallback", lambda *a, **k: None)
    monkeypatch.setattr(fetch, "_incr", lambda *a, **k: None)
    monkeypatch.setattr(fetch, "inc_provider_fallback", lambda *a, **k: None)
    monkeypatch.setattr(fetch.provider_monitor, "active_provider", lambda primary, backup: backup)

    calls: list[tuple[str, str]] = []

    def fake_backup(symbol: str, start: datetime, end: datetime, *, interval: str) -> pd.DataFrame:
        calls.append((symbol, interval))
        return pd.DataFrame(
            [
                {
                    "timestamp": start,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 100,
                }
            ]
        )

    monkeypatch.setattr(fetch, "_safe_backup_get_bars", fake_backup)

    start = datetime.now(UTC) - timedelta(minutes=5)
    end = datetime.now(UTC)

    df = fetch.get_minute_df("AAPL", start, end)
    assert not df.empty
    assert calls

    skip_logs = [rec for rec in caplog.records if rec.msg == "PRIMARY_PROVIDER_DISABLED_CYCLE_SKIP"]
    assert len(skip_logs) == 1

    caplog.clear()
    df2 = fetch.get_minute_df("AAPL", start, end)
    assert not df2.empty
    assert len(calls) >= 2
    skip_logs_repeat = [rec for rec in caplog.records if rec.msg == "PRIMARY_PROVIDER_DISABLED_CYCLE_SKIP"]
    assert not skip_logs_repeat


def test_resolve_data_provider_degraded_minutes(monkeypatch):
    monkeypatch.setattr(bot_engine.runtime_state, "observe_data_provider_state", lambda: {})
    monkeypatch.setattr(bot_engine, "safe_mode_reason", lambda: "minute_gap")
    monkeypatch.setattr(bot_engine.provider_monitor, "is_disabled", lambda *_args, **_kwargs: False)

    degraded, reason, fatal = bot_engine._resolve_data_provider_degraded()

    assert degraded is True
    assert reason == "minute_gap"
    assert fatal is False
