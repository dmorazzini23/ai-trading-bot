from __future__ import annotations

from ai_trading.data import bars as bars_mod


def test_minute_fallback_debug_path_emits_record(caplog):
    caplog.set_level("DEBUG")
    bars_mod.fetch_minute_fallback(None, "SPY", now_utc=bars_mod.now_utc())
    records = [r for r in caplog.records if r.message == "DATA_FALLBACK_WINDOW_DEBUG"]
    assert records, "expected debug window log"  # AI-AGENT-REF: ensure debug path coverage
