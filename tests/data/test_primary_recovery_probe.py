from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_trading.data import fetch


def test_request_primary_recovery_probe_expires_fallback_skip_and_marks_due() -> None:
    key = ("AAPL", "1Min")
    fetch._BACKUP_SKIP_UNTIL[key] = datetime.now(UTC) + timedelta(minutes=5)
    fetch._BACKUP_PRIMARY_PROBE_AT[key] = datetime.now(UTC) + timedelta(minutes=5)

    scheduled = fetch.request_primary_recovery_probe(["aapl"], timeframe="1Min")

    assert scheduled == 1
    assert key not in fetch._BACKUP_SKIP_UNTIL
    assert fetch._BACKUP_PRIMARY_PROBE_AT[key] <= datetime.now(UTC)
    assert fetch._backup_primary_probe_due("AAPL", "1Min") is True
