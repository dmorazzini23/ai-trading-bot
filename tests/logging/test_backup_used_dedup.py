from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from ai_trading.data import fetch


def test_mark_fallback_emits_single_backup_log(caplog):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)

    original_windows = fetch._FALLBACK_WINDOWS.copy()
    original_metadata = fetch._FALLBACK_METADATA.copy()
    try:
        fetch._FALLBACK_WINDOWS.clear()
        fetch._FALLBACK_METADATA.clear()
        with caplog.at_level(logging.INFO, logger="ai_trading.data.fetch"):
            fetch._mark_fallback(
                "AAPL",
                "1Min",
                start,
                end,
                from_provider="alpaca_iex",
                resolved_provider="yahoo",
                resolved_feed=None,
            )
    finally:
        fetch._FALLBACK_WINDOWS.clear()
        fetch._FALLBACK_WINDOWS.update(original_windows)
        fetch._FALLBACK_METADATA.clear()
        fetch._FALLBACK_METADATA.update(original_metadata)

    backup_logs = [
        record for record in caplog.records if record.getMessage() == "BACKUP_PROVIDER_USED"
    ]
    assert len(backup_logs) == 1
    assert backup_logs[0].name.startswith("ai_trading.data.fetch")
