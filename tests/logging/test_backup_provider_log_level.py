from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from ai_trading.logging import log_backup_provider_used


def _window() -> tuple[datetime, datetime]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return start, start + timedelta(minutes=1)


def test_backup_provider_used_logs_info_for_configured_override(caplog) -> None:
    start, end = _window()
    with caplog.at_level(logging.INFO, logger="ai_trading.logging"):
        log_backup_provider_used(
            "yahoo",
            symbol="AAPL",
            timeframe="1Day",
            start=start,
            end=end,
            extra={"reason": "configured_source_override"},
        )

    records = [record for record in caplog.records if record.getMessage() == "BACKUP_PROVIDER_USED"]
    assert records
    assert records[-1].levelname == "INFO"


def test_backup_provider_used_logs_info_for_unavailable_primary(caplog) -> None:
    start, end = _window()
    with caplog.at_level(logging.INFO, logger="ai_trading.logging"):
        log_backup_provider_used(
            "yahoo",
            symbol="AAPL",
            timeframe="1Day",
            start=start,
            end=end,
            extra={"reason": "upstream_unavailable"},
        )

    records = [record for record in caplog.records if record.getMessage() == "BACKUP_PROVIDER_USED"]
    assert records
    assert records[-1].levelname == "INFO"
