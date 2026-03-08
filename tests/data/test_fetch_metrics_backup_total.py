from __future__ import annotations

from ai_trading.data.fetch.metrics import (
    backup_provider_used_total,
    inc_backup_provider_used,
    reset,
)


def test_backup_provider_used_total_supports_provider_filter() -> None:
    reset()
    inc_backup_provider_used("yahoo", "AAPL", increment=True)
    inc_backup_provider_used("yahoo", "MSFT", increment=True)
    inc_backup_provider_used("finnhub", "AAPL", increment=True)

    assert backup_provider_used_total() == 3
    assert backup_provider_used_total("yahoo") == 2
    assert backup_provider_used_total("finnhub") == 1
