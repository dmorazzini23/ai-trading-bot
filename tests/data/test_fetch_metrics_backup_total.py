from __future__ import annotations

from ai_trading.data.fetch import metrics


def test_backup_provider_used_total_supports_provider_filter() -> None:
    metrics.reset()
    metrics.inc_backup_provider_used("yahoo", "AAPL", increment=True)
    metrics.inc_backup_provider_used("yahoo", "MSFT", increment=True)
    metrics.inc_backup_provider_used("finnhub", "AAPL", increment=True)

    assert metrics.backup_provider_used_total() == 3
    assert metrics.backup_provider_used_total("yahoo") == 2
    assert metrics.backup_provider_used_total("finnhub") == 1
