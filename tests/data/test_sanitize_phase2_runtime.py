from __future__ import annotations

import logging

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import sanitize
from ai_trading.data.sanitize import DataSanitizer, SanitizationConfig


def test_outlier_volume_and_stale_helpers_classify_specific_rows() -> None:
    idx = pd.date_range("2025-01-02 14:30", periods=24, freq="min", tz="UTC")
    idx = idx.delete(12).insert(12, pd.Timestamp("2025-01-02 18:00", tz="UTC"))
    close = [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.2, 9.8, 10.1, 9.9, 50.0, 10.0]
    close.extend([11.0, 11.0, 11.0, 11.0, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9])
    bars = pd.DataFrame(
        {
            "close": close,
            "trade_volume": [10, *([1_000] * 23)],
        },
        index=idx,
    )
    sanitizer = DataSanitizer(
        SanitizationConfig(
            mad_threshold=3.0,
            zscore_threshold=3.0,
            min_absolute_volume=100,
            min_volume_percentile=10.0,
            max_gap_hours=2.0,
            max_price_staleness=2,
            enable_price_validation=False,
            log_rejections=False,
        )
    )

    outlier_mask, outlier_reasons = sanitizer._detect_outliers(bars)
    volume_mask, volume_reasons = sanitizer._filter_low_volume(bars)
    stale_mask, stale_reasons = sanitizer._detect_stale_data(bars)

    assert bool(outlier_mask.iloc[10]) is True
    assert "outlier_close" in outlier_reasons.iloc[10]
    assert bool(volume_mask.iloc[0]) is True
    assert volume_reasons.iloc[0] in {
        "low_absolute_volume_trade_volume",
        "low_percentile_volume_trade_volume",
    }
    assert bool(stale_mask.iloc[12]) is True
    assert stale_reasons.iloc[12] == "time_gap"
    assert "stale_prices" in set(stale_reasons)

    stats = sanitizer.get_rejection_stats()
    assert stats["outliers"] >= 1
    assert stats["low_volume"] >= 1
    assert stats["stale_gaps"] == 1
    assert stats["stale_prices"] >= 1




