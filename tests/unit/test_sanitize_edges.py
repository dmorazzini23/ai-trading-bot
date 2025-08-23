from __future__ import annotations

import pandas as pd

from ai_trading.data.sanitize import DataSanitizer, SanitizationConfig


def test_sanitize_handles_non_datetime_index_and_missing_columns():
    df = pd.DataFrame({"close": [10.0, 10.1, 10.2, 10.0]})
    s = DataSanitizer(SanitizationConfig())
    cleaned, report = s.sanitize_bars(df.copy(), symbol="TEST")
    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(report, dict)
    assert report.get("time_range") is None


def test_sanitize_flags_negative_prices_and_low_volume():
    idx = pd.date_range("2025-01-01", periods=5, freq="min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [10.0, 10.1, -9.9, 10.2, 10.3],
            "high": [10.1, 10.2, -9.8, 10.3, 10.4],
            "low": [9.9, 10.0, -10.0, 10.1, 10.2],
            "close": [10.0, 10.1, -9.7, 10.2, 10.3],
            "volume": [0, 5, 0, 2, 3],
        },
        index=idx,
    )
    s = DataSanitizer(SanitizationConfig(min_absolute_volume=1, min_volume_percentile=10.0))
    cleaned, report = s.sanitize_bars(bars.copy(), symbol="TEST")
    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(report, dict)
    assert report["original_count"] == 5
    assert 0 <= report["cleaned_count"] <= 5
    assert report["rejected_count"] >= 1

