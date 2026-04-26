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


def test_sanitize_bars_combines_reasons_and_resets_stats() -> None:
    idx = pd.date_range("2025-01-02 14:30", periods=3, freq="min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [10.0, -1.0, 10.0],
            "close": [10.0, float("inf"), 20.0],
            "volume": [1_000, 1, 1_000],
        },
        index=idx,
    )
    sanitizer = DataSanitizer(
        SanitizationConfig(
            min_absolute_volume=100,
            max_price_change=0.25,
            enable_outlier_detection=False,
            enable_stale_detection=False,
            log_rejections=False,
        )
    )

    cleaned, report = sanitizer.sanitize_bars(bars, symbol="BAD")

    assert list(cleaned.index) == [idx[0]]
    assert report["rejected_count"] == 2
    assert report["rejections"]["excessive_price_move"] == 2
    assert report["rejections"]["low_absolute_volume_volume"] == 1
    assert report["time_range"] == {"start": idx[0].isoformat(), "end": idx[-1].isoformat()}
    stats = sanitizer.get_rejection_stats()
    assert stats["invalid_prices"] == 2
    assert stats["rejection_rate"] > 0

    sanitizer.reset_stats()
    assert all(value == 0 for value in sanitizer.get_rejection_stats().values())


def test_debug_rejection_logging_and_winsorize_dataframe(monkeypatch) -> None:
    idx = pd.date_range("2025-01-02", periods=12, freq="min", tz="UTC")
    df = pd.DataFrame({"px": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200], "label": ["x"] * 12}, index=idx)
    sanitizer = DataSanitizer(SanitizationConfig(winsorize_limits=(0.1, 0.1)))
    warnings: list[str] = []
    debugs: list[str] = []

    class _Logger:
        def warning(self, message: str) -> None:
            warnings.append(message)

        def debug(self, message: str) -> None:
            debugs.append(message)

        def isEnabledFor(self, level: int) -> bool:
            return level == logging.DEBUG

    monkeypatch.setattr(sanitizer, "logger", _Logger())
    sanitizer._log_rejections("DBG", df.iloc[:4], pd.Series(["a", "b", "c", "d"], index=idx[:4]))

    assert "Rejected 4 bars for DBG" in warnings[0]
    assert len(debugs) == 3

    monkeypatch.setattr(sanitize, "_global_sanitizer", sanitizer)
    clipped = sanitize.winsorize_dataframe(df, columns=["px", "missing"], limits=(0.1, 0.1))
    assert clipped["px"].iloc[0] > df["px"].iloc[0]
    assert clipped["px"].iloc[-1] < df["px"].iloc[-1]
    assert sanitize.winsorize_dataframe(pd.DataFrame()).empty
