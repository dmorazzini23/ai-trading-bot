from __future__ import annotations

import logging

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import sanitize
from ai_trading.data.sanitize import DataSanitizer, SanitizationConfig


def test_empty_frame_and_module_convenience_use_global_sanitizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sanitize, "_global_sanitizer", None)
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    cleaned, report = sanitize.sanitize_bars(empty, symbol="EMPTY", config=SanitizationConfig())

    assert cleaned is empty
    assert report == {"status": "empty", "rejections": {}}
    assert isinstance(sanitize.get_data_sanitizer(), DataSanitizer)


def test_mixed_case_and_non_string_columns_are_detected_without_crashing() -> None:
    idx = pd.date_range("2025-01-02 14:30", periods=3, freq="min", tz="UTC")
    bars = pd.DataFrame(
        {
            "Open": [10.0, 10.2, 10.3],
            "HIGH": [10.4, 10.5, 10.6],
            "low": [9.8, 9.9, 10.0],
            "ClosePrice": [10.1, 10.2, 10.3],
            "TotalVolume": [1_000, 25, 1_100],
            42: ["ignored", "metadata", "column"],
        },
        index=idx,
    )
    sanitizer = DataSanitizer(
        SanitizationConfig(
            min_absolute_volume=100,
            enable_outlier_detection=False,
            enable_stale_detection=False,
            log_rejections=False,
        )
    )

    cleaned, report = sanitizer.sanitize_bars(bars, symbol="CASE")

    assert list(cleaned.index) == [idx[0], idx[2]]
    assert report["rejections"] == {"low_absolute_volume_TotalVolume": 1}
    assert sanitizer._get_price_columns(bars) == ["Open", "HIGH", "low", "ClosePrice"]
    assert sanitizer._get_volume_columns(bars) == ["TotalVolume"]


def test_partial_inputs_nan_volume_and_non_datetime_range_are_safe() -> None:
    bars = pd.DataFrame(
        {
            "open_price": [10.0, 0.0, float("nan")],
            "shares": [float("nan"), float("nan"), float("nan")],
            "note": ["ok", "too-low", "bad-number"],
        },
        index=["a", "b", "c"],
    )
    sanitizer = DataSanitizer(
        SanitizationConfig(
            min_price=1.0,
            enable_outlier_detection=False,
            enable_stale_detection=True,
            log_rejections=False,
        )
    )

    cleaned, report = sanitizer.sanitize_bars(bars, symbol="PARTIAL")

    assert list(cleaned.index) == ["a"]
    assert report["time_range"] is None
    assert report["rejections"] == {
        "price_too_low_open_price": 1,
        "invalid_price_open_price": 1,
    }
    assert sanitizer._filter_low_volume(bars)[1].eq("").all()


def test_internal_missing_columns_and_blank_reasons_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2025-01-02 14:30", periods=2, freq="min", tz="UTC")
    bars = pd.DataFrame({"close": [10.0, 10.1], "volume": [100, 100]}, index=idx)
    sanitizer = DataSanitizer(SanitizationConfig(log_rejections=False))

    monkeypatch.setattr(sanitizer, "_get_price_columns", lambda _bars: ["close", "missing_price"])
    price_mask, price_reasons = sanitizer._validate_prices(bars)
    monkeypatch.setattr(sanitizer, "_get_price_columns", lambda _bars: ["missing_price"])
    outlier_mask, outlier_reasons = sanitizer._detect_outliers(bars)
    monkeypatch.setattr(sanitizer, "_get_volume_columns", lambda _bars: ["volume", "missing_volume"])
    volume_mask, volume_reasons = sanitizer._filter_low_volume(bars)

    assert price_mask.eq(False).all()
    assert price_reasons.eq("").all()
    assert outlier_mask.eq(False).all()
    assert outlier_reasons.eq("").all()
    assert volume_mask.eq(True).all()
    assert set(volume_reasons) == {"low_absolute_volume_volume"}
    assert sanitizer._count_rejection_reasons(pd.Series(["", "a,b", " b , c "])) == {
        "a": 1,
        "b": 2,
        "c": 1,
    }


def test_empty_rejection_logging_and_winsorize_defaults_select_numeric_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2025-01-02 14:30", periods=12, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "px": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0, 200.0],
            "qty": [100.0] * 11 + [10_000.0],
            "label": ["keep"] * 12,
        },
        index=idx,
    )
    sanitizer = DataSanitizer(SanitizationConfig(winsorize_limits=(0.2, 0.2)))
    warnings: list[str] = []

    class _Logger:
        def warning(self, message: str) -> None:
            warnings.append(message)

        def isEnabledFor(self, level: int) -> bool:
            return level == logging.DEBUG

    monkeypatch.setattr(sanitizer, "logger", _Logger())
    sanitizer._log_rejections("NONE", df.iloc[:0], pd.Series(dtype=str))
    monkeypatch.setattr(sanitize, "_global_sanitizer", sanitizer)

    short = pd.Series([1.0, 100.0])
    clipped = sanitize.winsorize_dataframe(df, limits=(0.2, 0.2))

    assert warnings == []
    assert sanitizer.winsorize_series(short) is short
    assert clipped["px"].iloc[0] > df["px"].iloc[0]
    assert clipped["px"].iloc[-1] < df["px"].iloc[-1]
    assert clipped["qty"].iloc[-1] < df["qty"].iloc[-1]
    assert clipped["label"].equals(df["label"])
