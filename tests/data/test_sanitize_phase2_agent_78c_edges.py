from __future__ import annotations

import logging

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import sanitize
from ai_trading.data.sanitize import DataSanitizer, SanitizationConfig








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


