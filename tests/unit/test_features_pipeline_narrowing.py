"""Tests for narrowed exceptions in feature pipeline."""
from __future__ import annotations


import math

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.features.pipeline import BuildFeatures


def test_transform_raises_value_error_on_bad_input() -> None:
    """Non-DataFrame input should raise ValueError."""  # AI-AGENT-REF: narrow exception test
    bf = BuildFeatures(include_regime=True, include_volatility=True)
    X = pd.DataFrame({"close": [1, 2, 3]})
    bf.fit(X)
    with pytest.raises(ValueError):
        bf.transform("bad")


def test_transform_rejects_invalid_raw_prices_before_fill() -> None:
    bf = BuildFeatures(include_regime=False, include_volatility=False)
    X = pd.DataFrame({"open": [1.0, 0.0], "high": [1.1, 1.2], "low": [0.9, 1.0], "close": [1.0, 1.1]})
    bf.fit(X)

    with pytest.raises(ValueError, match="finite and positive"):
        bf.transform(X)


def test_regime_without_volatility_keeps_short_history_thresholds_finite() -> None:
    bf = BuildFeatures(
        include_returns=False,
        include_volatility=False,
        include_volume=False,
        include_regime=True,
        regime_span=10,
    )
    X = pd.DataFrame({"close": [100.0]})

    bf.fit(X)
    transformed = bf.transform(X)

    assert math.isfinite(bf.feature_params_["regime_vol_low"])
    assert math.isfinite(bf.feature_params_["regime_vol_high"])
    assert transformed["vol_regime"].tolist() == [1]
    assert "vol_20d" not in transformed
