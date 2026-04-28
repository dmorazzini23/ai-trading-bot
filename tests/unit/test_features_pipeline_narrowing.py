"""Tests for narrowed exceptions in feature pipeline."""
from __future__ import annotations


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
