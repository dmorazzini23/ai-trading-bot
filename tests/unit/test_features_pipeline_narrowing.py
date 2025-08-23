"""Tests for narrowed exceptions in feature pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from ai_trading.features.pipeline import BuildFeatures


def test_transform_raises_value_error_on_bad_input() -> None:
    """Non-DataFrame input should raise ValueError."""  # AI-AGENT-REF: narrow exception test
    bf = BuildFeatures(include_regime=True, include_volatility=True)
    X = pd.DataFrame({"close": [1, 2, 3]})
    bf.fit(X)
    with pytest.raises(ValueError):
        bf.transform("bad")
