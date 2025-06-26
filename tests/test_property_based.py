import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import risk_engine
import signals
import utils


@given(
    prices=st.lists(
        st.floats(min_value=0, allow_nan=False, allow_infinity=False),
        min_size=35,
        max_size=60,
    )
)
def test_calculate_macd_basic(prices):
    ser = pd.Series(prices)
    df = signals.calculate_macd(ser)
    assert df is not None
    assert list(df.columns) == ["macd", "signal", "histogram"]
    assert len(df) == len(ser)
    assert not df.isna().any().any()


@given(
    prices=st.lists(
        st.floats(allow_nan=True, allow_infinity=False), min_size=1, max_size=30
    )
)
def test_calculate_macd_invalid(prices):
    ser = pd.Series(prices)
    if ser.isna().any() or len(ser) < 35:
        assert signals.calculate_macd(ser) is None
    else:
        pytest.skip("valid series covered in other test")


@given(
    values=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    )
)
def test_compute_volatility_matches_numpy(values):
    arr = np.array(values, dtype=float)
    eng = risk_engine.RiskEngine()
    res = eng.compute_volatility(arr)
    assert res["volatility"] == pytest.approx(float(np.std(arr)))


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(rows=st.integers(min_value=0, max_value=20))
def test_health_check_row_threshold(monkeypatch, rows):
    monkeypatch.setenv("HEALTH_MIN_ROWS", "10")
    df = pd.DataFrame({"close": [1] * rows})
    ok = utils.health_check(df, "daily")
    assert ok == (rows >= 10)
