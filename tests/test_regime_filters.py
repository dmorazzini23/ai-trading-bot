import pandas as pd
import pytest

def test_regime_changes():
    df = pd.read_csv("data/trades.csv")
    if "regime" not in df.columns:
        pytest.skip("Trades data missing regime column")
    assert df["regime"].nunique() > 1, "No regime changes detected, model might be static"
