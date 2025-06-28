import pandas as pd

def test_regime_changes():
    df = pd.read_csv("data/trades.csv")
    assert "regime" in df.columns, "Trades data missing regime column"
    assert df["regime"].nunique() > 1, "No regime changes detected, model might be static"
