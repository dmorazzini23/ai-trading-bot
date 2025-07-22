import pandas as pd
import pytest

def test_grid_search_results():
    try:
        df = pd.read_csv("logs/grid_results.csv")
    except FileNotFoundError:
        pytest.skip("grid_results.csv not found")
    assert "Sharpe" in df.columns, "Missing Sharpe column"
    assert df["Sharpe"].max() > 0.5, "Best Sharpe too low, grid may have failed"
