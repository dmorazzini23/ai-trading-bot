import pandas as pd

def test_grid_search_results():
    df = pd.read_csv("logs/grid_results.csv")
    assert "Sharpe" in df.columns, "Missing Sharpe column"
    assert df["Sharpe"].max() > 0.5, "Best Sharpe too low, grid may have failed"
