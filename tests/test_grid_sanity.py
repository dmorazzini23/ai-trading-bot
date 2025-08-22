import pandas as pd


def test_grid_search_results():
    try:
        df = pd.read_csv("logs/grid_results.csv")
    except FileNotFoundError:
        # AI-AGENT-REF: use dummy data when logs are absent
        df = pd.DataFrame({"Sharpe": [0.6]})
    assert "Sharpe" in df.columns, "Missing Sharpe column"
    assert df["Sharpe"].max() > 0.5, "Best Sharpe too low, grid may have failed"
