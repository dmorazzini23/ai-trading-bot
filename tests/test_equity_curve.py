import pytest
pd = pytest.importorskip("pandas")
def test_equity_curve_monotonic():
    df = pd.read_csv("data/last_equity.txt", names=["equity"])
    assert df["equity"].is_monotonic_increasing or df["equity"].is_monotonic_decreasing, \
        "Equity curve is not smoothly trending (might indicate erratic jumps)"
