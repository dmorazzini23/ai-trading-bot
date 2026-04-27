import pytest
pd = pytest.importorskip("pandas")
def test_slippage_limits():
    df = pd.read_csv("logs/slippage.csv")
    if df.empty:
        max_slip = 0
    elif "slippage_bps" in df.columns:
        max_slip = df["slippage_bps"].abs().max() / 10000.0
    else:
        max_slip = df["slippage_cents"].abs().max()
    assert max_slip < 0.5, "Slippage exceeded 50%, review execution quality"
