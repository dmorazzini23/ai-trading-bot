import pandas as pd

def test_slippage_limits():
    df = pd.read_csv("logs/slippage.csv")
    assert df["slippage_cents"].abs().max() < 0.5, \
        "Slippage exceeded 50%, review execution quality"
