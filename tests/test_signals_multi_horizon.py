import pandas as pd
from indicators import compute_ema, compute_sma, compute_bollinger, compute_atr


def test_multi_horizon_indicators():
    df = pd.DataFrame({
        'close': [i for i in range(100)],
        'high': [i + 1 for i in range(100)],
        'low': [i - 1 for i in range(100)],
    })
    df = compute_ema(df)
    df = compute_sma(df)
    df = compute_bollinger(df)
    df = compute_atr(df)
    for p in [5, 20, 50, 200]:
        assert f'EMA_{p}' in df.columns
        assert f'SMA_{p}' in df.columns or p < 51
    assert 'UB' in df.columns and 'LB' in df.columns
