import pytest
pd = pytest.importorskip("pandas")

from ai_trading import signals


def test_prepare_indicators_requires_open_column():
    df = pd.DataFrame({
        "high": [2.0],
        "low": [1.0],
        "close": [1.5],
        "volume": [100],
    })
    with pytest.raises(KeyError, match=r"missing required column\(s\): \['open'\]"):
        signals.prepare_indicators(df)
