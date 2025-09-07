import pytest
from ai_trading.regime.filters import load_trades

pd = pytest.importorskip("pandas")


def test_regime_changes():
    df = load_trades()
    if df is None or "regime" not in df.columns:
        pytest.skip("Trades data missing regime column")

    # Filter out empty/null regime values and check for diversity
    regime_values = df["regime"].dropna()
    regime_values = regime_values[regime_values != ""]  # Remove empty strings

    if len(regime_values) == 0:
        pytest.skip("No valid regime data found")

    unique_regimes = regime_values.nunique()
    assert (
        unique_regimes > 1
    ), f"No regime changes detected ({unique_regimes} unique values), model might be static"
