from tests.optdeps import require
require("pandas")
import pandas as pd
import pytest


def test_regime_changes():
    df = pd.read_csv(
        "data/trades.csv",
        engine="python",
        on_bad_lines="skip",
        skip_blank_lines=True,
    )
    if "regime" not in df.columns:
        pytest.skip("Trades data missing regime column")

    # Filter out empty/null regime values and check for diversity
    regime_values = df["regime"].dropna()
    regime_values = regime_values[regime_values != ""]  # Remove empty strings

    if len(regime_values) == 0:
        pytest.skip("No valid regime data found")

    unique_regimes = regime_values.nunique()
    assert unique_regimes > 1, f"No regime changes detected ({unique_regimes} unique values), model might be static"
