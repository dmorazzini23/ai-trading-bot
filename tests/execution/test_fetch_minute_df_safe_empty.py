import pandas as pd
import pytest

from ai_trading.core.bot_engine import fetch_minute_df_safe, DataFetchError


def test_empty_minute_raises(monkeypatch):
    """fetch_minute_df_safe should raise when providers return empty data."""
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.get_minute_df",
        lambda *a, **k: pd.DataFrame(),
    )
    with pytest.raises(DataFetchError):
        fetch_minute_df_safe("AAPL")
