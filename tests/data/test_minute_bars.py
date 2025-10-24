import pandas as pd
import pytest

from ai_trading.config import get_trading_config
from ai_trading.data.fetch import _apply_incomplete_row_policy, _drop_last_bar_enabled


@pytest.fixture(autouse=True)
def _reset_config(monkeypatch):
    monkeypatch.setenv("DATA_DROP_LAST_PARTIAL_BAR", "1")
    get_trading_config.cache_clear()
    _drop_last_bar_enabled.cache_clear()
    yield
    monkeypatch.delenv("DATA_DROP_LAST_PARTIAL_BAR", raising=False)


def test_apply_incomplete_row_policy_drops_partial_and_nan():
    index = pd.date_range("2024-01-01 14:30", periods=3, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, pd.NA],
            "high": [101.0, 102.0, pd.NA],
            "low": [99.5, 100.5, pd.NA],
            "close": [100.5, pd.NA, pd.NA],
            "volume": [1000, 900, pd.NA],
        },
        index=index,
    )
    sanitized = _apply_incomplete_row_policy(frame, "AAPL", "1Min")
    assert len(sanitized) == 1
    assert sanitized.iloc[-1]["close"] == pytest.approx(100.5)
    assert sanitized["close"].isna().sum() == 0
