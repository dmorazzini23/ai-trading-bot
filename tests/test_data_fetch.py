import os
from datetime import UTC, datetime

import pytest
from ai_trading.alpaca_api import get_bars_df  # AI-AGENT-REF
from ai_trading.data.fetch import bars_time_window_day

try:  # pragma: no cover - optional dependency
    import alpaca
    from alpaca.data.timeframe import TimeFrame
except ImportError:  # pragma: no cover - optional dependency
    alpaca = None
    TimeFrame = None


@pytest.mark.skip(reason="requires valid Alpaca credentials")
def test_get_bars_df_spy_day():
    if not (os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY")):
        pytest.skip("missing Alpaca credentials")
    df = get_bars_df("SPY", TimeFrame.Day)
    assert len(df) > 0
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns


def test_bars_time_window_day():
    s_dt, e_dt = bars_time_window_day()
    assert e_dt <= datetime.now(UTC)
    assert (e_dt - s_dt).days == 10
    assert s_dt.hour == 0 and s_dt.minute == 0 and s_dt.second == 0 and s_dt.microsecond == 0
