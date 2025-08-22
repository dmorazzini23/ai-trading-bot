import os
from datetime import UTC, datetime

import pytest

from ai_trading.alpaca_api import _bars_time_window, get_bars_df  # AI-AGENT-REF
from ai_trading.utils.optional_import import optional_import

alpaca = optional_import("alpaca_trade_api")
TimeFrame = optional_import("alpaca_trade_api.rest", "TimeFrame")

pytestmark = pytest.mark.skipif(
    alpaca is None or TimeFrame is None, reason="vendor modules not installed"
)


@pytest.mark.requires_credentials
def test_get_bars_df_spy_day():
    if not (os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY")):
        pytest.skip("missing Alpaca credentials")
    df = get_bars_df("SPY", TimeFrame.Day)
    assert len(df) > 0
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns


def test_bars_time_window_day():
    start, end = _bars_time_window(TimeFrame.Day)
    s_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    e_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    assert e_dt <= datetime.now(UTC)
    assert (e_dt - s_dt).days >= 10
