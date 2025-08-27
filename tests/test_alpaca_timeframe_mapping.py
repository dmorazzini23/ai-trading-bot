import sys
import types
from unittest.mock import MagicMock, patch

import pytest

pd = pytest.importorskip("pandas")
from ai_trading.alpaca_api import get_bars_df

from tests.helpers.asserts import assert_df_like

try:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except Exception:  # pragma: no cover - inject stub
    mod = types.ModuleType("alpaca.data.timeframe")

    class TimeFrameUnit:
        Day = type("Day", (), {"name": "Day"})()

    class TimeFrame:
        def __init__(self, amount, unit):
            self.amount = amount
            self.unit = unit

    mod.TimeFrame = TimeFrame
    mod.TimeFrameUnit = TimeFrameUnit
    sys.modules.setdefault("alpaca.data.timeframe", mod)
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


class _Resp:
    def __init__(self, df):
        self.df = df


@patch("ai_trading.alpaca_api._get_rest")
def test_day_timeframe_normalized(mock_rest_cls):
    mock_rest = MagicMock()
    mock_rest.get_bars.return_value = _Resp(pd.DataFrame({"open": [1.0], "close": [1.1]}))
    mock_rest_cls.return_value = mock_rest

    df = get_bars_df("SPY", "Day", feed="iex", adjustment="all")
    args, kwargs = mock_rest.get_bars.call_args
    assert kwargs["timeframe"] in ("1Day", "1D")
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode


@patch("ai_trading.alpaca_api._get_rest")
def test_tf_object_normalized(mock_rest_cls):
    mock_rest = MagicMock()
    mock_rest.get_bars.return_value = _Resp(pd.DataFrame({"open": [1.0], "close": [1.1]}))
    mock_rest_cls.return_value = mock_rest

    df = get_bars_df("SPY", TimeFrame(1, TimeFrameUnit.Day), feed="iex", adjustment="all")
    args, kwargs = mock_rest.get_bars.call_args
    assert kwargs["timeframe"] in ("1Day", "1D")
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode


@patch("ai_trading.alpaca_api._get_rest")
def test_minute_normalized(mock_rest_cls):
    mock_rest = MagicMock()
    mock_rest.get_bars.return_value = _Resp(pd.DataFrame({"open": [1.0], "close": [1.1]}))
    mock_rest_cls.return_value = mock_rest

    df = get_bars_df("SPY", "Minute", feed="iex", adjustment="all")
    args, kwargs = mock_rest.get_bars.call_args
    assert kwargs["timeframe"] in ("1Min", "1Minute")
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode
