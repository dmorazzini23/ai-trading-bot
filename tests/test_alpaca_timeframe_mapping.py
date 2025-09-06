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
        Minute = type("Minute", (), {"name": "Minute"})()

    class TimeFrame:
        def __init__(self, amount=1, unit=TimeFrameUnit.Day):
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
    mock_rest.get_stock_bars.return_value = _Resp(
        pd.DataFrame({"open": [1.0], "close": [1.1]})
    )
    mock_rest_cls.return_value = mock_rest

    df = get_bars_df("SPY", "1Day", feed="iex", adjustment="all")
    mock_rest_cls.assert_called_once_with(bars=True)
    (req,), kwargs = mock_rest.get_stock_bars.call_args
    assert getattr(req.timeframe, "amount", None) == 1
    assert getattr(req.timeframe.unit, "name", "") == "Day"
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode


@patch("ai_trading.alpaca_api._get_rest")
def test_tf_zero_arg_normalized(mock_rest_cls):
    mock_rest = MagicMock()
    mock_rest.get_stock_bars.return_value = _Resp(
        pd.DataFrame({"open": [1.0], "close": [1.1]})
    )
    mock_rest_cls.return_value = mock_rest

    df = get_bars_df("SPY", TimeFrame(), feed="iex", adjustment="all")
    mock_rest_cls.assert_called_once_with(bars=True)
    (req,), kwargs = mock_rest.get_stock_bars.call_args
    assert getattr(req.timeframe, "amount", None) == 1
    assert getattr(req.timeframe.unit, "name", "") == "Day"
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode


@patch("ai_trading.alpaca_api._get_rest")
def test_minute_normalized(mock_rest_cls):
    mock_rest = MagicMock()
    mock_rest.get_stock_bars.return_value = _Resp(
        pd.DataFrame({"open": [1.0], "close": [1.1]})
    )
    mock_rest_cls.return_value = mock_rest

    df = get_bars_df("SPY", "Minute", feed="iex", adjustment="all")
    mock_rest_cls.assert_called_once_with(bars=True)
    (req,), kwargs = mock_rest.get_stock_bars.call_args
    assert getattr(req.timeframe, "amount", None) == 1
    assert getattr(req.timeframe.unit, "name", "") in {"Minute", "Min"}
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode
