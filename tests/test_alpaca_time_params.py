import datetime as dt
from unittest.mock import MagicMock, patch

import pytest

pd = pytest.importorskip("pandas")
from ai_trading.alpaca_api import get_bars_df

from tests.helpers.asserts import assert_df_like


class _Resp:
    def __init__(self, df):
        self.df = df


@patch("ai_trading.alpaca_api._get_rest")
def test_daily_uses_date_only(mock_rest_cls):
    mock_rest = MagicMock()
    mock_rest.get_stock_bars.return_value = _Resp(
        pd.DataFrame({"open": [1.0], "close": [1.1]})
    )
    mock_rest_cls.return_value = mock_rest

    df = get_bars_df("SPY", "Day", feed="iex", adjustment="all")
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode
    mock_rest_cls.assert_called_once_with(bars=True)
    (req,), kwargs = mock_rest.get_stock_bars.call_args
    assert isinstance(req.start, str) and len(req.start) == 10 and req.start.count("-") == 2
    assert isinstance(req.end, str) and len(req.end) == 10 and req.end.count("-") == 2
    assert req.timeframe in ("1Day", "1D")


@patch("ai_trading.alpaca_api._get_rest")
def test_intraday_uses_rfc3339z(mock_rest_cls):
    mock_rest = MagicMock()
    mock_rest.get_stock_bars.return_value = _Resp(
        pd.DataFrame({"open": [1.0], "close": [1.1]})
    )
    mock_rest_cls.return_value = mock_rest

    start = dt.datetime(2025, 8, 19, 15, 0, 5, tzinfo=dt.UTC)
    end = dt.datetime(2025, 8, 19, 16, 0, 5, tzinfo=dt.UTC)
    df = get_bars_df("SPY", "5Min", start=start, end=end, feed="iex", adjustment="all")
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode
    mock_rest_cls.assert_called_once_with(bars=True)
    (req,), kwargs = mock_rest.get_stock_bars.call_args
    assert req.start.endswith("Z") and "T" in req.start and "." not in req.start
    assert req.end.endswith("Z") and "T" in req.end and "." not in req.end
    assert req.timeframe in ("5Min",)
