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

    df = get_bars_df("SPY", "1Day", feed="iex", adjustment="all")
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode
    mock_rest_cls.assert_called_once_with(bars=True)
    (req,), kwargs = mock_rest.get_stock_bars.call_args
    assert isinstance(req.start, dt.datetime) and req.start.tzinfo is None
    assert isinstance(req.end, dt.datetime) and req.end.tzinfo is None
    assert req.start.time() == dt.time(0)
    assert req.end.time() == dt.time(0)
    assert getattr(req.timeframe, "amount", None) == 1
    assert getattr(req.timeframe.unit, "name", "") == "Day"


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
    assert isinstance(req.start, dt.datetime) and req.start.tzinfo == dt.UTC
    assert isinstance(req.end, dt.datetime) and req.end.tzinfo == dt.UTC
    assert req.start == start.astimezone(dt.UTC)
    assert req.end == end.astimezone(dt.UTC)
    assert getattr(req.timeframe, "amount", None) == 5
    assert getattr(req.timeframe.unit, "name", "") in {"Minute", "Min"}


@patch("ai_trading.alpaca_api._get_rest")
def test_warns_if_end_date_in_future(mock_rest_cls, caplog):
    mock_rest = MagicMock()
    mock_rest.get_stock_bars.return_value = _Resp(pd.DataFrame())
    mock_rest_cls.return_value = mock_rest

    future = dt.date.today() + dt.timedelta(days=1)
    start = dt.datetime.combine(future - dt.timedelta(days=1), dt.time(0, tzinfo=dt.UTC))
    end = dt.datetime.combine(future, dt.time(0, tzinfo=dt.UTC))

    with caplog.at_level("WARNING"):
        get_bars_df("SPY", "1Day", start=start, end=end, feed="iex", adjustment="all")

    assert any("END_DATE_AFTER_TODAY" in rec.message for rec in caplog.records)
