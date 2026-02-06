import datetime as dt
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
    assert isinstance(req.start, dt.datetime) and req.start.tzinfo is None
    assert isinstance(req.end, dt.datetime) and req.end.tzinfo is None
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
    assert isinstance(req.start, dt.datetime) and req.start.tzinfo is None
    assert isinstance(req.end, dt.datetime) and req.end.tzinfo is None
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
    assert isinstance(req.start, dt.datetime) and req.start.tzinfo == dt.UTC
    assert isinstance(req.end, dt.datetime) and req.end.tzinfo == dt.UTC
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode


@patch("ai_trading.alpaca_api._get_rest")
def test_week_normalized(mock_rest_cls):
    mock_rest = MagicMock()
    mock_rest.get_stock_bars.return_value = _Resp(
        pd.DataFrame({"open": [1.0], "close": [1.1]})
    )
    mock_rest_cls.return_value = mock_rest

    df = get_bars_df("SPY", "1week", feed="iex", adjustment="all")
    mock_rest_cls.assert_called_once_with(bars=True)
    (req,), kwargs = mock_rest.get_stock_bars.call_args
    assert getattr(req.timeframe, "amount", None) == 1
    assert getattr(req.timeframe.unit, "name", "").lower() == "week"
    assert isinstance(req.start, dt.datetime)
    assert isinstance(req.end, dt.datetime)
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode


def test_timeframe_coerced_to_request_runtime_class(monkeypatch):
    import ai_trading.alpaca_api as api

    class _ExpectedUnit:
        Minute = type("Minute", (), {"name": "Minute"})()
        Day = type("Day", (), {"name": "Day"})()

    class _ExpectedTimeFrame:
        def __init__(self, amount=1, unit=_ExpectedUnit.Day):
            self.amount = amount
            self.unit = unit

    _ExpectedTimeFrame.Minute = _ExpectedTimeFrame(1, _ExpectedUnit.Minute)
    _ExpectedTimeFrame.Day = _ExpectedTimeFrame(1, _ExpectedUnit.Day)

    class _StrictRequest:
        def __init__(self, **kwargs):
            timeframe = kwargs.get("timeframe")
            if not isinstance(timeframe, _ExpectedTimeFrame):
                raise TypeError("timeframe_type_mismatch")
            self.__dict__.update(kwargs)

    captured: dict[str, object] = {}

    class _Rest:
        def get_stock_bars(self, req):
            captured["request"] = req
            return _Resp(pd.DataFrame({"open": [1.0], "close": [1.1]}))

    monkeypatch.setattr(api, "_get_rest", lambda bars=True: _Rest())
    monkeypatch.setattr(api, "get_stock_bars_request_cls", lambda: _StrictRequest)
    monkeypatch.setattr(
        api,
        "_data_classes",
        lambda: (_StrictRequest, _ExpectedTimeFrame, _ExpectedUnit),
    )

    foreign_tf = types.SimpleNamespace(
        amount=5,
        unit=types.SimpleNamespace(name="Minute"),
    )
    monkeypatch.setattr(
        api,
        "_normalize_timeframe_for_tradeapi",
        lambda _tf: ("5Minute", foreign_tf),
    )

    start = dt.datetime(2025, 8, 19, 15, 0, tzinfo=dt.UTC)
    end = dt.datetime(2025, 8, 19, 16, 0, tzinfo=dt.UTC)
    df = api.get_bars_df("SPY", "5Min", start=start, end=end, feed="iex", adjustment="all")
    assert_df_like(df)

    request = captured.get("request")
    assert request is not None
    assert isinstance(request.timeframe, _ExpectedTimeFrame)
    assert request.timeframe.amount == 5
    assert request.timeframe.unit is _ExpectedUnit.Minute
