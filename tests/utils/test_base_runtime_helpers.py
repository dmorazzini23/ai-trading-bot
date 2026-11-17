from __future__ import annotations

import datetime as dt
import types
from enum import Enum
from types import MappingProxyType
from uuid import UUID

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.utils import base


class _Side(Enum):
    BUY = "buy"




def test_log_warning_routes_health_stale_to_debug(monkeypatch):
    calls: list[tuple[str, str, object | None, bool]] = []

    class Logger:
        def debug(self, msg, *args, extra=None, exc_info=False):
            calls.append(("debug", msg, extra, bool(exc_info)))

        def warning(self, msg, *args, extra=None, exc_info=False):
            calls.append(("warning", msg, extra, bool(exc_info)))

    monkeypatch.setattr(base, "logger", Logger())

    base.log_warning("HEALTH_STALE_DATA", exc=RuntimeError("stale"), extra={"symbol": "SPY"})
    base.log_warning("OTHER_EVENT", exc=RuntimeError("boom"), extra={"symbol": "QQQ"})
    base.log_warning("HEALTH_STALE_DATA", extra={"symbol": "IWM"})
    base.log_warning("OTHER_EVENT", extra={"symbol": "DIA"})

    assert calls == [
        ("debug", "%s: %s", {"symbol": "SPY"}, True),
        ("warning", "%s: %s", {"symbol": "QQQ"}, True),
        ("debug", "HEALTH_STALE_DATA", {"symbol": "IWM"}, False),
        ("warning", "OTHER_EVENT", {"symbol": "DIA"}, False),
    ]


def test_backoff_delay_caps_and_applies_jitter(monkeypatch):
    monkeypatch.setattr(base.random, "uniform", lambda low, high: high)

    assert base.backoff_delay(4, base=2.0, cap=10.0, jitter=0.0) == 10.0
    assert base.backoff_delay(2, base=2.0, cap=10.0, jitter=0.25) == 5.0




def test_callable_lock_supports_context_and_callable_usage():
    lock = base._CallableLock()

    with lock() as acquired:
        assert acquired is lock
        assert lock.locked() is True

    assert lock.locked() is False


def test_get_latest_close_handles_aliases_invalid_and_nonfinite():
    assert base.get_latest_close(None) == 0.0
    assert base.get_latest_close(pd.DataFrame({"Close": ["bad", "7.5"]})) == 7.5
    assert base.get_latest_close(pd.DataFrame({"c": [float("inf")]})) == 0.0
    assert base.get_latest_close(pd.DataFrame({"open": [1.0]})) == 0.0






def test_to_serializable_recurses_mapping_proxy_and_tuples():
    nested = MappingProxyType({"items": ({"value": 1}, MappingProxyType({"inner": 2}))})

    assert base.to_serializable(nested) == {"items": [{"value": 1}, {"inner": 2}]}




def test_safe_to_datetime_handles_empty_none_scalar_and_naive_output():
    assert len(base.safe_to_datetime(None)) == 0
    assert len(base.safe_to_datetime([], utc=False)) == 0

    scalar = base.safe_to_datetime("2026-01-02 09:30", utc=False)
    assert scalar.tz is None
    assert scalar[0] == pd.Timestamp("2026-01-02 09:30")


def test_validate_ohlcv_rejects_invalid_timestamps_and_order():
    valid = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "open": [1, 2],
            "high": [2, 3],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [100, 200],
        }
    )
    base.validate_ohlcv(valid)

    with pytest.raises(ValueError, match="timestamp contains"):
        bad_ts = valid.copy()
        bad_ts.loc[1, "timestamp"] = "not-a-date"
        base.validate_ohlcv(bad_ts)

    with pytest.raises(ValueError, match="not monotonic"):
        reversed_ts = valid.iloc[::-1].reset_index(drop=True)
        base.validate_ohlcv(reversed_ts)

    with pytest.raises(ValueError, match="no rows"):
        base.validate_ohlcv(valid.iloc[0:0], require_monotonic=False)


def test_health_check_handles_length_errors(monkeypatch):
    monkeypatch.setattr(base, "get_env", lambda *_args, **_kwargs: "2")

    assert base.health_check([1, 2]) is True
    assert base.health_check([1]) is False
    assert base.health_check(object()) is False


def test_pre_trade_health_check_collects_symbol_results(monkeypatch):
    monkeypatch.setattr(base, "check_symbol", lambda symbol, _api: symbol == "SPY")

    result = base.pre_trade_health_check(["SPY", "BAD"], api=object())

    assert result == {"SPY": True, "BAD": False}
