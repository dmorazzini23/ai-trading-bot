"""Pending-order monitoring should escalate with warning severity."""

from __future__ import annotations

import logging
import sys
import types

import pytest

if "ai_trading.indicators" not in sys.modules:
    indicators_stub = types.ModuleType("ai_trading.indicators")

    def _unavailable_indicator(*_args, **_kwargs):  # pragma: no cover - safety stub
        raise RuntimeError("Indicator module unavailable in tests")

    indicators_stub.compute_atr = _unavailable_indicator
    indicators_stub.atr = _unavailable_indicator
    indicators_stub.mean_reversion_zscore = _unavailable_indicator
    indicators_stub.rsi = _unavailable_indicator
    sys.modules["ai_trading.indicators"] = indicators_stub

if "ai_trading.signals" not in sys.modules:
    signals_stub = types.ModuleType("ai_trading.signals")
    signals_indicators_stub = types.ModuleType("ai_trading.signals.indicators")

    def _composite_confidence_stub(*_args, **_kwargs):  # pragma: no cover - safety stub
        return {}

    signals_indicators_stub.composite_signal_confidence = _composite_confidence_stub
    sys.modules["ai_trading.signals"] = signals_stub
    sys.modules["ai_trading.signals.indicators"] = signals_indicators_stub
    signals_stub.indicators = signals_indicators_stub

if "ai_trading.features" not in sys.modules:
    features_stub = types.ModuleType("ai_trading.features")
    features_indicators_stub = types.ModuleType("ai_trading.features.indicators")

    def _feature_passthrough(df, **_kwargs):  # pragma: no cover - safety stub
        return df

    features_indicators_stub.compute_macd = _feature_passthrough
    features_indicators_stub.compute_macds = _feature_passthrough
    features_indicators_stub.compute_vwap = _feature_passthrough
    features_indicators_stub.compute_atr = _feature_passthrough
    features_indicators_stub.compute_sma = _feature_passthrough
    features_indicators_stub.ensure_columns = _feature_passthrough
    sys.modules["ai_trading.features"] = features_stub
    sys.modules["ai_trading.features.indicators"] = features_indicators_stub
    features_stub.indicators = features_indicators_stub

if "portalocker" not in sys.modules:
    portalocker_stub = types.ModuleType("portalocker")
    portalocker_stub.LOCK_EX = 1

    def _noop_lock(*_args, **_kwargs):  # pragma: no cover - safety stub
        return None

    portalocker_stub.lock = _noop_lock
    portalocker_stub.unlock = _noop_lock
    sys.modules["portalocker"] = portalocker_stub

if "bs4" not in sys.modules:
    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:  # pragma: no cover - safety stub
        def __init__(self, *_args, **_kwargs):
            self.text = ""

        def find(self, *_args, **_kwargs):
            return None

    bs4_stub.BeautifulSoup = _BeautifulSoup
    sys.modules["bs4"] = bs4_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = dotenv_stub

if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.array = lambda *a, **k: a
    numpy_stub.ndarray = object
    numpy_stub.float64 = float
    numpy_stub.int64 = int
    numpy_stub.nan = float("nan")
    numpy_stub.NaN = float("nan")
    numpy_stub.random = types.SimpleNamespace(seed=lambda *_a, **_k: None)
    sys.modules["numpy"] = numpy_stub

from ai_trading.core import bot_engine as be


def _order(status: str, oid: str = "order-1") -> types.SimpleNamespace:
    return types.SimpleNamespace(id=oid, status=status)


@pytest.fixture(autouse=True)
def _reset_pending_tracker():
    be._PENDING_ORDER_STATUSES  # ensure module imported for coverage
    yield


def test_pending_orders_log_warning_levels(monkeypatch, caplog):
    """Pending-order logs escalate from info to warning as age grows."""

    runtime = types.SimpleNamespace(state={})
    cancel_called: list[types.SimpleNamespace] = []

    monkeypatch.setattr(
        be,
        "cancel_all_open_orders",
        lambda rt: cancel_called.append(rt),
    )
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=60),
    )

    clock = types.SimpleNamespace(value=1000.0)
    monkeypatch.setattr(be.time, "time", lambda: clock.value)

    caplog.set_level(logging.INFO)

    pending = [_order("pending_new", "alpha")]

    assert be._handle_pending_orders(pending, runtime) is True
    assert caplog.records[0].message == "PENDING_ORDERS_DETECTED"
    assert caplog.records[0].levelno == logging.INFO

    caplog.clear()
    clock.value += be._PENDING_ORDER_LOG_INTERVAL_SECONDS + 1
    assert be._handle_pending_orders(pending, runtime) is False
    assert cancel_called == [runtime]
    messages = [rec.message for rec in caplog.records]
    assert "PENDING_ORDERS_STILL_PRESENT" in messages
    assert "PENDING_ORDERS_CANCELED" in messages
    warning_logs = [rec for rec in caplog.records if rec.message == "PENDING_ORDERS_STILL_PRESENT"]
    assert warning_logs and warning_logs[0].levelno == logging.WARNING

    caplog.clear()
    clock.value += 1000
    assert be._handle_pending_orders(pending, runtime) is True
    detection_logs = [
        rec for rec in caplog.records if rec.message == "PENDING_ORDERS_DETECTED"
    ]
    assert detection_logs and detection_logs[0].levelno == logging.INFO


def test_pending_orders_log_escalates_to_error(monkeypatch, caplog):
    """Long-lived pending orders escalate to error-level logs."""

    runtime = types.SimpleNamespace(state={})
    cancel_called: list[types.SimpleNamespace] = []

    monkeypatch.setattr(
        be,
        "cancel_all_open_orders",
        lambda rt: cancel_called.append(rt),
    )
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(
            order_stale_cleanup_interval=600,
            orders_pending_new_warn_s=60,
            orders_pending_new_error_s=180,
        ),
    )

    clock = types.SimpleNamespace(value=1000.0)
    monkeypatch.setattr(be.time, "time", lambda: clock.value)

    caplog.set_level(logging.INFO)
    pending = [_order("pending_new", "alpha")]

    assert be._handle_pending_orders(pending, runtime) is True
    caplog.clear()
    clock.value += 181.0
    assert be._handle_pending_orders(pending, runtime) is False
    assert cancel_called == [runtime]
    still_present = [
        rec for rec in caplog.records if rec.message == "PENDING_ORDERS_STILL_PRESENT"
    ]
    assert still_present and still_present[0].levelno == logging.ERROR


def test_pending_orders_symbol_scope_caps_error_to_warning(monkeypatch, caplog):
    """Per-symbol pending blocking should not escalate still-present logs to error."""

    runtime = types.SimpleNamespace(
        state={},
        execution_engine=types.SimpleNamespace(_apply_pending_new_timeout_policy=lambda: True),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_ORDERS_BLOCK_SCOPE", "symbol")
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(
            order_stale_cleanup_interval=600,
            orders_pending_new_warn_s=60,
            orders_pending_new_error_s=120,
        ),
    )
    monkeypatch.setattr(
        be,
        "_pending_order_broker_age_seconds",
        lambda _order, _now_dt: 300.0,
    )
    monkeypatch.setattr(be, "cancel_all_open_orders", lambda _runtime: None)

    clock = types.SimpleNamespace(value=1000.0)
    monkeypatch.setattr(be.time, "time", lambda: clock.value)

    caplog.set_level(logging.INFO)
    pending = [_order("pending_new", "alpha")]
    assert be._handle_pending_orders(pending, runtime) is True
    clock.value += be._PENDING_ORDER_LOG_INTERVAL_SECONDS + 1
    assert be._handle_pending_orders(pending, runtime) is True
    levels = [
        rec.levelno
        for rec in caplog.records
        if rec.message == "PENDING_ORDERS_STILL_PRESENT"
    ]
    assert levels
    assert max(levels) == logging.WARNING


def test_pending_orders_first_detection_stays_info_for_fresh_orders(monkeypatch, caplog):
    """First pending-order detection is informational even with aggressive thresholds."""

    runtime = types.SimpleNamespace(state={})

    monkeypatch.setattr(be, "cancel_all_open_orders", lambda _rt: None)
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(
            order_stale_cleanup_interval=60,
            orders_pending_new_warn_s=5,
            orders_pending_new_error_s=10,
        ),
    )
    monkeypatch.setattr(
        be,
        "_pending_order_broker_age_seconds",
        lambda _order, _now_dt: 8.0,
    )

    clock = types.SimpleNamespace(value=1000.0)
    monkeypatch.setattr(be.time, "time", lambda: clock.value)

    caplog.set_level(logging.INFO)
    pending = [_order("pending_new", "alpha")]
    assert be._handle_pending_orders(pending, runtime) is True
    assert caplog.records[0].message == "PENDING_ORDERS_DETECTED"
    assert caplog.records[0].levelno == logging.INFO
