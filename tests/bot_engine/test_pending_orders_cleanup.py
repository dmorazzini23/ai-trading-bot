"""Tests for pending-order reconciliation logic in the bot engine."""

from datetime import UTC, datetime, timedelta
import logging
import sys
import types
from unittest.mock import MagicMock

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

from ai_trading.core import bot_engine as be


def _order(status: str, oid: str = "order-1", **extra):
    payload = {"id": oid, "status": status}
    payload.update(extra)
    return types.SimpleNamespace(**payload)


def test_handle_pending_orders_grace_then_cleanup(monkeypatch, caplog):
    runtime = types.SimpleNamespace(state={})
    cancel_mock = MagicMock()
    monkeypatch.setattr(be, "cancel_all_open_orders", cancel_mock)
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=12),
    )
    clock = types.SimpleNamespace(value=100.0)
    monkeypatch.setattr(be.time, "time", lambda: clock.value)

    caplog.set_level(logging.INFO)
    orders = [_order("pending_new", "o-1")]

    assert be._handle_pending_orders(orders, runtime) is True
    first_record = caplog.records[0]
    assert first_record.message == "PENDING_ORDERS_DETECTED"
    assert first_record.pending_ids == ["o-1"]
    assert first_record.cleanup_after_s == 12
    assert cancel_mock.call_count == 0
    tracker = runtime.state[be._PENDING_ORDER_TRACKER_KEY]
    assert tracker[be._PENDING_ORDER_FIRST_SEEN_KEY] == clock.value

    clock.value = 105.0
    caplog.clear()
    assert be._handle_pending_orders(orders, runtime) is True
    assert cancel_mock.call_count == 0
    assert all(record.message != "PENDING_ORDERS_CANCELED" for record in caplog.records)

    clock.value = 120.0
    caplog.clear()
    assert be._handle_pending_orders(orders, runtime) is False
    cancel_mock.assert_called_once_with(runtime)
    messages = [record.message for record in caplog.records]
    assert "PENDING_ORDERS_CANCELED" in messages
    tracker = runtime.state[be._PENDING_ORDER_TRACKER_KEY]
    assert tracker[be._PENDING_ORDER_FIRST_SEEN_KEY] is None
    assert tracker[be._PENDING_ORDER_LAST_LOG_KEY] is None


def test_handle_pending_orders_clears_tracker(monkeypatch, caplog):
    runtime = types.SimpleNamespace(state={})
    tracker = runtime.state.setdefault(
        be._PENDING_ORDER_TRACKER_KEY,
        {
            be._PENDING_ORDER_FIRST_SEEN_KEY: 95.0,
            be._PENDING_ORDER_LAST_LOG_KEY: 95.0,
        },
    )
    monkeypatch.setattr(be.time, "time", lambda: 150.0)
    caplog.set_level(logging.INFO)

    assert be._handle_pending_orders([], runtime) is False
    assert tracker[be._PENDING_ORDER_FIRST_SEEN_KEY] is None
    assert tracker[be._PENDING_ORDER_LAST_LOG_KEY] is None
    assert any(record.message == "PENDING_ORDERS_CLEARED" for record in caplog.records)


def test_handle_pending_orders_creates_state(monkeypatch):
    runtime = types.SimpleNamespace()
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=15),
    )
    monkeypatch.setattr(be.time, "time", lambda: 10.0)
    orders = [_order("pending_new", "o-state")]

    assert be._handle_pending_orders(orders, runtime) is True
    assert isinstance(getattr(runtime, "state", None), dict)
    tracker = runtime.state[be._PENDING_ORDER_TRACKER_KEY]
    assert tracker[be._PENDING_ORDER_FIRST_SEEN_KEY] == 10.0


def test_handle_pending_orders_force_cleanup_for_stuck_pending(monkeypatch):
    runtime = types.SimpleNamespace(state={})
    cancel_mock = MagicMock()
    monkeypatch.setattr(be, "cancel_all_open_orders", cancel_mock)
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=300),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_FORCE_CANCEL_SEC", "30")

    clock = types.SimpleNamespace(value=100.0)
    monkeypatch.setattr(be.time, "time", lambda: clock.value)
    orders = [_order("pending_new", "o-force")]

    assert be._handle_pending_orders(orders, runtime) is True
    clock.value = 131.0
    assert be._handle_pending_orders(orders, runtime) is False
    cancel_mock.assert_called_once_with(runtime)


def test_handle_pending_orders_stale_broker_age_triggers_immediate_cleanup(monkeypatch):
    runtime = types.SimpleNamespace(state={})
    cancel_mock = MagicMock()
    monkeypatch.setattr(be, "cancel_all_open_orders", cancel_mock)
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=300),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_FORCE_CANCEL_SEC", "30")

    now_dt = datetime(2026, 2, 9, 18, 0, 0, tzinfo=UTC)
    now_epoch = now_dt.timestamp()
    monkeypatch.setattr(be.time, "time", lambda: now_epoch)
    stale_order = _order(
        "pending_new",
        "o-stale",
        created_at=now_dt - timedelta(seconds=120),
    )

    assert be._handle_pending_orders([stale_order], runtime) is False
    cancel_mock.assert_called_once_with(runtime)


def test_handle_pending_orders_cleanup_warmup(monkeypatch):
    runtime = types.SimpleNamespace(state={})
    cancel_mock = MagicMock()
    monkeypatch.setattr(be, "cancel_all_open_orders", cancel_mock)
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=10),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_CLEANUP_WARMUP_CYCLES", "2")

    clock = types.SimpleNamespace(value=100.0)
    monkeypatch.setattr(be.time, "time", lambda: clock.value)
    orders = [_order("pending_new", "o-warm")]

    assert be._handle_pending_orders(orders, runtime) is True
    clock.value = 112.0
    assert be._handle_pending_orders(orders, runtime) is True
    cancel_mock.assert_called_once_with(runtime)

    # One additional warmup cycle should be skipped even after pending clears.
    assert be._handle_pending_orders([], runtime) is True
    assert be._handle_pending_orders([], runtime) is False


def test_startup_cancel_decision_stale_only(monkeypatch):
    now = datetime(2026, 2, 17, 14, 0, tzinfo=UTC)
    monkeypatch.setattr(be, "datetime", types.SimpleNamespace(now=lambda tz=None: now))
    monkeypatch.setenv("AI_TRADING_STARTUP_CANCEL_STALE_SEC", "60")

    stale_order = _order("pending_new", "stale-1", created_at=now - timedelta(seconds=120))
    fresh_order = _order("pending_new", "fresh-1", created_at=now - timedelta(seconds=20))
    terminal_order = _order("filled", "filled-1", created_at=now - timedelta(seconds=300))

    should_cancel, details = be._startup_cancel_decision(
        [fresh_order, terminal_order],
        mode="stale_only",
    )
    assert should_cancel is False
    assert details["reason"] == "no_stale_pending"

    should_cancel, details = be._startup_cancel_decision(
        [fresh_order, stale_order],
        mode="stale_only",
    )
    assert should_cancel is True
    assert details["reason"] == "stale_pending"
    assert details["stale_ids"] == ["stale-1"]
