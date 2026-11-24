"""Tests for pending-order reconciliation logic in the bot engine."""

from datetime import UTC, datetime, timedelta
import importlib
import logging
import sys
import types
from typing import Any, cast
from unittest.mock import MagicMock


def _set_module_attr(module: types.ModuleType, name: str, value: Any) -> None:
    setattr(cast(Any, module), name, value)


if "ai_trading.indicators" not in sys.modules:
    try:
        importlib.import_module("ai_trading.indicators")
    except (ImportError, RuntimeError, AttributeError, OSError):
        indicators_stub = types.ModuleType("ai_trading.indicators")

        def _unavailable_indicator(*_args, **_kwargs):  # pragma: no cover - safety stub
            raise RuntimeError("Indicator module unavailable in tests")

        _set_module_attr(indicators_stub, "compute_atr", _unavailable_indicator)
        _set_module_attr(indicators_stub, "atr", _unavailable_indicator)
        _set_module_attr(indicators_stub, "mean_reversion_zscore", _unavailable_indicator)
        _set_module_attr(indicators_stub, "rsi", _unavailable_indicator)
        sys.modules["ai_trading.indicators"] = indicators_stub

if "ai_trading.signals" not in sys.modules:
    try:
        importlib.import_module("ai_trading.signals")
        importlib.import_module("ai_trading.signals.indicators")
    except (ImportError, RuntimeError, AttributeError, OSError):
        signals_stub = types.ModuleType("ai_trading.signals")
        signals_indicators_stub = types.ModuleType("ai_trading.signals.indicators")

        def _composite_confidence_stub(*_args, **_kwargs):  # pragma: no cover - safety stub
            return {}

        _set_module_attr(signals_indicators_stub, "composite_signal_confidence", _composite_confidence_stub)
        sys.modules["ai_trading.signals"] = signals_stub
        sys.modules["ai_trading.signals.indicators"] = signals_indicators_stub
        _set_module_attr(signals_stub, "indicators", signals_indicators_stub)

if "ai_trading.features" not in sys.modules:
    try:
        importlib.import_module("ai_trading.features")
        importlib.import_module("ai_trading.features.indicators")
    except (ImportError, RuntimeError, AttributeError, OSError):
        features_stub = types.ModuleType("ai_trading.features")
        features_indicators_stub = types.ModuleType("ai_trading.features.indicators")

        def _feature_passthrough(df, **_kwargs):  # pragma: no cover - safety stub
            return df

        _set_module_attr(features_indicators_stub, "compute_macd", _feature_passthrough)
        _set_module_attr(features_indicators_stub, "compute_macds", _feature_passthrough)
        _set_module_attr(features_indicators_stub, "compute_vwap", _feature_passthrough)
        _set_module_attr(features_indicators_stub, "compute_atr", _feature_passthrough)
        _set_module_attr(features_indicators_stub, "compute_sma", _feature_passthrough)
        _set_module_attr(features_indicators_stub, "ensure_columns", _feature_passthrough)
        sys.modules["ai_trading.features"] = features_stub
        sys.modules["ai_trading.features.indicators"] = features_indicators_stub
        _set_module_attr(features_stub, "indicators", features_indicators_stub)

if "portalocker" not in sys.modules:
    portalocker_stub = types.ModuleType("portalocker")
    _set_module_attr(portalocker_stub, "LOCK_EX", 1)

    def _noop_lock(*_args, **_kwargs):  # pragma: no cover - safety stub
        return None

    _set_module_attr(portalocker_stub, "lock", _noop_lock)
    _set_module_attr(portalocker_stub, "unlock", _noop_lock)
    sys.modules["portalocker"] = portalocker_stub

if "bs4" not in sys.modules:
    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:  # pragma: no cover - safety stub
        def __init__(self, *_args, **_kwargs):
            self.text = ""

        def find(self, *_args, **_kwargs):
            return None

    _set_module_attr(bs4_stub, "BeautifulSoup", _BeautifulSoup)
    sys.modules["bs4"] = bs4_stub

from ai_trading.core import bot_engine as be


def _order(status: str, oid: str = "order-1", **extra):
    payload = {"id": oid, "status": status}
    payload.update(extra)
    return types.SimpleNamespace(**payload)


def test_handle_pending_orders_grace_then_cleanup(monkeypatch, caplog):
    runtime = types.SimpleNamespace(state={})
    cancel_all_mock = MagicMock()
    monkeypatch.setattr(be, "cancel_all_open_orders", cancel_all_mock)
    subset_calls: list[list[Any]] = []

    def _cancel_subset(_runtime, orders, **_kwargs):
        subset_calls.append(list(orders))
        return be.CancelAllResult(
            total_open=1,
            cancelled=1,
            failed=0,
            reason_code="PENDING_ORDERS_CLEANUP",
            errors=[],
        )

    monkeypatch.setattr(be, "_cancel_open_orders_subset", _cancel_subset)
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
    first_record = next(record for record in caplog.records if record.message == "PENDING_ORDERS_DETECTED")
    assert first_record.message == "PENDING_ORDERS_DETECTED"
    assert first_record.pending_ids == ["o-1"]
    assert first_record.cleanup_after_s == 12
    assert cancel_all_mock.call_count == 0
    assert subset_calls == []
    tracker = runtime.state[be._PENDING_ORDER_TRACKER_KEY]
    assert tracker[be._PENDING_ORDER_FIRST_SEEN_KEY] == clock.value

    clock.value = 105.0
    caplog.clear()
    assert be._handle_pending_orders(orders, runtime) is True
    assert cancel_all_mock.call_count == 0
    assert subset_calls == []
    assert all(record.message != "PENDING_ORDERS_CANCELED" for record in caplog.records)

    clock.value = 120.0
    caplog.clear()
    assert be._handle_pending_orders(orders, runtime) is False
    cancel_all_mock.assert_not_called()
    assert subset_calls == [orders]
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


def test_handle_pending_orders_tolerates_invalid_tracker_values(monkeypatch):
    runtime = types.SimpleNamespace(
        state={
            be._PENDING_ORDER_TRACKER_KEY: {
                be._PENDING_ORDER_FIRST_SEEN_KEY: "bad-first-seen",
                be._PENDING_ORDER_LAST_LOG_KEY: "bad-last-log",
            },
            be._PENDING_BACKLOG_ACTIVE_KEY: True,
            be._PENDING_BACKLOG_LAST_WARN_TS_KEY: "bad-last-warn",
        }
    )
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=15),
    )
    monkeypatch.setattr(be.time, "time", lambda: 50.0)
    orders = [_order("pending_new", "o-invalid", symbol="AAPL")]

    assert be._handle_pending_orders(orders, runtime) is True
    tracker = runtime.state[be._PENDING_ORDER_TRACKER_KEY]
    assert tracker[be._PENDING_ORDER_FIRST_SEEN_KEY] == 50.0
    assert tracker[be._PENDING_ORDER_LAST_LOG_KEY] == 50.0


def test_handle_pending_orders_force_cleanup_for_stuck_pending(monkeypatch):
    runtime = types.SimpleNamespace(state={})
    cancel_all_mock = MagicMock()
    monkeypatch.setattr(be, "cancel_all_open_orders", cancel_all_mock)
    subset_calls: list[list[Any]] = []
    monkeypatch.setattr(
        be,
        "_cancel_open_orders_subset",
        lambda _runtime, orders, **_kwargs: subset_calls.append(list(orders))
        or be.CancelAllResult(
            total_open=1,
            cancelled=1,
            failed=0,
            reason_code="PENDING_ORDERS_CLEANUP",
            errors=[],
        ),
    )
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
    cancel_all_mock.assert_not_called()
    assert subset_calls == [orders]


def test_handle_pending_orders_applies_stale_sweep_before_full_cleanup(monkeypatch, caplog):
    runtime = types.SimpleNamespace(state={})
    cancel_all_mock = MagicMock()
    monkeypatch.setattr(be, "cancel_all_open_orders", cancel_all_mock)
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=300),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_MAX_CANCELS", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_COOLDOWN_SEC", "0")

    now_dt = datetime(2026, 3, 1, 15, 0, 0, tzinfo=UTC)
    now_epoch = now_dt.timestamp()
    monkeypatch.setattr(be.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        be,
        "_cancel_open_orders_subset",
        lambda *_a, **_k: be.CancelAllResult(
            total_open=1,
            cancelled=1,
            failed=0,
            reason_code="PENDING_STALE_SWEEP",
            errors=[],
        ),
    )
    stale_order = _order(
        "pending_new",
        "o-sweep",
        created_at=now_dt - timedelta(seconds=120),
    )

    caplog.set_level(logging.INFO)
    assert be._handle_pending_orders([stale_order], runtime) is True
    cancel_all_mock.assert_not_called()
    assert any(record.message == "PENDING_STALE_SWEEP_APPLIED" for record in caplog.records)


def test_stale_sweep_defers_verified_sampling_order_to_passive_reprice(monkeypatch):
    now_dt = datetime(2026, 3, 1, 15, 0, 0, tzinfo=UTC)
    stale_order = _order(
        "pending_new",
        "sampling-order",
        created_at=now_dt - timedelta(seconds=120),
    )
    ownership_checks: list[Any] = []
    execution_engine = types.SimpleNamespace(
        _paper_sampling_passive_reprice_manages_order=lambda order: (
            ownership_checks.append(order) or True
        )
    )
    runtime = types.SimpleNamespace(state={}, execution_engine=execution_engine)
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_MAX_CANCELS", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_COOLDOWN_SEC", "0")
    monkeypatch.setattr(
        be,
        "_cancel_open_orders_subset",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("stale sweep must not cancel passive-reprice-owned orders")
        ),
    )

    result = be._maybe_apply_pending_stale_sweep(
        runtime=runtime,
        pending_orders=[stale_order],
        now_dt=now_dt,
        now_ts=now_dt.timestamp(),
    )

    assert result is None
    assert ownership_checks == [stale_order]


def test_full_cleanup_defers_verified_sampling_order_to_passive_reprice(monkeypatch, caplog):
    runtime = types.SimpleNamespace(
        state={},
        execution_engine=types.SimpleNamespace(
            _paper_sampling_passive_reprice_manages_order=lambda _order: True,
            _apply_pending_new_timeout_policy=lambda: False,
        ),
    )
    monkeypatch.setattr(
        be,
        "_cancel_open_orders_subset",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("full cleanup must not cancel passive-reprice-owned orders")
        ),
    )
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=5),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_PENDING_ORDERS_BLOCK_SCOPE", "symbol")
    clock = types.SimpleNamespace(value=100.0)
    monkeypatch.setattr(be.time, "time", lambda: clock.value)
    order = _order("pending_new", "sampling-owned", symbol="AMZN")

    assert be._handle_pending_orders([order], runtime) is True
    clock.value = 106.0
    caplog.set_level(logging.INFO)
    assert be._handle_pending_orders([order], runtime) is True

    deferred = [
        record
        for record in caplog.records
        if record.message == "PENDING_ORDERS_DEFERRED_TO_PASSIVE_REPRICE"
    ]
    assert deferred
    assert deferred[-1].owned_ids == ["sampling-owned"]
    assert deferred[-1].cleanup_count == 0


def test_full_cleanup_only_cancels_orders_not_owned_by_passive_reprice(monkeypatch):
    owned = _order("pending_new", "sampling-owned", symbol="AMZN")
    ordinary = _order("pending_new", "ordinary", symbol="AAPL")
    runtime = types.SimpleNamespace(
        state={},
        execution_engine=types.SimpleNamespace(
            _paper_sampling_passive_reprice_manages_order=lambda order: order is owned,
            _apply_pending_new_timeout_policy=lambda: False,
        ),
    )
    cancelled: list[Any] = []
    monkeypatch.setattr(
        be,
        "_cancel_open_orders_subset",
        lambda _runtime, orders, **_kwargs: cancelled.extend(orders)
        or be.CancelAllResult(
            total_open=len(orders),
            cancelled=len(orders),
            failed=0,
            reason_code="PENDING_ORDERS_CLEANUP",
            errors=[],
        ),
    )
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=5),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_PENDING_ORDERS_BLOCK_SCOPE", "symbol")
    clock = types.SimpleNamespace(value=100.0)
    monkeypatch.setattr(be.time, "time", lambda: clock.value)

    assert be._handle_pending_orders([owned, ordinary], runtime) is True
    clock.value = 106.0
    assert be._handle_pending_orders([owned, ordinary], runtime) is True
    assert cancelled == [ordinary]


def test_handle_pending_orders_reads_mapping_order_status_for_stale_sweep(monkeypatch, caplog):
    runtime = types.SimpleNamespace(state={})
    cancel_all_mock = MagicMock()
    monkeypatch.setattr(be, "cancel_all_open_orders", cancel_all_mock)
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=300),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_MAX_CANCELS", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_COOLDOWN_SEC", "0")

    now_dt = datetime(2026, 3, 1, 15, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(be.time, "time", lambda: now_dt.timestamp())
    subset_calls: list[list[dict[str, Any]]] = []

    def _cancel_subset(_runtime, orders, **_kwargs):
        subset_calls.append(list(orders))
        return be.CancelAllResult(
            total_open=1,
            cancelled=1,
            failed=0,
            reason_code="PENDING_STALE_SWEEP",
            errors=[],
        )

    monkeypatch.setattr(be, "_cancel_open_orders_subset", _cancel_subset)
    stale_order = {
        "status": "pending_new",
        "id": "dict-order",
        "created_at": now_dt - timedelta(seconds=120),
    }

    caplog.set_level(logging.INFO)
    assert be._handle_pending_orders([stale_order], runtime) is True
    cancel_all_mock.assert_not_called()
    assert subset_calls == [[stale_order]]
    assert any(record.message == "PENDING_STALE_SWEEP_APPLIED" for record in caplog.records)


def test_handle_pending_orders_symbol_scope_applies_policy(monkeypatch, caplog):
    runtime = types.SimpleNamespace(
        state={},
        execution_engine=types.SimpleNamespace(),
    )
    policy_calls: list[str] = []
    runtime.execution_engine._apply_pending_new_timeout_policy = (
        lambda: policy_calls.append("called")
    )
    cancel_mock = MagicMock()
    monkeypatch.setattr(be, "cancel_all_open_orders", cancel_mock)
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=30),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_ORDERS_BLOCK_SCOPE", "symbol")

    clock = types.SimpleNamespace(value=100.0)
    monkeypatch.setattr(be.time, "time", lambda: clock.value)
    orders = [_order("pending_new", "o-symbol", symbol="AAPL")]

    caplog.set_level(logging.INFO)
    assert be._handle_pending_orders(orders, runtime) is True
    assert policy_calls == []

    caplog.clear()
    clock.value = 131.0
    assert be._handle_pending_orders(orders, runtime) is True
    assert policy_calls == ["called"]
    cancel_mock.assert_not_called()
    assert getattr(runtime, be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR) == ("AAPL",)
    assert any(record.message == "PENDING_ORDERS_POLICY_APPLIED" for record in caplog.records)


def test_handle_pending_orders_symbol_scope_policy_no_action_falls_back_to_cleanup(monkeypatch):
    runtime = types.SimpleNamespace(
        state={},
        execution_engine=types.SimpleNamespace(_apply_pending_new_timeout_policy=lambda: False),
    )
    cancel_all_mock = MagicMock()
    monkeypatch.setattr(be, "cancel_all_open_orders", cancel_all_mock)
    subset_calls: list[list[Any]] = []
    monkeypatch.setattr(
        be,
        "_cancel_open_orders_subset",
        lambda _runtime, orders, **_kwargs: subset_calls.append(list(orders))
        or be.CancelAllResult(
            total_open=1,
            cancelled=1,
            failed=0,
            reason_code="PENDING_ORDERS_CLEANUP",
            errors=[],
        ),
    )
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=30),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_ORDERS_BLOCK_SCOPE", "symbol")

    clock = types.SimpleNamespace(value=100.0)
    monkeypatch.setattr(be.time, "time", lambda: clock.value)
    orders = [_order("pending_new", "o-symbol-fallback", symbol="AAPL")]

    assert be._handle_pending_orders(orders, runtime) is True
    clock.value = 131.0
    assert be._handle_pending_orders(orders, runtime) is False
    cancel_all_mock.assert_not_called()
    assert subset_calls == [orders]


def test_handle_pending_orders_partial_fill_excluded_from_stuck_pending_slo(monkeypatch):
    runtime = types.SimpleNamespace(
        state={},
        execution_engine=types.SimpleNamespace(_apply_pending_new_timeout_policy=lambda: None),
    )
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=60),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_ORDERS_BLOCK_SCOPE", "symbol")
    monkeypatch.setattr(be.time, "time", lambda: 1000.0)
    monkeypatch.setattr(
        be,
        "_pending_order_broker_age_seconds",
        lambda _order, _now_dt: 600.0,
    )

    metric_calls: list[tuple[int, float]] = []
    monkeypatch.setattr(
        be,
        "_record_pending_order_slo_metrics",
        lambda *, pending_count, oldest_pending_age_s: metric_calls.append(
            (int(pending_count), float(oldest_pending_age_s))
        ),
    )

    orders = [_order("partially_filled", "o-partial", symbol="USB")]
    assert be._handle_pending_orders(orders, runtime) is True

    assert metric_calls
    pending_count, oldest_pending_age_s = metric_calls[-1]
    assert pending_count == 0
    assert oldest_pending_age_s == 0.0


def test_handle_pending_orders_symbol_scope_decay_releases_stale_symbol(monkeypatch, caplog):
    runtime = types.SimpleNamespace(
        state={},
        execution_engine=types.SimpleNamespace(),
    )
    cancel_mock = MagicMock()
    monkeypatch.setattr(be, "cancel_all_open_orders", cancel_mock)
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=600),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_ORDERS_BLOCK_SCOPE", "symbol")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_ACK_MULT", "2")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_MIN_SEC", "5")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_MAX_SEC", "300")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_MIN_CYCLES", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_RELEASE_COOLDOWN_SEC", "60")
    monkeypatch.setenv("ORDER_ACK_TIMEOUT_SECONDS", "8")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_FORCE_CANCEL_SEC", "300")

    now_dt = datetime(2026, 2, 24, 20, 0, 0, tzinfo=UTC)
    now_epoch = now_dt.timestamp()
    monkeypatch.setattr(be.time, "time", lambda: now_epoch)
    stale_order = _order(
        "pending_new",
        "o-decay",
        symbol="AAPL",
        created_at=now_dt - timedelta(seconds=120),
    )

    caplog.set_level(logging.INFO)
    assert be._handle_pending_orders([stale_order], runtime) is True
    assert getattr(runtime, be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR) == ("AAPL",)
    cancel_mock.assert_not_called()
    assert any(record.message == "PENDING_SYMBOL_BLOCK_DECAY_DEFERRED" for record in caplog.records)


def test_handle_pending_orders_symbol_scope_decay_cooldown_prevents_reblock(monkeypatch):
    runtime = types.SimpleNamespace(
        state={},
        execution_engine=types.SimpleNamespace(),
    )
    monkeypatch.setattr(be, "cancel_all_open_orders", lambda *_a, **_k: None)
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=600),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_ORDERS_BLOCK_SCOPE", "symbol")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_ACK_MULT", "2")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_MIN_SEC", "5")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_MAX_SEC", "300")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_MIN_CYCLES", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_RELEASE_COOLDOWN_SEC", "60")
    monkeypatch.setenv("ORDER_ACK_TIMEOUT_SECONDS", "8")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_FORCE_CANCEL_SEC", "300")

    clock = types.SimpleNamespace(value=1000.0)
    monkeypatch.setattr(be.time, "time", lambda: clock.value)

    def _stale_order(order_id: str):
        now_dt = datetime.fromtimestamp(clock.value, tz=UTC)
        return _order(
            "pending_new",
            order_id,
            symbol="AAPL",
            created_at=now_dt - timedelta(seconds=120),
        )

    assert be._handle_pending_orders([_stale_order("o-decay-1")], runtime) is True
    assert getattr(runtime, be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR) == ("AAPL",)

    clock.value += 10.0
    assert be._handle_pending_orders([_stale_order("o-decay-2")], runtime) is True
    assert getattr(runtime, be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR) == ("AAPL",)
    tracker = runtime.state[be._PENDING_SYMBOL_DECAY_TRACKER_KEY]
    assert "AAPL" in tracker
    assert float(tracker["AAPL"].get("released_until_ts", 0.0) or 0.0) == 0.0


def test_handle_pending_orders_symbol_scope_emits_cooldown_telemetry(monkeypatch, caplog):
    runtime = types.SimpleNamespace(
        state={},
        execution_engine=types.SimpleNamespace(),
    )
    monkeypatch.setattr(be, "cancel_all_open_orders", lambda *_a, **_k: None)
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=600),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_ORDERS_BLOCK_SCOPE", "symbol")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_ACK_MULT", "2")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_MIN_SEC", "5")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_MAX_SEC", "300")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_MIN_CYCLES", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_RELEASE_COOLDOWN_SEC", "60")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_COOLDOWN_TELEMETRY_LOG_TTL_SEC", "0")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_COOLDOWN_TELEMETRY_SAMPLE_LIMIT", "3")
    monkeypatch.setenv("ORDER_ACK_TIMEOUT_SECONDS", "8")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_FORCE_CANCEL_SEC", "300")

    now_dt = datetime(2026, 2, 24, 20, 0, 0, tzinfo=UTC)
    now_epoch = now_dt.timestamp()
    monkeypatch.setattr(be.time, "time", lambda: now_epoch)
    stale_order = _order(
        "pending_new",
        "o-decay-metric",
        symbol="AAPL",
        created_at=now_dt - timedelta(seconds=120),
    )

    caplog.set_level(logging.INFO)
    assert be._handle_pending_orders([stale_order], runtime) is True

    telemetry_records = [
        record
        for record in caplog.records
        if record.message == "PENDING_SYMBOL_COOLDOWN_TELEMETRY"
    ]
    assert telemetry_records
    latest = telemetry_records[-1]
    assert int(getattr(latest, "raw_blocked_count", 0)) == 1
    symbol_states = getattr(latest, "symbol_states", None)
    assert isinstance(symbol_states, list)
    assert symbol_states
    first = symbol_states[0]
    assert first.get("symbol") == "AAPL"
    assert first.get("state") == "deferred"
    assert float(first.get("cooldown_remaining_s", 0.0)) >= 0.0


def test_handle_pending_orders_stale_broker_age_triggers_immediate_cleanup(monkeypatch):
    runtime = types.SimpleNamespace(state={})
    cancel_all_mock = MagicMock()
    monkeypatch.setattr(be, "cancel_all_open_orders", cancel_all_mock)
    subset_calls: list[list[Any]] = []
    monkeypatch.setattr(
        be,
        "_cancel_open_orders_subset",
        lambda _runtime, orders, **_kwargs: subset_calls.append(list(orders))
        or be.CancelAllResult(
            total_open=1,
            cancelled=1,
            failed=0,
            reason_code="PENDING_ORDERS_CLEANUP",
            errors=[],
        ),
    )
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=300),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_FORCE_CANCEL_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_ENABLED", "0")

    now_dt = datetime(2026, 2, 9, 18, 0, 0, tzinfo=UTC)
    now_epoch = now_dt.timestamp()
    monkeypatch.setattr(be.time, "time", lambda: now_epoch)
    stale_order = _order(
        "pending_new",
        "o-stale",
        created_at=now_dt - timedelta(seconds=120),
    )

    assert be._handle_pending_orders([stale_order], runtime) is False
    cancel_all_mock.assert_not_called()
    assert subset_calls == [[stale_order]]


def test_handle_pending_orders_cleanup_warmup(monkeypatch):
    runtime = types.SimpleNamespace(state={})
    cancel_all_mock = MagicMock()
    monkeypatch.setattr(be, "cancel_all_open_orders", cancel_all_mock)
    subset_calls: list[list[Any]] = []
    monkeypatch.setattr(
        be,
        "_cancel_open_orders_subset",
        lambda _runtime, orders, **_kwargs: subset_calls.append(list(orders))
        or be.CancelAllResult(
            total_open=1,
            cancelled=1,
            failed=0,
            reason_code="PENDING_ORDERS_CLEANUP",
            errors=[],
        ),
    )
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
    cancel_all_mock.assert_not_called()
    assert subset_calls == [orders]

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


def test_select_startup_stale_orders(monkeypatch):
    now = datetime(2026, 2, 24, 19, 0, tzinfo=UTC)
    monkeypatch.setattr(be, "datetime", types.SimpleNamespace(now=lambda tz=None: now))
    monkeypatch.setenv("AI_TRADING_STARTUP_CANCEL_STALE_SEC", "60")

    stale = _order("pending_new", "stale-1", created_at=now - timedelta(seconds=120))
    fresh = _order("pending_new", "fresh-1", created_at=now - timedelta(seconds=20))
    terminal = _order("filled", "filled-1", created_at=now - timedelta(seconds=180))

    selected = be._select_startup_stale_orders([fresh, terminal, stale])
    selected_ids = [getattr(order, "id", None) for order in selected]
    assert selected_ids == ["stale-1"]


def test_pending_stale_sweep_defaults_to_startup_stale_threshold(monkeypatch):
    monkeypatch.setenv("AI_TRADING_STARTUP_CANCEL_STALE_SEC", "75")
    monkeypatch.delenv("AI_TRADING_PENDING_STALE_SWEEP_SEC", raising=False)

    assert be._pending_stale_sweep_age_seconds() == 75.0


def test_cancel_open_orders_subset_cancels_target_orders():
    cancelled_ids: list[str] = []

    class _API:
        def cancel_order(self, order_id):
            cancelled_ids.append(str(order_id))

    runtime = types.SimpleNamespace(api=_API())
    stale_a = _order("pending_new", "stale-a", client_order_id="client-a")
    stale_b = _order("pending_new", "stale-b", client_order_id="client-b")

    result = be._cancel_open_orders_subset(
        runtime,
        orders=[stale_a, stale_b],
        reason_code="STARTUP_STALE_PENDING",
    )

    assert result.reason_code == "STARTUP_STALE_PENDING"
    assert result.total_open == 2
    assert result.cancelled == 2
    assert result.failed == 0
    assert cancelled_ids == ["stale-a", "stale-b"]


def test_cancel_open_orders_subset_never_falls_back_to_cancel_all(monkeypatch):
    class _API:
        pass

    runtime = types.SimpleNamespace(api=_API())
    stale_order = _order("pending_new", "stale-a")
    cancel_all_called = {"value": False}

    def _unexpected_cancel_all(_runtime):
        cancel_all_called["value"] = True
        return be.CancelAllResult(
            total_open=1,
            cancelled=1,
            failed=0,
            reason_code="CANCEL_ALL_TRIGGERED",
            errors=[],
        )

    monkeypatch.setattr(be, "cancel_all_open_orders_oms", _unexpected_cancel_all)
    result = be._cancel_open_orders_subset(
        runtime,
        orders=[stale_order],
        reason_code="STARTUP_STALE_PENDING",
    )

    assert cancel_all_called["value"] is False
    assert result.reason_code == "STARTUP_STALE_PENDING"
    assert result.total_open == 1
    assert result.cancelled == 0
    assert result.failed == 1


def _symbol_scope_runtime_without_decay(monkeypatch, *, cleanup_interval: int) -> Any:
    monkeypatch.setattr(
        be,
        "get_trading_config",
        lambda: types.SimpleNamespace(order_stale_cleanup_interval=cleanup_interval),
    )
    monkeypatch.setenv("AI_TRADING_PENDING_ORDERS_BLOCK_SCOPE", "symbol")
    monkeypatch.setenv("AI_TRADING_PENDING_SYMBOL_DECAY_ENABLED", "0")
    # An api without get_order: pending confirmation keeps each order's listed status.
    return types.SimpleNamespace(
        state={},
        api=types.SimpleNamespace(),
        execution_engine=types.SimpleNamespace(_apply_pending_new_timeout_policy=lambda: False),
    )


def test_stale_sweep_holds_symbol_until_broker_stops_listing_cancelled_order(monkeypatch, caplog):
    runtime = _symbol_scope_runtime_without_decay(monkeypatch, cleanup_interval=3600)
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_FORCE_CANCEL_SEC", "3600")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_SEC", "60")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_MAX_CANCELS", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_COOLDOWN_SEC", "0")
    swept: list[Any] = []
    monkeypatch.setattr(
        be,
        "_cancel_open_orders_subset",
        lambda _runtime, orders, **_kwargs: swept.extend(orders)
        or be.CancelAllResult(
            total_open=len(orders),
            cancelled=len(orders),
            failed=0,
            reason_code="PENDING_STALE_SWEEP",
            errors=[],
        ),
    )
    now_dt = datetime(2026, 3, 2, 15, 0, 0, tzinfo=UTC)
    clock = types.SimpleNamespace(value=now_dt.timestamp())
    monkeypatch.setattr(be.time, "time", lambda: clock.value)
    exit_order = _order(
        "pending_new", "exit-aapl", symbol="AAPL", side="sell",
        created_at=now_dt - timedelta(seconds=120),
    )
    entry_order = _order(
        "pending_new", "entry-msft", symbol="MSFT", side="buy",
        created_at=now_dt - timedelta(seconds=5),
    )

    # AAPL is inside a block-decay release cooldown, so decay leaves it out
    # of the blocked set even though it has an open order. Without the hold
    # the netting cycle straight after the sweep could resubmit AAPL.
    runtime.state[be._PENDING_SYMBOL_DECAY_TRACKER_KEY] = {
        "AAPL": {"released_until_ts": clock.value + 600.0}
    }

    assert be._handle_pending_orders([exit_order, entry_order], runtime) is True
    assert [order.id for order in swept] == ["exit-aapl"]
    assert "AAPL" in getattr(runtime, be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR)

    # Next cycle: the broker still lists the exit (a fill raced the cancel, so
    # it shows partially_filled) beside a fresh entry. Pending confirmation
    # drops partially_filled orders, and the cancel is not confirmed, so AAPL
    # must stay blocked.
    clock.value += 30.0
    exit_cancelling = _order(
        "partially_filled", "exit-aapl", symbol="AAPL", side="sell",
        created_at=now_dt - timedelta(seconds=120),
    )
    caplog.set_level(logging.INFO)
    assert be._handle_pending_orders([exit_cancelling, entry_order], runtime) is True
    assert "AAPL" in getattr(runtime, be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR)
    assert any(
        record.message == "PENDING_CANCEL_UNCONFIRMED_SYMBOLS_HELD" for record in caplog.records
    )

    # Once the broker no longer lists the exit, the cancel is confirmed and
    # AAPL is released.
    clock.value += 30.0
    assert be._handle_pending_orders([entry_order], runtime) is True
    assert getattr(runtime, be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR) == ("MSFT",)


def test_stale_sweep_does_not_hold_symbol_when_cancel_request_failed(monkeypatch):
    runtime = _symbol_scope_runtime_without_decay(monkeypatch, cleanup_interval=3600)
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_SEC", "60")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_COOLDOWN_SEC", "0")
    monkeypatch.setattr(
        be,
        "_cancel_open_orders_subset",
        lambda _runtime, orders, **_kwargs: be.CancelAllResult(
            total_open=len(orders),
            cancelled=0,
            failed=len(orders),
            reason_code="PENDING_STALE_SWEEP",
            errors=[{"order_id": "exit-aapl", "error": "rejected"}],
        ),
    )
    now_dt = datetime(2026, 3, 2, 15, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(be.time, "time", lambda: now_dt.timestamp())
    exit_order = _order(
        "pending_new", "exit-aapl", symbol="AAPL", side="sell",
        created_at=now_dt - timedelta(seconds=120),
    )

    be._handle_pending_orders([exit_order], runtime)

    assert not runtime.state.get("_pending_cancel_unconfirmed")


def test_pending_cleanup_holds_symbol_until_broker_stops_listing_cancelled_order(monkeypatch):
    runtime = _symbol_scope_runtime_without_decay(monkeypatch, cleanup_interval=30)
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_ENABLED", "0")
    cleaned: list[Any] = []
    monkeypatch.setattr(
        be,
        "_cancel_open_orders_subset",
        lambda _runtime, orders, **_kwargs: cleaned.extend(orders)
        or be.CancelAllResult(
            total_open=len(orders),
            cancelled=len(orders),
            failed=0,
            reason_code="PENDING_ORDERS_CLEANUP",
            errors=[],
        ),
    )
    clock = types.SimpleNamespace(value=100.0)
    monkeypatch.setattr(be.time, "time", lambda: clock.value)
    exit_order = _order("pending_new", "exit-aapl", symbol="AAPL", side="sell")

    assert be._handle_pending_orders([exit_order], runtime) is True
    clock.value = 131.0
    assert be._handle_pending_orders([exit_order], runtime) is False
    assert [order.id for order in cleaned] == ["exit-aapl"]
    # The netting cycle runs straight after cleanup; AAPL must not be in it.
    assert getattr(runtime, be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR) == ("AAPL",)

    clock.value = 161.0
    exit_cancelling = _order("pending_cancel", "exit-aapl", symbol="AAPL", side="sell")
    entry_order = _order("pending_new", "entry-msft", symbol="MSFT", side="buy")
    be._handle_pending_orders([exit_cancelling, entry_order], runtime)
    assert "AAPL" in getattr(runtime, be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR)


def _partial_exit_sweep_setup(monkeypatch) -> tuple[Any, list[Any], datetime]:
    runtime = _symbol_scope_runtime_without_decay(monkeypatch, cleanup_interval=3600)
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_FORCE_CANCEL_SEC", "3600")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_SEC", "60")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_PARTIALLY_FILLED_SEC", "900")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_MAX_CANCELS", "3")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_COOLDOWN_SEC", "0")
    swept: list[Any] = []
    monkeypatch.setattr(
        be,
        "_cancel_open_orders_subset",
        lambda _runtime, orders, **_kwargs: swept.extend(orders)
        or be.CancelAllResult(
            total_open=len(orders),
            cancelled=len(orders),
            failed=0,
            reason_code="PENDING_STALE_SWEEP",
            errors=[],
        ),
    )
    now_dt = datetime(2026, 3, 2, 15, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(be.time, "time", lambda: now_dt.timestamp())
    return runtime, swept, now_dt


def test_stale_sweep_reaches_partly_filled_exit_while_another_order_is_pending(monkeypatch):
    runtime, swept, now_dt = _partial_exit_sweep_setup(monkeypatch)
    partial_exit = _order(
        "partially_filled", "exit-aapl", symbol="AAPL", side="sell",
        position_intent="sell_to_close",
        created_at=now_dt - timedelta(seconds=1000),
    )
    young_entry = _order(
        "pending_new", "entry-msft", symbol="MSFT", side="buy",
        created_at=now_dt - timedelta(seconds=5),
    )

    be._handle_pending_orders([partial_exit, young_entry], runtime)

    assert [order.id for order in swept] == ["exit-aapl"]
    # The cancel is a request: AAPL stays out of the cycle until it is confirmed.
    assert "AAPL" in getattr(runtime, be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR)


def test_stale_sweep_leaves_young_or_opening_partial_fills_alone(monkeypatch):
    runtime, swept, now_dt = _partial_exit_sweep_setup(monkeypatch)
    young_partial_exit = _order(
        "partially_filled", "exit-aapl", symbol="AAPL", side="sell",
        position_intent="sell_to_close",
        created_at=now_dt - timedelta(seconds=300),
    )
    old_partial_entry = _order(
        "partially_filled", "entry-nvda", symbol="NVDA", side="buy",
        position_intent="buy_to_open",
        created_at=now_dt - timedelta(seconds=1000),
    )
    young_entry = _order(
        "pending_new", "entry-msft", symbol="MSFT", side="buy",
        created_at=now_dt - timedelta(seconds=5),
    )

    be._handle_pending_orders([young_partial_exit, old_partial_entry, young_entry], runtime)

    assert swept == []


class _StrEnum(str):
    """Stand-in for alpaca-py str enums: ``.value`` holds the wire token."""

    def __new__(cls, token: str, name: str) -> "_StrEnum":
        obj = super().__new__(cls, f"Enum.{name}")
        obj.value = token  # type: ignore[attr-defined]
        return obj


def _hold_runtime(monkeypatch, *, scope: str = "symbol") -> Any:
    runtime = _symbol_scope_runtime_without_decay(monkeypatch, cleanup_interval=3600)
    monkeypatch.setenv("AI_TRADING_PENDING_ORDERS_BLOCK_SCOPE", scope)
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_FORCE_CANCEL_SEC", "3600")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_SEC", "60")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_MAX_CANCELS", "5")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_COOLDOWN_SEC", "0")
    return runtime


def _accept_all_cancels(monkeypatch) -> list[Any]:
    swept: list[Any] = []
    monkeypatch.setattr(
        be,
        "_cancel_open_orders_subset",
        lambda _runtime, orders, **_kwargs: swept.extend(orders)
        or be.CancelAllResult(
            total_open=len(orders),
            cancelled=len(orders),
            failed=0,
            reason_code="PENDING_STALE_SWEEP",
            errors=[],
        ),
    )
    return swept


def _clock(monkeypatch) -> Any:
    clock = types.SimpleNamespace(value=datetime(2026, 3, 2, 15, 0, 0, tzinfo=UTC).timestamp())
    monkeypatch.setattr(be.time, "time", lambda: clock.value)
    return clock


def _aged(status: Any, oid: str, symbol: str, age_s: float, clock: Any, **extra: Any) -> Any:
    created = datetime.fromtimestamp(clock.value, tz=UTC) - timedelta(seconds=age_s)
    return _order(status, oid, symbol=symbol, created_at=created, **extra)


def _blocked(runtime: Any) -> tuple[str, ...]:
    return tuple(getattr(runtime, be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR))


def test_hold_released_on_empty_open_list(monkeypatch):
    runtime = _hold_runtime(monkeypatch)
    _accept_all_cancels(monkeypatch)
    clock = _clock(monkeypatch)
    be._handle_pending_orders([_aged("pending_new", "x", "AAPL", 120, clock)], runtime)
    assert runtime.state["_pending_cancel_unconfirmed"]

    clock.value += 30.0
    assert be._handle_pending_orders([], runtime) is False
    assert runtime.state["_pending_cancel_unconfirmed"] == {}
    assert _blocked(runtime) == ()


def test_hold_released_when_order_listed_as_filled(monkeypatch):
    runtime = _hold_runtime(monkeypatch)
    _accept_all_cancels(monkeypatch)
    clock = _clock(monkeypatch)
    be._handle_pending_orders([_aged("pending_new", "x", "AAPL", 120, clock)], runtime)

    # The order filled before the cancel landed; a terminal status confirms it.
    clock.value += 30.0
    filled = _aged("filled", "x", "AAPL", 150, clock)
    entry = _aged("pending_new", "e", "MSFT", 5, clock)
    be._handle_pending_orders([filled, entry], runtime)
    assert "AAPL" not in _blocked(runtime)
    assert "x" not in runtime.state["_pending_cancel_unconfirmed"]


def test_partial_cancel_batch_holds_only_accepted_ids(monkeypatch):
    runtime = _hold_runtime(monkeypatch)
    monkeypatch.setattr(
        be,
        "_cancel_open_orders_subset",
        lambda _runtime, orders, **_kwargs: be.CancelAllResult(
            total_open=len(orders),
            cancelled=1,
            failed=1,
            reason_code="PENDING_STALE_SWEEP",
            errors=[{"order_id": "b", "error": "rejected"}],
        ),
    )
    clock = _clock(monkeypatch)
    orders = [
        _aged("pending_new", "a", "AAPL", 200, clock),
        _aged("pending_new", "b", "NVDA", 150, clock),
    ]
    be._handle_pending_orders(orders, runtime)
    assert set(runtime.state["_pending_cancel_unconfirmed"]) == {"a"}


def test_hold_handles_alpaca_enum_status_intent_and_side(monkeypatch):
    runtime = _hold_runtime(monkeypatch)
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_PARTIALLY_FILLED_SEC", "900")
    swept = _accept_all_cancels(monkeypatch)
    clock = _clock(monkeypatch)
    partial_exit = _aged(
        _StrEnum("partially_filled", "PARTIALLY_FILLED"), "x", "AAPL", 1000, clock,
        side=_StrEnum("sell", "SELL"),
        position_intent=_StrEnum("sell_to_close", "SELL_TO_CLOSE"),
    )
    entry = _aged(_StrEnum("pending_new", "PENDING_NEW"), "e", "MSFT", 5, clock)
    be._handle_pending_orders([partial_exit, entry], runtime)
    assert [order.id for order in swept] == ["x"]

    clock.value += 30.0
    cancelling = _aged(_StrEnum("pending_cancel", "PENDING_CANCEL"), "x", "AAPL", 1030, clock)
    be._handle_pending_orders([cancelling, entry], runtime)
    assert runtime.state["_pending_cancel_unconfirmed"]["x"]["last_status"] == "pending_cancel"
    assert "AAPL" in _blocked(runtime)


def test_position_intent_token_matches_name_fallback():
    from ai_trading.core import pending_order_runtime as por
    from ai_trading.execution.live_trading import _position_intent_token

    class _NameOnly:
        name = "SELL_TO_CLOSE"

    for value in (_NameOnly(), "SELL_TO_CLOSE", _StrEnum("sell_to_close", "X"), None):
        assert por._token(value) == _position_intent_token(value)


def test_hold_with_several_orders_on_one_symbol(monkeypatch):
    runtime = _hold_runtime(monkeypatch)
    _accept_all_cancels(monkeypatch)
    clock = _clock(monkeypatch)
    be._handle_pending_orders(
        [_aged("pending_new", "a1", "AAPL", 200, clock), _aged("pending_new", "a2", "AAPL", 150, clock)],
        runtime,
    )
    entry = _aged("pending_new", "e", "MSFT", 5, clock)

    clock.value += 30.0
    be._handle_pending_orders([_aged("partially_filled", "a2", "AAPL", 180, clock), entry], runtime)
    assert "AAPL" in _blocked(runtime)
    assert set(runtime.state["_pending_cancel_unconfirmed"]) == {"a2"}

    clock.value += 30.0
    be._handle_pending_orders([entry], runtime)
    assert "AAPL" not in _blocked(runtime)


def test_listed_pending_cancel_or_replace_blocks_symbol_after_restart(monkeypatch):
    # Fresh runtime state, as after a restart: nothing was remembered, but the
    # broker lists orders whose cancel or replace someone else requested.
    runtime = _hold_runtime(monkeypatch)
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_ENABLED", "0")
    clock = _clock(monkeypatch)
    orders = [
        _aged("pending_cancel", "c", "AAPL", 10, clock),
        _aged(_StrEnum("pending_replace", "PENDING_REPLACE"), "r", "NVDA", 10, clock),
        _aged("pending_new", "e", "MSFT", 5, clock),
    ]
    be._handle_pending_orders(orders, runtime)
    assert set(_blocked(runtime)) == {"AAPL", "NVDA", "MSFT"}

    # The replace was rejected and the order is back to accepted: a status
    # hold ends as soon as the order leaves pending_cancel/pending_replace.
    clock.value += 30.0
    orders[1] = _aged("accepted", "r", "NVDA", 40, clock)
    be._handle_pending_orders(orders, runtime)
    assert "NVDA" not in _blocked(runtime)
    assert "AAPL" in _blocked(runtime)


def test_long_hold_warns_after_300s_then_throttles(monkeypatch, caplog):
    runtime = _hold_runtime(monkeypatch)
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_ENABLED", "0")
    clock = _clock(monkeypatch)
    entry = _aged("pending_new", "e", "MSFT", 5, clock)
    stuck = _aged("pending_cancel", "c", "AAPL", 10, clock)
    caplog.set_level(logging.INFO)

    def _warnings() -> list[Any]:
        return [r for r in caplog.records if r.message == "PENDING_CANCEL_HOLD_LONG"]

    for step, expected in ((0, 0), (299, 0), (1, 1), (100, 1), (200, 2), (60, 2)):
        clock.value += step
        be._handle_pending_orders([stuck, entry], runtime)
        assert len(_warnings()) == expected, step
    last = _warnings()[-1]
    assert (last.symbol, last.order_id, last.last_status) == ("AAPL", "c", "pending_cancel")
    assert last.held_s == 600
    assert any(r.message == "PENDING_CANCEL_UNCONFIRMED_SYMBOLS_HELD" for r in caplog.records)


def test_cycle_scope_is_unchanged_by_holds(monkeypatch, caplog):
    runtime = _hold_runtime(monkeypatch, scope="cycle")
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_PARTIALLY_FILLED_SEC", "900")
    swept = _accept_all_cancels(monkeypatch)
    clock = _clock(monkeypatch)
    partial_exit = _aged(
        "partially_filled", "p", "AAPL", 1000, clock, side="sell", position_intent="sell_to_close"
    )
    cancelling = _aged("pending_cancel", "c", "NVDA", 10, clock)
    entry = _aged("pending_new", "e", "MSFT", 5, clock)
    caplog.set_level(logging.INFO)

    be._handle_pending_orders([partial_exit, cancelling, entry], runtime)

    # As on main: the sweep sees only confirmed pending orders and the blocked
    # set is just their symbols; no hold is applied or logged.
    assert [order.id for order in swept] == []
    assert _blocked(runtime) == ("MSFT",)
    assert not any(
        r.message in {"PENDING_CANCEL_UNCONFIRMED_SYMBOLS_HELD", "PENDING_CANCEL_HOLD_LONG"}
        for r in caplog.records
    )


def test_engine_timeout_cancel_holds_symbol_for_this_cycle_and_until_confirmed(monkeypatch):
    runtime = _hold_runtime(monkeypatch)
    monkeypatch.setenv("AI_TRADING_PENDING_STALE_SWEEP_ENABLED", "0")
    clock = _clock(monkeypatch)
    # An accepted exit beside a pending entry: pending confirmation drops
    # "accepted", so AAPL is not in the blocked set the handler computed.
    exit_order = _aged("accepted", "x", "AAPL", 40, clock, side="sell")
    entry = _aged("pending_new", "e", "MSFT", 5, clock)
    be._handle_pending_orders([exit_order, entry], runtime)
    assert "AAPL" not in _blocked(runtime)

    # start_cycle then runs the engine's timeout policy, which cancels it.
    runtime.execution_engine._pending_new_cancelled_orders = [exit_order]
    from ai_trading.core.pending_order_runtime import hold_engine_timeout_cancels

    hold_engine_timeout_cancels(runtime)
    assert "AAPL" in _blocked(runtime)
    assert runtime.execution_engine._pending_new_cancelled_orders == []

    clock.value += 30.0
    be._handle_pending_orders([_aged("accepted", "x", "AAPL", 70, clock, side="sell"), entry], runtime)
    assert "AAPL" in _blocked(runtime)
    clock.value += 30.0
    be._handle_pending_orders([entry], runtime)
    assert "AAPL" not in _blocked(runtime)


def test_engine_timeout_cancel_does_not_block_in_cycle_scope(monkeypatch):
    runtime = _hold_runtime(monkeypatch, scope="cycle")
    clock = _clock(monkeypatch)
    entry = _aged("pending_new", "e", "MSFT", 5, clock)
    be._handle_pending_orders([entry], runtime)
    runtime.execution_engine._pending_new_cancelled_orders = [
        _aged("accepted", "x", "AAPL", 40, clock, side="sell")
    ]
    from ai_trading.core.pending_order_runtime import hold_engine_timeout_cancels

    hold_engine_timeout_cancels(runtime)
    assert _blocked(runtime) == ("MSFT",)
