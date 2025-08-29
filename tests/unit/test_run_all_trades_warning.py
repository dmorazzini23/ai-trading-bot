"""Ensure run_all_trades_worker does not warn for a valid API client."""

from __future__ import annotations

import types
import sys
from unittest.mock import Mock, call

sklearn_stub = types.ModuleType("sklearn")
ensemble_stub = types.ModuleType("sklearn.ensemble")

class _GB:  # noqa: D401 - minimal placeholder
    pass


class _RF:  # noqa: D401 - minimal placeholder
    pass


ensemble_stub.GradientBoostingClassifier = _GB
ensemble_stub.RandomForestClassifier = _RF
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.ensemble", ensemble_stub)
metrics_stub = types.ModuleType("sklearn.metrics")
metrics_stub.accuracy_score = lambda *a, **k: 0.0
sys.modules.setdefault("sklearn.metrics", metrics_stub)
model_selection_stub = types.ModuleType("sklearn.model_selection")
model_selection_stub.train_test_split = lambda *a, **k: ([], [])
sys.modules.setdefault("sklearn.model_selection", model_selection_stub)
preproc_stub = types.ModuleType("sklearn.preprocessing")
preproc_stub.StandardScaler = type("StandardScaler", (), {})
sys.modules.setdefault("sklearn.preprocessing", preproc_stub)

import ai_trading.core.bot_engine as eng


def test_run_all_trades_no_warning_with_valid_api(monkeypatch):
    """A valid client with get_orders should not trigger a warning."""

    # Stub Alpaca modules so the shim translates status -> GetOrdersRequest
    enums_mod = types.ModuleType("alpaca.trading.enums")
    requests_mod = types.ModuleType("alpaca.trading.requests")

    class OrderStatus:
        OPEN = "open"

    class GetOrdersRequest:
        def __init__(self, *, statuses=None):
            self.statuses = statuses

    enums_mod.OrderStatus = OrderStatus
    requests_mod.GetOrdersRequest = GetOrdersRequest
    monkeypatch.setitem(sys.modules, "alpaca", types.ModuleType("alpaca"))
    monkeypatch.setitem(sys.modules, "alpaca.trading", types.ModuleType("alpaca.trading"))
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", enums_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", requests_mod)

    class DummyAPI:
        def __init__(self):
            self.called_with: dict | None = None

        def get_orders(self, *args, **kwargs):  # noqa: D401
            """Capture forwarded kwargs."""
            self.called_with = kwargs
            return []

    class DummyRiskEngine:
        def wait_for_exposure_update(self, timeout: float) -> None:  # noqa: D401
            """No-op risk update."""

    state = eng.BotState()
    api = DummyAPI()
    runtime = types.SimpleNamespace(api=api, risk_engine=DummyRiskEngine())

    # Minimal patches to isolate the order-check logic
    monkeypatch.setattr(eng, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(eng, "_init_metrics", lambda: None)
    monkeypatch.setattr(eng, "is_market_open", lambda: True)
    monkeypatch.setattr(eng, "ensure_alpaca_attached", lambda runtime: None)
    monkeypatch.setattr(eng, "check_pdt_rule", lambda runtime: False)
    monkeypatch.setattr(eng, "get_strategies", lambda: [])
    monkeypatch.setattr(eng, "get_verbose_logging", lambda: False)

    def _raise_df(*_a, **_k):
        raise eng.DataFetchError("boom")

    monkeypatch.setattr(eng, "_prepare_run", _raise_df)
    monkeypatch.setattr(eng.CFG, "log_market_fetch", False, raising=False)

    class DummyLock:
        def acquire(self, blocking: bool = False) -> bool:  # noqa: D401
            """Always acquire."""

            return True

        def release(self) -> None:
            """No-op release."""

    monkeypatch.setattr(eng, "run_lock", DummyLock())

    warn_mock = Mock()
    monkeypatch.setattr(eng.logger_once, "warning", warn_mock)

    eng.run_all_trades_worker(state, runtime)

    warn_mock.assert_has_calls(
        [
            call("API_GET_ORDERS_MAPPED", key="alpaca_get_orders_mapped"),
            call("ALPACA_API_ADAPTER", key="alpaca_api_adapter"),
        ]
    )
    assert api.called_with is not None
    assert "filter" in api.called_with
    assert api.called_with["filter"].statuses == [OrderStatus.OPEN]


def test_run_all_trades_creates_trade_log(tmp_path, monkeypatch):
    """Launching the bot should create the trade log with CSV headers."""

    enums_mod = types.ModuleType("alpaca.trading.enums")
    requests_mod = types.ModuleType("alpaca.trading.requests")

    class OrderStatus:
        OPEN = "open"

    class GetOrdersRequest:
        def __init__(self, *, statuses=None):
            self.statuses = statuses

    enums_mod.OrderStatus = OrderStatus
    requests_mod.GetOrdersRequest = GetOrdersRequest
    monkeypatch.setitem(sys.modules, "alpaca", types.ModuleType("alpaca"))
    monkeypatch.setitem(sys.modules, "alpaca.trading", types.ModuleType("alpaca.trading"))
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", enums_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", requests_mod)

    class DummyAPI:
        def get_orders(self, *args, **kwargs):
            return []

    class DummyRiskEngine:
        def wait_for_exposure_update(self, timeout: float) -> None:
            pass

    state = eng.BotState()
    runtime = types.SimpleNamespace(api=DummyAPI(), risk_engine=DummyRiskEngine())

    monkeypatch.setattr(eng, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(eng, "_init_metrics", lambda: None)
    monkeypatch.setattr(eng, "is_market_open", lambda: True)
    monkeypatch.setattr(eng, "ensure_alpaca_attached", lambda _rt: None)
    monkeypatch.setattr(eng, "check_pdt_rule", lambda _rt: False)
    monkeypatch.setattr(eng, "get_strategies", lambda: [])
    monkeypatch.setattr(eng, "get_verbose_logging", lambda: False)
    monkeypatch.setattr(eng.CFG, "log_market_fetch", False, raising=False)

    class DummyLock:
        def acquire(self, blocking: bool = False) -> bool:
            return True

        def release(self) -> None:
            pass

    monkeypatch.setattr(eng, "run_lock", DummyLock())

    def _raise_df(*_a, **_k):
        raise eng.DataFetchError("boom")

    monkeypatch.setattr(eng, "_prepare_run", _raise_df)

    trade_log = tmp_path / "trades.csv"
    reward_log = tmp_path / "reward.csv"
    monkeypatch.setattr(eng, "TRADE_LOG_FILE", str(trade_log))
    monkeypatch.setattr(eng, "REWARD_LOG_FILE", str(reward_log))
    monkeypatch.setattr(eng, "_TRADE_LOGGER_SINGLETON", None)
    monkeypatch.setattr(eng, "_global_ctx", None)

    eng.run_all_trades_worker(state, runtime)

    assert trade_log.exists()
    assert (
        trade_log.read_text().splitlines()[0]
        == "symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward"
    )
