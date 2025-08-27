from __future__ import annotations

import types
import sys
from unittest.mock import Mock

import ai_trading.core.bot_engine as eng


def test_run_all_trades_handles_empty_symbols(monkeypatch):
    """run_all_trades_worker should warn and return when no symbols are provided."""

    # Stub Alpaca modules so _validate_trading_api can shim list_orders
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
    runtime = types.SimpleNamespace(
        api=DummyAPI(), risk_engine=DummyRiskEngine(), model=object()
    )

    # Minimal patches to isolate logic
    monkeypatch.setattr(eng, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(eng, "_init_metrics", lambda: None)
    monkeypatch.setattr(eng, "is_market_open", lambda: True)
    monkeypatch.setattr(eng, "ensure_alpaca_attached", lambda _rt: None)
    monkeypatch.setattr(eng, "check_pdt_rule", lambda _rt: False)
    monkeypatch.setattr(eng, "get_strategies", lambda: [])
    monkeypatch.setattr(eng, "get_verbose_logging", lambda: False)
    monkeypatch.setattr(eng.CFG, "log_market_fetch", False, raising=False)
    monkeypatch.setattr(eng, "MEMORY_OPTIMIZATION_AVAILABLE", False, raising=False)

    class DummyLock:
        def acquire(self, blocking: bool = False) -> bool:
            return True

        def release(self) -> None:
            pass

    monkeypatch.setattr(eng, "run_lock", DummyLock())

    sleep_mock = Mock()
    monkeypatch.setattr(eng.time, "sleep", sleep_mock)
    warn_mock = Mock()
    monkeypatch.setattr(eng.logger_once, "warning", warn_mock)
    process_mock = Mock()
    monkeypatch.setattr(eng, "_process_symbols", process_mock)

    monkeypatch.setattr(eng, "_prepare_run", lambda *_a, **_k: (0.0, True, []))

    eng.run_all_trades_worker(state, runtime)

    warn_mock.assert_called_once()
    process_mock.assert_not_called()
    sleep_mock.assert_called_once()
