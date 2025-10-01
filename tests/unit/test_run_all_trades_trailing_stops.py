import logging
import types
import sys


numpy_stub = types.ModuleType("numpy")
numpy_stub.nan = float("nan")
numpy_stub.NaN = numpy_stub.nan
numpy_stub.random = types.SimpleNamespace(seed=lambda *_a, **_k: None)
sys.modules.setdefault("numpy", numpy_stub)

portalocker_stub = types.ModuleType("portalocker")
portalocker_stub.LOCK_EX = 1
portalocker_stub.lock = lambda *_a, **_k: None
portalocker_stub.unlock = lambda *_a, **_k: None
sys.modules.setdefault("portalocker", portalocker_stub)

bs4_stub = types.ModuleType("bs4")
bs4_stub.BeautifulSoup = object
sys.modules.setdefault("bs4", bs4_stub)

flask_stub = types.ModuleType("flask")
class _Flask:
    def __init__(self, *args, **kwargs):
        pass
    def route(self, *args, **kwargs):  # pragma: no cover - stub
        def _decorator(func):
            return func
        return _decorator

flask_stub.Flask = _Flask
sys.modules.setdefault("flask", flask_stub)

import ai_trading.core.bot_engine as eng
from ai_trading.execution.engine import ExecutionEngine


def test_run_all_trades_calls_trailing_stops(monkeypatch, caplog):
    """run_all_trades_worker should invoke check_trailing_stops and suppress errors."""

    # Stub Alpaca modules so _validate_trading_api works
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

        def cancel_order(self, *args, **kwargs):  # noqa: D401 - stub
            """Provide minimal cancel capability for validation."""
            return None

    class DummyRiskEngine:
        def wait_for_exposure_update(self, timeout: float) -> None:
            pass

    state = eng.BotState()
    runtime = types.SimpleNamespace(
        api=DummyAPI(), risk_engine=DummyRiskEngine(), model=object(), drawdown_circuit_breaker=None
    )

    # Minimal patches to isolate logic
    monkeypatch.setattr(eng, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(eng, "_init_metrics", lambda: None)
    monkeypatch.setattr(eng, "is_market_open", lambda: True)
    monkeypatch.setattr(eng, "_is_market_open_base", lambda: True)
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

    monkeypatch.setattr(eng, "_process_symbols", lambda ctx, symbols, cash: None)
    monkeypatch.setattr(eng, "_prepare_run", lambda *_a, **_k: (0.0, True, []))

    called = {"flag": False}

    def mock_check_trailing_stops(self):
        called["flag"] = True
        raise ValueError("boom")

    monkeypatch.setattr(ExecutionEngine, "check_trailing_stops", mock_check_trailing_stops)
    monkeypatch.setattr(ExecutionEngine, "check_stops", lambda self: None)

    with caplog.at_level(logging.INFO, logger="ai_trading.core.bot_engine"):
        eng.run_all_trades_worker(state, runtime)

    assert called["flag"]
    trailing_logs = [
        record
        for record in caplog.records
        if record.getMessage() == "TRAILING_STOP_CHECK_SUPPRESSED"
    ]
    assert trailing_logs, "Expected trailing stop suppression to be logged"
    assert trailing_logs[0].cause == "ValueError"
    assert "boom" in trailing_logs[0].detail
