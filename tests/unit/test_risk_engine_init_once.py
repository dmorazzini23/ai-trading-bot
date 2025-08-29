from __future__ import annotations

import sys
import types

def test_risk_engine_init_logged_once(monkeypatch, caplog):
    """Risk engine initialization should log only once per run."""

    # Stub Alpaca modules required by risk engine and API validation
    enums_mod = types.ModuleType("alpaca.trading.enums")
    requests_mod = types.ModuleType("alpaca.trading.requests")
    client_mod = types.ModuleType("alpaca.trading.client")
    common_mod = types.ModuleType("alpaca.common.exceptions")

    class OrderStatus:
        OPEN = "open"

    class GetOrdersRequest:
        def __init__(self, *, statuses=None):
            self.statuses = statuses

    class TradingClient:  # noqa: D401 - minimal stub
        def __init__(self, *a, **k):
            pass

    class APIError(Exception):
        pass

    enums_mod.OrderStatus = OrderStatus
    requests_mod.GetOrdersRequest = GetOrdersRequest
    client_mod.TradingClient = TradingClient
    common_mod.APIError = APIError

    monkeypatch.setitem(sys.modules, "alpaca", types.ModuleType("alpaca"))
    monkeypatch.setitem(sys.modules, "alpaca.trading", types.ModuleType("alpaca.trading"))
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", enums_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", requests_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", client_mod)
    monkeypatch.setitem(sys.modules, "alpaca.common", types.ModuleType("alpaca.common"))
    monkeypatch.setitem(sys.modules, "alpaca.common.exceptions", common_mod)

    # Import modules after stubbing dependencies
    import ai_trading.risk.engine as risk_engine_mod
    import ai_trading.core.bot_engine as eng

    # Create startup components
    caplog.set_level("INFO")
    risk_engine = risk_engine_mod.RiskEngine()

    class DummyAPI:
        def list_orders(self, *a, **k):
            return []

        def get_account(self):
            return types.SimpleNamespace(cash=0, equity=0)

    runtime = types.SimpleNamespace(
        api=DummyAPI(),
        risk_engine=risk_engine,
        model=object(),
    )

    # Minimal patches to isolate loop
    monkeypatch.setattr(eng, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(eng, "_init_metrics", lambda: None)
    monkeypatch.setattr(eng, "is_market_open", lambda: True)
    monkeypatch.setattr(eng, "ensure_alpaca_attached", lambda _rt: None)
    monkeypatch.setattr(eng, "check_pdt_rule", lambda _rt: False)
    monkeypatch.setattr(eng, "get_strategies", lambda: [])
    monkeypatch.setattr(eng, "get_verbose_logging", lambda: False)
    monkeypatch.setattr(eng.CFG, "log_market_fetch", False, raising=False)
    monkeypatch.setattr(eng, "MEMORY_OPTIMIZATION_AVAILABLE", False, raising=False)
    monkeypatch.setattr(risk_engine, "wait_for_exposure_update", lambda _t: None)
    monkeypatch.setattr(risk_engine, "refresh_positions", lambda _api: None)
    monkeypatch.setattr(risk_engine, "_adaptive_global_cap", lambda: 0.8)

    class DummyLock:
        def acquire(self, blocking: bool = False) -> bool:
            return True

        def release(self) -> None:
            pass

    monkeypatch.setattr(eng, "run_lock", DummyLock())
    monkeypatch.setattr(eng, "_prepare_run", lambda *_a, **_k: (0.0, True, []))
    monkeypatch.setattr(eng, "_process_symbols", lambda *_a, **_k: ([], {}))
    monkeypatch.setattr(eng, "_send_heartbeat", lambda: None)
    monkeypatch.setattr(eng.time, "sleep", lambda *_a, **_k: None)

    state = eng.BotState()

    eng.run_all_trades_worker(state, runtime)
    eng.run_all_trades_worker(state, runtime)

    init_logs = [r for r in caplog.records if "Risk engine initialized" in r.message]
    assert len(init_logs) == 1

