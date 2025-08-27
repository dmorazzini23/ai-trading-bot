from __future__ import annotations

import types
import sys

sklearn_stub = types.ModuleType("sklearn")
ensemble_stub = types.ModuleType("sklearn.ensemble")

class _GB:  # noqa: D401 - placeholder
    pass


class _RF:  # noqa: D401 - placeholder
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


def test_run_all_trades_handles_api_error(monkeypatch, caplog):
    # Provide Alpaca stubs so the shim forwards a GetOrdersRequest
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
        def wait_for_exposure_update(self, timeout: float) -> None:
            pass

    state = eng.BotState()
    api = DummyAPI()
    runtime = types.SimpleNamespace(api=api, risk_engine=DummyRiskEngine())

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

    def raise_api_error(*_a, **_k):
        raise eng.APIError("boom")

    monkeypatch.setattr(eng, "_prepare_run", raise_api_error)

    caplog.set_level("WARNING")
    eng.run_all_trades_worker(state, runtime)

    assert any(str(r.msg).startswith("PREP") for r in caplog.records)
    assert state.running is False
    assert api.called_with is not None
    assert "filter" in api.called_with
    assert api.called_with["filter"].statuses == [OrderStatus.OPEN]
