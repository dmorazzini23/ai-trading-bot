"""Ensure run_all_trades_worker does not warn for a valid API client."""

from __future__ import annotations

import types
import sys
from typing import Any, cast
from unittest.mock import Mock, call


from ai_trading.telemetry import runtime_state

sys = cast(Any, sys)


def _set_module_attr(module: types.ModuleType, attr_name: str, value: Any) -> None:
    setattr(module, attr_name, value)


def _zeros(shape: Any, dtype: Any = None) -> Any:
    if isinstance(shape, int):
        return [0] * shape
    return [[0 for _ in range(shape[1])] for _ in range(shape[0])]


def _ones(shape: Any, dtype: Any = None) -> Any:
    if isinstance(shape, int):
        return [1] * shape
    return [[1 for _ in range(shape[1])] for _ in range(shape[0])]


sklearn_stub = types.ModuleType("sklearn")
ensemble_stub = types.ModuleType("sklearn.ensemble")

class _GB:  # noqa: D401 - minimal placeholder
    pass


class _RF:  # noqa: D401 - minimal placeholder
    pass


_set_module_attr(ensemble_stub, "GradientBoostingClassifier", _GB)
_set_module_attr(ensemble_stub, "RandomForestClassifier", _RF)
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.ensemble", ensemble_stub)
metrics_stub = types.ModuleType("sklearn.metrics")
_set_module_attr(metrics_stub, "accuracy_score", lambda *a, **k: 0.0)
sys.modules.setdefault("sklearn.metrics", metrics_stub)
model_selection_stub = types.ModuleType("sklearn.model_selection")
_set_module_attr(model_selection_stub, "train_test_split", lambda *a, **k: ([], []))
sys.modules.setdefault("sklearn.model_selection", model_selection_stub)
preproc_stub = types.ModuleType("sklearn.preprocessing")
_set_module_attr(preproc_stub, "StandardScaler", type("StandardScaler", (), {}))
sys.modules.setdefault("sklearn.preprocessing", preproc_stub)

numpy_stub = types.ModuleType("numpy")
_set_module_attr(numpy_stub, "nan", float("nan"))
_set_module_attr(numpy_stub, "NaN", float("nan"))
_set_module_attr(numpy_stub, "isfinite", lambda *_a, **_k: True)
_set_module_attr(numpy_stub, "asarray", lambda value, *a, **k: value)
_set_module_attr(numpy_stub, "array", lambda value, *a, **k: value)
_set_module_attr(numpy_stub, "stack", lambda seq, *a, **k: list(seq))
_set_module_attr(numpy_stub, "std", lambda *_a, **_k: 0.0)
_set_module_attr(numpy_stub, "mean", lambda *_a, **_k: 0.0)
_set_module_attr(numpy_stub, "clip", lambda arr, *a, **k: arr)
_set_module_attr(numpy_stub, "argmax", lambda *_a, **_k: 0)
_set_module_attr(numpy_stub, "where", lambda cond, x, y: x if cond else y)
_set_module_attr(numpy_stub, "float32", float)
_set_module_attr(numpy_stub, "float64", float)
_set_module_attr(numpy_stub, "eye", lambda n, dtype=None: [
    [1 if i == j else 0 for j in range(n)] for i in range(n)
])
_set_module_attr(numpy_stub, "zeros", _zeros)
_set_module_attr(numpy_stub, "ones", _ones)
_set_module_attr(numpy_stub, "random", types.SimpleNamespace(seed=lambda *_a, **_k: None))
_set_module_attr(numpy_stub, "polyfit", lambda *_a, **_k: [0.0])
_set_module_attr(numpy_stub, "where", lambda cond, x, y: x if cond else y)
_set_module_attr(numpy_stub, "isnan", lambda *_a, **_k: False)
_set_module_attr(numpy_stub, "isscalar", lambda obj: not isinstance(obj, (list, tuple, dict, set)))
_set_module_attr(numpy_stub, "bool_", bool)
sys.modules.setdefault("numpy", numpy_stub)

portalocker_stub = types.ModuleType("portalocker")
_set_module_attr(portalocker_stub, "LOCK_EX", 1)
_set_module_attr(portalocker_stub, "lock", lambda *_a, **_k: None)
_set_module_attr(portalocker_stub, "unlock", lambda *_a, **_k: None)
sys.modules.setdefault("portalocker", portalocker_stub)

bs4_stub = types.ModuleType("bs4")
_set_module_attr(bs4_stub, "BeautifulSoup", type("BeautifulSoup", (), {}))
sys.modules.setdefault("bs4", bs4_stub)

class _Flask:
    def __init__(self, *_a, **_k):
        pass

    def route(self, *args, **kwargs):  # noqa: D401
        """Return a no-op decorator."""

        def decorator(fn):
            return fn

        return decorator

    def run(self, *args, **kwargs):  # noqa: D401
        """No-op run."""

        return None


flask_stub = types.ModuleType("flask")
_set_module_attr(flask_stub, "Flask", _Flask)
_set_module_attr(flask_stub, "jsonify", lambda obj=None, *a, **k: obj)
sys.modules.setdefault("flask", flask_stub)

cachetools_stub = types.ModuleType("cachetools")


class _TTLCache(dict):
    def __init__(self, maxsize=128, ttl=600):
        super().__init__()
        self.maxsize = maxsize
        self.ttl = ttl

    def __setitem__(self, key, value):  # noqa: D401
        """Insert while enforcing a crude max size."""

        if len(self) >= self.maxsize:
            try:
                first_key = next(iter(self))
                super().__delitem__(first_key)
            except StopIteration:  # pragma: no cover - defensive
                pass
        super().__setitem__(key, value)


_set_module_attr(cachetools_stub, "TTLCache", _TTLCache)
sys.modules.setdefault("cachetools", cachetools_stub)

import ai_trading.core.bot_engine as eng


def test_run_all_trades_no_warning_with_valid_api(monkeypatch):
    """A valid client with get_orders should not trigger a warning."""

    # Stub Alpaca modules so native get_orders receives a GetOrdersRequest.
    enums_mod = types.ModuleType("alpaca.trading.enums")
    requests_mod = types.ModuleType("alpaca.trading.requests")
    common_enums_mod = types.ModuleType("alpaca.common.enums")

    class Sort:
        ASC = "asc"

    class QueryOrderStatus:
        OPEN = "open"
        ALL = "all"

    class GetOrdersRequest:
        def __init__(self, *, status=None, limit=None, direction=None, after=None):
            self.status = status
            self.limit = limit
            self.direction = direction
            self.after = after

    _set_module_attr(common_enums_mod, "Sort", Sort)
    _set_module_attr(enums_mod, "QueryOrderStatus", QueryOrderStatus)
    _set_module_attr(requests_mod, "GetOrdersRequest", GetOrdersRequest)
    monkeypatch.setitem(sys.modules, "alpaca", types.ModuleType("alpaca"))
    monkeypatch.setitem(sys.modules, "alpaca.common", types.ModuleType("alpaca.common"))
    monkeypatch.setitem(sys.modules, "alpaca.common.enums", common_enums_mod)
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

        def cancel_order_by_id(self, *args, **kwargs):  # noqa: D401 - stub
            """Provide minimal cancel capability for validation."""
            return None

        def get_all_positions(self):
            return []

        def get_order_by_id(self, order_id):
            return None

        def submit_order(self, *args, **kwargs):
            return None

    class DummyRiskEngine:
        def wait_for_exposure_update(self, timeout: float) -> None:  # noqa: D401
            """No-op risk update."""

    state = eng.BotState()
    api = DummyAPI()
    # The native-client guard must see the same client class as this fixture.
    monkeypatch.setattr("ai_trading.core.alpaca_client.get_trading_client_cls", lambda: DummyAPI)
    runtime = types.SimpleNamespace(api=api, risk_engine=DummyRiskEngine())

    # Minimal patches to isolate the order-check logic
    monkeypatch.setattr(eng, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(eng, "_init_metrics", lambda: None)
    monkeypatch.setattr(eng, "log_env_diag", lambda *_a, **_k: {})
    monkeypatch.setattr(eng, "is_market_open", lambda: True)
    monkeypatch.setattr(eng, "ensure_alpaca_attached", lambda runtime: None)
    monkeypatch.setattr(eng, "get_verbose_logging", lambda: False)
    # The order check runs before the netting cycle; the cycle itself is out of scope here.
    monkeypatch.setattr(eng, "_run_netting_cycle", lambda *_a, **_k: None)
    monkeypatch.setattr(eng.CFG, "log_market_fetch", False, raising=False)
    stub_fetcher = types.SimpleNamespace()
    monkeypatch.setattr(
        eng.data_fetcher_module, "build_fetcher", lambda *_a, **_k: stub_fetcher
    )
    monkeypatch.setattr(eng, "data_fetcher", stub_fetcher, raising=False)

    class DummyLock:
        def acquire(self, blocking: bool = False) -> bool:  # noqa: D401
            """Always acquire."""

            return True

        def release(self) -> None:
            """No-op release."""

    monkeypatch.setattr(eng, "run_lock", DummyLock())

    warn_mock = Mock()
    info_mock = Mock()
    monkeypatch.setattr(eng.logger_once, "warning", warn_mock)
    monkeypatch.setattr(eng.logger_once, "info", info_mock)

    eng.run_all_trades_worker(state, runtime)

    info_mock.assert_not_called()
    warn_mock.assert_not_called()
    assert api.called_with is not None
    assert "filter" in api.called_with
    assert getattr(api.called_with["filter"], "status", None) in (
        QueryOrderStatus.OPEN,
        QueryOrderStatus.ALL,
        "open",
        "all",
    )


def test_run_all_trades_worker_netting_invokes_execution_cycle_hooks(monkeypatch):
    runtime_state.reset_data_provider_state()

    class DummyExecutionEngine:
        def __init__(self) -> None:
            self.start_cycle_called = 0
            self.end_cycle_called = 0
            self.sync_called = 0

        def start_cycle(self) -> None:
            self.start_cycle_called += 1

        def end_cycle(self) -> None:
            self.end_cycle_called += 1

        def synchronize_broker_state(self):
            self.sync_called += 1
            return types.SimpleNamespace(open_orders=[], positions=[])

    class DummyRiskEngine:
        def wait_for_exposure_update(self, _timeout: float) -> None:
            return None

    class DummyAPI:
        def get_orders(self, *args, **kwargs):  # noqa: D401 - minimal stub
            return []

        def cancel_order_by_id(self, *args, **kwargs):  # noqa: D401 - minimal stub
            return None

        def get_all_positions(self):
            return []

        def get_order_by_id(self, order_id):
            return None

        def submit_order(self, *args, **kwargs):
            return None

    class DummyLock:
        def acquire(self, blocking: bool = False) -> bool:
            return True

        def release(self) -> None:
            return None

    state = eng.BotState()
    state.startup_cleanup_done = True
    dummy_exec = DummyExecutionEngine()
    runtime = types.SimpleNamespace(
        api=DummyAPI(),
        risk_engine=DummyRiskEngine(),
        execution_engine=dummy_exec,
        exec_engine=dummy_exec,
        tickers=["AAPL"],
        universe_tickers=["AAPL"],
    )

    monkeypatch.setattr(eng, "_ALPACA_IMPORT_ERROR", None)
    monkeypatch.setattr(eng, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(eng, "_init_metrics", lambda: None)
    monkeypatch.setattr(eng, "run_lock", DummyLock())
    monkeypatch.setattr(eng, "_ensure_execution_engine", lambda _runtime: None)
    monkeypatch.setattr(eng, "_enforce_dependency_preflight", lambda _runtime: None)
    monkeypatch.setattr(
        eng,
        "_resolve_trading_config",
        lambda _runtime: types.SimpleNamespace(
            rth_only=False,
            allow_extended=True,
            post_submit_broker_sync=True,
        ),
    )
    monkeypatch.setattr(
        eng,
        "_active_effective_policy",
        lambda _state, _cfg: types.SimpleNamespace(
            trading_mode="balanced",
            objective=types.SimpleNamespace(objective_name="expected_net_edge_after_fees_slippage_borrow"),
        ),
    )
    monkeypatch.setattr(eng, "write_run_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(eng, "_persist_effective_policy_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(eng, "is_market_open", lambda: True)
    monkeypatch.setattr(eng, "ensure_alpaca_attached", lambda _runtime: None)
    monkeypatch.setattr(eng, "_validate_trading_api", lambda _api: True)
    monkeypatch.setattr(eng, "list_open_orders", lambda _api: [])
    monkeypatch.setattr(eng, "_handle_pending_orders", lambda _orders, _runtime: False)
    monkeypatch.setattr(eng, "_check_runtime_stops", lambda _runtime: None)
    monkeypatch.setattr(eng, "_log_loop_heartbeat", lambda *_a, **_k: None)
    monkeypatch.setattr(eng, "flush_log_throttle_summaries", lambda: None)
    monkeypatch.setattr(eng.provider_monitor, "is_safe_mode_active", lambda: False)
    broker_sync_metrics_calls = {"count": 0}

    def _record_broker_sync_metrics(_state, _snapshot):
        broker_sync_metrics_calls["count"] += 1

    monkeypatch.setattr(eng, "_record_broker_sync_metrics", _record_broker_sync_metrics)

    netting_calls = {"count": 0}

    def _run_netting_stub(*_args, **_kwargs):
        netting_calls["count"] += 1

    monkeypatch.setattr(eng, "_run_netting_cycle", _run_netting_stub)

    eng.run_all_trades_worker(state, runtime)

    assert netting_calls["count"] == 1
    assert dummy_exec.start_cycle_called == 1
    assert dummy_exec.end_cycle_called == 1
    assert dummy_exec.sync_called == 1
    assert broker_sync_metrics_calls["count"] == 1
    provider_state = runtime_state.observe_data_provider_state()
    assert provider_state.get("status") == "healthy"
    assert provider_state.get("data_status") == "ready"
    assert provider_state.get("reason") == "data_available_netting"
