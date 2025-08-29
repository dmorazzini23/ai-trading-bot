import types
import sys
import importlib.machinery

validation_stub = types.ModuleType("ai_trading.validation")
require_env_stub = types.ModuleType("ai_trading.validation.require_env")
pydantic_settings_stub = types.ModuleType("pydantic_settings")
sklearn_stub = types.ModuleType("sklearn")
ensemble_stub = types.ModuleType("sklearn.ensemble")
metrics_stub = types.ModuleType("sklearn.metrics")
model_selection_stub = types.ModuleType("sklearn.model_selection")
preproc_stub = types.ModuleType("sklearn.preprocessing")
sklearn_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
bs4_stub = types.ModuleType("bs4")
flask_stub = types.ModuleType("flask")
prometheus_stub = types.ModuleType("prometheus_client")


class BeautifulSoup:  # noqa: D401
    pass


bs4_stub.BeautifulSoup = BeautifulSoup
class Flask:  # noqa: D401
    def __init__(self, *a, **k):
        pass

    def route(self, *a, **k):  # noqa: ANN001, ANN002
        def decorator(func):
            return func

        return decorator


flask_stub.Flask = Flask
prometheus_stub.REGISTRY = object()
prometheus_stub.Counter = object
prometheus_stub.Gauge = object
prometheus_stub.Histogram = object
prometheus_stub.Summary = object
prometheus_stub.CollectorRegistry = object
prometheus_stub.CONTENT_TYPE_LATEST = "text/plain"
prometheus_stub.generate_latest = lambda *a, **k: b""
prometheus_stub.start_http_server = lambda *a, **k: None


class _Dummy:  # noqa: D401
    pass


ensemble_stub.GradientBoostingClassifier = _Dummy
ensemble_stub.RandomForestClassifier = _Dummy
metrics_stub.accuracy_score = _Dummy
model_selection_stub.train_test_split = _Dummy
preproc_stub.StandardScaler = _Dummy

sklearn_stub.ensemble = ensemble_stub
sklearn_stub.metrics = metrics_stub
sklearn_stub.model_selection = model_selection_stub
sklearn_stub.preprocessing = preproc_stub
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.ensemble", ensemble_stub)
sys.modules.setdefault("sklearn.metrics", metrics_stub)
sys.modules.setdefault("sklearn.model_selection", model_selection_stub)
sys.modules.setdefault("sklearn.preprocessing", preproc_stub)
sys.modules.setdefault("joblib", types.ModuleType("joblib"))
sys.modules.setdefault("portalocker", types.ModuleType("portalocker"))
sys.modules.setdefault("bs4", bs4_stub)
sys.modules.setdefault("flask", flask_stub)
sys.modules.setdefault("prometheus_client", prometheus_stub)
pydantic_settings_stub.BaseSettings = object
pydantic_settings_stub.SettingsConfigDict = dict


def _should_halt_trading(*_a, **_k):  # noqa: D401
    return False


def _require_env_vars(*_a, **_k):  # noqa: D401
    return None


def _require_env_vars_public(*_a, **_k):  # noqa: D401
    return True


require_env_stub.should_halt_trading = _should_halt_trading
require_env_stub._require_env_vars = _require_env_vars
require_env_stub.require_env_vars = _require_env_vars_public
validation_stub.require_env = require_env_stub
validation_stub.should_halt_trading = _should_halt_trading
validation_stub._require_env_vars = _require_env_vars
validation_stub.require_env_vars = _require_env_vars_public
sys.modules.setdefault("ai_trading.validation", validation_stub)
sys.modules.setdefault("ai_trading.validation.require_env", require_env_stub)
sys.modules.setdefault("pydantic_settings", pydantic_settings_stub)

import ai_trading.core.bot_engine as eng

def test_service_start_without_data_fetcher_warning(monkeypatch, caplog):
    class DummyRiskEngine:
        def wait_for_exposure_update(self, timeout: float) -> None:
            pass

    class DummyAPI:
        def get_account(self):
            return types.SimpleNamespace(equity="0", cash="0")

        def list_positions(self):
            return []

    runtime = types.SimpleNamespace(
        api=DummyAPI(), risk_engine=DummyRiskEngine(), drawdown_circuit_breaker=None
    )
    state = eng.BotState()

    dummy_fetcher = object()
    dummy_ctx = types.SimpleNamespace(data_fetcher=dummy_fetcher)
    monkeypatch.setattr(eng, "_get_runtime_context_or_none", lambda: dummy_ctx)

    monkeypatch.setattr(eng, "_prepare_run", lambda runtime, state, tickers: (0.0, True, []))
    monkeypatch.setattr(eng, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(eng, "_init_metrics", lambda: None)
    monkeypatch.setattr(eng, "is_market_open", lambda: True)
    monkeypatch.setattr(eng, "ensure_alpaca_attached", lambda runtime: None)
    monkeypatch.setattr(eng, "check_pdt_rule", lambda runtime: False)
    monkeypatch.setattr(eng, "get_strategies", lambda: [])
    monkeypatch.setattr(eng, "get_verbose_logging", lambda: False)
    monkeypatch.setattr(eng, "_ensure_execution_engine", lambda runtime: None)
    monkeypatch.setattr(eng, "list_open_orders", lambda api: [])
    monkeypatch.setattr(eng, "_validate_trading_api", lambda api: True)
    monkeypatch.setattr(eng, "_send_heartbeat", lambda: None)
    monkeypatch.setattr(eng, "check_halt_flag", lambda runtime: False)
    monkeypatch.setattr(eng, "MEMORY_OPTIMIZATION_AVAILABLE", False, raising=False)
    monkeypatch.setattr(eng.CFG, "log_market_fetch", False, raising=False)

    class DummyLock:
        def acquire(self, blocking: bool = False) -> bool:
            return True

        def release(self) -> None:
            pass

    monkeypatch.setattr(eng, "run_lock", DummyLock())

    with caplog.at_level("WARNING"):
        eng.run_all_trades_worker(state, runtime)

    messages = [record.getMessage() for record in caplog.records]
    assert "DATA_FETCHER_MISSING" not in messages
