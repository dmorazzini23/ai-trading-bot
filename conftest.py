import os
import random
import importlib
import importlib.util
import sys
import types
import gc
import time as _time
from pathlib import Path
from typing import Any

from tests.dummy_model_util import _DummyModel, _get_model

import pytest


_ORIGINAL_TIME_SLEEP = _time.sleep


def _load_repo_sitecustomize() -> None:
    """Load repository-local ``sitecustomize.py`` deterministically for tests."""

    sitecustomize_path = Path(__file__).with_name("sitecustomize.py")
    if not sitecustomize_path.exists():
        return
    spec = importlib.util.spec_from_file_location("sitecustomize", sitecustomize_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["sitecustomize"] = module
    spec.loader.exec_module(module)


_load_repo_sitecustomize()

# Ensure optional light-weight stubs are available only when real deps are missing
def _ensure_test_stubs() -> None:
    repo = Path(__file__).parent.resolve()
    stubs = repo / "tests" / "stubs"
    if not stubs.exists():
        return

    def _missing(mod: str) -> bool:
        try:
            return importlib.util.find_spec(mod) is None
        except ValueError:
            return True

    need_stubs = any(
        _missing(m)
        for m in (
            "pydantic",
            "pydantic_settings",
            # Use a stub for Retry if urllib3 not installed
            "urllib3",
        )
    )
    if need_stubs and str(stubs) not in sys.path:
        sys.path.insert(0, str(stubs))


_ensure_test_stubs()

# Provide a lightweight default model so bot initialization can preload it

_dummy_mod = types.ModuleType("dummy_model")

setattr(_dummy_mod, "get_model", _get_model)
setattr(_dummy_mod, "_DummyModel", _DummyModel)
sys.modules["dummy_model"] = _dummy_mod
os.environ.setdefault("AI_TRADING_MODEL_MODULE", "dummy_model")
os.environ.setdefault("MAX_DRAWDOWN_THRESHOLD", "0.1")
os.environ.setdefault("DOLLAR_RISK_LIMIT", "0.05")
os.environ.setdefault("WEBHOOK_SECRET", "test-webhook-secret")
os.environ.setdefault("ALPACA_API_KEY", "test-key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test-secret")
os.environ["AI_TRADING_TRADE_HISTORY_PATH"] = str(
    Path(__file__).with_name(".pytest_trade_history.parquet")
)
os.environ["ALPACA_ALLOW_SIP"] = "1"
os.environ["ALPACA_SIP_ENTITLED"] = "1"
os.environ["ALPACA_HAS_SIP"] = "1"


def _missing(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is None
    except ValueError:
        return True


@pytest.fixture(scope="session", autouse=True)
def _seed_tests() -> None:
    """Ensure deterministic test execution."""
    os.environ["PYTEST_RUNNING"] = "1"
    os.environ["ALPACA_ALLOW_SIP"] = "1"
    os.environ["ALPACA_SIP_ENTITLED"] = "1"
    os.environ["ALPACA_HAS_SIP"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    if not _missing("numpy"):
        import numpy as np

        np.random.seed(0)
    if not _missing("torch"):
        import torch

        torch.manual_seed(0)


@pytest.fixture(autouse=True)
def _force_sip_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure SIP-related env defaults reflect entitlement-friendly state during tests."""

    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_SIP_ENTITLED", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.delenv("TRADING__DEGRADED_FEED_MODE", raising=False)
    monkeypatch.delenv("DEGRADED_FEED_MODE", raising=False)
    monkeypatch.delenv("ALPACA_SIP_UNAUTHORIZED", raising=False)
    monkeypatch.delenv("BACKUP_DATA_PROVIDER", raising=False)
    monkeypatch.delenv("BACKUP_PROVIDER", raising=False)
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)


def _clear_mutable_state(value: object) -> None:
    if isinstance(value, (dict, set, list)):
        value.clear()


def _iter_live_modules(module_name: str):
    """Yield all live module objects matching ``module_name``."""

    seen: set[int] = set()
    module = sys.modules.get(module_name)
    if isinstance(module, types.ModuleType):
        seen.add(id(module))
        yield module

    try:
        for obj in gc.get_objects():
            if not isinstance(obj, types.ModuleType):
                continue
            if getattr(obj, "__name__", None) != module_name:
                continue
            obj_id = id(obj)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            yield obj
    except Exception:
        return


def _reset_loaded_singletons() -> None:
    try:
        _time.sleep = _ORIGINAL_TIME_SLEEP
    except Exception:
        pass

    sys.modules.pop("ai_trading.__main__", None)

    try:
        fetch_mod = importlib.import_module("ai_trading.data.fetch")
    except Exception:
        fetch_mod = None
    if isinstance(fetch_mod, types.ModuleType):
        required_fetch_attrs = ("_FALLBACK_WINDOWS", "_FALLBACK_UNTIL", "_MINUTE_CACHE")
        if not all(hasattr(fetch_mod, attr) for attr in required_fetch_attrs):
            try:
                fetch_mod = importlib.reload(fetch_mod)
            except Exception:
                pass
            else:
                try:
                    sys.modules["ai_trading.data.fetch"] = fetch_mod
                except Exception:
                    pass
                data_pkg = sys.modules.get("ai_trading.data")
                if isinstance(data_pkg, types.ModuleType):
                    try:
                        setattr(data_pkg, "fetch", fetch_mod)
                    except Exception:
                        pass

    try:
        provider_monitor_mod = importlib.import_module("ai_trading.data.provider_monitor")
    except Exception:
        provider_monitor_mod = None
    try:
        config_pkg = importlib.import_module("ai_trading.config")
        config_mgmt_mod = importlib.import_module("ai_trading.config.management")
    except Exception:
        config_pkg = None
        config_mgmt_mod = None
    if isinstance(config_pkg, types.ModuleType) and isinstance(config_mgmt_mod, types.ModuleType):
        try:
            setattr(config_pkg, "management", config_mgmt_mod)
        except Exception:
            pass
    try:
        runtime_state_mod = importlib.import_module("ai_trading.telemetry.runtime_state")
    except Exception:
        runtime_state_mod = None

    for bot_engine_mod in _iter_live_modules("ai_trading.core.bot_engine"):
        if isinstance(fetch_mod, types.ModuleType):
            try:
                setattr(bot_engine_mod, "data_fetcher_module", fetch_mod)
            except Exception:
                pass
            try:
                data_fetch_err = getattr(fetch_mod, "DataFetchError", None)
                if data_fetch_err is not None:
                    setattr(bot_engine_mod, "DataFetchError", data_fetch_err)
            except Exception:
                pass
        if isinstance(provider_monitor_mod, types.ModuleType):
            try:
                monitor = getattr(provider_monitor_mod, "provider_monitor", None)
                if monitor is not None:
                    setattr(bot_engine_mod, "provider_monitor", monitor)
            except Exception:
                pass
            try:
                safe_reason_fn = getattr(provider_monitor_mod, "safe_mode_reason", None)
                if callable(safe_reason_fn):
                    setattr(bot_engine_mod, "safe_mode_reason", safe_reason_fn)
            except Exception:
                pass
        if isinstance(runtime_state_mod, types.ModuleType):
            try:
                setattr(bot_engine_mod, "runtime_state", runtime_state_mod)
            except Exception:
                pass

    try:
        vendor_requests_mod = importlib.import_module("tests.vendor_stubs.alpaca.data.requests")
    except Exception:
        vendor_requests_mod = None
    if isinstance(vendor_requests_mod, types.ModuleType):
        req_cls = getattr(vendor_requests_mod, "StockBarsRequest", None)
        needs_reload = not (
            isinstance(req_cls, type)
            and getattr(req_cls, "__name__", "") == "StockBarsRequest"
            and getattr(req_cls, "__module__", "") == "tests.vendor_stubs.alpaca.data.requests"
        )
        if needs_reload:
            try:
                vendor_requests_mod = importlib.reload(vendor_requests_mod)
            except Exception:
                pass

    try:
        vendor_timeframe_mod = importlib.import_module("tests.vendor_stubs.alpaca.data.timeframe")
    except Exception:
        vendor_timeframe_mod = None
    if isinstance(vendor_timeframe_mod, types.ModuleType):
        tf_cls = getattr(vendor_timeframe_mod, "TimeFrame", None)
        needs_reload = not (
            isinstance(tf_cls, type)
            and getattr(tf_cls, "__name__", "") == "TimeFrame"
            and getattr(tf_cls, "__module__", "") == "tests.vendor_stubs.alpaca.data.timeframe"
        )
        if needs_reload:
            try:
                vendor_timeframe_mod = importlib.reload(vendor_timeframe_mod)
            except Exception:
                pass

    alpaca_data_mod = sys.modules.get("alpaca.data")
    if isinstance(alpaca_data_mod, types.ModuleType):
        requests_mod = sys.modules.get("alpaca.data.requests")
        if isinstance(requests_mod, types.ModuleType):
            request_cls = getattr(requests_mod, "StockBarsRequest", None)
            if request_cls is not None:
                try:
                    setattr(alpaca_data_mod, "StockBarsRequest", request_cls)
                except Exception:
                    pass
        timeframe_mod = sys.modules.get("alpaca.data.timeframe")
        if isinstance(timeframe_mod, types.ModuleType):
            tf_cls = getattr(timeframe_mod, "TimeFrame", None)
            tf_unit_cls = getattr(timeframe_mod, "TimeFrameUnit", None)
            if tf_cls is not None:
                try:
                    setattr(alpaca_data_mod, "TimeFrame", tf_cls)
                except Exception:
                    pass
            if tf_unit_cls is not None:
                try:
                    setattr(alpaca_data_mod, "TimeFrameUnit", tf_unit_cls)
                except Exception:
                    pass

    try:
        from ai_trading.config import settings as _settings_mod

        cache_clear = getattr(_settings_mod.get_settings, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()
    except Exception:
        pass

    logging_mod = sys.modules.get("ai_trading.logging")
    if logging_mod is not None:
        throttle = getattr(logging_mod, "_THROTTLE_FILTER", None)
        reset = getattr(throttle, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass
        else:
            state = getattr(throttle, "_state", None)
            _clear_mutable_state(state)
        reset_provider_dedupe = getattr(logging_mod, "reset_provider_log_dedupe", None)
        if callable(reset_provider_dedupe):
            try:
                reset_provider_dedupe()
            except Exception:
                pass
    for logger_once_mod in _iter_live_modules("ai_trading.logging.logger_once"):
        emitted_keys = getattr(logger_once_mod, "_emitted_keys", None)
        _clear_mutable_state(emitted_keys)

    for fetch_mod in _iter_live_modules("ai_trading.data.fetch"):
        try:
            from ai_trading.data import market_calendar as _market_calendar_mod
        except Exception:
            _market_calendar_mod = None
        for fn_name in (
            "_clear_sip_lockout_for_tests",
            "_reset_provider_auth_state_for_tests",
            "refresh_alpaca_credentials_cache",
        ):
            fn = getattr(fetch_mod, fn_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        for attr_name in (
            "_state",
            "_BACKUP_SKIP_UNTIL",
            "_SKIPPED_SYMBOLS",
            "_FEED_FAILOVER_ATTEMPTS",
            "_FEED_OVERRIDE_BY_TF",
            "_FEED_SWITCH_CACHE",
            "_FEED_SWITCH_LOGGED",
            "_FEED_SWITCH_HISTORY",
            "_IEX_EMPTY_COUNTS",
            "_ALPACA_SYMBOL_FAILURES",
            "_ALPACA_CLOSE_NAN_COUNTS",
            "_ALPACA_FAILURE_EVENTS",
            "_ALPACA_CONSECUTIVE_FAILURES",
            "_ALPACA_EMPTY_ERROR_COUNTS",
            "_EMPTY_BAR_COUNTS",
            "_BACKUP_USAGE_LOGGED",
            "_FALLBACK_METADATA",
            "_FALLBACK_WINDOWS",
            "_FALLBACK_UNTIL",
            "_FALLBACK_SUPPRESS_UNTIL",
            "_daily_memo",
            "_MINUTE_CACHE",
            "_HOST_COUNTS",
            "_HOST_LIMITS",
            "_CYCLE_FALLBACK_FEED",
            "_SIP_UNAVAILABLE_LOGGED",
            "_BACKUP_USAGE_LOGGED",
            "_cycle_feed_override",
            "_override_set_ts",
            "_SAFE_MODE_LOGGED",
            "_SAFE_MODE_CYCLE_STATE",
        ):
            _clear_mutable_state(getattr(fetch_mod, attr_name, None))
        if hasattr(fetch_mod, "_DATA_FEED_OVERRIDE"):
            setattr(fetch_mod, "_DATA_FEED_OVERRIDE", None)
        if hasattr(fetch_mod, "_LAST_OVERRIDE_LOGGED"):
            setattr(fetch_mod, "_LAST_OVERRIDE_LOGGED", None)
        for attr_name, attr_value in (
            ("_BOOTSTRAP_PRIMARY_ONCE", True),
            ("_BOOTSTRAP_BACKUP_REASON", None),
            ("_alpaca_disabled_until", None),
            ("_alpaca_empty_streak", 0),
            ("_ALPACA_DISABLED_ALERTED", False),
            ("_alpaca_disable_count", 0),
            ("_ALPACA_KEYS_MISSING_LOGGED", False),
        ):
            if hasattr(fetch_mod, attr_name):
                try:
                    setattr(fetch_mod, attr_name, attr_value)
                except Exception:
                    pass
        if _market_calendar_mod is not None:
            try:
                setattr(fetch_mod, "is_trading_day", _market_calendar_mod.is_trading_day)
            except Exception:
                pass
            try:
                setattr(fetch_mod, "rth_session_utc", _market_calendar_mod.rth_session_utc)
            except Exception:
                pass

    active_fetch_mod = sys.modules.get("ai_trading.data.fetch")
    for iex_fallback_mod in _iter_live_modules("ai_trading.data.fetch.iex_fallback"):
        if not isinstance(active_fetch_mod, types.ModuleType):
            continue
        for attr_name in (
            "_HTTP_SESSION",
            "_ALLOW_SIP",
            "_SIP_UNAUTHORIZED",
            "_IEX_EMPTY_COUNTS",
            "_IEX_EMPTY_THRESHOLD",
            "_ALPACA_CONSECUTIVE_FAILURE_THRESHOLD",
            "_consecutive_failure_count",
            "_record_alpaca_failure_event",
            "_clear_alpaca_failure_events",
        ):
            if not hasattr(active_fetch_mod, attr_name):
                continue
            try:
                setattr(iex_fallback_mod, attr_name, getattr(active_fetch_mod, attr_name))
            except Exception:
                pass

    for fetch_metrics_mod in _iter_live_modules("ai_trading.data.fetch.metrics"):
        reset_metrics = getattr(fetch_metrics_mod, "reset", None)
        if callable(reset_metrics):
            try:
                reset_metrics()
            except Exception:
                pass

    for fallback_order_mod in _iter_live_modules("ai_trading.data.fetch.fallback_order"):
        reset_fallback_order = getattr(fallback_order_mod, "reset", None)
        if callable(reset_fallback_order):
            try:
                reset_fallback_order()
            except Exception:
                pass

    for provider_monitor_mod in _iter_live_modules("ai_trading.data.provider_monitor"):
        provider_monitor = getattr(provider_monitor_mod, "provider_monitor", None)
        reset_monitor = getattr(provider_monitor, "reset", None)
        if callable(reset_monitor):
            try:
                reset_monitor()
            except Exception:
                pass
        disable_cb = getattr(active_fetch_mod, "_disable_alpaca", None) if isinstance(active_fetch_mod, types.ModuleType) else None
        register_cb = getattr(provider_monitor, "register_disable_callback", None)
        if callable(register_cb) and callable(disable_cb):
            try:
                register_cb("alpaca", disable_cb)
            except Exception:
                pass

    for live_trading_mod in _iter_live_modules("ai_trading.execution.live_trading"):
        if isinstance(provider_monitor_mod, types.ModuleType):
            for attr_name in ("provider_monitor", "is_safe_mode_active", "safe_mode_reason"):
                if not hasattr(provider_monitor_mod, attr_name):
                    continue
                try:
                    setattr(live_trading_mod, attr_name, getattr(provider_monitor_mod, attr_name))
                except Exception:
                    pass
        for attr_name, attr_value in (
            ("_LONG_ONLY_ACCOUNT_MODE", False),
            ("_LONG_ONLY_ACCOUNT_REASON", None),
            ("_ACCOUNT_MARGIN_WARNING_LOGGED", False),
            ("_ACCOUNT_SHORTING_WARNING_LOGGED", False),
            ("_CONFIG_LONG_ONLY_LOGGED", False),
        ):
            if hasattr(live_trading_mod, attr_name):
                try:
                    setattr(live_trading_mod, attr_name, attr_value)
                except Exception:
                    pass

    for runtime_state_mod in _iter_live_modules("ai_trading.telemetry.runtime_state"):
        reset_all = getattr(runtime_state_mod, "reset_all_states", None)
        if callable(reset_all):
            try:
                reset_all()
            except Exception:
                pass

    for shutdown_mod in _iter_live_modules("ai_trading.runtime.shutdown"):
        stop_event = getattr(shutdown_mod, "stop_event", None)
        clear = getattr(stop_event, "clear", None)
        if callable(clear):
            try:
                clear()
            except Exception:
                pass

    for main_mod in _iter_live_modules("ai_trading.main"):
        if hasattr(main_mod, "_TRADE_LOG_INITIALIZED"):
            try:
                setattr(main_mod, "_TRADE_LOG_INITIALIZED", False)
            except Exception:
                pass
        if hasattr(main_mod, "_AUTH_PREFLIGHT_LOGGED"):
            try:
                setattr(main_mod, "_AUTH_PREFLIGHT_LOGGED", False)
            except Exception:
                pass
        if hasattr(main_mod, "_TEST_ALPACA_CREDS_BACKFILLED"):
            try:
                setattr(main_mod, "_TEST_ALPACA_CREDS_BACKFILLED", False)
            except Exception:
                pass
        reset_warmup = getattr(main_mod, "_reset_warmup_cooldown_timestamp", None)
        if callable(reset_warmup):
            try:
                reset_warmup()
            except Exception:
                pass

    for persistence_mod in _iter_live_modules("ai_trading.meta_learning.persistence"):
        canonical_path = Path(os.environ["AI_TRADING_TRADE_HISTORY_PATH"]).resolve()
        try:
            setattr(persistence_mod, "_CANONICAL_PATH", canonical_path)
        except Exception:
            pass
        try:
            if canonical_path.exists():
                canonical_path.unlink()
        except Exception:
            pass


def _sync_package_module_exports() -> None:
    """Ensure package-level module attributes point at canonical sys.modules objects."""

    export_map = (
        ("ai_trading.core", "bot_engine", "ai_trading.core.bot_engine"),
        ("ai_trading.data", "fetch", "ai_trading.data.fetch"),
        ("ai_trading.config", "management", "ai_trading.config.management"),
        ("ai_trading", "alpaca_api", "ai_trading.alpaca_api"),
        ("ai_trading", "predict", "ai_trading.predict"),
    )

    for package_name, attr_name, module_name in export_map:
        if module_name not in sys.modules:
            try:
                importlib.import_module(module_name)
            except Exception:
                pass
        package_mod = sys.modules.get(package_name)
        target_mod = sys.modules.get(module_name)
        if not isinstance(package_mod, types.ModuleType):
            continue
        if not isinstance(target_mod, types.ModuleType):
            continue
        try:
            setattr(package_mod, attr_name, target_mod)
        except Exception:
            continue

    predict_mod = sys.modules.get("ai_trading.predict")
    if isinstance(predict_mod, types.ModuleType):
        sys.modules.setdefault("predict", predict_mod)


_REBOUND_MODULE_NAMES = {
    "ai_trading.data.fetch",
    "ai_trading.data.fetch.iex_fallback",
    "ai_trading.data.provider_monitor",
    "ai_trading.core.bot_engine",
    "ai_trading.main",
    "ai_trading.meta_learning",
    "ai_trading.execution.engine",
    "ai_trading.execution.live_trading",
    "ai_trading.config.mode_aliases",
}

_REBOUND_SYMBOL_MODULE_NAMES = {
    "ai_trading.data.fetch",
    "ai_trading.data.fetch.iex_fallback",
    "ai_trading.data.provider_monitor",
    "ai_trading.core.bot_engine",
    "ai_trading.alpaca_api",
    "ai_trading.main",
    "ai_trading.meta_learning",
    "ai_trading.execution.engine",
    "ai_trading.execution.live_trading",
    "ai_trading.utils",
    "ai_trading.utils.retry",
    "ai_trading.config.mode_aliases",
    "ai_trading.risk.adaptive_sizing",
    "ai_trading.strategies.regime_detector",
}

_REBOUND_MUTABLE_SYMBOLS: dict[str, set[str]] = {
    "ai_trading.data.fetch": {"_MINUTE_CACHE", "_IEX_EMPTY_COUNTS"},
    "ai_trading.data.fetch.empty_handling": {"_RETRY_COUNTS"},
    "ai_trading.utils.timing": {"HTTP_TIMEOUT", "clamp_timeout"},
    "tests.vendor_stubs.alpaca.data.requests": {"StockBarsRequest"},
    "tests.vendor_stubs.alpaca.data.timeframe": {"TimeFrame", "TimeFrameUnit"},
}


def _rebind_test_module_references(test_module: object | None) -> None:
    """Refresh stale module globals in collected test modules."""

    if not isinstance(test_module, types.ModuleType):
        return

    mock_config = getattr(test_module, "MockConfig", None)
    if mock_config is not None:
        try:
            sys.modules["config"] = mock_config
        except Exception:
            pass

    module_refs: dict[str, types.ModuleType] = {}
    for attr_value in vars(test_module).values():
        if not isinstance(attr_value, types.ModuleType):
            continue
        module_name = getattr(attr_value, "__name__", None)
        if isinstance(module_name, str):
            module_refs.setdefault(module_name, attr_value)

    for attr_name, attr_value in list(vars(test_module).items()):
        if isinstance(attr_value, types.ModuleType):
            module_name = getattr(attr_value, "__name__", None)
            if module_name not in _REBOUND_MODULE_NAMES:
                continue
            try:
                setattr(test_module, attr_name, importlib.import_module(module_name))
            except Exception:
                continue
            continue

        owner_module = getattr(attr_value, "__module__", None)
        if owner_module not in _REBOUND_SYMBOL_MODULE_NAMES:
            rebound = False
            for module_name, symbol_names in _REBOUND_MUTABLE_SYMBOLS.items():
                if (
                    module_name.startswith("tests.vendor_stubs.")
                    and getattr(test_module, "__name__", "") != "tests.test_vendor_stub_alpaca_requests"
                ):
                    continue
                if attr_name not in symbol_names:
                    continue
                current_module = module_refs.get(module_name)
                if not isinstance(current_module, types.ModuleType):
                    current_module = sys.modules.get(module_name)
                if not isinstance(current_module, types.ModuleType):
                    try:
                        current_module = importlib.import_module(module_name)
                    except Exception:
                        continue
                if not isinstance(current_module, types.ModuleType):
                    continue
                if not hasattr(current_module, attr_name):
                    continue
                try:
                    setattr(test_module, attr_name, getattr(current_module, attr_name))
                except Exception:
                    continue
                rebound = True
                break
            if rebound:
                continue
            continue
        current_module = module_refs.get(owner_module)
        if not isinstance(current_module, types.ModuleType):
            current_module = sys.modules.get(owner_module)
        if not isinstance(current_module, types.ModuleType):
            try:
                current_module = importlib.import_module(owner_module)
            except Exception:
                continue
        if not isinstance(current_module, types.ModuleType):
            continue
        if not hasattr(current_module, attr_name):
            continue
        try:
            setattr(test_module, attr_name, getattr(current_module, attr_name))
        except Exception:
            continue


@pytest.fixture(autouse=True)
def _reset_runtime_singletons(request: pytest.FixtureRequest) -> None:
    """Keep mutable process-global state isolated between tests."""

    _reset_loaded_singletons()
    _sync_package_module_exports()
    _rebind_test_module_references(getattr(request, "module", None))
    yield


@pytest.fixture(autouse=True)
def _restore_real_sleep_for_prof_budget(
    request: pytest.FixtureRequest,
) -> None:
    """Keep real sleep semantics for precision budget timing tests."""

    try:
        request.getfixturevalue("_short_sleep")
    except Exception:
        pass
    if "tests/test_prof_budget.py" in request.node.nodeid:
        try:
            _time.sleep = _ORIGINAL_TIME_SLEEP
        except Exception:
            pass
    yield


def pytest_ignore_collect(path: Path, config: Any) -> bool:
    """Ignore heavy or optional-dep tests when prerequisites are missing."""
    p = Path(str(path))
    needs_sklearn = p.name == "test_meta_learning_heavy.py" or "slow" in p.parts
    if needs_sklearn and _missing("sklearn"):
        return True
    if _missing("numpy"):
        repo = Path(__file__).parent.resolve()
        allowed = {
            repo / "tests" / "test_runner_smoke.py",
            repo / "tests" / "test_utils_timing.py",
            repo / "tests" / "unit" / "test_trading_config_aliases.py",
            repo / "tests" / "test_current_api.py",
        }
        try:
            rel = p.resolve()
        except FileNotFoundError:
            rel = p
        if p.is_dir():
            return not any(a.is_relative_to(rel) for a in allowed)
        return rel not in allowed
    return False
