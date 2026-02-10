"""Test configuration and shared fixtures."""

import os

os.environ.setdefault("PYTEST_RUNNING", "1")

import sys as _sys

import asyncio
import importlib
import random
import socket
import sys
import unittest.mock as _mock
from datetime import datetime, timezone
import pathlib
import types

_STUB_PATH = pathlib.Path(__file__).resolve().parent / "stubs"
_STUB_PATH_STR = str(_STUB_PATH)

if "dotenv" not in sys.modules:
    removed_stub_for_dotenv = False
    if _STUB_PATH_STR in _sys.path:
        _sys.path.remove(_STUB_PATH_STR)
        removed_stub_for_dotenv = True

    try:
        dotenv_mod = importlib.import_module("dotenv")
    except ModuleNotFoundError:
        dotenv_stub = types.ModuleType("dotenv")
        dotenv_stub.load_dotenv = lambda *args, **kwargs: None
        sys.modules["dotenv"] = dotenv_stub
    else:
        sys.modules["dotenv"] = dotenv_mod
    finally:
        if removed_stub_for_dotenv:
            _sys.path.append(_STUB_PATH_STR)

import pytest

# NOTE: Avoid importing `ai_trading` modules at collection time; lazy import in helpers.


_ORIGINAL_PATCH_DICT = _mock.patch.dict
_ESSENTIAL = (
    "importlib",
    "importlib._bootstrap",
    "importlib._bootstrap_external",
    "importlib.machinery",
    "pathlib",
    "posixpath",
    "ntpath",
    "os",
    "collections",
    "runpy",
    "zipimport",
    "pkgutil",
    "zoneinfo",
    "zoneinfo._tzpath",
)
for _name in _ESSENTIAL:
    try:
        __import__(_name)
    except Exception:
        continue

_SNAPSHOT_MODULES = dict(sys.modules)
try:  # Ensure real pandas module is loaded before tests that set stubs conditionally
    import pandas  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - environment without pandas
    pandas = None  # noqa: F841

for _module_name in (
    "ai_trading.portfolio",
    "sklearn",
    "sklearn.linear_model",
    "sklearn.metrics",
    "sklearn.model_selection",
    "sklearn.preprocessing",
    "sklearn.pipeline",
    "sklearn.ensemble",
):
    try:  # Pre-load selected modules so setdefault stubs used by tests do not replace them.
        importlib.import_module(_module_name)
    except Exception:  # pragma: no cover - optional dependency may be unavailable
        continue


def _safe_patch_dict(in_dict, values=(), clear: bool = False, **kwargs):  # pragma: no cover - test helper
    ctx = _ORIGINAL_PATCH_DICT(in_dict, values, clear, **kwargs)
    if clear and in_dict is sys.modules:
        original_enter = ctx.__enter__
        original_exit = ctx.__exit__

        def _enter():
            result = original_enter()
            sys.modules.update({name: module for name, module in _SNAPSHOT_MODULES.items() if module is not None})
            return result

        def _exit(exc_type, exc_val, exc_tb):
            try:
                return original_exit(exc_type, exc_val, exc_tb)
            finally:
                sys.modules.update({name: module for name, module in _SNAPSHOT_MODULES.items() if module is not None})

        ctx.__enter__ = _enter
        ctx.__exit__ = _exit
    return ctx


_mock.patch.dict = _safe_patch_dict


if "flask" not in _sys.modules:
    flask_mod = types.ModuleType("flask")
    flask_app_mod = types.ModuleType("flask.app")

    class _StubFlask:
        def __init__(self, *a, **k):
            self._routes = {}
            self.config = {}

        def route(self, path, *a, **k):
            def decorator(func):
                self._routes[path] = func
                return func

            return decorator

        def run(self, *a, **k):
            pass

        def test_client(self):
            app = self

            class _Response:
                def __init__(self, data, status=200):
                    self._data = data
                    self.status_code = status

                def get_json(self):
                    return self._data

            class _Client:
                def get(self, path):
                    handler = app._routes[path]
                    result = handler()
                    status = 200
                    data = result
                    if isinstance(result, tuple):
                        data = result[0]
                        if len(result) > 1:
                            status = result[1]
                    return _Response(data, status)

            return _Client()

    def _jsonify(payload=None, *args, **kwargs):
        if payload is not None:
            return payload
        if args:
            return args[0] if len(args) == 1 else list(args)
        return kwargs

    flask_mod.Flask = _StubFlask
    flask_mod.jsonify = getattr(flask_mod, "jsonify", _jsonify)
    flask_app_mod.Flask = _StubFlask
    flask_app_mod.jsonify = flask_mod.jsonify
    _sys.modules["flask"] = flask_mod
    _sys.modules["flask.app"] = flask_app_mod

try:
    from alpaca.trading.client import TradingClient  # type: ignore  # noqa: F401
    from alpaca.data import TimeFrame  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - dependency missing
    from tests.vendor_stubs import alpaca as _alpaca
    import sys as _sys

    _sys.modules.setdefault("alpaca", _alpaca)
    _sys.modules.setdefault("alpaca.trading", _alpaca.trading)
    _sys.modules.setdefault("alpaca.trading.client", _alpaca.trading.client)
    _sys.modules.setdefault("alpaca.data", _alpaca.data)
    _sys.modules.setdefault("alpaca.data.timeframe", _alpaca.data.timeframe)
    _sys.modules.setdefault("alpaca.data.requests", _alpaca.data.requests)

    from tests.vendor_stubs.alpaca.trading.client import TradingClient  # type: ignore  # noqa: F401
    from tests.vendor_stubs.alpaca.data.timeframe import TimeFrame  # type: ignore  # noqa: F401

_BASE_THIRD_PARTY_MODULES = {
    name: sys.modules.get(name)
    for name in (
        "pandas",
        "ai_trading.portfolio",
        "ai_trading.portfolio.core",
        "ai_trading.portfolio.optimizer",
        "sklearn",
        "sklearn.linear_model",
        "sklearn.metrics",
        "sklearn.pipeline",
        "sklearn.preprocessing",
        "alpaca",
        "alpaca.trading",
        "alpaca.trading.client",
        "alpaca.data",
        "alpaca.data.timeframe",
        "alpaca.data.requests",
        "alpaca.common",
        "alpaca.common.exceptions",
    )
}

try:
    from freezegun import freeze_time  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    from contextlib import contextmanager

    @contextmanager
    def freeze_time(*_a, **_k):  # AI-AGENT-REF: no-op when freezegun missing
        yield


@pytest.fixture(autouse=True, scope="session")
def _event_loop_policy():
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


@pytest.fixture(autouse=True)
def _freeze_clock():
    with freeze_time("2024-01-02 15:04:05", tz_offset=0):
        yield


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def guard(*a, **k):
        raise RuntimeError("Network disabled in tests")

    monkeypatch.setattr(socket, "create_connection", guard, raising=True)


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "dummy")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "dummy")


@pytest.fixture(autouse=True)
def _restore_third_party_modules():
    """Ensure third-party module stubs installed by tests do not leak globally."""

    def _restore_from_baseline(module_name: str) -> None:
        baseline = _BASE_THIRD_PARTY_MODULES.get(module_name)
        if baseline is None:
            sys.modules.pop(module_name, None)
            return
        sys.modules[module_name] = baseline

    yield
    restore_targets = (
        (
            "pandas",
            lambda m: getattr(m, "DataFrame", None) is not None,
            (),
        ),
        (
            "ai_trading.portfolio",
            lambda m: hasattr(m, "compute_portfolio_weights"),
            ("ai_trading.portfolio.core", "ai_trading.portfolio.optimizer"),
        ),
        (
            "sklearn",
            lambda m: getattr(m, "__file__", None) is not None,
            ("sklearn.linear_model", "sklearn.metrics", "sklearn.pipeline", "sklearn.preprocessing"),
        ),
        (
            "alpaca",
            lambda m: getattr(m, "__path__", None) is not None,
            (
                "alpaca.trading",
                "alpaca.trading.client",
                "alpaca.data",
                "alpaca.data.timeframe",
                "alpaca.data.requests",
                "alpaca.common",
                "alpaca.common.exceptions",
            ),
        ),
    )
    for module_name, validator, extra_modules in restore_targets:
        module = sys.modules.get(module_name)
        # Always hard-restore alpaca modules because some tests mutate module
        # attributes in-place (not only module objects). Validating only
        # ``__path__`` misses those leaks and causes cross-test contamination.
        if module_name == "alpaca":
            needs_restore = True
        else:
            needs_restore = module is not None and not validator(module)
        if needs_restore:
            _restore_from_baseline(module_name)
            for extra in extra_modules:
                _restore_from_baseline(extra)


@pytest.fixture(autouse=True)
def _seed_prng() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(0)
    try:
        import numpy as np

        np.random.seed(0)
    except Exception:
        pass
    try:
        import torch  # type: ignore
    except Exception:
        pass
    else:
        torch.manual_seed(0)


def reload_module(mod):
    """Reload a module by object or name for tests."""
    import importlib

    if isinstance(mod, str):
        return importlib.reload(importlib.import_module(mod))
    return importlib.reload(mod)

# Compatibility fixtures expected by some tests
@pytest.fixture(name="default_env")
def _default_env(_env_defaults):  # reuse existing autouse env defaults
    """Alias fixture: provides default env via autouse `_env_defaults`."""
    yield


@pytest.fixture(name="_monkeypatch")
def _monkeypatch_fixture(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Provide legacy `_monkeypatch` alias used by older tests."""

    return monkeypatch

@pytest.fixture
def dummy_data_fetcher():
    """Provide a minimal data_fetcher interface for unit tests.

    Exposes `get_daily_df(ctx, sym)` and `get_minute_bars(sym)` returning
    a simple 30-row OHLCV DataFrame for deterministic checks.
    """
    import pandas as pd

    class _F:
        def __init__(self):
            n = 30
            self._df = pd.DataFrame(
                {
                    "open": [1.0] * n,
                    "high": [1.0] * n,
                    "low": [1.0] * n,
                    "close": [1.0] * n,
                    "volume": [100] * n,
                }
            )

        def get_daily_df(self, ctx, sym):  # noqa: ARG002 - ctx unused in tests
            return self._df.copy()

        def get_minute_bars(self, sym):  # noqa: ARG002
            return self._df.copy()

    return _F()


@pytest.fixture
def dummy_data_fetcher_empty():
    """Provide a data_fetcher with empty minute bars for failure simulations."""

    import pandas as pd

    class _F:
        def __init__(self):
            self._df = pd.DataFrame(
                {"open": [], "high": [], "low": [], "close": [], "volume": []}
            )

        def get_daily_df(self, ctx, sym):  # noqa: ARG002 - tests expect method
            return self._df.copy()

        def get_minute_bars(self, sym):  # noqa: ARG002
            return self._df.copy()

    return _F()


@pytest.fixture(autouse=True)
def _reset_fallback_cache():
    """Reset fallback cache state without eager package imports."""

    import ai_trading.data.fetch as data_fetcher  # local import to avoid test import side effects

    def _reset() -> None:
        data_fetcher._FALLBACK_WINDOWS.clear()
        data_fetcher._FALLBACK_UNTIL.clear()
        if hasattr(data_fetcher, "_CYCLE_FALLBACK_FEED"):
            data_fetcher._CYCLE_FALLBACK_FEED.clear()
        cache_attrs = {
            "_BACKUP_SKIP_UNTIL",
            "_BACKUP_PRIMARY_PROBE_AT",
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
            "_BACKUP_USAGE_LOGGED",
            "_FALLBACK_METADATA",
            "_FALLBACK_WINDOWS",
            "_FALLBACK_UNTIL",
            "_daily_memo",
            "_MINUTE_CACHE",
            "_HOST_COUNTS",
            "_HOST_LIMITS",
            "_CYCLE_FALLBACK_FEED",
            "_SIP_UNAVAILABLE_LOGGED",
            "_cycle_feed_override",
            "_override_set_ts",
        }
        for attr in cache_attrs:
            target = getattr(data_fetcher, attr, None)
            if isinstance(target, (dict, set)):
                target.clear()
            elif isinstance(target, list):
                target.clear()
        data_fetcher._DATA_FEED_OVERRIDE = None
        data_fetcher._LAST_OVERRIDE_LOGGED = None
        data_fetcher._SIP_PRECHECK_DONE = False
        if hasattr(data_fetcher, "_reset_state"):
            data_fetcher._reset_state()
        env_override = os.getenv("ENABLE_HTTP_FALLBACK")
        if env_override is None:
            data_fetcher._ENABLE_HTTP_FALLBACK = True
        else:
            data_fetcher._ENABLE_HTTP_FALLBACK = env_override.strip().lower() not in {"0", "false", "no", "off"}
        data_fetcher._max_fallbacks_config = None
        if hasattr(data_fetcher, "_reset_provider_auth_state_for_tests"):
            data_fetcher._reset_provider_auth_state_for_tests()
        elif hasattr(data_fetcher, "_clear_sip_lockout_for_tests"):
            data_fetcher._clear_sip_lockout_for_tests()
        # Ensure no auth/disable lockout leaks between tests.
        if hasattr(data_fetcher, "_clear_sip_lockout_for_tests"):
            data_fetcher._clear_sip_lockout_for_tests()
        if hasattr(data_fetcher, "_alpaca_disabled_until"):
            data_fetcher._alpaca_disabled_until = None
        if hasattr(data_fetcher, "_ALPACA_DISABLED_ALERTED"):
            data_fetcher._ALPACA_DISABLED_ALERTED = False
        if hasattr(data_fetcher, "_alpaca_disable_count"):
            data_fetcher._alpaca_disable_count = 0
        if hasattr(data_fetcher, "_alpaca_empty_streak"):
            data_fetcher._alpaca_empty_streak = 0
        try:
            from ai_trading.data.provider_monitor import provider_monitor

            provider_monitor.reset()
        except Exception:
            pass

    _reset()
    try:
        yield
    finally:
        _reset()


@pytest.fixture(autouse=True)
def _reset_provider_monitor_state():
    """Ensure provider disable state does not leak between tests."""

    from ai_trading.data.provider_monitor import provider_monitor

    provider_monitor.reset()
    try:
        yield
    finally:
        provider_monitor.reset()


@pytest.fixture(autouse=True)
def _reset_bot_engine_state():
    """Clear mutable caches in core bot engine between tests."""

    import ai_trading.core.bot_engine as bot_engine  # local import to avoid circulars
    import time as _time
    from threading import Lock

    def _reset() -> None:
        bot_engine._GLOBAL_CYCLE_ID = None
        bot_engine._GLOBAL_INTRADAY_FALLBACK_FEED = None
        bot_engine._GLOBAL_CYCLE_MINUTE_FEED_OVERRIDE.clear()
        bot_engine._cycle_feature_cache.clear()
        bot_engine._cycle_feature_cache_cycle = None
        bot_engine._cycle_budget_context = None
        bot_engine._EMPTY_TRADE_LOG_INFO_EMITTED = False
        bot_engine._TRADE_LOG_CACHE = None
        bot_engine._TRADE_LOG_CACHE_LOADED = False
        bot_engine._TRADE_LOGGER_SINGLETON = None
        bot_engine._TRADE_LOG_FALLBACK_PATH = None
        bot_engine.TRADE_LOG_FILE = bot_engine.default_trade_log_path()
        if hasattr(bot_engine, "_RUNTIME_READY"):
            bot_engine._RUNTIME_READY = False
        if hasattr(bot_engine, "_HEALTH_CHECK_FAILURES"):
            bot_engine._HEALTH_CHECK_FAILURES = 0
        if hasattr(bot_engine, "trading_client"):
            bot_engine.trading_client = None
        if hasattr(bot_engine, "data_client"):
            bot_engine.data_client = None
        if hasattr(bot_engine, "stream"):
            bot_engine.stream = None
        if hasattr(bot_engine, "time") and hasattr(bot_engine.time, "sleep"):
            bot_engine.time.sleep = _time.sleep
        if hasattr(bot_engine, "_ctx"):
            bot_engine._ctx = None
        lazy_cls = getattr(bot_engine, "LazyBotContext", None)
        if isinstance(lazy_cls, type):
            fresh_ctx = lazy_cls()
            if hasattr(bot_engine, "_global_ctx"):
                bot_engine._global_ctx = fresh_ctx
            if hasattr(bot_engine, "ctx"):
                bot_engine.ctx = fresh_ctx
        else:
            if hasattr(bot_engine, "_global_ctx"):
                bot_engine._global_ctx = None
            if hasattr(bot_engine, "ctx"):
                bot_engine.ctx = None
        if hasattr(bot_engine, "_exec_engine"):
            bot_engine._exec_engine = None
        if hasattr(bot_engine, "run_lock"):
            bot_engine.run_lock = Lock()

    _reset()
    try:
        yield
    finally:
        _reset()


@pytest.fixture(autouse=True)
def _reset_logging_once_state():
    """Reset once/throttle logging caches so caplog assertions are deterministic."""

    import ai_trading.logging as logging_mod
    from ai_trading.logging.emit_once import reset_emit_once_state

    def _reset() -> None:
        try:
            logging_mod.logger_once._emitted_keys.clear()
        except Exception:
            pass
        reset_emit_once_state()
        throttle_filter = getattr(logging_mod, "_THROTTLE_FILTER", None)
        if throttle_filter is not None:
            lock = getattr(throttle_filter, "_lock", None)
            state = getattr(throttle_filter, "_state", None)
            if isinstance(state, dict):
                try:
                    if lock is not None:
                        with lock:
                            state.clear()
                    else:
                        state.clear()
                except Exception:
                    state.clear()
        try:
            logging_mod.reset_provider_log_dedupe()
        except Exception:
            pass

    _reset()
    try:
        yield
    finally:
        _reset()


@pytest.fixture
def dummy_order():
    """Provide a minimal order-like object for tests.

    Ensures required attributes are present and ``filled_qty`` defaults to ``0``
    so that code paths handling numeric comparisons do not encounter ``None``.
    """

    return types.SimpleNamespace(
        id="1",
        status="filled",
        filled_qty=0,
        symbol="TEST",
    )


@pytest.fixture
def dummy_alpaca_client():
    """Provide a callable-recording Alpaca client substitute for unit tests."""

    class _DummyClient:
        def __init__(self):
            self.calls: list[dict[str, object]] = []
            self.last_call: dict[str, object] | None = None

        def submit_order(self, *args, **kwargs):
            call = {"args": args, "kwargs": kwargs}
            self.calls.append(call)
            self.last_call = call
            return {"id": f"dummy-order-{len(self.calls)}"}

    return _DummyClient()
