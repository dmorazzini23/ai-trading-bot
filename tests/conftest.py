"""Test configuration and shared fixtures."""

import os
import tempfile

os.environ.setdefault("PYTEST_RUNNING", "1")
_PYTEST_MODEL_REGISTRY_TMPDIR = tempfile.TemporaryDirectory(
    prefix="ai-trading-pytest-registry-"
)
os.environ["MODEL_REGISTRY_DIR"] = _PYTEST_MODEL_REGISTRY_TMPDIR.name
os.environ["AI_TRADING_HEALTH_REQUIRE_DAY_SLEEVE_MODEL"] = "0"

# Keep every test run out of the production state directories. ``ai_trading.paths``
# resolves LOG_DIR, DATA_DIR, MODELS_DIR, CACHE_DIR and OUTPUT_DIR once at import
# time, and LOG_DIR defaults to /var/log/ai-trading-bot even when no .env is
# loaded, so a per-test monkeypatch is too late: these must be set before any
# ``ai_trading`` import. The per-test ``_guard_production_dirs`` fixture below
# fails any test that still writes into, or opens anything under, the production
# directories.
_PYTEST_DIRS_TMPDIR = tempfile.TemporaryDirectory(prefix="ai-trading-pytest-dirs-")
_PRODUCTION_LOG_DIR = os.environ.get("AI_TRADING_LOG_DIR") or "/var/log/ai-trading-bot"
_PRODUCTION_MODELS_DIR = (
    os.environ.get("AI_TRADING_MODELS_DIR") or "/var/lib/ai-trading-bot/models"
)


def _production_state_roots() -> tuple[str, ...]:
    """Return the production state roots the read guard protects.

    The canonical systemd StateDirectory/LogsDirectory/CacheDirectory roots are
    always protected. Directories inherited from the invoking shell (for example
    after ``set -a; . .env``) are added when they are absolute and do not contain
    the checkout or the temp directory the tests themselves use.
    """

    roots = [
        "/var/lib/ai-trading-bot",
        "/var/log/ai-trading-bot",
        "/var/cache/ai-trading-bot",
    ]
    protected_children = (
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
        os.path.abspath(tempfile.gettempdir()),
    )
    for key in (
        "AI_TRADING_DATA_DIR",
        "AI_TRADING_LOG_DIR",
        "AI_TRADING_MODELS_DIR",
        "AI_TRADING_CACHE_DIR",
        "AI_TRADING_OUTPUT_DIR",
        "AI_TRADING_RUNTIME_ROOT",
        "STATE_DIRECTORY",
    ):
        raw = str(os.environ.get(key) or "").split(":")[0].strip()
        if not raw or not os.path.isabs(raw):
            continue
        candidate = os.path.normpath(raw)
        if candidate == os.sep or any(
            child == candidate or child.startswith(candidate + os.sep)
            for child in protected_children
        ):
            continue
        if candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


_PRODUCTION_STATE_ROOTS = _production_state_roots()

# Drop inherited settings that point at production state (``DATABASE_URL`` for
# the OMS sqlite store, absolute decision-record/fill-event/report paths from a
# sourced .env, ...), so module defaults resolve under the isolated dirs below.
for _env_key, _env_value in list(os.environ.items()):
    if any(_root in _env_value for _root in _PRODUCTION_STATE_ROOTS):
        os.environ.pop(_env_key, None)

# Runtime artifacts whose defaults are relative (``runtime/...``) resolve under
# ``AI_TRADING_DATA_DIR``, which ``_env_defaults`` points at a per-test
# ``tmp_path``. Unset them unconditionally so a developer shell cannot redirect
# the OMS database, decision records, fill events, research reports or
# after-hours state to a shared location.
for _env_key in (
    "DATABASE_URL",
    "STATE_DIRECTORY",
    "AI_TRADING_OMS_INTENT_STORE_PATH",
    "AI_TRADING_OMS_EVENT_JSONL_PATH",
    "AI_TRADING_DECISION_LOG_PATH",
    "AI_TRADING_FILL_EVENTS_PATH",
    "AI_TRADING_TCA_PATH",
    "AI_TRADING_RESEARCH_REPORT_ROOT",
    "AI_TRADING_RESEARCH_DATA_DIR",
    "AI_TRADING_DAILY_RESEARCH_REPORT_PATH",
    "AI_TRADING_AFTER_HOURS_TRAINING_STATE_PATH",
    "AI_TRADING_AFTER_HOURS_TRAINING_MARKER_PATH",
    "AI_TRADING_AFTER_HOURS_REPORT_DIR",
    "AI_TRADING_AFTER_HOURS_REPORT_LATEST_PATH",
    "AI_TRADING_RUNTIME_PERF_REPORT_LATEST_PATH",
    "AI_TRADING_RUNTIME_PERF_FILL_EVENTS_PATH",
    "AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH",
    "AI_TRADING_RUNTIME_PERF_GATE_SUMMARY_PATH",
    "AI_TRADING_RUNTIME_PERF_GATE_LOG_PATH",
    "AI_TRADING_RUNTIME_PERF_EDGE_REALISM_STATE_PATH",
):
    os.environ.pop(_env_key, None)

for _env_key, _relative in (
    ("AI_TRADING_LOG_DIR", "logs"),
    ("AI_TRADING_DATA_DIR", "data"),
    ("AI_TRADING_MODELS_DIR", "models"),
    ("AI_TRADING_CACHE_DIR", "cache"),
    ("AI_TRADING_OUTPUT_DIR", "output"),
    # tools/mcp_* servers and scripts/market_close_recap.py default to
    # /var/lib/ai-trading-bot/runtime when these are unset.
    ("AI_TRADING_RUNTIME_ROOT", "runtime"),
    ("AI_TRADING_RECAP_RUNTIME_DIR", "runtime"),
    # ai_trading.data.fetch_yf defaults to /var/cache/ai-trading-bot/yf.
    ("YF_CACHE_DIR", os.path.join("cache", "yf")),
):
    _isolated_dir = os.path.join(_PYTEST_DIRS_TMPDIR.name, _relative)
    os.makedirs(_isolated_dir, exist_ok=True)
    os.environ[_env_key] = _isolated_dir
os.environ["SLIPPAGE_LOG_PATH"] = os.path.join(_PYTEST_DIRS_TMPDIR.name, "logs", "slippage.csv")
_PYTEST_TRADE_LOG_FILE = os.path.join(_PYTEST_DIRS_TMPDIR.name, "logs", "trades.jsonl")
# File settings whose code default is an absolute production path.
for _env_key, _relative in (
    # ai_trading.telemetry.runtime_prom_metrics, tools/mcp_metrics_query_server.py
    ("AI_TRADING_RUNTIME_DAILY_REPORT_PATH", "daily_performance_report.json"),
    # ai_trading.core.netting_execution_context
    (
        "AI_TRADING_GATE_AUTO_DISABLE_RUNTIME_GONOGO_PATH",
        "runtime_performance_report_latest.json",
    ),
    # tools/mcp_slack_alerts_server.py
    ("AI_TRADING_SLACK_INCIDENT_STATE_PATH", "slack_incident_state.json"),
    ("AI_TRADING_SLACK_EOD_STATE_PATH", "slack_eod_state.json"),
    # tools/mcp_infra_cloud_server.py
    ("AI_TRADING_INFRA_AUDIT_PATH", "infra_actions_audit.jsonl"),
):
    os.environ[_env_key] = os.path.join(_PYTEST_DIRS_TMPDIR.name, "runtime", _relative)

import sys as _sys

import asyncio
import importlib
import random
import socket
import sys
import unittest.mock as _mock
import warnings
from datetime import datetime, timezone
import pathlib
import types
from typing import Any, cast

_STUB_PATH = pathlib.Path(__file__).resolve().parent / "stubs"
_STUB_PATH_STR = str(_STUB_PATH)
_REAL_DATETIME = datetime

if "dotenv" not in sys.modules:
    removed_stub_for_dotenv = False
    if _STUB_PATH_STR in _sys.path:
        _sys.path.remove(_STUB_PATH_STR)
        removed_stub_for_dotenv = True

    try:
        dotenv_mod = importlib.import_module("dotenv")
    except ModuleNotFoundError:
        dotenv_stub = cast(Any, types.ModuleType("dotenv"))
        dotenv_stub.load_dotenv = lambda *args, **kwargs: None
        sys.modules["dotenv"] = dotenv_stub
    else:
        sys.modules["dotenv"] = dotenv_mod
    finally:
        if removed_stub_for_dotenv:
            _sys.path.append(_STUB_PATH_STR)

import pytest

# NOTE: Avoid importing `ai_trading` modules at collection time; lazy import in helpers.
try:
    from urllib3.exceptions import SystemTimeWarning
except Exception:  # pragma: no cover - optional dependency shape
    SystemTimeWarning = None  # type: ignore[assignment]
else:
    warnings.filterwarnings("ignore", category=SystemTimeWarning)


@pytest.fixture(autouse=True)
def _reset_runtime_env_overrides(monkeypatch):
    """Ensure process-local config overrides do not leak across tests."""

    from ai_trading.config.management import clear_runtime_env_overrides

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("AI_TRADING_RUNTIME_ENV_PATH", raising=False)
    clear_runtime_env_overrides()
    yield
    clear_runtime_env_overrides()


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

_SNAPSHOT_MODULES = dict(sys.modules)


def _restore_snapshot_modules() -> None:
    sys.modules.update(
        {name: module for name, module in _SNAPSHOT_MODULES.items() if module is not None}
    )


class _SysModulesPatchContext:
    def __init__(self, context: Any) -> None:
        self._context = context

    def __enter__(self) -> object:
        result = self._context.__enter__()
        _restore_snapshot_modules()
        return result

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        try:
            return self._context.__exit__(exc_type, exc_val, exc_tb)
        finally:
            _restore_snapshot_modules()


def _safe_patch_dict(
    in_dict: Any, values: Any = (), clear: bool = False, **kwargs: Any
):  # pragma: no cover - test helper
    ctx = _ORIGINAL_PATCH_DICT(in_dict, values, clear, **kwargs)
    if clear and in_dict is sys.modules:
        return _SysModulesPatchContext(ctx)
    return ctx


cast(Any, _mock.patch).dict = _safe_patch_dict


if "flask" not in _sys.modules:
    flask_mod = cast(Any, types.ModuleType("flask"))
    flask_app_mod = cast(Any, types.ModuleType("flask.app"))
    request_state = types.SimpleNamespace(_json=None)

    class _StubRequest:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}
            self.method = "GET"
            self.path = ""

        def get_json(self, silent: bool = False):
            payload = request_state._json
            if payload is None and not silent:
                raise RuntimeError("request JSON unavailable")
            return payload

    stub_request = _StubRequest()

    class _StubFlask:
        def __init__(self, *a, **k):
            self._routes = {}
            self.config = {}

        def route(self, path, *a, **k):
            methods = tuple(str(method).upper() for method in (k.get("methods") or ("GET",)))

            def decorator(func):
                for method in methods:
                    self._routes[(path, method)] = func
                if "GET" in methods:
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
                def _request(self, path, method="GET", json=None, headers=None):
                    handler = app._routes[(path, method.upper())]
                    request_state._json = json
                    stub_request.headers = dict(headers or {})
                    stub_request.method = str(method).upper()
                    stub_request.path = str(path)
                    handler_globals = getattr(handler, "__globals__", {})
                    previous_request = handler_globals.get("request")
                    handler_globals["request"] = stub_request
                    try:
                        result = handler()
                    finally:
                        request_state._json = None
                        stub_request.headers = {}
                        stub_request.method = "GET"
                        stub_request.path = ""
                        if previous_request is None:
                            handler_globals.pop("request", None)
                        else:
                            handler_globals["request"] = previous_request
                    status = 200
                    data = result
                    if isinstance(result, tuple):
                        data = result[0]
                        if len(result) > 1:
                            status = result[1]
                    return _Response(data, status)

                def open(self, path, method="GET", json=None, headers=None):
                    return self._request(path, method=method, json=json, headers=headers)

                def get(self, path, headers=None):
                    return self._request(path, method="GET", headers=headers)

                def post(self, path, json=None, headers=None):
                    return self._request(path, method="POST", json=json, headers=headers)

                def delete(self, path, headers=None):
                    return self._request(path, method="DELETE", headers=headers)

            return _Client()

    def _jsonify(payload=None, *args, **kwargs):
        if payload is not None:
            return payload
        if args:
            return args[0] if len(args) == 1 else list(args)
        return kwargs

    flask_mod.Flask = _StubFlask
    flask_mod.jsonify = getattr(flask_mod, "jsonify", _jsonify)
    flask_mod.request = stub_request
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
    if SystemTimeWarning is not None:
        warnings.simplefilter("ignore", SystemTimeWarning, append=False)
    with freeze_time(
        "2024-01-02 15:04:05",
        tz_offset=0,
        ignore=["transformers", "urllib3"],
    ):
        yield


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def guard(*a, **k):
        raise RuntimeError("Network disabled in tests")

    monkeypatch.setattr(socket, "create_connection", guard, raising=True)


def _production_dir_signature() -> dict[str, tuple[int, int]]:
    """Return (mtime_ns, size) for every regular file directly under the production dirs."""

    signature: dict[str, tuple[int, int]] = {}
    for directory in (_PRODUCTION_LOG_DIR, _PRODUCTION_MODELS_DIR):
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in entries:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    stat_result = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                signature[entry.path] = (stat_result.st_mtime_ns, stat_result.st_size)
    return signature


# Audit events whose first argument (or ``src``/``dst`` pair) is a filesystem path.
# ``os.stat``/``Path.exists`` are not audited, so existence probes do not trip the
# read guard; opening, listing, connecting to or mutating a production file does.
_PRODUCTION_ACCESS_EVENTS = frozenset(
    {
        "open",
        "os.listdir",
        "os.scandir",
        "glob.glob",
        "sqlite3.connect",
        "os.mkdir",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.truncate",
        "os.chmod",
        "os.utime",
        "shutil.copyfile",
        "shutil.copytree",
        "shutil.move",
        "shutil.rmtree",
    }
)
_PRODUCTION_ACCESS_STATE: dict[str, Any] = {"active": False, "hits": []}


def _production_path_hit(value: object) -> str | None:
    if isinstance(value, int) or value is None:
        return None
    try:
        raw = os.fsdecode(os.fspath(value))  # type: ignore[arg-type]
    except TypeError:
        return None
    if raw.startswith("file:"):
        raw = raw[len("file:") :].split("?", 1)[0]
    if not raw:
        return None
    resolved = os.path.normpath(raw if os.path.isabs(raw) else os.path.join(os.getcwd(), raw))
    for root in _PRODUCTION_STATE_ROOTS:
        if resolved == root or resolved.startswith(root + os.sep):
            return resolved
    return None


def _production_access_audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if not _PRODUCTION_ACCESS_STATE["active"] or event not in _PRODUCTION_ACCESS_EVENTS:
        return
    candidates = args[:2] if event in {"os.rename", "shutil.copyfile", "shutil.copytree", "shutil.move"} else args[:1]
    for candidate in candidates:
        try:
            hit = _production_path_hit(candidate)
        except Exception:  # noqa: BLE001 - an audit hook must never break the audited call
            hit = None
        if hit is not None:
            _PRODUCTION_ACCESS_STATE["hits"].append((event, hit))


# Audit hooks cannot be removed; it is installed once and only records while a
# test body runs (see ``_guard_production_dirs``).
sys.addaudithook(_production_access_audit_hook)


def _production_guard_enabled() -> bool:
    return str(os.environ.get("AI_TRADING_TEST_PRODUCTION_GUARD", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@pytest.fixture(autouse=True)
def _guard_production_dirs(request: pytest.FixtureRequest):
    """Fail the test that reads or writes production state.

    Historically tests appended fixture rows to the live slippage log and
    truncated the live reward log because the writers resolve their paths at
    import time, and other tests silently read the live runtime artifacts
    (``fill_events.jsonl``, runtime reports) because tool defaults point at
    /var/lib/ai-trading-bot/runtime. The env isolation at the top of this
    module prevents both; this guard catches any path that bypasses it:

    * read guard: a ``sys.addaudithook`` hook records every open/list/connect/
      mutate audit event under ``_PRODUCTION_STATE_ROOTS`` while the test runs;
    * write guard: (mtime, size) of files directly under the production log and
      models directories must not change during the test.

    Set ``AI_TRADING_TEST_PRODUCTION_GUARD=0`` to disable both, for example when
    the live service is known to be writing to those directories during the run.
    """

    if not _production_guard_enabled():
        yield
        return
    before = _production_dir_signature()
    hits: list[tuple[str, str]] = []
    _PRODUCTION_ACCESS_STATE["hits"] = hits
    _PRODUCTION_ACCESS_STATE["active"] = True
    try:
        yield
    finally:
        _PRODUCTION_ACCESS_STATE["active"] = False
        _PRODUCTION_ACCESS_STATE["hits"] = []
    after = _production_dir_signature()
    changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    if changed:
        pytest.fail(
            "test run modified production files (this test, or the live service during "
            f"the test): {changed}. Tests must write only under tmp_path or the isolated "
            "AI_TRADING_*_DIR locations set in tests/conftest.py.",
            pytrace=False,
        )
    if hits:
        accessed = sorted({f"{event} {path}" for event, path in hits})
        shown = accessed[:20]
        more = f" (+{len(accessed) - len(shown)} more)" if len(accessed) > len(shown) else ""
        pytest.fail(
            "test accessed production state under "
            f"{', '.join(_PRODUCTION_STATE_ROOTS)}: {shown}{more}. Give the test explicit "
            "fixture data under tmp_path (pass runtime_root/state_path/report paths or "
            "monkeypatch the path env var) instead of relying on production defaults. "
            "Set AI_TRADING_TEST_PRODUCTION_GUARD=0 to disable this guard.",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def _env_defaults(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    if request.node.get_closest_marker("no_test_credentials"):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    else:
        monkeypatch.setenv("ALPACA_API_KEY", "dummy")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "dummy")
        monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets")
        monkeypatch.setenv("WEBHOOK_SECRET", "dummy-webhook-secret")
        monkeypatch.setenv("DOLLAR_RISK_LIMIT", "0.05")
    runtime_data_dir = tmp_path / "runtime_data"
    runtime_data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(runtime_data_dir))
    monkeypatch.setenv("AI_TRADING_PAPER_SAMPLING_ENABLED", "0")
    monkeypatch.delenv("STATE_DIRECTORY", raising=False)
    # Enforce canonical env contract in tests; legacy aliases must not leak
    # from developer shells or prior tests.
    for key in (
        "TRADING_MODE",
        "TRADING_MODE_PRECEDENCE",
        "TRADING_MODE_ADAPTIVE_ENABLED",
        "CAPITAL_CAP",
        "DAILY_LOSS_LIMIT",
        "MAX_POSITION_SIZE",
        "AI_TRADING_MAX_POSITION_SIZE",
        "TAKE_PROFIT_FACTOR",
        "BUY_THRESHOLD",
        "MIN_CONFIDENCE",
        "CONF_THRESHOLD",
        "KELLY_FRACTION",
        "KELLY_FRACTION_MAX",
        "SIGNAL_CONFIRMATION_BARS",
    ):
        monkeypatch.delenv(key, raising=False)


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
            (
                "sklearn.linear_model",
                "sklearn.metrics",
                "sklearn.pipeline",
                "sklearn.preprocessing",
            ),
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
            self._df = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})

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
            "_GLOBAL_BACKUP_SKIP_UNTIL",
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
            data_fetcher._ENABLE_HTTP_FALLBACK = env_override.strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
        setattr(data_fetcher, "_max_fallbacks_config", None)
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
        bot_engine._CYCLE_MINUTE_DF_CACHE.clear()
        bot_engine._CYCLE_MINUTE_DF_CACHE_CYCLE = None
        bot_engine._cycle_feature_cache.clear()
        bot_engine._cycle_feature_cache_cycle = None
        bot_engine._cycle_budget_context = None
        bot_engine._EMPTY_TRADE_LOG_INFO_EMITTED = False
        bot_engine._TRADE_LOG_CACHE = None
        bot_engine._TRADE_LOG_CACHE_LOADED = False
        bot_engine._TRADE_LOGGER_SINGLETON = None
        bot_engine._TRADE_LOG_FALLBACK_PATH = None
        # Do not call ``default_trade_log_path()`` here: it creates ./logs/trades.jsonl
        # in the checkout and, when a test has made cwd/logs unwritable, falls back to
        # the production /var/log/ai-trading-bot directory. Tests that exercise path
        # resolution call it explicitly.
        bot_engine.TRADE_LOG_FILE = _PYTEST_TRADE_LOG_FILE
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
        if hasattr(bot_engine, "datetime"):
            cast(Any, bot_engine).datetime = _REAL_DATETIME
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

        def submit_order(self, *args: Any, **kwargs: Any):
            call: dict[str, object] = {"args": args, "kwargs": kwargs}
            self.calls.append(call)
            self.last_call = call
            return {"id": f"dummy-order-{len(self.calls)}"}

    return _DummyClient()


# Test modules that set os.environ at import time leak those values into every
# test collected after them (e.g. TESTING=1 forcing primary minute fetches).
# Record each module's import-time env delta, undo it after collection, and
# re-apply it only to that module's own tests via monkeypatch.
_MODULE_IMPORT_ENV: dict[str, tuple[dict[str, str], frozenset[str]]] = {}


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector: pytest.Collector):
    if not isinstance(collector, pytest.Module):
        yield
        return
    before = dict(os.environ)
    yield
    after = dict(os.environ)
    changed = {key: value for key, value in after.items() if before.get(key) != value}
    removed = frozenset(key for key in before if key not in after)
    if not changed and not removed:
        return
    _MODULE_IMPORT_ENV[str(collector.path)] = (changed, removed)
    for key in changed:
        if key in before:
            os.environ[key] = before[key]
        else:
            os.environ.pop(key, None)
    for key in removed:
        os.environ[key] = before[key]


@pytest.fixture(autouse=True)
def _module_import_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    delta = _MODULE_IMPORT_ENV.get(str(request.node.path))
    if delta is not None:
        changed, removed = delta
        for key, value in changed.items():
            monkeypatch.setenv(key, value)
        for key in removed:
            monkeypatch.delenv(key, raising=False)
    yield


_CONFIG_PACKAGE_KEY = pytest.StashKey[Any]()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item):
    item.stash[_CONFIG_PACKAGE_KEY] = sys.modules.get("ai_trading.config")
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item):
    """Error the test that leaves ``ai_trading.config`` replaced or incomplete.

    Popping the package from ``sys.modules`` while its submodules stay cached
    makes the next import build a fresh package object that lacks attributes
    such as ``settings``, breaking later ``monkeypatch.setattr`` dotted paths.
    Runs after fixture teardown so monkeypatch-restored entries are not flagged,
    and restores the package so only the offending test errors.
    """

    result = yield
    original = item.stash.get(_CONFIG_PACKAGE_KEY, None)
    package = sys.modules.get("ai_trading.config")
    problems: list[str] = []
    if original is not None and package is not original:
        problems.append("replaced or removed the ai_trading.config package")
        sys.modules["ai_trading.config"] = original
        parent = sys.modules.get("ai_trading")
        if parent is not None:
            setattr(parent, "config", original)
        package = original
    if package is not None:
        prefix = "ai_trading.config."
        missing = {
            name[len(prefix) :]: module
            for name, module in list(sys.modules.items())
            if module is not None
            and name.startswith(prefix)
            and "." not in name[len(prefix) :]
            and not hasattr(package, name[len(prefix) :])
        }
        if missing:
            problems.append(f"left ai_trading.config without submodule attributes {sorted(missing)}")
            for attr, module in missing.items():
                setattr(package, attr, module)
    if problems:
        pytest.fail(
            f"{item.nodeid} {'; '.join(problems)}; restore sys.modules instead of "
            "popping the package",
            pytrace=False,
        )
    return result
