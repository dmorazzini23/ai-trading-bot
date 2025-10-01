"""Regression tests for run_all_trades_worker throttle handling."""

from __future__ import annotations
from datetime import UTC, datetime
import sys
import types

if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.array = lambda *a, **k: a
    numpy_stub.ndarray = object
    numpy_stub.float64 = float
    numpy_stub.int64 = int
    numpy_stub.nan = float("nan")
    numpy_stub.NaN = float("nan")
    numpy_stub.random = types.SimpleNamespace(seed=lambda *_a, **_k: None)
    sys.modules["numpy"] = numpy_stub

if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")

    class _DataFrame(dict):  # pragma: no cover - minimal stub
        empty = False

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.columns = []

        def rename(self, *args, **kwargs):  # noqa: D401
            return self

        def reset_index(self, *args, **kwargs):  # noqa: D401
            return self

        def loc(self, *args, **kwargs):  # noqa: D401
            return self

    pandas_stub.DataFrame = _DataFrame
    pandas_stub.Timestamp = datetime
    pandas_stub.Series = dict
    pandas_stub.Index = list
    pandas_stub.MultiIndex = list
    pandas_stub.NaT = None
    pandas_stub.isna = lambda *_a, **_k: False
    pandas_stub.to_datetime = lambda *a, **k: datetime.now(UTC)
    pandas_stub.date_range = lambda *a, **k: []
    pandas_stub.concat = lambda frames, *a, **k: frames[0] if frames else _DataFrame()
    sys.modules["pandas"] = pandas_stub

if "sklearn" not in sys.modules:
    sklearn_stub = types.ModuleType("sklearn")
    ensemble_stub = types.ModuleType("sklearn.ensemble")
    metrics_stub = types.ModuleType("sklearn.metrics")
    model_selection_stub = types.ModuleType("sklearn.model_selection")
    preprocessing_stub = types.ModuleType("sklearn.preprocessing")

    class _GB:  # noqa: D401 - placeholder
        pass

    class _RF:  # noqa: D401 - placeholder
        pass

    ensemble_stub.GradientBoostingClassifier = _GB
    ensemble_stub.RandomForestClassifier = _RF
    metrics_stub.accuracy_score = lambda *_a, **_k: 0.0
    model_selection_stub.train_test_split = lambda *_a, **_k: ([], [])
    preprocessing_stub.StandardScaler = type("StandardScaler", (), {})

    sys.modules.setdefault("sklearn", sklearn_stub)
    sys.modules.setdefault("sklearn.ensemble", ensemble_stub)
    sys.modules.setdefault("sklearn.metrics", metrics_stub)
    sys.modules.setdefault("sklearn.model_selection", model_selection_stub)
    sys.modules.setdefault("sklearn.preprocessing", preprocessing_stub)

if "portalocker" not in sys.modules:
    portalocker_stub = types.ModuleType("portalocker")
    portalocker_stub.LOCK_EX = 1
    portalocker_stub.lock = lambda *_a, **_k: None
    portalocker_stub.unlock = lambda *_a, **_k: None
    sys.modules["portalocker"] = portalocker_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *_a, **_k: None
    sys.modules["dotenv"] = dotenv_stub

if "bs4" not in sys.modules:
    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:  # pragma: no cover - minimal stub
        def __init__(self, *_a, **_k):
            self.text = ""

        def find(self, *_a, **_k):
            return None

    bs4_stub.BeautifulSoup = _BeautifulSoup
    sys.modules["bs4"] = bs4_stub

import ai_trading.core.bot_engine as be


class _FixedDateTime(datetime):
    """Return a deterministic timestamp for ``datetime.now``."""

    @classmethod
    def now(cls, tz=None):  # noqa: D401 - match datetime API
        base = datetime(2024, 1, 3, 14, 30, tzinfo=UTC)
        if tz is None:
            return base.replace(tzinfo=None)
        return base.astimezone(tz)


def _stub_runtime(monkeypatch):
    """Create a minimal runtime object for run_all_trades_worker tests."""

    api = types.SimpleNamespace(
        get_account=lambda: types.SimpleNamespace(
            cash=0.0, equity=0.0, last_equity=0.0
        ),
        list_positions=lambda: [],
    )
    runtime = types.SimpleNamespace(
        api=api,
        risk_engine=types.SimpleNamespace(
            wait_for_exposure_update=lambda _timeout: None,
            refresh_positions=lambda _api: None,
            _adaptive_global_cap=lambda: 0.0,
        ),
        execution_engine=None,
        drawdown_circuit_breaker=None,
        signal_manager=types.SimpleNamespace(begin_cycle=lambda: None),
        portfolio_weights={},
        data_fetcher=types.SimpleNamespace(_minute_timestamps={}),
        model=None,
    )

    monkeypatch.setattr(be, "safe_alpaca_get_account", lambda _rt: api.get_account())
    return runtime


def _patch_minimal_runtime(monkeypatch) -> None:
    """Stub out heavy runtime dependencies for the trading loop."""

    monkeypatch.setattr(be, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(be, "_init_metrics", lambda: None)
    monkeypatch.setattr(be, "_ensure_execution_engine", lambda _rt: None)
    monkeypatch.setattr(be, "ensure_alpaca_attached", lambda _rt: None)
    monkeypatch.setattr(be, "ensure_data_fetcher", lambda _rt: None)
    monkeypatch.setattr(be, "get_trade_logger", lambda: None)
    monkeypatch.setattr(be, "get_strategies", lambda: [])
    monkeypatch.setattr(be, "is_market_open", lambda: True)
    monkeypatch.setattr(be, "check_pdt_rule", lambda _rt: False)
    monkeypatch.setattr(be, "get_verbose_logging", lambda: False)
    monkeypatch.setattr(be, "_validate_trading_api", lambda _api: True)
    monkeypatch.setattr(be.CFG, "log_market_fetch", False, raising=False)
    monkeypatch.setattr(be, "list_open_orders", lambda _api: [])
    monkeypatch.setattr(be, "_handle_pending_orders", lambda *_a, **_k: False)
    monkeypatch.setattr(be, "signal_manager", types.SimpleNamespace(begin_cycle=lambda: None))
    monkeypatch.setattr(be, "run_multi_strategy", lambda _rt: None)
    monkeypatch.setattr(be, "check_halt_flag", lambda _rt: False)
    monkeypatch.setattr(be, "_log_loop_heartbeat", lambda *_a, **_k: None)
    monkeypatch.setattr(be, "_check_runtime_stops", lambda _rt: None)
    monkeypatch.setattr(be, "monotonic_time", lambda: 0.0)
    monkeypatch.setattr(be.time, "sleep", lambda *_a, **_k: None)

    class _DummyLock:
        _locked = False

        def acquire(self, blocking: bool = False) -> bool:  # noqa: D401 - lock API
            if self._locked:
                return False
            self._locked = True
            return True

        def release(self) -> None:  # noqa: D401 - lock API
            self._locked = False

    monkeypatch.setattr(be, "run_lock", _DummyLock())


def test_missing_columns_failure_does_not_throttle(monkeypatch, caplog):
    """A MissingOHLCVColumnsError should not update the throttle timestamp."""

    _patch_minimal_runtime(monkeypatch)
    monkeypatch.setattr(be, "datetime", _FixedDateTime)

    runtime = _stub_runtime(monkeypatch)
    state = be.BotState()

    attempt_counter = {"calls": 0}
    missing_error = be.data_fetcher_module.MissingOHLCVColumnsError
    monkeypatch.setattr(be, "DataFetchError", missing_error)

    def _prepare_stub(_rt, _state, _tickers=None):  # noqa: D401 - match contract
        attempt_counter["calls"] += 1
        if attempt_counter["calls"] <= 3:
            raise missing_error("missing")
        return (0.0, True, [])

    monkeypatch.setattr(be, "_prepare_run", _prepare_stub)

    caplog.set_level("WARNING")

    be.run_all_trades_worker(state, runtime)

    assert attempt_counter["calls"] == 3
    assert state.last_run_at is None

    be.run_all_trades_worker(state, runtime)

    assert attempt_counter["calls"] == 4
    assert not any(
        record.message == "RUN_ALL_TRADES_SKIPPED_RECENT" for record in caplog.records
    )
