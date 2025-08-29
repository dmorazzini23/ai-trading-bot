from __future__ import annotations

from pathlib import Path

import pytest
import types
import sys
import importlib.machinery

from ai_trading.data import universe

validation_stub = types.ModuleType("ai_trading.validation")
require_env_stub = types.ModuleType("ai_trading.validation.require_env")
flask_stub = types.ModuleType("flask")
class Flask:  # minimal stub
    def __init__(self, *args, **kwargs):  # noqa: D401, ANN001, ANN002
        pass

    def route(self, *args, **kwargs):  # noqa: D401, ANN001, ANN002
        def decorator(func):
            return func

        return decorator

flask_stub.Flask = Flask
sys.modules.setdefault("flask", flask_stub)

bs4_stub = types.ModuleType("bs4")


class BeautifulSoup:  # minimal stub
    pass


bs4_stub.BeautifulSoup = BeautifulSoup
sys.modules.setdefault("bs4", bs4_stub)
sklearn_stub = types.ModuleType("sklearn")
ensemble_stub = types.ModuleType("sklearn.ensemble")
metrics_stub = types.ModuleType("sklearn.metrics")
model_selection_stub = types.ModuleType("sklearn.model_selection")
preprocessing_stub = types.ModuleType("sklearn.preprocessing")
sklearn_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
ensemble_stub.__spec__ = importlib.machinery.ModuleSpec(
    "sklearn.ensemble", loader=None
)

class _Dummy:
    pass

ensemble_stub.GradientBoostingClassifier = _Dummy
ensemble_stub.RandomForestClassifier = _Dummy
sklearn_stub.ensemble = ensemble_stub
metrics_stub.accuracy_score = _Dummy
sklearn_stub.metrics = metrics_stub
model_selection_stub.train_test_split = _Dummy
sklearn_stub.model_selection = model_selection_stub
preprocessing_stub.StandardScaler = _Dummy
sklearn_stub.preprocessing = preprocessing_stub
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.ensemble", ensemble_stub)
sys.modules.setdefault("sklearn.metrics", metrics_stub)
sys.modules.setdefault("sklearn.model_selection", model_selection_stub)
sys.modules.setdefault("sklearn.preprocessing", preprocessing_stub)


def _require_env_vars(*_a, **_k):
    return None


def require_env_vars(*_a, **_k):  # noqa: D401
    return True


def should_halt_trading(*_a, **_k):
    return False


require_env_stub._require_env_vars = _require_env_vars
require_env_stub.require_env_vars = require_env_vars
require_env_stub.should_halt_trading = should_halt_trading
validation_stub.require_env = require_env_stub
validation_stub._require_env_vars = _require_env_vars
validation_stub.require_env_vars = require_env_vars
validation_stub.should_halt_trading = should_halt_trading
sys.modules.setdefault("ai_trading.validation", validation_stub)
sys.modules.setdefault("ai_trading.validation.require_env", require_env_stub)

metrics_stub = types.ModuleType("ai_trading.metrics")


class _Metric:  # minimal stub classes
    def __init__(self, *a, **k):
        pass


metrics_stub.Counter = _Metric
metrics_stub.Gauge = _Metric
metrics_stub.Histogram = _Metric
metrics_stub.Summary = _Metric
metrics_stub.CollectorRegistry = _Metric
metrics_stub.REGISTRY = _Metric()
metrics_stub.PROMETHEUS_AVAILABLE = False
metrics_stub.start_http_server = lambda *a, **k: None
sys.modules.setdefault("ai_trading.metrics", metrics_stub)

from ai_trading.core import bot_engine


def test_env_override_path_preferred(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    """Env var should override packaged CSV."""
    # AI-AGENT-REF: verify env override path
    csv_path = tmp_path / "tickers.csv"
    csv_path.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")

    monkeypatch.setenv("AI_TRADING_TICKERS_CSV", str(csv_path))
    try:
        path = universe.locate_tickers_csv()
        assert path == str(csv_path.resolve())
        symbols = universe.load_universe()
        assert symbols == ["AAPL", "MSFT"]
    finally:
        monkeypatch.delenv("AI_TRADING_TICKERS_CSV", raising=False)


def test_packaged_loads_packaged_csv():
    """Uses packaged CSV when env not set."""  # AI-AGENT-REF: ensure packaged loader
    path = universe.locate_tickers_csv()
    assert path is not None and path.endswith("ai_trading/data/tickers.csv")
    symbols = universe.load_universe()
    assert isinstance(symbols, list) and len(symbols) > 0


def test_missing_package_raises_runtime_error_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Missing package should log and raise error."""  # AI-AGENT-REF: test missing pkg case

    def boom(_name: str):
        raise ModuleNotFoundError("ai_trading.data not importable")

    monkeypatch.setattr(universe, "pkg_files", boom, raising=True)
    monkeypatch.delenv("AI_TRADING_TICKERS_CSV", raising=False)
    monkeypatch.chdir(tmp_path)

    called: list[tuple[str, dict]] = []

    def fake_error(msg, *, extra):
        called.append((msg, extra))

    monkeypatch.setattr(universe.logger, "error", fake_error)
    with pytest.raises(RuntimeError):
        universe.load_universe()
    assert called and called[0][0] == "TICKERS_FILE_MISSING"


def test_malformed_empty_csv_logs_and_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Malformed/empty CSV should log read failure."""  # AI-AGENT-REF: test read guard

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    monkeypatch.setenv("AI_TRADING_TICKERS_CSV", str(empty_csv))

    called: list[tuple[str, dict]] = []

    def fake_error(msg, *, extra):
        called.append((msg, extra))

    monkeypatch.setattr(universe.logger, "error", fake_error)

    syms = universe.load_universe()
    assert syms == []
    assert called and called[0][0] == "TICKERS_FILE_READ_FAILED"


def test_brk_dot_b_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """BRK.B should be normalized to BRK-B for Yahoo Finance."""

    csv_path = tmp_path / "tickers.csv"
    csv_path.write_text("symbol\nBRK.B\n", encoding="utf-8")
    monkeypatch.setenv("AI_TRADING_TICKERS_CSV", str(csv_path))
    try:
        symbols = universe.load_universe()
    finally:
        monkeypatch.delenv("AI_TRADING_TICKERS_CSV", raising=False)
    assert symbols == ["BRK-B"]


def test_screen_candidates_empty_watchlist_returns_fallback():
    """screen_candidates returns fallback symbols when watchlist is empty."""
    runtime = types.SimpleNamespace()
    assert bot_engine.screen_candidates(runtime, []) == bot_engine.FALLBACK_SYMBOLS


def test_load_candidate_universe_loads_when_none(monkeypatch: pytest.MonkeyPatch):
    """load_candidate_universe loads tickers when input is None."""

    runtime = types.SimpleNamespace()

    called: list[bool] = []

    def fake_load() -> list[str]:
        called.append(True)
        return ["AAPL"]

    monkeypatch.setattr(bot_engine, "load_tickers", fake_load)

    symbols = bot_engine.load_candidate_universe(runtime)

    assert symbols == ["AAPL"]
    assert runtime.tickers == ["AAPL"]
    assert called


def test_load_candidate_universe_loads_when_empty_list(monkeypatch: pytest.MonkeyPatch):
    """load_candidate_universe loads tickers when input list is empty."""

    runtime = types.SimpleNamespace()

    called: list[bool] = []

    def fake_load() -> list[str]:
        called.append(True)
        return ["MSFT"]

    monkeypatch.setattr(bot_engine, "load_tickers", fake_load)

    symbols = bot_engine.load_candidate_universe(runtime, [])

    assert symbols == ["MSFT"]
    assert runtime.tickers == ["MSFT"]
    assert called


def test_load_candidate_universe_raises_when_csv_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    """load_candidate_universe propagates missing CSV errors."""

    def boom(path: str = bot_engine.TICKERS_FILE) -> list[str]:  # noqa: ARG001
        raise RuntimeError("tickers missing")

    monkeypatch.setattr(bot_engine, "load_tickers", boom)
    runtime = types.SimpleNamespace()
    with pytest.raises(RuntimeError):
        bot_engine.load_candidate_universe(runtime)

