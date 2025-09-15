from __future__ import annotations

import logging
from types import ModuleType, SimpleNamespace

import sys

import pytest

# Provide a lightweight numpy stub so the heavy dependency is optional for these tests.
if "numpy" not in sys.modules:  # pragma: no cover - import guard
    numpy_stub = ModuleType("numpy")

    def _stub_array(*args, **kwargs):
        return []

    def _stub_bool_array(arr):
        try:
            length = len(arr)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - defensive
            length = 0
        return [False] * int(length)

    numpy_stub.ndarray = list  # type: ignore[attr-defined]
    numpy_stub.array = _stub_array  # type: ignore[attr-defined]
    numpy_stub.asarray = lambda arr, dtype=None: list(arr) if hasattr(arr, "__iter__") else []  # type: ignore[attr-defined]
    numpy_stub.diff = _stub_array  # type: ignore[attr-defined]
    numpy_stub.where = lambda condition, x=None, y=None: []  # type: ignore[attr-defined]
    numpy_stub.zeros_like = _stub_array  # type: ignore[attr-defined]
    numpy_stub.isnan = _stub_bool_array  # type: ignore[attr-defined]
    numpy_stub.float64 = float  # type: ignore[attr-defined]
    numpy_stub.nan = float("nan")  # type: ignore[attr-defined]
    numpy_stub.NaN = numpy_stub.nan  # type: ignore[attr-defined]
    numpy_stub.inf = float("inf")  # type: ignore[attr-defined]
    numpy_stub.random = SimpleNamespace(  # type: ignore[attr-defined]
        seed=lambda *_args, **_kwargs: None,
        normal=_stub_array,
    )
    sys.modules["numpy"] = numpy_stub

if "portalocker" not in sys.modules:  # pragma: no cover - import guard
    sys.modules["portalocker"] = ModuleType("portalocker")

if "bs4" not in sys.modules:  # pragma: no cover - import guard
    bs4_stub = ModuleType("bs4")
    bs4_stub.BeautifulSoup = object  # type: ignore[attr-defined]
    sys.modules["bs4"] = bs4_stub

if "flask" not in sys.modules:  # pragma: no cover - import guard
    flask_stub = ModuleType("flask")

    class _Flask:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs) -> None:
            self.blueprints = []

        def route(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def register_blueprint(self, blueprint, *args, **kwargs):
            self.blueprints.append(blueprint)

    flask_stub.Flask = _Flask  # type: ignore[attr-defined]
    flask_stub.jsonify = lambda *args, **kwargs: {}  # type: ignore[attr-defined]
    sys.modules["flask"] = flask_stub

if "requests" not in sys.modules:  # pragma: no cover - import guard
    requests_stub = ModuleType("requests")
    requests_stub.get = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    requests_stub.post = requests_stub.get  # type: ignore[attr-defined]
    requests_stub.Session = SimpleNamespace  # type: ignore[attr-defined]
    requests_stub.exceptions = SimpleNamespace(  # type: ignore[attr-defined]
        RequestException=Exception,
        Timeout=Exception,
        ConnectionError=Exception,
        HTTPError=Exception,
    )
    sys.modules["requests"] = requests_stub

from ai_trading.core import bot_engine


class _EmptyDailyFetcher:
    def __init__(self) -> None:
        self.calls = 0

    def get_daily_df(self, ctx: object, symbol: str) -> None:
        self.calls += 1
        return None


class _UnavailableFetcher:
    def __init__(self) -> None:
        self.calls = 0

    def get_daily_df(self, ctx: object, symbol: str) -> None:
        self.calls += 1
        raise bot_engine.DataFetchError("DATA_FETCHER_UNAVAILABLE")


def _reset_vol_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("mean", "std", "last", "last_update"):
        monkeypatch.setitem(bot_engine._VOL_STATS, key, None)


def _make_runtime(fetcher) -> SimpleNamespace:
    return SimpleNamespace(data_fetcher=fetcher, halt_manager=None)


def test_spy_vol_fetch_aborts_on_empty_response(monkeypatch: pytest.MonkeyPatch, caplog):
    _reset_vol_stats(monkeypatch)
    fetcher = _EmptyDailyFetcher()
    runtime = _make_runtime(fetcher)
    caplog.set_level(logging.WARNING)

    with pytest.raises(bot_engine.DataFetchError):
        bot_engine.compute_spy_vol_stats(runtime)

    assert fetcher.calls == 1
    abort_logs = [rec for rec in caplog.records if rec.message == "SPY_VOL_FETCH_ABORT"]
    assert abort_logs, "expected abort warning for empty daily data"
    assert (
        getattr(abort_logs[0], "hint", "")
        == "historical data not available; manual backfill required"
    )


def test_spy_vol_fetch_abort_on_data_fetcher_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    _reset_vol_stats(monkeypatch)
    fetcher = _UnavailableFetcher()
    runtime = _make_runtime(fetcher)
    caplog.set_level(logging.WARNING)

    with pytest.raises(bot_engine.DataFetchError):
        bot_engine.compute_spy_vol_stats(runtime)

    assert fetcher.calls == 1
    abort_logs = [rec for rec in caplog.records if rec.message == "SPY_VOL_FETCH_ABORT"]
    assert abort_logs, "expected abort warning when data fetcher is unavailable"
    assert (
        getattr(abort_logs[0], "hint", "")
        == "historical data not available; manual backfill required"
    )
