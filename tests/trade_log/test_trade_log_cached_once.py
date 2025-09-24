"""Trade log boot cache is loaded exactly once per process."""

from __future__ import annotations

import sys
import types

if "ai_trading.indicators" not in sys.modules:
    indicators_stub = types.ModuleType("ai_trading.indicators")

    def _unavailable_indicator(*_args, **_kwargs):  # pragma: no cover - safety stub
        raise RuntimeError("Indicator module unavailable in tests")

    indicators_stub.compute_atr = _unavailable_indicator
    indicators_stub.atr = _unavailable_indicator
    indicators_stub.mean_reversion_zscore = _unavailable_indicator
    indicators_stub.rsi = _unavailable_indicator
    sys.modules["ai_trading.indicators"] = indicators_stub

if "ai_trading.signals" not in sys.modules:
    signals_stub = types.ModuleType("ai_trading.signals")
    signals_indicators_stub = types.ModuleType("ai_trading.signals.indicators")

    def _composite_confidence_stub(*_args, **_kwargs):  # pragma: no cover - safety stub
        return {}

    signals_indicators_stub.composite_signal_confidence = _composite_confidence_stub
    sys.modules["ai_trading.signals"] = signals_stub
    sys.modules["ai_trading.signals.indicators"] = signals_indicators_stub
    signals_stub.indicators = signals_indicators_stub

if "ai_trading.features" not in sys.modules:
    features_stub = types.ModuleType("ai_trading.features")
    features_indicators_stub = types.ModuleType("ai_trading.features.indicators")

    def _feature_passthrough(df, **_kwargs):  # pragma: no cover - safety stub
        return df

    features_indicators_stub.compute_macd = _feature_passthrough
    features_indicators_stub.compute_macds = _feature_passthrough
    features_indicators_stub.compute_vwap = _feature_passthrough
    features_indicators_stub.compute_atr = _feature_passthrough
    features_indicators_stub.compute_sma = _feature_passthrough
    features_indicators_stub.ensure_columns = _feature_passthrough
    sys.modules["ai_trading.features"] = features_stub
    sys.modules["ai_trading.features.indicators"] = features_indicators_stub
    features_stub.indicators = features_indicators_stub

if "portalocker" not in sys.modules:
    portalocker_stub = types.ModuleType("portalocker")
    portalocker_stub.LOCK_EX = 1

    def _noop_lock(*_args, **_kwargs):  # pragma: no cover - safety stub
        return None

    portalocker_stub.lock = _noop_lock
    portalocker_stub.unlock = _noop_lock
    sys.modules["portalocker"] = portalocker_stub

if "bs4" not in sys.modules:
    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:  # pragma: no cover - safety stub
        def __init__(self, *_args, **_kwargs):
            self.text = ""

        def find(self, *_args, **_kwargs):
            return None

    bs4_stub.BeautifulSoup = _BeautifulSoup
    sys.modules["bs4"] = bs4_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = dotenv_stub

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

from ai_trading.core import bot_engine as be


def test_load_trade_log_cache_only_hits_disk_once(monkeypatch):
    call_count = {"value": 0}

    def fake_read(path, **kwargs):  # noqa: ARG001 - signature mirrors real helper
        call_count["value"] += 1
        return {"rows": 1}

    monkeypatch.setattr(be, "_read_trade_log", fake_read)
    monkeypatch.setattr(be, "TRADE_LOG_FILE", "trades.csv")
    be._TRADE_LOG_CACHE = None
    be._TRADE_LOG_CACHE_LOADED = False

    first = be._load_trade_log_cache()
    second = be._load_trade_log_cache()

    assert first is second
    assert call_count["value"] == 1
