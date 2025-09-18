"""Tests for :func:`ai_trading.core.bot_engine.get_latest_price` fallbacks."""

from __future__ import annotations

import sys
import types

try:  # pragma: no cover - optional dependency in test env
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - provide minimal stub
    pd = None

    class _FakeILoc:
        def __init__(self, value: float) -> None:
            self._value = value

        def __getitem__(self, idx: int) -> float:
            if idx == -1:
                return self._value
            raise IndexError(idx)

    class _FakeSeries:
        def __init__(self, value: float) -> None:
            self._value = value
            self.iloc = _FakeILoc(value)

    class _FakeDataFrame:
        def __init__(self, value: float) -> None:
            self._value = value
            self.empty = False

        def __getitem__(self, key: str) -> _FakeSeries:
            if key != "close":
                raise KeyError(key)
            return _FakeSeries(self._value)


def _df(price: float):
    if pd is not None:
        return pd.DataFrame({"close": [price]})
    return _FakeDataFrame(price)


if "numpy" not in sys.modules:  # pragma: no cover - optional dependency stub
    class _NumpyStub(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("numpy")
            self.ndarray = object
            self.nan = float("nan")
            self.NAN = self.nan
            self.float64 = float
            self.random = types.SimpleNamespace(seed=lambda *_, **__: None)
            self.isscalar = lambda _v: True
            self.bool_ = bool

        def __getattr__(self, name: str):  # noqa: D401 - stub fallback
            def _stub(*args, **kwargs):  # noqa: ANN001, ANN002
                raise NotImplementedError(f"numpy stub invoked for {name}")

            return _stub

    sys.modules["numpy"] = _NumpyStub()


if "portalocker" not in sys.modules:  # pragma: no cover - optional dependency stub
    portalocker_stub = types.ModuleType("portalocker")
    portalocker_stub.LOCK_EX = 1
    portalocker_stub.LockException = RuntimeError
    portalocker_stub.AlreadyLocked = RuntimeError

    def _noop_lock(*args, **kwargs):  # noqa: D401 - stub
        return None

    portalocker_stub.lock = _noop_lock
    portalocker_stub.unlock = _noop_lock
    sys.modules["portalocker"] = portalocker_stub


if "bs4" not in sys.modules:  # pragma: no cover - optional dependency stub
    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:  # noqa: D401 - minimal stub
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - stub
            raise NotImplementedError("BeautifulSoup stub invoked")

    bs4_stub.BeautifulSoup = _BeautifulSoup
    sys.modules["bs4"] = bs4_stub


import ai_trading.alpaca_api as alpaca_api
from ai_trading.alpaca_api import AlpacaAuthenticationError, is_alpaca_service_available
from ai_trading.core import bot_engine
import ai_trading.data.fetch as data_fetcher


def test_get_latest_price_uses_yahoo_when_alpaca_none(monkeypatch):
    """Alpaca returning ``None`` should trigger Yahoo fallback."""

    monkeypatch.setattr(
        bot_engine,
        "_alpaca_symbols",
        lambda: (lambda *_a, **_k: {"ap": None}, None),
    )

    called: dict[str, bool] = {"yahoo": False}

    def fake_yahoo(symbol, start, end, interval):  # noqa: ARG001
        called["yahoo"] = True
        return _df(101.0)

    monkeypatch.setattr(data_fetcher, "_backup_get_bars", fake_yahoo)
    monkeypatch.setattr(bot_engine, "get_latest_close", lambda df: float(df["close"].iloc[-1]))

    price = bot_engine.get_latest_price("AAPL")

    assert called["yahoo"]
    assert price == 101.0


def test_get_latest_price_uses_alpaca_bid_when_ask_invalid(monkeypatch):
    """Positive bid/last data should prevent Yahoo fallback when ask is invalid."""

    monkeypatch.setattr(bot_engine, "_PRICE_SOURCE", {})
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.is_alpaca_service_available",
        lambda: True,
    )

    def fake_alpaca_get(*_a, **_k):
        return {
            "ap": 0.0,
            "bp": 94.5,
            "last": {"price": 95.1},
            "midpoint": 94.8,
        }

    monkeypatch.setattr(bot_engine, "_alpaca_symbols", lambda: (fake_alpaca_get, None))

    def yahoo_fail(*_a, **_k):  # pragma: no cover - defensive
        raise AssertionError("Yahoo fallback should not run when Alpaca provides prices")

    monkeypatch.setattr(data_fetcher, "_backup_get_bars", yahoo_fail)

    price = bot_engine.get_latest_price("AAPL")

    assert price == 94.5
    assert bot_engine._PRICE_SOURCE["AAPL"] == "alpaca_bid"


def test_get_latest_price_uses_latest_close_when_providers_fail(monkeypatch):
    """If Alpaca and Yahoo fail, fall back to ``get_latest_close`` from bars."""

    monkeypatch.setattr(
        bot_engine,
        "_alpaca_symbols",
        lambda: (lambda *_a, **_k: {"ap": None}, None),
    )
    monkeypatch.setattr(data_fetcher, "_backup_get_bars", lambda *a, **k: (_ for _ in ()).throw(RuntimeError))

    monkeypatch.setattr(bot_engine, "get_bars_df", lambda symbol: _df(55.0))
    monkeypatch.setattr(bot_engine, "get_latest_close", lambda df: float(df["close"].iloc[-1]) if not df.empty else None)

    price = bot_engine.get_latest_price("AAPL")

    assert price == 55.0


def test_get_latest_price_skips_non_positive_from_yahoo(monkeypatch):
    """Zeroes from Yahoo should be treated as invalid and fall back further."""

    monkeypatch.setattr(bot_engine, "_PRICE_SOURCE", {})

    monkeypatch.setattr(
        bot_engine,
        "_alpaca_symbols",
        lambda: (lambda *_a, **_k: {"ap": None}, None),
    )

    def yahoo_zero(symbol, start, end, interval):  # noqa: ARG001
        return _df(0.0)

    monkeypatch.setattr(data_fetcher, "_backup_get_bars", yahoo_zero)
    monkeypatch.setattr(bot_engine, "get_latest_close", lambda df: float(df["close"].iloc[-1]))

    def fail_bars(symbol):  # noqa: ARG001
        raise RuntimeError("bars unavailable")

    monkeypatch.setattr(bot_engine, "get_bars_df", fail_bars)

    price = bot_engine.get_latest_price("AAPL")

    assert price is None
    assert bot_engine._PRICE_SOURCE["AAPL"] == "yahoo_invalid"


def test_get_latest_price_handles_auth_failure(monkeypatch):
    monkeypatch.setattr(alpaca_api, "_ALPACA_SERVICE_AVAILABLE", True)

    def raise_auth(*_a, **_k):
        monkeypatch.setattr(alpaca_api, "_ALPACA_SERVICE_AVAILABLE", False)
        raise AlpacaAuthenticationError("Unauthorized")

    def fail_backup(*_a, **_k):  # pragma: no cover - defensive guard
        raise AssertionError("Backup provider should not be queried on auth failure")

    monkeypatch.setattr(bot_engine, "_alpaca_symbols", lambda: (raise_auth, None))
    monkeypatch.setattr(data_fetcher, "_backup_get_bars", fail_backup)
    monkeypatch.setattr(bot_engine, "get_bars_df", fail_backup)

    price = bot_engine.get_latest_price("AAPL")

    assert price is None
    assert bot_engine._PRICE_SOURCE["AAPL"] == "alpaca_auth_failed"
    assert not is_alpaca_service_available()

