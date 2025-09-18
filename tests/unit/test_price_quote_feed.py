import sys
import types

import pytest

if "numpy" not in sys.modules:  # pragma: no cover - test stub for optional dep

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

        def __getattr__(self, name):  # noqa: D401 - stub fallback
            def _stub(*args, **kwargs):
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

from ai_trading.alpaca_api import AlpacaOrderHTTPError
from ai_trading.core import bot_engine
from ai_trading.utils import base as utils_base


@pytest.fixture(autouse=True)
def _clear_price_source(monkeypatch):
    monkeypatch.setattr(bot_engine, "_PRICE_SOURCE", {})
    yield


def test_get_latest_price_uses_configured_feed(monkeypatch):
    symbol = "AAPL"
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.is_alpaca_service_available",
        lambda: True,
    )

    captured: dict[str, object] = {}

    def fake_alpaca_get(url, *, params=None, **_):
        captured["url"] = url
        captured["params"] = params
        return {"ap": "123.45"}

    def fake_symbols():
        return fake_alpaca_get, None

    monkeypatch.setattr(bot_engine, "_alpaca_symbols", fake_symbols)

    price = bot_engine.get_latest_price(symbol)

    assert price == pytest.approx(123.45)
    assert (
        captured["url"]
        == f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
    )
    assert captured["params"] == {"feed": "sip"}
    assert bot_engine._PRICE_SOURCE[symbol] == "alpaca_ask"


def test_get_latest_price_http_error_falls_back(monkeypatch):
    symbol = "MSFT"
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.is_alpaca_service_available",
        lambda: True,
    )

    def fake_alpaca_get(*_, **__):
        raise AlpacaOrderHTTPError(404, "missing", payload={"msg": "not found"})

    def fake_symbols():
        return fake_alpaca_get, None

    monkeypatch.setattr(bot_engine, "_alpaca_symbols", fake_symbols)
    monkeypatch.setattr("ai_trading.data.fetch._backup_get_bars", lambda *a, **k: object())
    monkeypatch.setattr("ai_trading.core.bot_engine.get_latest_close", lambda df: 77.0)

    price = bot_engine.get_latest_price(symbol)

    assert price == pytest.approx(77.0)
    assert bot_engine._PRICE_SOURCE[symbol] == "yahoo"


def test_get_current_price_uses_configured_feed(monkeypatch):
    symbol = "TSLA"
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")

    captured: dict[str, object] = {}

    def fake_alpaca_get(url, *, params=None, **_):
        captured["url"] = url
        captured["params"] = params
        return {"ap": 222.5}

    monkeypatch.setattr(utils_base, "alpaca_get", fake_alpaca_get)

    price = utils_base.get_current_price(symbol)

    assert price == pytest.approx(222.5)
    assert (
        captured["url"]
        == f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
    )
    assert captured["params"] == {"feed": "sip"}


def test_get_current_price_http_error_uses_fallback(monkeypatch):
    symbol = "NVDA"

    def fake_alpaca_get(*_, **__):
        raise AlpacaOrderHTTPError(422, "bad request")

    monkeypatch.setattr(utils_base, "alpaca_get", fake_alpaca_get)
    monkeypatch.setattr("ai_trading.data.fetch.get_daily_df", lambda *a, **k: object())
    monkeypatch.setattr(utils_base, "get_latest_close", lambda df: 33.0)

    price = utils_base.get_current_price(symbol)

    assert price == pytest.approx(33.0)

