import os
import sys
import types

import pytest

import ai_trading.data.fetch as data_fetch

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
from ai_trading.core.enums import OrderSide as CoreOrderSide
from ai_trading.execution import live_trading
from ai_trading.utils import base as utils_base


@pytest.fixture(autouse=True)
def _clear_price_source(monkeypatch):
    monkeypatch.setattr(bot_engine, "_PRICE_SOURCE", {})
    yield


def test_get_latest_price_uses_configured_feed(monkeypatch):
    symbol = "AAPL"
    bot_engine._reset_cycle_cache()
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setattr(bot_engine, "_INTRADAY_FEED_CACHE", "sip", raising=False)
    monkeypatch.setattr(
        bot_engine,
        "_GLOBAL_INTRADAY_FALLBACK_FEED",
        "yahoo",
        raising=False,
    )
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
    assert bot_engine._GLOBAL_INTRADAY_FALLBACK_FEED in {None, "sip", "iex"}
    assert bot_engine._GLOBAL_CYCLE_MINUTE_FEED_OVERRIDE.get(symbol) in {
        None,
        "sip",
        "iex",
    }


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
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
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


def test_get_latest_price_invalid_feed_skips_alpaca(monkeypatch, caplog):
    symbol = "AAPL"
    bot_engine._reset_cycle_cache()
    caplog.set_level("WARNING")
    monkeypatch.setattr(bot_engine, "_get_intraday_feed", lambda: "yahoo")
    monkeypatch.setattr(bot_engine, "_prefer_feed_this_cycle", lambda: None)
    monkeypatch.setattr(
        bot_engine,
        "_get_price_provider_order",
        lambda: ("alpaca_trade", "alpaca_quote", "yahoo"),
    )

    def _fail_alpaca_symbols():  # pragma: no cover - ensure sanitized skip
        raise AssertionError("alpaca should not be called with invalid feed")

    monkeypatch.setattr(bot_engine, "_alpaca_symbols", _fail_alpaca_symbols)
    sentinel = 12.34
    monkeypatch.setattr(bot_engine, "_attempt_yahoo_price", lambda _s: (sentinel, "yahoo"))
    monkeypatch.setattr(bot_engine, "_attempt_bars_price", lambda _s: (None, "bars_invalid"))
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.is_alpaca_service_available", lambda: True
    )

    first_price = bot_engine.get_latest_price(symbol)
    second_price = bot_engine.get_latest_price(symbol)

    assert first_price == pytest.approx(sentinel)
    assert second_price == pytest.approx(sentinel)
    assert bot_engine._PRICE_SOURCE[symbol] == "yahoo"
    invalid_logs = [
        (getattr(record, "provider", None), getattr(record, "requested_feed", None))
        for record in caplog.records
        if record.message == "ALPACA_INVALID_FEED_SKIPPED"
    ]
    assert len(invalid_logs) <= 1
    if invalid_logs:
        provider, feed = invalid_logs[0]
        assert provider == "alpaca"
        assert feed == "yahoo"


def test_get_latest_price_uses_fallback_when_sip_lockout(monkeypatch):
    symbol = "SPY"
    bot_engine._reset_cycle_cache()
    monkeypatch.setenv("PYTEST_RUNNING", "")
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_SIP_UNAUTHORIZED", "1")
    monkeypatch.setattr(
        "ai_trading.data.fetch._SIP_UNAUTHORIZED", True, raising=False
    )
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.is_alpaca_service_available", lambda: True
    )
    monkeypatch.setattr(
        bot_engine,
        "_get_price_provider_order",
        lambda: ("alpaca_trade", "alpaca_quote", "yahoo"),
    )

    def _fail_alpaca_symbols():
        raise AssertionError("alpaca should not be called when SIP is locked out")

    monkeypatch.setattr(bot_engine, "_alpaca_symbols", _fail_alpaca_symbols)

    sentinel = 245.67
    monkeypatch.setattr(
        bot_engine, "_attempt_yahoo_price", lambda _s: (sentinel, "yahoo")
    )

    price = bot_engine.get_latest_price(symbol)

    assert price == pytest.approx(sentinel)
    assert bot_engine._PRICE_SOURCE[symbol] == "yahoo"


def test_get_latest_price_resumes_primary_after_sip_lockout_cleared(monkeypatch):
    symbol = "QQQ"
    bot_engine._reset_cycle_cache()
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_SIP_UNAUTHORIZED", "1")
    monkeypatch.setattr(data_fetch, "_SIP_UNAUTHORIZED", True, raising=False)
    monkeypatch.setattr(data_fetch, "_SIP_UNAUTHORIZED_UNTIL", None, raising=False)
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.is_alpaca_service_available", lambda: True
    )
    monkeypatch.setattr(
        bot_engine,
        "_get_price_provider_order",
        lambda: ("alpaca_trade", "yahoo"),
    )

    fallback_price = 101.01
    primary_price = 212.12
    fallback_calls: list[str] = []
    alpaca_calls: list[str | None] = []

    def _mock_alpaca_trade(_symbol: str, feed: str | None, _cache: dict):
        alpaca_calls.append(feed)
        return primary_price, "alpaca_trade"

    def _mock_yahoo_price(_symbol: str):
        fallback_calls.append(_symbol)
        return fallback_price, "yahoo"

    monkeypatch.setattr(bot_engine, "_attempt_alpaca_trade", _mock_alpaca_trade)
    monkeypatch.setattr(bot_engine, "_attempt_yahoo_price", _mock_yahoo_price)

    # Lockout active without pytest fast path -> fallback feed handles pricing.
    monkeypatch.setenv("PYTEST_RUNNING", "")
    locked_out_price = bot_engine.get_latest_price(symbol)
    assert locked_out_price == pytest.approx(fallback_price)
    assert fallback_calls == [symbol]
    assert not alpaca_calls

    # Fast path under pytest bypasses SIP lockout and reuses Alpaca pricing.
    bot_engine._reset_cycle_cache()
    fallback_calls.clear()
    alpaca_calls.clear()
    monkeypatch.setenv("ALPACA_SIP_UNAUTHORIZED", "1")
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    assert os.getenv("ALPACA_SIP_UNAUTHORIZED") == "1"
    bypass_price = bot_engine.get_latest_price(symbol)
    assert bypass_price == pytest.approx(primary_price)
    assert len(alpaca_calls) == 1
    assert not fallback_calls
    assert bot_engine._PRICE_SOURCE[symbol].startswith("alpaca")

    # Clearing lockout flags restores Alpaca usage even outside pytest fast path.
    bot_engine._reset_cycle_cache()
    fallback_calls.clear()
    alpaca_calls.clear()
    monkeypatch.setenv("PYTEST_RUNNING", "")
    monkeypatch.delenv("ALPACA_SIP_UNAUTHORIZED", raising=False)
    monkeypatch.setattr(data_fetch, "_SIP_UNAUTHORIZED", False, raising=False)
    monkeypatch.setattr(data_fetch, "_SIP_UNAUTHORIZED_UNTIL", None, raising=False)
    cleared_price = bot_engine.get_latest_price(symbol)
    assert cleared_price == pytest.approx(primary_price)
    assert len(alpaca_calls) == 1
    assert not fallback_calls
    assert bot_engine._PRICE_SOURCE[symbol].startswith("alpaca")
    assert not os.getenv("ALPACA_SIP_UNAUTHORIZED")


def test_cached_alpaca_fallback_feed_sanitized(monkeypatch, caplog):
    symbol = "AAPL"
    bot_engine._reset_cycle_cache()
    caplog.set_level("WARNING")
    monkeypatch.setattr(bot_engine, "_INTRADAY_FEED_CACHE", None, raising=False)
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(
        bot_engine,
        "_GLOBAL_INTRADAY_FALLBACK_FEED",
        "yahoo",
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_sip_configured",
        lambda: True,
    )
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.is_alpaca_service_available",
        lambda: True,
    )
    monkeypatch.setattr(
        bot_engine,
        "_get_price_provider_order",
        lambda: ("alpaca_trade",),
    )

    captured: dict[str, object] = {}

    def fake_alpaca_get(url, *, params=None, **_):
        captured["url"] = url
        captured["params"] = params
        return {"price": 145.0}

    monkeypatch.setattr(
        bot_engine,
        "_alpaca_symbols",
        lambda: (fake_alpaca_get, None),
    )

    bot_engine._cache_cycle_fallback_feed("alpaca_sip")
    assert bot_engine._prefer_feed_this_cycle() == "sip"

    price = bot_engine.get_latest_price(symbol)

    assert price == pytest.approx(145.0)
    assert captured["params"] == {"feed": "sip"}
    assert not any(
        record.message == "ALPACA_INVALID_FEED_SKIPPED" for record in caplog.records
    )
    assert bot_engine._GLOBAL_INTRADAY_FALLBACK_FEED in {"sip", "iex"}
    assert bot_engine._GLOBAL_CYCLE_MINUTE_FEED_OVERRIDE.get(symbol) in {
        None,
        "sip",
        "iex",
    }


def test_execute_order_new_instance_no_attr_error(monkeypatch):
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)
    engine.stats = {}

    def fake_market(self, symbol, side, quantity, **_):
        return {
            "id": "test-order",
            "status": "filled",
            "filled_qty": quantity,
            "qty": quantity,
        }

    monkeypatch.setattr(engine, "submit_market_order", types.MethodType(fake_market, engine))

    result = engine.execute_order("AAPL", "buy", 1, order_type="market")

    assert result is not None
    status = getattr(result.status, "value", result.status)
    assert status == "filled"


def test_execute_order_routes_market(monkeypatch):
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)
    calls: dict[str, dict[str, object]] = {}

    def fake_market(self, symbol, side, quantity, **kwargs):
        calls["market"] = {
            "symbol": symbol,
            "side": side,
            "qty": quantity,
            "kwargs": kwargs,
        }
        return types.SimpleNamespace(id="OID-MARKET", client_order_id="CID-MARKET")

    def fake_limit(self, symbol, side, quantity, **kwargs):  # pragma: no cover - ensure unused
        calls["limit"] = {
            "symbol": symbol,
            "side": side,
            "qty": quantity,
            "kwargs": kwargs,
        }
        return {"id": "OID-LIMIT", "client_order_id": "CID-LIMIT"}

    monkeypatch.setattr(engine, "submit_market_order", types.MethodType(fake_market, engine))
    monkeypatch.setattr(engine, "submit_limit_order", types.MethodType(fake_limit, engine))

    order_id = engine.execute_order("AAPL", CoreOrderSide.BUY, 10)

    assert order_id == "OID-MARKET"
    assert "market" in calls and "limit" not in calls
    assert calls["market"]["side"] == "buy"
    assert calls["market"]["kwargs"] == {}


def test_execute_order_routes_limit(monkeypatch):
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)
    calls: dict[str, dict[str, object]] = {}

    def fake_market(self, symbol, side, quantity, **kwargs):  # pragma: no cover - ensure unused
        calls["market"] = {
            "symbol": symbol,
            "side": side,
            "qty": quantity,
            "kwargs": kwargs,
        }
        return types.SimpleNamespace(id="OID-MARKET")

    def fake_limit(self, symbol, side, quantity, **kwargs):
        calls["limit"] = {
            "symbol": symbol,
            "side": side,
            "qty": quantity,
            "kwargs": kwargs,
        }
        return {"id": "OID-LIMIT"}

    monkeypatch.setattr(engine, "submit_market_order", types.MethodType(fake_market, engine))
    monkeypatch.setattr(engine, "submit_limit_order", types.MethodType(fake_limit, engine))

    order_id = engine.execute_order(
        "MSFT",
        CoreOrderSide.SELL,
        7,
        price=123.45,
        tif="ioc",
        extended_hours=True,
    )

    assert order_id == "OID-LIMIT"
    assert "limit" in calls
    limit_call = calls["limit"]
    assert limit_call["side"] == "sell"
    assert limit_call["qty"] == 7
    assert limit_call["kwargs"]["limit_price"] == 123.45
    assert limit_call["kwargs"]["time_in_force"] == "ioc"
    assert limit_call["kwargs"]["extended_hours"] is True
    assert "market" not in calls


def test_execute_order_limit_slippage_does_not_raise(monkeypatch, caplog):
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)
    caplog.set_level("INFO")
    import ai_trading.core.constants as core_constants
    import ai_trading.execution.engine as engine_mod

    monkeypatch.setitem(core_constants.EXECUTION_PARAMETERS, "MAX_SLIPPAGE_BPS", 0.1)
    monkeypatch.setattr(engine_mod, "hash", lambda _symbol: 0, raising=False)

    calls: dict[str, dict[str, object]] = {}

    def fake_market(self, symbol, side, quantity, **kwargs):  # pragma: no cover - ensure unused
        calls["market"] = {"symbol": symbol, "side": side, "qty": quantity, "kwargs": kwargs}
        return {"id": "OID-MARKET"}

    def fake_limit(self, symbol, side, quantity, **kwargs):
        calls["limit"] = {"symbol": symbol, "side": side, "qty": quantity, "kwargs": kwargs}
        return {"id": "OID-LIMIT", "status": "submitted", "qty": quantity}

    monkeypatch.setattr(engine, "submit_market_order", types.MethodType(fake_market, engine))
    monkeypatch.setattr(engine, "submit_limit_order", types.MethodType(fake_limit, engine))

    result = engine.execute_order("FAKE", CoreOrderSide.BUY, 10, price=100.0)

    assert result is not None
    assert "limit" in calls and "market" not in calls
    assert any(record.message == "SLIPPAGE_THRESHOLD_LIMIT_ORDER" for record in caplog.records)


def test_execute_order_market_slippage_guard(monkeypatch, caplog):
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)
    caplog.set_level("WARNING")
    import ai_trading.core.constants as core_constants
    import ai_trading.execution.engine as engine_mod

    monkeypatch.setitem(core_constants.EXECUTION_PARAMETERS, "MAX_SLIPPAGE_BPS", 0.1)
    monkeypatch.setattr(engine_mod, "hash", lambda _symbol: 0, raising=False)

    calls: dict[str, dict[str, object]] = {}

    def fake_market(self, symbol, side, quantity, **kwargs):
        calls["market"] = {"symbol": symbol, "side": side, "qty": quantity, "kwargs": kwargs}
        return {"id": "OID-MARKET", "status": "submitted", "qty": quantity}

    def fake_limit(self, symbol, side, quantity, **kwargs):  # pragma: no cover - ensure unused
        calls["limit"] = {"symbol": symbol, "side": side, "qty": quantity, "kwargs": kwargs}
        return {"id": "OID-LIMIT"}

    monkeypatch.setattr(engine, "submit_market_order", types.MethodType(fake_market, engine))
    monkeypatch.setattr(engine, "submit_limit_order", types.MethodType(fake_limit, engine))

    with pytest.raises(AssertionError):
        engine.execute_order("FAKE", CoreOrderSide.BUY, 10, order_type="market", price=100.0)

    assert "limit" not in calls
    assert any(record.message == "SLIPPAGE_THRESHOLD_EXCEEDED" for record in caplog.records)


def test_submit_limit_order_quantizes_price(monkeypatch):
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)
    engine.is_initialized = True
    engine.shadow_mode = False
    engine.stats = {
        "total_execution_time": 0.0,
        "total_orders": 0,
        "successful_orders": 0,
        "failed_orders": 0,
    }

    monkeypatch.setattr(engine, "_refresh_settings", lambda: None)
    monkeypatch.setattr(engine, "_ensure_initialized", lambda: True)
    monkeypatch.setattr(engine, "_pre_execution_checks", lambda: True)

    captured: dict[str, object] = {}

    def fake_submit(self, order_data):
        captured["order_data"] = order_data
        return {"status": "submitted", "id": "OID-LIMIT"}

    def fake_execute(self, func, order_data):
        captured["execute_order_data"] = order_data
        return func(order_data)

    monkeypatch.setattr(
        engine,
        "_submit_order_to_alpaca",
        types.MethodType(fake_submit, engine),
    )
    monkeypatch.setattr(
        engine,
        "_execute_with_retry",
        types.MethodType(fake_execute, engine),
    )

    price = 935.8800048828125
    response = engine.submit_limit_order("AAPL", "buy", 5, limit_price=price)

    assert response == {"status": "submitted", "id": "OID-LIMIT"}
    assert "order_data" in captured
    submitted_price = captured["order_data"]["limit_price"]
    assert submitted_price == pytest.approx(935.88)
    assert captured["execute_order_data"]["limit_price"] == submitted_price
