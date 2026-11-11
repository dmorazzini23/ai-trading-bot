from __future__ import annotations

import asyncio
import builtins
import datetime as dt
import importlib
import types
from dataclasses import dataclass
from typing import Any

import pytest

import ai_trading.alpaca_api as api
from ai_trading.exc import RequestException


class _Counter:
    def __init__(self, *, fail: bool = False) -> None:
        self.count = 0
        self.fail = fail

    def inc(self) -> None:
        if self.fail:
            raise RuntimeError("counter unavailable")
        self.count += 1


class _Histogram:
    def __init__(self, *, fail: bool = False) -> None:
        self.values: list[float] = []
        self.fail = fail

    def observe(self, value: float) -> None:
        if self.fail:
            raise RuntimeError("histogram unavailable")
        self.values.append(value)


class _Response:
    def __init__(
        self,
        status_code: int,
        payload: Any = None,
        *,
        text: str = "",
        json_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self._json_error = json_error

    def json(self) -> Any:
        if self._json_error is not None:
            raise self._json_error
        return self._payload


@dataclass
class _Creds:
    api_key: str | None = None
    secret_key: str | None = None
    base_url: str | None = "https://paper-api.alpaca.markets"


@pytest.fixture(autouse=True)
def _clean_alpaca_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.delenv("ALPACA_SHADOW", raising=False)
    monkeypatch.setattr(api, "_HTTP_SESSION", None)
    monkeypatch.setattr(api, "_alpaca_calls_total", _Counter())
    monkeypatch.setattr(api, "_alpaca_errors_total", _Counter())
    monkeypatch.setattr(api, "_alpaca_call_latency", _Histogram())
    monkeypatch.setattr(
        api._AlpacaConfig,
        "from_env",
        staticmethod(lambda: api._AlpacaConfig("https://paper-api.alpaca.markets", "key", "secret", False)),
    )
    api.partial_fill_tracker.clear()
    api.partial_fills.clear()


def test_import_and_env_fallback_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {
            "ai_trading.data.feed_roles",
            "ai_trading.net.http",
            "ai_trading.config",
            "alpaca.trading.client",
        }:
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert api._default_execution_feed() == "iex"
    assert api._lazy_http_session() is None
    assert api._managed_env("ALPACA_API_KEY", "fallback") == "fallback"
    with pytest.raises(RuntimeError, match="alpaca-py==0.42.1 is required"):
        api._ensure_trading_client_cls()
    assert api.TradingClient is None
    assert api.ALPACA_AVAILABLE is False


def test_initialize_uses_data_module_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_import_module(name: str):
        calls.append(name)
        if name in {"alpaca.data.historical.stock", "alpaca.data.historical"}:
            raise ModuleNotFoundError(name)
        return types.SimpleNamespace(__name__=name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    api.initialize()

    assert calls == [
        "alpaca.trading.client",
        "alpaca.data.historical.stock",
        "alpaca.data.historical",
        "alpaca.data",
    ]


def test_cancel_order_request_constructor_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch
    assert not hasattr(api, "TradingClientAdapter")


def test_data_class_and_timeframe_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "alpaca.data":
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(api, "StockBarsRequest", object, raising=False)
    monkeypatch.setattr(api, "TimeFrame", object, raising=False)
    monkeypatch.setattr(api, "TimeFrameUnit", object, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="alpaca-py==0.42.1 is required"):
        api._data_classes()

    monkeypatch.setattr(api, "ALPACA_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="alpaca-py==0.42.1 is required"):
        api.get_stock_bars_request_cls()
    assert api.get_timeframe_unit_cls().__name__ == "object"

    naive = dt.datetime(2024, 1, 2, 3, 4, 5, 999)
    aware = dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    assert api._to_utc(naive).tzinfo is dt.timezone.utc
    assert api._fmt_rfc3339_z(aware) == "2024-01-02T01:04:05Z"
    with pytest.raises(ValueError, match="start and end"):
        api._format_start_end_for_tradeapi("1Min", None, aware)


def test_coerce_timeframe_edge_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    assert api._coerce_timeframe_for_request("1Day", "1Day", timeframe_cls=None, timeframe_unit_cls=None) == "1Day"

    class ExistingFrame:
        pass

    existing = ExistingFrame()
    assert (
        api._coerce_timeframe_for_request(
            existing,
            "1Day",
            timeframe_cls=ExistingFrame,
            timeframe_unit_cls=None,
        )
        is existing
    )

    class Member:
        name = "week"
        value = "weekly"

    class UnitMembers:
        __members__ = {"weekly": Member()}
        Day = "day"

    class KeywordFrame:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if args:
                raise TypeError("keyword only")
            self.amount = kwargs["amount"]
            self.unit = kwargs["unit"]

    converted = api._coerce_timeframe_for_request(
        types.SimpleNamespace(amount=2, unit=types.SimpleNamespace(value="week")),
        "2Week",
        timeframe_cls=KeywordFrame,
        timeframe_unit_cls=UnitMembers,
    )
    assert isinstance(converted, KeywordFrame)
    assert converted.amount == 2
    assert converted.unit is UnitMembers.__members__["weekly"]

    class BadAmount:
        @property
        def amount(self) -> int:
            raise RuntimeError("bad amount")

        unit = None

    class DayFallbackFrame:
        def __init__(self, amount: int, unit: Any) -> None:
            self.amount = amount
            self.unit = unit

    fallback = api._coerce_timeframe_for_request(
        BadAmount(),
        "not-a-timeframe",
        timeframe_cls=DayFallbackFrame,
        timeframe_unit_cls=types.SimpleNamespace(Day="day"),
    )
    assert isinstance(fallback, DayFallbackFrame)
    assert fallback.amount == 1
    assert fallback.unit == "day"

    class BadStringUnit:
        name = None
        value = None

        def __str__(self) -> str:
            raise RuntimeError("cannot stringify")

    string_fallback = api._coerce_timeframe_for_request(
        types.SimpleNamespace(amount=0, unit=BadStringUnit()),
        "1Day",
        timeframe_cls=DayFallbackFrame,
        timeframe_unit_cls=types.SimpleNamespace(Day="day"),
    )
    assert string_fallback.amount == 1

    zero_amount = api._coerce_timeframe_for_request(
        types.SimpleNamespace(amount=-1, unit=types.SimpleNamespace(name="day")),
        "0Day",
        timeframe_cls=DayFallbackFrame,
        timeframe_unit_cls=types.SimpleNamespace(Day="day"),
    )
    assert zero_amount.amount == 1

    monkeypatch.setattr(api, "canonicalize_timeframe", lambda _tf: (_ for _ in ()).throw(RuntimeError("bad tf")))
    canonicalize_fallback = api._coerce_timeframe_for_request(
        types.SimpleNamespace(amount=1, unit=None),
        "broken",
        timeframe_cls=DayFallbackFrame,
        timeframe_unit_cls=types.SimpleNamespace(Day="day"),
    )
    assert canonicalize_fallback.unit == "day"

    class UnitKeyMembers:
        __members__ = {"week": "matched-key"}

    assert (
        api._coerce_timeframe_for_request(
            types.SimpleNamespace(amount=2, unit=types.SimpleNamespace(value="week")),
            "2Week",
            timeframe_cls=DayFallbackFrame,
            timeframe_unit_cls=UnitKeyMembers,
        ).unit
        == "matched-key"
    )

    class ValueMember:
        name = "other"
        value = "month"

    class UnitValueMembers:
        __members__ = {"other": ValueMember()}

    assert (
        api._coerce_timeframe_for_request(
            types.SimpleNamespace(amount=2, unit=types.SimpleNamespace(value="month")),
            "2Month",
            timeframe_cls=DayFallbackFrame,
            timeframe_unit_cls=UnitValueMembers,
        ).unit
        is UnitValueMembers.__members__["other"]
    )

    class NoMatchMembers:
        __members__ = {"other": object()}

    assert (
        api._coerce_timeframe_for_request(
            types.SimpleNamespace(amount=2, unit=types.SimpleNamespace(value="hour")),
            "2Hour",
            timeframe_cls=DayFallbackFrame,
            timeframe_unit_cls=NoMatchMembers,
        ).unit
        is None
    )

    def callable_timeframe(*_args: Any, **_kwargs: Any) -> Any:
        return types.SimpleNamespace(amount=1, unit="day")

    callable_result = api._coerce_timeframe_for_request(
        types.SimpleNamespace(amount=1, unit=types.SimpleNamespace(name="day")),
        "1Day",
        timeframe_cls=callable_timeframe,
        timeframe_unit_cls=types.SimpleNamespace(Day="day"),
    )
    assert callable_result.amount == 1

    class AlwaysRaisesFrame:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ValueError("cannot build")

    original = types.SimpleNamespace(amount=1, unit=types.SimpleNamespace(name="month"))
    assert (
        api._coerce_timeframe_for_request(
            original,
            "1Month",
            timeframe_cls=AlwaysRaisesFrame,
            timeframe_unit_cls=None,
        )
        is original
    )


def test_coerce_timeframe_instance_checks_that_raise() -> None:
    class RaisingMeta(type):
        def __instancecheck__(cls, instance: Any) -> bool:
            raise RuntimeError("instance check failed")

    class RaisingFrame(metaclass=RaisingMeta):
        Day = object()

        def __init__(self, amount: int, unit: Any) -> None:
            self.amount = amount
            self.unit = unit

    coerced = api._coerce_timeframe_for_request(
        types.SimpleNamespace(amount=1, unit=types.SimpleNamespace(name="day")),
        "1Day",
        timeframe_cls=RaisingFrame,
        timeframe_unit_cls=types.SimpleNamespace(Day="day"),
    )

    assert isinstance(coerced, RaisingFrame)


def test_get_rest_credential_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    import ai_trading.broker.alpaca_credentials as credentials

    created: list[tuple[str, dict[str, Any]]] = []

    class DataClient:
        def __init__(self, **kwargs: Any) -> None:
            created.append(("data", kwargs))

    class TradingClient:
        def __init__(self, **kwargs: Any) -> None:
            created.append(("trading", kwargs))

    monkeypatch.setattr(api, "get_data_client_cls", lambda: DataClient)
    monkeypatch.setattr(api, "get_trading_client_cls", lambda: TradingClient)
    monkeypatch.setattr(credentials, "resolve_alpaca_credentials_with_base", lambda: _Creds())
    monkeypatch.setattr(api, "_managed_env", lambda key, default=None, **_kwargs: "oauth-token" if key == "ALPACA_OAUTH" else default)

    assert isinstance(api._get_rest(bars=True), DataClient)
    assert isinstance(api._get_rest(bars=False), TradingClient)
    assert created == [
        ("data", {"oauth_token": "oauth-token"}),
        (
            "trading",
            {
                "oauth_token": "oauth-token",
                "paper": True,
                "url_override": "https://paper-api.alpaca.markets",
            },
        ),
    ]

    monkeypatch.setattr(credentials, "resolve_alpaca_credentials_with_base", lambda: _Creds("key", "secret"))
    with pytest.raises(RuntimeError, match="either ALPACA_API_KEY"):
        api._get_rest()

    monkeypatch.setattr(api, "_managed_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(credentials, "resolve_alpaca_credentials_with_base", lambda: _Creds(None, None))
    with pytest.raises(RuntimeError, match="Missing Alpaca credentials"):
        api._get_rest()

    monkeypatch.setattr(
        credentials,
        "resolve_alpaca_credentials_with_base",
        lambda: _Creds("key", "secret", "https://live-api.alpaca.markets"),
    )
    assert isinstance(api._get_rest(), TradingClient)
    assert created[-1] == (
        "trading",
        {
            "api_key": "key",
            "secret_key": "secret",
            "paper": False,
            "url_override": "https://live-api.alpaca.markets",
        },
    )


def test_bars_window_and_retry_wrapper_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    class Unit:
        name = "Day"

    class Frame:
        unit = Unit()

    monkeypatch.setattr(api, "get_timeframe_cls", lambda: Frame)
    monkeypatch.setattr(api, "_managed_env", lambda key, default=None, **_kwargs: "3" if key == "DATA_LOOKBACK_DAYS_DAILY" else default)
    start, end = api._bars_time_window(Frame())
    assert start.tzinfo is None
    assert end.tzinfo is None
    assert (end.date() - start.date()).days == 3

    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RequestException("temporary")
        return "ok"

    with monkeypatch.context() as ctx:
        ctx.setattr(api, "retry", None)
        wrapped = api._with_retry(flaky)
        assert wrapped() == "ok"
        assert attempts["count"] == 2


def test_get_bars_df_factory_fallbacks_and_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    pd = pytest.importorskip("pandas")

    class Request:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class Rest:
        def get_stock_bars(self, request: Request) -> Any:
            assert request.symbol_or_symbols == ["AAPL"]
            return types.SimpleNamespace(df=pd.DataFrame())

    calls: list[tuple[Any, ...]] = []

    def rest_factory(*args: Any, **kwargs: Any) -> Rest:
        calls.append((args, kwargs))
        if kwargs:
            raise TypeError("positional only")
        return Rest()

    monkeypatch.setattr(api, "_get_rest", rest_factory)
    monkeypatch.setattr(api, "_data_classes", lambda: (_ for _ in ()).throw(RuntimeError("no sdk classes")))
    monkeypatch.setattr(api, "get_stock_bars_request_cls", lambda: Request)
    monkeypatch.setattr(api, "_require_pandas", lambda _consumer: pd)
    monkeypatch.setattr(api, "_canon_symbol", lambda symbol: symbol.upper())
    monkeypatch.setattr(api, "_normalize_timeframe_for_tradeapi", lambda _tf: ("1Min", "1Min"))
    monkeypatch.setattr(api, "_coerce_timeframe_for_request", lambda tf, *_args, **_kwargs: tf)
    monkeypatch.setattr(
        api,
        "_format_start_end_for_tradeapi",
        lambda _tf, start, end: (start, end, "start", "end"),
    )

    result = api.get_bars_df(
        "aapl",
        start=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        end="not-a-date",
    )

    assert result.empty
    assert calls == [((), {"bars": True}), ((True,), {})]


def test_get_bars_df_rest_factory_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    pd = pytest.importorskip("pandas")

    def rest_factory(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("unavailable")

    monkeypatch.setattr(api, "_get_rest", rest_factory)
    monkeypatch.setattr(api, "_require_pandas", lambda _consumer: pd)
    monkeypatch.setattr(api, "_canon_symbol", lambda symbol: symbol)
    monkeypatch.setattr(api, "get_stock_bars_request_cls", lambda: object)

    with pytest.raises(RuntimeError, match="_get_rest unavailable"):
        api.get_bars_df("AAPL")


def test_get_bars_df_api_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    pd = pytest.importorskip("pandas")

    class APIError(Exception):
        status_code = 400

        @property
        def response(self) -> Any:
            class BadResponse:
                @property
                def text(self) -> str:
                    raise ValueError("unreadable")

            return BadResponse()

    class Request:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class Rest:
        def get_stock_bars(self, _request: Request) -> Any:
            raise APIError("bad request")

    monkeypatch.setattr(api, "get_api_error_cls", lambda: APIError)
    monkeypatch.setattr(api, "get_stock_bars_request_cls", lambda: Request)
    monkeypatch.setattr(api, "_get_rest", lambda bars=True: Rest())
    monkeypatch.setattr(api, "_require_pandas", lambda _consumer: pd)
    monkeypatch.setattr(api, "_canon_symbol", lambda symbol: symbol)
    monkeypatch.setattr(api, "_normalize_timeframe_for_tradeapi", lambda _tf: ("1Min", "1Min"))
    monkeypatch.setattr(api, "_coerce_timeframe_for_request", lambda tf, *_args, **_kwargs: tf)
    monkeypatch.setattr(api, "_bars_time_window", lambda _tf: (dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc), dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)))
    monkeypatch.setattr(api, "_format_start_end_for_tradeapi", lambda _tf, start, end: (start, end, "start", "end"))

    assert api.get_bars_df("MSFT").empty


def test_get_bars_df_rate_limit_sleep_and_metrics_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    pd = pytest.importorskip("pandas")

    class APIError(Exception):
        def __init__(self) -> None:
            super().__init__("rate limited")
            self.status_code = 429
            self.response = types.SimpleNamespace(text="rate limited")

    class Request:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class Rest:
        def __init__(self) -> None:
            self.calls = 0

        def get_stock_bars(self, _request: Request) -> Any:
            self.calls += 1
            if self.calls == 1:
                raise APIError()
            frame = pd.DataFrame({"close": [1.0]}, index=pd.date_range("2024-01-01", periods=1, tz="UTC"))
            return types.SimpleNamespace(df=frame)

    class TimeModule:
        def sleep(self, _delay: float) -> None:
            raise RuntimeError("sleep interrupted")

    rest = Rest()
    monkeypatch.setattr(api, "get_api_error_cls", lambda: APIError)
    monkeypatch.setattr(api, "get_stock_bars_request_cls", lambda: Request)
    monkeypatch.setattr(api, "_get_rest", lambda bars=True: rest)
    monkeypatch.setattr(api, "_require_pandas", lambda _consumer: pd)
    monkeypatch.setattr(api, "_canon_symbol", lambda symbol: symbol)
    monkeypatch.setattr(api, "_normalize_timeframe_for_tradeapi", lambda _tf: ("1Min", "1Min"))
    monkeypatch.setattr(api, "_coerce_timeframe_for_request", lambda tf, *_args, **_kwargs: tf)
    monkeypatch.setattr(api, "_bars_time_window", lambda _tf: (dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc), dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)))
    monkeypatch.setattr(api, "_format_start_end_for_tradeapi", lambda _tf, start, end: (start, end, "start", "end"))
    monkeypatch.setattr(api, "time", TimeModule())
    monkeypatch.setattr(api, "_alpaca_calls_total", _Counter(fail=True))

    assert not api.get_bars_df("QQQ").empty


def test_get_bars_df_network_retry_sleep_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    pd = pytest.importorskip("pandas")

    class Request:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class Rest:
        def get_stock_bars(self, _request: Request) -> Any:
            raise RequestException("network down")

    class TimeModule:
        def sleep(self, _delay: float) -> None:
            raise RuntimeError("sleep interrupted")

    monkeypatch.setattr(api, "get_stock_bars_request_cls", lambda: Request)
    monkeypatch.setattr(api, "_get_rest", lambda bars=True: Rest())
    monkeypatch.setattr(api, "_require_pandas", lambda _consumer: pd)
    monkeypatch.setattr(api, "_canon_symbol", lambda symbol: symbol)
    monkeypatch.setattr(api, "_normalize_timeframe_for_tradeapi", lambda _tf: ("1Min", "1Min"))
    monkeypatch.setattr(api, "_coerce_timeframe_for_request", lambda tf, *_args, **_kwargs: tf)
    monkeypatch.setattr(api, "_bars_time_window", lambda _tf: (dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc), dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)))
    monkeypatch.setattr(api, "_format_start_end_for_tradeapi", lambda _tf, start, end: (start, end, "start", "end"))
    monkeypatch.setattr(api, "time", TimeModule())

    with pytest.raises(RequestException, match="network down"):
        api.get_bars_df("IBM")


def test_alpaca_get_payload_shapes_and_metrics_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _Response(200, json_error=ValueError("json unavailable")),
        _Response(200, ["one", "two"]),
        _Response(200, {"account_number": "acct-1"}),
    ]

    class Session:
        def get(self, *_args: Any, **_kwargs: Any) -> Any:
            return responses.pop(0)

    monkeypatch.setattr(api, "_get_http_session", lambda: Session())
    monkeypatch.setattr(api, "_alpaca_calls_total", _Counter(fail=True))

    assert api.alpaca_get("https://broker.test/v2/account") == {}
    assert api.alpaca_get("/v2/positions") == {"data": ["one", "two"]}
    assert api.alpaca_get("/v2/account") == {"account_number": "acct-1"}


def test_coerce_float_and_trade_update_state() -> None:
    assert api._coerce_float("12.5") == 12.5
    assert api._coerce_float("bad") is None
    assert asyncio.iscoroutinefunction(api.start_trade_updates_stream)

    asyncio.run(api.handle_trade_update(None))
    asyncio.run(api.handle_trade_update(types.SimpleNamespace(event="", order=types.SimpleNamespace(id="o-0"))))
    asyncio.run(api.handle_trade_update(types.SimpleNamespace(event="partial_fill", order=None)))
    asyncio.run(api.handle_trade_update(types.SimpleNamespace(event="partial_fill", order=types.SimpleNamespace(symbol="AAPL"))))

    order = types.SimpleNamespace(id="o-1", symbol="AAPL", filled_qty="1.5", filled_avg_price="100.25")
    asyncio.run(api.handle_trade_update(types.SimpleNamespace(event="partial_fill", order=order)))
    assert api.partial_fills["o-1"]["filled_qty"] == 1.5
    asyncio.run(api.handle_trade_update(types.SimpleNamespace(event_type="partial_filled", order_data=order)))
    assert "o-1" in api.partial_fills
    asyncio.run(api.handle_trade_update(types.SimpleNamespace(event="fill", order=order)))
    assert "o-1" not in api.partial_fills
