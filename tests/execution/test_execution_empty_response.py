import types
from decimal import Decimal

import pytest

from ai_trading.execution import live_trading


class _DummyTradingClient:
    def __init__(self, response):
        self._response = response
        self.submitted: list[object] = []

    def submit_order(self, *, order_data):
        self.submitted.append(order_data)
        return self._response


class _LookupTradingClient(_DummyTradingClient):
    def __init__(self, response, *, lookup_order=None, duplicate_error=None):
        super().__init__(response)
        self.lookup_order = lookup_order
        self.duplicate_error = duplicate_error
        self.last_client_order_id = None

    def submit_order(self, *, order_data):
        self.submitted.append(order_data)
        self.last_client_order_id = getattr(order_data, "client_order_id", None)
        if self.duplicate_error is not None:
            raise self.duplicate_error
        return self._response

    def get_order_by_client_order_id(self, client_order_id):
        if self.lookup_order is None:
            raise RuntimeError("order not found")
        if client_order_id != self.last_client_order_id:
            raise RuntimeError("order not found")
        return self.lookup_order


class _DummyLimitOrderRequest:
    def __init__(self, *, limit_price, **kwargs):
        self.limit_price = limit_price
        for key, value in kwargs.items():
            setattr(self, key, value)


class _DummyMarketOrderRequest:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture(autouse=True)
def patch_alpaca_classes(monkeypatch):
    monkeypatch.setattr(live_trading, "OrderSide", types.SimpleNamespace(BUY="buy", SELL="sell"))
    monkeypatch.setattr(live_trading, "TimeInForce", types.SimpleNamespace(DAY="day"))
    monkeypatch.setattr(live_trading, "LimitOrderRequest", _DummyLimitOrderRequest)
    monkeypatch.setattr(live_trading, "MarketOrderRequest", _DummyMarketOrderRequest)
    yield


def _make_engine(monkeypatch, response):
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)
    engine.is_initialized = True
    engine.shadow_mode = False
    engine.stats = {
        "total_execution_time": 0.0,
        "total_orders": 0,
        "successful_orders": 0,
        "failed_orders": 0,
        "retry_count": 0,
        "capacity_skips": 0,
        "skipped_orders": 0,
        "circuit_breaker_trips": 0,
    }
    engine.retry_config = {
        "base_delay": 0.01,
        "max_attempts": 1,
        "exponential_base": 2,
        "max_delay": 0.01,
    }
    engine.circuit_breaker = {
        "failure_count": 0,
        "max_failures": 5,
        "reset_time": 300,
        "last_failure": None,
        "is_open": False,
    }
    engine.trading_client = _DummyTradingClient(response)
    monkeypatch.setattr(engine, "_refresh_settings", lambda: None)
    monkeypatch.setattr(engine, "_ensure_initialized", lambda: True)
    monkeypatch.setattr(engine, "_pre_execution_checks", lambda: True)
    monkeypatch.setattr(
        engine,
        "_execute_with_retry",
        types.MethodType(lambda self, func, *args, **kwargs: func(*args, **kwargs), engine),
    )
    monkeypatch.setattr(
        live_trading,
        "get_trading_config",
        lambda: types.SimpleNamespace(
            nbbo_required_for_limit=False,
            execution_require_realtime_nbbo=False,
            execution_market_on_degraded=False,
            degraded_feed_mode="widen",
            degraded_feed_limit_widen_bps=0,
            min_quote_freshness_ms=1500,
        ),
    )
    monkeypatch.setattr(live_trading.provider_monitor, "is_disabled", lambda *_a, **_k: False)
    monkeypatch.setattr(live_trading, "_require_bid_ask_quotes", lambda: False)
    monkeypatch.setattr(live_trading, "guard_shadow_active", lambda: False)
    monkeypatch.setattr(live_trading, "is_safe_mode_active", lambda: False)
    return engine


def test_submit_limit_order_handles_empty_response(monkeypatch):
    engine = _make_engine(monkeypatch, response=None)
    result = engine.submit_limit_order("AAPL", "buy", 5, limit_price=123.45)

    assert result["symbol"] == "AAPL"
    assert result["qty"] == 5
    assert result["status"] == "accepted"
    assert result["id"] == result["client_order_id"]
    assert result["id"].startswith("alpaca-") or result["id"].startswith("order_")


def test_execute_order_uses_fallback_identifier(monkeypatch):
    fallback = {
        "id": "",
        "client_order_id": "client-123",
        "status": None,
        "symbol": "AAPL",
        "qty": 7,
        "limit_price": 99.5,
        "raw": None,
    }

    engine = _make_engine(monkeypatch, response=fallback)
    result = engine.execute_order("AAPL", "buy", 7, order_type="limit", limit_price=99.5)

    assert str(result) == "client-123"
    assert result.order.id == "client-123"
    assert result.requested_quantity == 7


def test_submit_order_recovers_when_submit_returns_none(monkeypatch):
    engine = _make_engine(monkeypatch, response=None)
    recovered = {
        "id": "recovered-1",
        "client_order_id": "recover-cid-1",
        "status": "pending_new",
        "symbol": "AAPL",
        "side": "buy",
        "qty": 5,
        "limit_price": 123.45,
    }
    engine.trading_client = _LookupTradingClient(None, lookup_order=recovered)
    monkeypatch.setattr(live_trading, "_runtime_env", lambda *_a, **_k: "")

    result = engine._submit_order_to_alpaca(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 5,
            "type": "limit",
            "limit_price": 123.45,
            "time_in_force": "day",
            "client_order_id": "recover-cid-1",
        }
    )

    assert result is not None
    assert result["id"] == "recovered-1"
    assert result["client_order_id"] == "recover-cid-1"


def test_submit_order_builds_notional_market_request(monkeypatch):
    engine = _make_engine(monkeypatch, response={"id": "ok", "status": "accepted"})

    result = engine._submit_order_to_alpaca(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 0,
            "notional": "125.50",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": "notional-cid-1",
        }
    )

    submitted = engine.trading_client.submitted[-1]
    assert result is not None
    assert getattr(submitted, "notional") == "125.50"
    assert not hasattr(submitted, "qty")


def test_submit_order_quantizes_limit_price_before_broker_request(monkeypatch):
    engine = _make_engine(monkeypatch, response={"id": "ok", "status": "accepted"})
    monkeypatch.setattr(live_trading, "get_tick_size", lambda _symbol: Decimal("0.1"))

    engine._submit_order_to_alpaca(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 5,
            "type": "limit",
            "limit_price": 123.47,
            "time_in_force": "day",
            "client_order_id": "cid-quantize-1",
        }
    )

    submitted = engine.trading_client.submitted
    assert submitted
    request_obj = submitted[-1]
    expected = float(live_trading.Money(123.47).quantize(Decimal("0.1")).amount)
    if isinstance(request_obj, dict):
        actual = request_obj.get("limit_price")
    else:
        actual = getattr(request_obj, "limit_price", None)
    assert actual == pytest.approx(expected)


def test_submit_order_rejects_invalid_limit_price_before_broker_submit(monkeypatch):
    engine = _make_engine(monkeypatch, response={"id": "ok", "status": "accepted"})

    with pytest.raises(live_trading.NonRetryableBrokerError) as exc:
        engine._submit_order_to_alpaca(
            {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 5,
                "type": "limit",
                "limit_price": None,
                "time_in_force": "day",
                "client_order_id": "cid-invalid-limit-1",
            }
        )

    assert "invalid_limit_price" in str(exc.value)
    assert engine.trading_client.submitted == []


def test_submit_order_recovers_on_duplicate_client_order_id(monkeypatch):
    class _DuplicateClientOrderIdError(Exception):
        def __init__(self):
            super().__init__("client_order_id already exists")
            self.status_code = 422
            self.code = "duplicate_client_order_id"
            self.message = "client_order_id already exists"
            self._error = {
                "code": "duplicate_client_order_id",
                "message": "client_order_id already exists",
            }

    engine = _make_engine(monkeypatch, response=None)
    recovered = {
        "id": "recovered-2",
        "client_order_id": "recover-cid-2",
        "status": "pending_new",
        "symbol": "AAPL",
        "side": "buy",
        "qty": 2,
        "limit_price": 99.5,
    }
    client = _LookupTradingClient(
        None,
        lookup_order=recovered,
        duplicate_error=_DuplicateClientOrderIdError(),
    )
    engine.trading_client = client
    monkeypatch.setattr(live_trading, "_runtime_env", lambda *_a, **_k: "")
    monkeypatch.setattr(live_trading, "APIError", _DuplicateClientOrderIdError)

    result = engine._submit_order_to_alpaca(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 2,
            "type": "limit",
            "limit_price": 99.5,
            "time_in_force": "day",
            "client_order_id": "recover-cid-2",
        }
    )

    assert result is not None
    assert result["id"] == "recovered-2"
    assert result["client_order_id"] == "recover-cid-2"


def test_execute_order_suppresses_submit_no_result_when_recent_outcome_exists(monkeypatch):
    engine = _make_engine(monkeypatch, response={"status": "accepted"})
    reasons: list[str] = []

    monkeypatch.setattr(
        engine,
        "submit_limit_order",
        types.MethodType(lambda self, *_a, **_k: None, engine),
    )
    monkeypatch.setattr(
        engine,
        "_record_submit_failure",
        types.MethodType(
            lambda self, **kwargs: reasons.append(str(kwargs.get("reason") or "")),
            engine,
        ),
    )
    engine._last_submit_outcome = {
        "status": "failed",
        "reason": "submit_exception",
        "symbol": "AAPL",
        "side": "buy",
        "recorded_at_mono": float(live_trading.monotonic_time()),
    }

    engine.execute_order("AAPL", "buy", 1, order_type="limit", limit_price=100.0)

    assert "submit_no_result" not in reasons


def test_execute_order_recovers_submit_no_result_from_client_lookup(monkeypatch):
    engine = _make_engine(monkeypatch, response={"status": "accepted"})
    reasons: list[str] = []
    recovered_order = {
        "id": "recovered-exec-1",
        "client_order_id": "recover-cid-exec-1",
        "status": "filled",
        "symbol": "AAPL",
        "side": "buy",
        "qty": 1,
    }

    class _RecoverClient:
        def get_order_by_client_order_id(self, client_order_id):
            assert client_order_id == "recover-cid-exec-1"
            return recovered_order

    engine.trading_client = _RecoverClient()
    monkeypatch.setattr(
        engine,
        "submit_limit_order",
        types.MethodType(lambda self, *_a, **_k: None, engine),
    )
    monkeypatch.setattr(
        engine,
        "_record_submit_failure",
        types.MethodType(
            lambda self, **kwargs: reasons.append(str(kwargs.get("reason") or "")),
            engine,
        ),
    )

    result = engine.execute_order(
        "AAPL",
        "buy",
        1,
        order_type="limit",
        limit_price=100.0,
        client_order_id="recover-cid-exec-1",
    )

    assert result.order.id == "recovered-exec-1"
    assert "submit_no_result" not in reasons


def test_execute_order_recovers_submit_no_result_from_recent_orders_scan(monkeypatch):
    engine = _make_engine(monkeypatch, response={"status": "accepted"})
    reasons: list[str] = []
    recovered_order = {
        "id": "recovered-exec-2",
        "client_order_id": "recover-cid-exec-2",
        "status": "filled",
        "symbol": "AAPL",
        "side": "buy",
        "qty": 1,
    }

    class _RecentScanClient:
        def get_order_by_client_order_id(self, _client_order_id):
            raise RuntimeError("order not found")

        def list_orders(self, **kwargs):
            status = str(kwargs.get("status") or "").strip().lower()
            if status == "open":
                return []
            return [recovered_order]

    engine.trading_client = _RecentScanClient()
    monkeypatch.setattr(
        engine,
        "submit_limit_order",
        types.MethodType(lambda self, *_a, **_k: None, engine),
    )
    monkeypatch.setattr(
        engine,
        "_record_submit_failure",
        types.MethodType(
            lambda self, **kwargs: reasons.append(str(kwargs.get("reason") or "")),
            engine,
        ),
    )

    result = engine.execute_order(
        "AAPL",
        "buy",
        1,
        order_type="limit",
        limit_price=100.0,
        client_order_id="recover-cid-exec-2",
    )

    assert result.order.id == "recovered-exec-2"
    assert "submit_no_result" not in reasons


def test_recent_order_scan_uses_adapter_for_native_get_orders(monkeypatch):
    engine = _make_engine(monkeypatch, response={"status": "accepted"})
    recovered_order = {
        "id": "recovered-native-1",
        "client_order_id": "recover-native-cid-1",
        "status": "filled",
        "symbol": "AAPL",
        "side": "buy",
        "qty": 1,
    }
    calls: list[tuple[object, str, list[str] | None]] = []

    class _NativeOnlyClient:
        def get_orders(self, *, filter):
            raise AssertionError("native get_orders must be called through list_alpaca_orders")

        def list_orders(self, **_kwargs):
            raise AssertionError("list_orders fallback should not be needed")

    def _fake_list_alpaca_orders(client, *, status="open", symbols=None):
        calls.append((client, status, symbols))
        return [recovered_order] if status == "all" else []

    engine.trading_client = _NativeOnlyClient()
    monkeypatch.setattr(live_trading, "list_alpaca_orders", _fake_list_alpaca_orders)

    result = engine._lookup_order_in_open_recent_orders_by_client_order_id(
        "recover-native-cid-1",
        symbol="AAPL",
    )

    assert result == recovered_order
    assert calls == [
        (engine.trading_client, "open", ["AAPL"]),
        (engine.trading_client, "all", ["AAPL"]),
    ]


def test_execute_order_submit_no_result_treats_alpaca_404_lookup_as_not_found(monkeypatch):
    engine = _make_engine(monkeypatch, response={"status": "accepted"})
    reasons: list[str] = []

    class _ImportedLookupAPIError(Exception):
        def __init__(self, message: str, *, status_code: int | None = None, code=None):
            super().__init__(message)
            self.status_code = status_code
            self.code = code

        @property
        def message(self) -> str:
            return str(self)

    class _CompatLookupAPIError(_ImportedLookupAPIError):
        pass

    class _Lookup404Client:
        def get_order_by_client_order_id(self, _client_order_id):
            raise _ImportedLookupAPIError(
                '{"code":40410000,"message":"order not found"}',
                status_code=404,
                code=40410000,
            )

        def list_orders(self, **_kwargs):
            return []

    engine.trading_client = _Lookup404Client()
    monkeypatch.setattr(live_trading, "_AlpacaAPIError", _ImportedLookupAPIError)
    monkeypatch.setattr(live_trading, "APIError", _CompatLookupAPIError)
    monkeypatch.setattr(
        live_trading,
        "LIVE_TRADING_ORDER_LOOKUP_EXC",
        live_trading.LIVE_TRADING_FALLBACK_EXC + (_ImportedLookupAPIError,),
    )
    monkeypatch.setattr(
        engine,
        "submit_limit_order",
        types.MethodType(lambda self, *_a, **_k: None, engine),
    )
    monkeypatch.setattr(
        engine,
        "_record_submit_failure",
        types.MethodType(
            lambda self, **kwargs: reasons.append(str(kwargs.get("reason") or "")),
            engine,
        ),
    )

    result = engine.execute_order(
        "AAPL",
        "buy",
        1,
        order_type="limit",
        limit_price=100.0,
        client_order_id="recover-cid-exec-404",
    )

    assert result is None
    assert reasons[-1] == "submit_no_result"


def test_execute_order_submit_no_result_records_client_order_id(monkeypatch):
    engine = _make_engine(monkeypatch, response={"status": "accepted"})
    captured: list[dict[str, object]] = []

    class _NoResultClient:
        def get_order_by_client_order_id(self, _client_order_id):
            raise RuntimeError("order not found")

        def list_orders(self, **_kwargs):
            return []

    engine.trading_client = _NoResultClient()
    monkeypatch.setattr(
        engine,
        "submit_limit_order",
        types.MethodType(lambda self, *_a, **_k: None, engine),
    )
    monkeypatch.setattr(
        engine,
        "_record_submit_failure",
        types.MethodType(lambda self, **kwargs: captured.append(dict(kwargs)), engine),
    )

    result = engine.execute_order(
        "AAPL",
        "buy",
        1,
        order_type="limit",
        limit_price=100.0,
        client_order_id="cid-no-result-1",
    )

    assert result is None
    assert captured
    assert captured[-1]["reason"] == "submit_no_result"
    assert captured[-1]["client_order_id"] == "cid-no-result-1"


def test_submit_limit_order_records_submit_exception(monkeypatch):
    engine = _make_engine(monkeypatch, response={"status": "accepted"})
    captured: list[dict[str, object]] = []

    def _raise_timeout(_self, *_a, **_k):
        raise TimeoutError("submit timed out")

    monkeypatch.setattr(
        engine,
        "_execute_with_retry",
        types.MethodType(_raise_timeout, engine),
    )
    monkeypatch.setattr(
        engine,
        "_record_submit_failure",
        types.MethodType(lambda self, **kwargs: captured.append(kwargs), engine),
    )

    result = engine.submit_limit_order("AAPL", "buy", 1, limit_price=123.45)

    assert result is None
    assert captured
    assert captured[-1]["reason"] == "submit_exception"
