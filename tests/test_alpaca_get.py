from __future__ import annotations

from types import SimpleNamespace

import pytest

import ai_trading.alpaca_api as alpaca_api
from ai_trading.alpaca_api import AlpacaOrderHTTPError
from ai_trading.exc import RequestException


class _Counter:
    def __init__(self) -> None:
        self.count = 0

    def inc(self) -> None:
        self.count += 1


class _Histogram:
    def __init__(self) -> None:
        self.values: list[float] = []

    def observe(self, value: float) -> None:
        self.values.append(value)


class _Response:
    def __init__(self, status_code: int, payload: dict[str, object] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict[str, object]:
        return self._payload


@pytest.fixture(autouse=True)
def _alpaca_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_BASE_URL", "https://example.com")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key-id")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    monkeypatch.delenv("ALPACA_SHADOW", raising=False)


@pytest.fixture
def _http_stub(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    stub = SimpleNamespace(response=None, called=None)

    def fake_get(url, headers=None, params=None, timeout=None):
        stub.called = {
            "url": url,
            "headers": headers,
            "params": params,
            "timeout": timeout,
        }
        if isinstance(stub.response, Exception):
            raise stub.response
        return stub.response

    stub.get = fake_get  # type: ignore[attr-defined]
    monkeypatch.setattr(alpaca_api, "_HTTP", stub)
    return stub


@pytest.fixture
def _metrics_stub(monkeypatch: pytest.MonkeyPatch) -> tuple[_Counter, _Counter, _Histogram]:
    calls = _Counter()
    errors = _Counter()
    latency = _Histogram()
    monkeypatch.setattr(alpaca_api, "_alpaca_calls_total", calls)
    monkeypatch.setattr(alpaca_api, "_alpaca_errors_total", errors)
    monkeypatch.setattr(alpaca_api, "_alpaca_call_latency", latency)
    return calls, errors, latency


def test_alpaca_get_success(_http_stub: SimpleNamespace, _metrics_stub: tuple[_Counter, _Counter, _Histogram]) -> None:
    calls, errors, latency = _metrics_stub
    _http_stub.response = _Response(
        status_code=200,
        payload={"symbol": "AAPL", "quote": {"ap": 123.45}},
    )

    result = alpaca_api.alpaca_get("/v2/stocks/AAPL/quotes/latest", params={"feed": "sip"}, timeout=5)

    assert result == {"ap": 123.45}
    assert _http_stub.called["url"] == "https://example.com/v2/stocks/AAPL/quotes/latest"
    assert _http_stub.called["params"] == {"feed": "sip"}
    assert _http_stub.called["timeout"] == 5
    headers = _http_stub.called["headers"]
    assert headers["APCA-API-KEY-ID"] == "key-id"
    assert headers["APCA-API-SECRET-KEY"] == "secret"
    assert headers["Accept"] == "application/json"
    assert calls.count == 1
    assert errors.count == 0
    assert latency.values, "latency histogram should record an observation"


def test_alpaca_get_http_error(_http_stub: SimpleNamespace, _metrics_stub: tuple[_Counter, _Counter, _Histogram]) -> None:
    calls, errors, _ = _metrics_stub
    _http_stub.response = _Response(
        status_code=429,
        payload={"message": "Too many requests"},
        text="Too many requests",
    )

    with pytest.raises(AlpacaOrderHTTPError) as excinfo:
        alpaca_api.alpaca_get("/v2/bad")

    assert excinfo.value.status_code == 429
    assert excinfo.value.payload == {"message": "Too many requests"}
    assert calls.count == 1
    assert errors.count == 1


def test_alpaca_get_network_error(_http_stub: SimpleNamespace, _metrics_stub: tuple[_Counter, _Counter, _Histogram]) -> None:
    calls, errors, _ = _metrics_stub
    _http_stub.response = RequestException("boom")

    with pytest.raises(RequestException) as excinfo:
        alpaca_api.alpaca_get("/v2/stocks/SPY/quotes/latest")

    assert "Network error calling" in str(excinfo.value)
    assert calls.count == 1
    # errors counter increments on network failures too
    assert errors.count == 1


def test_get_latest_price_uses_live_quote(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from ai_trading.core import bot_engine
    import ai_trading.data.fetch as data_fetcher

    def _fail(*_a, **_k):  # pragma: no cover - defensive within the test
        pytest.fail("Fallback should not be invoked")

    monkeypatch.setattr(bot_engine, "get_latest_close", _fail)
    monkeypatch.setattr(bot_engine, "get_bars_df", _fail)
    monkeypatch.setattr(data_fetcher, "_backup_get_bars", _fail)
    monkeypatch.setattr(
        bot_engine,
        "_alpaca_symbols",
        lambda: (lambda *_a, **_k: {"ap": "111.11"}, None),
    )

    caplog.clear()
    with caplog.at_level("WARNING"):
        price = bot_engine.get_latest_price("AAPL")

    assert price == pytest.approx(111.11)
    assert "ALPACA_PRICE_NONE" not in caplog.messages
