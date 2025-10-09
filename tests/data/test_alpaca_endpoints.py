from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from typing import Any

import pytest


class StubResponse:
    def __init__(self, *, status_code: int = 200, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:  # pragma: no cover - exercised via tests
        return self._payload


class StubSession:
    def __init__(self) -> None:
        self.last_request: SimpleNamespace | None = None
        self._response = StubResponse(
            payload={
                "bars": [
                    {
                        "t": "2024-01-02T15:30:00Z",
                        "o": 10.0,
                        "h": 10.5,
                        "l": 9.5,
                        "c": 10.2,
                        "v": 1000,
                    }
                ]
            }
        )

    def get(self, url: str, *, params: dict[str, Any], headers: dict[str, str], timeout: Any) -> StubResponse:
        self.last_request = SimpleNamespace(url=url, params=params, headers=headers, timeout=timeout)
        return self._response


@pytest.mark.parametrize("data_base", [None, "https://alt-data.alpaca.markets"])
def test_fetch_bars_uses_data_base_and_headers(monkeypatch, data_base):
    from ai_trading.data import fetch as fetch_mod

    session = StubSession()
    monkeypatch.setattr(fetch_mod, "_HTTP_SESSION", session)
    monkeypatch.setattr(fetch_mod, "_ensure_pandas", lambda: __import__("pandas"))
    monkeypatch.setenv("ALPACA_API_KEY", "key123")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret456")
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "0")
    monkeypatch.setenv("ALPACA_HAS_SIP", "0")
    monkeypatch.setenv("ALPACA_ADJUSTMENT", "all")
    if data_base:
        monkeypatch.setenv("ALPACA_DATA_BASE_URL", data_base)
    else:
        monkeypatch.delenv("ALPACA_DATA_BASE_URL", raising=False)

    start = dt.datetime(2024, 1, 2, 15, 0, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 2, 16, 0, tzinfo=dt.timezone.utc)

    result = fetch_mod._fetch_bars("AAPL", start, end, "1Min", feed="iex", adjustment="all")
    assert session.last_request is not None
    expected_base = data_base or "https://data.alpaca.markets"
    assert session.last_request.url == f"{expected_base}/v2/stocks/bars"
    assert session.last_request.params["feed"] == "iex"
    assert session.last_request.params["adjustment"] == "all"
    assert session.last_request.headers["APCA-API-KEY-ID"] == "key123"
    assert session.last_request.headers["APCA-API-SECRET-KEY"] == "secret456"
    assert not result.empty
