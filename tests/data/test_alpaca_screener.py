from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.data import alpaca_screener


class StubResponse:
    def __init__(self, *, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


def test_fetch_market_movers_uses_alpaca_data_base_and_headers(monkeypatch):
    requests: list[dict] = []

    def fake_get(url: str, *, params: dict, headers: dict, timeout):
        requests.append(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return StubResponse(
            payload={
                "gainers": [{"symbol": "AAPL", "percent_change": 3.2, "change": 5.1, "price": 182.5}],
                "losers": [{"symbol": "TSLA", "percent_change": -2.1, "change": -4.0, "price": 171.4}],
                "market_type": "stocks",
                "last_updated": "2026-04-17T14:35:00Z",
            }
        )

    alpaca_screener.reset_screener_cache()
    monkeypatch.setattr(alpaca_screener, "http_get", fake_get)
    monkeypatch.setattr(alpaca_screener, "alpaca_auth_headers", lambda: {"APCA-API-KEY-ID": "key", "APCA-API-SECRET-KEY": "secret"})
    monkeypatch.setattr(alpaca_screener, "get_alpaca_data_base_url", lambda: "https://data.alpaca.markets")

    snapshot = alpaca_screener.fetch_market_movers(top=7, ttl_seconds=0)

    assert requests
    assert requests[0]["url"] == "https://data.alpaca.markets/v1beta1/screener/stocks/movers"
    assert requests[0]["params"] == {"top": 7}
    assert requests[0]["headers"]["APCA-API-KEY-ID"] == "key"
    assert snapshot.gainers[0].symbol == "AAPL"
    assert snapshot.losers[0].symbol == "TSLA"
    assert snapshot.used_fallback is False


def test_fetch_market_movers_falls_back_to_last_good_snapshot(monkeypatch):
    calls = {"count": 0}

    def fake_get(url: str, *, params: dict, headers: dict, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            return StubResponse(
                payload={
                    "gainers": [{"symbol": "NVDA", "percent_change": 4.5, "change": 12.0, "price": 890.0}],
                    "losers": [],
                    "market_type": "stocks",
                    "last_updated": "2026-04-17T15:00:00Z",
                }
            )
        raise RuntimeError("boom")

    alpaca_screener.reset_screener_cache()
    monkeypatch.setattr(alpaca_screener, "http_get", fake_get)
    monkeypatch.setattr(alpaca_screener, "alpaca_auth_headers", lambda: {})
    monkeypatch.setattr(alpaca_screener, "get_alpaca_data_base_url", lambda: "https://data.alpaca.markets")

    now = datetime(2026, 4, 17, 16, 0, tzinfo=UTC)
    fresh = alpaca_screener.fetch_market_movers(top=5, ttl_seconds=0, now=now)
    stale = alpaca_screener.fetch_market_movers(top=5, ttl_seconds=0, now=now)

    assert fresh.used_fallback is False
    assert stale.used_fallback is True
    assert stale.gainers[0].symbol == "NVDA"


def test_fetch_market_movers_rejects_prior_day_last_good(monkeypatch):
    calls = {"count": 0}

    def fake_get(url: str, *, params: dict, headers: dict, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            return StubResponse(
                payload={
                    "gainers": [{"symbol": "NVDA", "percent_change": 4.5, "change": 12.0, "price": 890.0}],
                    "losers": [],
                    "market_type": "stocks",
                    "last_updated": "2026-04-16T15:00:00Z",
                }
            )
        raise RuntimeError("boom")

    alpaca_screener.reset_screener_cache()
    monkeypatch.setattr(alpaca_screener, "http_get", fake_get)
    monkeypatch.setattr(alpaca_screener, "alpaca_auth_headers", lambda: {})
    monkeypatch.setattr(alpaca_screener, "get_alpaca_data_base_url", lambda: "https://data.alpaca.markets")

    fresh = alpaca_screener.fetch_market_movers(
        top=5,
        ttl_seconds=0,
        now=datetime(2026, 4, 16, 16, 0, tzinfo=UTC),
    )
    stale = alpaca_screener.fetch_market_movers(
        top=5,
        ttl_seconds=0,
        now=datetime(2026, 4, 17, 16, 0, tzinfo=UTC),
    )

    assert fresh.used_fallback is False
    assert stale.used_fallback is True
    assert stale.gainers == []


def test_fetch_most_actives_rejects_expired_last_good(monkeypatch):
    calls = {"count": 0}

    def fake_get(url: str, *, params: dict, headers: dict, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            return StubResponse(
                payload={
                    "most_actives": [{"symbol": "AMD", "volume": 1_250_000, "trade_count": 42_000}],
                    "last_updated": "2026-04-17T15:05:00Z",
                }
            )
        raise RuntimeError("boom")

    alpaca_screener.reset_screener_cache()
    monkeypatch.setenv("AI_TRADING_SCREENER_LAST_GOOD_MAX_AGE_SEC", "60")
    monkeypatch.setattr(alpaca_screener, "http_get", fake_get)
    monkeypatch.setattr(alpaca_screener, "alpaca_auth_headers", lambda: {})
    monkeypatch.setattr(alpaca_screener, "get_alpaca_data_base_url", lambda: "https://data.alpaca.markets")
    monkeypatch.setattr(
        alpaca_screener,
        "_now_utc",
        lambda: datetime(2026, 4, 17, 15, 5, tzinfo=UTC),
    )

    fresh = alpaca_screener.fetch_most_actives(
        top=3,
        by="volume",
        ttl_seconds=0,
        now=datetime(2026, 4, 17, 15, 5, tzinfo=UTC),
    )

    monkeypatch.setattr(
        alpaca_screener,
        "_now_utc",
        lambda: datetime(2026, 4, 17, 15, 7, tzinfo=UTC),
    )
    stale = alpaca_screener.fetch_most_actives(
        top=3,
        by="volume",
        ttl_seconds=0,
        now=datetime(2026, 4, 17, 15, 7, tzinfo=UTC),
    )

    assert fresh.used_fallback is False
    assert stale.used_fallback is True
    assert stale.most_actives == []


def test_fetch_most_actives_parses_payload(monkeypatch):
    def fake_get(url: str, *, params: dict, headers: dict, timeout):
        return StubResponse(
            payload={
                "most_actives": [{"symbol": "AMD", "volume": 1_250_000, "trade_count": 42_000}],
                "last_updated": datetime(2026, 4, 17, 15, 5, tzinfo=UTC),
            }
        )

    alpaca_screener.reset_screener_cache()
    monkeypatch.setattr(alpaca_screener, "http_get", fake_get)
    monkeypatch.setattr(alpaca_screener, "alpaca_auth_headers", lambda: {})
    monkeypatch.setattr(alpaca_screener, "get_alpaca_data_base_url", lambda: "https://data.alpaca.markets")

    snapshot = alpaca_screener.fetch_most_actives(top=3, by="volume", ttl_seconds=0)

    assert snapshot.most_actives[0].symbol == "AMD"
    assert snapshot.most_actives[0].volume == 1_250_000
    assert snapshot.used_fallback is False
