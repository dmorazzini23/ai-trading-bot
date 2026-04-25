from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.data import bars as bars_mod


class _StatusError(ValueError):
    status_code = 403


def _bar_frame() -> Any:
    pd = bars_mod.pd
    return pd.DataFrame(
        {
            "timestamp": [datetime(2026, 4, 20, 14, 30, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.5],
            "close": [100.5],
            "volume": [1200],
        }
    )


def test_http_get_bars_turns_non_2xx_response_into_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    start = datetime(2026, 4, 20, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)

    def fake_get_bars(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(status_code=503, text=b"temporarily unavailable")

    monkeypatch.setattr(bars_mod, "_raw_http_get_bars", fake_get_bars)

    result = bars_mod.http_get_bars("spy", "1Min", start, end, feed="iex")

    assert isinstance(result, bars_mod.BarsFetchFailed)
    assert result.symbol == "SPY"
    assert result.feed == "iex"
    assert result.status == 503
    assert result.error == "temporarily unavailable"


def test_client_fetch_stock_bars_falls_back_to_get_bars_with_iso_params() -> None:
    start = datetime(2026, 4, 20, 14, 30, tzinfo=UTC)
    end = start + timedelta(days=1)
    calls: list[dict[str, Any]] = []

    class Client:
        def get_bars(self, symbol: str, timeframe: str, **params: Any) -> Any:
            calls.append({"symbol": symbol, "timeframe": timeframe, **params})
            return _bar_frame()

    request = SimpleNamespace(
        symbol_or_symbols="SPY",
        timeframe="1Day",
        start=start,
        end=end,
        limit=2,
        feed="iex",
    )

    frame = bars_mod._client_fetch_stock_bars(Client(), request)

    assert not frame.empty
    assert calls == [
        {
            "symbol": "SPY",
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 2,
            "feed": "iex",
        }
    ]


def test_safe_get_stock_bars_retries_unauthorized_feed_with_entitled_alt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2026, 4, 20, 14, 30, tzinfo=UTC)
    attempts: list[str] = []

    class Client:
        pass

    request = SimpleNamespace(
        symbol_or_symbols="SPY",
        timeframe="1Day",
        start=start,
        end=start + timedelta(days=1),
        feed="sip",
    )

    def fake_client_fetch(_client: Any, req: Any) -> Any:
        attempts.append(str(req.feed))
        if len(attempts) == 1:
            raise _StatusError("forbidden")
        return _bar_frame()

    feed_resolutions = iter(["sip", "iex"])

    monkeypatch.setattr(bars_mod, "_client_fetch_stock_bars", fake_client_fetch)
    monkeypatch.setattr(bars_mod, "_ensure_entitled_feed", lambda _client, _feed: next(feed_resolutions))
    monkeypatch.setattr(bars_mod.time, "sleep", lambda _seconds: None)

    frame = bars_mod.safe_get_stock_bars(Client(), request, "spy", context="daily")

    assert attempts == ["sip", "iex"]
    assert not frame.empty
    assert request.feed == "iex"


def test_safe_get_stock_bars_minute_exception_recovers_from_minute_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2026, 4, 20, 14, 30, tzinfo=UTC)
    fallback = _bar_frame()

    class Client:
        pass

    request = SimpleNamespace(
        symbol_or_symbols="SPY",
        timeframe="1Min",
        start=start,
        end=start + timedelta(minutes=2),
        feed="iex",
    )

    monkeypatch.setattr(bars_mod, "_client_fetch_stock_bars", lambda *_args: (_ for _ in ()).throw(ValueError("boom")))
    monkeypatch.setattr(bars_mod, "get_minute_df", lambda *_args, **_kwargs: fallback)

    frame = bars_mod.safe_get_stock_bars(Client(), request, "spy", context="minute")

    assert not frame.empty
    assert list(frame["close"]) == [100.5]


def test_fetch_minute_fallback_uses_sip_when_iex_snapshot_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 4, 20, 15, 0, tzinfo=UTC)
    small = bars_mod.pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1],
        },
        index=bars_mod.pd.date_range(now, periods=1, freq="min", tz="UTC"),
    )
    large = bars_mod.pd.DataFrame(
        {
            "open": [1.0] * 301,
            "high": [1.0] * 301,
            "low": [1.0] * 301,
            "close": [1.0] * 301,
            "volume": [1] * 301,
        },
        index=bars_mod.pd.date_range(now, periods=301, freq="min", tz="UTC"),
    )
    feeds: list[str] = []

    def fake_get_minute_bars(
        _symbol: str,
        _start_dt: datetime,
        _end_dt: datetime,
        *,
        feed: str,
        adjustment: str | None = None,
    ) -> Any:
        del adjustment
        feeds.append(feed)
        return small if feed == "iex" else large

    monkeypatch.setattr(bars_mod, "_minute_fallback_window", lambda _now: (now, now + timedelta(hours=6)))
    monkeypatch.setattr(bars_mod, "_get_minute_bars", fake_get_minute_bars)
    monkeypatch.setattr(bars_mod, "expected_regular_minutes", lambda: 390)

    frame = bars_mod.fetch_minute_fallback(object(), "spy", now)

    assert feeds == ["iex", "sip"]
    assert len(frame) == 301


def test_parse_bars_handles_result_payload_and_bad_payload() -> None:
    parsed = bars_mod._parse_bars(
        {
            "results": [
                {
                    "timestamp": "2026-04-20T14:30:00+00:00",
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10.5,
                    "volume": 100,
                }
            ]
        },
        "SPY",
        "UTC",
    )
    bad = bars_mod._parse_bars({"results": object()}, "SPY", "UTC")

    assert list(parsed["close"]) == [10.5]
    assert bad.empty


def test_entitlement_helpers_collect_account_and_generation_sources() -> None:
    class Account:
        market_data_subscription = "sip"
        has_iex = True
        updated_at = "2026-04-20T12:00:00Z"

    class Client:
        entitlements = {"SIP": True}
        permitted_feeds = ["iex", "ignored"]

        def get_account(self) -> Account:
            return Account()

    client = Client()

    assert bars_mod._extract_entitlements(client) == {"iex", "sip"}
    assert bars_mod._extract_generation(client) == datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
