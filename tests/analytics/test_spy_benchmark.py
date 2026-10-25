from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_trading.analytics import spy_benchmark

FILLS = [
    {"ts": "2026-10-20T14:00:30+00:00", "symbol": "AAPL", "side": "buy", "fill_qty": 2, "fill_price": 100.0},
    {"ts": "2026-10-20T14:30:30+00:00", "symbol": "AAPL", "side": "sell", "fill_qty": 2, "fill_price": 99.0},
    {"ts": "2026-10-20T15:00:30+00:00", "symbol": "MSFT", "side": "sell_short", "fill_qty": 1, "fill_price": 50.0},
    {"ts": "2026-10-20T15:10:30+00:00", "symbol": "MSFT", "side": "buy", "fill_qty": 1, "fill_price": 49.0},
]


def _minute(ts: str, close: float) -> dict[str, object]:
    return {"ts": ts, "close": close}


def test_timed_round_trips_fifo_match_with_entry_and_exit_times():
    trips = spy_benchmark.timed_round_trips(FILLS)

    assert [(trip["symbol"], trip["direction"], trip["qty"], trip["pnl"]) for trip in trips] == [
        ("AAPL", "long", 2.0, -2.0),
        ("MSFT", "short", 1.0, 1.0),
    ]
    assert trips[0]["entry_ts"] == datetime(2026, 10, 20, 14, 0, 30, tzinfo=UTC)
    assert trips[0]["exit_ts"] == datetime(2026, 10, 20, 14, 30, 30, tzinfo=UTC)
    assert trips[0]["session"] == "2026-10-20"


def test_trades_vs_spy_uses_completed_bars_only_and_matches_notional():
    minute = [
        _minute("2026-10-20T13:59:00+00:00", 500.0),  # completes 14:00:00, before the 14:00:30 entry
        _minute("2026-10-20T14:00:00+00:00", 999.0),  # still forming at 14:00:30: never used for the entry
        _minute("2026-10-20T14:29:00+00:00", 505.0),
        _minute("2026-10-20T14:59:00+00:00", 510.0),
        _minute("2026-10-20T15:09:00+00:00", 505.0),
    ]

    result = spy_benchmark.trades_vs_spy(FILLS, minute)

    # AAPL long: 200 notional, SPY 500 -> 505 = +1%. MSFT short: 50 notional, SPY 510 -> 505.
    msft_spy = -50.0 * (505.0 / 510.0 - 1.0)
    assert result["closed_round_trips"] == 2
    assert result["priced_round_trips"] == 2
    assert result["winning_round_trips"] == 1
    assert result["bot_pnl_usd"] == pytest.approx(-1.0)
    assert result["spy_equivalent_pnl_usd"] == pytest.approx(2.0 + msft_spy)
    assert result["bot_minus_spy_usd"] == pytest.approx(-1.0 - 2.0 - msft_spy)
    assert result["traded_notional_usd"] == pytest.approx(250.0)
    assert result["by_symbol"]["AAPL"]["spy_equivalent_pnl_usd"] == pytest.approx(2.0)


def test_trades_vs_spy_leaves_trips_without_recent_spy_bars_unpriced():
    minute = [_minute("2026-10-20T13:50:00+00:00", 500.0), _minute("2026-10-20T14:29:00+00:00", 505.0)]

    result = spy_benchmark.trades_vs_spy(FILLS, minute)

    assert result["closed_round_trips"] == 2
    assert result["priced_round_trips"] == 0
    assert result["bot_pnl_usd"] is None
    assert result["bot_pnl_usd_all_trips"] == pytest.approx(-1.0)
    assert result["bot_minus_spy_bps_of_notional"] is None


def test_account_vs_spy_uses_prior_session_close_as_start():
    broker = {
        "available": True,
        "daily": [
            {"date": "2026-10-19", "broker_equity_start": 1000.0, "broker_equity_end": 1001.0},
            {"date": "2026-10-20", "broker_equity_start": 1001.0, "broker_equity_end": 999.0},
        ],
    }
    daily = [{"date": "2026-10-16", "close": 400.0}, {"date": "2026-10-19", "close": 402.0}, {"date": "2026-10-20", "close": 404.0}]

    result = spy_benchmark.account_vs_spy(broker, daily)

    assert result["available"] is True
    assert result["spy_close_start"] == {"date": "2026-10-16", "close": 400.0}
    assert result["account_return_bps"] == pytest.approx(-10.0)
    assert result["spy_return_bps"] == pytest.approx(100.0)
    assert result["excess_return_bps"] == pytest.approx(-110.0)
    assert result["spy_equivalent_pnl_usd"] == pytest.approx(10.0)
    assert spy_benchmark.account_vs_spy({"available": False}, daily)["available"] is False
    assert spy_benchmark.account_vs_spy(broker, daily[1:])["reason"] == "missing_equity_or_spy_close"


class _FakeDataClient:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def get_stock_bars(self, request: object) -> SimpleNamespace:
        self.requests.append(request)
        start = datetime(2026, 10, 20, 4, tzinfo=UTC)
        if len(self.requests) == 1:
            bars = [SimpleNamespace(timestamp=start, close=404.0), SimpleNamespace(timestamp=start - timedelta(days=1), close=0.0)]
        else:
            bars = [SimpleNamespace(timestamp=datetime(2026, 10, 20, 14, tzinfo=UTC), close=404.5)]
        return SimpleNamespace(data={"SPY": bars})


def test_fetch_spy_bars_normalizes_daily_and_minute_closes():
    client = _FakeDataClient()
    now = datetime(2026, 10, 25, tzinfo=UTC)

    bars = spy_benchmark.fetch_spy_bars(now - timedelta(days=7), now, client=client)

    assert bars["available"] is True
    assert bars["daily"] == [{"date": "2026-10-20", "close": 404.0}]  # zero close dropped
    assert bars["minute"] == [{"ts": "2026-10-20T14:00:00+00:00", "close": 404.5}]
    assert [str(request.feed) for request in client.requests] == ["DataFeed.IEX", "DataFeed.IEX"]
    json.dumps(bars, allow_nan=False)


def test_fetch_spy_bars_reports_failure_without_raising():
    class _Broken:
        def get_stock_bars(self, request: object) -> None:
            raise RuntimeError("network down")

    now = datetime(2026, 10, 25, tzinfo=UTC)
    bars = spy_benchmark.fetch_spy_bars(now - timedelta(days=7), now, client=_Broken())

    assert bars["available"] is False
    assert bars["reason"] == "bar_fetch_failed"
    assert "network down" in bars["error"]


def test_weekly_benchmark_is_reported_only_and_unavailable_without_bars():
    result = spy_benchmark.weekly_benchmark({"available": False}, FILLS, {"available": False, "reason": "not_requested"})

    assert result["reported_only"] is True
    assert result["available"] is False
    assert result["reason"] == "not_requested"
