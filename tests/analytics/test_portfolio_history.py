"""Broker portfolio-history normalisation, caching and cross-checks (reported only)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from ai_trading.analytics import portfolio_history as ph

_NY = ZoneInfo("America/New_York")


def _close_epoch(day: str) -> int:
    # Alpaca stamps daily rows at 20:00 New York time on the session date.
    return int(datetime.fromisoformat(f"{day}T20:00:00").replace(tzinfo=_NY).timestamp())


_RAW = {
    "timestamp": [_close_epoch(d) for d in ("2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16")],
    "equity": [81660.62, 81659.90, 81665.36, 81666.08],
    "profit_loss": [-2.20, -0.72, 5.46, 0.72],
    "profit_loss_pct": [0.0, 0.0, 0.0001, 0.0],
    "base_value": 81662.2,
    "timeframe": "1D",
}


class _FakeClient:
    def __init__(self, raw: Any = None, error: Exception | None = None) -> None:
        self.raw = _RAW if raw is None else raw
        self.error = error
        self.calls: list[Any] = []

    def get_portfolio_history(self, history_filter: Any = None) -> Any:
        self.calls.append(history_filter)
        if self.error is not None:
            raise self.error
        return self.raw


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _clear_cache():
    ph.clear_portfolio_history_cache()
    yield
    ph.clear_portfolio_history_cache()


def test_normalise_maps_broker_timestamps_to_new_york_session_dates() -> None:
    history = ph.normalize_portfolio_history(
        SimpleNamespace(**_RAW), period="1M", timeframe="1D", fetched_at="2026-10-19T03:00:00+00:00"
    )

    assert history["available"] is True
    assert history["evidence_class"] == "broker_reported_portfolio_history"
    assert [row["date"] for row in history["rows"]] == [
        "2026-10-13",
        "2026-10-14",
        "2026-10-15",
        "2026-10-16",
    ]
    last = history["rows"][-1]
    # 20:00 EDT on the 16th is 00:00 UTC on the 17th; the session date stays the 16th.
    assert last["broker_timestamp_utc"] == "2026-10-17T00:00:00+00:00"
    assert last["equity"] == 81666.08
    assert last["profit_loss"] == 0.72
    assert last["profit_loss_pct"] == 0.0
    assert last["base_value"] == 81662.2
    assert history["fetched_at"] == "2026-10-19T03:00:00+00:00"


def test_normalise_empty_series_is_unavailable() -> None:
    history = ph.normalize_portfolio_history({}, period="1M", timeframe="1D", fetched_at="x")
    assert history["available"] is False
    assert history["reason"] == "empty_broker_series"


def test_fetch_requests_daily_period_and_caches_by_period_and_timeframe(tmp_path: Path) -> None:
    client = _FakeClient()
    clock = _Clock(1_000_000.0)
    cache_path = tmp_path / "cache.json"

    first = ph.fetch_portfolio_history(client=client, cache_path=cache_path, clock=clock)
    clock.now += 60
    second = ph.fetch_portfolio_history(client=client, cache_path=cache_path, clock=clock)

    assert len(client.calls) == 1
    assert client.calls[0].period == "1M"
    assert client.calls[0].timeframe == "1D"
    assert first["cache"]["source"] == "broker"
    assert second["cache"] == {"source": "memory", "age_s": 60.0}
    assert second["rows"] == first["rows"]

    # A different key is a separate fetch.
    ph.fetch_portfolio_history(period="1W", client=client, cache_path=cache_path, clock=clock)
    assert len(client.calls) == 2


def test_file_cache_serves_a_fresh_process_until_ttl(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    clock = _Clock(2_000_000.0)
    ph.fetch_portfolio_history(client=_FakeClient(), cache_path=cache_path, clock=clock)
    ph.clear_portfolio_history_cache()  # a new dispatcher process

    client = _FakeClient()
    clock.now += 120
    from_file = ph.fetch_portfolio_history(client=client, cache_path=cache_path, clock=clock)
    assert client.calls == []
    assert from_file["cache"]["source"] == "file"

    ph.clear_portfolio_history_cache()
    clock.now += ph.CACHE_TTL_SECONDS
    refreshed = ph.fetch_portfolio_history(client=client, cache_path=cache_path, clock=clock)
    assert len(client.calls) == 1
    assert refreshed["cache"]["source"] == "broker"


def test_fetch_failure_never_raises_and_is_not_cached(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    failing = _FakeClient(error=RuntimeError("401 unauthorized"))

    result = ph.fetch_portfolio_history(client=failing, cache_path=cache_path, clock=_Clock(10.0))

    assert result["available"] is False
    assert result["reason"] == "broker_fetch_failed"
    assert "RuntimeError" in result["error"]
    assert result["evidence_class"] == "broker_reported_portfolio_history"
    assert not cache_path.exists()

    recovered = ph.fetch_portfolio_history(client=_FakeClient(), cache_path=cache_path, clock=_Clock(11.0))
    assert recovered["available"] is True


def test_default_client_failure_is_reported(monkeypatch) -> None:
    def _boom() -> Any:
        raise RuntimeError("Missing Alpaca credentials")

    monkeypatch.setattr(ph, "_default_trading_client", _boom)
    result = ph.fetch_portfolio_history(clock=_Clock(10.0))
    assert result["available"] is False
    assert "Missing Alpaca credentials" in result["error"]


def _history() -> dict[str, Any]:
    return ph.normalize_portfolio_history(_RAW, period="1M", timeframe="1D", fetched_at="t")


def test_daily_crosscheck_reports_broker_change_next_to_internal_pnl() -> None:
    section = ph.daily_crosscheck(_history(), "2026-10-16", internal_pnl=0.76, internal_status="ready")

    assert section["available"] is True
    assert section["reported_only"] is True
    assert section["evidence_class"] == "broker_reported_portfolio_history"
    assert section["internal_evidence_class"] == "internal_same_day_fill_round_trips"
    assert section["broker_equity_start"] == 81665.36
    assert section["broker_equity_end"] == 81666.08
    assert section["broker_daily_pnl"] == 0.72
    assert section["broker_daily_pnl_pct"] == 0.0
    assert section["broker_daily_pnl_bps_of_start_equity"] == pytest.approx(0.72 / 81665.36 * 1e4)
    assert section["internal_same_day_fill_pnl"] == 0.76
    assert section["difference_usd"] == pytest.approx(-0.04)
    assert section["difference_bps_of_start_equity"] == pytest.approx(-0.04 / 81665.36 * 1e4)
    assert "unrealised" in section["basis_note"]


def test_daily_crosscheck_first_row_derives_start_from_broker_pnl() -> None:
    section = ph.daily_crosscheck(_history(), "2026-10-13", internal_pnl=None, internal_status="no_same_day_fills")
    assert section["broker_equity_start"] == pytest.approx(81662.82)
    assert section["difference_usd"] is None


def test_daily_crosscheck_unavailable_paths() -> None:
    missing_day = ph.daily_crosscheck(_history(), "2026-10-19", internal_pnl=1.0, internal_status="ready")
    assert missing_day["available"] is False
    assert missing_day["reason"] == "report_date_not_in_broker_series"
    assert missing_day["broker_latest_date"] == "2026-10-16"

    failed = ph.daily_crosscheck(
        {"available": False, "reason": "broker_fetch_failed", "error": "x"},
        "2026-10-16",
        internal_pnl=0.76,
        internal_status="ready",
    )
    assert failed["available"] is False
    assert failed["reason"] == "broker_fetch_failed"
    assert failed["internal_same_day_fill_pnl"] == 0.76


def test_weekly_crosscheck_series_difference_and_partial_sessions() -> None:
    report = ph.weekly_crosscheck(
        _history(),
        {"2026-10-14": -0.50, "2026-10-16": 0.76},
        start_date=date(2026, 10, 13),
        end_date=date(2026, 10, 16),
        partial_sessions=["2026-10-13"],
    )

    assert report["available"] is True
    assert report["reported_only"] is True
    assert report["broker_equity_start"] == pytest.approx(81662.82)
    assert report["broker_equity_end"] == 81666.08
    assert report["broker_weekly_equity_change"] == pytest.approx(81666.08 - 81662.82)
    daily = {row["date"]: row for row in report["daily"]}
    assert daily["2026-10-13"]["internal_same_day_fill_pnl"] is None
    assert daily["2026-10-13"]["internal_partial_window"] is True
    assert daily["2026-10-15"]["internal_same_day_fill_pnl"] == 0.0  # no round trips that day
    assert daily["2026-10-16"]["difference_usd"] == pytest.approx(-0.04)
    assert report["comparable_sessions"] == 3
    assert report["broker_comparable_pnl"] == pytest.approx(-0.72 + 5.46 + 0.72)
    assert report["internal_comparable_pnl"] == pytest.approx(0.26)
    assert report["weekly_difference_usd"] == pytest.approx(5.46 - 0.26)


def test_weekly_crosscheck_without_fill_source_has_no_internal_figures() -> None:
    report = ph.weekly_crosscheck(
        _history(), {}, start_date=date(2026, 10, 13), end_date=date(2026, 10, 16), internal_available=False
    )
    assert report["internal_weekly_same_day_fill_pnl"] is None
    assert all(row["internal_same_day_fill_pnl"] is None for row in report["daily"])
    assert report["weekly_difference_usd"] is None


def test_internal_pnl_by_session_uses_new_york_days_and_fifo() -> None:
    rows = [
        {"ts": "2026-10-15T14:00:00+00:00", "symbol": "AAPL", "side": "buy", "fill_qty": 2, "fill_price": 100.0},
        {"ts": "2026-10-15T15:00:00+00:00", "symbol": "AAPL", "side": "sell", "fill_qty": 2, "fill_price": 101.0},
        {"ts": "2026-10-16T15:30:00+00:00", "symbol": "MSFT", "side": "buy", "fill_qty": 1, "fill_price": 50.0},
        # 00:30 UTC on the 17th is the 16th's session in New York.
        {"ts": "2026-10-17T00:30:00+00:00", "symbol": "MSFT", "side": "sell", "fill_qty": 1, "fill_price": 48.0},
        {"ts": "not-a-time", "symbol": "MSFT", "side": "sell", "fill_qty": 1, "fill_price": 48.0},
    ]
    assert ph.internal_same_day_fill_pnl_by_session(rows) == {"2026-10-15": 2.0, "2026-10-16": -2.0}


def test_round_trips_track_open_lots() -> None:
    trips = ph.same_day_round_trips(
        [
            {"symbol": "QQQ", "side": "buy", "qty": 3.0, "price": 10.0},
            {"symbol": "QQQ", "side": "sell", "qty": 1.0, "price": 12.0},
        ]
    )
    assert trips["pnl_by_symbol"] == {"QQQ": 2.0}
    assert trips["open_qty_by_symbol"] == {"QQQ": 2.0}
    assert trips["closed_trades"] == 1


def test_fetched_at_is_utc_iso(tmp_path: Path) -> None:
    now = datetime(2026, 10, 19, 3, 0, tzinfo=UTC).timestamp()
    result = ph.fetch_portfolio_history(client=_FakeClient(), clock=_Clock(now))
    assert result["fetched_at"] == "2026-10-19T03:00:00+00:00"
