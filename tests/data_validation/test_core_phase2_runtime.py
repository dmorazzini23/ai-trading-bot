from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data_validation import core


def _ohlcv(index: Any | None = None) -> Any:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [12.0, 12.0],
            "low": [9.0, 10.0],
            "close": [11.0, 11.5],
            "volume": [100, 200],
        },
        index=index,
    )


def test_market_hours_thresholds_and_ohlcv_validation() -> None:
    weekday_open = datetime(2026, 4, 24, 14, 30, tzinfo=UTC)
    weekday_after_close = datetime(2026, 4, 24, 21, 0, tzinfo=UTC)
    weekend = datetime(2026, 4, 25, 16, 0, tzinfo=UTC)

    assert core.is_market_hours(weekday_open) is True
    assert core.is_market_hours(weekday_after_close) is False
    assert core.is_market_hours(weekend) is False
    assert core.is_market_hours(datetime(2026, 4, 24, 10, 0)) is False
    assert core.get_staleness_threshold("AAPL", weekday_open) == 15
    assert core.get_staleness_threshold("AAPL", weekday_after_close) == 120
    assert core.get_staleness_threshold("AAPL", weekend) == 360
    assert core.is_valid_ohlcv(_ohlcv(), min_rows=2) is True
    assert core.is_valid_ohlcv(_ohlcv().drop(columns=["volume"]), min_rows=2) is False
    assert core.is_valid_ohlcv(pd.DataFrame(), min_rows=1) is False


def test_freshness_stale_symbols_and_validate_trading_data(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 4, 24, 15, 0, tzinfo=UTC)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
            return now if tz is not None else now.replace(tzinfo=None)

    monkeypatch.setattr(core, "datetime", FixedDatetime)
    fresh = _ohlcv(pd.DatetimeIndex([now - timedelta(minutes=2), now - timedelta(minutes=1)]))
    stale = _ohlcv(pd.DatetimeIndex([now - timedelta(hours=1), now - timedelta(minutes=30)]))
    epoch_index = pd.Index([now.timestamp() - 120, now.timestamp() - 60])

    fresh_info = core.check_data_freshness(fresh, "AAPL", max_staleness_minutes=5)
    stale_info = core.check_data_freshness(stale, "MSFT", max_staleness_minutes=5)
    epoch_info = core.check_data_freshness(_ohlcv(epoch_index), "TSLA", max_staleness_minutes=5)
    bad_info = core.check_data_freshness(
        _ohlcv(pd.Index(["not-a-date", "still-bad"])),
        "BAD",
        max_staleness_minutes=5,
    )

    assert fresh_info["is_fresh"] is True
    assert stale_info["is_fresh"] is False
    assert epoch_info["is_fresh"] is True
    assert bad_info["minutes_stale"] == float("inf")
    assert core.get_stale_symbols(
        {"AAPL": {"trading_ready": True}, "MSFT": {"is_fresh": False}, "TSLA": stale},
        max_staleness_minutes=5,
    ) == ["MSFT", "TSLA"]
    assert core.validate_trading_data({"AAPL": fresh}, max_staleness_minutes=5)["AAPL"][
        "trading_ready"
    ] is True


def test_emergency_data_check_and_market_data_validator() -> None:
    good = pd.DataFrame({"close": [1.0, 2.0]})
    bad = pd.DataFrame({"close": [1.0, -2.0]})
    calls: list[str] = []

    def fetcher(symbol: str, *_args: Any) -> Any:
        calls.append(symbol)
        if symbol == "BAD":
            raise ValueError("provider")
        if symbol == "GOOD":
            return good
        return pd.DataFrame()

    assert core.emergency_data_check(good, "AAPL") is True
    assert core.emergency_data_check(bad, "AAPL") is False
    assert core.emergency_data_check(b"BAD", fetcher=fetcher) is False
    assert core.emergency_data_check(["BAD", "GOOD"], fetcher=fetcher) is True
    assert calls == ["BAD", "BAD", "GOOD"]

    validator = core.MarketDataValidator()
    assert validator.validate_ohlc_data(_ohlcv(), "AAPL").severity is core.ValidationSeverity.INFO
    broken = _ohlcv()
    broken.loc[0, "high"] = 1.0
    assert validator.validate_ohlc_data(broken, "AAPL").is_valid is False
    assert validator.validate_ohlc_data(pd.DataFrame(), "AAPL").data_quality_score == 0.0
    assert validator.positive_prices(pd.DataFrame({"close": [1.0, -1.0]}))["close"].tolist() == [
        1.0
    ]
    assert core.monitor_real_time_data_quality({"AAPL": 1.0, "BAD": 0.0}) == {
        "data_quality_ok": False,
        "critical_symbols": ["BAD"],
        "anomalies_detected": ["BAD"],
    }


def test_validate_trade_log_integrity_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = core.validate_trade_log_integrity(tmp_path / "missing.csv")
    assert missing["file_exists"] is False

    invalid = tmp_path / "invalid.csv"
    invalid.write_text("symbol,pnl\nAAPL,1\n")
    invalid_report = core.validate_trade_log_integrity(invalid)
    assert invalid_report["file_readable"] is True
    assert invalid_report["valid_format"] is False

    valid = tmp_path / "trades.csv"
    valid.write_text(
        "timestamp,symbol,side,entry_price,exit_price,quantity,pnl\n"
        "2026-01-01,AAPL,buy,100,101,2,2\n"
        "bad,MSFT,buy,nope,101,qty,1\n"
    )
    report = core.validate_trade_log_integrity(valid)
    assert report["total_trades"] == 2
    assert report["corrupted_rows"] == [1]
    assert report["integrity_score"] == 0.5

    monkeypatch.setattr(core.pd, "read_csv", lambda _path: (_ for _ in ()).throw(ValueError("bad csv")))
    unreadable = core.validate_trade_log_integrity(valid)
    assert unreadable["file_readable"] is False
