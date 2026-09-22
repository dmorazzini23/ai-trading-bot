from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ai_trading.core import bot_engine


class _EmptyDailyFetcher:
    def __init__(self) -> None:
        self.calls = 0

    def get_daily_df(self, ctx: object, symbol: str) -> None:
        self.calls += 1
        return None


class _UnavailableFetcher:
    def __init__(self) -> None:
        self.calls = 0

    def get_daily_df(self, ctx: object, symbol: str) -> None:
        self.calls += 1
        raise bot_engine.DataFetchError("DATA_FETCHER_UNAVAILABLE")


def _reset_vol_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("mean", "std", "last", "last_update"):
        monkeypatch.setitem(bot_engine._VOL_STATS, key, None)


def _make_runtime(fetcher) -> SimpleNamespace:
    return SimpleNamespace(data_fetcher=fetcher, halt_manager=None)


def test_spy_vol_fetch_aborts_on_empty_response(monkeypatch: pytest.MonkeyPatch, caplog):
    _reset_vol_stats(monkeypatch)
    fetcher = _EmptyDailyFetcher()
    runtime = _make_runtime(fetcher)
    caplog.set_level(logging.WARNING)

    with pytest.raises(bot_engine.DataFetchError):
        bot_engine.compute_spy_vol_stats(runtime)

    assert fetcher.calls == 1
    abort_logs = [rec for rec in caplog.records if rec.message == "SPY_VOL_FETCH_ABORT"]
    assert abort_logs, "expected abort warning for empty daily data"
    assert (
        getattr(abort_logs[0], "hint", "")
        == "historical data not available; manual backfill required"
    )


def test_spy_vol_fetch_abort_on_data_fetcher_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    _reset_vol_stats(monkeypatch)
    fetcher = _UnavailableFetcher()
    runtime = _make_runtime(fetcher)
    caplog.set_level(logging.WARNING)

    with pytest.raises(bot_engine.DataFetchError):
        bot_engine.compute_spy_vol_stats(runtime)

    assert fetcher.calls == 1
    abort_logs = [rec for rec in caplog.records if rec.message == "SPY_VOL_FETCH_ABORT"]
    assert abort_logs, "expected abort warning when data fetcher is unavailable"
    assert (
        getattr(abort_logs[0], "hint", "")
        == "historical data not available; manual backfill required"
    )
