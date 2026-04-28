from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.strategies.cross_sectional_momentum import (
    CrossSectionalMomentumStrategy,
    _zscore_map,
)
from ai_trading.strategies.low_beta_defensive import LowBetaDefensiveStrategy
from ai_trading.strategies.pairs_stat_arb import PairsStatArbStrategy
from ai_trading.strategies.pead_event import PEADEventStrategy
from ai_trading.strategies.time_series_momentum_overlay import (
    TimeSeriesMomentumOverlayStrategy,
)


def _series(start: float, stop: float, rows: int = 140) -> Any:
    return pd.Series(np.linspace(start, stop, rows), index=pd.date_range("2026-01-01", periods=rows, freq="D"))


class _Fetcher:
    def __init__(self, frames: dict[str, Any]) -> None:
        self.frames = frames

    def get_daily_df(self, _ctx: Any, symbol: str) -> Any:
        if symbol == "ERR":
            raise ValueError("bad symbol")
        return self.frames.get(symbol)


def test_cross_sectional_momentum_ranks_tails_and_loads_prices() -> None:
    assert _zscore_map({}) == {}
    assert _zscore_map({"A": 1.0, "B": 1.0}) == {"A": 0.0, "B": 0.0}
    strategy = CrossSectionalMomentumStrategy(lookback=20, top_quantile=0.25, min_universe=4)
    prices = {
        "AAA": _series(100, 160, 60),
        "BBB": _series(100, 140, 60),
        "CCC": _series(100, 80, 60),
        "DDD": _series(100, 60, 60),
        "ZERO": pd.Series([0.0] * 60),
        "BAD": object(),
    }

    signals = strategy.generate_signals({"prices": prices})

    assert {(signal.symbol, signal.side) for signal in signals} == {
        ("AAA", "buy"),
        ("DDD", "sell_short"),
    }
    assert all(signal.metadata["expected_edge_bps"] >= 2.0 for signal in signals)

    frames = {
        "AAA": pd.DataFrame({"close": _series(100, 160, 60)}),
        "SHORT": pd.DataFrame({"close": _series(100, 110, 5)}),
        "NOCLOSE": pd.DataFrame({"open": [1.0, 2.0]}),
    }
    ctx = SimpleNamespace(data_fetcher=_Fetcher(frames), tickers=["AAA", "SHORT", "NOCLOSE", "ERR"])
    assert set(strategy._load_prices(ctx)) == {"AAA"}  # noqa: SLF001
    assert strategy.generate(SimpleNamespace(data_fetcher=None, tickers=["AAA"])) == []


def test_pairs_stat_arb_emits_positive_and_negative_spread_pairs() -> None:
    strategy = PairsStatArbStrategy(lookback=30, z_entry=0.5, min_universe=4)
    base = np.linspace(-0.01, 0.01, 30)
    returns = {
        "AAA": np.r_[base[:-1], 0.08],
        "BBB": np.r_[base[:-1], -0.08],
        "CCC": np.sin(np.linspace(0.0, 3.0, 30)) * 0.003,
        "DDD": np.cos(np.linspace(0.0, 3.0, 30)) * 0.003,
    }

    wide = strategy.generate_signals({"returns": returns})
    narrow = strategy.generate_signals({"returns": {**returns, "AAA": np.r_[base[:-1], -0.08], "BBB": np.r_[base[:-1], 0.08]}})

    assert {(signal.symbol, signal.side) for signal in wide} == {
        ("AAA", "sell_short"),
        ("BBB", "buy"),
    }
    assert {(signal.symbol, signal.side) for signal in narrow} == {
        ("AAA", "buy"),
        ("BBB", "sell_short"),
    }
    assert strategy.generate_signals({"returns": {"A": base}}) == []
    assert PairsStatArbStrategy(z_entry=99, min_universe=1).z_entry == 4.0

    prices = np.cumprod(1.0 + np.r_[0.0, base, base]) * 100.0
    frames = {
        "AAA": pd.DataFrame({"close": prices}),
        "BBB": pd.DataFrame({"close": prices * 1.01}),
        "BAD": pd.DataFrame({"open": prices}),
        "ERR": None,
    }
    ctx = SimpleNamespace(data_fetcher=_Fetcher(frames), tickers=["AAA", "BBB", "BAD", "ERR"])
    assert set(PairsStatArbStrategy(lookback=20, min_universe=2)._load_returns(ctx)) == {"AAA", "BBB"}  # noqa: SLF001


def test_low_beta_defensive_generates_only_under_stress_and_loads_daily() -> None:
    strategy = LowBetaDefensiveStrategy(beta_lookback=30, stress_vol_threshold=0.002)
    proxy = pd.Series(100.0 * np.cumprod(1.0 + np.tile([0.03, -0.025], 35)))
    series = {
        "LOW": pd.Series(100.0 * np.cumprod(1.0 + np.tile([0.01, -0.008], 35))),
        "MID": pd.Series(100.0 * np.cumprod(1.0 + np.tile([0.02, -0.016], 35))),
        "HIGH": pd.Series(100.0 * np.cumprod(1.0 + np.tile([0.04, -0.035], 35))),
        "INV": pd.Series(100.0 * np.cumprod(1.0 + np.tile([-0.01, 0.008], 35))),
    }

    signals = strategy.generate_signals({"series": series, "market_proxy": proxy})

    assert len(signals) == 1
    assert signals[0].side == "buy"
    assert signals[0].metadata["market_stress_vol"] >= strategy.stress_vol_threshold
    assert LowBetaDefensiveStrategy(stress_vol_threshold=1.0).stress_vol_threshold == 0.08
    assert strategy.generate_signals({"series": series, "market_proxy": pd.Series([100.0] * 70)}) == []

    frames = {
        "SPY": pd.DataFrame({"close": proxy}),
        "LOW": pd.DataFrame({"close": series["LOW"]}),
        "BAD": pd.DataFrame({"open": series["LOW"]}),
    }
    ctx = SimpleNamespace(data_fetcher=_Fetcher(frames), tickers=["LOW", "BAD", "ERR"])
    loaded, loaded_proxy = strategy._load_daily(ctx)  # noqa: SLF001
    assert set(loaded) == {"LOW"}
    assert loaded_proxy is not None


def test_pead_event_detects_buy_sell_and_filters_frames() -> None:
    strategy = PEADEventStrategy(gap_threshold=0.03, volume_multiple=1.5, lookback=20)

    def frame(gap: float, follow: float, volume: float = 3_000.0) -> Any:
        rows = 24
        close = [100.0] * (rows - 1) + [100.0 * (1.0 + gap) * (1.0 + follow)]
        open_ = [100.0] * (rows - 1) + [100.0 * (1.0 + gap)]
        vol = [1_000.0] * (rows - 1) + [volume]
        return pd.DataFrame({"open": open_, "close": close, "volume": vol})

    signals = strategy.generate_signals(
        {"frames": {"up": frame(0.05, 0.01), "down": frame(-0.05, -0.01), "fade": frame(0.05, -0.01)}}
    )

    assert {(signal.symbol, signal.side) for signal in signals} == {("UP", "buy"), ("DOWN", "sell_short")}
    assert all(signal.metadata["event_volume_multiple"] >= 1.5 for signal in signals)
    assert strategy.generate_signals({"frames": []}) == []

    ctx = SimpleNamespace(
        data_fetcher=_Fetcher({"UP": frame(0.05, 0.01), "BAD": pd.DataFrame({"open": [1.0]})}),
        tickers=["up", "bad", "err"],
    )
    assert [signal.symbol for signal in strategy.generate(ctx)] == ["UP"]
    assert strategy.generate(SimpleNamespace(data_fetcher=None, tickers=["UP"])) == []


def test_time_series_momentum_overlay_trades_trends_and_loads_closes() -> None:
    strategy = TimeSeriesMomentumOverlayStrategy(fast_lookback=10, slow_lookback=30, trend_floor=0.02)
    closes = {
        "UP": _series(100, 150, 80),
        "DOWN": _series(150, 100, 80),
        "FLAT": _series(100, 101, 80),
        "BAD": object(),
        "ZERO": pd.Series([0.0] * 80),
    }

    signals = strategy.generate_signals({"closes": closes})

    assert {(signal.symbol, signal.side) for signal in signals} == {("UP", "buy"), ("DOWN", "sell")}
    assert all(signal.metadata["expected_edge_bps"] >= 2.0 for signal in signals)
    assert TimeSeriesMomentumOverlayStrategy(fast_lookback=1, slow_lookback=5).slow_lookback == 20

    frames = {
        "UP": pd.DataFrame({"close": _series(100, 150, 80)}),
        "SHORT": pd.DataFrame({"close": _series(100, 150, 5)}),
        "BAD": pd.DataFrame({"open": [1.0]}),
    }
    ctx = SimpleNamespace(data_fetcher=_Fetcher(frames), tickers=["up", "short", "bad", "err"])
    assert set(strategy._load_closes(ctx)) == {"UP"}  # noqa: SLF001
    assert strategy.generate(SimpleNamespace(data_fetcher=None, tickers=["UP"])) == []
