from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.core.enums import RiskLevel
from ai_trading.strategies.multi_factor_quality_value import (
    MultiFactorQualityValueStrategy,
    _zscore,
)


def _series(start: float, stop: float, rows: int = 70) -> Any:
    return pd.Series(np.linspace(start, stop, rows), index=pd.date_range("2026-01-01", periods=rows, freq="D"))


def test_zscore_empty_flat_and_scaled_values() -> None:
    assert _zscore({}) == {}
    assert _zscore({"A": 2.0, "B": 2.0}) == {"A": 0.0, "B": 0.0}

    scored = _zscore({"LOW": 1.0, "MID": 2.0, "HIGH": 3.0})

    assert scored["LOW"] < 0.0
    assert scored["MID"] == pytest.approx(0.0)
    assert scored["HIGH"] > 0.0


def test_strategy_clamps_configuration_and_rejects_bad_market_data() -> None:
    strategy = MultiFactorQualityValueStrategy(
        risk_level=RiskLevel.AGGRESSIVE,
        lookback=5,
        top_quantile=0.99,
        min_universe=1,
    )

    assert strategy.lookback == 30
    assert strategy.top_quantile == 0.45
    assert strategy.min_universe == 6
    assert strategy.generate_signals({"closes": []}) == []
    assert strategy.generate_signals({}) == []


def test_generate_signals_ranks_winners_and_losers() -> None:
    strategy = MultiFactorQualityValueStrategy(lookback=30, top_quantile=0.25, min_universe=8)
    closes = {
        "AAA": _series(100.0, 150.0),
        "BBB": _series(100.0, 140.0),
        "CCC": _series(100.0, 130.0),
        "DDD": _series(100.0, 120.0),
        "EEE": _series(100.0, 110.0),
        "FFF": _series(100.0, 90.0),
        "GGG": _series(100.0, 80.0),
        "HHH": _series(100.0, 70.0),
    }

    signals = strategy.generate_signals({"closes": closes})
    by_side = {signal.side: [] for signal in signals}
    for signal in signals:
        by_side[signal.side].append(signal)

    assert len(signals) == 4
    assert {signal.symbol for signal in by_side["buy"]} == {"DDD", "EEE"}
    assert {signal.symbol for signal in by_side["sell_short"]} == {"GGG", "HHH"}
    assert all(signal.metadata["factor_composite"] > 0.0 for signal in by_side["buy"])
    assert all(signal.metadata["factor_composite"] < 0.0 for signal in by_side["sell_short"])
    assert all(signal.strategy_id == strategy.strategy_id for signal in signals)
    assert all(signal.metadata["expected_edge_bps"] >= 2.0 for signal in signals)
    assert all(0.05 <= signal.strength <= 1.0 for signal in signals)
    assert all(0.55 <= signal.confidence <= 0.95 for signal in signals)


def test_generate_skips_short_bad_or_nonfinite_series() -> None:
    strategy = MultiFactorQualityValueStrategy(lookback=30, min_universe=6)
    closes = {
        "GOOD1": _series(100, 130),
        "GOOD2": _series(100, 125),
        "GOOD3": _series(100, 120),
        "GOOD4": _series(100, 115),
        "GOOD5": _series(100, 110),
        "GOOD6": _series(100, 105),
        "SHORT": _series(1, 2, rows=10),
        "ZERO": _series(0, 10),
        "BAD": object(),
        "INF": pd.Series([100.0] * 69 + [float("inf")]),
    }

    signals = strategy.generate_signals({"closes": closes})
    symbols = {signal.symbol for signal in signals}

    assert signals
    assert "BAD" not in symbols
    assert "INF" not in symbols


def test_load_daily_closes_filters_context_inputs_and_generate() -> None:
    frames = {
        "AAPL": pd.DataFrame({"close": _series(100, 130)}),
        "MSFT": pd.DataFrame({"close": _series(100, 125)}),
        "TSLA": pd.DataFrame({"open": _series(100, 125)}),
        "SHORT": pd.DataFrame({"close": _series(1, 2, rows=10)}),
    }

    class Fetcher:
        def get_daily_df(self, _ctx: Any, symbol: str) -> Any:
            if symbol == "ERR":
                raise ValueError("bad symbol")
            return frames.get(symbol)

    strategy = MultiFactorQualityValueStrategy(lookback=30, min_universe=6)
    ctx = SimpleNamespace(data_fetcher=Fetcher(), tickers=[" aapl ", "msft", "tsla", "short", "err", ""])

    closes = strategy._load_daily_closes(ctx)  # noqa: SLF001

    assert set(closes) == {"AAPL", "MSFT"}
    assert strategy.generate(SimpleNamespace(data_fetcher=None, tickers=["AAPL"])) == []
    assert strategy.generate(ctx) == []
