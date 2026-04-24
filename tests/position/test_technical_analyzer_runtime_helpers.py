from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest

from ai_trading.position.technical_analyzer import (
    DivergenceType,
    SignalStrength,
    TechnicalSignalAnalyzer,
)


def _market_frame(rows: int = 80, *, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    close = [start + i * step for i in range(rows)]
    return pd.DataFrame(
        {
            "close": close,
            "high": [price + 2.0 for price in close],
            "low": [price - 2.0 for price in close],
            "volume": [1000 + i * 10 for i in range(rows)],
        }
    )


def test_analyze_signals_returns_default_when_data_missing() -> None:
    signals = TechnicalSignalAnalyzer().analyze_signals("AAPL")

    assert signals.symbol == "AAPL"
    assert signals.hold_recommendation is SignalStrength.NEUTRAL
    assert signals.divergence_type is DivergenceType.NONE
    assert signals.exit_urgency == 0.0


def test_analyze_signals_composes_indicator_helpers() -> None:
    class _Fetcher:
        def get_minute_df(self, _ctx, symbol):
            if symbol == "SPY":
                return _market_frame(start=100.0, step=0.05)
            return _market_frame(start=100.0, step=0.8)

        def get_daily_df(self, _ctx, symbol):
            return self.get_minute_df(_ctx, symbol)

    ctx = SimpleNamespace(data_fetcher=_Fetcher())
    signals = TechnicalSignalAnalyzer(ctx).analyze_signals("AAPL")

    assert signals.symbol == "AAPL"
    assert signals.momentum_score > 0.5
    assert signals.volume_strength > 0.5
    assert signals.relative_strength_score > 0.5
    assert signals.distance_to_support > 0
    assert signals.hold_recommendation in {SignalStrength.STRONG, SignalStrength.VERY_STRONG}


def test_momentum_defaults_for_short_or_malformed_data() -> None:
    analyzer = TechnicalSignalAnalyzer()

    assert analyzer._analyze_momentum(pd.DataFrame({"open": [1.0, 2.0]})) == {  # noqa: SLF001
        "score": 0.5,
        "rsi": 50.0,
        "macd": 0.0,
    }
    assert analyzer._analyze_momentum(pd.DataFrame({"close": ["bad"] * 30})) == {  # noqa: SLF001
        "score": 0.5,
        "rsi": 50.0,
        "macd": 0.0,
    }


def test_divergence_detects_bearish_and_bullish_patterns(monkeypatch) -> None:
    analyzer = TechnicalSignalAnalyzer()
    analyzer.momentum_period = 3
    analyzer.divergence_lookback = 4
    bearish_rsi = iter([80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0])
    monkeypatch.setattr(analyzer, "_calculate_rsi", lambda *_args: next(bearish_rsi))

    bearish = analyzer._analyze_divergence(pd.DataFrame({"close": [10, 11, 12, 13, 14, 15, 16, 17, 18]}))  # noqa: SLF001

    assert bearish["type"] is DivergenceType.BEARISH
    assert bearish["strength"] > 0

    bullish_rsi = iter([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    monkeypatch.setattr(analyzer, "_calculate_rsi", lambda *_args: next(bullish_rsi))

    bullish = analyzer._analyze_divergence(pd.DataFrame({"close": [18, 17, 16, 15, 14, 13, 12, 11, 10]}))  # noqa: SLF001

    assert bullish["type"] is DivergenceType.BULLISH
    assert bullish["strength"] > 0


def test_volume_analysis_covers_confirmation_and_defaults() -> None:
    analyzer = TechnicalSignalAnalyzer()
    increasing = _market_frame(rows=30, start=100.0, step=1.0)
    decreasing = _market_frame(rows=30, start=130.0, step=-1.0)
    decreasing["volume"] = list(range(2000, 1970, -1))

    strong = analyzer._analyze_volume(increasing)  # noqa: SLF001
    weak = analyzer._analyze_volume(decreasing)  # noqa: SLF001
    default = analyzer._analyze_volume(pd.DataFrame({"close": [1.0, 2.0]}))  # noqa: SLF001

    assert strong["trend"] == "increasing"
    assert strong["strength"] > 0.5
    assert weak["trend"] == "decreasing"
    assert default == {"strength": 0.5, "trend": "neutral"}


def test_relative_strength_uses_market_data_and_fallback(monkeypatch) -> None:
    analyzer = TechnicalSignalAnalyzer()
    symbol_data = _market_frame(rows=30, start=100.0, step=1.0)
    spy_data = _market_frame(rows=30, start=100.0, step=0.1)
    monkeypatch.setattr(analyzer, "_get_market_data", lambda symbol: spy_data if symbol == "SPY" else None)

    relative = analyzer._analyze_relative_strength("AAPL", symbol_data)  # noqa: SLF001

    assert relative["score"] > 0.5
    assert relative["relative_strength"] > 0

    monkeypatch.setattr(analyzer, "_get_market_data", lambda _symbol: None)
    fallback = analyzer._analyze_relative_strength("AAPL", pd.DataFrame({"close": [1.0] * 30}))  # noqa: SLF001
    assert fallback["score"] == pytest.approx(0.5)


def test_support_resistance_distances_and_default() -> None:
    analyzer = TechnicalSignalAnalyzer()
    data = _market_frame(rows=60, start=100.0, step=1.0)

    levels = analyzer._analyze_support_resistance(data)  # noqa: SLF001
    default = analyzer._analyze_support_resistance(pd.DataFrame({"close": [1.0] * 10}))  # noqa: SLF001

    assert levels["support_distance"] > 0
    assert levels["resistance_distance"] > 0
    assert levels["confidence"] == pytest.approx(2 / 3)
    assert levels["support_levels"]
    assert default == {
        "support_distance": 10.0,
        "resistance_distance": 10.0,
        "confidence": 0.5,
    }


@pytest.mark.parametrize(
    ("momentum", "divergence", "volume", "relative", "sr", "expected"),
    [
        (
            {"score": 0.8},
            {"type": DivergenceType.BULLISH, "strength": 0.8},
            {"strength": 0.9, "trend": "increasing"},
            {"score": 0.8},
            {"support_distance": 5.0, "resistance_distance": 5.0},
            SignalStrength.VERY_STRONG,
        ),
        (
            {"score": 0.2},
            {"type": DivergenceType.BEARISH, "strength": 0.9},
            {"strength": 0.2, "trend": "decreasing"},
            {"score": 0.2},
            {"support_distance": 5.0, "resistance_distance": 1.0},
            SignalStrength.VERY_WEAK,
        ),
        (
            {"score": 0.55},
            {"type": DivergenceType.NONE, "strength": 0.0},
            {"strength": 0.5, "trend": "neutral"},
            {"score": 0.5},
            {"support_distance": 5.0, "resistance_distance": 5.0},
            SignalStrength.NEUTRAL,
        ),
    ],
)
def test_hold_recommendation_thresholds(momentum, divergence, volume, relative, sr, expected) -> None:
    analyzer = TechnicalSignalAnalyzer()

    assert analyzer._calculate_hold_recommendation(momentum, divergence, volume, relative, sr) is expected  # noqa: SLF001


def test_exit_urgency_and_trend_helpers() -> None:
    analyzer = TechnicalSignalAnalyzer()

    urgency = analyzer._calculate_exit_urgency(  # noqa: SLF001
        {"score": 0.25},
        {"type": DivergenceType.BEARISH, "strength": 0.9},
        {"trend": "increasing"},
    )

    assert urgency == pytest.approx(0.9)
    assert analyzer._calculate_exit_urgency({"score": 0.6}, {}, {}) == 0.0  # noqa: SLF001
    assert analyzer._calculate_trend([1.0, 2.0]) == 0.0  # noqa: SLF001
    assert analyzer._calculate_trend([1.0, 2.0, 3.0, 4.0]) > 0.0  # noqa: SLF001
    assert analyzer._calculate_trend([4.0, 3.0, 2.0, 1.0]) < 0.0  # noqa: SLF001
    assert analyzer._calculate_trend(cast(list[float], ["bad", object(), None])) == 0.0  # noqa: SLF001


def test_get_market_data_prefers_minute_then_daily_and_handles_errors() -> None:
    daily = _market_frame(rows=20)

    class _Fetcher:
        def get_minute_df(self, _ctx, _symbol):
            return pd.DataFrame()

        def get_daily_df(self, _ctx, _symbol):
            return daily

    assert TechnicalSignalAnalyzer(SimpleNamespace(data_fetcher=_Fetcher()))._get_market_data("AAPL") is daily  # noqa: SLF001

    class _FailingFetcher:
        def get_minute_df(self, _ctx, _symbol):
            raise AttributeError("missing")

    assert TechnicalSignalAnalyzer(SimpleNamespace(data_fetcher=_FailingFetcher()))._get_market_data("AAPL") is None  # noqa: SLF001


def test_indicator_helpers_handle_short_and_valid_series() -> None:
    analyzer = TechnicalSignalAnalyzer()
    prices = pd.Series([float(i) for i in range(1, 60)])

    rsi = analyzer._calculate_rsi(prices, 14)  # noqa: SLF001
    macd_line, macd_signal = analyzer._calculate_macd(prices)  # noqa: SLF001

    assert 0 <= rsi <= 100
    assert macd_line != 0
    assert macd_signal != 0
    assert analyzer._calculate_rsi([1.0, 2.0], 14) == 50.0  # noqa: SLF001
    assert analyzer._calculate_macd(pd.Series([1.0, 2.0]), slow=26, signal=9) == (0.0, 0.0)  # noqa: SLF001
