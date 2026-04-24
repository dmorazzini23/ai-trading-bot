from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pandas as pd
import pytest

from ai_trading.position.correlation_analyzer import (
    ConcentrationLevel,
    CorrelationStrength,
    PortfolioCorrelationAnalyzer,
    PositionCorrelation,
    SectorExposure,
)


def _price_frame(rows: int = 40, *, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"close": [start + i * step for i in range(rows)]})


def _position(symbol: str, qty: object = 10, value: object = 1000.0) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, qty=qty, market_value=value, avg_entry_price=100.0)


def test_pair_key_empty_analysis_and_sector_mapping() -> None:
    analyzer = PortfolioCorrelationAnalyzer()

    assert analyzer._pair_key("MSFT", "AAPL") == ("AAPL", "MSFT")  # noqa: SLF001
    assert analyzer.analyze_portfolio([]).total_positions == 0
    assert analyzer._get_symbol_sector("AAPL") == "Technology"  # noqa: SLF001
    assert analyzer._get_symbol_sector("JPM") == "Financials"  # noqa: SLF001
    assert analyzer._get_symbol_sector("PFE") == "Healthcare"  # noqa: SLF001
    assert analyzer._get_symbol_sector("X") == "Other"  # noqa: SLF001


def test_context_position_normalization_and_mock_filtering() -> None:
    ctx = SimpleNamespace(
        current_positions={"AAPL": "5", "BAD": "not-a-number"},
        cached_positions=[_position("MSFT", qty="3", value="900")],
        portfolio_state={"positions": {"GOOG": {"qty": 2, "market_value": 500.0}}},
    )
    analyzer = PortfolioCorrelationAnalyzer(ctx)

    collected = analyzer._collect_ctx_positions()  # noqa: SLF001
    symbols = {item["symbol"] for item in collected}

    assert {"AAPL", "MSFT", "GOOG"} <= symbols
    assert analyzer._coerce_position_payload(Mock(), symbol_hint="MOCK") is None  # noqa: SLF001
    assert analyzer._coerce_position_payload(4, symbol_hint="TSLA") == {"symbol": "TSLA", "qty": 4}  # noqa: SLF001
    assert analyzer._coerce_position_payload("6", symbol_hint="NVDA") == {"symbol": "NVDA", "qty": 6.0}  # noqa: SLF001
    assert analyzer._coerce_position_payload("bad", symbol_hint="NVDA") is None  # noqa: SLF001


def test_extract_position_data_builds_records_and_uses_price_fallback(monkeypatch) -> None:
    analyzer = PortfolioCorrelationAnalyzer()
    monkeypatch.setattr(analyzer, "_get_current_price", lambda symbol: 25.0 if symbol == "PRICE" else 0.0)

    positions = [
        {"symbol": "AAPL", "qty": "2", "market_value": "300"},
        {"symbol": "MSFT", "quantity": "3", "avg_price": "20"},
        {"symbol": "PRICE", "shares": "4"},
        {"symbol": "ZERO", "qty": 0, "market_value": 10},
        {"symbol": "", "qty": 1, "market_value": 1},
    ]

    data = analyzer._extract_position_data(positions)  # noqa: SLF001

    assert data["AAPL"]["market_value"] == 300.0
    assert data["MSFT"]["market_value"] == 60.0
    assert data["PRICE"]["market_value"] == 100.0
    assert "ZERO" not in data


def test_price_data_pair_correlation_and_cache() -> None:
    class _Fetcher:
        def get_daily_df(self, _ctx, symbol):
            if symbol == "SHORT":
                return _price_frame(rows=5)
            step = 1.0 if symbol != "MSFT" else 2.0
            return _price_frame(rows=45, start=100.0, step=step)

    analyzer = PortfolioCorrelationAnalyzer(SimpleNamespace(data_fetcher=_Fetcher()))

    returns = analyzer._get_price_data("AAPL")  # noqa: SLF001
    assert returns is not None
    assert len(returns) >= analyzer.min_data_points
    assert analyzer._get_price_data("SHORT") is None  # noqa: SLF001

    corr = analyzer._calculate_pair_correlation("AAPL", "MSFT")  # noqa: SLF001
    assert corr is not None
    assert corr.symbol1 == "AAPL"
    assert corr.symbol2 == "MSFT"
    assert corr.strength in set(CorrelationStrength)

    correlations = analyzer._calculate_position_correlations(  # noqa: SLF001
        {"AAPL": {"market_value": 1}, "MSFT": {"market_value": 1}}
    )
    assert len(correlations) == 1
    assert analyzer.get_position_correlation("MSFT", "AAPL") is correlations[0]


def test_alignment_and_correlation_strength_thresholds() -> None:
    analyzer = PortfolioCorrelationAnalyzer()

    assert analyzer._align_price_data([1.0], [1.0]) is None  # noqa: SLF001
    aligned = analyzer._align_price_data(list(range(25)), list(range(30)))  # noqa: SLF001
    assert aligned is not None
    assert len(aligned[0]) == len(aligned[1]) == 25

    assert analyzer._classify_correlation_strength(0.90) is CorrelationStrength.VERY_HIGH  # noqa: SLF001
    assert analyzer._classify_correlation_strength(0.75) is CorrelationStrength.HIGH  # noqa: SLF001
    assert analyzer._classify_correlation_strength(0.55) is CorrelationStrength.MODERATE  # noqa: SLF001
    assert analyzer._classify_correlation_strength(0.35) is CorrelationStrength.LOW  # noqa: SLF001
    assert analyzer._classify_correlation_strength(0.10) is CorrelationStrength.VERY_LOW  # noqa: SLF001


def test_sector_exposures_metrics_and_recommendations() -> None:
    analyzer = PortfolioCorrelationAnalyzer()
    analyzer.correlation_cache[("AAPL", "MSFT")] = PositionCorrelation(
        "AAPL",
        "MSFT",
        0.92,
        CorrelationStrength.VERY_HIGH,
        30,
        pd.Timestamp.utcnow().to_pydatetime(),
    )
    position_data = {
        "AAPL": {"symbol": "AAPL", "market_value": 60_000.0, "sector": "Technology"},
        "MSFT": {"symbol": "MSFT", "market_value": 30_000.0, "sector": "Technology"},
        "JPM": {"symbol": "JPM", "market_value": 10_000.0, "sector": "Financials"},
    }

    exposures = analyzer._analyze_sector_exposures(position_data)  # noqa: SLF001
    metrics = analyzer._calculate_portfolio_metrics(  # noqa: SLF001
        position_data,
        list(analyzer.correlation_cache.values()),
        exposures,
    )
    recommendations = analyzer._generate_recommendations(  # noqa: SLF001
        position_data,
        list(analyzer.correlation_cache.values()),
        exposures,
        metrics,
    )

    assert exposures[0].sector == "Technology"
    assert exposures[0].concentration_level is ConcentrationLevel.EXTREME
    assert exposures[0].avg_correlation == pytest.approx(0.92)
    assert metrics["largest_position_pct"] == pytest.approx(60.0)
    assert metrics["concentration_level"] is ConcentrationLevel.EXTREME
    assert {"AAPL", "MSFT"} <= set(recommendations["reduce_exposure"])
    assert any("largest position" in item for item in recommendations["rebalance"])
    assert any("high correlation" in item for item in recommendations["rebalance"])
    assert any("High sector concentration" in item for item in recommendations["rebalance"])


def test_concentration_classifiers_and_sector_correlation_defaults() -> None:
    analyzer = PortfolioCorrelationAnalyzer()

    assert analyzer._classify_sector_concentration(70.0) is ConcentrationLevel.EXTREME  # noqa: SLF001
    assert analyzer._classify_sector_concentration(45.0) is ConcentrationLevel.HIGH  # noqa: SLF001
    assert analyzer._classify_sector_concentration(30.0) is ConcentrationLevel.MODERATE  # noqa: SLF001
    assert analyzer._classify_sector_concentration(10.0) is ConcentrationLevel.LOW  # noqa: SLF001
    assert analyzer._classify_position_concentration(60.0) is ConcentrationLevel.EXTREME  # noqa: SLF001
    assert analyzer._classify_position_concentration(40.0) is ConcentrationLevel.HIGH  # noqa: SLF001
    assert analyzer._classify_position_concentration(25.0) is ConcentrationLevel.MODERATE  # noqa: SLF001
    assert analyzer._classify_position_concentration(10.0) is ConcentrationLevel.LOW  # noqa: SLF001
    assert analyzer._calculate_sector_correlation(["AAPL"]) == 0.0  # noqa: SLF001


def test_should_reduce_position_uses_analysis_sector_and_position_size(monkeypatch) -> None:
    analyzer = PortfolioCorrelationAnalyzer()
    analyzer.last_analysis = cast(Any, SimpleNamespace(
        reduce_exposure_symbols=["AAPL"],
        sector_exposures=[],
        largest_position_pct=0.0,
        total_value=100_000.0,
    ))
    assert analyzer.should_reduce_position("AAPL", []) == (
        True,
        "High correlation/concentration risk",
    )

    analyzer.last_analysis = cast(Any, SimpleNamespace(
        reduce_exposure_symbols=[],
        sector_exposures=[
            SectorExposure("Technology", ["MSFT"], 50_000.0, 50.0, ConcentrationLevel.HIGH, 0.0)
        ],
        largest_position_pct=0.0,
        total_value=100_000.0,
    ))
    assert analyzer.should_reduce_position("MSFT", [])[0] is True

    analyzer.last_analysis = cast(Any, SimpleNamespace(
        reduce_exposure_symbols=[],
        sector_exposures=[],
        largest_position_pct=45.0,
        total_value=100_000.0,
    ))
    monkeypatch.setattr(
        analyzer,
        "_extract_position_data",
        lambda _positions: {"TSLA": {"market_value": 35_000.0}},
    )
    assert analyzer.should_reduce_position("TSLA", [_position("TSLA")]) == (
        True,
        "Position size: 35.0%",
    )


def test_correlation_adjustment_factor_thresholds() -> None:
    analyzer = PortfolioCorrelationAnalyzer()
    analyzer.last_analysis = cast(Any, SimpleNamespace(position_correlations=[]))
    assert analyzer.get_correlation_adjustment_factor("AAPL") == 1.0

    for corr_value, expected in [(0.9, 0.6), (0.7, 0.8), (0.2, 1.2), (0.45, 1.0)]:
        analyzer.last_analysis = cast(Any, SimpleNamespace(
            position_correlations=[
                PositionCorrelation("AAPL", "MSFT", corr_value, CorrelationStrength.HIGH, 30, pd.Timestamp.utcnow().to_pydatetime())
            ]
        ))
        assert analyzer.get_correlation_adjustment_factor("AAPL") == expected


def test_analyze_portfolio_end_to_end_with_recommendations() -> None:
    class _Fetcher:
        def get_daily_df(self, _ctx, symbol):
            step = 1.0 if symbol in {"AAPL", "MSFT"} else -0.5
            return _price_frame(rows=45, start=100.0, step=step)

    analyzer = PortfolioCorrelationAnalyzer(SimpleNamespace(data_fetcher=_Fetcher()))
    analysis = analyzer.analyze_portfolio(
        [
            _position("AAPL", 100, 60_000.0),
            _position("MSFT", 100, 30_000.0),
            _position("JPM", 100, 10_000.0),
        ]
    )

    assert analysis.total_positions == 3
    assert analysis.total_value == 100_000.0
    assert analysis.position_correlations
    assert analysis.sector_exposures
    assert analysis.largest_position_pct == pytest.approx(60.0)
    assert analysis.reduce_exposure_symbols
    assert analyzer.last_analysis is analysis


def test_current_price_prefers_minute_then_daily_and_handles_errors() -> None:
    minute = pd.DataFrame({"close": [101.0]})
    daily = pd.DataFrame({"close": [99.0]})

    class _Fetcher:
        def get_minute_df(self, _ctx, symbol):
            if symbol == "DAILY":
                return pd.DataFrame()
            if symbol == "ERR":
                raise ValueError("bad")
            return minute

        def get_daily_df(self, _ctx, _symbol):
            return daily

    analyzer = PortfolioCorrelationAnalyzer(SimpleNamespace(data_fetcher=_Fetcher()))

    assert analyzer._get_current_price("AAPL") == 101.0  # noqa: SLF001
    assert analyzer._get_current_price("DAILY") == 99.0  # noqa: SLF001
    assert analyzer._get_current_price("ERR") == 0.0  # noqa: SLF001
