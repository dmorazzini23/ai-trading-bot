from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.execution.microstructure import (
    BidAskSpreadAnalyzer,
    MarketMicrostructureData,
    MarketMicrostructureEngine,
    MarketRegimeFeature,
    OrderFlowAnalyzer,
)


def _trades(count: int = 24) -> list[dict]:
    start = datetime(2026, 4, 24, 14, 30, tzinfo=UTC)
    rows: list[dict] = []
    for idx in range(count):
        side = "buy" if idx % 2 == 0 else "sell"
        price = 100.0 + idx * 0.05
        rows.append(
            {
                "price": price,
                "size": 100 + idx * 25,
                "side": side,
                "timestamp": start + timedelta(seconds=idx * 10),
                "quote_mid": price - 0.01 if side == "buy" else price + 0.01,
                "quote_mid_before": price,
            }
        )
    return rows


def _quotes(count: int = 24) -> list[dict]:
    start = datetime(2026, 4, 24, 14, 30, tzinfo=UTC)
    return [
        {
            "timestamp": start + timedelta(seconds=idx * 10 + 5),
            "bid_price": 99.9 + idx * 0.04,
            "ask_price": 100.1 + idx * 0.04,
            "bid_size": 1200 + idx * 25,
            "ask_size": 800,
        }
        for idx in range(count)
    ]


def _micro_data(**updates) -> MarketMicrostructureData:
    payload = {
        "symbol": "AAPL",
        "timestamp": datetime.now(UTC),
        "bid_price": 99.95,
        "ask_price": 100.05,
        "last_price": 100.0,
        "bid_size": 1500,
        "ask_size": 1200,
        "volume": 100_000,
        "bid_ask_spread": 0.10,
        "spread_bps": 10.0,
        "quoted_spread_pct": 0.10,
        "effective_spread_pct": 0.02,
        "realized_spread_pct": 0.01,
        "market_depth": 2700.0,
        "order_imbalance": 0.1,
        "price_impact_estimate": 0.05,
        "trade_intensity": 2.0,
        "order_flow_toxicity": 0.1,
        "information_content": 0.2,
        "realized_volatility": 0.02,
        "price_variance_ratio": 1.0,
        "microstructure_noise": 0.1,
    }
    payload.update(updates)
    return MarketMicrostructureData(**payload)


def test_spread_features_include_trade_spreads_depth_and_quality() -> None:
    analyzer = BidAskSpreadAnalyzer()
    features = analyzer.analyze_spread_features(
        {
            "bid_price": 99.95,
            "ask_price": 100.05,
            "last_price": 100.04,
            "bid_size": 1500,
            "ask_size": 1000,
        },
        _trades(),
    )

    assert features["bid_ask_spread"] == pytest.approx(0.10)
    assert features["spread_bps"] == pytest.approx(10.0)
    assert features["relative_spread_position"] > 0
    assert features["effective_spread_pct"] > 0
    assert "realized_spread_pct" in features
    assert features["market_depth"] == 2500
    assert features["depth_imbalance"] == pytest.approx(0.2)
    assert 0 <= features["spread_quality"] <= 1


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        ({"spread_bps": 3, "market_depth": 20_000, "spread_quality": 0.9}, MarketRegimeFeature.HIGH_FREQUENCY),
        ({"spread_bps": 60, "market_depth": 20_000, "spread_quality": 0.7}, MarketRegimeFeature.ILLIQUID),
        ({"spread_bps": 25, "market_depth": 2_000, "spread_quality": 0.2}, MarketRegimeFeature.STRESSED),
        ({"spread_bps": 10, "market_depth": 8_000, "spread_quality": 0.7}, MarketRegimeFeature.INSTITUTIONAL),
        ({"spread_bps": 18, "market_depth": 2_000, "spread_quality": 0.5}, MarketRegimeFeature.RETAIL_DOMINATED),
    ],
)
def test_spread_regime_classification_thresholds(features, expected) -> None:
    assert BidAskSpreadAnalyzer().classify_spread_regime(features) is expected


def test_trade_spread_and_quality_helpers_handle_short_and_bad_inputs() -> None:
    analyzer = BidAskSpreadAnalyzer()

    assert analyzer._calculate_trade_spreads([{"price": 100.0}]) == (0.0, 0.0)  # noqa: SLF001
    effective, realized = analyzer._calculate_trade_spreads(_trades())  # noqa: SLF001
    assert effective > 0
    assert isinstance(realized, float)
    assert analyzer._assess_spread_quality({"spread_bps": 5, "market_depth": 10_000, "depth_imbalance": 0.0}) == pytest.approx(0.95)  # noqa: SLF001
    assert analyzer.analyze_spread_features({"bid_price": "bad", "ask_price": object()}, []) == {}


def test_order_flow_features_and_toxicity_detection() -> None:
    analyzer = OrderFlowAnalyzer()
    trades = _trades()
    quotes = _quotes()
    features = analyzer.analyze_order_flow(trades, quotes)

    assert features["trade_intensity"] > 0
    assert features["order_imbalance"] > 0
    assert features["information_content"] >= 0
    assert features["vwap_deviation"] >= 0
    assert features["avg_trade_size"] > 0
    assert features["trade_size_volatility"] > 0

    toxic = analyzer.detect_toxic_flow(
        {
            "order_flow_toxicity": 0.8,
            "trade_intensity": 11.0,
            "order_imbalance": 0.9,
            "information_content": 0.8,
        }
    )
    assert toxic["is_toxic"] is True
    assert toxic["risk_level"] == "extreme"
    assert toxic["primary_concern"] == "High flow toxicity"
    assert toxic["recommendations"]

    normal = analyzer.detect_toxic_flow({"order_flow_toxicity": 0.0, "trade_intensity": 1.0})
    assert normal["is_toxic"] is False
    assert normal["risk_level"] == "normal"


def test_order_flow_helper_defaults_and_recommendations() -> None:
    analyzer = OrderFlowAnalyzer()

    assert analyzer._calculate_trade_intensity([{"timestamp": datetime.now(UTC)}]) == 0.0  # noqa: SLF001
    assert analyzer._calculate_order_imbalance([]) == 0.0  # noqa: SLF001
    assert analyzer._calculate_order_imbalance([{"bid_size": 100, "ask_size": 300}]) == pytest.approx(-0.5)  # noqa: SLF001
    assert analyzer._calculate_flow_toxicity([], []) == 0.0  # noqa: SLF001
    assert analyzer._estimate_information_content([{"price": 1.0}]) == 0.0  # noqa: SLF001
    assert analyzer._calculate_vwap_deviation([{"price": 1.0}]) == 0.0  # noqa: SLF001
    assert analyzer._identify_primary_concern({"trade_intensity": 6.0}) == "Elevated trade intensity"  # noqa: SLF001
    recommendations = analyzer._generate_flow_recommendations(  # noqa: SLF001
        {"order_flow_toxicity": 0.4, "trade_intensity": 6.0, "order_imbalance": -0.7}
    )
    assert len(recommendations) >= 5


def test_market_microstructure_engine_analysis_and_default() -> None:
    engine = MarketMicrostructureEngine()
    market_data = {
        "bid_price": 99.95,
        "ask_price": 100.05,
        "last_price": 100.0,
        "bid_size": 1500,
        "ask_size": 1200,
        "volume": 100_000,
    }

    data = engine.analyze_market_microstructure("AAPL", market_data, _trades(), _quotes())
    default = engine.analyze_market_microstructure("BAD", {"bid_price": "bad"}, [], [])

    assert data.symbol == "AAPL"
    assert data.spread_bps > 0
    assert data.trade_intensity > 0
    assert data.realized_volatility >= 0
    assert default.symbol == "BAD"
    assert default.spread_bps == 0.0


def test_execution_impact_confidence_and_market_impact_helpers() -> None:
    engine = MarketMicrostructureEngine()
    data = _micro_data(spread_bps=12.0, realized_volatility=0.03, order_flow_toxicity=0.2)

    impact = engine.estimate_execution_impact(10_000, data)

    assert impact["participation_rate"] == pytest.approx(0.1)
    assert impact["total_impact_bps"] > impact["spread_impact_bps"]
    assert 0 <= impact["confidence_level"] <= 1
    assert engine._calculate_market_impact(0.0, data) >= 1.0  # noqa: SLF001
    assert engine._calculate_market_impact(10.0, data) <= 200.0  # noqa: SLF001

    low_conf = engine._assess_impact_confidence(  # noqa: SLF001
        _micro_data(bid_size=0, ask_size=0, spread_bps=80.0, order_flow_toxicity=0.8)
    )
    assert low_conf < impact["confidence_level"]


def test_price_impact_volatility_autocorrelation_and_defaults() -> None:
    engine = MarketMicrostructureEngine()
    trades = _trades(30)

    price_impact = engine._estimate_price_impact({"bid_size": 1000, "ask_size": 1000}, trades)  # noqa: SLF001
    default_impact = engine._estimate_price_impact({"bid_size": 0, "ask_size": 0}, [])  # noqa: SLF001
    vol = engine._analyze_microstructure_volatility(trades)  # noqa: SLF001
    default_vol = engine._analyze_microstructure_volatility([])  # noqa: SLF001

    assert price_impact["estimated_impact"] > 0
    assert default_impact == {"estimated_impact": 0.05}
    assert vol["realized_volatility"] >= 0
    assert "price_variance_ratio" in vol
    assert "microstructure_noise" in vol
    assert default_vol == {
        "realized_volatility": 0.0,
        "price_variance_ratio": 1.0,
        "microstructure_noise": 0.0,
    }
    assert engine._calculate_autocorrelation([1.0], lag=1) == 0.0  # noqa: SLF001
    assert engine._calculate_autocorrelation([1.0, 2.0, 3.0, 4.0], lag=1) > 0.0  # noqa: SLF001
