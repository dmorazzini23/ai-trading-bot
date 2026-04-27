from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.position.intelligent_manager import IntelligentPositionManager, PositionAction
from ai_trading.position.market_regime import MarketRegime
from ai_trading.position.technical_analyzer import DivergenceType, SignalStrength, TechnicalSignals


def test_analyze_position_integrates_component_outputs() -> None:
    manager = IntelligentPositionManager(ctx=None)
    position = SimpleNamespace(qty=100)
    technical = TechnicalSignals(
        symbol="SPY",
        timestamp=datetime.now(UTC),
        momentum_score=0.2,
        divergence_type=DivergenceType.BEARISH,
        divergence_strength=0.8,
        volume_strength=0.5,
        volume_trend="falling",
        relative_strength_score=0.3,
        outperformance_rank=0.4,
        distance_to_support=10.0,
        distance_to_resistance=2.0,
        sr_confidence=0.5,
        hold_recommendation=SignalStrength.VERY_WEAK,
        exit_urgency=0.9,
    )
    target = SimpleNamespace(quantity_pct=30.0)

    manager._get_current_price = lambda _symbol: 100.0  # type: ignore[method-assign]
    manager._analyze_market_regime = lambda: {  # type: ignore[method-assign]
        "regime": MarketRegime.HIGH_VOLATILITY,
        "confidence": 0.8,
        "parameters": {"profit_taking_patience": 0.6},
    }
    manager._analyze_technical_signals = lambda *_args: {  # type: ignore[method-assign]
        "signals": technical,
        "hold_strength": SignalStrength.VERY_WEAK,
        "exit_urgency": 0.9,
        "divergence": DivergenceType.BEARISH,
        "momentum": 0.2,
    }
    manager._analyze_profit_opportunities = lambda *_args: {  # type: ignore[method-assign]
        "triggered_targets": [target],
        "profit_plan": None,
        "velocity": 0.1,
        "has_targets": True,
    }
    manager._analyze_trailing_stops = lambda *_args: {  # type: ignore[method-assign]
        "stop_level": SimpleNamespace(stop_price=95.0),
        "is_triggered": False,
        "stop_price": 95.0,
        "trail_distance": 5.0,
    }
    manager._analyze_portfolio_context = lambda *_args: {  # type: ignore[method-assign]
        "portfolio_analysis": None,
        "should_reduce": True,
        "reduce_reason": "sector concentration",
        "correlation_factor": 0.7,
    }

    recommendation = manager.analyze_position("SPY", position, [position])

    assert recommendation.action is PositionAction.FULL_SELL
    assert recommendation.quantity_to_sell == 100
    assert recommendation.stop_price == 95.0
    assert recommendation.primary_reason == "Profit targets triggered (30.0%)"
    assert "Bearish momentum divergence" in recommendation.contributing_factors


def test_analyze_position_returns_default_without_price_or_on_failure() -> None:
    manager = IntelligentPositionManager(ctx=None)
    manager._get_current_price = lambda _symbol: 0.0  # type: ignore[method-assign]

    no_price = manager.analyze_position("SPY", SimpleNamespace(qty=10), [])
    assert no_price.action is PositionAction.NO_ACTION
    assert no_price.primary_reason == "No current price data"

    manager._get_current_price = lambda _symbol: (_ for _ in ()).throw(ValueError("bad"))  # type: ignore[method-assign]
    failed = manager.analyze_position("SPY", SimpleNamespace(qty=10), [])
    assert failed.action is PositionAction.NO_ACTION
    assert "Analysis error: bad" in failed.primary_reason
