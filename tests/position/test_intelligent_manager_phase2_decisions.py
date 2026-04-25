from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai_trading.position.intelligent_manager import (
    IntelligentPositionManager,
    PositionAction,
    PositionRecommendation,
)
from ai_trading.position.market_regime import MarketRegime
from ai_trading.position.technical_analyzer import DivergenceType


def _manager() -> IntelligentPositionManager:
    return IntelligentPositionManager(ctx=None)


def test_determine_action_from_scores_covers_decision_boundaries() -> None:
    manager = _manager()

    assert manager._determine_action_from_scores(0.1, 0.1, 0.1) == (
        PositionAction.NO_ACTION,
        0.3,
        0.0,
    )
    assert manager._determine_action_from_scores(0.1, 0.7, 0.2)[0] is PositionAction.FULL_SELL
    assert manager._determine_action_from_scores(0.1, 0.2, 0.5)[0] is PositionAction.PARTIAL_SELL
    assert manager._determine_action_from_scores(0.5, 0.1, 0.1)[0] is PositionAction.HOLD
    assert manager._determine_action_from_scores(0.1, 0.3, 0.1)[0] is PositionAction.REDUCE_SIZE
    assert manager._determine_action_from_scores(0.25, 0.2, 0.1)[0] is PositionAction.HOLD


def test_calculate_action_details_handles_full_partial_reduce_and_stop() -> None:
    manager = _manager()
    position = SimpleNamespace(qty=100)
    target = SimpleNamespace(quantity_pct=30.0)
    stop = SimpleNamespace(stop_price=91.5)

    assert manager._calculate_action_details(
        PositionAction.FULL_SELL,
        position,
        100.0,
        [],
        {"stop_level": stop},
    ) == (100, 100.0, 99.5, 91.5)
    assert manager._calculate_action_details(
        PositionAction.PARTIAL_SELL,
        position,
        100.0,
        [target],
        {},
    ) == (30, 30.0, 99.8, None)
    assert manager._calculate_action_details(
        PositionAction.PARTIAL_SELL,
        position,
        100.0,
        [],
        {},
    ) == (25, 25.0, 99.8, None)
    assert manager._calculate_action_details(
        PositionAction.REDUCE_SIZE,
        position,
        100.0,
        [],
        {},
    ) == (20, 20.0, 99.8, None)


@pytest.mark.parametrize(
    ("action", "inputs", "expected"),
    [
        (
            PositionAction.FULL_SELL,
            {"stop_analysis": {"is_triggered": True}},
            "Trailing stop loss triggered",
        ),
        (
            PositionAction.PARTIAL_SELL,
            {"profit_analysis": {"triggered_targets": [SimpleNamespace(quantity_pct=25.0)]}},
            "Profit targets triggered (25.0%)",
        ),
        (
            PositionAction.FULL_SELL,
            {
                "technical_analysis": {
                    "signals": SimpleNamespace(
                        exit_urgency=0.8,
                        divergence_type=DivergenceType.NONE,
                    )
                }
            },
            "Technical exit signal (urgency: 0.80)",
        ),
        (
            PositionAction.FULL_SELL,
            {
                "technical_analysis": {
                    "signals": SimpleNamespace(
                        exit_urgency=0.2,
                        divergence_type=DivergenceType.BEARISH,
                    )
                }
            },
            "Bearish momentum divergence detected",
        ),
        (
            PositionAction.REDUCE_SIZE,
            {"correlation_analysis": {"should_reduce": True, "reduce_reason": "clustered beta"}},
            "Portfolio risk management: clustered beta",
        ),
        (
            PositionAction.HOLD,
            {"regime_analysis": {"regime": MarketRegime.TRENDING_BULL}},
            "Market regime supports holding (trending_bull)",
        ),
        (
            PositionAction.PARTIAL_SELL,
            {"regime_analysis": {"regime": MarketRegime.RANGE_BOUND}},
            "Market regime favors profit taking (range_bound)",
        ),
    ],
)
def test_determine_primary_reason_prioritizes_runtime_causes(
    action: PositionAction,
    inputs: dict[str, Any],
    expected: str,
) -> None:
    manager = _manager()
    reason = manager._determine_primary_reason(
        action,
        inputs.get("regime_analysis", {"regime": MarketRegime.RANGE_BOUND}),
        inputs.get("technical_analysis", {}),
        inputs.get("profit_analysis", {}),
        inputs.get("stop_analysis", {}),
        inputs.get("correlation_analysis", {}),
    )

    assert reason == expected


def test_get_current_price_prefers_minute_then_daily_and_degrades_on_errors() -> None:
    pd = pytest.importorskip("pandas")
    minute = pd.DataFrame({"close": [101.5]})
    daily = pd.DataFrame({"close": [99.25]})
    data_fetcher = SimpleNamespace(
        get_minute_df=lambda _ctx, _symbol: minute,
        get_daily_df=lambda _ctx, _symbol: daily,
    )
    ctx = SimpleNamespace(data_fetcher=data_fetcher)
    manager = IntelligentPositionManager(ctx)

    assert manager._get_current_price("SPY") == 101.5

    data_fetcher.get_minute_df = lambda _ctx, _symbol: None
    assert manager._get_current_price("SPY") == 99.25

    data_fetcher.get_daily_df = lambda _ctx, _symbol: (_ for _ in ()).throw(ValueError("bad data"))
    assert manager._get_current_price("SPY") == 0.0


def test_should_hold_position_maps_recommendation_actions_and_fallback() -> None:
    manager = _manager()

    def rec(action: PositionAction, confidence: float) -> PositionRecommendation:
        return PositionRecommendation(
            symbol="SPY",
            action=action,
            confidence=confidence,
            urgency=0.2,
            timestamp=datetime.now(UTC),
        )

    manager.analyze_position = lambda *_args, **_kwargs: rec(PositionAction.HOLD, 0.9)  # type: ignore[method-assign]
    assert manager.should_hold_position("SPY", object(), -10.0, 10) is True

    manager.analyze_position = lambda *_args, **_kwargs: rec(PositionAction.FULL_SELL, 0.9)  # type: ignore[method-assign]
    assert manager.should_hold_position("SPY", object(), 20.0, 10) is False

    manager.analyze_position = lambda *_args, **_kwargs: rec(PositionAction.PARTIAL_SELL, 0.7)  # type: ignore[method-assign]
    assert manager.should_hold_position("SPY", object(), 20.0, 10) is True

    manager.analyze_position = lambda *_args, **_kwargs: rec(PositionAction.REDUCE_SIZE, 0.85)  # type: ignore[method-assign]
    assert manager.should_hold_position("SPY", object(), 20.0, 10) is False

    manager.analyze_position = lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad"))  # type: ignore[method-assign]
    assert manager.should_hold_position("SPY", object(), 6.0, 10) is True
    assert manager.should_hold_position("SPY", object(), -1.0, 1) is True
    assert manager.should_hold_position("SPY", object(), -1.0, 5) is False


def test_get_portfolio_recommendations_sorts_by_urgency_and_handles_failures() -> None:
    manager = _manager()
    positions = [SimpleNamespace(symbol="LOW"), SimpleNamespace(symbol="HIGH")]

    cast(Any, manager.correlation_analyzer).analyze_portfolio = lambda _positions: object()

    def fake_analyze(
        symbol: str,
        position_data: Any,
        current_positions: list[Any],
    ) -> PositionRecommendation:
        del position_data, current_positions
        urgency = 0.9 if symbol == "HIGH" else 0.2
        return PositionRecommendation(
            symbol=symbol,
            action=PositionAction.HOLD,
            confidence=0.5,
            urgency=urgency,
            timestamp=datetime.now(UTC),
        )

    manager.analyze_position = fake_analyze  # type: ignore[method-assign]

    recommendations = manager.get_portfolio_recommendations(positions)

    assert [item.symbol for item in recommendations] == ["HIGH", "LOW"]

    cast(Any, manager.correlation_analyzer).analyze_portfolio = lambda _positions: (
        _ for _ in ()
    ).throw(ValueError("bad"))
    assert manager.get_portfolio_recommendations(positions) == []
