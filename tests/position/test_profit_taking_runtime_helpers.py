from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from ai_trading.position.profit_taking import (
    ProfitTakingEngine,
    ProfitTakingPlan,
    ProfitTakingStrategy,
    ProfitTarget,
)


def _market_frame(rows: int = 60, *, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    close = [start + i * step for i in range(rows)]
    return pd.DataFrame(
        {
            "close": close,
            "high": [price + 3.0 for price in close],
            "low": [price - 2.0 for price in close],
            "volume": [1000 + i for i in range(rows)],
        }
    )


def _position(qty: int = 100) -> SimpleNamespace:
    return SimpleNamespace(qty=qty, avg_entry_price=100.0)


def _plan(*, current_price: float = 110.0, created_days_ago: int = 1) -> ProfitTakingPlan:
    return ProfitTakingPlan(
        symbol="AAPL",
        entry_price=100.0,
        current_price=current_price,
        position_size=100,
        risk_amount=300.0,
        targets=[],
        remaining_quantity=100,
        total_profit_realized=0.0,
        created_time=datetime.now(UTC) - timedelta(days=created_days_ago),
        last_updated=datetime.now(UTC),
    )


def test_create_profit_plan_builds_risk_technical_and_time_targets() -> None:
    class _Fetcher:
        def get_minute_df(self, _ctx, _symbol):
            return pd.DataFrame({"close": [106.0]})

        def get_daily_df(self, _ctx, _symbol):
            return _market_frame(rows=80, start=100.0, step=0.5)

    engine = ProfitTakingEngine(SimpleNamespace(data_fetcher=_Fetcher()))
    engine.overbought_threshold = 10

    plan = engine.create_profit_plan("AAPL", _position(), 100.0, 300.0)

    assert plan is not None
    assert engine.get_profit_plan("AAPL") is plan
    assert plan.current_price == 106.0
    strategies = {target.strategy for target in plan.targets}
    assert ProfitTakingStrategy.RISK_MULTIPLE in strategies
    assert ProfitTakingStrategy.TECHNICAL_LEVELS in strategies
    assert ProfitTakingStrategy.TIME_BASED in strategies
    assert plan.targets == sorted(plan.targets, key=lambda target: (target.priority, target.level))


def test_short_profit_plan_uses_absolute_size_and_downside_targets() -> None:
    class _Fetcher:
        def get_minute_df(self, _ctx, _symbol):
            return pd.DataFrame({"close": [100.0]})

        def get_daily_df(self, _ctx, _symbol):
            return pd.DataFrame({"close": [100.0] * 30, "high": [101.0] * 30})

    engine = ProfitTakingEngine(SimpleNamespace(data_fetcher=_Fetcher()))

    plan = engine.create_profit_plan("AAPL", _position(qty=-100), 100.0, 300.0)

    assert plan is not None
    assert plan.position_size == 100
    assert plan.remaining_quantity == 100
    assert plan.side == "short"
    risk_targets = [target for target in plan.targets if target.strategy is ProfitTakingStrategy.RISK_MULTIPLE]
    assert [target.level for target in risk_targets] == pytest.approx([94.0, 91.0, 85.0])

    assert engine.update_profit_plan("AAPL", 99.0, _position(qty=-100)) == []
    triggered = engine.update_profit_plan("AAPL", 94.0, _position(qty=-80))

    assert [target.level for target in triggered] == pytest.approx([94.0])
    assert plan.remaining_quantity == 80


def test_create_profit_plan_returns_none_for_empty_position_or_missing_price() -> None:
    engine = ProfitTakingEngine()

    assert engine.create_profit_plan("AAPL", _position(qty=0), 100.0, 300.0) is None
    assert engine.create_profit_plan("AAPL", _position(qty=10), 100.0, 300.0) is None


def test_update_profit_plan_triggers_targets_and_correlation_once() -> None:
    engine = ProfitTakingEngine()
    plan = _plan(current_price=100.0)
    plan.targets = [
        ProfitTarget(
            level=105.0,
            quantity_pct=25.0,
            strategy=ProfitTakingStrategy.RISK_MULTIPLE,
            priority=1,
        ),
        ProfitTarget(
            level=130.0,
            quantity_pct=25.0,
            strategy=ProfitTakingStrategy.PERCENTAGE_BASED,
            priority=2,
        ),
    ]
    engine.profit_plans["AAPL"] = plan

    triggered = engine.update_profit_plan("AAPL", 112.0, _position(qty=80))

    assert [target.strategy for target in triggered] == [
        ProfitTakingStrategy.RISK_MULTIPLE,
        ProfitTakingStrategy.CORRELATION_BASED,
    ]
    assert plan.remaining_quantity == 80
    assert plan.targets[0].triggered is True
    assert plan.targets[-1].strategy is ProfitTakingStrategy.CORRELATION_BASED

    second = engine.update_profit_plan("AAPL", 113.0, _position(qty=80))

    assert second == []


def test_update_profit_plan_returns_empty_for_missing_symbol() -> None:
    assert ProfitTakingEngine().update_profit_plan("MISSING", 120.0) == []


def test_remove_profit_plan_deletes_existing_plan() -> None:
    engine = ProfitTakingEngine()
    engine.profit_plans["AAPL"] = _plan()

    engine.remove_profit_plan("AAPL")

    assert engine.get_profit_plan("AAPL") is None


def test_calculate_profit_velocity_uses_days_held() -> None:
    engine = ProfitTakingEngine()
    engine.profit_plans["AAPL"] = _plan(current_price=120.0, created_days_ago=4)

    assert engine.calculate_profit_velocity("AAPL") == pytest.approx(5.0)
    assert engine.calculate_profit_velocity("MSFT") == 0.0

    engine.profit_plans["ZERO"] = _plan(current_price=110.0, created_days_ago=0)
    assert engine.calculate_profit_velocity("ZERO") == 0.0


def test_create_risk_multiple_targets_falls_back_to_percentage_targets() -> None:
    engine = ProfitTakingEngine()

    targets = engine._create_risk_multiple_targets(100.0, 0.0, 100)  # noqa: SLF001

    assert [target.strategy for target in targets] == [ProfitTakingStrategy.PERCENTAGE_BASED] * 3
    assert [target.level for target in targets] == pytest.approx([105.0, 110.0, 120.0])


def test_create_risk_multiple_targets_uses_risk_per_share() -> None:
    engine = ProfitTakingEngine()

    targets = engine._create_risk_multiple_targets(100.0, 300.0, 100)  # noqa: SLF001

    assert [target.level for target in targets] == pytest.approx([106.0, 109.0, 115.0])
    assert all(target.strategy is ProfitTakingStrategy.RISK_MULTIPLE for target in targets)


def test_short_percentage_and_time_targets_are_side_aware() -> None:
    engine = ProfitTakingEngine()

    percentage_targets = engine._create_percentage_targets(100.0, -100)  # noqa: SLF001
    time_targets = engine._create_time_based_targets(100.0, 96.0, -100)  # noqa: SLF001
    short_plan = _plan(current_price=94.0)
    short_plan.side = "short"

    assert [target.level for target in percentage_targets] == pytest.approx([95.0, 90.0, 80.0])
    assert len(time_targets) == 1
    assert time_targets[0].level == pytest.approx(94.08)
    assert engine._is_target_triggered(ProfitTarget(94.0, 25.0, ProfitTakingStrategy.RISK_MULTIPLE, 1), short_plan) is True  # noqa: SLF001


def test_create_technical_targets_uses_resistance_and_rsi(monkeypatch) -> None:
    engine = ProfitTakingEngine()
    monkeypatch.setattr(engine, "_get_market_data", lambda _symbol: _market_frame(rows=80))
    monkeypatch.setattr(engine, "_calculate_rsi", lambda *_args: 80.0)

    targets = engine._create_technical_targets("AAPL", 120.0, 100)  # noqa: SLF001

    assert len(targets) >= 2
    assert all(target.strategy is ProfitTakingStrategy.TECHNICAL_LEVELS for target in targets)
    assert any("Resistance level" in target.reason for target in targets)
    assert any("RSI overbought" in target.reason for target in targets)


def test_create_time_based_targets_only_for_high_velocity_gain() -> None:
    engine = ProfitTakingEngine()

    assert engine._create_time_based_targets(100.0, 102.0, 100) == []  # noqa: SLF001
    targets = engine._create_time_based_targets(100.0, 104.0, 100)  # noqa: SLF001

    assert len(targets) == 1
    assert targets[0].strategy is ProfitTakingStrategy.TIME_BASED
    assert targets[0].level == pytest.approx(106.08)


def test_is_target_triggered_covers_strategy_rules() -> None:
    engine = ProfitTakingEngine()
    plan = _plan(current_price=110.0, created_days_ago=20)

    assert engine._is_target_triggered(ProfitTarget(105.0, 25.0, ProfitTakingStrategy.RISK_MULTIPLE, 1), plan) is True  # noqa: SLF001
    assert engine._is_target_triggered(ProfitTarget(115.0, 25.0, ProfitTakingStrategy.TECHNICAL_LEVELS, 1), plan) is False  # noqa: SLF001
    assert engine._is_target_triggered(ProfitTarget(111.0, 25.0, ProfitTakingStrategy.TIME_BASED, 1), plan) is True  # noqa: SLF001
    assert engine._is_target_triggered(ProfitTarget(105.0, 25.0, ProfitTakingStrategy.CORRELATION_BASED, 1), plan) is False  # noqa: SLF001


def test_find_resistance_levels_and_rsi_target(monkeypatch) -> None:
    engine = ProfitTakingEngine()
    data = _market_frame(rows=60, start=100.0, step=0.5)

    levels = engine._find_resistance_levels(data, current_price=100.0)  # noqa: SLF001
    missing = engine._find_resistance_levels(pd.DataFrame({"close": [1.0] * 10}), 100.0)  # noqa: SLF001
    monkeypatch.setattr(engine, "_calculate_rsi", lambda *_args: 90.0)
    target = engine._create_rsi_overbought_target("AAPL", data, 100)  # noqa: SLF001
    monkeypatch.setattr(engine, "_calculate_rsi", lambda *_args: 40.0)
    no_target = engine._create_rsi_overbought_target("AAPL", data, 100)  # noqa: SLF001

    assert levels
    assert missing == []
    assert target is not None
    assert target.quantity_pct == 10.0
    assert no_target is None


def test_price_and_market_data_helpers_prefer_available_frames() -> None:
    minute = pd.DataFrame({"close": [101.0]})
    daily = pd.DataFrame({"close": [99.0]})

    class _Fetcher:
        def get_minute_df(self, _ctx, _symbol):
            return minute

        def get_daily_df(self, _ctx, _symbol):
            return daily

    engine = ProfitTakingEngine(SimpleNamespace(data_fetcher=_Fetcher()))

    assert engine._get_current_price("AAPL") == 101.0  # noqa: SLF001
    assert engine._get_market_data("AAPL") is daily  # noqa: SLF001


def test_rsi_helper_accepts_valid_short_and_bad_inputs() -> None:
    engine = ProfitTakingEngine()
    prices = pd.Series([float(i) for i in range(1, 40)])

    assert 0 <= engine._calculate_rsi(prices, 14) <= 100  # noqa: SLF001
    assert engine._calculate_rsi([1.0, 2.0], 14) == 50.0  # noqa: SLF001
    assert engine._calculate_rsi(object(), 14) == 50.0  # noqa: SLF001
