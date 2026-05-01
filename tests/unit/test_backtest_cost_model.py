from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.core.enums import RiskLevel
from ai_trading.strategies.backtest import BacktestEngine
from ai_trading.strategies.base import BaseStrategy, StrategySignal


class _AlwaysBuyStrategy(BaseStrategy):
    def __init__(self) -> None:
        super().__init__(
            strategy_id="backtest-cost-model",
            name="backtest-cost-model",
            risk_level=RiskLevel.MODERATE,
        )
        self.symbols = ["AAPL"]

    def generate_signals(self, market_data: dict) -> list[StrategySignal]:
        _ = market_data
        return [StrategySignal("AAPL", "buy", strength=0.9, confidence=0.9)]

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_position: float = 0,
    ) -> int:
        _ = signal
        _ = portfolio_value
        _ = current_position
        return 1


def test_next_open_execution_costs_use_signal_bar_inputs() -> None:
    engine = BacktestEngine(
        initial_capital=10_000.0,
        commission_bps=0.0,
        commission_flat=0.0,
        enable_slippage=True,
        enable_partial_fills=False,
    )
    engine.estimate_half_spread = lambda volatility, _price, _liquidity: volatility
    engine.calculate_slippage = lambda **kwargs: kwargs["volatility"] + kwargs["trade_size"]

    signal = StrategySignal("AAPL", "buy", strength=0.9, confidence=0.9)
    base_market_data = {
        "open": 110.0,
        "close": 100.0,
        "high": 200.0,
        "low": 50.0,
        "volume": 1.0,
        "_signal_price": 100.0,
        "_cost_close": 100.0,
        "_cost_high": 101.0,
        "_cost_low": 99.0,
        "_cost_volume": 1_000.0,
    }
    changed_next_bar = dict(base_market_data, high=500.0, low=1.0, volume=2.0)

    first = engine._simulate_trade(signal, 10, base_market_data)
    second = engine._simulate_trade(signal, 10, changed_next_bar)

    assert first["execution_price"] == pytest.approx(second["execution_price"])
    assert first["half_spread"] == pytest.approx(0.02)
    assert first["slippage_amount"] == pytest.approx(0.03)


def test_losing_backtest_reports_negative_gross_return_and_cost_drag() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    historical_data = {
        "AAPL": [
            {
                "timestamp": start + timedelta(days=offset),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10_000.0,
            }
            for offset in range(3)
        ]
    }
    historical_data["AAPL"][2]["open"] = 110.0
    historical_data["AAPL"][2]["close"] = 100.0

    engine = BacktestEngine(
        initial_capital=10_000.0,
        commission_bps=0.0,
        commission_flat=1.0,
        enable_slippage=False,
        enable_partial_fills=False,
    )
    engine.estimate_half_spread = lambda *_args, **_kwargs: 0.0
    engine._normal = lambda *_args, **_kwargs: 0.0
    result = engine.run_backtest(
        strategy=_AlwaysBuyStrategy(),
        historical_data=historical_data,
        start_date=start,
        end_date=start + timedelta(days=2),
    )

    assert result["gross_pnl_total"] == pytest.approx(-10.0)
    assert result["gross_return"] == pytest.approx(-0.001)
    assert result["net_return"] == pytest.approx(-0.0011)
    assert result["cost_drag"] == pytest.approx(0.0001)
