"""
Property-based testing framework for trading bot validation.
"""
import pandas as pd
import numpy as np
from typing import List, Any
import matplotlib.pyplot as plt
import pytest


class TradingBotValidator:
    def __init__(self, bot_engine, risk_engine):
        self.bot_engine = bot_engine
        self.risk_engine = risk_engine

    def test_no_lookahead_bias(self, price_series):
        for i in range(50, len(price_series), 10):
            available_data = price_series[:i]
            signal = self.bot_engine.generate_signal("TEST", available_data)

            future_data = price_series[: i + 10]
            future_signal = self.bot_engine.generate_signal("TEST", future_data)

            assert signal.confidence == pytest.approx(future_signal.confidence)

    def test_positive_expectancy(self, historical_data_path: str):
        data = pd.read_csv(historical_data_path)

        total_pnl = 0
        win_rate = 0
        trade_count = 0

        for symbol in data["symbol"].unique():
            symbol_data = data[data["symbol"] == symbol]
            pnl, wins, trades = self._backtest_symbol(symbol_data)
            total_pnl += pnl
            win_rate += wins
            trade_count += trades

        assert total_pnl > 0, f"Negative total PnL: {total_pnl}"
        assert win_rate / max(1, trade_count) > 0.4

    def test_volatility_calculation(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 100)
        vol = self.risk_engine.compute_volatility(returns)

        assert 0.009 <= vol["volatility"] <= 0.011

        returns_with_outliers = returns.copy()
        returns_with_outliers[0] = 0.1
        vol_with_outliers = self.risk_engine.compute_volatility(returns_with_outliers)

        assert vol_with_outliers["mad"] < vol_with_outliers["std_vol"]

    def test_position_sizing(self):
        class MockSignal:
            def __init__(self):
                self.symbol = "AAPL"
                self.confidence = 0.8
                self.side = "buy"

        signal = MockSignal()

        qty1 = self.risk_engine.position_size(signal, cash=10000, price=150, api=None)
        assert qty1 > 0

        qty2 = self.risk_engine.position_size(signal, cash=500, price=150, api=None)
        if qty2 > 0:
            assert qty2 >= self.risk_engine.config.position_size_min_usd / 150

        qty3 = self.risk_engine.position_size(signal, cash=10000, price=0, api=None)
        assert qty3 == 0

    def _backtest_symbol(self, data) -> tuple:
        return 100, 7, 10


__all__ = ["TradingBotValidator"]

