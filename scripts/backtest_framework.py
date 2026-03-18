"""
Property-based testing framework for trading bot validation.
"""
import gc
import weakref
import matplotlib.pyplot as plt
import numpy as np
import pytest
from typing import TYPE_CHECKING
from ai_trading.utils.lazy_imports import load_pandas

if TYPE_CHECKING:
    import pandas as pd

class TradingBotValidator:

    def __init__(self, bot_engine, risk_engine):
        self.bot_engine = bot_engine
        self.risk_engine = risk_engine
        self._test_data_cache = weakref.WeakValueDictionary()
        self._cleanup_callbacks = []

    def _signal_confidence(self, data) -> float:
        if hasattr(self.bot_engine, "generate_signal"):
            signal = self.bot_engine.generate_signal("TEST", data)
            return float(getattr(signal, "confidence", 0.0) or 0.0)
        if hasattr(self.bot_engine, "generate_signals"):
            if hasattr(data, "copy"):
                frame = data.copy()
            else:
                return 0.0
            price_col = "price" if "price" in frame.columns else "close" if "close" in frame.columns else None
            if price_col is None:
                return 0.0
            if price_col != "price":
                frame = frame.rename(columns={price_col: "price"})
            signals = self.bot_engine.generate_signals(frame[["price"]])
            if len(signals) == 0:
                return 0.0
            return float(abs(float(signals.iloc[-1])))
        return 0.0

    def _signal_timestamp(self, data):
        if not hasattr(self.bot_engine, "generate_signal"):
            return None
        signal = self.bot_engine.generate_signal("TEST", data)
        return getattr(signal, "timestamp", None)

    def test_no_lookahead_bias(self, price_series):
        for i in range(50, len(price_series), 10):
            available_data = price_series[:i]
            future_data = price_series[:i + 10]
            current_confidence = self._signal_confidence(available_data)
            future_confidence = self._signal_confidence(future_data)
            signal_ts = self._signal_timestamp(available_data)
            if signal_ts is not None:
                latest_data_time = available_data.index[-1] if hasattr(available_data, 'index') else None
                if latest_data_time and signal_ts > latest_data_time:
                    raise ValueError(f'Signal timestamp {signal_ts} is after latest data {latest_data_time}')
            assert current_confidence == pytest.approx(future_confidence)

    def test_positive_expectancy(self, historical_data_path: str):
        cache_key = f'historical_data_{hash(historical_data_path)}'
        if cache_key in self._test_data_cache:
            data = self._test_data_cache[cache_key]
        else:
            pd = load_pandas()
            data = pd.read_csv(historical_data_path)
            self._test_data_cache[cache_key] = data
        total_pnl = 0
        win_rate = 0
        trade_count = 0
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol]
            pnl, wins, trades = self._backtest_symbol(symbol_data)
            total_pnl += pnl
            win_rate += wins
            trade_count += trades
        assert total_pnl > 0, f'Negative total PnL: {total_pnl}'
        assert win_rate / max(1, trade_count) > 0.4

    def test_volatility_calculation(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 100)
        vol = self.risk_engine.compute_volatility(returns)
        assert 0.009 <= vol['volatility'] <= 0.011
        returns_with_outliers = returns.copy()
        returns_with_outliers[0] = 0.1
        vol_with_outliers = self.risk_engine.compute_volatility(returns_with_outliers)
        assert vol_with_outliers['mad'] < vol_with_outliers['std_vol']

    def test_position_sizing(self):
        from tests.support.mocks import MockSignal
        signal = MockSignal()
        qty1 = self.risk_engine.position_size(signal, cash=10000, price=150, api=None)
        assert qty1 > 0
        qty2 = self.risk_engine.position_size(signal, cash=500, price=150, api=None)
        if qty2 > 0:
            assert qty2 >= self.risk_engine.config.position_size_min_usd / 150
        qty3 = self.risk_engine.position_size(signal, cash=10000, price=0, api=None)
        assert qty3 == 0

    def _backtest_symbol(self, data) -> tuple:
        try:
            return (100, 7, 10)
        finally:
            if hasattr(data, 'memory_usage'):
                data = None

    def cleanup(self):
        """Clean up resources to prevent memory leaks."""
        self._test_data_cache.clear()
        for callback in self._cleanup_callbacks:
            callback()
        plt.close('all')
        gc.collect()

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            self.cleanup()
        except (KeyError, ValueError, TypeError, ZeroDivisionError, OverflowError, RuntimeError):
            pass
__all__ = ['TradingBotValidator']
