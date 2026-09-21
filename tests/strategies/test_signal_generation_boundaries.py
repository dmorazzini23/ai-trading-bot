from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from ai_trading.strategies import mean_reversion as reversal, momentum


@pytest.mark.parametrize('last,side', [(80., 'buy'), (120., 'sell')])
def test_reversal_signal_has_bar_provenance_and_duplicate_guard(monkeypatch, last, side):
    monkeypatch.setattr(reversal, 'load_strategy_profile', lambda: None)
    frame = pd.DataFrame({'close': [100., 100., 100., 100., last]},
                         index=pd.date_range('2024-01-02', periods=5, tz='UTC'))
    strategy = reversal.MeanReversionStrategy(lookback=5, z=1.)
    ctx = NS(tickers=['AAPL'], data_fetcher=NS(get_daily_df=lambda *a: frame))
    signals = strategy.generate(ctx)
    assert len(signals) == 1 and signals[0].side == side
    assert signals[0].metadata['bar_ts'] == frame.index[-1].isoformat()
    assert 0 <= signals[0].confidence <= 1
    assert strategy.generate(ctx) == []
    assert strategy._guard_skips == 1


@pytest.mark.parametrize('prices', [[100.], [100.] * 5, [np.nan] * 5])
def test_reversal_requires_finite_nonzero_variance(monkeypatch, prices):
    monkeypatch.setattr(reversal, 'load_strategy_profile', lambda: None)
    strategy = reversal.MeanReversionStrategy()
    ctx = NS(tickers=['AAPL'], data_fetcher=NS(get_daily_df=lambda *a: pd.DataFrame({'close': prices})))
    assert strategy.generate(ctx) == []


def test_momentum_rejects_missing_short_and_nonfinite_series(monkeypatch):
    monkeypatch.setattr(momentum, 'load_strategy_profile', lambda: None)
    strategy = momentum.MomentumStrategy(lookback=2)
    prices = {'AAPL': pd.Series([100., 101., 102., 103., 104.]),
              'SHORT': pd.Series([100.]), 'BAD': pd.Series([100., 100., 100., 100., np.nan])}
    result = strategy.generate_signals({'symbols': ['AAPL', 'SHORT', 'BAD', 'ABSENT'], 'prices': prices})
    assert [signal.symbol for signal in result] == ['AAPL']
    assert result[0].side == 'buy'
    assert result[0].metadata['lookback'] == 2
    assert strategy.generate_signals({'prices': prices}) == []
