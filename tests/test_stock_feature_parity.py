import numpy as np
import pandas as pd
import pytest

from ai_trading.features.day_sleeve import build_day_sleeve_features
from ai_trading.tools.stock_feature_parity import compare_history
from ai_trading.tools.train_replay_aligned_model import _feature_frame


def bars():
    close = 100 + np.arange(510) * .02 + np.sin(np.arange(510) / 9)
    return pd.DataFrame(dict(open=close, high=close + 1, low=close - 1,
                             close=close, volume=1000),
                        index=pd.date_range('2024-01-02T14:30:00Z', periods=510, freq='5min'))


def test_same_history_and_prefix_causality():
    result = compare_history(bars(), symbol='AAPL', end=500)
    assert result['same_history_equal']
    assert result['prefix_causal']


def test_training_fallback_is_not_runtime_readiness():
    short = bars().iloc[:20]
    assert _feature_frame(short, symbol='AAPL')['sma_200'].isna().all()
    with pytest.raises(ValueError, match='non-finite'):
        build_day_sleeve_features(short)


def test_runtime_window_is_distinct_from_minimum_feature_history():
    index = pd.DatetimeIndex([
        stamp
        for day in pd.bdate_range('2024-01-02', periods=20)
        for stamp in pd.date_range(day.tz_localize('UTC') + pd.Timedelta(hours=14, minutes=30),
                                   periods=78, freq='5min')
    ])
    close = 100 + np.arange(len(index)) * .02 + np.sin(np.arange(len(index)) / 9)
    frame = pd.DataFrame(dict(open=close, high=close + 1, low=close - 1,
                              close=close, volume=1000), index=index)
    result = compare_history(frame, symbol='AAPL', end=1500)
    assert result['same_history_equal']
    assert result['prefix_causal']
    assert result['runtime_window_bars'] < 1500
    assert result['runtime_window_equal']
    assert result['runtime_window_mismatch_columns'] == []
    assert not result['truncated_history_equal']
