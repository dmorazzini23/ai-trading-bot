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
    assert np.isfinite(_feature_frame(short, symbol='AAPL').to_numpy()).all()
    with pytest.raises(ValueError, match='non-finite'):
        build_day_sleeve_features(short)
