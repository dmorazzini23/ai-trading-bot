import numpy as np
import pandas as pd
import pytest

from ai_trading.tools.momentum_horizon_study import evaluate, opportunities


def test_grid_next_bar_prices_gaps_and_fold_boundaries():
    index = pd.date_range('2026-03-30 13:30Z', periods=390, freq='min')
    frame = pd.DataFrame({k: np.full(390, 100.0) for k in ('open', 'high', 'low', 'close')}, index=index)
    frame.loc[index[60], 'open'] = 101
    features = pd.DataFrame({'sma_spread': np.ones(390)}, index=index)
    folds = [{'fold_index': 1, 'test_start': index[0].isoformat(), 'test_end': index[-1].isoformat()}]
    rows, rejected = opportunities(frame, features, folds, 'AAPL')
    assert len(rows) == 11
    assert rows[0]['signal'] == index[29].isoformat()
    assert rows[0]['entry'] == index[30].isoformat()
    assert rows[0]['exit'] == index[60].isoformat()
    assert rows[0]['gross_bps'] == pytest.approx(100)
    assert rejected == {'exit_outside_session': 1}
    assert all(a['exit'] <= b['entry'] for a, b in zip(rows, rows[1:]))
    missing, reasons = opportunities(frame.drop(index[45]), features, folds, 'AAPL')
    assert len(missing) == 10
    assert reasons['missing_contiguous_bars'] == 1
    limited = [dict(folds[0], test_end=index[59].isoformat())]
    assert opportunities(frame, features, limited, 'AAPL')[0] == []
    with pytest.raises(ValueError, match='overlapping'):
        opportunities(frame, features, folds * 2, 'AAPL')


def test_decision_costs_and_session_bootstrap():
    rows = [{'fold': fold, 'session': f'2026-04-{fold:02d}', 'symbol': symbol, 'momentum': i % 2 == 0, 'gross_bps': 20.0 if i % 2 == 0 else -20.0} for fold in range(1, 6) for symbol in ('AAPL', 'AMZN', 'MSFT') for i in range(30)]
    data = pd.DataFrame(rows)
    report = evaluate(data)
    assert report['decision'] == 'eligible_for_untouched_test_planning'
    assert report['aggregate']['momentum']['net_per_opportunity_bps'] == 7
    assert report['aggregate']['always_long']['net_per_opportunity_bps'] == -6
    assert report['session_block_bootstrap_95_interval_bps'] == [7, 7]
    assert report['bootstrap_sessions'] == 5
    data['gross_bps'] = 1.0
    assert evaluate(data)['decision'] == 'retired'
    assert evaluate(data.head(10))['decision'] == 'inconclusive'


def test_canonical_feature_is_causal():
    from ai_trading.tools.train_replay_aligned_model import _feature_frame

    index = pd.date_range('2026-03-30 13:30Z', periods=390, freq='min')
    frame = pd.DataFrame({k: np.linspace(100, 102, 390) for k in ('open', 'high', 'low', 'close')}, index=index)
    frame['volume'] = 1000
    original = _feature_frame(frame, symbol='AAPL')['sma_spread']
    frame.loc[index[250]:, 'close'] *= 10
    changed = _feature_frame(frame, symbol='AAPL')['sma_spread']
    pd.testing.assert_series_equal(original.iloc[:250], changed.iloc[:250])
