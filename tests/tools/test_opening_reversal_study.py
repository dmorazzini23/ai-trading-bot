import numpy as np
import pandas as pd

from ai_trading.tools.opening_reversal_study import build_opportunities, evaluate
from ai_trading.utils.market_calendar import is_trading_day, session_info


def test_prior_session_threshold_and_latency_are_causal():
    parts = []
    for day in pd.date_range('2024-01-02', '2024-02-15'):
        if not is_trading_day(day.date()):
            continue
        session = session_info(day.date())
        index = pd.date_range(session.start_utc, session.end_utc, freq='min', inclusive='left')
        frame = pd.DataFrame({'open': 100., 'high': 101., 'low': 99., 'close': 100.}, index=index)
        parts.append(frame)
    frame = pd.concat(parts)
    target = parts[20].index
    frame.loc[target[29], 'close'] = 98
    frame.loc[target[31], 'open'] = 99
    frame.loc[target[151], 'open'] = 100
    rows, rejected = build_opportunities(frame, 'SPY', '2024-01-02', '2024-02-15')
    assert rejected['warmup_complete_sessions'] == 20
    assert rows[0]['selected'] is True
    assert rows[0]['threshold'] == -.01
    assert rows[0]['entry'] == target[31].isoformat()
    assert rows[0]['exit'] == target[151].isoformat()
    frame.loc[target[200]:, 'high'] = 200
    changed, _ = build_opportunities(frame, 'SPY', '2024-01-02', '2024-02-15')
    assert changed[0] == rows[0]


def test_reversal_stopping_rule_and_cost_stress():
    data = pd.DataFrame([{'fold': str(fold), 'session': str(fold), 'symbol': str(symbol), 'selected': i % 2 == 0, 'gross_bps': 30 if i % 2 == 0 else -30} for fold in range(8) for symbol in range(6) for i in range(10)])
    result = evaluate(data)
    assert result['decision'] == 'eligible_for_untouched_test_planning'
    assert result['aggregate']['net_per_opportunity_bps'] == 10
    assert result['session_bootstrap_95_bps'] == [10, 10]
    data['gross_bps'] = np.zeros(len(data))
    assert evaluate(data)['decision'] == 'retired'
