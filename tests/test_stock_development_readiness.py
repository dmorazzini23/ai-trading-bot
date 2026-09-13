import numpy as np
import pandas as pd

from ai_trading.tools.stock_development_readiness import assess, valid_bars


def test_invalid_prices_and_volume_are_not_feature_evidence():
    frame = pd.DataFrame(dict(open=[10]*5, high=[11, 9, 11, 11, 11],
                              low=[9]*5, close=[10, 10, np.nan, 10, 10],
                              volume=[1, 1, 1, 0, -1]))
    assert valid_bars(frame).tolist() == [True, False, False, False, False]


def test_feature_warmup_and_duplicates_do_not_pass():
    start = pd.Timestamp('2024-01-02T14:30:00Z')
    end = start + pd.Timedelta(minutes=20)
    frame = pd.DataFrame(dict(open=10, high=11, low=9, close=10, volume=1),
                         index=pd.date_range(start, end, freq='min', inclusive='left'))
    result = assess(frame, [(start, end)])['counts']
    assert result['complete_timestamp_windows'] == 2
    assert result['feature_input_supported'] == 0
    assert result['feature_history_unavailable'] == 2
    duplicate = assess(pd.concat([frame, frame.iloc[:1]]), [(start, end)])['counts']
    assert duplicate['missing_or_duplicate_window'] == 1


def test_complete_history_supports_inputs_but_not_execution():
    sessions = [(pd.Timestamp(f'2024-01-0{day}T14:30:00Z'),
                 pd.Timestamp(f'2024-01-0{day}T21:00:00Z')) for day in [2, 3, 4]]
    parts = [pd.date_range(start, end, freq='min', inclusive='left') for start, end in sessions]
    index = parts[0].append(parts[1:])
    frame = pd.DataFrame(dict(open=10, high=11, low=9, close=10, volume=1), index=index)
    result = assess(frame, sessions)
    assert result['counts']['feature_input_supported'] == 33
    assert result['counts']['feature_history_unavailable'] == 195
    assert result['feature_status'] == 'necessary_input_support_not_full_feature_or_execution_validation'
