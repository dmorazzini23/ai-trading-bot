import numpy as np
import pandas as pd
import pytest

from ai_trading.features.day_sleeve import build_day_sleeve_features
from ai_trading.features.input_provenance import REQUIRED, VERSION, compare_input_contracts
from ai_trading.training.label_timing import valid_five_minute_labels


@pytest.mark.parametrize('value', [{}, [], False, 0, float('nan')])
def test_malformed_contract_cannot_match_itself(value):
    contract = dict.fromkeys(REQUIRED, value)
    contract['version'] = VERSION
    assert compare_input_contracts(contract, contract)['status'] == 'unverified'


@pytest.mark.parametrize('column,value', [('high', 1), ('low', 500), ('volume', 0), ('open', -1)])
def test_invalid_historical_bar_is_rejected(column, value):
    close = 100 + np.arange(250) * .01
    frame = pd.DataFrame(dict(open=close, high=close+1, low=close-1, close=close, volume=1000),
                         index=pd.date_range('2024-01-02T14:30Z', periods=250, freq='5min'))
    frame.iloc[20, frame.columns.get_loc(column)] = value
    with pytest.raises(ValueError, match='OHLCV'):
        build_day_sleeve_features(frame)


def test_labels_do_not_cross_missing_intervals_or_sessions():
    index = pd.to_datetime(['2024-01-02T14:30Z', '2024-01-02T14:35Z',
                            '2024-01-02T14:45Z', '2024-01-03T14:30Z'])
    assert valid_five_minute_labels(index, 1).tolist() == [True, False, False, False]
    assert not valid_five_minute_labels(index, 2).any()


def test_label_grid_and_early_close():
    off_grid = pd.to_datetime(['2024-01-02T14:31Z', '2024-01-02T14:36Z'])
    assert not valid_five_minute_labels(off_grid, 1).any()
    close = pd.to_datetime(['2024-07-03T16:55Z', '2024-07-03T17:00Z'])
    assert not valid_five_minute_labels(close, 1).any()
