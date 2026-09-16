import pandas as pd

from ai_trading.core.data_contract import normalize_bars
from ai_trading.models.contracts import DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION
from ai_trading.features.input_provenance import (
    REQUIRED, VERSION, compare_input_contracts, describe_batch, frame_identity,
)


def test_exchange_calendar_excludes_holiday_and_early_close():
    frame = pd.DataFrame({'close': [1, 2, 3, 4]}, index=pd.to_datetime([
        '2024-07-03T16:55:00Z', '2024-07-03T17:00:00Z',
        '2024-07-04T15:00:00Z', '2024-07-05T13:30:00Z']))
    assert normalize_bars(frame, '5Min')['close'].tolist() == [1, 4]
    assert len(normalize_bars(frame, '5Min', rth_only=False)) == 4
    assert len(normalize_bars(frame, '1Day')) == 4


def test_provenance_distinguishes_request_from_effective_source():
    frame = pd.DataFrame({'close': [10.]}, index=pd.to_datetime(['2024-01-02T14:30Z']))
    frame.attrs.update(requested_feed='iex', requested_adjustment='all', data_provider='yahoo')
    record = describe_batch(frame)
    assert record['source_evidence']['requested_adjustment'] == 'all'
    assert record['source_evidence']['effective_adjustment'] is None
    assert not record['effective_adjustment_verified']
    assert frame_identity(frame.copy()) == record['input_sha256']
    changed = frame.copy()
    changed.iloc[0, 0] = 11
    assert frame_identity(changed) != record['input_sha256']


def test_missing_historical_contract_never_matches():
    serving = {'feed': 'iex', 'adjustment': 'all', 'timeframe': '5Min',
               'session_policy': 'canonical_exchange_regular_session_v1',
               'history_policy': {'kind': 'rolling_calendar_days', 'days': 10},
               'finality_policy': {'bar_label': 'start', 'grace_seconds': 2},
               'feature_version': DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION}
    serving['version'] = VERSION
    assert compare_input_contracts(None, serving)['status'] == 'unverified'
    assert compare_input_contracts(serving, serving)['status'] == 'matched'
    training = dict(serving, adjustment='split')
    assert compare_input_contracts(training, serving)['mismatched_fields'] == ['adjustment']
    assert not compare_input_contracts(serving, serving)['qualification_authority']


def test_empty_batch_and_extended_session_evidence():
    assert describe_batch(None)['status'] == 'no_batch'
    frame = pd.DataFrame({'close': [10.]}, index=pd.to_datetime(['2024-01-02T14:30Z']))
    frame.attrs.update(data_feed='iex', effective_adjustment='all',
                       history_policy={'kind': 'rolling_calendar_days', 'days': 10})
    record = describe_batch(frame, rth_only=False, grace_seconds=3)
    assert record['input_contract']['session_policy'] == 'extended_sessions'
    assert record['input_contract']['finality_policy']['grace_seconds'] == 3
    assert record['input_contract']['history_policy']['days'] == 10
