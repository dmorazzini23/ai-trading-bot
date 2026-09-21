import json
from datetime import date
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pandas as pd
import pytest

from ai_trading.core import bot_engine as bot


@pytest.mark.parametrize('retry', [False, True])
@pytest.mark.parametrize('layout', ['flat', 'multi'])
def test_historical_minutes_normalize_sort_deduplicate_and_retry_iex(retry, layout):
    fetcher = bot.DataFetcher.__new__(bot.DataFetcher)
    fetcher.settings = NS(alpaca_execution_feed='sip')
    fetcher._resolve_feed = lambda value: value
    frame = pd.DataFrame(dict(Open=[100., 101.], High=[102., 102.], Low=[99., 99.],
                              Close=[101., 100.], Volume=[1000, 2000]),
                         index=pd.to_datetime(['2024-01-02T15:01Z', '2024-01-02T15:00Z']))
    if layout == 'multi':
        frame = pd.concat({'AAPL': frame}, axis=1)
    outcomes = [frame, frame]
    if retry:
        outcomes.insert(0, bot.APIError(json.dumps({'message': 'subscription does not permit'})))
    fetcher._get_stock_bars = Mock(side_effect=outcomes)
    result = fetcher.get_historical_minute(NS(data_client=object()), 'AAPL', date(2024, 1, 2), date(2024, 1, 3))
    assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert result.index.is_monotonic_increasing and result.index.is_unique
    assert len(result) == 2
    assert result.close.tolist() == [100., 101.]
    if retry:
        assert fetcher._get_stock_bars.call_args_list[1].kwargs['feed'] == 'iex'


def test_unreadable_day_does_not_destroy_other_days():
    fetcher = bot.DataFetcher.__new__(bot.DataFetcher)
    fetcher.settings = NS(data_feed='iex')
    fetcher._resolve_feed = lambda value: value
    fetcher._get_stock_bars = Mock(side_effect=ValueError('malformed'))
    assert fetcher.get_historical_minute(NS(data_client=object()), 'AAPL', date(2024, 1, 2), date(2024, 1, 2)) is None
