import json
from datetime import date
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pandas as pd
import pytest

from ai_trading.core import bot_engine as bot


@pytest.fixture
def bulk(monkeypatch):
    monkeypatch.setattr(bot, 'get_settings', lambda: NS(alpaca_api_key='unit-key', alpaca_secret_key_plain='unit-secret'))
    monkeypatch.setattr(bot, 'StockHistoricalDataClient', lambda **k: object())
    monkeypatch.setattr(bot.bars, 'StockBarsRequest', lambda **k: NS(**k))
    frame = pd.DataFrame({'symbol': ['AAPL', 'AMZN'], 'Open': [10., 20.],
                          'Close': [11., 21.]}, index=pd.to_datetime(['2024-01-02', '2024-01-02'], utc=True))
    fetch = Mock(return_value=frame)
    monkeypatch.setattr(bot.bars, 'safe_get_stock_bars', fetch)
    return frame, fetch


def run():
    return bot.prefetch_daily_data(['AAPL', 'AMZN'], date(2024, 1, 2), date(2024, 1, 3))


@pytest.mark.parametrize('layout', ['flat', 'multi'])
@pytest.mark.parametrize('retry', [False, True])
def test_bulk_results_preserve_symbol_identity_and_iex_retry(bulk, layout, retry):
    frame, fetch = bulk
    if layout == 'multi':
        frame = pd.concat({s: frame.loc[frame.symbol == s].drop(columns='symbol') for s in ['AAPL', 'AMZN']}, axis=1)
    if retry:
        fetch.side_effect = [bot.APIError(json.dumps({'message': 'subscription does not permit querying recent sip data'})), frame]
    else:
        fetch.return_value = frame
    result = run()
    assert set(result) == {'AAPL', 'AMZN'}
    assert result['AAPL'].close.tolist() == [11.]
    assert result['AMZN'].close.tolist() == [21.]
    assert 'symbol' not in result['AAPL'].columns
    assert result['AAPL'].index.tz is not None
    assert fetch.call_count == 1 + int(retry)
    if retry:
        assert fetch.call_args.args[1].feed == 'iex'


def test_failed_iex_bulk_recovers_each_symbol_independently(bulk):
    frame, fetch = bulk
    fetch.side_effect = [bot.APIError(json.dumps({'message': 'subscription does not permit querying recent sip data'})),
                         ValueError('bulk failure'), frame.iloc[:1], None]
    result = run()
    assert list(result) == ['AAPL']
    assert result['AAPL'].close.tolist() == [11.]
    assert fetch.call_count == 4


@pytest.mark.parametrize('error', [ValueError('bad payload'), bot.APIError(json.dumps({'message': 'unknown error'}))])
def test_failed_bulk_never_manufactures_positive_prices(bulk, error):
    _, fetch = bulk
    fetch.side_effect = error
    result = run()
    assert set(result) == {'AAPL', 'AMZN'}
    assert all((frame.close == 0).all() for frame in result.values())


def test_absent_bulk_response_is_empty(bulk):
    _, fetch = bulk
    fetch.return_value = None
    assert run() == {}
