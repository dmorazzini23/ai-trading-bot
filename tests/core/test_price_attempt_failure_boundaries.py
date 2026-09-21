from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from ai_trading import alpaca_api
from ai_trading.core import bot_engine as bot


@pytest.fixture
def price_request(monkeypatch):
    get = Mock()
    monkeypatch.setattr(bot, '_alpaca_symbols', lambda: (get, None))
    monkeypatch.setattr(bot, 'data_client', None)
    monkeypatch.setattr(bot, '_PRICE_SOURCE', {})
    monkeypatch.setattr(alpaca_api, '_set_alpaca_service_available', Mock())
    monkeypatch.setattr(bot.runtime_state, 'update_data_provider_state', Mock())
    return get


@pytest.mark.parametrize('kind', ['trade', 'quote', 'minute_close'])
@pytest.mark.parametrize('error', [404, 401, 403, 500, 'auth', 'invalid'])
def test_failed_price_attempt_is_cached_without_fabricated_price(price_request, kind, error):
    price_request.side_effect = (bot.AlpacaAuthenticationError('unauthorized') if error == 'auth'
                          else ValueError('invalid response') if error == 'invalid'
                          else bot.AlpacaOrderHTTPError(error, 'price_request failed'))
    attempt = getattr(bot, '_attempt_alpaca_' + kind)
    cache = {}
    first = attempt('AAPL', 'iex', cache)
    assert first[0] is None
    assert attempt('AAPL', 'iex', cache) == first
    price_request.assert_called_once()
    if kind != 'minute_close' and error in {401, 403, 'auth'}:
        assert cache['alpaca_auth_failed'] is True


@pytest.mark.parametrize('kind,payload,expected', [
    ('trade', {'trade': {'p': 101.}}, 101.),
    ('minute_close', {'bar': {'c': 102.}}, 102.),
    ('minute_close', {'bars': {'close': 103.}}, 103.),
    ('minute_close', {'AAPL': {'c': 104.}}, 104.),
    ('minute_close', {'bar': {'c': -1.}}, None),
])
def test_price_payload_shapes_and_feed_forwarding(price_request, kind, payload, expected):
    price_request.return_value = payload
    cache = {}
    result = getattr(bot, '_attempt_alpaca_' + kind)('AAPL', 'iex', cache)
    assert result[0] == expected
    assert price_request.call_args.kwargs['params']['feed'] == 'iex'


@pytest.mark.parametrize('kind', ['trade', 'quote', 'minute_close'])
def test_invalid_feed_does_not_request_data(price_request, kind):
    result = getattr(bot, '_attempt_alpaca_' + kind)('AAPL', 'invalid', {})
    assert result[0] is None and result[1].endswith('invalid_feed')
    price_request.assert_not_called()


@pytest.mark.parametrize('payload', [
    {'ask_price': 101., 'bid_price': 99.},
    SimpleNamespace(ask_price=101., bid_price=99.),
])
def test_sdk_quotes_use_same_extraction_and_cache_without_http(price_request, monkeypatch, payload):
    stamp = datetime.now(UTC).isoformat()
    if isinstance(payload, dict):
        payload['timestamp'] = stamp
    else:
        payload.timestamp = stamp
    monkeypatch.setattr(bot, 'data_client', object())
    monkeypatch.setattr(bot, '_stock_quote_request_ready', lambda: True)
    fetch = Mock(return_value=payload)
    monkeypatch.setattr(bot, '_fetch_quote', fetch)
    cache = {}
    result = bot._attempt_alpaca_quote('AAPL', 'iex', cache)
    assert result[0] == 101.
    assert cache['quote_values']['alpaca_ask'] == 101.
    assert bot._attempt_alpaca_quote('AAPL', 'iex', cache) == result
    fetch.assert_called_once()
    price_request.assert_not_called()


def test_startup_prefetch_retries_only_failed_symbols(monkeypatch):
    calls = []
    def fetch(symbol, **kwargs):
        calls.append(symbol)
        if symbol == 'AMZN' and calls.count(symbol) == 1:
            raise ValueError('temporary failure')
        return pd.DataFrame({'close': [100.]})
    monkeypatch.setattr(bot.data_fetcher_module, 'get_daily_df', fetch)
    monkeypatch.setattr(bot, 'get_env', lambda key, default=None, **kwargs:
                        1. if key.endswith('MIN_SUCCESS_RATIO') else default)
    monkeypatch.setattr(bot.time, 'sleep', lambda seconds: None)
    result = bot._startup_prefetch_daily_data(SimpleNamespace(data_fetcher=object()), ['aapl', 'AAPL', '', 'AMZN'])
    assert result['success_count'] == 2 and result['failure_count'] == 0
    assert result['attempts'] == 2
    assert calls == ['AAPL', 'AMZN', 'AMZN']


def test_startup_prefetch_reports_exhaustion_and_missing_context(monkeypatch):
    monkeypatch.setattr(bot, 'get_env', lambda key, default=None, **kwargs: default)
    monkeypatch.setattr(bot.data_fetcher_module, 'get_daily_df', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(bot.time, 'sleep', lambda seconds: None)
    result = bot._startup_prefetch_daily_data(SimpleNamespace(data_fetcher=object()), ['AAPL'])
    assert result['success_count'] == 0 and result['failed_reasons'] == {'AAPL': 'empty'}
    assert result['attempts'] == 2
    assert bot._startup_prefetch_daily_data(object(), ['AAPL'])['reason'] == 'missing_data_fetcher'
    assert bot._startup_prefetch_daily_data(SimpleNamespace(data_fetcher=object()), [])['reason'] == 'no_symbols'
