from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from ai_trading.config import management
from ai_trading.core import alpaca_client as clients, bot_engine as bot


@pytest.fixture
def setup(monkeypatch):
    state = NS(trading_client=None, data_client=None)
    monkeypatch.setattr(clients, '_get_bot_engine_module', lambda: state)
    monkeypatch.setattr(clients, '_get_bot_logger_once', lambda: Mock())
    monkeypatch.setattr(clients, '_get_active_alpaca_api_module', lambda: NS(ALPACA_AVAILABLE=True))
    monkeypatch.setattr(clients, '_set_alpaca_service_available', Mock())
    monkeypatch.setattr(clients, 'log_env_diag', Mock())
    monkeypatch.setattr(clients, 'get_env', lambda key, default=None, **kw: 'paper' if key == 'EXECUTION_MODE' else default)
    monkeypatch.setattr(clients, 'get_api_error_cls', lambda: ValueError)
    monkeypatch.setattr(bot, '_ensure_alpaca_env_or_raise', lambda: ('test', 'test', 'https://paper-api.alpaca.markets'))
    trading = Mock(return_value=NS(list_positions=lambda: []))
    data = Mock(return_value=object())
    monkeypatch.setattr(clients, 'get_trading_client_cls', lambda: trading)
    monkeypatch.setattr(clients, 'get_data_client_cls', lambda: data)
    monkeypatch.setattr(clients, '_validate_trading_api', lambda api: True)
    return state, trading, data


def test_initialization_uses_paper_endpoint_and_reuses_client(setup):
    state, trading, data = setup
    assert clients._initialize_alpaca_clients() is True
    assert state.trading_client is trading.return_value and state.data_client is data.return_value
    assert trading.call_args.kwargs['paper'] is True
    assert trading.call_args.kwargs['url_override'] == 'https://paper-api.alpaca.markets'
    assert clients._initialize_alpaca_clients() is True
    trading.assert_called_once()


@pytest.mark.parametrize('failure', ['disabled', 'credentials', 'validation', 'constructor'])
def test_incomplete_initialization_cannot_attach_clients(setup, monkeypatch, failure):
    state, trading, _ = setup
    if failure == 'disabled':
        monkeypatch.setattr(clients, 'get_env', lambda key, default=None, **kw: 'disabled')
    elif failure == 'credentials':
        monkeypatch.setattr(bot, '_ensure_alpaca_env_or_raise', lambda: (None, None, 'paper'))
    elif failure == 'validation':
        monkeypatch.setattr(clients, '_validate_trading_api', lambda api: False)
    else:
        trading.side_effect = ValueError('invalid client')
    assert clients._initialize_alpaca_clients() is False
    assert state.trading_client is None and state.data_client is None


def test_environment_resolution_retries_once_without_overriding_env(setup, monkeypatch):
    resolver = Mock(side_effect=[ValueError('missing'), ('test', 'test', 'paper')])
    reload = Mock()
    monkeypatch.setattr(bot, '_ensure_alpaca_env_or_raise', resolver)
    monkeypatch.setattr(management, 'reload_env', reload)
    assert clients._initialize_alpaca_clients() is True
    reload.assert_called_once_with(override=False)
    assert resolver.call_count == 2


def test_attach_keeps_existing_context_client_and_attaches_new_one(setup, monkeypatch):
    state, _, _ = setup
    monkeypatch.setattr(clients._alpaca_api, 'ALPACA_AVAILABLE', True)
    ctx = NS(api=None)
    clients.ensure_alpaca_attached(ctx)
    assert ctx.api is state.trading_client
    existing = object()
    ctx.api = existing
    clients.ensure_alpaca_attached(ctx)
    assert ctx.api is existing
