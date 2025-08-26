import importlib
import types

import pytest


def test_lazy_context_model_loaded_once(monkeypatch):
    be = importlib.reload(__import__('ai_trading.core.bot_engine', fromlist=['dummy']))
    load_calls = {'count': 0}
    def fake_load():
        load_calls['count'] += 1
        return object()
    monkeypatch.setattr(be, '_load_required_model', fake_load)

    build_calls = {'count': 0}
    def fake_build_fetcher(params):
        build_calls['count'] += 1
        return object()
    monkeypatch.setattr(be.data_fetcher_module, 'build_fetcher', fake_build_fetcher)

    monkeypatch.setattr(be, '_init_metrics', lambda: None)
    monkeypatch.setattr(be, '_initialize_alpaca_clients', lambda: None)
    monkeypatch.setattr(be, 'ensure_alpaca_attached', lambda ctx: None)
    monkeypatch.setattr(be, 'ExecutionEngine', lambda ctx: object())
    monkeypatch.setattr(be, 'CapitalScalingEngine', lambda params: object())
    monkeypatch.setattr(be, 'get_risk_engine', lambda: types.SimpleNamespace(capital_scaler=None))
    monkeypatch.setattr(be, 'get_allocator', lambda: None)
    monkeypatch.setattr(be, 'get_strategies', lambda: [])
    monkeypatch.setattr(be, 'get_volume_threshold', lambda: 0)
    monkeypatch.setattr(be, 'ENTRY_START_OFFSET', 0)
    monkeypatch.setattr(be, 'ENTRY_END_OFFSET', 0)
    monkeypatch.setattr(be, 'MARKET_OPEN', 0)
    monkeypatch.setattr(be, 'MARKET_CLOSE', 0)
    monkeypatch.setattr(be, 'REGIME_LOOKBACK', 0)
    monkeypatch.setattr(be, 'REGIME_ATR_THRESHOLD', 0.0)
    monkeypatch.setattr(be, 'get_daily_loss_limit', lambda: 0.0)
    monkeypatch.setattr(be, 'DrawdownCircuitBreaker', None)
    monkeypatch.setattr(be, 'CFG', types.SimpleNamespace(max_drawdown_threshold=0))
    monkeypatch.setattr(be, 'get_trade_logger', lambda: None)
    monkeypatch.setattr(be, 'Semaphore', lambda n: object())
    monkeypatch.setattr(be, 'BotContext', types.SimpleNamespace)
    monkeypatch.setattr(be, 'params', {})
    monkeypatch.setattr(be, 'trading_client', object())
    monkeypatch.setattr(be, 'data_client', object())
    monkeypatch.setattr(be, 'signal_manager', object())
    monkeypatch.setattr(be, 'stream', None)
    monkeypatch.setenv('PYTEST_RUNNING', '1')

    wrapper = be.LazyBotContext()
    wrapper._ensure_initialized()
    wrapper._ensure_initialized()

    assert build_calls['count'] == 1
    assert load_calls['count'] == 1


def test_model_loaded_once_across_wrappers(monkeypatch):
    be = importlib.reload(__import__('ai_trading.core.bot_engine', fromlist=['dummy']))
    calls = {'count': 0}
    orig_load = be._load_required_model

    def spy_load():
        calls['count'] += 1
        return orig_load()

    monkeypatch.setattr(be, '_load_required_model', spy_load)

    mod = types.ModuleType('fake_mod')
    mod.get_model = lambda: object()
    import sys
    sys.modules['fake_mod'] = mod
    monkeypatch.setenv('AI_TRADING_MODEL_MODULE', 'fake_mod')
    monkeypatch.delenv('AI_TRADING_MODEL_PATH', raising=False)

    for name in ['_init_metrics', '_initialize_alpaca_clients']:
        monkeypatch.setattr(be, name, lambda: None)
    monkeypatch.setattr(be.data_fetcher_module, 'build_fetcher', lambda params: object())
    monkeypatch.setattr(be, 'ensure_alpaca_attached', lambda ctx: None)
    monkeypatch.setattr(be, 'ExecutionEngine', lambda ctx: object())
    monkeypatch.setattr(be, 'CapitalScalingEngine', lambda params: object())
    monkeypatch.setattr(be, 'get_risk_engine', lambda: types.SimpleNamespace(capital_scaler=None))
    monkeypatch.setattr(be, 'get_allocator', lambda: None)
    monkeypatch.setattr(be, 'get_strategies', lambda: [])
    monkeypatch.setattr(be, 'get_volume_threshold', lambda: 0)
    monkeypatch.setattr(be, 'ENTRY_START_OFFSET', 0)
    monkeypatch.setattr(be, 'ENTRY_END_OFFSET', 0)
    monkeypatch.setattr(be, 'MARKET_OPEN', 0)
    monkeypatch.setattr(be, 'MARKET_CLOSE', 0)
    monkeypatch.setattr(be, 'REGIME_LOOKBACK', 0)
    monkeypatch.setattr(be, 'REGIME_ATR_THRESHOLD', 0.0)
    monkeypatch.setattr(be, 'get_daily_loss_limit', lambda: 0.0)
    monkeypatch.setattr(be, 'DrawdownCircuitBreaker', None)
    monkeypatch.setattr(be, 'CFG', types.SimpleNamespace(max_drawdown_threshold=0))
    monkeypatch.setattr(be, 'get_trade_logger', lambda: None)
    monkeypatch.setattr(be, 'Semaphore', lambda n: object())
    monkeypatch.setattr(be, 'BotContext', types.SimpleNamespace)
    monkeypatch.setattr(be, 'params', {})
    monkeypatch.setattr(be, 'trading_client', object())
    monkeypatch.setattr(be, 'data_client', object())
    monkeypatch.setattr(be, 'signal_manager', object())
    monkeypatch.setattr(be, 'stream', None)
    monkeypatch.setenv('PYTEST_RUNNING', '1')

    be.LazyBotContext()._ensure_initialized()
    be.LazyBotContext()._ensure_initialized()

    assert calls['count'] == 1
