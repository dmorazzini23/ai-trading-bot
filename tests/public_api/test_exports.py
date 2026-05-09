import importlib
from importlib import import_module
import pytest

EXPECTED = {
    'ai_trading': [
        'DataFetchError',
        'ExecutionEngine',
        'alpaca_api',
        'app',
        'audit',
        'capital_scaling',
        'config',
        'core',
        'data',
        'data_validation',
        'execution',
        'indicator_manager',
        'indicators',
        'logging',
        'main',
        'meta_learning',
        'ml_model',
        'paths',
        'portfolio',
        'position_sizing',
        'predict',
        'production_system',
        'rebalancer',
        'settings',
        'signals',
        'strategy_allocator',
        'trade_logic',
        'utils',
    ],
    'ai_trading.config': sorted([
        '_require_env_vars',
        'AlpacaConfig',
        'get_max_drawdown_threshold',
        'META_LEARNING_BOOTSTRAP_ENABLED',
        'META_LEARNING_BOOTSTRAP_WIN_RATE',
        'META_LEARNING_MIN_TRADES_REDUCED',
        'MODE_PARAMETERS',
        'ORDER_FILL_RATE_TARGET',
        'SENTIMENT_API_KEY',
        'SENTIMENT_API_URL',
        'SENTIMENT_ENHANCED_CACHING',
        'SENTIMENT_FALLBACK_SOURCES',
        'SENTIMENT_RECOVERY_TIMEOUT_SECS',
        'SENTIMENT_SUCCESS_RATE_TARGET',
        'Settings',
        'TradingConfig',
        'broker_keys',
        'derive_cap_from_settings',
        'get_alpaca_config',
        'get_env',
        'get_settings',
        'log_config',
        'reload_env',
        'require_env_vars',
        'validate_alpaca_credentials',
        'validate_environment',
        'validate_env_vars',
    ]),
    'ai_trading.core': [
        'AssetClass',
        'OrderSide',
        'OrderStatus',
        'OrderType',
        'RiskLevel',
        'TRADING_CONSTANTS',
        'TimeFrame',
    ],
    'ai_trading.utils': [
        'EASTERN_TZ',
        'HTTP_TIMEOUT',
        'OptionalDependencyError',
        'SUBPROCESS_TIMEOUT_DEFAULT',
        'capital_scaling',
        'clamp_request_timeout',
        'clamp_timeout',
        'datetime',
        'device',
        'ensure_utc',
        'get_free_port',
        'get_latest_close',
        'get_pid_on_port',
        'health_check',
        'http',
        'is_market_open',
        'log_warning',
        'market_open_between',
        'model_lock',
        'module_ok',
        'paths',
        'portfolio_lock',
        'retry',
        'safe_subprocess_run',
        'safe_to_datetime',
        'sleep',
        'timing',
        'validate_ohlcv',
    ],
}


def test_exports_lists_are_stable():
    for module_name, expected in EXPECTED.items():
        mod = import_module(module_name)
        assert hasattr(mod, '__all__')
        assert sorted(mod.__all__) == expected


def test_package_import_does_not_patch_runtime_override_helpers():
    import ai_trading
    from ai_trading.config import management

    original_clear = management.clear_runtime_env_overrides

    importlib.reload(ai_trading)

    assert management.clear_runtime_env_overrides is original_clear
    assert not hasattr(management, "_test_override_guard_installed")


def test_legacy_live_api_exports_warn_as_research_only():
    import ai_trading

    importlib.reload(ai_trading)
    vars(ai_trading).pop("predict", None)
    vars(ai_trading).pop("trade_logic", None)
    with pytest.warns(DeprecationWarning, match="research utilities"):
        assert ai_trading.predict is import_module("ai_trading.predict")
    vars(ai_trading).pop("trade_logic", None)
    with pytest.warns(DeprecationWarning, match="research utilities"):
        assert ai_trading.trade_logic is import_module("ai_trading.trade_logic")


def test_root_execution_engine_export_uses_runtime_selector(monkeypatch):
    import ai_trading
    import ai_trading.execution as execution_mod

    class SelectedExecutionEngine:
        pass

    monkeypatch.setattr(
        execution_mod,
        "select_execution_engine",
        lambda: SelectedExecutionEngine,
    )
    vars(ai_trading).pop("ExecutionEngine", None)

    assert ai_trading.ExecutionEngine is SelectedExecutionEngine
    assert "ExecutionEngine" not in vars(ai_trading)
