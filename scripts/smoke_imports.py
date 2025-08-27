"""
Smoke test script to validate all key modules import correctly.

This script imports key modules and instantiates critical classes to ensure
hard dependencies are present and import guards have been removed properly.
"""
import sys

def test_import(module_name: str, description: str='') -> tuple[bool, str]:
    """Test importing a module."""
    try:
        __import__(module_name)
        return (True, f'✅ {module_name} {description}')
    except (KeyError, ValueError, TypeError) as e:
        return (False, f'❌ {module_name} {description}: {e}')

def test_class_instantiation(module_name: str, class_name: str, args=None, kwargs=None, description: str='') -> tuple[bool, str]:
    """Test importing and instantiating a class."""
    args = args or []
    kwargs = kwargs or {}
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        cls(*args, **kwargs)
        return (True, f'✅ {module_name}.{class_name} {description}')
    except (KeyError, ValueError, TypeError) as e:
        return (False, f'❌ {module_name}.{class_name} {description}: {e}')

def main():
    """Run smoke tests for critical imports."""
    tests = []
    tests.append(test_import('ai_trading', '- core package'))
    tests.append(test_import('ai_trading.monitoring', '- monitoring module'))
    tests.append(test_import('ai_trading.execution.production_engine', '- production engine'))
    tests.append(test_import('ai_trading.config.management', '- config management'))
    tests.append(test_import('ai_trading.config.settings', '- settings'))
    tests.append(test_import('ai_trading.core.bot_engine', '- bot engine'))
    tests.append(test_import('ai_trading.execution.live_trading', '- live trading'))
    tests.append(test_import('ai_trading.rebalancer', '- rebalancer'))
    tests.append(test_import('ai_trading.signals', '- signals'))
    tests.append(test_import('ai_trading.utils.base', '- utils'))
    tests.append(test_import('ai_trading.integrations.rate_limit', '- rate limiter'))
    try:
        from ai_trading.monitoring import MetricsCollector, PerformanceMonitor
        tests.append((True, '✅ MetricsCollector and PerformanceMonitor imported successfully'))
    except ImportError as e:
        tests.append((False, f'❌ Failed to import monitoring classes: {e}'))
    tests.append(test_class_instantiation('ai_trading.monitoring', 'MetricsCollector', description='- metrics collector instantiation'))
    tests.append(test_class_instantiation('ai_trading.monitoring', 'PerformanceMonitor', description='- performance monitor instantiation'))
    tests.append(test_import('pandas_market_calendars', '- market calendars'))
    tests.append(test_import('alpaca_trade_api.rest', '- Alpaca legacy API'))
    tests.append(test_import('ai_trading.portfolio.optimizer', '- portfolio optimizer'))
    tests.append(test_import('ai_trading.execution.transaction_costs', '- transaction costs'))
    tests.append(test_import('scripts.strategy_allocator', '- strategy allocator'))
    tests.append(test_class_instantiation('ai_trading.config.management', 'TradingConfig', description='- config with all new attributes'))
    tests.append(test_class_instantiation('ai_trading.integrations.rate_limit', 'RateLimiter', description='- rate limiter with config support'))
    try:
        from ai_trading.config.management import TradingConfig
        cfg = TradingConfig.from_env()
        required_attrs = ['trading_mode', 'alpaca_base_url', 'sleep_interval', 'max_retries', 'backoff_factor', 'max_backoff_interval', 'pct', 'MODEL_PATH', 'scheduler_iterations', 'scheduler_sleep_seconds', 'window', 'enabled', 'capacity', 'refill_rate', 'queue_timeout']
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(cfg, attr):
                missing_attrs.append(attr)
        if missing_attrs:
            tests.append((False, f'❌ TradingConfig missing attributes: {missing_attrs}'))
        else:
            tests.append((True, '✅ TradingConfig has all required attributes'))
        safe_dict = cfg.to_dict(safe=True)
        if 'ALPACA_API_KEY' in safe_dict and safe_dict['ALPACA_API_KEY'] == '***REDACTED***':
            tests.append((True, '✅ TradingConfig.to_dict(safe=True) redacts secrets'))
        else:
            tests.append((False, '❌ TradingConfig.to_dict(safe=True) does not redact secrets properly'))
    except (OSError, PermissionError, KeyError, ValueError, TypeError) as e:
        tests.append((False, f'❌ TradingConfig.from_env() failed: {e}'))
    try:
        from ai_trading.core.bot_engine import StrategyAllocator
        allocator = StrategyAllocator()
        if hasattr(allocator._alloc, 'allocate'):
            tests.append((True, '✅ StrategyAllocator resolves to real implementation'))
        else:
            tests.append((False, '❌ StrategyAllocator missing allocate method'))
    except (KeyError, ValueError, TypeError) as e:
        tests.append((False, f'❌ StrategyAllocator instantiation failed: {e}'))
    passed = 0
    failed = 0
    for success, message in tests:
        if success:
            passed += 1
        else:
            failed += 1
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)
if __name__ == '__main__':
    main()