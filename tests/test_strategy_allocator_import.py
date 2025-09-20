from importlib import import_module
import sys

from ai_trading.config.management import get_trading_config


def test_strategy_allocator_module_import_does_not_raise():
    module_name = "ai_trading.strategy_allocator"
    sys.modules.pop(module_name, None)
    module = import_module(module_name)
    assert hasattr(module, "StrategyAllocator")
    allocator = module.StrategyAllocator()
    assert hasattr(allocator, "config")


def test_strategy_allocator_config_isolation():
    from ai_trading import strategy_allocator

    base_cfg = get_trading_config()
    allocator = strategy_allocator.StrategyAllocator()

    assert allocator.config is not base_cfg
    assert allocator.config.signal_confirmation_bars == base_cfg.signal_confirmation_bars

    bumped = base_cfg.signal_confirmation_bars + 1
    allocator.replace_config(signal_confirmation_bars=bumped)

    assert get_trading_config().signal_confirmation_bars == base_cfg.signal_confirmation_bars
