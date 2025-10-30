from __future__ import annotations

import types


def test_live_trading_executionengine_alias_has_broker_sync():
    # Import within test to ensure module-level aliases are applied
    from ai_trading.execution import ExecutionEngine as SelectedEngine  # type: ignore
    from ai_trading.execution.live_trading import (
        ExecutionEngine as LiveModuleEngine,  # type: ignore
        LiveTradingExecutionEngine,
    )

    # The live module should export the subclass for runtime use
    assert LiveModuleEngine is LiveTradingExecutionEngine

    # The selector should ultimately point to a concrete class type
    assert isinstance(SelectedEngine, type)

    # Instance created from the exported class must provide broker sync hook
    engine = LiveModuleEngine(ctx=None)
    assert hasattr(engine, "_fetch_broker_state"), "Live engine missing _fetch_broker_state()"

