from __future__ import annotations


def test_live_trading_executionengine_alias_has_broker_sync() -> None:
    """Live trading module should export broker-sync capable engine."""

    from ai_trading.execution import ExecutionEngine as SelectedEngine
    from ai_trading.execution.live_trading import (
        ExecutionEngine as LiveModuleEngine,
        LiveTradingExecutionEngine,
    )

    assert LiveModuleEngine is LiveTradingExecutionEngine
    assert isinstance(SelectedEngine, type)
    engine = LiveModuleEngine(ctx=None)
    fetch_hook = getattr(engine, "_fetch_broker_state", None)
    assert callable(fetch_hook), "Live engine missing _fetch_broker_state()"
