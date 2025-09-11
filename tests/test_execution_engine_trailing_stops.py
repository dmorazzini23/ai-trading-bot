from ai_trading.execution.engine import ExecutionEngine


def test_check_trailing_stops_noop():
    eng = ExecutionEngine()
    # Should not raise even when no handler is present
    eng.check_trailing_stops()
