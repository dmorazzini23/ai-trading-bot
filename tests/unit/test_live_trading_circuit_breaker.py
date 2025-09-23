"""Tests for the live trading execution engine circuit breaker behaviour."""

from ai_trading.execution import live_trading


def _make_engine():
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)
    engine.retry_config = {
        'max_attempts': 3,
        'base_delay': 0,
        'max_delay': 0,
        'exponential_base': 1,
    }
    engine.circuit_breaker = {
        'failure_count': 0,
        'max_failures': 1,
        'reset_time': 300,
        'last_failure': None,
        'is_open': False,
    }
    engine.stats = {
        'retry_count': 0,
        'circuit_breaker_trips': 0,
    }
    engine.is_initialized = True
    engine.trading_client = None
    return engine


def test_circuit_breaker_closes_after_success():
    engine = _make_engine()

    # Trip the breaker on failure and ensure it blocks subsequent work.
    engine._handle_execution_failure(RuntimeError('forced failure'))
    assert engine.circuit_breaker['is_open'] is True
    assert not engine._pre_execution_checks()

    # Successful execution should immediately clear breaker state.
    result = engine._execute_with_retry(lambda: 'ok')
    assert result == 'ok'
    assert engine.circuit_breaker['failure_count'] == 0
    assert engine.circuit_breaker['is_open'] is False
    assert engine.circuit_breaker['last_failure'] is None
    assert engine._pre_execution_checks()
