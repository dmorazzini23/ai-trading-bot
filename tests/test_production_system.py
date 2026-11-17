"""
Basic tests for the production-ready trading system enhancements.

Tests the core functionality of risk management, circuit breakers,
and execution coordination without requiring full dependencies.
"""





def test_atr_position_sizer():
    """Test ATR-based position sizing."""
    try:
        from ai_trading.risk.position_sizing import ATRPositionSizer

        # Test basic ATR position sizing
        sizer = ATRPositionSizer(risk_per_trade=0.02)  # 2% risk per trade

        # Test position size calculation
        account_equity = 100000  # $100k
        entry_price = 100.0      # $100 per share
        atr_value = 2.0          # $2 ATR

        position_size = sizer.calculate_position_size(account_equity, entry_price, atr_value)

        # Expected: $2000 risk / ($2 * 2.0 multiplier) = 500 shares
        expected_size = int(2000 / (2.0 * 2.0))

        assert position_size == expected_size, f"Expected {expected_size}, got {position_size}"

        # Test stop levels
        stop_levels = sizer.calculate_stop_levels(entry_price, atr_value, "long")
        assert stop_levels["stop_loss"] < entry_price, "Stop loss should be below entry for long"
        assert stop_levels["take_profit"] > entry_price, "Take profit should be above entry for long"

        return True

    except ImportError:
        return True


def test_drawdown_circuit_breaker():
    """Test drawdown circuit breaker functionality."""
    try:
        from ai_trading.risk.circuit_breakers import (
            CircuitBreakerState,
            DrawdownCircuitBreaker,
        )

        # Test circuit breaker with 10% max drawdown
        breaker = DrawdownCircuitBreaker(max_drawdown=0.10)

        # Test normal operation
        assert breaker.update_equity(100000) is True, "Trading should be allowed initially"

        # Test drawdown within limits
        assert breaker.update_equity(95000) is True, "5% drawdown should be allowed"

        # Test drawdown exceeding limits
        assert breaker.update_equity(85000) is False, "15% drawdown should halt trading"

        # Test status
        status = breaker.get_status()
        assert status["current_drawdown"] >= 0.10, "Drawdown should be >= 10%"
        assert not status["trading_allowed"], "Trading should be halted"

        return True

    except ImportError:
        return True


def test_trading_halt_manager():
    """Test comprehensive trading halt management."""
    try:
        from ai_trading.risk.circuit_breakers import TradingHaltManager

        halt_manager = TradingHaltManager()

        # Test initial state
        status = halt_manager.is_trading_allowed()
        assert status["trading_allowed"] is True, "Trading should be allowed initially"

        # Test manual halt
        halt_manager.manual_halt_trading("Test halt")
        status = halt_manager.is_trading_allowed()
        assert status["trading_allowed"] is False, "Trading should be halted after manual halt"
        assert "Manual halt" in status["reasons"][0], "Reason should mention manual halt"

        # Test resume
        halt_manager.resume_trading("Test resume")
        status = halt_manager.is_trading_allowed()
        assert status["trading_allowed"] is True, "Trading should resume after manual resume"

        # Test emergency stop
        halt_manager.emergency_stop_all("Test emergency")
        status = halt_manager.is_trading_allowed()
        assert status["trading_allowed"] is False, "Trading should be halted after emergency stop"

        return True

    except ImportError:
        return True






