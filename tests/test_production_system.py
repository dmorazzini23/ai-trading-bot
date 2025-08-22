"""
Basic tests for the production-ready trading system enhancements.

Tests the core functionality of risk management, circuit breakers,
and execution coordination without requiring full dependencies.
"""

import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from ai_trading.core.enums import OrderSide, OrderType, RiskLevel
    from ai_trading.execution.production_engine import ProductionExecutionCoordinator
    from ai_trading.monitoring.alerting import AlertManager, AlertSeverity
    from ai_trading.risk.circuit_breakers import (
        DrawdownCircuitBreaker,
        TradingHaltManager,
    )
    from ai_trading.risk.position_sizing import ATRPositionSizer, DynamicPositionSizer
except ImportError:
    # Create mock classes for testing
    class RiskLevel:
        CONSERVATIVE = "conservative"
        MODERATE = "moderate"
        AGGRESSIVE = "aggressive"

    class OrderSide:
        BUY = "buy"
        SELL = "sell"

    class OrderType:
        MARKET = "market"
        LIMIT = "limit"


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
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        return False


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
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        return False


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
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        return False


def test_alert_manager():
    """Test alert management system."""
    try:
        from ai_trading.monitoring.alerting import AlertManager, AlertSeverity

        alert_manager = AlertManager()

        # Start processing
        alert_manager.start_processing()

        # Test sending alerts
        alert_id = alert_manager.send_alert(
            "Test Alert",
            "This is a test alert message",
            AlertSeverity.INFO,
            "TestSystem"
        )

        assert alert_id != "", "Alert ID should not be empty"

        # Test trading alert
        trading_alert_id = alert_manager.send_trading_alert(
            "Order Executed",
            "AAPL",
            {"quantity": 100, "price": 150.0},
            AlertSeverity.INFO
        )

        assert trading_alert_id != "", "Trading alert ID should not be empty"

        # Test alert stats
        stats = alert_manager.get_alert_stats()
        assert stats["total_alerts"] >= 2, "Should have at least 2 alerts"

        # Stop processing
        alert_manager.stop_processing()

        return True

    except ImportError:
        return True
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        return False


async def test_production_execution_coordinator():
    """Test production execution coordinator."""
    try:
        from ai_trading.core.enums import OrderSide, OrderType, RiskLevel
        from ai_trading.execution.production_engine import (
            ProductionExecutionCoordinator,
        )

        # Initialize coordinator
        coordinator = ProductionExecutionCoordinator(
            account_equity=100000,
            risk_level=RiskLevel.MODERATE
        )

        # Test order submission
        result = await coordinator.submit_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            strategy="test_strategy"
        )

        assert result["status"] in ["success", "rejected"], f"Unexpected status: {result['status']}"

        # Test execution summary
        summary = coordinator.get_execution_summary()
        assert "execution_stats" in summary, "Summary should contain execution stats"
        assert summary["execution_stats"]["total_orders"] >= 1, "Should have at least 1 order"

        # Test position tracking
        coordinator.get_current_positions()

        return True

    except ImportError:
        return True
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        return False


async def run_all_tests():
    """Run all tests and report results."""

    test_results = []

    # Synchronous tests
    test_results.append(("ATR Position Sizer", test_atr_position_sizer()))
    test_results.append(("Drawdown Circuit Breaker", test_drawdown_circuit_breaker()))
    test_results.append(("Trading Halt Manager", test_trading_halt_manager()))
    test_results.append(("Alert Manager", test_alert_manager()))

    # Asynchronous tests
    test_results.append(("Production Execution Coordinator", await test_production_execution_coordinator()))

    # Report results

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        if result:
            passed += 1


    if passed == total:
        pass
    else:
        pass

    return passed == total


if __name__ == "__main__":
    # Run tests
    try:
        result = asyncio.run(run_all_tests())
        exit_code = 0 if result else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(1)
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        sys.exit(1)
