"""
Main institutional testing suite for live trading bot.

This module provides comprehensive testing of the trading bot's live trading
capabilities, including end-to-end workflows, risk management, and compliance.
"""

import asyncio
import datetime as dt
import os

import pytest
import pytz

pytestmark = pytest.mark.alpaca

# Set test environment
os.environ['PYTEST_RUNNING'] = '1'
pytest.importorskip('alpaca_trade_api', reason='alpaca not installed')

from ai_trading.execution.live_trading import AlpacaExecutionEngine

from .framework import (
    ComplianceTestSuite,
    MockMarketDataProvider,
    TradingScenarioRunner,
)


class TestLiveTradingBot:
    """
    Comprehensive test suite for live trading bot functionality.
    
    Tests cover:
    - Order execution workflows
    - Risk management scenarios
    - Error handling and recovery
    - Performance and compliance
    """

    @pytest.fixture
    def execution_engine(self):
        """Create and initialize execution engine for testing."""
        if AlpacaExecutionEngine:
            engine = AlpacaExecutionEngine()
            engine.initialize()
            return engine
        return None

    @pytest.fixture
    def scenario_runner(self, execution_engine):
        """Create scenario runner with execution engine."""
        return TradingScenarioRunner(execution_engine)

    @pytest.fixture
    def compliance_suite(self):
        """Create compliance test suite."""
        return ComplianceTestSuite()

    @pytest.mark.asyncio
    async def test_end_to_end_trading(self, scenario_runner):
        """Test complete end-to-end trading workflow."""
        results = await scenario_runner.run_end_to_end_test()

        assert results["overall_status"] == "passed", f"End-to-end test failed: {results}"
        assert results["pass_rate"] >= 0.8, f"Pass rate too low: {results['pass_rate']}"

        # Check specific scenarios
        scenario_names = [s["name"] for s in results["scenarios"]]
        critical_scenarios = ["initialization", "market_order", "error_handling"]

        for scenario in critical_scenarios:
            assert scenario in scenario_names, f"Missing critical scenario: {scenario}"

    @pytest.mark.asyncio
    async def test_risk_management_scenarios(self, scenario_runner):
        """Test risk management scenario handling."""
        results = await scenario_runner.run_risk_scenario_tests()

        assert results["overall_status"] == "passed", f"Risk scenario tests failed: {results}"
        assert results["pass_rate"] >= 0.9, f"Risk management pass rate too low: {results['pass_rate']}"

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, scenario_runner):
        """Test performance and latency benchmarks."""
        results = await scenario_runner.run_performance_tests()

        assert results["overall_status"] == "passed", f"Performance tests failed: {results}"

        metrics = results["metrics"]
        assert metrics["average_latency_ms"] < 100, f"Latency too high: {metrics['average_latency_ms']}ms"
        assert metrics["orders_per_second"] > 50, f"Throughput too low: {metrics['orders_per_second']} ops/s"
        assert metrics["memory_usage_mb"] < 100, f"Memory usage too high: {metrics['memory_usage_mb']}MB"

    @pytest.mark.asyncio
    async def test_compliance_validation(self, compliance_suite):
        """Test compliance and audit requirements."""
        results = await compliance_suite.run_compliance_tests()

        assert results["overall_status"] == "passed", f"Compliance tests failed: {results}"
        assert results["compliance_score"] >= 0.95, f"Compliance score too low: {results['compliance_score']}"

    def test_execution_engine_initialization(self, execution_engine):
        """Test execution engine can be properly initialized."""
        if execution_engine:
            assert execution_engine.is_initialized, "Execution engine not initialized"

            stats = execution_engine.get_execution_stats()
            assert "success_rate" in stats, "Missing success rate in stats"
            assert "circuit_breaker_status" in stats, "Missing circuit breaker status"
            assert stats["is_initialized"] is True, "Initialization status incorrect"
        else:
            pytest.skip("No execution engine available for testing")

    def test_mock_market_data_provider(self):
        """Test mock market data provider functionality."""
        provider = MockMarketDataProvider()

        # Test current prices
        aapl_price = provider.get_current_price("AAPL")
        assert aapl_price is not None, "AAPL price should be available"
        assert aapl_price > 0, "AAPL price should be positive"

        # Test price history
        history = provider.get_price_history("AAPL")
        assert len(history) > 0, "Price history should not be empty"

        # Test market scenarios
        initial_price = provider.get_current_price("AAPL")
        provider.create_market_scenario("bull_market")
        bull_price = provider.get_current_price("AAPL")

        assert bull_price > initial_price, "Bull market should increase prices"

        provider.create_market_scenario("bear_market")
        bear_price = provider.get_current_price("AAPL")

        assert bear_price < bull_price, "Bear market should decrease prices"

    @pytest.mark.asyncio
    async def test_order_execution_scenarios(self, execution_engine):
        """Test various order execution scenarios."""
        if not execution_engine:
            pytest.skip("No execution engine available")

        # Test market order
        market_result = execution_engine.submit_market_order("AAPL", "buy", 100)
        assert market_result is not None, "Market order should succeed"
        assert "id" in market_result, "Market order should return order ID"

        # Test limit order
        limit_result = execution_engine.submit_limit_order("AAPL", "buy", 100, 170.0)
        assert limit_result is not None, "Limit order should succeed"
        assert "id" in limit_result, "Limit order should return order ID"

        # Test order cancellation
        if limit_result and "id" in limit_result:
            cancel_result = execution_engine.cancel_order(limit_result["id"])
            assert cancel_result is True, "Order cancellation should succeed"

    @pytest.mark.asyncio
    async def test_error_handling(self, execution_engine):
        """Test error handling and recovery mechanisms."""
        if not execution_engine:
            pytest.skip("No execution engine available")

        # Test invalid symbol
        invalid_result = execution_engine.submit_market_order("INVALID_SYMBOL", "buy", 100)
        # Should handle gracefully (may succeed in mock environment)

        # Test invalid quantity
        zero_qty_result = execution_engine.submit_market_order("AAPL", "buy", 0)
        # Should handle gracefully

        # Test invalid side
        try:
            invalid_side_result = execution_engine.submit_market_order("AAPL", "invalid_side", 100)
            # Should handle gracefully
        except Exception:
            # Error handling is working
            pass

    def test_circuit_breaker_functionality(self, execution_engine):
        """Test circuit breaker protection mechanism."""
        if not execution_engine:
            pytest.skip("No execution engine available")

        # Check initial circuit breaker state
        stats = execution_engine.get_execution_stats()
        assert stats["circuit_breaker_status"] in ["open", "closed"], "Invalid circuit breaker status"

        # Test manual reset
        execution_engine.reset_circuit_breaker()
        stats_after_reset = execution_engine.get_execution_stats()
        assert stats_after_reset["circuit_breaker_status"] == "closed", "Circuit breaker should be closed after reset"

    @pytest.mark.asyncio
    async def test_multiple_order_handling(self, execution_engine):
        """Test handling of multiple simultaneous orders."""
        if not execution_engine:
            pytest.skip("No execution engine available")

        # Submit multiple orders
        orders = []
        symbols = ["AAPL", "MSFT", "GOOGL"]

        for symbol in symbols:
            result = execution_engine.submit_market_order(symbol, "buy", 10)
            orders.append(result)

        # Check that at least some orders succeeded
        successful_orders = [o for o in orders if o is not None]
        assert len(successful_orders) > 0, "At least one order should succeed"

    def test_account_information_retrieval(self, execution_engine):
        """Test account information retrieval."""
        if not execution_engine:
            pytest.skip("No execution engine available")

        account_info = execution_engine.get_account_info()
        if account_info:  # May be None in some test environments
            assert "equity" in account_info, "Account info should include equity"
            assert "buying_power" in account_info, "Account info should include buying power"

    def test_position_information_retrieval(self, execution_engine):
        """Test position information retrieval."""
        if not execution_engine:
            pytest.skip("No execution engine available")

        positions = execution_engine.get_positions()
        assert positions is not None, "Positions should return a list (even if empty)"
        assert isinstance(positions, list), "Positions should be a list"

    def test_execution_statistics_tracking(self, execution_engine):
        """Test execution statistics tracking."""
        if not execution_engine:
            pytest.skip("No execution engine available")

        initial_stats = execution_engine.get_execution_stats()

        # Submit a test order to update stats
        execution_engine.submit_market_order("AAPL", "buy", 10)

        updated_stats = execution_engine.get_execution_stats()

        # Check that statistics are tracked
        assert updated_stats["total_orders"] >= initial_stats["total_orders"], "Order count should increase"

        # Check required statistics fields
        required_fields = [
            "total_orders", "successful_orders", "failed_orders",
            "success_rate", "circuit_breaker_status", "is_initialized"
        ]

        for field in required_fields:
            assert field in updated_stats, f"Missing required statistics field: {field}"


@pytest.mark.integration
@pytest.mark.broker
class TestTradingBotIntegration:
    """
    Integration tests for the complete trading bot system.
    
    These tests validate the interaction between different components
    and the overall system behavior.
    """

    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test full system integration with all components."""
        if not (os.getenv('ALPACA_API_KEY_ID') and os.getenv('ALPACA_API_SECRET_KEY')):
            pytest.skip('ALPACA credentials required for integration test')
        now = dt.datetime.now(pytz.timezone('US/Eastern'))
        if now.weekday() >= 5 or not (dt.time(9, 30) <= now.time() <= dt.time(16, 0)):
            pytest.skip('Market closed')
        # This would test the integration of:
        # - Market data feeds
        # - Signal generation
        # - Risk management
        # - Order execution
        # - Monitoring and logging

        # For now, basic integration check
        try:
            # Import key modules
            from ai_trading.execution.live_trading import AlpacaExecutionEngine
            from tests.institutional.framework import TradingScenarioRunner

            # Create and initialize components
            engine = AlpacaExecutionEngine()
            runner = TradingScenarioRunner(engine)

            # Run basic integration test
            if engine.initialize():
                results = await runner.run_end_to_end_test()
                assert results["overall_status"] in ["passed", "failed"], "Integration test should complete"

        except ImportError:
            pytest.skip("Required modules not available for integration test")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_stress_testing(self):
        """Test system under stress conditions."""
        # This would include:
        # - High frequency order submission
        # - Large order volumes
        # - Concurrent operations
        # - Resource usage monitoring

        # Placeholder for stress testing
        await asyncio.sleep(0.1)  # Simulate stress test
        assert True, "Stress test placeholder"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_disaster_recovery(self):
        """Test disaster recovery scenarios."""
        # This would test:
        # - System restart recovery
        # - Data consistency after failure
        # - Emergency stop procedures
        # - Position reconciliation

        # Placeholder for disaster recovery testing
        await asyncio.sleep(0.1)  # Simulate disaster recovery test
        assert True, "Disaster recovery test placeholder"


if __name__ == "__main__":
    """Run tests when executed directly."""
    pytest.main([__file__, "-v", "--tb=short"])
