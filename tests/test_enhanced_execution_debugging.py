"""Test enhanced execution debugging and tracking system.

This test validates the complete signal-to-execution debugging pipeline,
position reconciliation, and PnL attribution functionality.
"""

import time
import unittest
from unittest.mock import Mock

try:
    # Try to import the enhanced execution modules
    from ai_trading.execution.debug_tracker import (
        ExecutionDebugTracker,
        ExecutionPhase,
        OrderStatus,
        get_debug_tracker,
        log_signal_to_execution,
    )
    from ai_trading.execution.pnl_attributor import (
        PnLAttributor,
        PnLEvent,
        PnLSource,
        get_pnl_attributor,
    )
    from ai_trading.execution.position_reconciler import (
        PositionDiscrepancy,
        PositionReconciler,
        get_position_reconciler,
    )
    ENHANCED_DEBUGGING_AVAILABLE = True
except ImportError:
    ENHANCED_DEBUGGING_AVAILABLE = False


class TestExecutionDebugging(unittest.TestCase):
    """Test the enhanced execution debugging system."""

    def setUp(self):
        """Set up test fixtures."""
        if not ENHANCED_DEBUGGING_AVAILABLE:
            self.skipTest("Enhanced debugging modules not available")

        # Create fresh instances for each test
        self.debug_tracker = ExecutionDebugTracker()
        self.debug_tracker.set_debug_mode(verbose=True, trace=True)

    def test_correlation_id_generation(self):
        """Test correlation ID generation is unique and traceable."""
        symbol = "AAPL"
        side = "buy"

        # Generate multiple correlation IDs
        id1 = self.debug_tracker.generate_correlation_id(symbol, side)
        time.sleep(0.001)  # Ensure different timestamps
        id2 = self.debug_tracker.generate_correlation_id(symbol, side)

        # Check they are unique
        self.assertNotEqual(id1, id2)

        # Check they contain symbol and side
        self.assertIn(symbol, id1)
        self.assertIn(side, id1)

    def test_execution_tracking_lifecycle(self):
        """Test complete execution tracking from signal to completion."""
        symbol = "AAPL"
        side = "buy"
        qty = 100

        # Start tracking
        self.debug_tracker.start_execution_tracking(
            correlation_id="test_id_123",
            symbol=symbol,
            qty=qty,
            side=side,
            signal_data={'strategy': 'test_strategy', 'confidence': 0.8}
        )

        # Check order is in active tracking
        active_orders = self.debug_tracker.get_active_orders()
        self.assertIn("test_id_123", active_orders)
        self.assertEqual(active_orders["test_id_123"]["symbol"], symbol)
        self.assertEqual(active_orders["test_id_123"]["qty"], qty)
        self.assertEqual(active_orders["test_id_123"]["side"], side)

        # Log execution phases
        self.debug_tracker.log_execution_event(
            "test_id_123",
            ExecutionPhase.RISK_CHECK,
            {'risk_score': 0.3}
        )

        self.debug_tracker.log_execution_event(
            "test_id_123",
            ExecutionPhase.ORDER_SUBMITTED,
            {'order_id': 'alpaca_123', 'price': 150.00}
        )

        self.debug_tracker.log_execution_event(
            "test_id_123",
            ExecutionPhase.ORDER_FILLED,
            {'fill_price': 150.05, 'fill_qty': qty}
        )

        # Check status updates
        active_orders = self.debug_tracker.get_active_orders()
        self.assertEqual(active_orders["test_id_123"]["status"], OrderStatus.FILLED.value)

        # Log successful completion
        self.debug_tracker.log_order_result(
            "test_id_123",
            success=True,
            order_data={'final_price': 150.05, 'total_cost': 15005.00}
        )

        # Check order moved to completed
        active_orders = self.debug_tracker.get_active_orders()
        self.assertNotIn("test_id_123", active_orders)

        # Check execution timeline
        timeline = self.debug_tracker.get_execution_timeline("test_id_123")
        self.assertEqual(len(timeline), 4)  # signal + risk + submit + fill

        # Check recent executions
        recent = self.debug_tracker.get_recent_executions(limit=1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]['symbol'], symbol)
        self.assertTrue(recent[0].get('success', False))

    def test_failed_execution_tracking(self):
        """Test tracking of failed executions."""
        symbol = "INVALID"
        correlation_id = "failed_test_123"

        self.debug_tracker.start_execution_tracking(
            correlation_id, symbol, 50, "buy"
        )

        # Log rejection
        self.debug_tracker.log_execution_event(
            correlation_id,
            ExecutionPhase.ORDER_REJECTED,
            {'reason': 'Invalid symbol'}
        )

        # Log failure
        self.debug_tracker.log_order_result(
            correlation_id,
            success=False,
            error="Symbol not found"
        )

        # Check in failed executions
        failed = self.debug_tracker.get_failed_executions(limit=1)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]['symbol'], symbol)
        self.assertFalse(failed[0].get('success', True))

    def test_position_update_tracking(self):
        """Test position update tracking with correlation."""
        symbol = "MSFT"
        correlation_id = "position_test_123"

        # Log position change
        self.debug_tracker.log_position_update(
            symbol=symbol,
            old_qty=0,
            new_qty=100,
            correlation_id=correlation_id
        )

        # Get position updates
        updates = self.debug_tracker.get_position_updates(symbol=symbol, limit=1)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]['symbol'], symbol)
        self.assertEqual(updates[0]['old_qty'], 0)
        self.assertEqual(updates[0]['new_qty'], 100)
        self.assertEqual(updates[0]['qty_change'], 100)

    def test_execution_statistics(self):
        """Test execution statistics calculation."""
        # Start some orders
        for i in range(3):
            correlation_id = f"stats_test_{i}"
            self.debug_tracker.start_execution_tracking(
                correlation_id, "TEST", 100, "buy"
            )

        # Complete 2 successfully, fail 1
        self.debug_tracker.log_order_result("stats_test_0", True)
        self.debug_tracker.log_order_result("stats_test_1", True)
        self.debug_tracker.log_order_result("stats_test_2", False, error="Test failure")

        stats = self.debug_tracker.get_execution_stats()

        self.assertEqual(stats['active_orders'], 0)
        self.assertEqual(stats['recent_successes'], 2)
        self.assertEqual(stats['recent_failures'], 1)
        self.assertAlmostEqual(stats['success_rate'], 2/3, places=2)


class TestPositionReconciliation(unittest.TestCase):
    """Test position reconciliation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not ENHANCED_DEBUGGING_AVAILABLE:
            self.skipTest("Enhanced debugging modules not available")

        # Create mock API client
        self.mock_api = Mock()
        self.reconciler = PositionReconciler(self.mock_api)

    def test_bot_position_tracking(self):
        """Test bot position tracking updates."""
        symbol = "AAPL"

        # Update position
        self.reconciler.update_bot_position(symbol, 100, "live_trading")

        positions = self.reconciler.get_bot_positions()
        self.assertEqual(positions[symbol], 100)

        # Adjust position
        self.reconciler.adjust_bot_position(symbol, 50, "additional_trade")

        positions = self.reconciler.get_bot_positions()
        self.assertEqual(positions[symbol], 150)

    def test_discrepancy_detection(self):
        """Test position discrepancy detection."""
        symbol = "MSFT"

        # Set bot position
        self.reconciler.update_bot_position(symbol, 100, "test")

        # Mock broker returning different position
        self.mock_api.get_all_positions.return_value = []  # Empty positions from broker

        # Override the get_broker_positions method for testing
        def mock_get_broker_positions():
            return {symbol: 0}  # Broker shows no position

        self.reconciler.get_broker_positions = mock_get_broker_positions

        # Run reconciliation
        discrepancies = self.reconciler.reconcile_positions()

        # Should detect discrepancy
        self.assertEqual(len(discrepancies), 1)
        self.assertEqual(discrepancies[0].symbol, symbol)
        self.assertEqual(discrepancies[0].bot_qty, 100)
        self.assertEqual(discrepancies[0].broker_qty, 0)
        self.assertEqual(discrepancies[0].discrepancy_type, "phantom_position")

    def test_discrepancy_classification(self):
        """Test different types of discrepancy classification."""

        # Test missing position (bot=0, broker=100)
        discrepancy_type = self.reconciler._classify_discrepancy(0, 100)
        self.assertEqual(discrepancy_type, "missing_position")

        # Test phantom position (bot=100, broker=0)
        discrepancy_type = self.reconciler._classify_discrepancy(100, 0)
        self.assertEqual(discrepancy_type, "phantom_position")

        # Test direction mismatch (bot=100, broker=-50)
        discrepancy_type = self.reconciler._classify_discrepancy(100, -50)
        self.assertEqual(discrepancy_type, "direction_mismatch")

        # Test quantity mismatch (bot=100, broker=150)
        discrepancy_type = self.reconciler._classify_discrepancy(100, 150)
        self.assertEqual(discrepancy_type, "quantity_mismatch")

    def test_severity_determination(self):
        """Test discrepancy severity determination."""
        symbol = "TEST"

        # High severity (>=10 shares difference)
        severity = self.reconciler._determine_severity(symbol, 0, 15)
        self.assertEqual(severity, "high")

        # Medium severity (>=1 share difference)
        severity = self.reconciler._determine_severity(symbol, 100, 105)
        self.assertEqual(severity, "medium")

        # Low severity (<1 share difference)
        severity = self.reconciler._determine_severity(symbol, 100, 100.5)
        self.assertEqual(severity, "low")

    def test_auto_resolution(self):
        """Test automatic discrepancy resolution."""
        symbol = "NVDA"

        # Create low severity discrepancy
        discrepancy = PositionDiscrepancy(
            symbol=symbol,
            bot_qty=100,
            broker_qty=100.5,
            discrepancy_type="quantity_mismatch",
            severity="low"
        )

        # Auto-resolve should update bot position
        resolved = self.reconciler.auto_resolve_discrepancies([discrepancy])
        self.assertEqual(resolved, 1)

        # Check bot position was updated
        positions = self.reconciler.get_bot_positions()
        self.assertEqual(positions[symbol], 100.5)


class TestPnLAttribution(unittest.TestCase):
    """Test PnL attribution functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not ENHANCED_DEBUGGING_AVAILABLE:
            self.skipTest("Enhanced debugging modules not available")

        self.attributor = PnLAttributor()

    def test_trade_pnl_recording(self):
        """Test recording PnL from trades."""
        symbol = "AAPL"
        correlation_id = "trade_pnl_test"

        # Record a profitable trade
        self.attributor.add_trade_pnl(
            symbol=symbol,
            trade_qty=100,
            execution_price=150.00,
            avg_cost=145.00,
            fees=1.00,
            slippage=-0.50,
            correlation_id=correlation_id
        )

        # Check PnL events were created
        recent_events = self.attributor.get_recent_pnl_events(symbol=symbol, limit=10)

        # Should have 3 events: trade PnL, fees, slippage
        self.assertEqual(len(recent_events), 3)

        # Find trade PnL event
        trade_event = next(e for e in recent_events if e['source'] == PnLSource.POSITION_CHANGE.value)
        self.assertEqual(trade_event['pnl_amount'], 500.00)  # 100 * (150 - 145)

        # Find fees event
        fees_event = next(e for e in recent_events if e['source'] == PnLSource.FEES.value)
        self.assertEqual(fees_event['pnl_amount'], -1.00)

        # Find slippage event
        slippage_event = next(e for e in recent_events if e['source'] == PnLSource.SLIPPAGE.value)
        self.assertEqual(slippage_event['pnl_amount'], -0.50)

    def test_position_snapshot_pnl_attribution(self):
        """Test PnL attribution from position snapshots."""
        symbol = "MSFT"

        # Initial position snapshot
        self.attributor.update_position_snapshot(
            symbol=symbol,
            quantity=100,
            avg_cost=200.00,
            market_price=200.00
        )

        # Update with price movement
        self.attributor.update_position_snapshot(
            symbol=symbol,
            quantity=100,
            avg_cost=200.00,
            market_price=205.00  # $5 price increase
        )

        # Check market movement PnL was recorded
        recent_events = self.attributor.get_recent_pnl_events(symbol=symbol, limit=5)
        market_events = [e for e in recent_events if e['source'] == PnLSource.MARKET_MOVEMENT.value]

        self.assertEqual(len(market_events), 1)
        self.assertEqual(market_events[0]['pnl_amount'], 500.00)  # 100 * $5

    def test_pnl_summary_and_breakdown(self):
        """Test PnL summary and breakdown functionality."""
        symbol = "GOOGL"

        # Add various PnL events
        self.attributor.add_trade_pnl(symbol, 50, 2800.00, 2750.00, fees=2.50)
        self.attributor.add_dividend_pnl(symbol, 0.50, 50)
        self.attributor.add_manual_adjustment(symbol, -10.00, "Test adjustment")

        # Get symbol breakdown
        breakdown = self.attributor.get_pnl_by_symbol(symbol)

        # Should have position change, fees, dividend, and adjustment
        self.assertIn(PnLSource.POSITION_CHANGE.value, breakdown)
        self.assertIn(PnLSource.FEES.value, breakdown)
        self.assertIn(PnLSource.DIVIDEND.value, breakdown)
        self.assertIn(PnLSource.ADJUSTMENT.value, breakdown)

        # Check values
        self.assertEqual(breakdown[PnLSource.POSITION_CHANGE.value], 2500.00)  # 50 * (2800 - 2750)
        self.assertEqual(breakdown[PnLSource.FEES.value], -2.50)
        self.assertEqual(breakdown[PnLSource.DIVIDEND.value], 25.00)  # 0.50 * 50
        self.assertEqual(breakdown[PnLSource.ADJUSTMENT.value], -10.00)

    def test_pnl_explanation(self):
        """Test PnL change explanation functionality."""
        symbol = "TSLA"

        # Add some PnL events
        self.attributor.add_trade_pnl(symbol, 25, 800.00, 750.00, fees=1.25)
        self.attributor.add_manual_adjustment(symbol, 5.00, "Price correction")

        # Get explanation
        explanation = self.attributor.explain_pnl_change(symbol, time_window_minutes=60)

        self.assertEqual(explanation['symbol'], symbol)
        self.assertEqual(explanation['total_change'], 1253.75)  # 1250 - 1.25 + 5
        self.assertIn('gained', explanation['explanation'])
        self.assertEqual(explanation['events_count'], 3)  # trade, fees, adjustment

    def test_dividend_pnl_recording(self):
        """Test dividend PnL recording."""
        symbol = "KO"

        self.attributor.add_dividend_pnl(
            symbol=symbol,
            dividend_amount=0.44,
            shares=100,
            correlation_id="dividend_test"
        )

        recent_events = self.attributor.get_recent_pnl_events(symbol=symbol, limit=5)
        dividend_events = [e for e in recent_events if e['source'] == PnLSource.DIVIDEND.value]

        self.assertEqual(len(dividend_events), 1)
        self.assertEqual(dividend_events[0]['pnl_amount'], 44.00)  # 0.44 * 100


class TestIntegratedExecutionDebugging(unittest.TestCase):
    """Test integrated execution debugging across all modules."""

    def setUp(self):
        """Set up integrated test environment."""
        if not ENHANCED_DEBUGGING_AVAILABLE:
            self.skipTest("Enhanced debugging modules not available")

        # Get global instances
        self.debug_tracker = get_debug_tracker()
        self.reconciler = get_position_reconciler()
        self.attributor = get_pnl_attributor()

        # Enable debug mode
        self.debug_tracker.set_debug_mode(verbose=True, trace=True)

    def test_complete_trade_lifecycle_debugging(self):
        """Test complete trade lifecycle with all debugging features."""
        symbol = "AMZN"
        qty = 50
        side = "buy"

        # Start execution tracking
        correlation_id = log_signal_to_execution(
            symbol=symbol,
            side=side,
            qty=qty,
            signal_data={'strategy': 'momentum', 'confidence': 0.9}
        )

        # Simulate trade execution phases
        from ai_trading.execution.debug_tracker import (
            ExecutionPhase,
            log_execution_phase,
        )

        log_execution_phase(correlation_id, ExecutionPhase.RISK_CHECK, {'risk_score': 0.2})
        log_execution_phase(correlation_id, ExecutionPhase.ORDER_PREPARED, {'order_type': 'market'})
        log_execution_phase(correlation_id, ExecutionPhase.ORDER_SUBMITTED, {'order_id': 'test_123'})

        # Simulate successful fill
        execution_price = 3200.00
        avg_cost = 3200.00

        log_execution_phase(correlation_id, ExecutionPhase.ORDER_FILLED, {
            'fill_price': execution_price,
            'fill_qty': qty
        })

        # Update position tracking
        from ai_trading.execution.position_reconciler import update_bot_position
        update_bot_position(symbol, qty, f"live_trading_{correlation_id}")

        # Record PnL
        from ai_trading.execution.pnl_attributor import (
            record_trade_pnl,
            update_position_for_pnl,
        )
        record_trade_pnl(symbol, qty, execution_price, avg_cost, fees=2.50, correlation_id=correlation_id)
        update_position_for_pnl(symbol, qty, avg_cost, execution_price, correlation_id)

        # Log successful completion
        from ai_trading.execution.debug_tracker import log_order_outcome
        log_order_outcome(correlation_id, True, {'final_qty': qty, 'avg_price': execution_price})

        # Verify all systems tracked the trade

        # Check debug tracker
        timeline = self.debug_tracker.get_execution_timeline(correlation_id)
        self.assertGreater(len(timeline), 4)  # Should have multiple phases

        recent_executions = self.debug_tracker.get_recent_executions(limit=1)
        self.assertEqual(len(recent_executions), 1)
        self.assertTrue(recent_executions[0].get('success', False))

        # Check position reconciler
        bot_positions = self.reconciler.get_bot_positions()
        self.assertEqual(bot_positions.get(symbol, 0), qty)

        # Check PnL attributor
        symbol_pnl = self.attributor.get_pnl_by_symbol(symbol)
        self.assertIn(PnLSource.FEES.value, symbol_pnl)
        self.assertEqual(symbol_pnl[PnLSource.FEES.value], -2.50)

        # Get integrated statistics
        debug_stats = self.debug_tracker.get_execution_stats()
        reconcile_stats = self.reconciler.get_reconciliation_stats()
        pnl_stats = self.attributor.calculate_attribution_statistics()

        # Verify statistics are consistent
        self.assertGreater(debug_stats['recent_successes'], 0)
        self.assertEqual(reconcile_stats['bot_positions_count'], 1)
        self.assertGreater(pnl_stats['total_events'], 0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestExecutionDebugging,
        TestPositionReconciliation,
        TestPnLAttribution,
        TestIntegratedExecutionDebugging
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary

    if result.failures:
        for test, traceback in result.failures:
            pass

    if result.errors:
        for test, traceback in result.errors:
            pass
