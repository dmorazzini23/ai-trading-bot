#!/usr/bin/env python3
"""Simple test to validate enhanced execution debugging functionality."""

import os
import sys

# Set required environment variables
os.environ['ALPACA_API_KEY'] = 'test_key'
os.environ['ALPACA_SECRET_KEY'] = 'test_secret'  
os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets'
os.environ['WEBHOOK_SECRET'] = 'test_webhook'
os.environ['FLASK_PORT'] = '9000'

def test_debug_tracker():
    """Test the execution debug tracker."""
    print("Testing ExecutionDebugTracker...")
    
    # Import and create tracker
    from ai_trading.execution.debug_tracker import ExecutionDebugTracker, ExecutionPhase
    tracker = ExecutionDebugTracker()
    print("✓ Debug tracker created")
    
    # Test correlation ID generation
    correlation_id = tracker.generate_correlation_id('AAPL', 'buy')
    print(f"✓ Generated correlation ID: {correlation_id}")
    assert 'AAPL' in correlation_id
    assert 'buy' in correlation_id
    
    # Test execution tracking
    tracker.start_execution_tracking(correlation_id, 'AAPL', 100, 'buy', 
                                   signal_data={'strategy': 'test'})
    print("✓ Started execution tracking")
    
    # Test event logging
    tracker.log_execution_event(correlation_id, ExecutionPhase.ORDER_SUBMITTED, 
                               {'order_id': 'test_123'})
    print("✓ Logged execution event")
    
    # Test order completion
    tracker.log_order_result(correlation_id, True, {'price': 150.00})
    print("✓ Logged order result")
    
    # Get statistics
    stats = tracker.get_execution_stats()
    print(f"✓ Execution stats: {stats}")
    assert stats['recent_successes'] == 1
    
    print("ExecutionDebugTracker test PASSED\n")
    return True

def test_position_reconciler():
    """Test the position reconciler."""
    print("Testing PositionReconciler...")
    
    from ai_trading.execution.position_reconciler import PositionReconciler
    reconciler = PositionReconciler()
    print("✓ Position reconciler created")
    
    # Test position updates
    reconciler.update_bot_position('MSFT', 100, 'test_trade')
    print("✓ Updated bot position")
    
    positions = reconciler.get_bot_positions()
    print(f"✓ Bot positions: {positions}")
    assert positions.get('MSFT') == 100
    
    # Test position adjustment
    reconciler.adjust_bot_position('MSFT', 50, 'adjustment')
    positions = reconciler.get_bot_positions()
    print(f"✓ Adjusted position: {positions}")
    assert positions.get('MSFT') == 150
    
    # Test reconciliation stats
    stats = reconciler.get_reconciliation_stats()
    print(f"✓ Reconciliation stats: {stats}")
    
    print("PositionReconciler test PASSED\n")
    return True

def test_pnl_attributor():
    """Test the PnL attributor."""
    print("Testing PnLAttributor...")
    
    from ai_trading.execution.pnl_attributor import PnLAttributor, PnLSource
    attributor = PnLAttributor()
    print("✓ PnL attributor created")
    
    # Test trade PnL recording
    attributor.add_trade_pnl('GOOGL', 50, 2800.00, 2750.00, fees=2.50)
    print("✓ Recorded trade PnL")
    
    # Test dividend recording
    attributor.add_dividend_pnl('GOOGL', 0.25, 50)
    print("✓ Recorded dividend PnL")
    
    # Get symbol breakdown
    breakdown = attributor.get_pnl_by_symbol('GOOGL')
    print(f"✓ PnL breakdown: {breakdown}")
    assert PnLSource.POSITION_CHANGE.value in breakdown
    assert PnLSource.FEES.value in breakdown
    assert PnLSource.DIVIDEND.value in breakdown
    
    # Get recent events
    events = attributor.get_recent_pnl_events(symbol='GOOGL', limit=5)
    print(f"✓ Recent events count: {len(events)}")
    assert len(events) == 3  # trade, fees, dividend
    
    # Get statistics
    stats = attributor.calculate_attribution_statistics()
    print(f"✓ Attribution stats: {stats}")
    
    print("PnLAttributor test PASSED\n")
    return True

def test_integration():
    """Test integration across all modules."""
    print("Testing integrated functionality...")
    
    from ai_trading.execution.debug_tracker import get_debug_tracker, log_signal_to_execution
    from ai_trading.execution.position_reconciler import get_position_reconciler, update_bot_position
    from ai_trading.execution.pnl_attributor import get_pnl_attributor, record_trade_pnl
    
    # Start a complete trade cycle
    symbol = 'AMZN'
    qty = 25
    side = 'buy'
    
    # 1. Start execution tracking
    correlation_id = log_signal_to_execution(symbol, side, qty, 
                                           signal_data={'strategy': 'momentum'})
    print(f"✓ Started tracking: {correlation_id}")
    
    # 2. Update position
    update_bot_position(symbol, qty, f'trade_{correlation_id}')
    print(f"✓ Updated position for {symbol}")
    
    # 3. Record PnL
    record_trade_pnl(symbol, qty, 3200.00, 3150.00, fees=1.50, 
                    correlation_id=correlation_id)
    print(f"✓ Recorded PnL for {symbol}")
    
    # 4. Verify all systems have the data
    debug_tracker = get_debug_tracker()
    reconciler = get_position_reconciler()
    attributor = get_pnl_attributor()
    
    # Check debug tracker
    timeline = debug_tracker.get_execution_timeline(correlation_id)
    print(f"✓ Timeline events: {len(timeline)}")
    assert len(timeline) >= 1
    
    # Check position reconciler
    positions = reconciler.get_bot_positions()
    print(f"✓ Positions: {positions}")
    assert positions.get(symbol) == qty
    
    # Check PnL attributor
    pnl_breakdown = attributor.get_pnl_by_symbol(symbol)
    print(f"✓ PnL breakdown: {pnl_breakdown}")
    assert len(pnl_breakdown) >= 2  # Should have position change and fees
    
    print("Integration test PASSED\n")
    return True

def main():
    """Run all tests."""
    print("Starting Enhanced Execution Debugging Tests")
    print("=" * 50)
    
    try:
        # Run individual tests
        test_debug_tracker()
        test_position_reconciler()
        test_pnl_attributor()
        test_integration()
        
        print("=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("\nEnhanced execution debugging system is working correctly.")
        print("Features validated:")
        print("- Order correlation tracking with unique IDs")
        print("- Complete execution lifecycle logging")  
        print("- Position reconciliation between bot and broker")
        print("- Detailed PnL attribution by source")
        print("- Integrated debugging across all modules")
        
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)