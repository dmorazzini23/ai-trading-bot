"""Example usage of the enhanced execution debugging system.

This script demonstrates how to use the new debugging features to track
and debug trading execution issues like missing orders and PnL discrepancies.
"""

import os
import time
from datetime import datetime, timezone

# Set environment variables for demo
os.environ.setdefault('ALPACA_API_KEY', 'demo_key')
os.environ.setdefault('ALPACA_SECRET_KEY', 'demo_secret')
os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
os.environ.setdefault('WEBHOOK_SECRET', 'demo_webhook')
os.environ.setdefault('FLASK_PORT', '9000')

def demonstrate_signal_to_execution_debugging():
    """Demonstrate complete signal-to-execution debugging."""
    print("=" * 60)
    print("DEMONSTRATION: Signal-to-Execution Debugging")
    print("=" * 60)
    
    from ai_trading.execution import (
        log_signal_to_execution, log_execution_phase, log_order_outcome,
        ExecutionPhase, get_debug_tracker, enable_debug_mode
    )
    
    # Enable verbose debugging
    enable_debug_mode(verbose=True, trace=True)
    print("✓ Enabled verbose debugging mode")
    
    # Simulate a trading signal
    signal_data = {
        'strategy': 'momentum_breakout',
        'confidence': 0.85,
        'trigger_price': 149.50,
        'target_price': 155.00,
        'stop_loss': 145.00
    }
    
    # Start tracking execution
    correlation_id = log_signal_to_execution(
        symbol="AAPL",
        side="buy", 
        qty=100,
        signal_data=signal_data
    )
    print(f"✓ Started tracking execution with ID: {correlation_id}")
    
    # Simulate execution phases
    print("\nSimulating execution phases...")
    
    # Risk check phase
    log_execution_phase(correlation_id, ExecutionPhase.RISK_CHECK, {
        'risk_score': 0.25,
        'position_size_check': 'passed',
        'buying_power_check': 'passed',
        'volatility_check': 'passed'
    })
    print("  ✓ Risk check completed")
    
    # Order preparation phase
    log_execution_phase(correlation_id, ExecutionPhase.ORDER_PREPARED, {
        'order_type': 'market',
        'quantity': 100,
        'estimated_cost': 14950.00,
        'account_equity': 50000.00
    })
    print("  ✓ Order prepared")
    
    # Order submission phase
    log_execution_phase(correlation_id, ExecutionPhase.ORDER_SUBMITTED, {
        'alpaca_order_id': 'ord_12345678',
        'submission_time': datetime.now(timezone.utc).isoformat(),
        'order_type': 'market'
    })
    print("  ✓ Order submitted to broker")
    
    # Simulate a small delay
    time.sleep(0.1)
    
    # Order acknowledgment
    log_execution_phase(correlation_id, ExecutionPhase.ORDER_ACKNOWLEDGED, {
        'broker_status': 'accepted',
        'estimated_fill_time': '2-5 seconds'
    })
    print("  ✓ Order acknowledged by broker")
    
    # Order fill
    log_execution_phase(correlation_id, ExecutionPhase.ORDER_FILLED, {
        'fill_price': 149.75,
        'fill_quantity': 100,
        'fill_time': datetime.now(timezone.utc).isoformat(),
        'execution_quality': 'good'
    })
    print("  ✓ Order filled")
    
    # Position update
    log_execution_phase(correlation_id, ExecutionPhase.POSITION_UPDATED, {
        'old_position': 0,
        'new_position': 100,
        'avg_cost': 149.75
    })
    print("  ✓ Position updated")
    
    # Final outcome
    log_order_outcome(correlation_id, True, {
        'final_price': 149.75,
        'total_cost': 14975.00,
        'fees': 1.00,
        'slippage': 0.25  # Paid 0.25 more than expected
    })
    print("  ✓ Execution completed successfully")
    
    # Show execution timeline
    tracker = get_debug_tracker()
    timeline = tracker.get_execution_timeline(correlation_id)
    
    print(f"\n📊 EXECUTION TIMELINE ({len(timeline)} events):")
    for i, event in enumerate(timeline, 1):
        print(f"  {i}. {event['phase']} at {event['timestamp']}")
        if event.get('data'):
            for key, value in event['data'].items():
                print(f"     {key}: {value}")
    
    return correlation_id

def demonstrate_position_reconciliation():
    """Demonstrate position reconciliation between bot and broker."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Position Reconciliation")
    print("=" * 60)
    
    from ai_trading.execution import (
        get_position_reconciler, update_bot_position, force_position_reconciliation
    )
    
    reconciler = get_position_reconciler()
    
    # Simulate bot tracking some positions
    positions = [
        ("AAPL", 100),
        ("MSFT", 50),
        ("GOOGL", 25),
        ("TSLA", 75)
    ]
    
    print("Setting up bot positions...")
    for symbol, qty in positions:
        update_bot_position(symbol, qty, f"demo_trade_{symbol}")
        print(f"  ✓ {symbol}: {qty} shares")
    
    # Simulate a discrepancy (broker shows different position for TSLA)
    print(f"\nBot positions: {reconciler.get_bot_positions()}")
    
    # Mock broker positions that differ from bot
    def mock_broker_positions():
        return {
            "AAPL": 100,  # Matches
            "MSFT": 50,   # Matches  
            "GOOGL": 25,  # Matches
            "TSLA": 50,   # DISCREPANCY: Bot thinks 75, broker has 50
            "NVDA": 30    # DISCREPANCY: Bot doesn't know about this position
        }
    
    # Override the method for demo
    reconciler.get_broker_positions = mock_broker_positions
    
    print(f"Simulated broker positions: {mock_broker_positions()}")
    
    # Run reconciliation
    print("\nRunning position reconciliation...")
    discrepancies = force_position_reconciliation()
    
    if discrepancies:
        print(f"⚠️  Found {len(discrepancies)} discrepancies:")
        for disc in discrepancies:
            print(f"  {disc.symbol}: Bot={disc.bot_qty}, Broker={disc.broker_qty}")
            print(f"    Type: {disc.discrepancy_type}, Severity: {disc.severity}")
    else:
        print("✓ All positions reconciled successfully")
    
    # Show reconciliation statistics
    stats = reconciler.get_reconciliation_stats()
    print("\n📊 RECONCILIATION STATS:")
    print(f"  Current discrepancies: {stats['current_discrepancies']}")
    print(f"  Historical discrepancies: {stats['total_historical_discrepancies']}")
    print(f"  Bot positions: {stats['bot_positions_count']}")
    print(f"  Broker positions: {stats['broker_positions_count']}")

def demonstrate_pnl_attribution():
    """Demonstrate detailed PnL attribution."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: PnL Attribution")
    print("=" * 60)
    
    from ai_trading.execution import (
        get_pnl_attributor, record_trade_pnl, record_dividend_income,
        update_position_for_pnl, get_symbol_pnl_breakdown, explain_recent_pnl_changes
    )
    
    attributor = get_pnl_attributor()
    
    # Simulate various PnL events
    print("Recording PnL events...")
    
    # Trade PnL (profitable AAPL trade)
    record_trade_pnl(
        symbol="AAPL",
        trade_qty=100,
        execution_price=149.75,
        avg_cost=145.00,  # Bought cheaper earlier
        fees=1.00,
        slippage=0.25,
        correlation_id="demo_trade_aapl"
    )
    print("  ✓ AAPL trade: +$475 profit, -$1 fees, -$0.25 slippage")
    
    # Market movement (AAPL price went up)
    update_position_for_pnl("AAPL", 100, 149.75, 152.00)  # Price moved up $2.25
    print("  ✓ AAPL market movement: +$225 unrealized gain")
    
    # Dividend income
    record_dividend_income("AAPL", 0.24, 100, "dividend_q4_2024")
    print("  ✓ AAPL dividend: +$24")
    
    # Losing trade (TSLA)
    record_trade_pnl(
        symbol="TSLA",
        trade_qty=50,
        execution_price=180.00,
        avg_cost=190.00,  # Sold at a loss
        fees=1.50,
        correlation_id="demo_trade_tsla"
    )
    print("  ✓ TSLA trade: -$500 loss, -$1.50 fees")
    
    # Show PnL breakdown by symbol
    print("\n📊 PnL BREAKDOWN BY SYMBOL:")
    
    for symbol in ["AAPL", "TSLA"]:
        breakdown = get_symbol_pnl_breakdown(symbol)
        print(f"\n{symbol}:")
        total = 0
        for source, amount in breakdown.items():
            if amount != 0:
                print(f"  {source}: ${amount:+.2f}")
                if source != 'unrealized':
                    total += amount
        print(f"  TOTAL REALIZED: ${total:+.2f}")
    
    # Explain recent changes
    print("\n📝 RECENT PnL EXPLANATION FOR AAPL:")
    explanation = explain_recent_pnl_changes("AAPL", minutes=60)
    print(f"  {explanation['explanation']}")
    print(f"  Total change: ${explanation['total_change']:+.2f}")
    
    # Show portfolio summary
    from ai_trading.execution import get_portfolio_pnl_summary
    summary = get_portfolio_pnl_summary()
    print("\n📊 PORTFOLIO PnL SUMMARY:")
    print(f"  Total realized PnL: ${summary['total_realized_pnl']:+.2f}")
    print(f"  Total unrealized PnL: ${summary['total_unrealized_pnl']:+.2f}")
    print(f"  Net PnL: ${summary['total_pnl']:+.2f}")

def demonstrate_debugging_a_problem():
    """Demonstrate how to debug the specific problem mentioned: '$514 PnL with 0 positions'."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Debugging '$514 PnL with 0 positions' Problem")
    print("=" * 60)
    
    from ai_trading.execution import (
        get_debug_tracker, get_position_reconciler, get_pnl_attributor,
        force_position_reconciliation, get_portfolio_pnl_summary
    )
    
    # This is how you would investigate the problem
    print("🔍 INVESTIGATING PnL/POSITION DISCREPANCY...")
    
    # 1. Check current execution statistics
    debug_tracker = get_debug_tracker()
    exec_stats = debug_tracker.get_execution_stats()
    
    print("\n1. EXECUTION STATISTICS:")
    print(f"   Recent successful orders: {exec_stats['recent_successes']}")
    print(f"   Recent failed orders: {exec_stats['recent_failures']}")
    print(f"   Success rate: {exec_stats['success_rate']:.1%}")
    print(f"   Active orders: {exec_stats['active_orders']}")
    
    if exec_stats['recent_successes'] > 0:
        print("   → Orders were executed successfully, but where did positions go?")
    
    # 2. Check recent executions for clues
    recent_executions = debug_tracker.get_recent_executions(limit=5)
    print(f"\n2. RECENT EXECUTIONS ({len(recent_executions)} found):")
    for execution in recent_executions:
        symbol = execution.get('symbol', 'Unknown')
        side = execution.get('side', 'Unknown')
        qty = execution.get('qty', 'Unknown')
        success = execution.get('success', False)
        print(f"   {symbol} {side} {qty} shares - {'✓' if success else '✗'}")
    
    # 3. Force position reconciliation to find discrepancies
    print("\n3. POSITION RECONCILIATION:")
    reconciler = get_position_reconciler()
    bot_positions = reconciler.get_bot_positions()
    print(f"   Bot positions: {bot_positions}")
    
    # Force reconciliation (would check against broker API in real scenario)
    discrepancies = force_position_reconciliation()
    if discrepancies:
        print(f"   ⚠️  Found {len(discrepancies)} position discrepancies!")
        for disc in discrepancies:
            print(f"   {disc.symbol}: Bot={disc.bot_qty}, Broker={disc.broker_qty}")
    else:
        print("   ✓ No position discrepancies found")
    
    # 4. Analyze PnL attribution
    print("\n4. PnL ATTRIBUTION ANALYSIS:")
    attributor = get_pnl_attributor()
    portfolio_summary = get_portfolio_pnl_summary()
    
    print(f"   Total realized PnL: ${portfolio_summary['total_realized_pnl']:+.2f}")
    print(f"   Total unrealized PnL: ${portfolio_summary['total_unrealized_pnl']:+.2f}")
    print(f"   Net PnL: ${portfolio_summary['total_pnl']:+.2f}")
    
    # Break down PnL by source
    pnl_by_source = portfolio_summary['pnl_by_source']
    print("   PnL by source:")
    for source, amount in pnl_by_source.items():
        if amount != 0:
            print(f"     {source}: ${amount:+.2f}")
    
    # 5. Check for recent PnL events that might explain the situation
    recent_events = attributor.get_recent_pnl_events(limit=10)
    print(f"\n5. RECENT PnL EVENTS ({len(recent_events)} found):")
    for event in recent_events[-5:]:  # Show last 5
        symbol = event['symbol']
        amount = event['pnl_amount']
        source = event['source']
        print(f"   {symbol}: ${amount:+.2f} from {source}")
    
    # 6. Provide debugging recommendations
    print("\n6. DEBUGGING RECOMMENDATIONS:")
    print("   ✓ Check execution timeline for each order using correlation IDs")
    print("   ✓ Verify position updates were properly recorded after fills")
    print("   ✓ Check if orders were submitted but never filled")
    print("   ✓ Look for position adjustments or manual corrections")
    print("   ✓ Verify broker API position sync is working")
    print("   ✓ Check for unreported fills or partial fills")

def main():
    """Run all demonstrations."""
    print("Enhanced Execution Debugging System - Demonstrations")
    print("This shows how the new debugging features help track and resolve trading issues")
    
    try:
        # Demonstrate each feature
        correlation_id = demonstrate_signal_to_execution_debugging()
        demonstrate_position_reconciliation()
        demonstrate_pnl_attribution()
        demonstrate_debugging_a_problem()
        
        print("\n" + "=" * 60)
        print("SUMMARY: Enhanced Debugging Features")
        print("=" * 60)
        print("✓ Signal-to-execution correlation tracking")
        print("✓ Complete execution timeline with phase logging")
        print("✓ Position reconciliation with discrepancy detection")
        print("✓ Detailed PnL attribution by source")
        print("✓ Comprehensive debugging workflow for problem resolution")
        print("\nThese features provide complete visibility into the trading pipeline")
        print("and will help quickly identify and resolve issues like:")
        print("- Missing orders that should have been placed")
        print("- Position discrepancies between bot and broker")
        print("- Unexplained PnL changes")
        print("- Failed order executions")
        
        print("\n🔧 TO USE IN PRODUCTION:")
        print("1. Import: from ai_trading.execution import enable_debug_mode")
        print("2. Enable: enable_debug_mode(verbose=True)")
        print("3. Monitor: Check logs for execution events and discrepancies")
        print("4. Debug: Use correlation IDs to trace specific order issues")
        
    except Exception as e:
        print(f"\nDemonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()