import logging
'Example usage of the enhanced execution debugging system.\n\nThis script demonstrates how to use the new debugging features to track\nand debug trading execution issues like missing orders and PnL discrepancies.\n'
import os
import time
from datetime import UTC, datetime
os.environ.setdefault('ALPACA_API_KEY', 'demo_key')
os.environ.setdefault('ALPACA_SECRET_KEY', 'demo_secret')
os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
os.environ.setdefault('WEBHOOK_SECRET', 'demo_webhook')
os.environ.setdefault('FLASK_PORT', '9000')

def demonstrate_signal_to_execution_debugging():
    """Demonstrate complete signal-to-execution debugging."""
    logging.info(str('=' * 60))
    logging.info('DEMONSTRATION: Signal-to-Execution Debugging')
    logging.info(str('=' * 60))
    from ai_trading.execution import ExecutionPhase, enable_debug_mode, get_debug_tracker, log_execution_phase, log_order_outcome, log_signal_to_execution
    enable_debug_mode(verbose=True, trace=True)
    logging.info('✓ Enabled verbose debugging mode')
    signal_data = {'strategy': 'momentum_breakout', 'confidence': 0.85, 'trigger_price': 149.5, 'target_price': 155.0, 'stop_loss': 145.0}
    correlation_id = log_signal_to_execution(symbol='AAPL', side='buy', qty=100, signal_data=signal_data)
    logging.info(f'✓ Started tracking execution with ID: {correlation_id}')
    logging.info('\nSimulating execution phases...')
    log_execution_phase(correlation_id, ExecutionPhase.RISK_CHECK, {'risk_score': 0.25, 'position_size_check': 'passed', 'buying_power_check': 'passed', 'volatility_check': 'passed'})
    logging.info('  ✓ Risk check completed')
    log_execution_phase(correlation_id, ExecutionPhase.ORDER_PREPARED, {'order_type': 'market', 'quantity': 100, 'estimated_cost': 14950.0, 'account_equity': 50000.0})
    logging.info('  ✓ Order prepared')
    log_execution_phase(correlation_id, ExecutionPhase.ORDER_SUBMITTED, {'alpaca_order_id': 'ord_12345678', 'submission_time': datetime.now(UTC).isoformat(), 'order_type': 'market'})
    logging.info('  ✓ Order submitted to broker')
    time.sleep(0.1)
    log_execution_phase(correlation_id, ExecutionPhase.ORDER_ACKNOWLEDGED, {'broker_status': 'accepted', 'estimated_fill_time': '2-5 seconds'})
    logging.info('  ✓ Order acknowledged by broker')
    log_execution_phase(correlation_id, ExecutionPhase.ORDER_FILLED, {'fill_price': 149.75, 'fill_quantity': 100, 'fill_time': datetime.now(UTC).isoformat(), 'execution_quality': 'good'})
    logging.info('  ✓ Order filled')
    log_execution_phase(correlation_id, ExecutionPhase.POSITION_UPDATED, {'old_position': 0, 'new_position': 100, 'avg_cost': 149.75})
    logging.info('  ✓ Position updated')
    log_order_outcome(correlation_id, True, {'final_price': 149.75, 'total_cost': 14975.0, 'fees': 1.0, 'slippage': 0.25})
    logging.info('  ✓ Execution completed successfully')
    tracker = get_debug_tracker()
    timeline = tracker.get_execution_timeline(correlation_id)
    logging.info(f'\n📊 EXECUTION TIMELINE ({len(timeline)} events):')
    for i, event in enumerate(timeline, 1):
        logging.info(str(f"  {i}. {event['phase']} at {event['timestamp']}"))
        if event.get('data'):
            for key, value in event['data'].items():
                logging.info(f'     {key}: {value}')
    return correlation_id

def demonstrate_position_reconciliation():
    """Demonstrate position reconciliation between bot and broker."""
    logging.info(str('\n' + '=' * 60))
    logging.info('DEMONSTRATION: Position Reconciliation')
    logging.info(str('=' * 60))
    from ai_trading.execution import force_position_reconciliation, get_position_reconciler, update_bot_position
    reconciler = get_position_reconciler()
    positions = [('AAPL', 100), ('MSFT', 50), ('GOOGL', 25), ('TSLA', 75)]
    logging.info('Setting up bot positions...')
    for symbol, qty in positions:
        update_bot_position(symbol, qty, f'demo_trade_{symbol}')
        logging.info(f'  ✓ {symbol}: {qty} shares')
    logging.info(f'\nBot positions: {reconciler.get_bot_positions()}')

    def mock_broker_positions():
        return {'AAPL': 100, 'MSFT': 50, 'GOOGL': 25, 'TSLA': 50, 'NVDA': 30}
    reconciler.get_broker_positions = mock_broker_positions
    logging.info(f'Simulated broker positions: {mock_broker_positions()}')
    logging.info('\nRunning position reconciliation...')
    discrepancies = force_position_reconciliation()
    if discrepancies:
        logging.info(f'⚠️  Found {len(discrepancies)} discrepancies:')
        for disc in discrepancies:
            logging.info(f'  {disc.symbol}: Bot={disc.bot_qty}, Broker={disc.broker_qty}')
            logging.info(f'    Type: {disc.discrepancy_type}, Severity: {disc.severity}')
    else:
        logging.info('✓ All positions reconciled successfully')
    stats = reconciler.get_reconciliation_stats()
    logging.info('\n📊 RECONCILIATION STATS:')
    logging.info(str(f"  Current discrepancies: {stats['current_discrepancies']}"))
    logging.info(str(f"  Historical discrepancies: {stats['total_historical_discrepancies']}"))
    logging.info(str(f"  Bot positions: {stats['bot_positions_count']}"))
    logging.info(str(f"  Broker positions: {stats['broker_positions_count']}"))

def demonstrate_pnl_attribution():
    """Demonstrate detailed PnL attribution."""
    logging.info(str('\n' + '=' * 60))
    logging.info('DEMONSTRATION: PnL Attribution')
    logging.info(str('=' * 60))
    from ai_trading.execution import explain_recent_pnl_changes, get_pnl_attributor, get_symbol_pnl_breakdown, record_dividend_income, record_trade_pnl, update_position_for_pnl
    get_pnl_attributor()
    logging.info('Recording PnL events...')
    record_trade_pnl(symbol='AAPL', trade_qty=100, execution_price=149.75, avg_cost=145.0, fees=1.0, slippage=0.25, correlation_id='demo_trade_aapl')
    logging.info('  ✓ AAPL trade: +$475 profit, -$1 fees, -$0.25 slippage')
    update_position_for_pnl('AAPL', 100, 149.75, 152.0)
    logging.info('  ✓ AAPL market movement: +$225 unrealized gain')
    record_dividend_income('AAPL', 0.24, 100, 'dividend_q4_2024')
    logging.info('  ✓ AAPL dividend: +$24')
    record_trade_pnl(symbol='TSLA', trade_qty=50, execution_price=180.0, avg_cost=190.0, fees=1.5, correlation_id='demo_trade_tsla')
    logging.info('  ✓ TSLA trade: -$500 loss, -$1.50 fees')
    logging.info('\n📊 PnL BREAKDOWN BY SYMBOL:')
    for symbol in ['AAPL', 'TSLA']:
        breakdown = get_symbol_pnl_breakdown(symbol)
        logging.info(f'\n{symbol}:')
        total = 0
        for source, amount in breakdown.items():
            if amount != 0:
                logging.info(f'  {source}: ${amount:+.2f}')
                if source != 'unrealized':
                    total += amount
        logging.info(f'  TOTAL REALIZED: ${total:+.2f}')
    logging.info('\n📝 RECENT PnL EXPLANATION FOR AAPL:')
    explanation = explain_recent_pnl_changes('AAPL', minutes=60)
    logging.info(str(f"  {explanation['explanation']}"))
    logging.info(str(f"  Total change: ${explanation['total_change']:+.2f}"))
    from ai_trading.execution import get_portfolio_pnl_summary
    summary = get_portfolio_pnl_summary()
    logging.info('\n📊 PORTFOLIO PnL SUMMARY:')
    logging.info(str(f"  Total realized PnL: ${summary['total_realized_pnl']:+.2f}"))
    logging.info(str(f"  Total unrealized PnL: ${summary['total_unrealized_pnl']:+.2f}"))
    logging.info(str(f"  Net PnL: ${summary['total_pnl']:+.2f}"))

def demonstrate_debugging_a_problem():
    """Demonstrate how to debug the specific problem mentioned: '$514 PnL with 0 positions'."""
    logging.info(str('\n' + '=' * 60))
    logging.info("DEMONSTRATION: Debugging '$514 PnL with 0 positions' Problem")
    logging.info(str('=' * 60))
    from ai_trading.execution import force_position_reconciliation, get_debug_tracker, get_pnl_attributor, get_portfolio_pnl_summary, get_position_reconciler
    logging.info('🔍 INVESTIGATING PnL/POSITION DISCREPANCY...')
    debug_tracker = get_debug_tracker()
    exec_stats = debug_tracker.get_execution_stats()
    logging.info('\n1. EXECUTION STATISTICS:')
    logging.info(str(f"   Recent successful orders: {exec_stats['recent_successes']}"))
    logging.info(str(f"   Recent failed orders: {exec_stats['recent_failures']}"))
    logging.info(str(f"   Success rate: {exec_stats['success_rate']:.1%}"))
    logging.info(str(f"   Active orders: {exec_stats['active_orders']}"))
    if exec_stats['recent_successes'] > 0:
        logging.info('   → Orders were executed successfully, but where did positions go?')
    recent_executions = debug_tracker.get_recent_executions(limit=5)
    logging.info(f'\n2. RECENT EXECUTIONS ({len(recent_executions)} found):')
    for execution in recent_executions:
        symbol = execution.get('symbol', 'Unknown')
        side = execution.get('side', 'Unknown')
        qty = execution.get('qty', 'Unknown')
        success = execution.get('success', False)
        logging.info(str(f"   {symbol} {side} {qty} shares - {('✓' if success else '✗')}"))
    logging.info('\n3. POSITION RECONCILIATION:')
    reconciler = get_position_reconciler()
    bot_positions = reconciler.get_bot_positions()
    logging.info(f'   Bot positions: {bot_positions}')
    discrepancies = force_position_reconciliation()
    if discrepancies:
        logging.info(f'   ⚠️  Found {len(discrepancies)} position discrepancies!')
        for disc in discrepancies:
            logging.info(f'   {disc.symbol}: Bot={disc.bot_qty}, Broker={disc.broker_qty}')
    else:
        logging.info('   ✓ No position discrepancies found')
    logging.info('\n4. PnL ATTRIBUTION ANALYSIS:')
    attributor = get_pnl_attributor()
    portfolio_summary = get_portfolio_pnl_summary()
    logging.info(str(f"   Total realized PnL: ${portfolio_summary['total_realized_pnl']:+.2f}"))
    logging.info(str(f"   Total unrealized PnL: ${portfolio_summary['total_unrealized_pnl']:+.2f}"))
    logging.info(str(f"   Net PnL: ${portfolio_summary['total_pnl']:+.2f}"))
    pnl_by_source = portfolio_summary['pnl_by_source']
    logging.info('   PnL by source:')
    for source, amount in pnl_by_source.items():
        if amount != 0:
            logging.info(f'     {source}: ${amount:+.2f}')
    recent_events = attributor.get_recent_pnl_events(limit=10)
    logging.info(f'\n5. RECENT PnL EVENTS ({len(recent_events)} found):')
    for event in recent_events[-5:]:
        symbol = event['symbol']
        amount = event['pnl_amount']
        source = event['source']
        logging.info(f'   {symbol}: ${amount:+.2f} from {source}')
    logging.info('\n6. DEBUGGING RECOMMENDATIONS:')
    logging.info('   ✓ Check execution timeline for each order using correlation IDs')
    logging.info('   ✓ Verify position updates were properly recorded after fills')
    logging.info('   ✓ Check if orders were submitted but never filled')
    logging.info('   ✓ Look for position adjustments or manual corrections')
    logging.info('   ✓ Verify broker API position sync is working')
    logging.info('   ✓ Check for unreported fills or partial fills')

def main():
    """Run all demonstrations."""
    logging.info('Enhanced Execution Debugging System - Demonstrations')
    logging.info('This shows how the new debugging features help track and resolve trading issues')
    try:
        demonstrate_signal_to_execution_debugging()
        demonstrate_position_reconciliation()
        demonstrate_pnl_attribution()
        demonstrate_debugging_a_problem()
        logging.info(str('\n' + '=' * 60))
        logging.info('SUMMARY: Enhanced Debugging Features')
        logging.info(str('=' * 60))
        logging.info('✓ Signal-to-execution correlation tracking')
        logging.info('✓ Complete execution timeline with phase logging')
        logging.info('✓ Position reconciliation with discrepancy detection')
        logging.info('✓ Detailed PnL attribution by source')
        logging.info('✓ Comprehensive debugging workflow for problem resolution')
        logging.info('\nThese features provide complete visibility into the trading pipeline')
        logging.info('and will help quickly identify and resolve issues like:')
        logging.info('- Missing orders that should have been placed')
        logging.info('- Position discrepancies between bot and broker')
        logging.info('- Unexplained PnL changes')
        logging.info('- Failed order executions')
        logging.info('\n🔧 TO USE IN PRODUCTION:')
        logging.info('1. Import: from ai_trading.execution import enable_debug_mode')
        logging.info('2. Enable: enable_debug_mode(verbose=True)')
        logging.info('3. Monitor: Check logs for execution events and discrepancies')
        logging.info('4. Debug: Use correlation IDs to trace specific order issues')
    except (ValueError, TypeError) as e:
        logging.info(f'\nDemonstration failed: {e}')
        import traceback
        traceback.print_exc()
if __name__ == '__main__':
    main()