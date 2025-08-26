import logging
from datetime import UTC

# Minimal stubs so this guide script lints without heavy imports
def get_current_position(symbol: str) -> int:  # pragma: no cover
    return 0

def get_current_market_price(symbol: str) -> float:  # pragma: no cover
    return 0.0
'Integration guide for enhanced execution debugging in existing bot engine.\n\nThis shows how to integrate the new debugging features into the existing\ntrading bot without breaking current functionality.\n'

def enable_enhanced_debugging():
    """Enable enhanced debugging during bot startup."""
    from ai_trading.execution import enable_debug_mode
    enable_debug_mode(verbose=True, trace=False)
    logging.info('✓ Enhanced execution debugging enabled')

def enhance_existing_execution_engine():
    """Example of how to add debugging to existing ExecutionEngine calls."""
    from ai_trading.execution import ExecutionPhase, log_execution_phase, log_signal_to_execution

    def execute_order_with_debugging(engine, symbol, qty, side, signal_data=None):
        correlation_id = log_signal_to_execution(symbol, side, qty, signal_data)
        log_execution_phase(correlation_id, ExecutionPhase.RISK_CHECK, {'requested_qty': qty, 'available_cash': engine.ctx.get_account().cash if hasattr(engine.ctx, 'get_account') else 'unknown'})
        try:
            result = engine.execute_order(symbol, qty, side)
            if result:
                log_execution_phase(correlation_id, ExecutionPhase.ORDER_FILLED, {'order_id': getattr(result, 'id', 'unknown'), 'status': getattr(result, 'status', 'unknown')})
                from ai_trading.execution import update_bot_position
                current_qty = engine._available_qty(engine.api, symbol)
                update_bot_position(symbol, current_qty, f'execution_{correlation_id}')
            else:
                log_execution_phase(correlation_id, ExecutionPhase.ORDER_REJECTED, {'reason': 'execute_order returned None'})
        except (ValueError, TypeError) as e:
            log_execution_phase(correlation_id, ExecutionPhase.ORDER_REJECTED, {'error': str(e)})
            raise
        return result
    return execute_order_with_debugging

def add_debugging_to_signals():
    """Example of adding debugging to signal generation."""

    def generate_trading_signals_with_debugging(data, strategy_name):
        """Generate signals with execution tracking."""
        from ai_trading.execution import log_signal_to_execution
        signals = []
        for signal in signals:
            signal.correlation_id = log_signal_to_execution(symbol=signal.symbol, side=signal.side, qty=signal.quantity, signal_data={'strategy': strategy_name, 'confidence': signal.confidence, 'trigger_price': getattr(signal, 'price', None), 'timestamp': signal.timestamp.isoformat() if hasattr(signal, 'timestamp') else None})
        return signals

def add_pnl_tracking_to_fills():
    """Example of adding PnL tracking when orders are filled."""

    def handle_order_fill_with_pnl_tracking(order, fill_price, fill_qty):
        """Handle order fill with PnL attribution."""
        from ai_trading.execution import record_trade_pnl, update_position_for_pnl
        symbol = order.symbol
        side = order.side
        correlation_id = getattr(order, 'correlation_id', None)
        fees = getattr(order, 'fees', 0)
        expected_price = getattr(order, 'limit_price', fill_price)
        slippage = (fill_price - expected_price) * fill_qty if side == 'buy' else (expected_price - fill_price) * fill_qty
        avg_cost = fill_price
        record_trade_pnl(symbol=symbol, trade_qty=fill_qty if side == 'buy' else -fill_qty, execution_price=fill_price, avg_cost=avg_cost, fees=fees, slippage=slippage, correlation_id=correlation_id)
        current_qty = get_current_position(symbol)
        current_market_price = get_current_market_price(symbol)
        update_position_for_pnl(symbol=symbol, quantity=current_qty, avg_cost=avg_cost, market_price=current_market_price, correlation_id=correlation_id)

def setup_periodic_checks():
    """Setup periodic reconciliation and health checks."""

    def periodic_health_check():
        """Run periodic health checks on execution system."""
        from ai_trading.execution import force_position_reconciliation, get_execution_statistics, get_portfolio_pnl_summary
        discrepancies = force_position_reconciliation()
        if discrepancies:
            logging.info(f'⚠️ Found {len(discrepancies)} position discrepancies')
            for disc in discrepancies:
                if disc.severity in ['high', 'medium']:
                    logging.info(f'  {disc.symbol}: Bot={disc.bot_qty}, Broker={disc.broker_qty} ({disc.severity})')
        exec_stats = get_execution_statistics()
        success_rate = exec_stats.get('success_rate', 0)
        if success_rate < 0.95:
            logging.info(f'⚠️ Low execution success rate: {success_rate:.1%}')
        pnl_summary = get_portfolio_pnl_summary()
        if abs(pnl_summary['total_pnl']) > 1000:
            logging.info(str(f"📊 Large PnL movement: ${pnl_summary['total_pnl']:+.2f}"))

def setup_error_alerting():
    """Setup error alerting for debugging events."""

    def check_for_execution_issues():
        """Check for and alert on execution issues."""
        from ai_trading.execution import get_debug_tracker
        tracker = get_debug_tracker()
        failed_executions = tracker.get_failed_executions(limit=10)
        recent_failures = [e for e in failed_executions if was_recent(e.get('timestamp'))]
        if len(recent_failures) > 3:
            logging.info(f'🚨 ALERT: {len(recent_failures)} failed executions recently')
            for failure in recent_failures:
                logging.info(f"  {failure['symbol']} {failure['side']}: {failure.get('error', 'Unknown error')}")
        active_orders = tracker.get_active_orders()
        stuck_orders = [o for o in active_orders.values() if was_old(o.get('start_time'))]
        if stuck_orders:
            logging.info(f'🚨 ALERT: {len(stuck_orders)} orders appear stuck')
            for order in stuck_orders:
                logging.info(str(f"  {order['symbol']} {order['side']} - started {order['start_time']}"))

    def was_recent(timestamp_str, minutes=30):
        """Check if timestamp was within last N minutes."""
        if not timestamp_str:
            return False
        from datetime import datetime, timedelta
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
        return timestamp > cutoff

    def was_old(timestamp_str, minutes=60):
        """Check if timestamp is older than N minutes."""
        return not was_recent(timestamp_str, minutes)

def complete_integration_example():
    """Complete example of integrating debugging into existing bot."""

    class EnhancedTradingBot:
        """Example of enhanced trading bot with debugging."""

        def __init__(self):
            enable_enhanced_debugging()
            self.execution_engine = None
            self.strategy = None

        def run_trading_cycle(self):
            """Enhanced trading cycle with debugging."""
            try:
                signals = self.generate_signals_with_debugging()
                for signal in signals:
                    self.execute_signal_with_debugging(signal)
                self.run_health_checks()
            except (ValueError, TypeError) as e:
                logging.info(f'Trading cycle error: {e}')

        def generate_signals_with_debugging(self):
            """Generate signals with debugging integration."""
            signals = []
            from ai_trading.execution import log_signal_to_execution
            for signal in signals:
                signal.correlation_id = log_signal_to_execution(symbol=signal.symbol, side=signal.side, qty=signal.quantity, signal_data={'strategy': signal.strategy, 'confidence': signal.confidence})
            return signals

        def execute_signal_with_debugging(self, signal):
            """Execute signal with debugging."""
            from ai_trading.execution import ExecutionPhase, log_execution_phase
            correlation_id = signal.correlation_id
            log_execution_phase(correlation_id, ExecutionPhase.RISK_CHECK)
            result = self.execution_engine.execute_order(signal.symbol, signal.quantity, signal.side)
            if result:
                log_execution_phase(correlation_id, ExecutionPhase.ORDER_FILLED)
            else:
                log_execution_phase(correlation_id, ExecutionPhase.ORDER_REJECTED)

        def run_health_checks(self):
            """Run periodic health checks."""
            from ai_trading.execution import force_position_reconciliation, get_execution_statistics
            discrepancies = force_position_reconciliation()
            if discrepancies:
                logging.info(f'Position discrepancies found: {len(discrepancies)}')
            stats = get_execution_statistics()
            if stats['success_rate'] < 0.9:
                logging.info(str(f"Low success rate: {stats['success_rate']:.1%}"))
if __name__ == '__main__':
    logging.info('Enhanced Execution Debugging - Integration Guide')
    logging.info(str('=' * 50))
    logging.info('This file shows how to integrate the new debugging features')
    logging.info('into your existing trading bot code.')
    logging.info('Key integration points:')
    logging.info('1. Enable debugging at startup')
    logging.info('2. Add correlation tracking to signal generation')
    logging.info('3. Enhance order execution with debugging hooks')
    logging.info('4. Add PnL tracking to order fills')
    logging.info('5. Setup periodic reconciliation checks')
    logging.info('6. Add error alerting for execution issues')
    logging.info('See the function examples above for implementation details.')