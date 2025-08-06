"""Integration guide for enhanced execution debugging in existing bot engine.

This shows how to integrate the new debugging features into the existing
trading bot without breaking current functionality.
"""

# STEP 1: Enable debugging in bot startup
def enable_enhanced_debugging():
    """Enable enhanced debugging during bot startup."""
    from ai_trading.execution import enable_debug_mode, start_position_monitoring
    
    # Enable verbose debugging (set to False for production)
    enable_debug_mode(verbose=True, trace=False)
    
    # Start position monitoring every 5 minutes
    # start_position_monitoring(api_client=your_api_client, interval=300)
    
    print("✓ Enhanced execution debugging enabled")


# STEP 2: Integration with existing ExecutionEngine
def enhance_existing_execution_engine():
    """Example of how to add debugging to existing ExecutionEngine calls."""
    
    # BEFORE (existing code):
    # engine = ExecutionEngine(ctx)
    # result = engine.execute_order("AAPL", 100, "buy")
    
    # AFTER (with debugging):
    from ai_trading.execution import log_signal_to_execution, log_execution_phase, ExecutionPhase
    
    def execute_order_with_debugging(engine, symbol, qty, side, signal_data=None):
        # Start tracking
        correlation_id = log_signal_to_execution(symbol, side, qty, signal_data)
        
        # Log pre-execution phase
        log_execution_phase(correlation_id, ExecutionPhase.RISK_CHECK, {
            'requested_qty': qty,
            'available_cash': engine.ctx.get_account().cash if hasattr(engine.ctx, 'get_account') else 'unknown'
        })
        
        # Execute the order (existing code)
        try:
            result = engine.execute_order(symbol, qty, side)
            
            if result:
                # Log success
                log_execution_phase(correlation_id, ExecutionPhase.ORDER_FILLED, {
                    'order_id': getattr(result, 'id', 'unknown'),
                    'status': getattr(result, 'status', 'unknown')
                })
                
                # Update position tracking
                from ai_trading.execution import update_bot_position
                current_qty = engine._available_qty(engine.api, symbol)
                update_bot_position(symbol, current_qty, f"execution_{correlation_id}")
                
            else:
                # Log failure
                log_execution_phase(correlation_id, ExecutionPhase.ORDER_REJECTED, {
                    'reason': 'execute_order returned None'
                })
                
        except Exception as e:
            # Log error
            log_execution_phase(correlation_id, ExecutionPhase.ORDER_REJECTED, {
                'error': str(e)
            })
            raise
        
        return result
    
    return execute_order_with_debugging


# STEP 3: Add debugging to signal generation
def add_debugging_to_signals():
    """Example of adding debugging to signal generation."""
    
    def generate_trading_signals_with_debugging(data, strategy_name):
        """Generate signals with execution tracking."""
        from ai_trading.execution import log_signal_to_execution
        
        # Existing signal generation logic
        signals = []  # Your existing signal generation
        
        # Add debugging to each signal
        for signal in signals:
            signal.correlation_id = log_signal_to_execution(
                symbol=signal.symbol,
                side=signal.side,
                qty=signal.quantity,
                signal_data={
                    'strategy': strategy_name,
                    'confidence': signal.confidence,
                    'trigger_price': getattr(signal, 'price', None),
                    'timestamp': signal.timestamp.isoformat() if hasattr(signal, 'timestamp') else None
                }
            )
        
        return signals


# STEP 4: Add PnL tracking to fills
def add_pnl_tracking_to_fills():
    """Example of adding PnL tracking when orders are filled."""
    
    def handle_order_fill_with_pnl_tracking(order, fill_price, fill_qty):
        """Handle order fill with PnL attribution."""
        from ai_trading.execution import record_trade_pnl, update_position_for_pnl
        
        symbol = order.symbol
        side = order.side
        correlation_id = getattr(order, 'correlation_id', None)
        
        # Calculate fees and slippage
        fees = getattr(order, 'fees', 0)
        expected_price = getattr(order, 'limit_price', fill_price)
        slippage = (fill_price - expected_price) * fill_qty if side == 'buy' else (expected_price - fill_price) * fill_qty
        
        # Get average cost basis (you'll need to track this)
        avg_cost = fill_price  # Simplified - in reality you'd calculate this properly
        
        # Record the trade PnL
        record_trade_pnl(
            symbol=symbol,
            trade_qty=fill_qty if side == 'buy' else -fill_qty,
            execution_price=fill_price,
            avg_cost=avg_cost,
            fees=fees,
            slippage=slippage,
            correlation_id=correlation_id
        )
        
        # Update position snapshot for unrealized PnL tracking
        current_qty = get_current_position(symbol)  # Your existing position tracking
        current_market_price = get_current_market_price(symbol)  # Your price feed
        
        update_position_for_pnl(
            symbol=symbol,
            quantity=current_qty,
            avg_cost=avg_cost,
            market_price=current_market_price,
            correlation_id=correlation_id
        )


# STEP 5: Periodic reconciliation checks
def setup_periodic_checks():
    """Setup periodic reconciliation and health checks."""
    
    def periodic_health_check():
        """Run periodic health checks on execution system."""
        from ai_trading.execution import (
            force_position_reconciliation, get_execution_statistics,
            get_portfolio_pnl_summary, get_position_discrepancies
        )
        
        # Check for position discrepancies
        discrepancies = force_position_reconciliation()
        if discrepancies:
            print(f"⚠️ Found {len(discrepancies)} position discrepancies")
            for disc in discrepancies:
                if disc.severity in ['high', 'medium']:
                    print(f"  {disc.symbol}: Bot={disc.bot_qty}, Broker={disc.broker_qty} ({disc.severity})")
        
        # Check execution statistics
        exec_stats = get_execution_statistics()
        success_rate = exec_stats.get('success_rate', 0)
        if success_rate < 0.95:  # Alert if success rate drops below 95%
            print(f"⚠️ Low execution success rate: {success_rate:.1%}")
        
        # Check for unusual PnL patterns
        pnl_summary = get_portfolio_pnl_summary()
        if abs(pnl_summary['total_pnl']) > 1000:  # Alert for large PnL changes
            print(f"📊 Large PnL movement: ${pnl_summary['total_pnl']:+.2f}")
    
    # Schedule periodic checks (you'd integrate this with your existing scheduler)
    # schedule.every(5).minutes.do(periodic_health_check)


# STEP 6: Error handling and alerting
def setup_error_alerting():
    """Setup error alerting for debugging events."""
    
    def check_for_execution_issues():
        """Check for and alert on execution issues."""
        from ai_trading.execution import get_debug_tracker
        
        tracker = get_debug_tracker()
        
        # Check for failed executions
        failed_executions = tracker.get_failed_executions(limit=10)
        recent_failures = [e for e in failed_executions if was_recent(e.get('timestamp'))]
        
        if len(recent_failures) > 3:  # Alert if more than 3 failures recently
            print(f"🚨 ALERT: {len(recent_failures)} failed executions recently")
            for failure in recent_failures:
                print(f"  {failure['symbol']} {failure['side']}: {failure.get('error', 'Unknown error')}")
        
        # Check for stuck orders
        active_orders = tracker.get_active_orders()
        stuck_orders = [o for o in active_orders.values() if was_old(o.get('start_time'))]
        
        if stuck_orders:
            print(f"🚨 ALERT: {len(stuck_orders)} orders appear stuck")
            for order in stuck_orders:
                print(f"  {order['symbol']} {order['side']} - started {order['start_time']}")
    
    def was_recent(timestamp_str, minutes=30):
        """Check if timestamp was within last N minutes."""
        if not timestamp_str:
            return False
        from datetime import datetime, timezone, timedelta
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return timestamp > cutoff
    
    def was_old(timestamp_str, minutes=60):
        """Check if timestamp is older than N minutes."""
        return not was_recent(timestamp_str, minutes)


# STEP 7: Complete integration example
def complete_integration_example():
    """Complete example of integrating debugging into existing bot."""
    
    class EnhancedTradingBot:
        """Example of enhanced trading bot with debugging."""
        
        def __init__(self):
            # Initialize debugging
            enable_enhanced_debugging()
            
            # Your existing initialization
            self.execution_engine = None  # Your ExecutionEngine
            self.strategy = None  # Your strategy
            
        def run_trading_cycle(self):
            """Enhanced trading cycle with debugging."""
            try:
                # Generate signals with debugging
                signals = self.generate_signals_with_debugging()
                
                # Execute signals with debugging
                for signal in signals:
                    self.execute_signal_with_debugging(signal)
                
                # Run health checks
                self.run_health_checks()
                
            except Exception as e:
                print(f"Trading cycle error: {e}")
                # Your existing error handling
        
        def generate_signals_with_debugging(self):
            """Generate signals with debugging integration."""
            # Your existing signal generation
            signals = []  # self.strategy.generate_signals()
            
            # Add debugging correlation IDs
            from ai_trading.execution import log_signal_to_execution
            for signal in signals:
                signal.correlation_id = log_signal_to_execution(
                    symbol=signal.symbol,
                    side=signal.side,
                    qty=signal.quantity,
                    signal_data={
                        'strategy': signal.strategy,
                        'confidence': signal.confidence
                    }
                )
            
            return signals
        
        def execute_signal_with_debugging(self, signal):
            """Execute signal with debugging."""
            from ai_trading.execution import log_execution_phase, ExecutionPhase
            
            correlation_id = signal.correlation_id
            
            # Log pre-execution
            log_execution_phase(correlation_id, ExecutionPhase.RISK_CHECK)
            
            # Execute (your existing code)
            result = self.execution_engine.execute_order(
                signal.symbol, signal.quantity, signal.side
            )
            
            # Log result
            if result:
                log_execution_phase(correlation_id, ExecutionPhase.ORDER_FILLED)
            else:
                log_execution_phase(correlation_id, ExecutionPhase.ORDER_REJECTED)
        
        def run_health_checks(self):
            """Run periodic health checks."""
            from ai_trading.execution import (
                force_position_reconciliation, get_execution_statistics
            )
            
            # Check positions
            discrepancies = force_position_reconciliation()
            if discrepancies:
                print(f"Position discrepancies found: {len(discrepancies)}")
            
            # Check execution health
            stats = get_execution_statistics()
            if stats['success_rate'] < 0.9:
                print(f"Low success rate: {stats['success_rate']:.1%}")


if __name__ == '__main__':
    print("Enhanced Execution Debugging - Integration Guide")
    print("=" * 50)
    print("This file shows how to integrate the new debugging features")
    print("into your existing trading bot code.")
    print()
    print("Key integration points:")
    print("1. Enable debugging at startup")
    print("2. Add correlation tracking to signal generation")
    print("3. Enhance order execution with debugging hooks")
    print("4. Add PnL tracking to order fills")
    print("5. Setup periodic reconciliation checks")
    print("6. Add error alerting for execution issues")
    print()
    print("See the function examples above for implementation details.")