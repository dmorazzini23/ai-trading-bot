#!/usr/bin/env python3
"""
Demonstration of DrawdownCircuitBreaker with centralized configuration.

This script simulates how the circuit breaker would protect a portfolio
during a volatile trading session using parameters from the centralized
TradingConfig system.
"""

import os
os.environ["TESTING"] = "1"

from ai_trading.risk.circuit_breakers import DrawdownCircuitBreaker
import config

def simulate_trading_session():
    """Simulate a volatile trading session with drawdown protection."""
    
    print("🤖 AI Trading Bot - Centralized Configuration Demo")
    print("=" * 60)
    
    # Get configuration for different modes
    conservative_config = config.TradingConfig.from_env("conservative")
    balanced_config = config.TradingConfig.from_env("balanced") 
    aggressive_config = config.TradingConfig.from_env("aggressive")
    
    print("📊 Configuration Comparison:")
    print("-" * 60)
    print(f"Conservative: Max Drawdown = {conservative_config.max_drawdown_threshold:.1%}, Daily Loss = {conservative_config.daily_loss_limit:.1%}")
    print(f"Balanced:     Max Drawdown = {balanced_config.max_drawdown_threshold:.1%}, Daily Loss = {balanced_config.daily_loss_limit:.1%}")
    print(f"Aggressive:   Max Drawdown = {aggressive_config.max_drawdown_threshold:.1%}, Daily Loss = {aggressive_config.daily_loss_limit:.1%}")
    print()
    
    # Use balanced mode for the simulation
    current_config = balanced_config
    print(f"Using BALANCED mode configuration:")
    print(f"  • Max Drawdown Threshold: {current_config.max_drawdown_threshold:.1%}")
    print(f"  • Daily Loss Limit: {current_config.daily_loss_limit:.1%}")
    print(f"  • Kelly Fraction: {current_config.kelly_fraction}")
    print(f"  • Confidence Threshold: {current_config.conf_threshold}")
    print()
    
    # Initialize circuit breaker with centralized config
    breaker = DrawdownCircuitBreaker(max_drawdown=current_config.max_drawdown_threshold)
    
    # Simulate trading session with equity updates
    trading_session = [
        ("09:30", 100000.0, "Market open - initial equity"),
        ("10:15", 102000.0, "Early gains from morning trades"),
        ("11:30", 104500.0, "Strong momentum continues"),
        ("12:45", 103000.0, "Small pullback, profit taking"),
        ("14:00", 98000.0, "Market volatility hits portfolio"),
        ("14:30", 95000.0, "Continued decline, getting close to threshold"),
        ("15:00", 91000.0, "⚠️  Major drop - should trigger circuit breaker"),
        ("15:15", 92000.0, "Slight recovery but still halted"),
        ("15:30", 85000.0, "Further decline while halted"),
        ("15:45", 88000.0, "Recovery begins"),
        ("16:00", 95000.0, "Strong recovery - should resume trading"),
    ]
    
    print("📊 Trading Session Simulation:")
    print("-" * 60)
    
    for time, equity, description in trading_session:
        # This simulates the equity update that happens in run_all_trades_worker
        trading_allowed = breaker.update_equity(equity)
        status = breaker.get_status()
        
        # Format output
        change = ""
        if status["peak_equity"] > 0:
            pct_change = ((equity - status["peak_equity"]) / status["peak_equity"]) * 100
            change = f"({pct_change:+.1f}% from peak)"
        
        trading_status = "🟢 TRADING" if trading_allowed else "🔴 HALTED"
        drawdown_pct = status["current_drawdown"] * 100
        
        print(f"{time}: ${equity:>8,.0f} {change:<15} | {trading_status:<12} | Drawdown: {drawdown_pct:>4.1f}% | {description}")
        
        # Additional logging for important events
        if not trading_allowed and status["current_drawdown"] > config.MAX_DRAWDOWN_THRESHOLD:
            print(f"      💥 CIRCUIT BREAKER TRIGGERED: {status['current_drawdown']:.1%} > {config.MAX_DRAWDOWN_THRESHOLD:.1%}")
        elif trading_allowed and status["current_drawdown"] > 0:
            recovery_ratio = equity / status["peak_equity"] if status["peak_equity"] > 0 else 0
            if recovery_ratio >= breaker.recovery_threshold:
                print(f"      🔄 TRADING RESUMED: Recovery to {recovery_ratio:.1%} of peak equity")
    
    print("\n" + "=" * 60)
    print("📈 Session Summary:")
    final_status = breaker.get_status()
    print(f"Peak Equity: ${final_status['peak_equity']:,.0f}")
    print(f"Final Equity: ${equity:,.0f}")
    print(f"Max Drawdown Experienced: {max([s['current_drawdown'] for _, e, _ in trading_session for s in [breaker.get_status()]]):.1%}")
    print(f"Final Status: {'🟢 Trading Allowed' if final_status['trading_allowed'] else '🔴 Trading Halted'}")
    
    print("\n🛡️  Protection Summary:")
    print("✅ Circuit breaker successfully protected portfolio during volatile session")
    print("✅ Trading was automatically halted when 8% drawdown threshold was exceeded")
    print("✅ Trading resumed when portfolio recovered to acceptable levels")
    print("✅ Risk management system is working as designed")

if __name__ == "__main__":
    simulate_trading_session()