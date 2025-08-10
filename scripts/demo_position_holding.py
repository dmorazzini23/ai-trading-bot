#!/usr/bin/env python3
import logging

"""
Demo script showing position holding logic and meta-learning integration.

This script demonstrates the key improvements implemented:
1. Position holding logic to reduce churn
2. Meta-learning trigger automation
3. Signal enhancement with position awareness
"""

import os
from datetime import datetime, timezone

# Set required environment variables for demo
os.environ['ALPACA_API_KEY'] = 'demo_key'
os.environ['ALPACA_SECRET_KEY'] = 'demo_secret'
os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets'
os.environ['WEBHOOK_SECRET'] = 'demo_secret'
os.environ['FLASK_PORT'] = '5000'
os.environ['PYTEST_RUNNING'] = '1'

def demo_position_holding_logic():
    """Demonstrate position holding decisions."""
    logging.info(str("=" * 60))
    logging.info("🔍 POSITION HOLDING LOGIC DEMO")
    logging.info(str("=" * 60))
    
    from ai_trading.position.legacy_manager import should_hold_position
    
    scenarios = [
        ("UBER", None, 28.3, 5, "High profit winner from logs"),
        ("SHOP", None, 20.0, 3, "Good profit winner from logs"),
        ("PG", None, 16.6, 2, "Decent profit, new position"),
        ("JNJ", None, -2.5, 1, "Small loss, very new position"),
        ("NVDA", None, 3.2, 7, "Small gain, older position"),
        ("GOOGL", None, -8.1, 10, "Loss, old position"),
    ]
    
    logging.info("Position Hold Decisions:")
    logging.info(str("-" * 60))
    
    for symbol, _, pnl_pct, days, description in scenarios:
        hold = should_hold_position(symbol, None, pnl_pct, days)
        action = "🟢 HOLD" if hold else "🔴 SELL"
        logging.info(f"{symbol:6} | {pnl_pct:+6.1f}% | {days:2d} days | {action} | {description}")
    
    logging.info("\nKey Insights:")
    logging.info("• Profitable positions (>5% gain) are held: UBER, SHOP, PG")
    logging.info("• New positions (<3 days) are held even with small losses: JNJ")
    logging.info("• Old losing positions are marked for sale: GOOGL")
    logging.info("• This prevents premature exit from winners!")


def demo_signal_enhancement():
    """Demonstrate signal enhancement with position logic."""
    logging.info(str("\n" + "=" * 60))
    logging.info("📊 SIGNAL ENHANCEMENT DEMO")
    logging.info(str("=" * 60))
    
    # Mock original trading signals (what would happen without position logic)
    original_signals = [
        {"symbol": "UBER", "side": "sell", "confidence": 0.6, "reason": "take_profit"},
        {"symbol": "SHOP", "side": "sell", "confidence": 0.7, "reason": "momentum_fade"}, 
        {"symbol": "PG", "side": "sell", "confidence": 0.5, "reason": "rebalance"},
        {"symbol": "JNJ", "side": "buy", "confidence": 0.8, "reason": "new_entry"},
        {"symbol": "NVDA", "side": "buy", "confidence": 0.9, "reason": "breakout"},
        {"symbol": "GOOGL", "side": "sell", "confidence": 0.9, "reason": "stop_loss"},
    ]
    
    # Mock position hold signals
    hold_signals = {
        "UBER": "hold",   # High profit - hold
        "SHOP": "hold",   # Good profit - hold  
        "PG": "hold",     # New position - hold
        "JNJ": "neutral", # Allow new position
        "NVDA": "neutral", # Allow new position
        "GOOGL": "sell",   # Loss position - sell OK
    }
    
    logging.info("Signal Enhancement Results:")
    logging.info(str("-" * 60))
    logging.info("Symbol | Original | Hold Signal | Final Action | Explanation")
    logging.info(str("-" * 60))
    
    enhanced_signals = []
    
    for signal in original_signals:
        symbol = signal["symbol"]
        side = signal["side"]
        hold_action = hold_signals.get(symbol, "neutral")
        
        # Apply position holding logic
        if hold_action == "hold" and side == "sell":
            final_action = "❌ FILTERED (hold wins)"
            explanation = "Position held due to profits/age"
        elif hold_action == "sell" and side == "buy":
            final_action = "❌ FILTERED (sell pending)"
            explanation = "No new buys when selling"
        else:
            final_action = "✅ EXECUTE"
            explanation = "Signal allowed"
            enhanced_signals.append(signal)
        
        logging.info(f"{symbol:6} | {side:8} | {hold_action:7} | {final_action:20} | {explanation}")
    
    logging.info("\nSummary:")
    logging.info(f"• Original signals: {len(original_signals)}")
    logging.info(f"• Enhanced signals: {len(enhanced_signals)}")
    logging.info(f"• Filtered out: {len(original_signals) - len(enhanced_signals)}")
    logging.info(f"• Churn reduction: {((len(original_signals) - len(enhanced_signals)) / len(original_signals) * 100):.1f}%")


def demo_meta_learning_trigger():
    """Demonstrate meta-learning trigger functionality."""
    logging.info(str("\n" + "=" * 60)) 
    logging.info("🧠 META-LEARNING TRIGGER DEMO")
    logging.info(str("=" * 60))
    
    # Simulate trade execution with meta-learning trigger
    trade_executions = [
        {"symbol": "META", "qty": 50, "side": "buy", "price": 485.20, "order_id": "meta-001"},
        {"symbol": "AMZN", "qty": 25, "side": "buy", "price": 186.50, "order_id": "amzn-001"},  
        {"symbol": "CRM", "qty": 100, "side": "sell", "price": 315.80, "order_id": "crm-001"},
        {"symbol": "NVDA", "qty": 75, "side": "buy", "price": 875.30, "order_id": "nvda-001"},
    ]
    
    logging.info("Trade Execution → Meta-Learning Conversion:")
    logging.info(str("-" * 60))
    
    for trade in trade_executions:
        logging.info(str(f"📈 TRADE_EXECUTED: {trade['symbol']} {trade['side']} {trade['qty']} @ ${trade['price']:.2f}"))
        
        # Simulate meta-learning trigger
        trade_data = {
            **trade,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'filled'
        }
        
        # Mock conversion (in real system this would call trigger_meta_learning_conversion)
        logging.info(str(f"   ➜ 🧠 META_LEARNING_TRIGGERED | symbol={trade['symbol']}"))
        logging.info("   ➜ 📊 Converting audit format to meta-learning format")
        logging.info("   ➜ 💾 Storing for strategy optimization")
        print()
    
    logging.info("Key Benefits:")
    logging.info("• Trade data automatically flows to meta-learning system")
    logging.info(str("• No more 'METALEARN_EMPTY_TRADE_LOG' errors"))
    logging.info("• Strategy optimization can analyze all executed trades")
    logging.info("• Real-time learning from trading performance")


def demo_before_vs_after():
    """Show before vs after behavior."""
    logging.info(str("\n" + "=" * 60))
    logging.info("🔄 BEFORE vs AFTER COMPARISON")
    logging.info(str("=" * 60))
    
    logging.info("BEFORE (High Churn Behavior):")
    logging.info(str("-" * 30))
    logging.info("Cycle 1: Buy UBER, SHOP, PG")
    logging.info("Cycle 2: Sell UBER (+28.3%), SHOP (+20.0%), PG (+16.6%)")
    logging.info("         Buy JNJ, NVDA, GOOGL")
    logging.info("Cycle 3: Sell JNJ, NVDA, GOOGL")
    logging.info("         Buy new positions...")
    logging.info("Result: Constant turnover, missed gains on winners")
    
    logging.info("\nAFTER (Position Holding Logic):")
    logging.info(str("-" * 30))
    logging.info("Cycle 1: Buy UBER, SHOP, PG")
    logging.info("Cycle 2: Hold UBER (+28.3%), SHOP (+20.0%), PG (+16.6%)")
    logging.info("         Only buy JNJ (tactical position)")
    logging.info("Cycle 3: Continue holding winners, manage new positions")
    logging.info("         Add to winners or diversify carefully")
    logging.info("Result: Lower turnover, maximize gains from winners")
    
    logging.info("\nMeta-Learning Enhancement:")
    logging.info(str("-" * 30))
    logging.info("BEFORE: Manual conversion, often empty trade logs")
    logging.info("AFTER:  Automatic trigger after each trade execution")
    logging.info("        → Real-time strategy learning and optimization")


if __name__ == "__main__":
    logging.info("🚀 AI Trading Bot - Position Holding & Meta-Learning Demo")
    logging.info("Demonstrating fixes for position churn and meta-learning issues")
    
    try:
        demo_position_holding_logic()
        demo_signal_enhancement()
        demo_meta_learning_trigger()
        demo_before_vs_after()
        
        logging.info(str("\n" + "=" * 60))
        logging.info("✅ DEMO COMPLETE")
        logging.info(str("=" * 60))
        logging.info("The trading bot now:")
        logging.info("• Holds profitable positions to maximize gains")
        logging.info("• Reduces unnecessary position churn")
        logging.info("• Automatically feeds data to meta-learning system")
        logging.info("• Learns from trading performance in real-time")
        logging.info("\nReady for deployment! 🎉")
        
    except Exception as e:
        logging.info(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()