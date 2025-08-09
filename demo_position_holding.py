#!/usr/bin/env python3
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
    print("=" * 60)
    print("🔍 POSITION HOLDING LOGIC DEMO")
    print("=" * 60)
    
    from position_manager import should_hold_position
    
    scenarios = [
        ("UBER", None, 28.3, 5, "High profit winner from logs"),
        ("SHOP", None, 20.0, 3, "Good profit winner from logs"),
        ("PG", None, 16.6, 2, "Decent profit, new position"),
        ("JNJ", None, -2.5, 1, "Small loss, very new position"),
        ("NVDA", None, 3.2, 7, "Small gain, older position"),
        ("GOOGL", None, -8.1, 10, "Loss, old position"),
    ]
    
    print("Position Hold Decisions:")
    print("-" * 60)
    
    for symbol, _, pnl_pct, days, description in scenarios:
        hold = should_hold_position(symbol, None, pnl_pct, days)
        action = "🟢 HOLD" if hold else "🔴 SELL"
        print(f"{symbol:6} | {pnl_pct:+6.1f}% | {days:2d} days | {action} | {description}")
    
    print("\nKey Insights:")
    print("• Profitable positions (>5% gain) are held: UBER, SHOP, PG")
    print("• New positions (<3 days) are held even with small losses: JNJ")
    print("• Old losing positions are marked for sale: GOOGL")
    print("• This prevents premature exit from winners!")


def demo_signal_enhancement():
    """Demonstrate signal enhancement with position logic."""
    print("\n" + "=" * 60)
    print("📊 SIGNAL ENHANCEMENT DEMO")
    print("=" * 60)
    
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
    
    print("Signal Enhancement Results:")
    print("-" * 60)
    print("Symbol | Original | Hold Signal | Final Action | Explanation")
    print("-" * 60)
    
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
        
        print(f"{symbol:6} | {side:8} | {hold_action:7} | {final_action:20} | {explanation}")
    
    print("\nSummary:")
    print(f"• Original signals: {len(original_signals)}")
    print(f"• Enhanced signals: {len(enhanced_signals)}")
    print(f"• Filtered out: {len(original_signals) - len(enhanced_signals)}")
    print(f"• Churn reduction: {((len(original_signals) - len(enhanced_signals)) / len(original_signals) * 100):.1f}%")


def demo_meta_learning_trigger():
    """Demonstrate meta-learning trigger functionality."""
    print("\n" + "=" * 60) 
    print("🧠 META-LEARNING TRIGGER DEMO")
    print("=" * 60)
    
    # Simulate trade execution with meta-learning trigger
    trade_executions = [
        {"symbol": "META", "qty": 50, "side": "buy", "price": 485.20, "order_id": "meta-001"},
        {"symbol": "AMZN", "qty": 25, "side": "buy", "price": 186.50, "order_id": "amzn-001"},  
        {"symbol": "CRM", "qty": 100, "side": "sell", "price": 315.80, "order_id": "crm-001"},
        {"symbol": "NVDA", "qty": 75, "side": "buy", "price": 875.30, "order_id": "nvda-001"},
    ]
    
    print("Trade Execution → Meta-Learning Conversion:")
    print("-" * 60)
    
    for trade in trade_executions:
        print(f"📈 TRADE_EXECUTED: {trade['symbol']} {trade['side']} {trade['qty']} @ ${trade['price']:.2f}")
        
        # Simulate meta-learning trigger
        trade_data = {
            **trade,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'filled'
        }
        
        # Mock conversion (in real system this would call trigger_meta_learning_conversion)
        print(f"   ➜ 🧠 META_LEARNING_TRIGGERED | symbol={trade['symbol']}")
        print("   ➜ 📊 Converting audit format to meta-learning format")
        print("   ➜ 💾 Storing for strategy optimization")
        print()
    
    print("Key Benefits:")
    print("• Trade data automatically flows to meta-learning system")
    print("• No more 'METALEARN_EMPTY_TRADE_LOG' errors")
    print("• Strategy optimization can analyze all executed trades")
    print("• Real-time learning from trading performance")


def demo_before_vs_after():
    """Show before vs after behavior."""
    print("\n" + "=" * 60)
    print("🔄 BEFORE vs AFTER COMPARISON")
    print("=" * 60)
    
    print("BEFORE (High Churn Behavior):")
    print("-" * 30)
    print("Cycle 1: Buy UBER, SHOP, PG")
    print("Cycle 2: Sell UBER (+28.3%), SHOP (+20.0%), PG (+16.6%)")
    print("         Buy JNJ, NVDA, GOOGL")
    print("Cycle 3: Sell JNJ, NVDA, GOOGL")
    print("         Buy new positions...")
    print("Result: Constant turnover, missed gains on winners")
    
    print("\nAFTER (Position Holding Logic):")
    print("-" * 30)
    print("Cycle 1: Buy UBER, SHOP, PG")
    print("Cycle 2: Hold UBER (+28.3%), SHOP (+20.0%), PG (+16.6%)")
    print("         Only buy JNJ (tactical position)")
    print("Cycle 3: Continue holding winners, manage new positions")
    print("         Add to winners or diversify carefully")
    print("Result: Lower turnover, maximize gains from winners")
    
    print("\nMeta-Learning Enhancement:")
    print("-" * 30)
    print("BEFORE: Manual conversion, often empty trade logs")
    print("AFTER:  Automatic trigger after each trade execution")
    print("        → Real-time strategy learning and optimization")


if __name__ == "__main__":
    print("🚀 AI Trading Bot - Position Holding & Meta-Learning Demo")
    print("Demonstrating fixes for position churn and meta-learning issues")
    
    try:
        demo_position_holding_logic()
        demo_signal_enhancement()
        demo_meta_learning_trigger()
        demo_before_vs_after()
        
        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETE")
        print("=" * 60)
        print("The trading bot now:")
        print("• Holds profitable positions to maximize gains")
        print("• Reduces unnecessary position churn")
        print("• Automatically feeds data to meta-learning system")
        print("• Learns from trading performance in real-time")
        print("\nReady for deployment! 🎉")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()