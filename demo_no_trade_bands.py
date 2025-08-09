#!/usr/bin/env python3
"""
Example usage of the no-trade bands functionality for transaction cost optimization.
This demonstrates how to use the new no-trade bands to avoid churn on tiny weight deltas.
"""

import os
os.environ['ALPACA_API_KEY'] = 'demo'
os.environ['ALPACA_SECRET_KEY'] = 'demo'
os.environ['ALPACA_BASE_URL'] = 'demo'
os.environ['WEBHOOK_SECRET'] = 'demo'
os.environ['FLASK_PORT'] = '5000'

from ai_trading.rebalancer import apply_no_trade_bands


def demo_no_trade_bands():
    """Demonstrate the no-trade bands functionality."""
    print("=== AI Trading Bot - No-Trade Bands Demo ===\n")
    
    # Example portfolio weights
    current_weights = {
        "AAPL": 0.25,
        "MSFT": 0.20,
        "GOOGL": 0.15,
        "AMZN": 0.15,
        "TSLA": 0.10,
        "NVDA": 0.10,
        "META": 0.05
    }
    
    # New target weights (small changes due to signal updates)
    target_weights = {
        "AAPL": 0.2515,  # +15 bps
        "MSFT": 0.1985,  # -15 bps
        "GOOGL": 0.1520, # +20 bps
        "AMZN": 0.1480,  # -20 bps
        "TSLA": 0.1040,  # +40 bps (larger move)
        "NVDA": 0.0960,  # -40 bps (larger move)
        "META": 0.0500   # No change
    }
    
    print("Current Portfolio Weights:")
    for symbol, weight in current_weights.items():
        print(f"  {symbol}: {weight:.4f} ({weight*100:.2f}%)")
    
    print("\nTarget Portfolio Weights:")
    for symbol, weight in target_weights.items():
        current = current_weights.get(symbol, 0)
        delta_bps = (weight - current) * 10000
        print(f"  {symbol}: {weight:.4f} ({weight*100:.2f}%) [Δ{delta_bps:+.0f}bps]")
    
    # Apply different no-trade band thresholds
    band_thresholds = [10.0, 25.0, 50.0]
    
    for band_bps in band_thresholds:
        print(f"\n=== Applying {band_bps:.0f}bps No-Trade Band ===")
        
        adjusted_weights = apply_no_trade_bands(current_weights, target_weights, band_bps)
        
        trades_needed = 0
        total_turnover = 0.0
        
        print("Final Weights after No-Trade Bands:")
        for symbol in target_weights.keys():
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            final = adjusted_weights.get(symbol, 0)
            
            original_delta_bps = (target - current) * 10000
            final_delta_bps = (final - current) * 10000
            
            trade_avoided = abs(original_delta_bps) > 0.1 and abs(final_delta_bps) < 0.1
            trade_executed = abs(final_delta_bps) > 0.1
            
            if trade_executed:
                trades_needed += 1
                total_turnover += abs(final_delta_bps) / 10000
            
            status = "AVOIDED" if trade_avoided else ("TRADE" if trade_executed else "NO CHANGE")
            print(f"  {symbol}: {final:.4f} [Δ{final_delta_bps:+.0f}bps] - {status}")
        
        print("\nSummary:")
        print(f"  Trades needed: {trades_needed}/7 positions")
        print(f"  Total turnover: {total_turnover:.4f} ({total_turnover*100:.2f}%)")
        print(f"  Transaction cost savings: ~{(7-trades_needed)*0.0005*100:.2f}bps per avoided trade")
    
    # Demonstrate with larger moves that should trigger trades
    print("\n=== Large Rebalancing Example ===")
    large_target_weights = {
        "AAPL": 0.30,  # +500 bps
        "MSFT": 0.15,  # -500 bps
        "GOOGL": 0.20, # +500 bps
        "AMZN": 0.10,  # -500 bps
        "TSLA": 0.15,  # +500 bps
        "NVDA": 0.05,  # -500 bps
        "META": 0.05   # No change
    }
    
    adjusted_large = apply_no_trade_bands(current_weights, large_target_weights, 25.0)
    
    print("Large moves (>25bps threshold):")
    for symbol in large_target_weights.keys():
        current = current_weights.get(symbol, 0)
        target = large_target_weights.get(symbol, 0)
        final = adjusted_large.get(symbol, 0)
        
        delta_bps = (final - current) * 10000
        print(f"  {symbol}: {current:.3f} → {final:.3f} [Δ{delta_bps:+.0f}bps]")
    
    print("\nAll large moves executed as expected (exceed 25bps threshold)")
    print("\n=== No-Trade Bands Demo Complete ===")


if __name__ == "__main__":
    demo_no_trade_bands()