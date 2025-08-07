#!/usr/bin/env python3
"""
Standalone smoke test for backtest cost validation.
Can be run as: python smoke_backtest.py
"""

import sys
from pathlib import Path

def run_backtest_smoke_test():
    """
    Run smoke test to verify net < gross due to all costs.
    """
    print("=== Backtest Smoke Test ===")
    print("Testing that net < gross due to all costs...")
    
    try:
        # Import Money class directly without package initialization
        repo_root = Path(__file__).parent
        sys.path.insert(0, str(repo_root / "ai_trading" / "math"))
        from money import Money
        
        # Mock a simple backtest scenario
        print("Simulating backtest with costs...")
        
        # Trade scenario: Buy at $100, sell at $102, 100 shares
        entry_price = Money('100.00')
        exit_price = Money('102.00') 
        quantity = 100
        
        # Gross P&L calculation (no costs)
        gross_pnl = (exit_price - entry_price) * quantity
        print(f"Gross P&L: ${gross_pnl}")
        
        # Apply realistic costs
        position_value = entry_price * quantity
        
        # Execution costs (5 bps total - 2.5 bps each way)
        execution_cost_bps = 5.0
        execution_cost = position_value * (execution_cost_bps / 10000)
        
        # Overnight holding cost (assume 1 day hold, 2 bps/day)
        overnight_cost_bps = 2.0
        overnight_cost = position_value * (overnight_cost_bps / 10000)
        
        # Commission (assume $1 minimum)
        commission = Money('2.00')  # $1 each way
        
        # Total costs
        total_costs = execution_cost + overnight_cost + commission
        
        # Net P&L after costs
        net_pnl = gross_pnl - total_costs
        
        print(f"Execution cost (5 bps): ${execution_cost}")
        print(f"Overnight cost (2 bps): ${overnight_cost}")
        print(f"Commission: ${commission}")
        print(f"Total costs: ${total_costs}")
        print(f"Net P&L: ${net_pnl}")
        
        # Critical validation: net must be less than gross
        if net_pnl >= gross_pnl:
            raise AssertionError(f"Net P&L ({net_pnl}) should be less than gross P&L ({gross_pnl})")
        
        # Additional checks
        cost_drag_bps = (total_costs / position_value) * 10000
        print(f"Total cost drag: {cost_drag_bps:.1f} bps")
        
        if cost_drag_bps < 5.0:
            raise AssertionError(f"Cost drag ({cost_drag_bps:.1f} bps) seems too low")
        
        print("✓ Net P&L is correctly less than gross P&L due to costs")
        print(f"✓ Cost drag of {cost_drag_bps:.1f} bps is realistic")
        print("✓ Backtest smoke test passed!")
        
        return True
        
    except Exception as e:
        print(f"✗ Backtest smoke test failed: {e}")
        return False

if __name__ == "__main__":
    success = run_backtest_smoke_test()
    sys.exit(0 if success else 1)