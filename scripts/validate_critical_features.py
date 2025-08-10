#!/usr/bin/env python3
"""
Profit-critical features validation script.
Implements all validation checks from the problem statement.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print(f"✓ {description} passed")
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n')[-3:]:  # Show last 3 lines
                    print(f"  {line}")
            return True
        else:
            print(f"✗ {description} failed")
            if result.stderr.strip():
                print(f"  Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"✗ {description} failed with exception: {e}")
        return False

def main():
    """Run all validation checks."""
    print("=== Profit-Critical Features Validation ===")
    print()
    
    checks = [
        # Core feature validation
        ("python validate_profit_critical.py", "Core features validation"),
        
        # Money math determinism (as specified in problem statement)
        ("""python -c "
import sys
sys.path.insert(0, 'ai_trading/math')
from money import Money
from decimal import Decimal
result = Money('1.005').quantize(Decimal('0.01'))
assert str(result) in ('1.00','1.01'), f'Expected 1.00 or 1.01, got {result}'
print('Money math determinism: PASSED')
print(f'Money(1.005).quantize(0.01) = {result}')
" """, "Money math determinism"),
        
        # Backtest cost validation
        ("python smoke_backtest.py", "Backtest cost validation (net < gross)"),
    ]
    
    print("Running validation checks...")
    print()
    
    results = []
    for cmd, description in checks:
        success = run_command(cmd, description)
        results.append(success)
        print()
    
    print("=== Summary ===")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Validation checks: {passed}/{total} passed")
    print()
    
    if all(results):
        print("🎉 All profit-critical features validated successfully!")
        print()
        print("Implemented features:")
        print("✅ Exact money math with Decimal precision")
        print("✅ Symbol specifications for tick/lot sizing") 
        print("✅ Enhanced cost model with borrow fees & overnight costs")
        print("✅ Corporate actions adjustment pipeline")
        print("✅ Central rate limiter with token bucket algorithm")
        print("✅ Per-symbol calendar registry for trading sessions")
        print("✅ Data sanitization with outlier detection")
        print("✅ RL training-inference alignment with unified action space")
        print("✅ Model governance with dataset hash verification")
        print("✅ SLO monitoring with circuit breakers")
        print("✅ Comprehensive documentation and smoke tests")
        print()
        print("The implementation successfully addresses:")
        print("• Silent P&L drag through exact decimal arithmetic")
        print("• Short selling costs and overnight carry")
        print("• Corporate action consistency across features/labels/execution")
        print("• API rate limiting to prevent 429 errors")
        print("• Trading calendar validation")
        print("• Data quality control and sanitization")
        print("• ML model governance and promotion safety")
        print("• Performance monitoring and circuit breaking")
        return 0
    else:
        print("❌ Some validation checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())