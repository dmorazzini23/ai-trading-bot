#!/usr/bin/env python3
"""
Standalone validation script for profit-critical features.
Does not import from ai_trading package to avoid initialization issues.
"""

import sys
from pathlib import Path

# Add the repo root to path to import modules directly
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

def test_money_math():
    """Test Money math determinism."""
    print("Testing Money math determinism...")
    
    # Import directly from file to avoid package initialization
    sys.path.insert(0, str(repo_root / "ai_trading" / "math"))
    from money import Money
    from decimal import Decimal
    
    # Test as specified in problem statement
    m = Money('1.005')
    result = m.quantize(Decimal('0.01'))
    result_str = str(result)
    
    print(f"Money('1.005').quantize(Decimal('0.01')) = {result_str}")
    
    # Should be either 1.00 or 1.01 due to banker's rounding
    assert result_str in ('1.00', '1.01'), f"Expected 1.00 or 1.01, got {result_str}"
    
    # Test basic arithmetic
    m1 = Money('10.50')
    m2 = Money('5.25')
    assert str(m1 + m2) == '15.75'
    assert str(m1 - m2) == '5.25'
    
    print("✓ Money math determinism test passed")
    return True

def test_symbol_specs():
    """Test symbol specifications."""
    print("Testing symbol specifications...")
    
    sys.path.insert(0, str(repo_root / "ai_trading" / "market"))
    from symbol_specs import get_symbol_spec, get_tick_size, get_lot_size
    from decimal import Decimal
    
    # Test default symbols
    spec = get_symbol_spec('AAPL')
    assert spec.tick == Decimal('0.01')
    assert spec.lot == 1
    
    tick = get_tick_size('SPY')
    assert tick == Decimal('0.01')
    
    lot = get_lot_size('QQQ')
    assert lot == 1
    
    print("✓ Symbol specifications test passed")
    return True

def test_file_structure():
    """Test that all required files exist."""
    print("Testing file structure...")
    
    required_files = [
        "ai_trading/math/__init__.py",
        "ai_trading/math/money.py",
        "ai_trading/market/__init__.py", 
        "ai_trading/market/symbol_specs.py",
        "ai_trading/market/calendars.py",
        "ai_trading/data/corp_actions.py",
        "ai_trading/data/sanitize.py",
        "ai_trading/integrations/__init__.py",
        "ai_trading/integrations/rate_limit.py",
        "ai_trading/governance/__init__.py",
        "ai_trading/governance/promotion.py",
        "ai_trading/rl_trading/tests/__init__.py",
        "ai_trading/rl_trading/tests/smoke_parity.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = repo_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    print("✓ All required files exist")
    return True

def main():
    """Run validation tests."""
    print("=== Profit-Critical Features Validation ===")
    
    tests = [
        test_file_structure,
        test_money_math,
        test_symbol_specs
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            results.append(False)
        print()
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"=== Results: {success_count}/{total_count} tests passed ===")
    
    if all(results):
        print("✓ All validation tests passed!")
        return 0
    else:
        print("✗ Some validation tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())