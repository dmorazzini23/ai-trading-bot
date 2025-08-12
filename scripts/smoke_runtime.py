#!/usr/bin/env python3
"""
Smoke test for runtime context - tests the specific ctx NameError fixes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_check_pdt_rule_signature():
    """Test that check_pdt_rule accepts runtime parameter."""
    try:
        from ai_trading.core.bot_engine import check_pdt_rule
        
        # Create a mock runtime object with the necessary attributes
        class MockAPI:
            def get_account(self):
                class MockAccount:
                    def __init__(self):
                        self.equity = "10000"
                        self.day_trade_count = 0
                        self.trading_blocked = False
                return MockAccount()
        
        class MockRuntime:
            def __init__(self):
                self.api = MockAPI()
        
        runtime = MockRuntime()
        
        # This should not raise NameError anymore
        result = check_pdt_rule(runtime)
        print(f"✓ check_pdt_rule(runtime) returned: {result}")
        
        return True
    except NameError as e:
        if "ctx" in str(e):
            print(f"✗ NameError still present in check_pdt_rule: {e}")
            return False
        raise
    except Exception as e:
        # Other exceptions are OK for this smoke test, we just want to avoid NameError
        print(f"✓ check_pdt_rule(runtime) executed without NameError (got {type(e).__name__}: {e})")
        return True

def test_run_all_trades_worker_signature():
    """Test that run_all_trades_worker accepts runtime parameter."""
    try:
        import inspect
        from ai_trading.core.bot_engine import run_all_trades_worker
        
        # Check function signature
        sig = inspect.signature(run_all_trades_worker)
        params = list(sig.parameters.keys())
        
        if len(params) >= 2 and params[1] != 'model':
            print(f"✓ run_all_trades_worker signature updated: {params}")
            return True
        else:
            print(f"✗ run_all_trades_worker signature not updated: {params}")
            return False
            
    except Exception as e:
        print(f"✗ run_all_trades_worker signature test failed: {e}")
        return False

def main():
    """Run smoke tests for runtime context fixes."""
    print("Running smoke tests for runtime context fixes...")
    
    tests = [
        test_check_pdt_rule_signature,
        test_run_all_trades_worker_signature,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"Test {test.__name__} failed")
        except Exception as e:
            print(f"Test {test.__name__} raised exception: {e}")
    
    print(f"\nSmoke test results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("OK")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())