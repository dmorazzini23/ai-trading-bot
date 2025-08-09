#!/usr/bin/env python3
"""Debug the drawdown circuit breaker status variable issue."""

import os
import sys
import traceback

# Set testing environment
os.environ["TESTING"] = "1"

def test_drawdown_circuit_breaker():
    """Test drawdown circuit breaker for the status variable issue."""
    try:
        print("🔍 Testing DrawdownCircuitBreaker...")
        
        from ai_trading.risk.circuit_breakers import DrawdownCircuitBreaker
        
        # Initialize circuit breaker
        breaker = DrawdownCircuitBreaker(max_drawdown=0.08)  # 8% like in your logs
        print("✅ Circuit breaker initialized successfully")
        
        # Test normal equity update
        print("\n📊 Testing equity updates...")
        result1 = breaker.update_equity(100000)
        print(f"Initial equity update: {result1}")
        
        # Test status method
        print("\n🔍 Testing get_status method...")
        status = breaker.get_status()
        print(f"Status keys: {list(status.keys())}")
        print(f"Full status: {status}")
        
        # Test drawdown scenario
        print("\n⚠️ Testing drawdown scenario...")
        result2 = breaker.update_equity(95000)  # 5% loss
        print(f"5% loss result: {result2}")
        
        result3 = breaker.update_equity(90000)  # 10% loss (should trigger)
        print(f"10% loss result: {result3}")
        
        final_status = breaker.get_status()
        print(f"Final status: {final_status}")
        
        print("✅ All tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error in drawdown test: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_status_variable_issue():
    """Specifically test for the status variable issue."""
    try:
        print("\n🔍 Testing for status variable issue...")
        
        from ai_trading.risk.circuit_breakers import DrawdownCircuitBreaker
        
        breaker = DrawdownCircuitBreaker(max_drawdown=0.08)
        
        # Try to reproduce the exact scenario from logs
        breaker.update_equity(88519.46)  # Your equity from logs
        
        # Force multiple updates to see if we can reproduce the error
        for i in range(5):
            try:
                equity = 88519.46 * (1 - 0.01 * i)  # Gradual loss
                result = breaker.update_equity(equity)
                status = breaker.get_status()
                print(f"Update {i+1}: equity=${equity:.2f}, result={result}, state={status['state']}")
            except Exception as e:
                print(f"❌ Error on update {i+1}: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                return False
        
        print("✅ Status variable test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error in status variable test: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🚀 Starting DrawdownCircuitBreaker Debug Session")
    print("=" * 60)
    
    success1 = test_drawdown_circuit_breaker()
    success2 = test_status_variable_issue()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ All tests passed - Circuit breaker appears functional")
    else:
        print("❌ Some tests failed - Issues detected")
        sys.exit(1)
