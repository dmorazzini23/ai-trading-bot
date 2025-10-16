#!/usr/bin/env python3
"""
Verification script to ensure PDT integration is properly loaded.
Run this before starting the trading bot to verify all fixes are in place.
"""

import sys
import os

print("=" * 80)
print("PDT INTEGRATION VERIFICATION")
print("=" * 80)

# Add the bot directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

errors = []
warnings = []

# Test 1: Check if modules exist
print("\n1. Checking if PDT modules exist...")
try:
    from ai_trading.execution import pdt_manager, swing_mode
    print("   ✅ PDT manager module found")
    print("   ✅ Swing mode module found")
except ImportError as e:
    errors.append(f"Module import failed: {e}")
    print(f"   ❌ {e}")

# Test 2: Check if integration code exists in live_trading.py
print("\n2. Checking integration code in live_trading.py...")
try:
    with open('ai_trading/execution/live_trading.py', 'r') as f:
        content = f.read()
        
    if 'PDT_STATUS_CHECK' in content:
        print("   ✅ PDT status check code found")
    else:
        errors.append("PDT_STATUS_CHECK not found in live_trading.py")
        print("   ❌ PDT_STATUS_CHECK not found")
        
    if 'PDT_LIMIT_EXCEEDED_SWING_MODE_ACTIVATED' in content:
        print("   ✅ Swing mode activation code found")
    else:
        errors.append("Swing mode activation code not found")
        print("   ❌ Swing mode activation code not found")
        
    if 'SWING_MODE_ENTRY_RECORDED' in content:
        print("   ✅ Swing entry recording code found")
    else:
        errors.append("Swing entry recording code not found")
        print("   ❌ Swing entry recording code not found")
        
except Exception as e:
    errors.append(f"Failed to read live_trading.py: {e}")
    print(f"   ❌ {e}")

# Test 3: Verify PDT manager can be instantiated
print("\n3. Testing PDT manager instantiation...")
try:
    from ai_trading.execution.pdt_manager import PDTManager
    from types import SimpleNamespace
    
    manager = PDTManager()
    test_account = SimpleNamespace(
        pattern_day_trader=True,
        daytrade_count=6,
        daytrade_limit=3
    )
    
    status = manager.get_pdt_status(test_account)
    if status.strategy_recommendation == "swing_only":
        print("   ✅ PDT manager working correctly")
        print(f"      - Detected PDT: {status.is_pattern_day_trader}")
        print(f"      - Day trades: {status.daytrade_count}/{status.daytrade_limit}")
        print(f"      - Strategy: {status.strategy_recommendation}")
    else:
        warnings.append("PDT manager not recommending swing_only for 6/3 scenario")
        print("   ⚠️  PDT manager logic may be incorrect")
except Exception as e:
    errors.append(f"PDT manager test failed: {e}")
    print(f"   ❌ {e}")

# Test 4: Verify swing mode can be enabled
print("\n4. Testing swing mode functionality...")
try:
    from ai_trading.execution.swing_mode import get_swing_mode, enable_swing_mode
    
    swing_mode = get_swing_mode()
    initial_state = swing_mode.enabled
    
    enable_swing_mode()
    enabled_state = swing_mode.enabled
    
    if enabled_state:
        print("   ✅ Swing mode can be enabled")
        print(f"      - Initial state: {initial_state}")
        print(f"      - After enable: {enabled_state}")
    else:
        errors.append("Swing mode failed to enable")
        print("   ❌ Swing mode failed to enable")
except Exception as e:
    errors.append(f"Swing mode test failed: {e}")
    print(f"   ❌ {e}")

# Test 5: Check git branch
print("\n5. Checking git branch...")
try:
    import subprocess
    result = subprocess.run(['git', 'branch', '--show-current'], 
                          capture_output=True, text=True, cwd=os.path.dirname(__file__))
    branch = result.stdout.strip()
    if branch in ['fix/pdt-complete-solution', 'fix/critical-quantity-and-order-bugs']:
        print(f"   ✅ On correct branch: {branch}")
    else:
        warnings.append(f"On unexpected branch: {branch}")
        print(f"   ⚠️  On branch: {branch} (expected fix/pdt-complete-solution)")
except Exception as e:
    warnings.append(f"Could not check git branch: {e}")
    print(f"   ⚠️  {e}")

# Summary
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

if errors:
    print(f"\n❌ {len(errors)} ERRORS FOUND:")
    for error in errors:
        print(f"   - {error}")
    print("\n⚠️  THE BOT WILL NOT WORK CORRECTLY!")
    print("   Please ensure you're in the correct directory and on the right branch.")
    sys.exit(1)
elif warnings:
    print(f"\n⚠️  {len(warnings)} WARNINGS:")
    for warning in warnings:
        print(f"   - {warning}")
    print("\n✅ Core functionality appears to be working, but review warnings.")
    sys.exit(0)
else:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nThe PDT integration is properly installed and ready to use.")
    print("\nWhen you start the bot, you should see these log messages:")
    print("  - PDT_STATUS_CHECK")
    print("  - PDT_LIMIT_EXCEEDED_SWING_MODE_ACTIVATED (if PDT limit exceeded)")
    print("  - SWING_MODE_ENTRY_RECORDED (when orders are placed)")
    print("\nYou can now start the trading bot.")
    sys.exit(0)

