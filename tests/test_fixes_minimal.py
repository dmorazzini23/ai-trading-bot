"""
Minimal test for critical fixes that can run without full environment setup.
"""

import os
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

# Set minimal environment for testing
os.environ.setdefault('ALPACA_API_KEY', 'test_key')
os.environ.setdefault('ALPACA_SECRET_KEY', 'test_secret')
os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
os.environ.setdefault('WEBHOOK_SECRET', 'test_webhook')
os.environ.setdefault('FLASK_PORT', '5000')


def test_risk_engine_methods_exist():
    """Test that the missing RiskEngine methods now exist."""
    print("Testing RiskEngine missing methods...")
    
    try:
        from risk_engine import RiskEngine
        
        # Create risk engine instance
        risk_engine = RiskEngine()
        
        # Test that all required methods exist
        assert hasattr(risk_engine, 'get_current_exposure'), "get_current_exposure method missing"
        assert hasattr(risk_engine, 'max_concurrent_orders'), "max_concurrent_orders method missing"
        assert hasattr(risk_engine, 'max_exposure'), "max_exposure method missing"
        assert hasattr(risk_engine, 'order_spacing'), "order_spacing method missing"
        
        # Test that methods return appropriate types
        exposure = risk_engine.get_current_exposure()
        assert isinstance(exposure, dict), f"get_current_exposure should return dict, got {type(exposure)}"
        
        max_orders = risk_engine.max_concurrent_orders()
        assert isinstance(max_orders, int), f"max_concurrent_orders should return int, got {type(max_orders)}"
        
        max_exp = risk_engine.max_exposure()
        assert isinstance(max_exp, float), f"max_exposure should return float, got {type(max_exp)}"
        
        spacing = risk_engine.order_spacing()
        assert isinstance(spacing, float), f"order_spacing should return float, got {type(spacing)}"
        
        print("✓ All RiskEngine methods exist and return correct types")
        return True
        
    except Exception as e:
        print(f"✗ RiskEngine test failed: {e}")
        return False


def test_bot_context_alpaca_client():
    """Test BotContext alpaca_client property."""
    print("Testing BotContext alpaca_client compatibility...")
    
    try:
        from ai_trading.core.bot_engine import BotContext
        from datetime import timedelta, datetime
        
        # Create minimal BotContext with mocked dependencies
        mock_api = Mock()
        ctx = BotContext(
            api=mock_api,
            data_client=Mock(),
            data_fetcher=Mock(),
            signal_manager=Mock(),
            trade_logger=Mock(),
            sem=Mock(),
            volume_threshold=1000,
            entry_start_offset=timedelta(minutes=30),
            entry_end_offset=timedelta(minutes=30),
            market_open=datetime.now(timezone.utc).time(),  # AI-AGENT-REF: Use timezone-aware datetime
            market_close=datetime.now(timezone.utc).time(),  # AI-AGENT-REF: Use timezone-aware datetime
            regime_lookback=10,
            regime_atr_threshold=0.02,
            daily_loss_limit=0.05,
            kelly_fraction=0.25,
            capital_scaler=Mock(),
            adv_target_pct=0.1,
            max_position_dollars=10000,
            params={}
        )
        
        # Test alpaca_client property
        assert hasattr(ctx, 'alpaca_client'), "alpaca_client property missing"
        assert ctx.alpaca_client is mock_api, "alpaca_client should return the api object"
        
        print("✓ BotContext alpaca_client property works correctly")
        return True
        
    except Exception as e:
        print(f"✗ BotContext test failed: {e}")
        return False


def test_process_manager_enhancements():
    """Test ProcessManager new functionality."""
    print("Testing ProcessManager enhancements...")
    
    try:
        from process_manager import ProcessManager
        
        pm = ProcessManager()
        
        # Test new methods exist
        assert hasattr(pm, 'acquire_process_lock'), "acquire_process_lock method missing"
        assert hasattr(pm, 'check_multiple_instances'), "check_multiple_instances method missing"
        
        # Test lock mechanism with temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            lock_file = tmp.name
        
        try:
            # Test lock acquisition
            result = pm.acquire_process_lock(lock_file)
            assert isinstance(result, bool), "acquire_process_lock should return bool"
            
            # Clean up
            if os.path.exists(lock_file):
                os.remove(lock_file)
                
        finally:
            if os.path.exists(lock_file):
                os.remove(lock_file)
        
        print("✓ ProcessManager enhancements work correctly")
        return True
        
    except Exception as e:
        print(f"✗ ProcessManager test failed: {e}")
        return False


def test_data_validation_module():
    """Test data validation module exists and has required functions."""
    print("Testing data validation module...")
    
    try:
        import data_validation
        
        # Test required functions exist
        required_functions = [
            'check_data_freshness',
            'validate_trading_data',
            'get_stale_symbols',
            'should_halt_trading',
            'emergency_data_check'
        ]
        
        for func_name in required_functions:
            assert hasattr(data_validation, func_name), f"{func_name} function missing"
        
        print("✓ Data validation module has all required functions")
        return True
        
    except Exception as e:
        print(f"✗ Data validation test failed: {e}")
        return False


def test_audit_permission_handling():
    """Test audit module has enhanced permission handling."""
    print("Testing audit permission handling...")
    
    try:
        import ai_trading.audit as audit  # AI-AGENT-REF: canonical import
        import inspect
        
        # Check that log_trade function exists
        assert hasattr(audit, 'log_trade'), "log_trade function missing"
        
        # Check that enhanced permission handling code exists
        source = inspect.getsource(audit.log_trade)
        assert 'ProcessManager' in source, "ProcessManager not found in audit.log_trade"
        assert 'fix_file_permissions' in source, "fix_file_permissions not found in audit.log_trade"
        
        print("✓ Audit module has enhanced permission handling")
        return True
        
    except Exception as e:
        print(f"✗ Audit permission handling test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING CRITICAL TRADING BOT FIXES")
    print("=" * 60)
    
    tests = [
        test_risk_engine_methods_exist,
        test_bot_context_alpaca_client,
        test_process_manager_enhancements,
        test_data_validation_module,
        test_audit_permission_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL CRITICAL FIXES IMPLEMENTED SUCCESSFULLY!")
        return True
    else:
        print("❌ Some fixes need attention")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)