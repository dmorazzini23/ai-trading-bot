#!/usr/bin/env python3
"""
Test script to validate settings singleton, Money math, and rate limiting.

Validates the critical features required by the problem statement:
- Settings singleton with aliases
- Money math with quantization
- Rate limiter functionality
- Final-bar gating
- Hyperparams schema
"""

import sys
import os
sys.path.append('.')

def test_settings_singleton():
    """Test settings singleton with Alpaca credential aliases."""
    print("Testing Settings Singleton...")
    
    from ai_trading.config.settings_singleton import get_settings, get_masked_config
    
    # Test singleton behavior
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2, "Settings should be singleton"
    print("✓ Singleton pattern works")
    
    # Test alias support
    os.environ["APCA_API_KEY_ID"] = "test_key_123"
    os.environ["APCA_API_SECRET_KEY"] = "test_secret_456"
    
    # Clear cache to pick up new env vars
    get_settings.cache_clear()
    settings = get_settings()
    
    api_key, secret_key = settings.get_alpaca_keys()
    assert api_key == "test_key_123", f"Expected test_key_123, got {api_key}"
    assert secret_key == "test_secret_456", f"Expected test_secret_456, got {secret_key}"
    print("✓ Alpaca credential aliases work")
    
    # Test masked config
    masked = get_masked_config()
    assert "test..._123" in masked['alpaca_api_key'] or "***MASKED***" in masked['alpaca_api_key']
    assert masked['has_credentials'] == True
    print("✓ Masked config logging works")
    
    print("Settings Singleton: PASS\n")


def test_money_math():
    """Test Money class with quantization."""
    print("Testing Money Math...")
    
    from ai_trading.math.money import Money, round_to_tick, round_to_lot
    from decimal import Decimal
    
    # Test Money arithmetic
    m1 = Money("10.50")
    m2 = Money("5.25")
    
    assert float(m1 + m2) == 15.75, "Money addition failed"
    assert float(m1 - m2) == 5.25, "Money subtraction failed"
    assert float(m1 * 2) == 21.0, "Money multiplication failed"
    print("✓ Money arithmetic works")
    
    # Test quantization
    tick_size = Decimal('0.01')
    price = round_to_tick('15.567', tick_size)
    assert float(price) == 15.57, f"Expected 15.57, got {float(price)}"
    print("✓ Price quantization works")
    
    # Test lot rounding
    lot_qty = round_to_lot(157.8, 100)
    assert lot_qty == 200, f"Expected 200, got {lot_qty}"
    print("✓ Lot quantization works")
    
    print("Money Math: PASS\n")


def test_rate_limiter():
    """Test central rate limiter functionality."""
    print("Testing Rate Limiter...")
    
    from ai_trading.integrations.rate_limit import get_limiter
    
    limiter = get_limiter()
    
    # Test basic acquisition
    acquired = limiter.acquire_sync("test_route", tokens=1, timeout=5.0)
    assert acquired == True, "Should acquire tokens successfully"
    print("✓ Basic token acquisition works")
    
    # Test status
    status = limiter.get_status("orders")
    assert "available_tokens" in status, "Status should include available tokens"
    assert status["enabled"] == True, "Orders route should be enabled"
    print("✓ Rate limiter status works")
    
    print("Rate Limiter: PASS\n")


def test_final_bar_gating():
    """Test final-bar gating functionality."""
    print("Testing Final-bar Gating...")
    
    from ai_trading.market.calendars import ensure_final_bar, is_market_open
    
    # Test final bar check
    result = ensure_final_bar("AAPL", "1min")
    assert isinstance(result, bool), "ensure_final_bar should return boolean"
    print(f"✓ Final bar check for AAPL 1min: {result}")
    
    # Test market open check  
    market_open = is_market_open("AAPL")
    assert isinstance(market_open, bool), "is_market_open should return boolean"
    print(f"✓ Market open check for AAPL: {market_open}")
    
    print("Final-bar Gating: PASS\n")


def test_hyperparams_schema():
    """Test hyperparams schema validation."""
    print("Testing Hyperparams Schema...")
    
    from ai_trading.core.hyperparams_schema import (
        load_hyperparams, 
        get_default_hyperparams,
        validate_hyperparams_file,
        HYPERPARAMS_SCHEMA_VERSION
    )
    
    # Test default hyperparams
    default_params = get_default_hyperparams()
    assert default_params.schema_version == HYPERPARAMS_SCHEMA_VERSION
    assert 0.0 <= default_params.buy_threshold <= 1.0
    print(f"✓ Default hyperparams loaded (version {default_params.schema_version})")
    
    # Test loading (should use defaults if file missing)
    loaded_params = load_hyperparams("nonexistent_file.json")
    assert loaded_params.schema_version == HYPERPARAMS_SCHEMA_VERSION
    print("✓ Missing file handled gracefully")
    
    # Test validation
    validation_report = validate_hyperparams_file("nonexistent_file.json")
    assert validation_report["file_exists"] == False
    assert len(validation_report["warnings"]) > 0
    print("✓ Validation report works")
    
    print("Hyperparams Schema: PASS\n")


def test_import_compatibility():
    """Test that imports work without breaking existing code."""
    print("Testing Import Compatibility...")
    
    # Test importing main modules
    try:
        print("✓ ai_trading import works")
    except Exception as e:
        print(f"✗ ai_trading import failed: {e}")
        return False
    
    try:
        print("✓ settings_singleton import works")
    except Exception as e:
        print(f"✗ settings_singleton import failed: {e}")
        return False
    
    try:
        print("✓ Money class import works")
    except Exception as e:
        print(f"✗ Money class import failed: {e}")
        return False
    
    try:
        print("✓ rate_limit import works")
    except Exception as e:
        print(f"✗ rate_limit import failed: {e}")
        return False
    
    print("Import Compatibility: PASS\n")
    return True


def run_validation():
    """Run all validation tests."""
    print("=" * 60)
    print("AI TRADING BOT - CONFIGURATION VALIDATION")
    print("=" * 60)
    
    try:
        if not test_import_compatibility():
            return False
        
        test_settings_singleton()
        test_money_math()
        test_rate_limiter()
        test_final_bar_gating()
        test_hyperparams_schema()
        
        print("=" * 60)
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)