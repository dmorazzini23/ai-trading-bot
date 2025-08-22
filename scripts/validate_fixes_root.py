#!/usr/bin/env python3
"""Validation script for the ellipsis and risk exposure fixes."""

import json
import logging
import sys

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/ai-trading-bot/ai-trading-bot')

def test_json_dumps_ensure_ascii():
    """Test that json.dumps with ensure_ascii=False preserves Unicode."""
    test_data = {
        "msg": "MARKET WATCH — Real Alpaca Trading SDK imported successfully",
        "symbol": "AAPL",
        "note": "Unicode ellipsis: …"
    }

    # Test with ensure_ascii=True (old behavior)
    ascii_json = json.dumps(test_data, ensure_ascii=True)

    # Test with ensure_ascii=False (new behavior)
    unicode_json = json.dumps(test_data, ensure_ascii=False)

    # Verify differences
    assert "\\u2014" in ascii_json, "Expected escaped Unicode in ensure_ascii=True"
    assert "—" in unicode_json, "Expected actual Unicode character in ensure_ascii=False"
    assert "\\u2026" in ascii_json, "Expected escaped ellipsis in ensure_ascii=True"
    assert "…" in unicode_json, "Expected actual ellipsis character in ensure_ascii=False"

    return True

def test_compilation():
    """Test that the modified files compile correctly."""
    import compileall

    files_to_check = [
        '/home/runner/work/ai-trading-bot/ai-trading-bot/ai_trading/logging.py',
        '/home/runner/work/ai-trading-bot/ai-trading-bot/ai_trading/core/bot_engine.py'
    ]

    for file_path in files_to_check:
        if not compileall.compile_file(file_path, quiet=True):
            return False
        else:
            pass

    return True

def test_logging_formatter_imports():
    """Test that logging module imports and JSONFormatter can be accessed."""
    try:
        # Test basic import
        import ai_trading.logging as logger_module

        # Test that JSONFormatter exists
        formatter = logger_module.JSONFormatter("%(asctime)sZ")

        # Create a test log record
        rec = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Test message with Unicode — characters",
            args=None,
            exc_info=None,
        )

        # Test formatting
        output = formatter.format(rec)
        data = json.loads(output)

        # Verify the message is preserved
        assert "msg" in data
        assert "Unicode — characters" in data["msg"]

        return True

    except Exception:
        return False

def validate_bot_engine_functions():
    """Validate that the new functions exist in bot_engine without importing."""
    bot_engine_path = '/home/runner/work/ai-trading-bot/ai-trading-bot/ai_trading/core/bot_engine.py'

    with open(bot_engine_path) as f:
        content = f.read()

    # Check for the new functions
    required_functions = [
        '_get_runtime_context_or_none',
        '_update_risk_engine_exposure'
    ]

    for func_name in required_functions:
        if f"def {func_name}(" in content:
            pass
        else:
            return False

    # Check for the scheduled task
    if 'target=_update_risk_engine_exposure' in content:
        pass
    else:
        return False

    return True

def validate_logging_changes():
    """Validate that logging.py has the ensure_ascii=False changes."""
    logging_path = '/home/runner/work/ai-trading-bot/ai-trading-bot/ai_trading/logging.py'

    with open(logging_path) as f:
        content = f.read()

    # Check for ensure_ascii=False in json.dumps calls
    ensure_ascii_false_count = content.count('ensure_ascii=False')

    if ensure_ascii_false_count >= 2:
        return True
    else:
        return False

def main():
    """Run all validation tests."""

    tests = [
        ("JSON Unicode handling", test_json_dumps_ensure_ascii),
        ("File compilation", test_compilation),
        ("Logging changes", validate_logging_changes),
        ("Bot engine functions", validate_bot_engine_functions),
        ("Logging formatter", test_logging_formatter_imports),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception:
            pass


    if passed == total:
        return 0
    else:
        return 1

if __name__ == '__main__':
    sys.exit(main())
