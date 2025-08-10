#!/usr/bin/env python3
import logging

"""
Validation script for runtime hardening changes.
This script validates that all the key changes are working properly.
"""

import pathlib
import re
import sys
from datetime import UTC


def test_utc_helper():
    """Test UTC timestamp helper functionality."""
    logging.info("Testing UTC timestamp helper...")
    sys.path.insert(0, 'ai_trading/utils')
    from timefmt import format_datetime_utc, utc_now_iso

    # Test utc_now_iso
    timestamp = utc_now_iso()
    assert timestamp.endswith('Z'), "UTC timestamp should end with Z"
    assert timestamp.count('Z') == 1, "UTC timestamp should have exactly one Z"

    # Test format_datetime_utc
    from datetime import datetime
    dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    formatted = format_datetime_utc(dt)
    assert formatted == "2024-01-01T12:00:00Z", f"Expected '2024-01-01T12:00:00Z', got '{formatted}'"

    logging.info("✓ UTC timestamp helper working correctly")

def test_http_timeouts():
    """Test that all HTTP requests have timeout parameters."""
    logging.info("Testing HTTP timeout enforcement...")

    root = pathlib.Path('.').resolve()
    offenders = []
    important_files = []

    for p in root.rglob('*.py'):
        if 'backup' in str(p) or 'original' in str(p) or '__pycache__' in str(p):
            continue
        txt = p.read_text(encoding='utf-8', errors='ignore')
        for m in re.finditer(r'requests\.(get|post|put|delete|patch)\s*\(', txt):
            # Find the complete function call by matching parentheses
            start = m.start()
            i = start + len(m.group(0)) - 1  # Position of opening parenthesis
            paren_count = 1
            j = i + 1
            while j < len(txt) and paren_count > 0:
                if txt[j] == '(':
                    paren_count += 1
                elif txt[j] == ')':
                    paren_count -= 1
                j += 1
            # Check if timeout appears in the full function call
            full_call = txt[start:j]
            if 'timeout=' not in full_call:
                line_no = txt[:start].count('\n') + 1
                first_line = full_call.split('\n')[0]
                offenders.append(f'{p}:{line_no}:{first_line.strip()[:100]}')
                important_files.append(str(p))

    if offenders:
        logging.info(f"✗ Found {len(offenders)} requests without timeout:")
        for o in offenders:
            logging.info(f"  {o}")
        return False
    else:
        logging.info("✓ All HTTP requests have timeout parameters")
        return True

def test_package_imports():
    """Test that package-safe import patterns exist."""
    logging.info("Testing package-safe import patterns...")

    # Check key files for package-safe import patterns
    files_to_check = [
        'ai_trading/core/bot_engine.py',
        'ai_trading/runner.py'
    ]

    found_patterns = 0
    for filepath in files_to_check:
        if not pathlib.Path(filepath).exists():
            continue

        with open(filepath) as f:
            content = f.read()

        # Look for package-safe import patterns
        if "from ai_trading." in content and "except Exception:" in content:
            found_patterns += 1

    if found_patterns > 0:
        logging.info(f"✓ Found package-safe import patterns in {found_patterns} files")
        return True
    else:
        logging.info("✗ No package-safe import patterns found")
        return False

def test_lazy_loading():
    """Test that lazy loading patterns exist."""
    logging.info("Testing lazy loading patterns...")

    # Check for lazy loading in base.py
    base_file = 'ai_trading/utils/base.py'
    if pathlib.Path(base_file).exists():
        with open(base_file) as f:
            content = f.read()

        if '_get_alpaca_rest' in content and 'REST = None' in content:
            logging.info("✓ Lazy loading pattern found for Alpaca SDK")
            return True

    logging.info("✗ Lazy loading pattern not found")
    return False

def test_version_warning():
    """Test that Python version warning is relaxed."""
    logging.info("Testing Python version warning...")

    # Check bot_engine.py for version check
    bot_engine_file = 'ai_trading/core/bot_engine.py'
    if pathlib.Path(bot_engine_file).exists():
        with open(bot_engine_file) as f:
            content = f.read()

        if "sys.version_info < (3, 10)" in content:
            logging.info("✓ Python version warning relaxed to accept 3.10+")
            return True
        elif "sys.version_info < (3, 12" in content:
            logging.info("✗ Python version warning still too restrictive")
            return False

    logging.info("✗ Python version check not found")
    return False

def main():
    """Run all validation tests."""
    logging.info("=== Runtime Hardening Validation ===\n")

    tests = [
        test_utc_helper,
        test_http_timeouts,
        test_package_imports,
        test_lazy_loading,
        test_version_warning
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
            logging.info(f"✗ {test.__name__} failed with error: {e}")
            failed += 1
        print()

    logging.info("=== Summary ===")
    logging.info(f"Passed: {passed}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Total:  {len(tests)}")

    if failed == 0:
        logging.info("\n🎉 All validation tests passed!")
        return 0
    else:
        logging.info(f"\n⚠️  {failed} validation tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
