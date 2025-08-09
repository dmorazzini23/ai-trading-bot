#!/usr/bin/env python3
"""
Test script to verify the duplicate logging fix.

This script tests that the logging configuration changes eliminate
duplicate log messages that were previously appearing in both
standard and JSON formats.
"""

import logging
import io
import sys
from contextlib import redirect_stderr, redirect_stdout


def test_no_duplicate_logging():
    """Test that memory optimizer doesn't produce duplicate log messages."""
    
    # Capture all logging output
    log_capture = io.StringIO()
    
    # Set up basic logging to capture output
    handler = logging.StreamHandler(log_capture)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Clear any existing handlers
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    # Test the fixed memory optimizer
    from memory_optimizer import get_memory_optimizer
    optimizer = get_memory_optimizer()
    
    # Generate a test log message
    test_message = "Test message for duplicate detection"
    optimizer.logger.info(test_message)
    
    # Get the captured log output
    log_output = log_capture.getvalue()
    
    # Count how many times our test message appears
    message_count = log_output.count(test_message)
    
    print(f"Log output captured:")
    print("=" * 50)
    print(log_output)
    print("=" * 50)
    print(f"Test message '{test_message}' appears {message_count} times")
    
    # Verify the message appears exactly once
    if message_count == 1:
        print("✅ SUCCESS: No duplicate logging detected")
        return True
    else:
        print(f"❌ FAILURE: Message appears {message_count} times (expected 1)")
        return False


def test_deprecated_logging_config():
    """Test that logging_config.py is properly deprecated."""
    
    # Capture warnings
    import warnings
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        
        # Import and use the deprecated module
        import logging_config
        logging_config.setup_logging()
        
        # Check if deprecation warning was issued
        deprecation_warnings = [w for w in warning_list 
                              if issubclass(w.category, DeprecationWarning)
                              and "logging_config.py is deprecated" in str(w.message)]
        
        if deprecation_warnings:
            print("✅ SUCCESS: Deprecation warning properly issued")
            return True
        else:
            print("❌ FAILURE: No deprecation warning found")
            return False


def main():
    """Run all logging fix tests."""
    print("Testing duplicate logging fix...")
    print()
    
    # Test 1: No duplicate logging
    print("Test 1: Checking for duplicate logging...")
    test1_passed = test_no_duplicate_logging()
    print()
    
    # Test 2: Deprecation warning
    print("Test 2: Checking deprecation warning...")
    test2_passed = test_deprecated_logging_config()
    print()
    
    # Summary
    total_tests = 2
    passed_tests = sum([test1_passed, test2_passed])
    
    print("=" * 60)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Duplicate logging issue is fixed.")
        return 0
    else:
        print("⚠️  Some tests failed. Issue may not be fully resolved.")
        return 1


if __name__ == "__main__":
    exit(main())