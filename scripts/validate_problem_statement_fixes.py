#!/usr/bin/env python3
import logging

"""
Manual validation script for critical trading bot fixes.
This script validates that all fixes from the problem statement have been properly implemented.
"""

import os
import re
import sys

# Set up minimal environment
os.environ.setdefault('ALPACA_API_KEY', 'test_key')
os.environ.setdefault('ALPACA_SECRET_KEY', 'test_secret')
os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
os.environ.setdefault('WEBHOOK_SECRET', 'test_webhook_secret')
os.environ.setdefault('FLASK_PORT', '5000')

def validate_sentiment_circuit_breaker():
    """Validate Fix 2: Sentiment Circuit Breaker improvements."""
    logging.info("Fix 2: Sentiment Circuit Breaker Thresholds")
    logging.info(str("="*50))

    try:
        import ai_trading.analysis.sentiment as sentiment

        # Expected values from problem statement:
        # - Increase failure threshold from current to 15 (more tolerant)
        # - Extend recovery timeout from 300s to 1800s (30 minutes)

        expected_failures = 15
        expected_recovery = 1800

        actual_failures = sentiment.SENTIMENT_FAILURE_THRESHOLD
        actual_recovery = sentiment.SENTIMENT_RECOVERY_TIMEOUT

        logging.info(f"Failure threshold: {actual_failures} (expected: {expected_failures}) - {'✓' if actual_failures == expected_failures else '✗'}")
        logging.info(f"Recovery timeout: {actual_recovery}s (expected: {expected_recovery}s) - {'✓' if actual_recovery == expected_recovery else '✗'}")

        # Also check bot_engine.py for consistency
        with open('bot_engine.py', 'r') as f:
            content = f.read()

        bot_failures = re.search(r'SENTIMENT_FAILURE_THRESHOLD = (\d+)', content)
        bot_recovery = re.search(r'SENTIMENT_RECOVERY_TIMEOUT = (\d+)', content)

        if bot_failures and bot_recovery:
            bot_failures_val = int(bot_failures.group(1))
            bot_recovery_val = int(bot_recovery.group(1))
            logging.info(str(f"bot_engine.py consistency: failures={bot_failures_val}, recovery={bot_recovery_val} - {'✓' if bot_failures_val == expected_failures and bot_recovery_val == expected_recovery else '✗'}"))

        return actual_failures == expected_failures and actual_recovery == expected_recovery

    except Exception as e:
        logging.info(f"Error validating sentiment circuit breaker: {e}")
        return False

def validate_meta_learning():
    """Validate Fix 3: Meta-Learning System improvements."""
    logging.info("\nFix 3: Meta-Learning Minimum Trade Requirement")
    logging.info(str("="*50))

    try:
        with open('bot_engine.py', 'r') as f:
            content = f.read()

        # Expected: Reduce minimum trade requirement from 10 to 3
        pattern = r'def load_global_signal_performance\(\s*min_trades: int = (\d+)'
        match = re.search(pattern, content)

        if match:
            current_value = int(match.group(1))
            expected_value = 3
            logging.info(f"min_trades default: {current_value} (expected: {expected_value}) - {'✓' if current_value == expected_value else '✗'}")
            return current_value == expected_value
        else:
            logging.info("✗ Could not find min_trades parameter")
            return False

    except Exception as e:
        logging.info(f"Error validating meta-learning: {e}")
        return False

def validate_pltr_sector():
    """Validate Fix 5: Sector Classification for PLTR."""
    logging.info("\nFix 5: PLTR Sector Classification")
    logging.info(str("="*50))

    try:
        with open('bot_engine.py', 'r') as f:
            content = f.read()

        # Expected: Add PLTR to Technology sector mapping
        if '"PLTR": "Technology"' in content:
            logging.info("PLTR sector mapping: Technology ✓")
            return True
        else:
            logging.info("✗ PLTR not found in Technology sector mapping")
            return False

    except Exception as e:
        logging.info(f"Error validating PLTR sector: {e}")
        return False

def validate_execution_optimizations():
    """Validate Fix 4: Order Execution Optimizations."""
    logging.info("\nFix 4: Order Execution Optimizations")
    logging.info(str("="*50))

    try:
        with open('trade_execution.py', 'r') as f:
            content = f.read()

        optimizations = {
            'Pre-validation function': '_pre_validate_order' in content,
            'Market hours optimization': '_is_market_open' in content,
            'Validation caching': '_VALIDATION_CACHE' in content,
            'Market status caching': '_MARKET_STATUS_CACHE' in content,
            'Order validation integration': 'ORDER_VALIDATION_FAILED' in content,
            'Async validation integration': 'Pre-validate order to reduce execution latency (async version)' in content
        }

        all_implemented = True
        for feature, implemented in optimizations.items():
            logging.info(str(f"{feature}: {'✓' if implemented else '✗'}"))
            if not implemented:
                all_implemented = False

        return all_implemented

    except Exception as e:
        logging.info(f"Error validating execution optimizations: {e}")
        return False

def validate_quantity_tracking():
    """Validate Fix 1: Order Quantity Tracking improvements."""
    logging.info("\nFix 1: Order Quantity Tracking")
    logging.info(str("="*50))

    try:
        with open('trade_execution.py', 'r') as f:
            content = f.read()

        # Check for clear field names in logging
        tracking_features = {
            'FULL_FILL_SUCCESS includes requested_qty': '"requested_qty": requested_qty' in content,
            'FULL_FILL_SUCCESS includes filled_qty': '"filled_qty": filled_qty' in content,
            'ORDER_FILL_CONSOLIDATED uses total_filled_qty': '"total_filled_qty": buf["qty"]' in content,
            'Clear field name documentation': 'AI-AGENT-REF: Clarify this is the total filled quantity' in content
        }

        all_implemented = True
        for feature, implemented in tracking_features.items():
            logging.info(str(f"{feature}: {'✓' if implemented else '✗'}"))
            if not implemented:
                all_implemented = False

        return all_implemented

    except Exception as e:
        logging.info(f"Error validating quantity tracking: {e}")
        return False

def main():
    """Run all validation checks."""
    logging.info("Critical Trading Bot Fixes - Manual Validation")
    logging.info(str("="*60))
    logging.info("Validating implementation of fixes from problem statement...")
    print()

    results = []

    # Run all validations
    results.append(('Sentiment Circuit Breaker', validate_sentiment_circuit_breaker()))
    results.append(('Meta-Learning System', validate_meta_learning()))
    results.append(('PLTR Sector Classification', validate_pltr_sector()))
    results.append(('Execution Optimizations', validate_execution_optimizations()))
    results.append(('Quantity Tracking', validate_quantity_tracking()))

    # Summary
    logging.info(str("\n" + "="*60))
    logging.info("VALIDATION SUMMARY")
    logging.info(str("="*60))

    all_passed = True
    for fix_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logging.info(f"{fix_name:<30} {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        logging.info("🎉 ALL FIXES VALIDATED SUCCESSFULLY!")
        logging.info("The trading bot critical issues have been resolved according to the problem statement.")
    else:
        logging.info("❌ SOME FIXES FAILED VALIDATION")
        logging.info("Please review the failed items above.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
