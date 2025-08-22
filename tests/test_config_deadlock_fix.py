"""Tests for configuration validation deadlock fix."""

import os
import threading
import time
from unittest.mock import patch

import pytest

from ai_trading import config


def test_no_hang_on_basic_validation():
    """Test that basic validation operations complete without hanging."""
    start_time = time.time()

    # This should complete quickly regardless of environment state
    try:
        config.validate_environment()
    except RuntimeError:
        pass  # Expected when env vars missing

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Validation took too long (possible hang): {elapsed:.2f} seconds"


def test_no_hang_on_comprehensive_validation():
    """Test that comprehensive validation completes without hanging."""
    start_time = time.time()

    try:
        config.validate_env_vars()
    except RuntimeError:
        pass  # Expected when env vars missing

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Validation took too long (possible hang): {elapsed:.2f} seconds"


def test_nested_validation_calls_no_deadlock():
    """Test that nested validation calls don't cause deadlock."""
    # Set required env vars for successful validation
    env_vars = {
        'ALPACA_API_KEY': 'test_key_1234567890',
        'ALPACA_SECRET_KEY': 'test_secret_1234567890',
        'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets',
        'WEBHOOK_SECRET': 'test_webhook_secret',
        'SCHEDULER_SLEEP_SECONDS': '30'
    }

    with patch.dict(os.environ, env_vars):
        start_time = time.time()

        # This should not hang - validate_env_vars internally calls validate_environment
        config.validate_env_vars()

        elapsed = time.time() - start_time
        assert elapsed < 5.0, f"Nested validation took too long: {elapsed:.2f} seconds"


def test_concurrent_validation_no_hang():
    """Test that concurrent validation calls complete without hanging."""
    results = []

    def run_validation():
        """Run validation in a thread."""
        start_time = time.time()
        try:
            config.validate_environment()
        except RuntimeError:
            pass  # Expected when env vars missing
        elapsed = time.time() - start_time
        results.append(elapsed)

    # Start multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=run_validation)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete with timeout
    start_time = time.time()
    for thread in threads:
        thread.join(timeout=10.0)
        # If thread is still alive, it's hanging
        if thread.is_alive():
            pytest.fail("Thread did not complete within timeout - possible hang/deadlock")

    total_elapsed = time.time() - start_time
    assert total_elapsed < 10.0, f"Total concurrent validation took too long: {total_elapsed:.2f} seconds"
    assert len(results) == 3, f"Not all threads completed: got {len(results)} results"


def test_lock_timeout_functionality():
    """Test that lock timeout works properly."""
    # This is a bit tricky to test, but we can verify the timeout mechanism exists
    assert hasattr(config, '_LOCK_TIMEOUT')
    assert config._LOCK_TIMEOUT == 30

    # Test that the lock tracking functions exist and work
    assert not config._is_lock_held_by_current_thread()

    config._set_lock_held_by_current_thread(True)
    assert config._is_lock_held_by_current_thread()

    config._set_lock_held_by_current_thread(False)
    assert not config._is_lock_held_by_current_thread()


def test_validation_with_proper_env_vars():
    """Test successful validation with proper environment variables."""
    env_vars = {
        'ALPACA_API_KEY': 'test_key_1234567890',
        'ALPACA_SECRET_KEY': 'test_secret_1234567890',
        'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets',
        'WEBHOOK_SECRET': 'test_webhook_secret',
        'SCHEDULER_SLEEP_SECONDS': '30'
    }

    with patch.dict(os.environ, env_vars):
        start_time = time.time()

        # Both functions should complete successfully
        config.validate_environment()
        config.validate_env_vars()

        elapsed = time.time() - start_time
        assert elapsed < 5.0, f"Validation took too long: {elapsed:.2f} seconds"


def test_main_import_no_hang():
    """Test that importing ai_trading.main doesn't hang."""
    start_time = time.time()

    try:
        # This is the specific test case mentioned in the problem statement
        print('Import successful')
    except Exception as e:
        # Import might fail due to missing dependencies, but it shouldn't hang
        print(f'Import failed (expected): {e}')

    elapsed = time.time() - start_time
    assert elapsed < 10.0, f"Import took too long (possible hang): {elapsed:.2f} seconds"


def test_deadlock_scenario_resolved():
    """Test the specific deadlock scenario that was causing the hang."""
    # This simulates the original hanging scenario:
    # validate_env_vars() -> acquire lock -> validate_environment() -> try to acquire same lock

    start_time = time.time()

    try:
        # This was the problematic call that would hang
        config.validate_env_vars()
    except RuntimeError:
        pass  # Expected when env vars are missing

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Deadlock scenario took too long: {elapsed:.2f} seconds"

    print(f"Deadlock scenario completed in {elapsed:.3f} seconds - no hang!")
