"""
Test centralized logging system to ensure no duplicate logging setup.
"""
import logging
import os
import threading
from unittest.mock import patch

# Mock dependencies if needed for testing
try:
    from ai_trading.logging import (
        _LOGGING_CONFIGURED,
        _LOGGING_LOCK,
        get_logger,
        setup_logging,
        validate_logging_setup,
    )
    CENTRALIZED_LOGGING_AVAILABLE = True
except ImportError:
    CENTRALIZED_LOGGING_AVAILABLE = False


def test_centralized_logging_prevents_duplicates():
    """Test that setup_logging prevents duplicate handler creation."""
    if not CENTRALIZED_LOGGING_AVAILABLE:
        return  # Skip if module not available

    # Reset logging state
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()
    root_logger.handlers.clear()

    try:
        # Mock environment variables for testing
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test',
            'ALPACA_SECRET_KEY': 'test',
            'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets'
        }):
            # Reset global state
            import ai_trading.logging as logging_module
            with logging_module._LOGGING_LOCK:
                logging_module._LOGGING_CONFIGURED = False
                logging_module._configured = False

            # First setup
            logger1 = setup_logging(debug=True)
            handlers_after_first = len(root_logger.handlers)

            # Second setup (should not add more handlers)
            logger2 = setup_logging(debug=True)
            handlers_after_second = len(root_logger.handlers)

            # Third setup with different params (should still not add handlers)
            logger3 = setup_logging(debug=False)
            handlers_after_third = len(root_logger.handlers)

            # Validate results
            assert handlers_after_first <= 2, f"Too many handlers after first setup: {handlers_after_first}"
            assert handlers_after_first == handlers_after_second, "Handler count changed on second setup"
            assert handlers_after_second == handlers_after_third, "Handler count changed on third setup"
            assert logger1 is logger2 is logger3, "Different logger instances returned"

            # Validate logging setup
            validation_result = validate_logging_setup()
            assert validation_result['validation_passed'], f"Validation failed: {validation_result['issues']}"

    finally:
        # Restore original handlers
        root_logger.handlers = original_handlers


def test_deprecated_modules_removed():
    """Test that deprecated logging modules can no longer be imported."""

    # Test that logging_config cannot be imported
    try:
        import logging_config
        assert False, "logging_config should not be importable after removal"
    except ImportError:
        pass  # Expected

    # Test that logger cannot be imported
    try:
        import logger
        assert False, "logger should not be importable after removal"
    except ImportError:
        pass  # Expected


def test_centralized_logging_thread_safety():
    """Test that centralized logging setup is thread-safe."""
    if not CENTRALIZED_LOGGING_AVAILABLE:
        return  # Skip if module not available

    # Reset logging state
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()
    root_logger.handlers.clear()

    results = []
    exceptions = []

    def setup_in_thread():
        try:
            with patch.dict(os.environ, {
                'ALPACA_API_KEY': 'test',
                'ALPACA_SECRET_KEY': 'test',
                'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets'
            }):
                setup_logging(debug=True)
                results.append(len(logging.getLogger().handlers))
        # noqa: BLE001 TODO: narrow exception
        except Exception as e:
            exceptions.append(e)

    try:
        # Reset global state
        import ai_trading.logging as logging_module
        with logging_module._LOGGING_LOCK:
            logging_module._LOGGING_CONFIGURED = False
            logging_module._configured = False

        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=setup_in_thread)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no exceptions occurred
        assert not exceptions, f"Exceptions in threads: {exceptions}"

        # Verify all threads saw the same handler count
        assert len(set(results)) == 1, f"Different handler counts: {results}"

        # Verify reasonable handler count
        assert results[0] <= 2, f"Too many handlers: {results[0]}"

    finally:
        # Restore original handlers
        root_logger.handlers = original_handlers


if __name__ == "__main__":
    test_centralized_logging_prevents_duplicates()
    test_deprecated_modules_removed()
    test_centralized_logging_thread_safety()
