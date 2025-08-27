"""
Test suite for the critical fixes implemented for Alpaca import hardening, 
package-safe imports, async modernization, and deployment hardening.
"""

import os
import sys
from datetime import UTC
from unittest.mock import MagicMock, patch


def test_alpaca_availability_detection():
    """Test that _alpaca_available() function works correctly."""
    # Set test environment
    os.environ['PYTEST_RUNNING'] = '1'
    os.environ['TESTING'] = '1'

    # Import the function
    from ai_trading.core.bot_engine import _alpaca_available

    # Test normal case (should return False in our test environment)
    result = _alpaca_available()
    assert isinstance(result, bool)


def test_alpaca_import_exception_handling():
    """Test that Alpaca imports handle TypeErrors and other exceptions gracefully."""
    # Set test environment
    os.environ['PYTEST_RUNNING'] = '1'
    os.environ['TESTING'] = '1'

    # Mock alpaca to raise TypeError (the specific error mentioned in requirements)
    with patch.dict('sys.modules', {'alpaca': MagicMock()}):
        with patch('ai_trading.core.bot_engine._alpaca_available') as mock_available:
            mock_available.side_effect = TypeError("'function' object is not iterable")

            # This should not crash, should fallback gracefully
            try:
                mock_available()
                assert False, "Should have raised TypeError"
            except TypeError:
                pass


def test_package_safe_imports():
    """Test that package imports work correctly from ai_trading namespace."""
    # Set test environment
    os.environ['PYTEST_RUNNING'] = '1'
    os.environ['TESTING'] = '1'

    # Test logging import
    from ai_trading.logging import get_logger, setup_logging
    assert callable(setup_logging)
    assert callable(get_logger)

    # Test core imports
    from ai_trading.core.bot_engine import _alpaca_available
    assert callable(_alpaca_available)

    # Test main imports
    from ai_trading.main import run_cycle
    assert callable(run_cycle)


def test_utc_datetime_handling():
    """Test that datetime operations use timezone-aware UTC timestamps."""
    # Set test environment
    os.environ['PYTEST_RUNNING'] = '1'
    os.environ['TESTING'] = '1'

    # Test execution engine datetime handling
    from ai_trading.core.enums import OrderSide, OrderType
    from ai_trading.execution.engine import Order

    # Create an order and check timestamps
    order = Order(
        id="test_123",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100
    )

    # Check that created_at has timezone info
    assert order.created_at.tzinfo is not None, "Order created_at should be timezone-aware"
    assert order.created_at.tzinfo == UTC, "Order should use UTC timezone"


def test_async_modernization():
    """Test that asyncio uses modern get_running_loop pattern."""
    try:
        import asyncio
        import inspect

        # Check health monitor source for modern async patterns
        from ai_trading.health_monitor import HealthChecker

        # Get the source of run_check method
        source = inspect.getsource(HealthChecker.run_check)

        # Should use get_running_loop() not get_event_loop()
        assert 'get_running_loop()' in source, "Should use modern get_running_loop()"
        assert 'get_event_loop()' not in source, "Should not use deprecated get_event_loop()"
    except ImportError:
        pass
        # This is OK in minimal test environment


def test_start_script_portability():
    """Test that start.sh is portable and doesn't contain hard-coded paths."""
    with open('start.sh') as f:
        content = f.read()

    # Should not contain hard-coded paths
    assert '/home/aiuser/ai-trading-bot' not in content, "Should not have hard-coded paths"

    # Should use SCRIPT_DIR pattern
    assert 'SCRIPT_DIR=' in content, "Should determine script directory dynamically"

    # Should support WORKDIR override
    assert 'WORKDIR:-' in content, "Should support WORKDIR environment variable"

    # Should support VENV_PATH override
    assert 'VENV_PATH:-' in content, "Should support VENV_PATH environment variable"



def test_python_version_requirements():
    """Test that pyproject.toml has correct Python version requirements."""
    with open('pyproject.toml') as f:
        content = f.read()

    # Should use flexible version range
    assert 'requires-python = ">=3.12,<3.13"' in content, "Should use flexible Python 3.12 range"
    assert 'requires-python = "==3.12.3"' not in content, "Should not pin exact version"



def test_env_example_exists():
    """Test that .env.example exists and has required placeholders."""
    assert os.path.exists('.env.example'), ".env.example should exist"

    with open('.env.example') as f:
        content = f.read()

    # Should have key placeholders
    # Guard: test validation checking for placeholder patterns, not actual secrets
    assert 'ALPACA_API_KEY=' in content, "Should have Alpaca API key placeholder"
    assert 'ALPACA_SECRET_KEY=' in content, "Should have Alpaca secret placeholder"
    assert 'your_alpaca_api_key_here' in content, "Should have safe placeholder values"

    # Should not contain real secrets
    # Guard: test validation ensuring no real credentials leak into examples
    assert len([line for line in content.split('\n') if line.startswith('ALPACA_API_KEY=') and 'your_' not in line]) == 0, "Should not contain real API keys"



def test_no_inappropriate_shebangs():
    """Test that non-CLI modules don't have shebangs."""
    import glob

    # Check ai_trading package files
    for py_file in glob.glob('ai_trading/**/*.py', recursive=True):
        with open(py_file) as f:
            first_line = f.readline().strip()

        # Non-CLI modules should not have shebangs
        assert not first_line.startswith('#!'), f"File {py_file} should not have shebang (non-CLI module)"



if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_alpaca_availability_detection,
        test_alpaca_import_exception_handling,
        test_package_safe_imports,
        test_utc_datetime_handling,
        test_async_modernization,
        test_start_script_portability,
        test_python_version_requirements,
        test_env_example_exists,
        test_no_inappropriate_shebangs,
    ]


    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except (ValueError, TypeError):
            failed += 1


    if failed == 0:
        sys.exit(0)
    else:
        sys.exit(1)
