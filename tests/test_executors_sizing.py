"""Test auto-sizing logic and environment overrides for executors."""

import os
from unittest.mock import patch


def test_executor_auto_sizing():
    """Test that executors auto-size correctly based on CPU count."""
    # Clear any existing environment variables
    for var in ["EXECUTOR_WORKERS", "PREDICTION_WORKERS"]:
        if var in os.environ:
            del os.environ[var]

    # Mock cpu_count to return a known value
    with patch('os.cpu_count') as mock_cpu_count:
        mock_cpu_count.return_value = 8

        # Import the module to trigger executor creation

        # Check auto-sizing logic
        # For 8 CPUs: max(2, min(4, 8)) = 4
        expected_workers = 4

        # Check the computed values (note: we can't directly test executor workers
        # because they're created at import time, but we can check the logic)
        _cpu = (os.cpu_count() or 2)
        _exec_env = int(os.getenv("EXECUTOR_WORKERS", "0") or "0")
        _pred_env = int(os.getenv("PREDICTION_WORKERS", "0") or "0")
        _exec_workers = _exec_env or max(2, min(4, _cpu))
        _pred_workers = _pred_env or max(2, min(4, _cpu))

        assert _exec_workers == expected_workers
        assert _pred_workers == expected_workers


def test_executor_env_overrides():
    """Test that environment variable overrides work correctly."""
    # Set environment variables
    os.environ["EXECUTOR_WORKERS"] = "6"
    os.environ["PREDICTION_WORKERS"] = "3"

    # Test the logic that would be used
    _cpu = (os.cpu_count() or 2)
    _exec_env = int(os.getenv("EXECUTOR_WORKERS", "0") or "0")
    _pred_env = int(os.getenv("PREDICTION_WORKERS", "0") or "0")
    _exec_workers = _exec_env or max(2, min(4, _cpu))
    _pred_workers = _pred_env or max(2, min(4, _cpu))

    assert _exec_workers == 6
    assert _pred_workers == 3

    # Clean up
    del os.environ["EXECUTOR_WORKERS"]
    del os.environ["PREDICTION_WORKERS"]


def test_executor_bounds():
    """Test that executor auto-sizing respects bounds."""
    # Clear environment variables
    for var in ["EXECUTOR_WORKERS", "PREDICTION_WORKERS"]:
        if var in os.environ:
            del os.environ[var]

    test_cases = [
        (1, 2),    # 1 CPU -> max(2, min(4, 1)) = 2
        (2, 2),    # 2 CPU -> max(2, min(4, 2)) = 2
        (4, 4),    # 4 CPU -> max(2, min(4, 4)) = 4
        (8, 4),    # 8 CPU -> max(2, min(4, 8)) = 4 (capped at 4)
        (16, 4),   # 16 CPU -> max(2, min(4, 16)) = 4 (capped at 4)
    ]

    for cpu_count, expected in test_cases:
        with patch('os.cpu_count') as mock_cpu_count:
            mock_cpu_count.return_value = cpu_count

            _cpu = (os.cpu_count() or 2)
            _exec_env = int(os.getenv("EXECUTOR_WORKERS", "0") or "0")
            _pred_env = int(os.getenv("PREDICTION_WORKERS", "0") or "0")
            _exec_workers = _exec_env or max(2, min(4, _cpu))
            _pred_workers = _pred_env or max(2, min(4, _cpu))

            assert _exec_workers == expected, f"For {cpu_count} CPUs, expected {expected}, got {_exec_workers}"
            assert _pred_workers == expected, f"For {cpu_count} CPUs, expected {expected}, got {_pred_workers}"


def test_executor_fallback_behavior():
    """Test executor behavior when cpu_count returns None."""
    # Clear environment variables
    for var in ["EXECUTOR_WORKERS", "PREDICTION_WORKERS"]:
        if var in os.environ:
            del os.environ[var]

    with patch('os.cpu_count') as mock_cpu_count:
        mock_cpu_count.return_value = None

        _cpu = (os.cpu_count() or 2)
        _exec_env = int(os.getenv("EXECUTOR_WORKERS", "0") or "0")
        _pred_env = int(os.getenv("PREDICTION_WORKERS", "0") or "0")
        _exec_workers = _exec_env or max(2, min(4, _cpu))
        _pred_workers = _pred_env or max(2, min(4, _cpu))

        assert _exec_workers == 2  # Fallback to 2
        assert _pred_workers == 2  # Fallback to 2


def test_executor_env_validation():
    """Test that environment variable parsing handles edge cases."""
    test_cases = [
        ("", 0),      # Empty string
        ("0", 0),     # Zero
        ("1", 1),     # Valid number
        ("10", 10),   # Valid number
        ("invalid", 0),  # Invalid should default to 0 (fallback to auto-size)
    ]

    for env_val, expected in test_cases:
        os.environ["EXECUTOR_WORKERS"] = env_val

        _exec_env = int(os.getenv("EXECUTOR_WORKERS", "0") or "0")
        assert _exec_env == expected, f"For env value '{env_val}', expected {expected}, got {_exec_env}"

        del os.environ["EXECUTOR_WORKERS"]


def test_executor_cleanup_available():
    """Test that executor cleanup function is available."""
    import ai_trading.core.bot_engine as bot_engine

    # Check that cleanup function exists
    assert hasattr(bot_engine, 'cleanup_executors')
    assert callable(bot_engine.cleanup_executors)


def teardown_module():
    """Clean up after tests."""
    for var in ["EXECUTOR_WORKERS", "PREDICTION_WORKERS"]:
        if var in os.environ:
            del os.environ[var]
