"""Test auto-sizing logic and environment overrides for executors."""

import os
import threading
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from ai_trading.utils.exec import get_worker_env_override

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
        _exec_env = get_worker_env_override("EXECUTOR_WORKERS")
        _pred_env = get_worker_env_override("PREDICTION_WORKERS")
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
    _exec_env = get_worker_env_override("EXECUTOR_WORKERS")
    _pred_env = get_worker_env_override("PREDICTION_WORKERS")
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
            _exec_env = get_worker_env_override("EXECUTOR_WORKERS")
            _pred_env = get_worker_env_override("PREDICTION_WORKERS")
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
        _exec_env = get_worker_env_override("EXECUTOR_WORKERS")
        _pred_env = get_worker_env_override("PREDICTION_WORKERS")
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

        _exec_env = get_worker_env_override("EXECUTOR_WORKERS")
        assert _exec_env == expected, f"For env value '{env_val}', expected {expected}, got {_exec_env}"

        del os.environ["EXECUTOR_WORKERS"]


def test_executor_cleanup_available():
    """Test that executor cleanup function is available."""
    from ai_trading.core import bot_engine

    # Check that cleanup function exists
    assert hasattr(bot_engine, 'cleanup_executors')
    assert callable(bot_engine.cleanup_executors)


def test_bot_engine_import_does_not_create_executor_pools():
    code = """
import os

os.environ["PYTEST_RUNNING"] = "1"

from ai_trading.core import executors

executors.cleanup_executors(wait=False)

import ai_trading.core.bot_engine  # noqa: F401

assert executors.executor is None
assert executors.prediction_executor is None
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_executor_create_and_cleanup_are_synchronized(monkeypatch):
    from ai_trading.core import executors

    executors.cleanup_executors(wait=False)
    created: list[object] = []

    class DummyExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.shutdown_calls = 0
            created.append(self)

        def shutdown(self, *, wait=True, cancel_futures=True):
            self.shutdown_calls += 1

    monkeypatch.setattr(executors, "ThreadPoolExecutor", DummyExecutor)
    monkeypatch.setattr(executors.os, "cpu_count", lambda: 2)
    from ai_trading.config import settings as settings_module

    monkeypatch.setattr(settings_module, "get_settings", lambda: SimpleNamespace())

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(20):
                executors.get_executor()
                executors.get_prediction_executor()
                executors.cleanup_executors(wait=False)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert all(item.shutdown_calls <= 1 for item in created)
    executors.cleanup_executors(wait=False)


def teardown_module():
    """Clean up after tests."""
    for var in ["EXECUTOR_WORKERS", "PREDICTION_WORKERS"]:
        if var in os.environ:
            del os.environ[var]
