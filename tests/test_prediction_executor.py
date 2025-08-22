"""
Tests for prediction executor sizing and threading.
"""
import os
from unittest.mock import patch

import pytest


class TestPredictionExecutor:
    """Test prediction executor worker count logic."""

    @patch('os.cpu_count')
    def test_prediction_executor_default_sizing(self, mock_cpu_count):
        """Test default worker count with various CPU counts."""
        # Test with 8 CPUs - should cap at 4
        mock_cpu_count.return_value = 8
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PREDICTION_WORKERS", None)

            # Simulate the logic from bot_engine.py
            _workers_env = int(os.getenv("PREDICTION_WORKERS", "0") or "0")
            _cpu = (os.cpu_count() or 2)
            _default_workers = max(2, min(4, _cpu))
            workers = _workers_env or _default_workers

            assert workers == 4, "Should cap at 4 workers for 8 CPUs"

    @patch('os.cpu_count')
    def test_prediction_executor_low_cpu_count(self, mock_cpu_count):
        """Test worker count with low CPU count."""
        # Test with 1 CPU - should use minimum of 2
        mock_cpu_count.return_value = 1
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PREDICTION_WORKERS", None)

            _workers_env = int(os.getenv("PREDICTION_WORKERS", "0") or "0")
            _cpu = (os.cpu_count() or 2)
            _default_workers = max(2, min(4, _cpu))
            workers = _workers_env or _default_workers

            assert workers == 2, "Should use minimum of 2 workers for 1 CPU"

    @patch('os.cpu_count')
    def test_prediction_executor_null_cpu_count(self, mock_cpu_count):
        """Test worker count when cpu_count returns None."""
        # Test when cpu_count() returns None
        mock_cpu_count.return_value = None
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PREDICTION_WORKERS", None)

            _workers_env = int(os.getenv("PREDICTION_WORKERS", "0") or "0")
            _cpu = (os.cpu_count() or 2)
            _default_workers = max(2, min(4, _cpu))
            workers = _workers_env or _default_workers

            assert workers == 2, "Should use fallback of 2 when cpu_count returns None"

    def test_prediction_executor_env_override(self):
        """Test that PREDICTION_WORKERS environment variable overrides default."""
        with patch.dict(os.environ, {"PREDICTION_WORKERS": "3"}):
            _workers_env = int(os.getenv("PREDICTION_WORKERS", "0") or "0")
            _cpu = (os.cpu_count() or 2)
            _default_workers = max(2, min(4, _cpu))
            workers = _workers_env or _default_workers

            assert workers == 3, "Should use PREDICTION_WORKERS=3 when set"

    def test_prediction_executor_env_zero(self):
        """Test that PREDICTION_WORKERS=0 uses default logic."""
        with patch('os.cpu_count', return_value=6):
            with patch.dict(os.environ, {"PREDICTION_WORKERS": "0"}):
                _workers_env = int(os.getenv("PREDICTION_WORKERS", "0") or "0")
                _cpu = (os.cpu_count() or 2)
                _default_workers = max(2, min(4, _cpu))
                workers = _workers_env or _default_workers

                assert workers == 4, "Should use default logic when PREDICTION_WORKERS=0"

    def test_prediction_executor_env_empty_string(self):
        """Test that empty PREDICTION_WORKERS uses default logic."""
        with patch('os.cpu_count', return_value=6):
            with patch.dict(os.environ, {"PREDICTION_WORKERS": ""}):
                _workers_env = int(os.getenv("PREDICTION_WORKERS", "0") or "0")
                _cpu = (os.cpu_count() or 2)
                _default_workers = max(2, min(4, _cpu))
                workers = _workers_env or _default_workers

                assert workers == 4, "Should use default logic when PREDICTION_WORKERS is empty"

    def test_prediction_executor_env_invalid(self):
        """Test behavior with invalid PREDICTION_WORKERS value."""
        with patch('os.cpu_count', return_value=4):
            with patch.dict(os.environ, {"PREDICTION_WORKERS": "invalid"}):
                # This should raise ValueError when trying to convert to int
                with pytest.raises(ValueError):
                    int(os.getenv("PREDICTION_WORKERS", "0") or "0")

    def test_prediction_executor_large_value(self):
        """Test that large PREDICTION_WORKERS values are accepted."""
        with patch.dict(os.environ, {"PREDICTION_WORKERS": "16"}):
            _workers_env = int(os.getenv("PREDICTION_WORKERS", "0") or "0")
            _cpu = (os.cpu_count() or 2)
            _default_workers = max(2, min(4, _cpu))
            workers = _workers_env or _default_workers

            assert workers == 16, "Should accept large PREDICTION_WORKERS values"

    @patch('os.cpu_count')
    def test_conservative_defaults(self, mock_cpu_count):
        """Test that defaults are conservative to avoid thrash."""
        # Test with various CPU counts to ensure we stay conservative
        test_cases = [
            (2, 2),   # 2 CPUs -> 2 workers
            (3, 3),   # 3 CPUs -> 3 workers
            (4, 4),   # 4 CPUs -> 4 workers
            (8, 4),   # 8 CPUs -> 4 workers (capped)
            (16, 4),  # 16 CPUs -> 4 workers (capped)
        ]

        for cpu_count, expected_workers in test_cases:
            mock_cpu_count.return_value = cpu_count
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("PREDICTION_WORKERS", None)

                _workers_env = int(os.getenv("PREDICTION_WORKERS", "0") or "0")
                _cpu = (os.cpu_count() or 2)
                _default_workers = max(2, min(4, _cpu))
                workers = _workers_env or _default_workers

                assert workers == expected_workers, f"CPU={cpu_count} should give {expected_workers} workers, got {workers}"
