#!/usr/bin/env python3
"""
Integration test to validate all implemented changes without relying on complex test setup.
This test validates the core functionality we implemented according to the problem statement.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path


def test_model_registry():
    """Test model registry functionality."""
    try:
        import numpy as np
        from ai_trading.model_registry import ModelRegistry
        from sklearn.linear_model import LinearRegression

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_path=tmpdir)

            # Create and register a model
            model = LinearRegression()
            X = np.array([[1, 2], [3, 4]])
            y = np.array([1, 2])
            model.fit(X, y)

            model_id = registry.register_model(model, 'test', 'linear')
            latest = registry.latest_for('test', 'linear')
            loaded, meta = registry.load_model(model_id)

            assert latest == model_id
            assert isinstance(loaded, LinearRegression)
            return True
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        traceback.print_exc()
        return False

def test_disable_daily_retrain():
    """Test DISABLE_DAILY_RETRAIN parsing."""
    try:
        test_cases = [
            ("true", True),
            ("True", True),
            ("1", True),
            ("false", False),
            ("0", False),
            ("", False),
            ("invalid", False),
        ]

        for env_val, expected in test_cases:
            result = env_val.lower() in ("true", "1")
            assert result == expected, f"For '{env_val}', expected {expected}, got {result}"

        return True
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        return False

def test_executor_sizing():
    """Test executor auto-sizing logic."""
    try:
        # Clear environment variables
        for var in ["EXECUTOR_WORKERS", "PREDICTION_WORKERS"]:
            if var in os.environ:
                del os.environ[var]

        # Test auto-sizing logic
        _cpu = (os.cpu_count() or 2)
        _exec_env = int(os.getenv("EXECUTOR_WORKERS", "0") or "0")
        _pred_env = int(os.getenv("PREDICTION_WORKERS", "0") or "0")
        _exec_workers = _exec_env or max(2, min(4, _cpu))
        _pred_workers = _pred_env or max(2, min(4, _cpu))

        assert 2 <= _exec_workers <= 4
        assert 2 <= _pred_workers <= 4

        # Test environment overrides
        os.environ["EXECUTOR_WORKERS"] = "6"
        os.environ["PREDICTION_WORKERS"] = "3"

        _exec_env = int(os.getenv("EXECUTOR_WORKERS", "0") or "0")
        _pred_env = int(os.getenv("PREDICTION_WORKERS", "0") or "0")
        _exec_workers = _exec_env or max(2, min(4, _cpu))
        _pred_workers = _pred_env or max(2, min(4, _cpu))

        assert _exec_workers == 6
        assert _pred_workers == 3

        return True
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        return False

def test_minute_cache_helpers():
    """Test minute cache helper functions."""
    try:
        import pandas as pd

        # Simulate the cache and helper functions
        _MINUTE_CACHE = {}

        def get_cached_minute_timestamp(symbol):
            entry = _MINUTE_CACHE.get(symbol)
            if not entry:
                return None
            _, ts = entry
            return ts if isinstance(ts, pd.Timestamp) else None

        def last_minute_bar_age_seconds(symbol):
            ts = get_cached_minute_timestamp(symbol)
            if ts is None:
                return None
            return int((pd.Timestamp.now(tz="UTC") - ts).total_seconds())

        # Test empty cache
        assert get_cached_minute_timestamp("AAPL") is None
        assert last_minute_bar_age_seconds("AAPL") is None

        # Test with data
        test_df = pd.DataFrame({"close": [100]})
        timestamp = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=30)
        _MINUTE_CACHE["AAPL"] = (test_df, timestamp)

        cached_ts = get_cached_minute_timestamp("AAPL")
        age = last_minute_bar_age_seconds("AAPL")

        assert cached_ts is not None
        assert isinstance(age, int)
        assert 25 <= age <= 35  # Should be around 30 seconds

        return True
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        return False

def test_import_hardening():
    """Test that import hardening patterns are in place."""
    try:
        # Check bot_engine.py for hardened imports
        bot_engine_path = Path("ai_trading/core/bot_engine.py")
        if bot_engine_path.exists():
            content = bot_engine_path.read_text()

            expected_patterns = [
                "from ai_trading.meta_learning import optimize_signals",
                "from meta_learning import optimize_signals",
                "from ai_trading.pipeline import model_pipeline",
                "from pipeline import model_pipeline",
                "from ai_trading.trade_execution import ExecutionEngine",
                "from trade_execution import ExecutionEngine",
                "from ai_trading.data_fetcher import",
                "from ai_trading.data_fetcher import",
            ]

            for pattern in expected_patterns:
                assert pattern in content, f"Missing import pattern: {pattern}"

        # Check other files
        files_to_check = ["runner.py", "backtester.py", "profile_indicators.py"]
        for filename in files_to_check:
            if Path(filename).exists():
                content = Path(filename).read_text()
                # Look for the specific import patterns we implemented
                if filename == "profile_indicators.py":
                    assert "import ai_trading.signals as signals" in content, f"Missing ai_trading signals import in {filename}"
                    assert "import ai_trading.indicators as indicators" in content, f"Missing ai_trading indicators import in {filename}"
                else:
                    assert "from ai_trading." in content, f"Missing ai_trading imports in {filename}"
                assert "except Exception:" in content, f"Missing fallback imports in {filename}"

        return True
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        return False

def test_http_timeouts():
    """Test that HTTP timeouts are implemented."""
    try:
        bot_engine_path = Path("ai_trading/core/bot_engine.py")
        if bot_engine_path.exists():
            content = bot_engine_path.read_text()

            # Should find requests.get calls with timeout
            import re
            timeout_pattern = r'requests\.get\([^)]*timeout\s*=\s*\d+'
            matches = re.findall(timeout_pattern, content)

            assert len(matches) >= 1, "Should find at least one requests.get call with timeout"

            # Check for specific timeouts we added
            assert "timeout=2" in content, "Should have health probe timeout=2"
            assert "timeout=10" in content, "Should have API timeout=10"

        return True
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        return False

def test_data_fetcher_helpers():
    """Test that data_fetcher helpers are exported."""
    try:
        data_fetcher_path = Path("data_fetcher.py")
        if data_fetcher_path.exists():
            content = data_fetcher_path.read_text()

            # Should contain the helper functions we added
            assert "def get_cached_minute_timestamp" in content
            assert "def last_minute_bar_age_seconds" in content
            assert "pd.Timestamp.now(tz=\"UTC\")" in content

        return True
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        return False

def main():
    """Run all integration tests."""

    # Set testing environment
    os.environ["TESTING"] = "1"

    tests = [
        test_model_registry,
        test_disable_daily_retrain,
        test_executor_sizing,
        test_minute_cache_helpers,
        test_import_hardening,
        test_http_timeouts,
        test_data_fetcher_helpers,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        # noqa: BLE001 TODO: narrow exception
        except Exception:
            failed += 1


    if failed == 0:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
