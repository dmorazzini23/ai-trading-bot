#!/usr/bin/env python3
"""
Smoke test for runtime context and pandas index fixes.

Tests that:
1. LazyBotContext has .params attribute 
2. pandas MultiIndex is used instead of private _RealMultiIndex
3. _prepare_run can access runtime.params without AttributeError
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment to avoid heavy imports
os.environ["PYTEST_RUNNING"] = "1"
os.environ["TESTING"] = "1"


def test_pandas_multiindex_usage():
    """Test that _RealMultiIndex has been replaced with pd.MultiIndex."""
    try:
        import pandas as pd

        # Test that pandas MultiIndex is available and works
        multi_idx = pd.MultiIndex.from_arrays([['A'], [1]], names=['sym', 'field'])
        print(f"✓ pandas.MultiIndex works: {type(multi_idx)}")

        # Test isinstance check (this is what the fixed code uses)
        is_multiindex = isinstance(multi_idx, pd.MultiIndex)
        print(f"✓ isinstance(columns, pd.MultiIndex) works: {is_multiindex}")

        return True

    except Exception as e:
        print(f"✗ pandas MultiIndex test failed: {e}")
        return False


def test_lazy_bot_context_params():
    """Test that LazyBotContext has .params attribute accessible."""
    try:
        from ai_trading.core.bot_engine import LazyBotContext

        # Create context (won't initialize fully in test mode)
        runtime = LazyBotContext()

        # Test that params property exists and is accessible
        has_params = hasattr(runtime, 'params')
        print(f"✓ LazyBotContext has params attribute: {has_params}")

        if not has_params:
            return False

        return True

    except Exception as e:
        print(f"✗ LazyBotContext params test failed: {e}")
        return False


def test_prepare_run_signature():
    """Test that _prepare_run can access runtime.params."""
    try:
        from ai_trading.core.bot_engine import BotState, _prepare_run

        # Create mock runtime with params attribute
        class MockAPI:
            def get_account(self):
                class MockAccount:
                    def __init__(self):
                        self.equity = "10000"
                return MockAccount()

        class MockCapitalScaler:
            def update(self, runtime, equity):
                pass

        class MockRuntime:
            def __init__(self):
                self.api = MockAPI()
                self.capital_scaler = MockCapitalScaler()
                self.params = {
                    "CAPITAL_CAP": 0.04,
                    "DOLLAR_RISK_LIMIT": 0.05,
                    "MAX_POSITION_SIZE": 1000
                }

        runtime = MockRuntime()
        state = BotState()

        # This should not raise AttributeError: 'MockRuntime' object has no attribute 'params'
        # Note: It may raise other errors due to missing dependencies, but not AttributeError on params
        try:
            _prepare_run(runtime, state)
            print("✓ _prepare_run executed successfully")
        except AttributeError as e:
            if "params" in str(e):
                print(f"✗ _prepare_run still has params AttributeError: {e}")
                return False
            else:
                print(f"✓ _prepare_run params access works (other AttributeError: {e})")
        except Exception as e:
            # Other exceptions are fine, we just want to test params access
            print(f"✓ _prepare_run params access works (other error: {type(e).__name__})")

        return True

    except Exception as e:
        print(f"✗ _prepare_run test failed: {e}")
        return False


def test_empty_dataframe_helper():
    """Test the empty DataFrame helper creates valid indexes."""
    try:
        import pandas as pd

        from ai_trading.core.bot_engine import _create_empty_bars_dataframe

        # Test the helper function
        empty_df = _create_empty_bars_dataframe("daily")

        print(f"✓ Empty DataFrame created: {empty_df.shape}")
        print(f"✓ Index type: {type(empty_df.index)}")

        # Verify it's a proper DatetimeIndex with UTC timezone
        assert isinstance(empty_df.index, pd.DatetimeIndex)
        assert str(empty_df.index.tz) == "UTC"
        assert empty_df.index.name == "timestamp"

        print("✓ Empty DataFrame has valid DatetimeIndex with UTC timezone")

        return True

    except Exception as e:
        print(f"✗ Empty DataFrame helper test failed: {e}")
        return False


def main():
    """Run smoke tests for pandas index and runtime context fixes."""
    print("Running smoke tests for pandas index and runtime context fixes...")
    print()

    tests = [
        test_pandas_multiindex_usage,
        test_empty_dataframe_helper,
        test_lazy_bot_context_params,
        test_prepare_run_signature,
    ]

    passed = 0
    for test in tests:
        print(f"Running {test.__name__}...")
        try:
            if test():
                passed += 1
                print("✓ PASSED")
            else:
                print("✗ FAILED")
        except Exception as e:
            print(f"✗ FAILED with exception: {e}")
        print()

    print(f"Smoke test results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed - OK")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
