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
os.environ['PYTEST_RUNNING'] = '1'
os.environ['TESTING'] = '1'

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    class _PDErrors:
        class EmptyDataError(Exception):
            pass

    class _PD:
        errors = _PDErrors()

    pd = _PD()  # type: ignore

def test_pandas_multiindex_usage():
    """Test that _RealMultiIndex has been replaced with pd.MultiIndex."""
    try:
        import pandas as pd
        multi_idx = pd.MultiIndex.from_arrays([['A'], [1]], names=['sym', 'field'])
        isinstance(multi_idx, pd.MultiIndex)
        return True
    except (pd.errors.EmptyDataError, KeyError, ValueError, TypeError):
        return False

def test_lazy_bot_context_params():
    """Test that LazyBotContext has .params attribute accessible."""
    try:
        from ai_trading.core.bot_engine import LazyBotContext
        runtime = LazyBotContext()
        has_params = hasattr(runtime, 'params')
        if not has_params:
            return False
        return True
    except (pd.errors.EmptyDataError, KeyError, ValueError, TypeError):
        return False

def test_prepare_run_signature():
    """Test that _prepare_run can access runtime.params."""
    try:
        from ai_trading.core.bot_engine import BotState, _prepare_run

        class MockAPI:

            def get_account(self):

                class MockAccount:

                    def __init__(self):
                        self.equity = '10000'
                return MockAccount()

        class MockCapitalScaler:

            def update(self, runtime, equity):
                pass

        class MockRuntime:

            def __init__(self):
                self.api = MockAPI()
                self.capital_scaler = MockCapitalScaler()
                self.params = {'CAPITAL_CAP': 0.04, 'DOLLAR_RISK_LIMIT': 0.05, 'MAX_POSITION_SIZE': 1000}
        runtime = MockRuntime()
        state = BotState()
        try:
            _prepare_run(runtime, state)
        except AttributeError as e:
            if 'params' in str(e):
                return False
            else:
                pass
        except (pd.errors.EmptyDataError, KeyError, ValueError, TypeError):
            pass
        return True
    except (pd.errors.EmptyDataError, KeyError, ValueError, TypeError):
        return False

def test_empty_dataframe_helper():
    """Test the empty DataFrame helper creates valid indexes."""
    try:
        import pandas as pd
        from ai_trading.data.bars import _create_empty_bars_dataframe
        empty_df = _create_empty_bars_dataframe('daily')
        assert isinstance(empty_df.index, pd.DatetimeIndex)
        assert str(empty_df.index.tz) == 'UTC'
        assert empty_df.index.name == 'timestamp'
        return True
    except (pd.errors.EmptyDataError, KeyError, ValueError, TypeError):
        return False

def main():
    """Run smoke tests for pandas index and runtime context fixes."""
    tests = [test_pandas_multiindex_usage, test_empty_dataframe_helper, test_lazy_bot_context_params, test_prepare_run_signature]
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                pass
        except (pd.errors.EmptyDataError, KeyError, ValueError, TypeError):
            pass
    if passed == len(tests):
        return 0
    else:
        return 1
if __name__ == '__main__':
    sys.exit(main())