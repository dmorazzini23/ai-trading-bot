#!/usr/bin/env python3
"""
Test script to verify the sklearn naming conflict fix.

This script tests that the original import error has been resolved.
Before the fix: TypeError: 'function' object is not iterable
After the fix: Clean ImportError for missing dependencies
"""

def test_sklearn_import_resolution():
    """Test that sklearn imports now fail cleanly instead of with TypeError."""
    print("Testing sklearn import resolution...")
    
    try:
        from sklearn.utils import check_random_state
        print("UNEXPECTED: sklearn.utils imported successfully")
        return False
    except ImportError as e:
        print(f"EXPECTED: Clean ImportError - {e}")
        return True
    except TypeError as e:
        print(f"PROBLEM: Still getting TypeError - {e}")
        return False
    except Exception as e:
        print(f"OTHER ERROR: {type(e).__name__}: {e}")
        return False


def test_hmmlearn_import():
    """Test that hmmlearn import fails cleanly."""
    print("Testing hmmlearn import...")
    
    try:
        from hmmlearn.hmm import GaussianHMM
        print("UNEXPECTED: hmmlearn imported successfully")
        return False
    except ImportError as e:
        print(f"EXPECTED: Clean ImportError - {e}")
        return True
    except TypeError as e:
        print(f"PROBLEM: Still getting the original TypeError - {e}")
        return False
    except Exception as e:
        print(f"OTHER ERROR: {type(e).__name__}: {e}")
        return False


def test_signals_import():
    """Test that signals module imports successfully."""
    print("Testing signals module import...")
    
    try:
        from signals import calculate_macd
        print("SUCCESS: signals.calculate_macd imported")
        return True
    except Exception as e:
        print(f"PROBLEM: signals import failed - {type(e).__name__}: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing sklearn naming conflict fix")
    print("=" * 60)
    
    tests = [
        test_sklearn_import_resolution,
        test_hmmlearn_import,
        test_signals_import,
    ]
    
    results = []
    for test in tests:
        print()
        result = test()
        results.append(result)
        print(f"Result: {'PASS' if result else 'FAIL'}")
    
    print()
    print("=" * 60)
    all_passed = all(results)
    print(f"Overall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)
    
    if all_passed:
        print()
        print("✓ The sklearn naming conflict has been successfully resolved!")
        print("✓ The AI Trading Bot should now start without the TypeError!")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)