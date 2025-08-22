"""Test to ensure real tenacity package is imported from PyPI."""

import inspect


def test_real_tenacity_import():
    """Test that tenacity is imported from site-packages, not local mock."""
    import tenacity
    tenacity_path = inspect.getfile(tenacity)
    assert 'site-packages' in tenacity_path or '/tmp' in tenacity_path, \
        f"Using local tenacity mock! Import path: {tenacity_path}"


def test_tenacity_functionality():
    """Test that real tenacity has expected functionality."""
    from tenacity import retry, stop_after_attempt, wait_exponential

    # Test that these are callable
    assert callable(retry)
    assert callable(stop_after_attempt)
    assert callable(wait_exponential)

    # Test basic decorator functionality
    @retry(stop=stop_after_attempt(2))
    def test_func():
        return "success"

    result = test_func()
    assert result == "success"
