"""Test to ensure real tenacity package is imported from PyPI."""
import inspect

from tests.optdeps import require
from ai_trading.utils import retry as retry_utils

# Skip when tenacity isn't installed (CI smoke with SKIP_INSTALL=1)
require("tenacity")


def test_real_tenacity_import():
    """Test that tenacity is imported from site-packages, not local mock."""
    import tenacity

    tenacity_path = inspect.getfile(tenacity)
    assert "site-packages" in tenacity_path or "/tmp" in tenacity_path, (
        f"Using local tenacity mock! Import path: {tenacity_path}"
    )
    assert retry_utils.retry is tenacity.retry


def test_tenacity_functionality():
    """Test that retry utilities are callable and usable."""
    from ai_trading.utils.retry import retry, stop_after_attempt, wait_exponential

    assert callable(retry)
    assert callable(stop_after_attempt)
    assert callable(wait_exponential)

    @retry(stop=stop_after_attempt(2))
    def test_func():
        return "success"

    assert test_func() == "success"
