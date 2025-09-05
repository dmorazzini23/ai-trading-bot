import pytest

from tests.optdeps import require

from ai_trading.utils.retry import RetryError, retry

# Skip when tenacity is not installed
require("tenacity")


def test_retry_reraise_false_wraps_exception():
    @retry(retries=2, delay=0.001, reraise=False)
    def boom():
        raise ValueError("boom")

    with pytest.raises(RetryError):
        boom()


def test_retry_reraise_true_propagates_exception():
    @retry(retries=2, delay=0.001, reraise=True)
    def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        boom()
