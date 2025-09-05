import pytest

from ai_trading.utils.retry import RetryError, retry


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


def test_retry_reraise_forwards_kwargs():
    captured: dict[str, object] = {}

    @retry(retries=1, delay=0.001, reraise=True)
    def boom(**kw):  # type: ignore[no-untyped-def]
        captured.update(kw)
        raise ValueError("boom")

    with pytest.raises(ValueError):
        boom(reraise="call", extra=123)

    assert captured["reraise"] == "call"
    assert captured["extra"] == 123
