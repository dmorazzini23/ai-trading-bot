from ai_trading.utils.retry import RetryError, retry
from ai_trading.utils.retry_mode import retry_mode


def test_retry_mode_fixed_and_linear():
    calls = {"fixed": 0, "linear": 0}

    @retry(retries=2, delay=0.001, backoff=1.0, mode="fixed")
    def fixed():
        calls["fixed"] += 1
        raise RuntimeError

    @retry(retries=3, delay=0.001, backoff=0.001, mode="linear")
    def linear():
        calls["linear"] += 1
        raise RuntimeError

    for fn in (fixed, linear):
        try:
            fn()
        except Exception:
            pass

    assert calls["fixed"] == 2
    assert calls["linear"] == 3


def test_retry_mode_returns_fallback_without_retryerror():
    calls = {"fail": 0}

    @retry_mode(retries=2, delay=0.001, fallback="ok", exceptions=(RuntimeError,))
    def always_fail():
        calls["fail"] += 1
        raise RuntimeError

    try:
        result = always_fail()
    except RetryError as exc:  # pragma: no cover - shouldn't happen
        raise AssertionError("RetryError raised") from exc

    assert result == "ok"
    assert calls["fail"] == 2

