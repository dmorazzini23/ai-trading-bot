from ai_trading.utils.retry import retry


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
        except RuntimeError:
            pass

    assert calls["fixed"] == 2
    assert calls["linear"] == 3

