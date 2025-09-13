from ai_trading.utils import sleep


def test_utils_sleep_zero_returns_zero() -> None:
    """sleep should return 0 when duration is 0."""
    assert sleep(0) == 0.0
