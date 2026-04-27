import pytest

from ai_trading.config.management import get_env


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),
        ("true", True),
        ("1", True),
        ("false", False),
        ("0", False),
        ("TRUE", True),
        ("invalid", False),
    ],
)
def test_disable_daily_retrain_uses_config_bool_parser(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("DISABLE_DAILY_RETRAIN", raising=False)
    else:
        monkeypatch.setenv("DISABLE_DAILY_RETRAIN", raw)

    assert get_env("DISABLE_DAILY_RETRAIN", False, cast=bool) is expected
