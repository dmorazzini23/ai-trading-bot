from ai_trading.core import bot_engine


def test_retry_budget_short_circuits_when_primary_disabled():
    attempts, delay, reason = bot_engine._short_circuit_retry_budget(  # type: ignore[attr-defined]
        prefer_backup=False,
        primary_disabled=True,
        attempts=3,
        delay=0.5,
    )
    assert attempts == 1
    assert delay == 0.0
    assert reason == "primary_disabled"


def test_retry_budget_short_circuits_when_prefer_backup():
    attempts, delay, reason = bot_engine._short_circuit_retry_budget(  # type: ignore[attr-defined]
        prefer_backup=True,
        primary_disabled=False,
        attempts=4,
        delay=1.0,
    )
    assert attempts == 1
    assert delay == 0.0
    assert reason == "prefer_backup_quotes"
