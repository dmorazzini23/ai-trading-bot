import threading


def test_portfolio_lock_is_exported_and_is_same_object():
    from ai_trading.utils import portfolio_lock
    from ai_trading.utils.base import portfolio_lock as base_lock
    from ai_trading.utils.locks import portfolio_lock as locks_lock

    assert isinstance(portfolio_lock, type(threading.Lock()))
    # all modules should reference the exact same lock instance
    assert portfolio_lock is base_lock is locks_lock
