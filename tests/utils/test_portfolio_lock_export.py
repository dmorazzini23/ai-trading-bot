import threading


def test_portfolio_lock_is_exported_and_is_same_object():
    from ai_trading.utils import portfolio_lock
    from ai_trading.utils.base import portfolio_lock as base_lock
    assert isinstance(portfolio_lock, type(threading.Lock()))
    # same underlying object (identity), not a new lock
    assert portfolio_lock is base_lock
