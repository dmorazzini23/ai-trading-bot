import time

from ai_trading import main
import ai_trading.core.bot_engine as bot_engine
from ai_trading.utils.prof import SoftBudget


def test_run_cycle_respects_budget(monkeypatch):
    """run_cycle should complete within the configured time budget."""
    # Avoid heavy operations by stubbing the worker
    monkeypatch.setattr(bot_engine, "run_all_trades_worker", lambda state, runtime: None)

    budget = SoftBudget(500)
    start = time.perf_counter()
    main.run_cycle()
    duration = time.perf_counter() - start

    assert duration < 0.5
    assert not budget.over_budget()
