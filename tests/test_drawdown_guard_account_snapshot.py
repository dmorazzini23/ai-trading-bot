import logging
import types

from ai_trading.core import bot_engine
from ai_trading.core.run_all_trades_execution import _drawdown_guard_halts_cycle


class _Breaker:
    def __init__(self, max_drawdown: float = 0.10) -> None:
        self.peak = 0.0
        self.max_drawdown = max_drawdown
        self.scored: list[float] = []

    def update_equity(self, equity: float) -> bool:
        self.scored.append(equity)
        self.peak = max(self.peak, equity)
        return (self.peak - equity) / self.peak < self.max_drawdown

    def get_status(self) -> dict[str, float | bool]:
        current = self.scored[-1]
        return {
            "current_drawdown": (self.peak - current) / self.peak,
            "max_drawdown": self.max_drawdown,
            "peak_equity": self.peak,
            "trading_allowed": True,
        }


def _runtime():
    return types.SimpleNamespace(api=types.SimpleNamespace(list_positions=lambda: []))


def test_missing_account_snapshot_is_not_scored_as_zero_equity(caplog):
    breaker = _Breaker()
    runtime = _runtime()
    assert _drawdown_guard_halts_cycle(
        be=bot_engine, runtime=runtime, dbc=breaker,
        account_snapshot=types.SimpleNamespace(equity=100_000.0),
    ) is False

    with caplog.at_level(logging.WARNING):
        for snapshot in (None, types.SimpleNamespace(equity=None)):
            assert _drawdown_guard_halts_cycle(
                be=bot_engine, runtime=runtime, dbc=breaker, account_snapshot=snapshot
            ) is False

    assert breaker.scored == [100_000.0]
    messages = [r.message for r in caplog.records]
    assert messages.count("DRAWDOWN_CHECK_SKIPPED") == 2
    assert "TRADING_HALTED_DRAWDOWN_PROTECTION" not in messages


def test_real_drawdown_still_halts_the_cycle(monkeypatch):
    managed = []
    monkeypatch.setattr(bot_engine, "manage_position_risk", lambda ctx, pos: managed.append(pos))
    breaker = _Breaker()
    runtime = types.SimpleNamespace(api=types.SimpleNamespace(list_positions=lambda: ["AAPL"]))
    _drawdown_guard_halts_cycle(
        be=bot_engine, runtime=runtime, dbc=breaker,
        account_snapshot=types.SimpleNamespace(equity="100000"),
    )
    assert _drawdown_guard_halts_cycle(
        be=bot_engine, runtime=runtime, dbc=breaker,
        account_snapshot=types.SimpleNamespace(equity="85000"),
    ) is True
    assert managed == ["AAPL"]


def test_unparseable_equity_is_logged_not_raised(caplog):
    breaker = _Breaker()
    with caplog.at_level(logging.ERROR):
        assert _drawdown_guard_halts_cycle(
            be=bot_engine, runtime=_runtime(), dbc=breaker,
            account_snapshot=types.SimpleNamespace(equity="n/a"),
        ) is False
    assert breaker.scored == []
    assert "DRAWDOWN_CHECK_FAILED" in [r.message for r in caplog.records]
