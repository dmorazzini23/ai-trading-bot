"""Tests for allocation delta sizing and execution logging helpers."""

from types import SimpleNamespace

import logging
import pytest

from ai_trading.core import bot_engine


@pytest.mark.parametrize(
    "side,target,position,open_buy,open_sell,expected",
    [
        ("buy", 50, 10, 12, 4, 32),  # subtract existing long exposure (10 + 12 - 4)
        ("sell_short", 30, -5, 3, 6, 22),  # short exposure subtracts open sells minus open buys
        ("sell", 13, 15, 2, 5, 10),  # cannot exceed available long shares
    ],
)
def test_delta_quantity_accounts_for_open_orders(
    side: str, target: int, position: int, open_buy: int, open_sell: int, expected: int
) -> None:
    """Delta sizing subtracts open orders and clamps to available inventory."""

    result = bot_engine._delta_quantity(side, target, position, open_buy, open_sell)
    assert result == expected


def test_record_broker_sync_metrics_updates_state(caplog) -> None:
    """Broker sync helper should update metrics and emit structured log."""

    state = bot_engine.BotState()
    state.execution_metrics = bot_engine.ExecutionCycleMetrics()
    snapshot = SimpleNamespace(open_orders=(1, 2, 3), positions=("AAPL",))

    caplog.set_level(logging.INFO)
    bot_engine._record_broker_sync_metrics(state, snapshot)

    assert state.execution_metrics.open_orders == 3
    assert state.execution_metrics.positions == 1
    record = next(rec for rec in caplog.records if rec.msg == "BROKER_SYNC")
    assert record.open_orders == 3
    assert record.positions == 1


def test_log_execution_summary_emits_expected_payload(caplog) -> None:
    """Execution summary helper emits consolidated metrics."""

    metrics = bot_engine.ExecutionCycleMetrics(
        submitted=4,
        open_orders=2,
        positions=1,
        exposure_pct=17.345,
        provider_mode="backup",
    )

    caplog.set_level(logging.INFO)
    bot_engine._log_execution_summary(metrics)

    entry = next(rec for rec in caplog.records if rec.msg == "EXEC_SUMMARY")
    assert entry.submitted == 4
    assert entry.open == 2
    assert entry.positions == 1
    expected_exposure = round(metrics.exposure_pct, 2)
    assert entry.exposure_pct == pytest.approx(expected_exposure, rel=1e-6)
    assert entry.provider == "backup"
