from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ai_trading.core.enums import RiskLevel
from ai_trading.oms.event_store import EventStore
from ai_trading.strategies.backtest import BacktestEngine as LegacyBacktestEngine
from ai_trading.strategies.base import BaseStrategy, StrategySignal


pytest.importorskip("sqlalchemy")


class _AlwaysBuyLegacyStrategy(BaseStrategy):
    def __init__(self) -> None:
        super().__init__(
            strategy_id="legacy-oms-events",
            name="legacy-oms-events",
            risk_level=RiskLevel.MODERATE,
        )
        self.symbols = ["AAPL"]

    def generate_signals(self, market_data: dict) -> list[StrategySignal]:
        _ = market_data
        return [
            StrategySignal(
                symbol="AAPL",
                side="buy",
                strength=0.9,
                confidence=0.9,
            )
        ]

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_position: float = 0,
    ) -> int:
        _ = signal
        _ = portfolio_value
        _ = current_position
        return 1


def test_legacy_backtest_emits_oms_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy_backtest_oms_events.db"
    monkeypatch.setenv("AI_TRADING_BACKTEST_OMS_EVENTS_ENABLED", "1")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_PATH", str(db_path))
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    start = datetime(2025, 1, 1, tzinfo=UTC)
    historical_data = {
        "AAPL": [
            {
                "timestamp": start + timedelta(days=offset),
                "open": 100.0 + float(offset),
                "high": 100.5 + float(offset),
                "low": 99.5 + float(offset),
                "close": 100.0 + float(offset),
                "volume": 10_000.0,
            }
            for offset in range(6)
        ]
    }
    strategy = _AlwaysBuyLegacyStrategy()
    engine = LegacyBacktestEngine(
        initial_capital=10_000.0,
        commission_bps=0.0,
        commission_flat=0.0,
        enable_slippage=False,
        enable_partial_fills=False,
    )
    result = engine.run_backtest(
        strategy=strategy,
        historical_data=historical_data,
        start_date=start,
        end_date=start + timedelta(days=5),
    )
    assert int(result.get("total_trades", 0)) > 0

    event_store = EventStore(path=str(db_path))
    rows = event_store.list_oms_events(limit=5000)
    event_store.close()
    assert rows
    event_sources = {str(row.get("event_source")) for row in rows}
    assert "intent_store" in event_sources
    event_types = {str(row.get("event_type")) for row in rows}
    assert "INTENT_CREATED" in event_types
    assert "SUBMIT_CLAIMED" in event_types
    assert "SUBMIT_ATTEMPTED" in event_types
    assert "SUBMIT_ACK" in event_types
    assert "ORDER_PARTIALLY_FILLED" in event_types
    assert "ORDER_FILLED" in event_types
    assert "INTENT_CLOSED" in event_types


def test_legacy_backtest_executes_signal_on_next_bar_open() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    historical_data = {
        "AAPL": [
            {
                "timestamp": start + timedelta(days=offset),
                "open": 90.0 + float(offset),
                "high": 110.0 + float(offset),
                "low": 90.0 + float(offset),
                "close": 100.0 + float(offset),
                "volume": 10_000.0,
            }
            for offset in range(4)
        ]
    }
    historical_data["AAPL"][2]["open"] = 123.0

    strategy = _AlwaysBuyLegacyStrategy()
    engine = LegacyBacktestEngine(
        initial_capital=10_000.0,
        commission_bps=0.0,
        commission_flat=0.0,
        enable_slippage=False,
        enable_partial_fills=False,
    )
    engine.estimate_half_spread = lambda *_args, **_kwargs: 0.0
    engine._normal = lambda *_args, **_kwargs: 0.0

    result = engine.run_backtest(
        strategy=strategy,
        historical_data=historical_data,
        start_date=start,
        end_date=start + timedelta(days=3),
    )

    first_trade = result["trades"][0]
    assert first_trade["signal_price"] == 101.0
    assert first_trade["execution_price"] == 123.0


def test_legacy_backtest_marks_pnl_to_execution_bar_close() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    historical_data = {
        "AAPL": [
            {
                "timestamp": start + timedelta(days=offset),
                "open": 100.0,
                "high": 140.0,
                "low": 90.0,
                "close": 100.0,
                "volume": 10_000.0,
            }
            for offset in range(4)
        ]
    }
    historical_data["AAPL"][2]["open"] = 120.0
    historical_data["AAPL"][2]["close"] = 132.0

    strategy = _AlwaysBuyLegacyStrategy()
    engine = LegacyBacktestEngine(
        initial_capital=10_000.0,
        commission_bps=0.0,
        commission_flat=0.0,
        enable_slippage=False,
        enable_partial_fills=False,
    )
    engine.estimate_half_spread = lambda *_args, **_kwargs: 0.0
    normal_draws = [0.0, 99.0]
    engine._normal = lambda *_args, **_kwargs: normal_draws.pop(0) if normal_draws else 99.0

    result = engine.run_backtest(
        strategy=strategy,
        historical_data=historical_data,
        start_date=start,
        end_date=start + timedelta(days=3),
    )

    first_trade = result["trades"][0]
    assert first_trade["execution_price"] == 120.0
    assert first_trade["exit_price"] == 132.0
    assert first_trade["gross_pnl"] == pytest.approx(12.0)
