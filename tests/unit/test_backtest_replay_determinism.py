from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ai_trading.core.enums import RiskLevel
from ai_trading.oms.event_store import EventStore
from ai_trading.strategies.backtest import BacktestEngine as OmsBacktestEngine
from ai_trading.strategies.backtester import BacktestEngine, DefaultExecutionModel
from ai_trading.strategies.base import BaseStrategy, StrategySignal


pd = pytest.importorskip("pandas")
pytest.importorskip("sqlalchemy")


class _AlwaysBuyStrategy(BaseStrategy):
    def __init__(self) -> None:
        super().__init__(
            strategy_id="replay-determinism",
            name="replay-determinism",
            risk_level=RiskLevel.MODERATE,
        )
        self.symbols = ["AAPL"]

    def generate_signals(self, market_data: dict[str, Any]) -> list[StrategySignal]:
        _ = market_data
        return [
            StrategySignal(
                symbol="AAPL",
                side="buy",
                strength=0.8,
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


def _normalize_scalar(value: Any) -> Any:
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return str(iso())
    return value


def _normalize_trade_frame(frame: Any) -> list[dict[str, Any]]:
    if getattr(frame, "empty", True):
        return []
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        normalized = {
            str(key): _normalize_scalar(value)
            for key, value in dict(row).items()
        }
        records.append(normalized)
    return records


def _normalize_equity_frame(frame: Any) -> list[dict[str, Any]]:
    if getattr(frame, "empty", True):
        return []
    records: list[dict[str, Any]] = []
    reset = frame.reset_index()
    for row in reset.to_dict("records"):
        normalized = {
            str(key): _normalize_scalar(value)
            for key, value in dict(row).items()
        }
        records.append(normalized)
    return records


def _normalized_oms_rows(db_path: Path) -> list[dict[str, Any]]:
    event_store = EventStore(url=f"sqlite:///{db_path}")
    try:
        rows = event_store.list_oms_events(limit=5000)
    finally:
        event_store.close()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "intent_id": row.get("intent_id"),
                "event_type": row.get("event_type"),
                "event_source": row.get("event_source"),
                "idempotency_key": row.get("idempotency_key"),
                "sequence_no": row.get("sequence_no"),
                "event_ts": row.get("event_ts"),
                "error_code": row.get("error_code"),
                "broker_order_id": row.get("broker_order_id"),
                "fill_id": row.get("fill_id"),
                "payload_json": row.get("payload_json"),
            }
        )
    return normalized


def test_modern_backtester_replay_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AI_TRADING_BACKTEST_OMS_EVENTS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    index = pd.date_range("2025-01-01", periods=18, freq="D")
    closes = [100, 100, 100, 101, 102, 103, 101, 99, 100, 102, 104, 103, 105, 106, 107, 106, 108, 109]
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": [value + 0.5 for value in closes],
            "low": [value - 0.5 for value in closes],
            "close": closes,
            "volume": [2_500] * len(closes),
        },
        index=index,
    )

    first_db = tmp_path / "modern_replay_first.db"
    second_db = tmp_path / "modern_replay_second.db"

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{first_db}")
    first_engine = BacktestEngine(
        {"AAPL": frame},
        DefaultExecutionModel(per_share_fee=0.05, slippage_pips=0.01, latency=2),
    )
    first_result = first_engine.run(["AAPL"])

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{second_db}")
    second_engine = BacktestEngine(
        {"AAPL": frame},
        DefaultExecutionModel(per_share_fee=0.05, slippage_pips=0.01, latency=2),
    )
    second_result = second_engine.run(["AAPL"])

    assert _normalize_trade_frame(first_result.trades) == _normalize_trade_frame(second_result.trades)
    assert _normalize_equity_frame(first_result.equity_curve) == _normalize_equity_frame(second_result.equity_curve)
    assert _normalized_oms_rows(first_db) == _normalized_oms_rows(second_db)


def test_backtest_replay_is_deterministic_without_input_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AI_TRADING_BACKTEST_OMS_EVENTS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_LEGACY_BACKTEST_SEED", "17")

    start = datetime(2025, 1, 1, tzinfo=UTC)
    historical_data = {
        "AAPL": [
            {
                "open": 100.0 + float(offset),
                "high": 100.5 + float(offset),
                "low": 99.5 + float(offset),
                "close": 100.0 + float(offset),
                "volume": 10_000.0,
            }
            for offset in range(8)
        ]
    }
    strategy = _AlwaysBuyStrategy()
    first_db = tmp_path / "backtest_replay_first.db"
    second_db = tmp_path / "backtest_replay_second.db"

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{first_db}")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_PATH", str(first_db))
    first_engine = OmsBacktestEngine(
        initial_capital=10_000.0,
        commission_bps=1.0,
        commission_flat=0.25,
        enable_slippage=True,
        enable_partial_fills=True,
    )
    first_result = first_engine.run_backtest(
        strategy=strategy,
        historical_data=historical_data,
        start_date=start,
        end_date=start + timedelta(days=7),
    )

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{second_db}")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_PATH", str(second_db))
    second_engine = OmsBacktestEngine(
        initial_capital=10_000.0,
        commission_bps=1.0,
        commission_flat=0.25,
        enable_slippage=True,
        enable_partial_fills=True,
    )
    second_result = second_engine.run_backtest(
        strategy=strategy,
        historical_data=historical_data,
        start_date=start,
        end_date=start + timedelta(days=7),
    )

    assert dict(first_result) == dict(second_result)
    assert _normalized_oms_rows(first_db) == _normalized_oms_rows(second_db)
