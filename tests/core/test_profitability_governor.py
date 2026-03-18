from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import ai_trading.core.bot_engine as bot_engine


def _write_trade_history(path: str, rows: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)


def test_profitability_governor_blocks_negative_symbol(monkeypatch, tmp_path):
    trade_history_path = tmp_path / "trade_history.json"
    now_iso = datetime.now(UTC).isoformat()
    _write_trade_history(
        str(trade_history_path),
        [
            {
                "symbol": "AAPL",
                "net_pnl": -100.0,
                "entry_notional": 10_000.0,
                "regime": "trending",
                "exit_time": now_iso,
            },
            {
                "symbol": "MSFT",
                "net_pnl": 150.0,
                "entry_notional": 10_000.0,
                "regime": "trending",
                "exit_time": now_iso,
            },
        ],
    )

    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH", str(trade_history_path))
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_BLOCK_GLOBAL", "0")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_BLOCK_REGIME", "0")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_BLOCK_SYMBOL", "1")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_MIN_SYMBOL_TRADES", "1")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_MIN_SYMBOL_NET_EDGE_BPS", "0")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_LOOKBACK_DAYS", "5")

    state = bot_engine.BotState()

    assert (
        bot_engine._profitability_governor_allows_entry(
            state,
            symbol="AAPL",
            regime="trending",
            side="long",
        )
        is False
    )
    assert (
        bot_engine._profitability_governor_allows_entry(
            state,
            symbol="MSFT",
            regime="trending",
            side="long",
        )
        is True
    )


def test_profitability_governor_blocks_negative_regime(monkeypatch, tmp_path):
    trade_history_path = tmp_path / "trade_history.json"
    now_iso = datetime.now(UTC).isoformat()
    _write_trade_history(
        str(trade_history_path),
        [
            {
                "symbol": "AAPL",
                "net_pnl": -100.0,
                "entry_notional": 10_000.0,
                "regime": "high_volatility",
                "exit_time": now_iso,
            },
            {
                "symbol": "MSFT",
                "net_pnl": -50.0,
                "entry_notional": 5_000.0,
                "regime": "high_volatility",
                "exit_time": now_iso,
            },
            {
                "symbol": "KO",
                "net_pnl": 100.0,
                "entry_notional": 10_000.0,
                "regime": "sideways",
                "exit_time": now_iso,
            },
        ],
    )

    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH", str(trade_history_path))
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_BLOCK_GLOBAL", "0")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_BLOCK_SYMBOL", "0")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_BLOCK_REGIME", "1")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_MIN_REGIME_TRADES", "2")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_MIN_REGIME_NET_EDGE_BPS", "0")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_LOOKBACK_DAYS", "5")

    state = bot_engine.BotState()

    assert (
        bot_engine._profitability_governor_allows_entry(
            state,
            symbol="AAPL",
            regime="high_volatility",
            side="long",
        )
        is False
    )
    assert (
        bot_engine._profitability_governor_allows_entry(
            state,
            symbol="KO",
            regime="sideways",
            side="long",
        )
        is True
    )


def test_profitability_governor_fail_open_when_history_missing(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing_trade_history.json"

    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH", str(missing_path))
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_FAIL_CLOSED", "0")

    state = bot_engine.BotState()

    assert (
        bot_engine._profitability_governor_allows_entry(
            state,
            symbol="AAPL",
            regime="trending",
            side="short",
        )
        is True
    )


def test_profitability_governor_symbol_cooldown_blocks_until_expiry(monkeypatch, tmp_path):
    trade_history_path = tmp_path / "trade_history.json"
    now_iso = datetime.now(UTC).isoformat()
    _write_trade_history(
        str(trade_history_path),
        [
            {
                "symbol": "AAPL",
                "net_pnl": -100.0,
                "entry_notional": 10_000.0,
                "regime": "trending",
                "exit_time": now_iso,
            }
        ],
    )

    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH", str(trade_history_path))
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_BLOCK_GLOBAL", "0")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_BLOCK_REGIME", "0")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_BLOCK_SYMBOL", "1")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_MIN_SYMBOL_TRADES", "1")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_MIN_SYMBOL_NET_EDGE_BPS", "0")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_LOOKBACK_DAYS", "5")
    monkeypatch.setenv("AI_TRADING_PROFITABILITY_GOVERNOR_SYMBOL_COOLDOWN_MIN", "30")

    state = bot_engine.BotState()

    assert (
        bot_engine._profitability_governor_allows_entry(
            state,
            symbol="AAPL",
            regime="trending",
            side="long",
        )
        is False
    )
    assert "AAPL" in state.profitability_governor_symbol_block_until

    _write_trade_history(
        str(trade_history_path),
        [
            {
                "symbol": "AAPL",
                "net_pnl": 100.0,
                "entry_notional": 10_000.0,
                "regime": "trending",
                "exit_time": now_iso,
            }
        ],
    )
    state.profitability_governor_cache_until_mono = 0.0

    assert (
        bot_engine._profitability_governor_allows_entry(
            state,
            symbol="AAPL",
            regime="trending",
            side="long",
        )
        is False
    )

    state.profitability_governor_symbol_block_until["AAPL"] = datetime.now(UTC) - timedelta(seconds=1)
    state.profitability_governor_cache_until_mono = 0.0

    assert (
        bot_engine._profitability_governor_allows_entry(
            state,
            symbol="AAPL",
            regime="trending",
            side="long",
        )
        is True
    )
