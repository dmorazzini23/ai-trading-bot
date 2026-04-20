from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from ai_trading.core import bot_engine
from ai_trading.core.legacy_decision_journal import LegacyDecisionJournalRecorder


class _ImmediateExecutor:
    def submit(self, fn, symbol):  # noqa: ANN001
        return SimpleNamespace(result=lambda: fn(symbol))


def test_process_symbols_records_market_closed_skip(monkeypatch) -> None:
    captured: list[dict[str, Any]] = []
    state = bot_engine.BotState()
    state.position_cache = {}
    setattr(
        state,
        "_legacy_decision_recorder",
        LegacyDecisionJournalRecorder(
            path=None,
            write_impl=lambda record, path: captured.append(record.to_dict()),
        ),
    )
    bot_engine.state = state

    monkeypatch.setattr(bot_engine, "get_ctx", lambda: SimpleNamespace())
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: False)
    monkeypatch.setattr(bot_engine, "_trade_limit_reached", lambda *_a, **_k: False)
    monkeypatch.setattr(bot_engine, "_pre_rank_execution_candidates", lambda symbols, runtime=None: symbols)
    monkeypatch.setattr(bot_engine.executors, "_ensure_executors", lambda: None)
    monkeypatch.setattr(bot_engine, "prediction_executor", _ImmediateExecutor(), raising=False)
    monkeypatch.setattr(bot_engine.provider_monitor, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(bot_engine, "_safe_mode_blocks_trading", lambda: False)

    processed, row_counts, fetch_attempts = bot_engine._process_symbols(
        ["AAPL"], 1000.0, None, True
    )

    assert processed == []
    assert row_counts == {}
    assert fetch_attempts == 0
    assert captured
    journal = captured[-1]["decision_journal"]
    assert journal["event"] == "legacy_process_symbols_market_closed_skip"
    assert journal["reasons"] == ["MARKET_CLOSED_SKIP_SYMBOL"]
