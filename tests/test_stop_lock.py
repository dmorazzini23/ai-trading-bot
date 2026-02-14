from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from ai_trading.config.runtime import TradingConfig
from ai_trading.core import bot_engine


def test_stop_lock_blocks_reentry(monkeypatch):
    cfg = TradingConfig.from_env(allow_missing_drawdown=True)
    cfg.update(
        netting_enabled=True,
        data_contract_enabled=False,
        recon_enabled=False,
        ledger_enabled=False,
        rth_only=False,
        allow_extended=True,
        decision_log_path=None,
    )
    runtime = SimpleNamespace(cfg=cfg, tickers=["AAPL"], universe_tickers=["AAPL"], api=None)
    state = bot_engine.BotState()

    bar_ts = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1000],
        },
        index=pd.DatetimeIndex([bar_ts - timedelta(minutes=1), bar_ts], tz=UTC),
    )

    monkeypatch.setattr("ai_trading.data.fetch.get_bars_batch", lambda symbols, timeframe, start, end: {"AAPL": df})
    monkeypatch.setattr(bot_engine, "ensure_data_fetcher", lambda runtime: None)
    monkeypatch.setattr(bot_engine, "compute_current_positions", lambda runtime: {})

    records: list[dict] = []

    def _capture(record, path):
        records.append(record.to_dict())

    monkeypatch.setattr(bot_engine, "_write_decision_record", _capture)

    state.stop_lock["AAPL"] = {"bar_ts": bar_ts, "direction": "long"}

    bot_engine._run_netting_cycle(state, runtime, "loop", 0.0)
    assert records
    assert any("STOP_LOCK" in r.get("gates", []) for r in records)
