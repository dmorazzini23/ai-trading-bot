from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_trading.meta_learning.core import retrain_meta_learner


def test_retrain_meta_learner_strict_schema_requires_signal_tags(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trade_log = tmp_path / "trades_missing_signal_tags.csv"
    pd.DataFrame(
        {
            "entry_price": [100.0, 101.0],
            "exit_price": [100.5, 100.8],
            "side": ["buy", "sell"],
        }
    ).to_csv(trade_log, index=False)

    monkeypatch.setenv("AI_TRADING_META_STRICT_SCHEMA_ENABLED", "1")
    ok = retrain_meta_learner(
        trade_log_path=str(trade_log),
        model_path=str(tmp_path / "meta.pkl"),
        history_path=str(tmp_path / "meta_history.pkl"),
        min_samples=1,
    )

    assert ok is False
