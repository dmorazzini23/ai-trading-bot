from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_trading import meta_learning
from ai_trading.tools.refresh_meta_model import main


def test_refresh_meta_model_tool_retrains_and_writes_model(tmp_path: Path) -> None:
    trade_log = tmp_path / "trades.csv"
    pd.DataFrame(
        {
            "entry_price": [100.0, 101.0, 102.0],
            "exit_price": [101.0, 102.5, 101.5],
            "signal_tags": ["momentum", "trend", "momentum+trend"],
            "side": ["buy", "buy", "sell"],
        }
    ).to_csv(trade_log, index=False)

    model_path = tmp_path / "meta_model.pkl"
    history_path = tmp_path / "meta_retrain_history.pkl"

    rc = main(
        [
            "--trade-log-path",
            str(trade_log),
            "--model-path",
            str(model_path),
            "--history-path",
            str(history_path),
            "--min-samples",
            "1",
        ]
    )
    assert rc == 0
    assert model_path.exists()
    assert history_path.exists()
    history = meta_learning.load_model_checkpoint(str(history_path))
    assert isinstance(history, list)
    assert history
