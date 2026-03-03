from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tca.rollups import calibrate_cost_model_from_tca


def test_calibrate_cost_model_uses_slippage_bootstrap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    slippage_path = tmp_path / "slippage.csv"
    slippage_path.write_text(
        "\n".join(
            [
                "2026-03-01T13:30:00+00:00,AAPL,buy,1,100.0,100.12,12.0",
                "2026-03-01T13:31:00+00:00,MSFT,sell,1,250.0,249.6,16.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "execution_cost_model.json"

    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_COST_FLOOR_MIN_SAMPLES", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_COST_MODEL_BOOTSTRAP_FROM_SLIPPAGE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_SLIPPAGE_LOG_PATH", str(slippage_path))

    result = calibrate_cost_model_from_tca(
        tca_path=tmp_path / "missing_tca.jsonl",
        model_path=model_path,
        lookback_days=30,
    )

    assert int(result.get("tca_records", 0)) == 0
    assert int(result.get("slippage_records", 0)) == 2
    after = result.get("after", {})
    assert int(after.get("sample_count", 0)) >= 2
    persisted = json.loads(model_path.read_text(encoding="utf-8"))
    assert int(persisted.get("sample_count", 0)) >= 2
