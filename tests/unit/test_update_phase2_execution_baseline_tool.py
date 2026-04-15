from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trading.tools.update_phase2_execution_baseline import (
    main,
    update_phase2_execution_baseline,
)


def _write_tca(path: Path) -> None:
    now = datetime.now(UTC)
    rows = [
        {
            "ts": (now - timedelta(hours=1)).isoformat(),
            "symbol": "AAPL",
            "status": "filled",
            "order_type": "limit",
            "midpoint_offset_bps": 4.0,
            "is_bps": 7.0,
            "execution_drift_bps": 4.0,
        },
        {
            "ts": (now - timedelta(hours=1)).isoformat(),
            "symbol": "AAPL",
            "status": "filled",
            "order_type": "limit",
            "midpoint_offset_bps": 4.0,
            "is_bps": 9.0,
            "execution_drift_bps": 6.0,
        },
        {
            "ts": (now - timedelta(hours=1)).isoformat(),
            "symbol": "AAPL",
            "status": "canceled",
            "order_type": "limit",
            "pending_terminal_nonfill": True,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_update_phase2_execution_baseline_writes_payload(tmp_path: Path) -> None:
    tca_path = tmp_path / "tca_records.jsonl"
    out_path = tmp_path / "phase2_baseline.json"
    _write_tca(tca_path)

    summary = update_phase2_execution_baseline(
        tca_path=str(tca_path),
        output_path=str(out_path),
        window_days=30,
    )
    assert summary["ok"] is True
    assert summary["output_path"] == str(out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    baselines = payload["baselines"]
    assert baselines["slippage_median_abs_bps"] == 8.0
    assert baselines["target_limit_fill_rate"] == 1.0
    assert baselines["stale_pending_count"] == 1.0


def test_update_phase2_execution_baseline_cli(tmp_path: Path, capsys) -> None:
    tca_path = tmp_path / "tca_records.jsonl"
    out_path = tmp_path / "phase2_baseline_cli.json"
    _write_tca(tca_path)

    exit_code = main(
        [
            "--tca-path",
            str(tca_path),
            "--output-path",
            str(out_path),
            "--window-days",
            "7",
        ]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)
    assert payload["ok"] is True
    assert payload["output_path"] == str(out_path)
