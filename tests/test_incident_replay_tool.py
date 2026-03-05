from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import incident_replay


def test_incident_replay_builds_dataset_and_summary(tmp_path: Path) -> None:
    source_log = tmp_path / "decision_records.jsonl"
    source_log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-03-04T20:29:00+00:00",
                        "symbol": "AAPL",
                        "price": 210.5,
                        "volume": 100,
                        "msg": "DECISION_RECORD",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-04T20:30:00+00:00",
                        "symbol": "AAPL",
                        "price": 211.1,
                        "volume": 80,
                        "msg": "ORDER_SUBMIT",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "replay_out"
    summary_path = tmp_path / "summary.json"

    payload = incident_replay.run_incident_replay(
        [
            "--log-path",
            str(source_log),
            "--output-dir",
            str(output_dir),
            "--seed",
            "7",
            "--no-run-offline-replay",
            "--summary-json",
            str(summary_path),
        ]
    )

    assert payload["status"] == "ok"
    assert Path(payload["dataset"]["manifest_path"]).exists()
    assert summary_path.exists()
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["dataset"]["events"] >= 1
