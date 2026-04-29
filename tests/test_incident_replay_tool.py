from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_incident_replay_scopes_offline_replay_to_manifest_symbols(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_log = tmp_path / "decision_records.jsonl"
    source_log.write_text(
        json.dumps(
            {
                "timestamp": "2026-03-04T20:29:00+00:00",
                "symbol": "AAPL",
                "price": 210.5,
                "volume": 100,
                "msg": "DECISION_RECORD",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "replay_out"
    output_dir.mkdir()
    stale_csv = output_dir / "MSFT.csv"
    stale_csv.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    captured: dict[str, list[str]] = {}

    def _run_replay(args: list[str]) -> dict[str, object]:
        captured["args"] = args
        return {"aggregate": {"symbols": 1, "total_trades": 0}}

    monkeypatch.setattr(incident_replay, "run_replay", _run_replay)

    payload = incident_replay.run_incident_replay(
        [
            "--log-path",
            str(source_log),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert payload["dataset"]["symbols"] == ["AAPL"]
    assert not stale_csv.exists()
    assert captured["args"] == ["--data-dir", str(output_dir), "--symbols", "AAPL"]


def test_incident_replay_fails_requested_replay_with_zero_events(tmp_path: Path) -> None:
    source_log = tmp_path / "decision_records.jsonl"
    source_log.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="zero replay events"):
        incident_replay.run_incident_replay(
            [
                "--log-path",
                str(source_log),
                "--output-dir",
                str(tmp_path / "replay_out"),
            ]
        )
