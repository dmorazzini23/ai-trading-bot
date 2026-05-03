from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trading.tools import live_cost_model


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) if not isinstance(row, str) else row for row in rows)
        + "\n",
        encoding="utf-8",
    )


def test_live_cost_model_preserves_zero_cost_samples_and_ignores_stale_rows(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 5, 1, 15, 0, tzinfo=UTC)
    events_path = tmp_path / "execution_quality_events.jsonl"
    fill_events_path = tmp_path / "fill_events.jsonl"
    tca_path = tmp_path / "tca_records.jsonl"
    _write_jsonl(
        events_path,
        [
            {
                "ts": (now - timedelta(minutes=5)).isoformat(),
                "event": "submit_outcome",
                "status": "filled",
                "symbol": "AAPL",
                "side": "buy",
                "spread_bps": 0.0,
                "quote_age_ms": 0.0,
                "slippage_bps": 0.0,
            },
            {
                "ts": (now - timedelta(days=3)).isoformat(),
                "event": "submit_outcome",
                "status": "filled",
                "symbol": "AAPL",
                "side": "buy",
                "spread_bps": 100.0,
                "slippage_bps": 100.0,
            },
            "{bad json",
            [],
        ],
    )
    _write_jsonl(fill_events_path, [])
    _write_jsonl(tca_path, [])

    report = live_cost_model.build_live_cost_model(
        events_path=events_path,
        fill_events_path=fill_events_path,
        tca_path=tca_path,
        window_minutes=390,
        min_samples=1,
        now=now,
    )

    assert report["status"]["status"] == "ready"
    assert report["window"]["sample_count"] == 1
    assert report["observed"]["mean_total_cost_bps"] == 0.0
    rows = report["by_symbol_side_session"]
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["mean_spread_bps"] == 0.0
    assert rows[0]["mean_slippage_bps"] == 0.0
    assert rows[0]["mean_total_cost_bps"] == 0.0


def test_live_cost_model_uses_explicit_total_cost_precedence(tmp_path: Path) -> None:
    now = datetime(2026, 5, 1, 15, 30, tzinfo=UTC)
    events_path = tmp_path / "execution_quality_events.jsonl"
    _write_jsonl(
        events_path,
        [
            {
                "ts": now.isoformat(),
                "symbol": "MSFT",
                "side": "sell",
                "market": {"spread_bps": 50.0},
                "cost": {"total_cost_bps": 7.0, "quote_age_ms": 125.0},
            }
        ],
    )

    report = live_cost_model.build_live_cost_model(
        events_path=events_path,
        fill_events_path=None,
        tca_path=None,
        window_minutes=60,
        min_samples=1,
        now=now,
    )

    row = report["by_symbol_side_session"][0]
    assert row["symbol"] == "MSFT"
    assert row["mean_spread_bps"] == 50.0
    assert row["mean_quote_age_ms"] == 125.0
    assert row["mean_total_cost_bps"] == 7.0


def test_live_cost_model_cli_writes_artifact(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    events_path = tmp_path / "execution_quality_events.jsonl"
    fill_events_path = tmp_path / "fill_events.jsonl"
    tca_path = tmp_path / "tca_records.jsonl"
    output_path = tmp_path / "live_cost_model_latest.json"
    _write_jsonl(
        events_path,
        [
            {
                "ts": now.isoformat(),
                "symbol": "AAPL",
                "side": "buy",
                "spread_bps": 2.0,
                "slippage_bps": 1.0,
            }
        ],
    )
    _write_jsonl(fill_events_path, [])
    _write_jsonl(tca_path, [])

    exit_code = live_cost_model.main(
        [
            "--events-jsonl",
            str(events_path),
            "--fill-events-jsonl",
            str(fill_events_path),
            "--tca-jsonl",
            str(tca_path),
            "--output-json",
            str(output_path),
            "--window-minutes",
            "60",
            "--min-samples",
            "1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "live_cost_model"
    assert payload["paths"]["report"] == str(output_path)
