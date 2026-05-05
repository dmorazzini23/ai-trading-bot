from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_trading.tools.microstructure_replay_bridge import build_microstructure_bridge_report


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_microstructure_bridge_joins_quote_telemetry_to_candidates(tmp_path: Path) -> None:
    shadow_path = tmp_path / "shadow.jsonl"
    candidates_path = tmp_path / "accepted_candidates.jsonl"
    output_path = tmp_path / "bridge.json"
    _write_jsonl(
        shadow_path,
        [
            {
                "mode": "ml_signal_shadow",
                "ts": "2026-01-02T14:30:10Z",
                "symbol": "AAPL",
                "market": {
                    "bar_timestamp": "2026-01-02T14:30:00Z",
                    "spread_bps": 4.0,
                    "quote_age_ms": 100.0,
                },
            },
            {
                "mode": "ml_signal_shadow",
                "ts": "2026-01-02T14:31:05Z",
                "symbol": "AAPL",
                "market": {
                    "bar_timestamp": "2026-01-02T14:31:00Z",
                    "spread_bps": 60.0,
                    "quote_age_ms": 2000.0,
                },
            },
        ],
    )
    _write_jsonl(
        candidates_path,
        [
            {
                "ts": "2026-01-02T14:30:20Z",
                "symbol": "AAPL",
                "net_markout_bps": 8.0,
            },
            {
                "ts": "2026-01-02T14:31:30Z",
                "symbol": "AAPL",
                "net_markout_bps": -12.0,
            },
        ],
    )

    report = build_microstructure_bridge_report(
        argparse.Namespace(
            shadow_jsonl=shadow_path,
            accepted_candidates_jsonl=candidates_path,
            output_json=output_path,
            bucket="min",
            match_time_of_day=False,
            max_spread_bps=25.0,
            max_quote_age_ms=1500.0,
            reject_missing=True,
        )
    )

    assert output_path.is_file()
    assert report["join"]["joined_rows"] == 2
    assert report["gate"]["would_reject_count"] == 1
    assert report["gate"]["reason_counts"] == {"stale_quote": 1, "wide_spread": 1}
    assert report["markout"]["retained"]["mean_bps"] == 8.0
    assert report["markout"]["rejected"]["mean_bps"] == -12.0
    assert report["recommendation"] == "eligible_for_enforcement_review"


def test_microstructure_bridge_can_match_time_of_day_across_dates(tmp_path: Path) -> None:
    shadow_path = tmp_path / "shadow.jsonl"
    candidates_path = tmp_path / "accepted_candidates.jsonl"
    output_path = tmp_path / "bridge.json"
    _write_jsonl(
        shadow_path,
        [
            {
                "mode": "ml_signal_shadow",
                "symbol": "AMZN",
                "market": {
                    "bar_timestamp": "2026-05-04T15:45:00Z",
                    "spread_bps": 3.0,
                    "quote_age_ms": 100.0,
                },
            }
        ],
    )
    _write_jsonl(
        candidates_path,
        [
            {
                "ts": "2026-04-10T15:45:30Z",
                "symbol": "AMZN",
                "net_markout_bps": 5.0,
            }
        ],
    )

    report = build_microstructure_bridge_report(
        argparse.Namespace(
            shadow_jsonl=shadow_path,
            accepted_candidates_jsonl=candidates_path,
            output_json=output_path,
            bucket="min",
            match_time_of_day=True,
            max_spread_bps=25.0,
            max_quote_age_ms=1500.0,
            reject_missing=True,
        )
    )

    assert report["inputs"]["match_time_of_day"] is True
    assert report["join"]["joined_rows"] == 1
    assert report["gate"]["would_reject_count"] == 0
