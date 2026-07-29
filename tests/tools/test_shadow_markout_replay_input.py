from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pandas as pd
import pytest

from ai_trading.analytics.opportunity_markouts import resolve_opportunity_markouts
from ai_trading.tools.shadow_markout_replay_input import (
    build_shadow_markout_replay_rows,
    main,
    write_shadow_markout_replay_input,
)


def _decision(
    correlation_id: str,
    source_timestamp: datetime,
    *,
    submitted: bool,
) -> dict[str, object]:
    timestamp = source_timestamp.isoformat()
    return {
        "correlation_id": correlation_id,
        "symbol": "AAPL",
        "bar_ts": timestamp,
        "metrics": {
            "opportunity_eligible": True,
            "source_timestamp": timestamp,
            "reference_price": 100.0,
            "spread_bps": 2.0,
        },
        "decision_journal": {
            "correlation_id": correlation_id,
            "symbol": "AAPL",
            "bar_ts": timestamp,
            "source_timestamp": timestamp,
            "decision_ts": timestamp,
            "submitted": submitted,
            "event": "decision_record",
            "signal": {"symbol": "AAPL", "side": "buy"},
            "metadata": {
                "opportunity_eligible": True,
                "reference_price": 100.0,
                "spread_bps": 2.0,
                "session_regime": "midday",
                "market_regime": "sideways",
            },
        },
    }


def _markout_report() -> dict[str, object]:
    source_timestamp = datetime(2026, 7, 21, 14, 30, tzinfo=UTC)
    bars = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
        index=pd.date_range(source_timestamp, periods=6, freq="min"),
    )
    report = resolve_opportunity_markouts(
        [
            _decision("opp-not-submitted", source_timestamp, submitted=False),
            _decision("opp-submitted", source_timestamp, submitted=True),
        ],
        {"AAPL": bars},
    )
    report["report_date"] = "2026-07-21"
    return report


def test_build_research_rows_is_resolved_only_deterministic_and_bounded() -> None:
    report = _markout_report()

    rows = build_shadow_markout_replay_rows(
        report,
        source_report_sha256="a" * 64,
        max_rows=6,
    )

    assert len(rows) == 6
    assert len({row["replay_row_id"] for row in rows}) == 6
    assert {row["horizon_bars"] for row in rows} == {1, 3, 5}
    assert {row["submitted"] for row in rows} == {False, True}
    assert all(row["bar_timeframe"] == "1Min" for row in rows)
    assert all(row["research_only"] is True for row in rows)
    assert all(row["fill_based_evidence"] is False for row in rows)
    assert all(row["promotion_eligible"] is False for row in rows)
    assert all(row["runtime_authority"] is False for row in rows)
    assert all(row["promotion_authority"] is False for row in rows)
    assert all(row["live_money_authority"] is False for row in rows)

    with pytest.raises(ValueError, match="exceed bounded maximum"):
        build_shadow_markout_replay_rows(
            report,
            source_report_sha256="a" * 64,
            max_rows=5,
        )


def test_research_input_writer_is_atomic_and_publishes_manifest(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "opportunity_markouts.json"
    report_path.write_text(json.dumps(_markout_report()), encoding="utf-8")
    output = tmp_path / "shadow_markouts_20260721.jsonl"
    latest = tmp_path / "shadow_markouts_latest.jsonl"
    manifest = tmp_path / "shadow_markouts_20260721.manifest.json"
    latest_manifest = tmp_path / "shadow_markouts_latest.manifest.json"

    payload = write_shadow_markout_replay_input(
        markout_report_path=report_path,
        output_jsonl=output,
        latest_jsonl=latest,
        manifest_json=manifest,
        latest_manifest_json=latest_manifest,
        max_rows=10,
    )

    assert output.read_bytes() == latest.read_bytes()
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6
    assert payload["row_count"] == 6
    assert payload["resolved_only"] is True
    assert payload["research_only"] is True
    assert payload["promotion_authority"] is False
    assert json.loads(manifest.read_text(encoding="utf-8")) == payload
    assert latest_manifest.read_bytes() == manifest.read_bytes()
    assert not list(tmp_path.glob(".*.tmp"))


def test_research_input_rejects_mixed_authority_and_governance_target(
    tmp_path: Path,
) -> None:
    report = _markout_report()
    outcomes = report["outcomes"]
    assert isinstance(outcomes, list)
    outcomes[0]["promotion_eligible"] = True
    report_path = tmp_path / "markouts.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="promotion_eligible"):
        write_shadow_markout_replay_input(
            markout_report_path=report_path,
            output_jsonl=tmp_path / "research.jsonl",
        )

    clean_path = tmp_path / "clean.json"
    clean_path.write_text(json.dumps(_markout_report()), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot target replay governance"):
        write_shadow_markout_replay_input(
            markout_report_path=clean_path,
            output_jsonl=tmp_path / "governance_tca_recent.jsonl",
        )


def test_shadow_markout_replay_cli_contract(tmp_path: Path) -> None:
    report_path = tmp_path / "markouts.json"
    report_path.write_text(json.dumps(_markout_report()), encoding="utf-8")
    output = tmp_path / "research.jsonl"
    manifest = tmp_path / "manifest.json"

    assert main(
        [
            "--markout-report-json",
            str(report_path),
            "--output-jsonl",
            str(output),
            "--manifest-json",
            str(manifest),
            "--max-rows",
            "10",
        ]
    ) == 0

    assert output.exists()
    assert json.loads(manifest.read_text(encoding="utf-8"))["row_count"] == 6
