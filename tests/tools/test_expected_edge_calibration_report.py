from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import expected_edge_calibration_report as report_tool


def _rows(count: int, *, expected: float, realized: float, symbol: str = "AAPL") -> list[dict[str, object]]:
    return [
        {
            "ts": f"2026-05-05T14:{index:02d}:00Z",
            "symbol": symbol,
            "side": "buy",
            "session_regime": "midday",
            "expected_net_edge_bps": expected,
            "realized_net_edge_bps": realized,
            "slippage_bps": 1.0,
            "spread_bps": 4.0,
            "quote_age_ms": 250.0,
            "order_type": "limit",
        }
        for index in range(count)
    ]


def test_expected_edge_calibration_classifies_calibrated_edge() -> None:
    payload = report_tool.build_expected_edge_calibration_report(
        report_date="2026-05-05",
        fills=_rows(12, expected=8.0, realized=6.0),
        min_samples=10,
    )

    assert payload["status"] == "calibrated"
    assert payload["summary"]["capture_ratio"] == 0.75
    assert payload["execution_capture_diagnosis"]["attribution_counts"] == {
        "captured_expected_edge": 12
    }


def test_expected_edge_calibration_classifies_overestimated_edge() -> None:
    payload = report_tool.build_expected_edge_calibration_report(
        report_date="2026-05-05",
        fills=_rows(12, expected=30.0, realized=-2.0),
        min_samples=10,
    )

    assert payload["status"] == "overestimated"
    assert payload["recommended_next_action"] == "keep_tiny_sampling_and_recalibrate_signal"
    assert payload["execution_capture_diagnosis"]["worst_buckets"][0]["bucket"] == "edge_25_50"


def test_expected_edge_calibration_classifies_inverted_edge() -> None:
    fills = _rows(6, expected=2.0, realized=3.0) + _rows(6, expected=50.0, realized=-4.0)

    payload = report_tool.build_expected_edge_calibration_report(
        report_date="2026-05-05",
        fills=fills,
        min_samples=10,
    )

    assert payload["status"] == "inverted"
    assert payload["recommended_next_action"] == "pause_scaling_and_retrain_expected_edge_labels"
    assert payload["calibration_correction"]["expected_edge_multiplier"] == 0.0
    assert payload["calibration_correction"]["production_scaling_allowed"] is False


def test_expected_edge_calibration_reports_sell_exit_quality() -> None:
    buy_rows = _rows(8, expected=4.0, realized=2.0)
    sell_rows = [
        {
            **row,
            "side": "sell",
            "expected_net_edge_bps": 1.0,
            "realized_net_edge_bps": -3.0,
        }
        for row in _rows(8, expected=1.0, realized=-3.0, symbol="MSFT")
    ]

    payload = report_tool.build_expected_edge_calibration_report(
        report_date="2026-05-05",
        fills=[*buy_rows, *sell_rows],
        min_samples=10,
    )

    assert payload["exit_quality_diagnostics"]["status"] == "inverted"
    assert (
        payload["exit_quality_diagnostics"]["recommended_action"]
        == "review_exit_timing_and_order_type_before_scaling"
    )
    assert payload["calibration_correction"]["side_multipliers"]["sell"]["expected_edge_multiplier"] == 0.0


def test_expected_edge_calibration_enriches_exact_tca_metadata_without_changing_gate() -> None:
    fill = {
        "ts": "2026-05-05T14:00:03Z",
        "client_order_id": "client-1",
        "symbol": "AAPL",
        "side": "buy",
        "realized_net_edge_bps": 6.0,
    }
    tca = {
        "ts": "2026-05-05T14:00:00Z",
        "client_order_id": "client-1",
        "symbol": "AAPL",
        "side": "buy",
        "expected_net_edge_bps": 8.0,
        "order_type": "limit",
        "session": "opening",
        "decision_quote_age_ms": 125.0,
        "decision_spread_bps": 3.0,
        "market_regime": "sideways",
        "execution_profile": "patient_passive",
        "regime_profile": "legacy_execution_profile",
    }

    payload = report_tool.build_expected_edge_calibration_report(
        report_date="2026-05-05",
        fills=[fill],
        tca_rows=[tca],
        min_samples=2,
    )

    assert payload["sample_gate"] == {
        "min_samples": 2,
        "realized_samples": 1,
        "non_fill_rows_excluded": 0,
        "sufficient": False,
    }
    row = payload["normalized_rows"][0]
    assert row["metadata_join_method"] == "exact_id"
    assert row["order_type"] == "limit"
    assert row["session"] == "opening"
    assert row["quote_age_ms"] == 125.0
    assert row["spread_bps"] == 3.0
    assert row["market_regime"] == "sideways"
    assert row["regime"] == "sideways"
    assert row["execution_profile"] == "patient_passive"
    assert payload["bucketed_by_regime"]["sideways"]["count"] == 1
    assert payload["bucketed_by_execution_profile"]["patient_passive"]["count"] == 1
    assert payload["metadata_quality"]["status"] == "complete"
    assert payload["metadata_quality"]["join_method_counts"] == {"exact_id": 1}


def test_expected_edge_calibration_excludes_shadow_counterfactual_rows() -> None:
    fill = _rows(1, expected=8.0, realized=6.0)[0]
    shadow = {
        **fill,
        "correlation_id": "opp-shadow",
        "evidence_type": "shadow_counterfactual",
        "evidence_partition": "shadow",
        "fill_based_evidence": False,
        "executed": False,
    }

    payload = report_tool.build_expected_edge_calibration_report(
        report_date="2026-05-05",
        fills=[fill, shadow],
        min_samples=2,
    )

    assert payload["sample_gate"]["realized_samples"] == 1
    assert payload["sample_gate"]["non_fill_rows_excluded"] == 1


def test_explicit_research_and_shadow_rows_cannot_inflate_calibration_gate() -> None:
    non_fill_types = (
        "historical_research",
        "decision_opportunity",
        "execution_intent",
        "order_execution",
        "shadow_counterfactual",
        "hypothetical",
    )
    excluded = [
        {
            **_rows(1, expected=100.0, realized=100.0)[0],
            "evidence_type": non_fill_types[index % len(non_fill_types)],
        }
        for index in range(10_000)
    ]
    real_fill = {
        **_rows(1, expected=8.0, realized=6.0)[0],
        "evidence_type": "fill_execution",
    }

    payload = report_tool.build_expected_edge_calibration_report(
        report_date="2026-05-05",
        fills=[*excluded, real_fill],
        min_samples=2,
    )

    assert payload["sample_gate"]["realized_samples"] == 1
    assert payload["sample_gate"]["non_fill_rows_excluded"] == 10_000
    assert payload["sample_gate"]["sufficient"] is False
    assert len(payload["normalized_rows"]) == 1
    assert payload["normalized_rows"][0]["realized_net_edge_bps"] == 6.0
    assert payload["promotion_authority"] is False


def test_expected_edge_calibration_cli_writes_latest(tmp_path: Path) -> None:
    fills = tmp_path / "fills.jsonl"
    out = tmp_path / "calibration.json"
    latest = tmp_path / "latest.json"
    fills.write_text(
        "\n".join(json.dumps(row) for row in _rows(3, expected=8.0, realized=1.0)) + "\n",
        encoding="utf-8",
    )

    rc = report_tool.main(
        [
            "--report-date",
            "2026-05-05",
            "--fills-jsonl",
            str(fills),
            "--min-samples",
            "10",
            "--output-json",
            str(out),
            "--latest-json",
            str(latest),
        ]
    )

    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "insufficient_samples"
    assert latest.is_file()
