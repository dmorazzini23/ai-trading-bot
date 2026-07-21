from __future__ import annotations

import json

import pytest

from ai_trading.tools.execution_capture_improvement_report import (
    build_execution_capture_improvement_report,
    is_fill_based_execution_evidence,
    main,
)


def _bad_fill(symbol: str = "AAPL") -> dict[str, object]:
    return {
        "ts": "2026-07-01T15:30:00Z",
        "symbol": symbol,
        "side": "buy",
        "session": "rth",
        "order_type": "market",
        "regime": "normal",
        "expected_net_edge_bps": 4.0,
        "realized_net_edge_bps": -2.0,
        "spread_bps": 8.0,
        "quote_age_ms": 250.0,
        "qty": 1,
    }


def test_execution_capture_improvement_builds_bad_buckets_and_training_labels() -> None:
    report = build_execution_capture_improvement_report(
        report_date="2026-07-01",
        fills=[_bad_fill(), _bad_fill(), _bad_fill("MSFT")],
        min_bucket_samples=2,
        min_capture_ratio=0.35,
    )

    assert report["artifact_type"] == "execution_capture_improvement_report"
    assert report["status"] == "needs_review"
    assert report["runtime_authority"] is False
    assert report["promotion_authority"] is False
    assert report["live_money_authority"] is False
    assert report["summary"]["adverse_selection_count"] == 3
    assert report["bad_buckets"]["by_symbol"][0]["bucket_key"] == "AAPL"
    assert report["edge_haircuts"]["by_symbol"]["AAPL"]["action"] == "shadow"
    assert report["edge_haircuts"]["by_symbol"]["AAPL"]["qty_scale"] == 0.0
    assert "realized_net_edge_after_cost" in report["training_labels"]["recommended_targets"]
    assert report["training_labels"]["runtime_authority"] is False


def test_execution_capture_enriches_unique_bounded_fallback_and_derives_spread() -> None:
    fill = {
        "ts": "2026-07-01T15:30:30Z",
        "symbol": "AAPL",
        "side": "buy",
        "realized_net_edge_bps": 2.0,
        "qty": 1,
    }
    tca = {
        "ts": "2026-07-01T15:30:00Z",
        "symbol": "AAPL",
        "side": "buy",
        "expected_net_edge_bps": 4.0,
        "order_type": "limit",
        "session": "rth",
        "market_regime": "sideways",
        "execution_profile": "passive",
        "benchmark": {
            "bid_at_arrival": 99.9,
            "ask_at_arrival": 100.1,
            "mid_at_arrival": 100.0,
        },
    }

    report = build_execution_capture_improvement_report(
        report_date="2026-07-01",
        fills=[fill],
        tca_rows=[tca],
        min_bucket_samples=2,
    )

    assert report["sample_gate"]["samples"] == 1
    assert report["sample_gate"]["sufficient"] is False
    row = report["normalized_rows"][0]
    assert row["metadata_join_method"] == "unique_symbol_side_time"
    assert row["spread_bps"] == pytest.approx(20.0)
    assert row["spread_bucket"] == "spread_normal"
    assert row["metadata_sources"]["spread_bps"] == "derived_bid_ask_mid"
    assert row["quote_age_ms"] is None
    assert row["quote_age_bucket"] == "quote_age_unknown"
    assert row["metadata_missing_reasons"]["quote_age_ms"] == (
        "field_not_recorded_after_unique_symbol_side_time"
    )
    assert row["market_regime"] == "sideways"
    assert row["execution_profile"] == "passive"
    assert report["metadata_quality"]["join_method_counts"] == {
        "unique_symbol_side_time": 1
    }
    assert report["metadata_status"] == "metadata_incomplete"
    assert report["warnings"] == ["metadata_incomplete"]
    assert report["metadata_quality"]["predominantly_unknown_fields"] == [
        "quote_age_ms"
    ]


def test_execution_capture_enriches_exact_broker_order_id_alias() -> None:
    report = build_execution_capture_improvement_report(
        report_date="2026-07-01",
        fills=[
            {
                **_bad_fill(),
                "broker_order_id": "broker-1",
                "order_type": None,
            }
        ],
        tca_rows=[
            {
                "order_id": "broker-1",
                "symbol": "AAPL",
                "side": "buy",
                "order_type": "limit",
            }
        ],
        min_bucket_samples=1,
    )

    row = report["normalized_rows"][0]
    assert row["metadata_join_method"] == "exact_id"
    assert row["order_type"] == "limit"


def test_execution_capture_keeps_ambiguous_and_unmatched_metadata_unknown() -> None:
    base_fill = {
        "ts": "2026-07-01T15:30:30Z",
        "symbol": "AAPL",
        "side": "buy",
        "expected_net_edge_bps": 4.0,
        "realized_net_edge_bps": 2.0,
        "qty": 1,
    }
    ambiguous_tca = [
        {
            "ts": f"2026-07-01T15:30:0{index}Z",
            "symbol": "AAPL",
            "side": "buy",
            "order_type": order_type,
            "session": "rth",
            "decision_quote_age_ms": 100.0,
            "decision_spread_bps": 2.0,
            "market_regime": "sideways",
        }
        for index, order_type in enumerate(("limit", "market"))
    ]

    ambiguous = build_execution_capture_improvement_report(
        report_date="2026-07-01",
        fills=[base_fill],
        tca_rows=ambiguous_tca,
        min_bucket_samples=1,
    )
    ambiguous_row = ambiguous["normalized_rows"][0]
    assert ambiguous_row["metadata_join_method"] == "ambiguous_symbol_side_time"
    assert ambiguous_row["order_type"] == "unknown"
    assert ambiguous_row["market_regime"] == "unknown"
    assert ambiguous["metadata_status"] == "metadata_incomplete"

    unmatched = build_execution_capture_improvement_report(
        report_date="2026-07-01",
        fills=[{**base_fill, "client_order_id": "fill-id"}],
        tca_rows=[{**ambiguous_tca[0], "client_order_id": "different-id"}],
        min_bucket_samples=1,
    )
    unmatched_row = unmatched["normalized_rows"][0]
    assert unmatched_row["metadata_join_method"] == "unmatched_id"
    assert unmatched_row["order_type"] == "unknown"

    outside_window = build_execution_capture_improvement_report(
        report_date="2026-07-01",
        fills=[base_fill],
        tca_rows=[
            {
                **ambiguous_tca[0],
                "ts": "2026-07-01T15:20:00Z",
            }
        ],
        min_bucket_samples=1,
    )
    outside_row = outside_window["normalized_rows"][0]
    assert outside_row["metadata_join_method"] == "no_match"
    assert outside_row["order_type"] == "unknown"


def test_execution_capture_disambiguates_same_second_children_and_excludes_shadow() -> None:
    common_fill = {
        "ts": "2026-07-01T15:30:30Z",
        "correlation_id": "opp-shared",
        "decision_trace_id": "trace-shared",
        "symbol": "AAPL",
        "side": "buy",
        "realized_net_edge_bps": 2.0,
        "qty": 1,
    }
    common_tca = {
        "ts": "2026-07-01T15:30:00Z",
        "correlation_id": "opp-shared",
        "decision_trace_id": "trace-shared",
        "symbol": "AAPL",
        "side": "buy",
        "expected_net_edge_bps": 4.0,
        "session": "midday",
        "market_regime": "sideways",
        "decision_spread_bps": 2.0,
    }
    report = build_execution_capture_improvement_report(
        report_date="2026-07-01",
        fills=[
            {**common_fill, "client_order_id": "child-1"},
            {**common_fill, "client_order_id": "child-2"},
            {
                **common_fill,
                "client_order_id": "shadow-child",
                "evidence_type": "shadow_counterfactual",
                "evidence_partition": "shadow",
                "fill_based_evidence": False,
                "executed": False,
            },
        ],
        tca_rows=[
            {
                **common_tca,
                "client_order_id": "child-1",
                "order_type": "limit",
                "decision_quote_age_ms": 100.0,
            },
            {
                **common_tca,
                "client_order_id": "child-2",
                "order_type": "market",
                "decision_quote_age_ms": 900.0,
            },
        ],
        min_bucket_samples=1,
    )

    assert report["sample_gate"]["samples"] == 2
    assert report["sample_gate"]["non_fill_rows_excluded"] == 1
    rows = {
        row["client_order_id"]: row for row in report["normalized_rows"]
    }
    assert rows["child-1"]["metadata_join_method"] == "correlation_and_order_id"
    assert rows["child-1"]["order_type"] == "limit"
    assert rows["child-1"]["quote_age_ms"] == 100.0
    assert rows["child-2"]["metadata_join_method"] == "correlation_and_order_id"
    assert rows["child-2"]["order_type"] == "market"
    assert rows["child-2"]["quote_age_ms"] == 900.0


def test_fill_evidence_classifier_uses_explicit_real_fill_allowlist() -> None:
    for evidence_type in (
        "fill_execution",
        "paper_fill",
        "live_fill",
        "executed_passive_fill",
        "broker_fill",
        "runtime_fill",
    ):
        assert is_fill_based_execution_evidence({"evidence_type": evidence_type})
    for evidence_type in (
        "historical_research",
        "decision_opportunity",
        "execution_intent",
        "order_execution",
        "shadow_counterfactual",
        "hypothetical",
    ):
        assert not is_fill_based_execution_evidence({"evidence_type": evidence_type})
    assert is_fill_based_execution_evidence({"symbol": "AAPL"})


def test_execution_capture_improvement_cli_writes_latest(tmp_path) -> None:
    fills = tmp_path / "fills.jsonl"
    output = tmp_path / "execution_capture_improvement.json"
    latest = tmp_path / "execution_capture_improvement_latest.json"
    fills.write_text(
        "\n".join(json.dumps(_bad_fill()) for _ in range(2)) + "\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "--report-date",
            "2026-07-01",
            "--fills-jsonl",
            str(fills),
            "--min-bucket-samples",
            "2",
            "--output-json",
            str(output),
            "--latest-json",
            str(latest),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    latest_payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "needs_review"
    assert latest_payload["edge_haircuts"]["runtime_authority"] is False
