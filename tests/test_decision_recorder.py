from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_trading.contracts import OrderIntent
from ai_trading.core.decision_log import DecisionRecorder
from ai_trading.core.netting import NettedTarget, SleeveProposal


def test_decision_recorder_tracks_counts_and_observations() -> None:
    written: list[object] = []
    recorder = DecisionRecorder(
        runtime=type(
            "Runtime",
            (),
            {"execution_candidate_rank_expected_edge_bps": {"AAPL": 12.5}},
        )(),
        path="runtime/decision_records.jsonl",
        write_impl=lambda record, path: written.append((record, path)),
        dedupe_gate_root_causes=lambda gates: list(dict.fromkeys(gates)),
        session_bucket_from_ts=lambda _ts: "rth",
        safe_float=lambda value: float(value) if value is not None else None,
    )
    bar_ts = datetime(2026, 4, 19, 14, 30, tzinfo=UTC)
    proposal = SleeveProposal(
        symbol="AAPL",
        sleeve="intraday",
        bar_ts=bar_ts,
        target_dollars=1000.0,
        expected_edge_bps=12.5,
        expected_cost_bps=4.0,
        score=0.8,
        confidence=0.7,
    )
    target = NettedTarget(
        symbol="AAPL",
        bar_ts=bar_ts,
        target_dollars=1000.0,
        target_shares=5.0,
        proposals=[proposal],
    )

    recorder.record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=target,
        sleeves=[proposal],
        gates=["OK_TRADE", "OK_TRADE"],
        order={"side": "buy", "qty": 5},
        metrics={"expected_net_edge_bps": 12.5},
        config_snapshot={"liquidity_regime": "normal"},
    )

    assert len(written) == 1
    assert recorder.decision_records_total == 1
    assert recorder.decision_gate_counts["OK_TRADE"] == 1
    assert recorder.decision_observations[0]["accepted"] is True
    assert recorder.decision_observations[0]["expected_net_edge_bps"] == 12.5
    assert recorder.decision_observations[0]["session_bucket"] == "rth"
    assert str(recorder.decision_observations[0]["correlation_id"]).startswith("opp_")


def test_decision_recorder_can_record_global_block() -> None:
    written: list[object] = []
    recorder = DecisionRecorder(
        runtime=type("Runtime", (), {"execution_candidate_rank_expected_edge_bps": {}})(),
        path="runtime/decision_records.jsonl",
        write_impl=lambda record, path: written.append((record, path)),
        dedupe_gate_root_causes=lambda gates: list(dict.fromkeys(gates)),
        session_bucket_from_ts=lambda _ts: "offhours",
        safe_float=lambda value: float(value) if value is not None else None,
    )
    bar_ts = datetime(2026, 4, 19, 20, 0, tzinfo=UTC)

    record = recorder.record_global_block(
        bar_ts=bar_ts,
        gates=["MARKET_CLOSED_BLOCK", "IDLE_MARKET_CLOSED"],
        config_snapshot={"liquidity_regime": "closed"},
    )

    assert len(written) == 1
    assert record.symbol == "ALL"
    assert record.net_target.target_dollars == 0.0
    assert recorder.decision_gate_counts["MARKET_CLOSED_BLOCK"] == 1
    assert recorder.decision_observations[0]["symbol"] == "ALL"


def test_decision_recorder_partitions_stale_model_counterfactual() -> None:
    recorder = DecisionRecorder(
        runtime=type("Runtime", (), {"execution_candidate_rank_expected_edge_bps": {}})(),
        path=None,
        write_impl=lambda _record, _path: None,
        dedupe_gate_root_causes=lambda gates: list(gates),
        session_bucket_from_ts=lambda _ts: "opening",
        safe_float=lambda value: float(value) if value is not None else None,
        quote_snapshot_func=lambda _symbol: {"bid": 99.9, "ask": 100.1},
    )
    bar_ts = datetime(2026, 8, 3, 13, 40, tzinfo=UTC)
    proposal = SleeveProposal(
        symbol="AAPL",
        sleeve="day",
        bar_ts=bar_ts,
        target_dollars=0.0,
        expected_edge_bps=5.0,
        expected_cost_bps=2.0,
        score=0.7,
        confidence=0.6,
        blocked=True,
        reason_code="DAY_SLEEVE_ML_REQUIRED_UNAVAILABLE",
        debug={
            "evidence_partition": "stale_model_shadow",
            "counterfactual_side": "buy",
            "counterfactual_target_dollars": 500.0,
            "counterfactual_score": 0.7,
            "counterfactual_confidence": 0.6,
        },
    )
    record = recorder.build_record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=NettedTarget(
            symbol="AAPL",
            bar_ts=bar_ts,
            target_dollars=0.0,
            target_shares=0.0,
            proposals=[proposal],
        ),
        gates=["DAY_SLEEVE_ML_REQUIRED_UNAVAILABLE"],
    )

    assert record.metrics["opportunity_eligible"] is True
    assert record.metrics["opportunity_side"] == "buy"
    assert record.metrics["opportunity_quantity"] == pytest.approx(5.0)
    assert record.metrics["evidence_partition"] == "stale_model_shadow"
    assert record.metrics["research_only"] is True
    assert record.metrics["promotion_eligible"] is False
    assert record.metrics["model_authority"] is False
    assert record.metrics["runtime_authority"] is False
    assert record.metrics["live_money_authority"] is False


def test_decision_recorder_correlation_is_not_derived_from_order_intent() -> None:
    recorder = _lineage_test_recorder([])
    bar_ts = datetime(2026, 7, 21, 14, 30, tzinfo=UTC)
    proposal = _lineage_test_proposal(bar_ts=bar_ts, ml_influenced=False)
    target = NettedTarget(
        symbol="AAPL",
        bar_ts=bar_ts,
        target_dollars=1000.0,
        target_shares=5.0,
        proposals=[proposal],
    )

    blocked = recorder.build_record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=target,
        sleeves=[proposal],
        gates=["NET_EDGE_FLOOR_GATE"],
    )
    submitted = recorder.build_record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=target,
        sleeves=[proposal],
        gates=["OK_TRADE"],
        order={"side": "buy", "qty": 5},
        order_intent=OrderIntent(
            symbol="AAPL",
            side="buy",
            bar_ts=bar_ts,
            qty=5.0,
            strategy_id="late_order_strategy",
            correlation_id="opp_conflicting_order_intent",
        ),
    )

    assert blocked.correlation_id == submitted.correlation_id
    assert submitted.order_intent is not None
    assert submitted.order_intent.correlation_id == blocked.correlation_id
    assert submitted.order_intent.metadata["correlation_id"] == blocked.correlation_id


def test_decision_recorder_preserves_quote_and_regime_metadata() -> None:
    recorder = DecisionRecorder(
        runtime=type("Runtime", (), {"execution_candidate_rank_expected_edge_bps": {}})(),
        path="runtime/decision_records.jsonl",
        write_impl=lambda _record, _path: None,
        dedupe_gate_root_causes=lambda gates: list(dict.fromkeys(gates)),
        session_bucket_from_ts=lambda _ts: "opening",
        safe_float=lambda value: float(value) if value is not None else None,
        quote_snapshot_func=lambda _symbol: {
            "status": "fresh",
            "source": "alpaca_iex",
            "bid": 100.0,
            "ask": 100.1,
            "quote_age_ms": 250.0,
            "quote_timestamp": "2026-07-21T14:29:59.750000+00:00",
            "anomaly_spread_bps": 5.0,
            "execution_profile": "paper_sampling_passive",
            "market_regime": "sideways",
        },
    )
    bar_ts = datetime(2026, 7, 21, 14, 30, tzinfo=UTC)
    proposal = SleeveProposal(
        symbol="AAPL",
        sleeve="day",
        bar_ts=bar_ts,
        target_dollars=1000.0,
        expected_edge_bps=12.0,
        expected_cost_bps=3.0,
        score=0.7,
        confidence=0.8,
        debug={"ml_serving_regime": "volatile"},
    )
    target = NettedTarget(
        symbol="AAPL",
        bar_ts=bar_ts,
        target_dollars=1000.0,
        target_shares=5.0,
        proposals=[proposal],
    )

    record = recorder.build_record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=target,
        sleeves=[proposal],
        gates=["NET_EDGE_FLOOR_GATE"],
        config_snapshot={
            "regime_signal_profile": "conservative",
            "liquidity_regime": "THIN",
        },
    )

    assert record.metrics is not None
    assert record.metrics["quote_age_ms"] == 250.0
    assert record.metrics["spread_bps"] == pytest.approx(9.99500249874909)
    assert record.metrics["raw_quote_bid"] == 100.0
    assert record.metrics["raw_quote_ask"] == 100.1
    assert record.metrics["raw_quote_spread_bps"] == pytest.approx(9.99500249874909)
    assert record.metrics["raw_quote_source"] == "alpaca_iex"
    assert record.metrics["raw_quote_age_ms"] == 250.0
    assert record.metrics["quote_quality_status"] == "anomalous"
    assert (
        record.metrics["quote_anomaly_reason"]
        == "spread_at_or_above_anomaly_threshold"
    )
    assert record.metrics["order_type"] == "not_submitted"
    assert record.metrics["session_regime"] == "opening"
    assert record.metrics["market_regime"] == "volatile"
    assert record.metrics["volatility_regime"] == "volatile"
    assert record.metrics["regime_profile"] == "conservative"
    assert record.metrics["liquidity_regime"] == "thin"
    assert record.metrics["execution_profile"] == "paper_sampling_passive"
    assert record.metrics["metadata_quality_status"] == "complete"
    assert record.metrics["metadata_missing_reasons"] == {}

    metadata = record.to_dict()["decision_journal"]["metadata"]
    assert metadata["quote_age_ms"] == 250.0
    assert metadata["spread_bps"] == pytest.approx(9.99500249874909)
    assert metadata["order_type"] == "not_submitted"
    assert metadata["session_regime"] == "opening"
    assert metadata["market_regime"] == "volatile"
    assert metadata["execution_profile"] == "paper_sampling_passive"
    assert metadata["metadata_quality_status"] == "complete"
    assert metadata["metadata_missing_reasons"] == {}


def test_decision_recorder_marks_explicit_monitor_only_replay_evidence() -> None:
    recorder = DecisionRecorder(
        runtime=type("Runtime", (), {"execution_candidate_rank_expected_edge_bps": {}})(),
        path="runtime/decision_records.jsonl",
        write_impl=lambda _record, _path: None,
        dedupe_gate_root_causes=lambda gates: list(dict.fromkeys(gates)),
        session_bucket_from_ts=lambda _ts: "midday",
        safe_float=lambda value: float(value) if value is not None else None,
        quote_snapshot_func=lambda _symbol: {
            "bid": 100.0,
            "ask": 100.1,
            "quote_timestamp": "2026-07-21T16:00:00+00:00",
        },
        replay_entry_control_func=lambda: {
            "status": "monitor_only",
            "monitor_only": True,
            "block_exposure_increasing": False,
            "gate_reason": "replay_counterfactual",
        },
    )
    bar_ts = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)
    proposal = _lineage_test_proposal(bar_ts=bar_ts, ml_influenced=True)
    record = recorder.build_record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=NettedTarget(
            symbol="AAPL",
            bar_ts=bar_ts,
            target_dollars=1000.0,
            target_shares=0.0,
            proposals=[proposal],
        ),
        sleeves=[proposal],
        gates=["CAPACITY_THROTTLE_BLOCK"],
    )

    assert record.metrics["replay_shadow_evidence"] is True
    assert record.metrics["replay_live_entry_control"]["status"] == "monitor_only"
    assert record.metrics["opportunity_side"] == "buy"
    assert record.metrics["opportunity_quantity"] == pytest.approx(1000.0 / 100.05)
    assert record.metrics["opportunity_price"] == pytest.approx(100.05)


def test_decision_recorder_does_not_mark_inconsistent_replay_control() -> None:
    recorder = _lineage_test_recorder([])
    recorder.replay_entry_control_func = lambda: {
        "status": "monitor_only",
        "monitor_only": False,
    }
    bar_ts = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)
    proposal = _lineage_test_proposal(bar_ts=bar_ts, ml_influenced=False)
    record = recorder.build_record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=NettedTarget(
            symbol="AAPL",
            bar_ts=bar_ts,
            target_dollars=1000.0,
            target_shares=5.0,
            proposals=[proposal],
        ),
        sleeves=[proposal],
        gates=["NET_EDGE_FLOOR_GATE"],
    )

    assert "replay_shadow_evidence" not in record.metrics


def test_decision_recorder_reports_quote_metadata_failure_without_crashing() -> None:
    def _raise_quote_error(_symbol: str) -> dict[str, object]:
        raise RuntimeError("quote cache unavailable")

    recorder = DecisionRecorder(
        runtime=type("Runtime", (), {"execution_candidate_rank_expected_edge_bps": {}})(),
        path="runtime/decision_records.jsonl",
        write_impl=lambda _record, _path: None,
        dedupe_gate_root_causes=lambda gates: list(dict.fromkeys(gates)),
        session_bucket_from_ts=lambda _ts: "midday",
        safe_float=lambda value: float(value) if value is not None else None,
        quote_snapshot_func=_raise_quote_error,
    )
    bar_ts = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)
    proposal = _lineage_test_proposal(bar_ts=bar_ts, ml_influenced=False)
    target = NettedTarget(
        symbol="AAPL",
        bar_ts=bar_ts,
        target_dollars=1000.0,
        target_shares=5.0,
        proposals=[proposal],
    )

    record = recorder.build_record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=target,
        sleeves=[proposal],
        gates=["NET_EDGE_FLOOR_GATE"],
        config_snapshot={"regime_signal_profile": "conservative"},
    )

    assert record.metrics is not None
    assert record.metrics["metadata_quality_status"] == "partial"
    assert record.metrics["metadata_missing_reasons"]["quote_age_ms"] == (
        "quote_snapshot_error"
    )
    assert record.metrics["metadata_missing_reasons"]["spread_bps"] == (
        "quote_snapshot_error"
    )
    assert record.metrics["order_type"] == "not_submitted"


def _lineage_test_recorder(written: list[object]) -> DecisionRecorder:
    return DecisionRecorder(
        runtime=type("Runtime", (), {"execution_candidate_rank_expected_edge_bps": {}})(),
        path="runtime/decision_records.jsonl",
        write_impl=lambda record, path: written.append((record, path)),
        dedupe_gate_root_causes=lambda gates: list(dict.fromkeys(gates)),
        session_bucket_from_ts=lambda _ts: "rth",
        safe_float=lambda value: float(value) if value is not None else None,
    )


def _lineage_test_proposal(
    *,
    bar_ts: datetime,
    ml_influenced: bool,
) -> SleeveProposal:
    return SleeveProposal(
        symbol="AAPL",
        sleeve="day",
        bar_ts=bar_ts,
        target_dollars=1000.0,
        expected_edge_bps=12.0,
        expected_cost_bps=3.0,
        score=0.7,
        confidence=0.8,
        debug={
            "ml_influenced": ml_influenced,
            "model_lineage": {
                "model_id": "ml-edge-1",
                "model_version": "v2026.07.10",
                "dataset_hash": "dataset-1",
                "feature_version": "features-1",
                "model_artifact_hash": "artifact-1",
            },
        },
    )


def test_decision_recorder_propagates_ml_lineage_to_accepted_decision() -> None:
    written: list[object] = []
    recorder = _lineage_test_recorder(written)
    bar_ts = datetime(2026, 7, 10, 14, 30, tzinfo=UTC)
    proposal = _lineage_test_proposal(bar_ts=bar_ts, ml_influenced=True)
    target = NettedTarget(
        symbol="AAPL",
        bar_ts=bar_ts,
        target_dollars=1000.0,
        target_shares=5.0,
        proposals=[proposal],
    )

    record = recorder.record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=target,
        sleeves=[proposal],
        gates=["OK_TRADE"],
        order={"side": "buy", "qty": 5, "status": "accepted"},
    )

    journal = record.to_dict()["decision_journal"]
    assert journal["model_id"] == "ml-edge-1"
    assert journal["model_version"] == "v2026.07.10"
    assert journal["dataset_hash"] == "dataset-1"
    assert journal["feature_version"] == "features-1"
    assert journal["model_artifact_hash"] == "artifact-1"
    assert journal["correlation_id"] == record.correlation_id
    assert journal["metadata"]["opportunity_eligible"] is True


def test_decision_recorder_propagates_ml_lineage_to_blocked_no_order_decision() -> None:
    written: list[object] = []
    recorder = _lineage_test_recorder(written)
    bar_ts = datetime(2026, 7, 10, 14, 35, tzinfo=UTC)
    proposal = _lineage_test_proposal(bar_ts=bar_ts, ml_influenced=True)
    target = NettedTarget(
        symbol="AAPL",
        bar_ts=bar_ts,
        target_dollars=0.0,
        target_shares=0.0,
        proposals=[proposal],
    )

    record = recorder.record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=target,
        sleeves=[proposal],
        gates=["NET_EDGE_FLOOR_GATE"],
    )

    journal = record.to_dict()["decision_journal"]
    assert journal["accepted"] is False
    assert journal["submitted"] is False
    assert journal["order_intent"] is None
    assert journal["model_id"] == "ml-edge-1"
    assert journal["model_artifact_hash"] == "artifact-1"


def test_decision_recorder_does_not_attribute_heuristic_proposal_to_ml() -> None:
    written: list[object] = []
    recorder = _lineage_test_recorder(written)
    bar_ts = datetime(2026, 7, 10, 14, 40, tzinfo=UTC)
    proposal = _lineage_test_proposal(bar_ts=bar_ts, ml_influenced=False)
    target = NettedTarget(
        symbol="AAPL",
        bar_ts=bar_ts,
        target_dollars=1000.0,
        target_shares=5.0,
        proposals=[proposal],
    )

    record = recorder.record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=target,
        sleeves=[proposal],
        gates=["OK_TRADE"],
    )

    journal = record.to_dict()["decision_journal"]
    for key in (
        "model_id",
        "model_version",
        "dataset_hash",
        "feature_version",
        "model_artifact_hash",
    ):
        assert journal[key] is None
