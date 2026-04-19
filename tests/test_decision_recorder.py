from __future__ import annotations

from datetime import UTC, datetime

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
