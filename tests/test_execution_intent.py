from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.core.execution_intent import build_execution_intent_context


def test_build_execution_intent_context_builds_intent_and_annotations() -> None:
    bar_ts = datetime(2026, 4, 19, 14, 30, tzinfo=UTC)

    context = build_execution_intent_context(
        salt="seed-1",
        symbol="AAPL",
        side="buy",
        delta_shares=5,
        price=100.0,
        bar_ts=bar_ts,
        spread_bps=12.0,
        liquidity_bucket="NORMAL",
        quote_quality_ok=True,
        sector="TECH",
        event_risk=True,
        slo_derisk_details={
            "rolling_volume": 15000.0,
            "execution_drift_bps": 2.5,
            "reject_rate_pct": 1.1,
        },
        config_snapshot={
            "config_snapshot_hash": "cfg-1",
            "effective_policy_hash": "policy-1",
        },
        execution_model_lineage={
            "model_id": "ml-main",
            "model_version": "v1",
            "dataset_hash": "ds-1",
            "feature_version": "fv-1",
            "model_artifact_hash": "artifact-1",
        },
        submit_quote_source="iex",
        submit_bid_at_arrival=99.9,
        submit_ask_at_arrival=100.1,
        submit_mid_at_arrival=100.0,
    )

    assert context.client_order_id
    assert context.decision_trace_id == context.client_order_id
    assert context.pretrade_intent.symbol == "AAPL"
    assert context.pretrade_intent.qty == 5
    assert context.pretrade_intent.event_type == "earnings"
    assert context.pretrade_intent.quote_quality_ok is True
    assert context.order_lineage_metadata["model_id"] == "ml-main"
    assert context.order_lineage_metadata["policy_hash"] == "policy-1"
    assert context.order_lineage_metadata["price_source"] == "iex"
    assert context.order_annotations["decision_trace_id"] == context.client_order_id
    assert context.order_annotations["quote_source"] == "broker_nbbo"
    assert context.order_annotations["quote"]["midpoint"] == 100.0
