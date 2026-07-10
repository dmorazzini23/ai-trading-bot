from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.replay.replay_engine import ReplayConfig, ReplayEngine


def _pipeline(event: dict[str, object]) -> dict[str, object]:
    close_raw = event["close"]
    close = float(close_raw) if isinstance(close_raw, (int, float)) else 0.0
    return {
        "symbol": event["symbol"],
        "ts": event["ts"],
        "sleeve": "day",
        "regime_profile": "balanced",
        "order": {
            "side": "buy",
            "qty": 10,
            "price": close,
            "order_type": "limit",
            "client_order_id": f"{event['symbol']}-{event['ts']}",
        },
    }


def test_replay_engine_is_deterministic() -> None:
    config = ReplayConfig(symbols=("AAPL",), seed=42)
    bars = [
        {"symbol": "AAPL", "ts": "2025-01-01T10:00:00Z", "close": 100.0, "mid": 100.0},
        {"symbol": "AAPL", "ts": "2025-01-01T10:05:00Z", "close": 101.0, "mid": 101.0},
    ]
    engine = ReplayEngine(config, pipeline=_pipeline)
    first = engine.run(bars)
    second = engine.run(bars)
    assert first == second


def test_replay_engine_next_bar_fill_avoids_same_bar_leakage() -> None:
    config = ReplayConfig(symbols=("AAPL",), seed=42, fill_slippage_bps=0.0)
    bars = [
        {"symbol": "AAPL", "ts": "2025-01-01T10:00:00Z", "close": 100.0, "mid": 100.0},
        {"symbol": "AAPL", "ts": "2025-01-01T10:05:00Z", "close": 101.0, "mid": 101.25},
    ]

    result = ReplayEngine(config, pipeline=_pipeline).run(bars)

    filled_entries = [row for row in result["ledger_entries"] if row.get("status", "filled") == "filled"]
    assert len(filled_entries) == 1
    assert any(row.get("status") == "canceled" for row in result["ledger_entries"])
    assert filled_entries[0]["price"] == pytest.approx(101.25)
    tca = result["tca_records"][0]
    assert tca["decision_price"] == pytest.approx(100.0)
    assert tca["fill_price"] == pytest.approx(101.25)
    assert tca["fill_latency_ms"] == 300_000
    assert tca["benchmark"]["decision_ts"] == "2025-01-01T10:00:00+00:00"
    assert tca["benchmark"]["first_fill_ts"] == "2025-01-01T10:05:00+00:00"


def test_replay_engine_enforces_oms_gate_blocks() -> None:
    config = ReplayConfig(symbols=("AAPL",), seed=42, fill_model="close")

    def blocked_pipeline(event: dict[str, object]) -> dict[str, object]:
        decision = _pipeline(event)
        decision["gates"] = ["OK_TRADE", "RISK_PORTFOLIO_HARD_BLOCK"]
        return decision

    result = ReplayEngine(config, pipeline=blocked_pipeline).run(
        [{"symbol": "AAPL", "ts": "2025-01-01T10:00:00Z", "close": 100.0}]
    )

    assert len(result["decision_records"]) == 1
    assert result["ledger_entries"] == []
    assert result["tca_records"] == []


def test_replay_engine_blocks_real_broker_submit() -> None:
    config = ReplayConfig(symbols=("AAPL",), seed=42)
    engine = ReplayEngine(config, pipeline=_pipeline, broker_submit=lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="must not execute real broker submits"):
        engine.run([])


def test_replay_engine_uses_conservative_live_cost_and_preserves_lineage() -> None:
    now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
    live_cost_model = {
        "generated_at": now.isoformat(),
        "by_symbol_side_session_order_type_volatility": [
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "order_type": "limit",
                "volatility_bucket": "normal",
                "sample_count": 12,
                "sufficient_samples": True,
                "p90_total_cost_bps": 15.0,
                "last_observed_at": now.isoformat(),
            }
        ],
    }
    config = ReplayConfig(
        symbols=("AAPL",),
        fill_slippage_bps=5.0,
        fill_fee_bps=0.0,
        live_cost_model=live_cost_model,
        cost_alignment_now=now,
    )

    def lineage_pipeline(event: dict[str, object]) -> dict[str, object]:
        decision = _pipeline(event)
        decision.update(
            {
                "session_regime": "midday",
                "volatility_bucket": "normal",
                "prediction_id": "prediction-a",
                "decision_id": "decision-a",
                "model_id": "model-a",
                "model_version": "v4",
                "model_artifact_hash": "sha256:abc",
                "feature_version": "features-v2",
                "required_bar_timeframe": "5Min",
            }
        )
        return decision

    bars = [
        {"symbol": "AAPL", "ts": "2025-01-01T10:00:00Z", "close": 100.0, "mid": 100.0},
        {"symbol": "AAPL", "ts": "2025-01-01T10:05:00Z", "close": 101.0, "mid": 101.0},
    ]
    engine = ReplayEngine(config, pipeline=lineage_pipeline)

    first = engine.run(bars)
    second = engine.run(bars)

    assert first == second
    filled = [
        row
        for row in first["ledger_entries"]
        if row.get("status", "filled") == "filled"
    ][0]
    assert filled["price"] == pytest.approx(101.1515)
    assert filled["prediction_id"] == "prediction-a"
    assert filled["model_id"] == "model-a"
    assert filled["model_artifact_hash"] == "sha256:abc"
    assert filled["feature_version"] == "features-v2"
    replay_cost = filled["replay_cost"]
    assert replay_cost["source"] == "live"
    assert replay_cost["alignment"] == "pessimism"
    assert replay_cost["resolved_cost_bps"] == pytest.approx(15.0)
    diagnostics = first["cost_diagnostics"]
    assert diagnostics["source_counts"] == {"fallback": 0, "fixed": 0, "live": 1}
    assert diagnostics["max_resolved_cost_bps"] == pytest.approx(15.0)
    tca = first["tca_records"][0]
    assert tca["model_id"] == "model-a"
    assert tca["model_version"] == "v4"
    assert tca["required_bar_timeframe"] == "5Min"
    assert tca["replay_cost"]["source"] == "live"


@pytest.mark.parametrize(
    ("generated_delta", "sample_count", "observed_cost", "alignment"),
    [
        (timedelta(0), 10, 2.0, "optimism"),
        (timedelta(0), 2, 30.0, "insufficient_samples"),
        (timedelta(days=-2), 10, 30.0, "stale"),
    ],
)
def test_replay_engine_falls_back_for_unusable_live_cost(
    generated_delta: timedelta,
    sample_count: int,
    observed_cost: float,
    alignment: str,
) -> None:
    now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
    observed_at = now + generated_delta
    live_cost_model = {
        "generated_at": observed_at.isoformat(),
        "by_symbol_side_session_order_type_volatility": [
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "order_type": "limit",
                "volatility_bucket": "normal",
                "sample_count": sample_count,
                "sufficient_samples": True,
                "p90_total_cost_bps": observed_cost,
                "last_observed_at": observed_at.isoformat(),
            }
        ],
    }
    config = ReplayConfig(
        symbols=("AAPL",),
        fill_slippage_bps=5.0,
        live_cost_model=live_cost_model,
        cost_alignment_now=now,
        cost_max_age_seconds=600.0,
    )

    def contextual_pipeline(event: dict[str, object]) -> dict[str, object]:
        decision = _pipeline(event)
        decision["session_regime"] = "midday"
        decision["volatility_bucket"] = "normal"
        return decision

    result = ReplayEngine(config, pipeline=contextual_pipeline).run(
        [
            {"symbol": "AAPL", "ts": "2025-01-01T10:00:00Z", "close": 100.0, "mid": 100.0},
            {"symbol": "AAPL", "ts": "2025-01-01T10:05:00Z", "close": 101.0, "mid": 101.0},
        ]
    )

    cost = result["cost_diagnostics"]["items"][0]
    assert cost["source"] == "fallback"
    assert cost["alignment"] == alignment
    assert cost["resolved_cost_bps"] == pytest.approx(5.0)
    filled = [
        row
        for row in result["ledger_entries"]
        if row.get("status", "filled") == "filled"
    ][0]
    assert filled["price"] == pytest.approx(101.0505)
