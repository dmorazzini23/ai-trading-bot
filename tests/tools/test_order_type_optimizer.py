from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trading.tools import order_type_optimizer


def _cost_model(generated_at: str) -> dict[str, object]:
    return {
        "generated_at": generated_at,
        "status": {"status": "ready", "available": True},
        "by_symbol_side_session_order_type_volatility": [
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "order_type": "market",
                "volatility_bucket": "tight_spread",
                "sample_count": 10,
                "sufficient_samples": True,
                "p90_total_cost_bps": 8.0,
            },
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "order_type": "limit",
                "volatility_bucket": "tight_spread",
                "sample_count": 10,
                "sufficient_samples": True,
                "p90_total_cost_bps": 3.0,
            },
        ],
    }


def test_order_type_optimizer_recommends_lowest_shadow_cost() -> None:
    now = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)
    report = order_type_optimizer.build_order_type_optimizer(
        candidates=[
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "order_type_options": ["market", "limit"],
                "spread_bps": 2.0,
                "quote_age_ms": 100.0,
            }
        ],
        live_cost_model=_cost_model(now.isoformat()),
        launch_profile_name="live_canary",
        now=now,
    )

    assert report["status"] == "ready"
    assert report["mode"] == "research_shadow"
    assert report["live_enabled"] is False
    assert report["recommendations"][0]["recommended_order_type"] == "limit"
    assert report["recommendations"][0]["shadow_only"] is True


def test_order_type_optimizer_blocks_stale_cost_model() -> None:
    now = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)
    stale = (now - timedelta(hours=30)).isoformat()

    report = order_type_optimizer.build_order_type_optimizer(
        candidates=[{"symbol": "AAPL", "side": "buy", "session_regime": "midday"}],
        live_cost_model=_cost_model(stale),
        max_cost_model_age_hours=24.0,
        now=now,
    )

    assert report["status"] == "blocked"
    assert report["recommendations_enabled"] is False
    assert report["recommendations"] == []
    assert "live_cost_model_stale" in report["reasons"]


def test_order_type_optimizer_cli_writes_artifact(tmp_path: Path) -> None:
    now = datetime.now(UTC).isoformat()
    model_path = tmp_path / "cost.json"
    candidates_path = tmp_path / "candidates.jsonl"
    output = tmp_path / "optimizer.json"
    model_path.write_text(json.dumps(_cost_model(now)), encoding="utf-8")
    candidates_path.write_text(
        json.dumps(
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "order_type_options": ["market", "limit"],
                "spread_bps": 2.0,
                "quote_age_ms": 100.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = order_type_optimizer.main(
        [
            "--candidates-jsonl",
            str(candidates_path),
            "--live-cost-model-json",
            str(model_path),
            "--output-json",
            str(output),
            "--launch-profile",
            "live_canary",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "order_type_optimizer"
    assert payload["recommendations"][0]["recommended_order_type"] == "limit"

