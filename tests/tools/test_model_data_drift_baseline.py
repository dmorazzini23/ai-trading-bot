from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ai_trading.tools import model_data_drift_baseline


def _rows(now: datetime, count: int = 30) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fills: list[dict[str, object]] = []
    tca: list[dict[str, object]] = []
    for index in range(count):
        timestamp = (now - timedelta(minutes=index)).isoformat()
        symbol = "AAPL" if index % 2 == 0 else "AMZN"
        fills.append(
            {
                "ts": timestamp,
                "symbol": symbol,
                "confidence": 0.55 + (index % 5) * 0.02,
                "expected_net_edge_bps": 3.0 + (index % 3),
                "realized_net_edge_bps": 1.5 if index % 3 else -0.5,
                "slippage_bps": 0.5 + (index % 4) * 0.1,
                "fee_bps": 0.1,
            }
        )
        tca.append(
            {
                "ts": timestamp,
                "symbol": symbol,
                "provider": "alpaca",
                "market_regime": "sideways" if index % 2 == 0 else "downtrend",
                "fill_latency_ms": 100.0 + index,
                "spread_paid_bps": 1.0 + (index % 4) * 0.1,
                "decision_quote_age_ms": 50.0 + index,
            }
        )
    return fills, tca


def test_build_governed_baseline_requires_complete_evidence() -> None:
    now = datetime(2026, 7, 18, 5, 0, tzinfo=UTC)
    fills, tca = _rows(now)
    evidence = model_data_drift_baseline.build_model_data_drift_evidence(
        fills=fills,
        tca_rows=tca,
        generated_at=now,
        min_samples=25,
        model_id="shadow-1",
        model_hash="abc123",
    )

    baseline = model_data_drift_baseline.build_governed_drift_baseline(
        evidence,
        baseline_id="shadow-1-20260718",
        approved_by="operator",
        approved_at=now,
    )

    assert evidence["status"] == "ready"
    assert evidence["coverage"]["complete"] is True
    assert baseline["artifact_type"] == "model_data_drift_baseline"
    assert baseline["status"] == "approved"
    assert baseline["approval"]["automatic_roll_forward"] is False
    assert baseline["promotion_authority"] is False
    assert baseline["live_money_authority"] is False


def test_governed_baseline_rejects_incomplete_category_coverage() -> None:
    now = datetime(2026, 7, 18, 5, 0, tzinfo=UTC)
    fills, tca = _rows(now, count=10)
    evidence = model_data_drift_baseline.build_model_data_drift_evidence(
        fills=fills,
        tca_rows=tca,
        generated_at=now,
        min_samples=25,
    )

    with pytest.raises(ValueError, match="baseline_evidence_incomplete"):
        model_data_drift_baseline.build_governed_drift_baseline(
            evidence,
            baseline_id="too-small",
            approved_by="operator",
            approved_at=now,
        )


def test_baseline_cli_records_sources_and_refuses_overwrite(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    fills, tca = _rows(now)
    fills_path = tmp_path / "fills.jsonl"
    tca_path = tmp_path / "tca.jsonl"
    output = tmp_path / "baseline.json"
    fills_path.write_text("".join(json.dumps(row) + "\n" for row in fills), encoding="utf-8")
    tca_path.write_text("".join(json.dumps(row) + "\n" for row in tca), encoding="utf-8")
    args = [
        "--fills-jsonl",
        str(fills_path),
        "--tca-jsonl",
        str(tca_path),
        "--output-json",
        str(output),
        "--model-id",
        "shadow-1",
        "--baseline-id",
        "shadow-1-baseline",
        "--approved-by",
        "operator",
    ]

    assert model_data_drift_baseline.main(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "approved"
    assert len(payload["sources"]) == 2
    assert all(len(source["sha256"]) == 64 for source in payload["sources"])
    with pytest.raises(SystemExit) as exc_info:
        model_data_drift_baseline.main(args)
    assert exc_info.value.code == 2


def test_current_evidence_cli_binds_viable_shadow_registry_identity(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    fills, tca = _rows(now)
    fills_path = tmp_path / "fills.jsonl"
    tca_path = tmp_path / "tca.jsonl"
    registry_path = tmp_path / "registry_evaluation.json"
    output = tmp_path / "current.json"
    fills_path.write_text("".join(json.dumps(row) + "\n" for row in fills), encoding="utf-8")
    tca_path.write_text("".join(json.dumps(row) + "\n" for row in tca), encoding="utf-8")
    registry_path.write_text(
        json.dumps(
            {
                "status": "blocked",
                "blocked_reasons": ["champion_artifact_missing"],
                "active_champion": None,
                "active_challenger": {
                    "model_id": "one-bar-shadow",
                    "model_hash": "abc123",
                    "dataset_fingerprint": "dataset456",
                    "artifact_viability": {
                        "ok": True,
                        "path": str(tmp_path / "one-bar.joblib"),
                    },
                },
                "promotion_authority": False,
            }
        ),
        encoding="utf-8",
    )

    assert (
        model_data_drift_baseline.main(
            [
                "--fills-jsonl",
                str(fills_path),
                "--tca-jsonl",
                str(tca_path),
                "--model-registry-json",
                str(registry_path),
                "--output-json",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "model_data_drift_evidence"
    assert payload["model"] == {
        "model_id": "one-bar-shadow",
        "model_hash": "abc123",
        "dataset_hash": "dataset456",
        "registry_role": "challenger",
    }
    assert len(payload["sources"]) == 3
    assert payload["promotion_authority"] is False
    assert payload["live_money_authority"] is False
