from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib

from ai_trading.models.artifacts import write_artifact_manifest
from ai_trading.tools import promotion_pipeline


def _write_model(path: Path) -> Path:
    joblib.dump({"model": "candidate"}, path)
    return write_artifact_manifest(
        model_path=path,
        model_version="candidate-v1",
        metadata={"strategy": "replay_aligned"},
    )


def _replay_payload(expectancy_bps: float = 5.0) -> dict[str, object]:
    return {
        "artifact_type": "offline_replay_summary",
        "generated_at": "2026-05-05T20:00:00Z",
        "aggregate": {
            "expectancy_bps": expectancy_bps,
            "total_trades": 25,
            "violation_count": 0,
        },
    }


def _shadow_payload() -> dict[str, object]:
    return {
        "generated_at": "2026-05-05T20:00:00Z",
        "sample_gate": {"status": "review_eligible"},
        "decision_summary": {"rows": 150, "skew_breach_rate": 0.0},
        "markout_summary": {"challenger_mean_net_markout_bps": 3.5},
    }


def test_promotion_report_passes_when_all_hard_gates_pass(tmp_path: Path) -> None:
    model = tmp_path / "candidate.joblib"
    manifest = _write_model(model)

    report = promotion_pipeline.build_promotion_report(
        model_path=model,
        manifest_path=manifest,
        full_replay=_replay_payload(),
        tail_replay=_replay_payload(),
        recent_replay=_replay_payload(),
        shadow_report=_shadow_payload(),
        live_cost_model={
            "generated_at": "2026-05-05T20:00:00Z",
            "status": {"available": True, "status": "ready", "breach_count": 0},
        },
        runtime_decay_controls={
            "generated_at": "2026-05-05T20:00:00Z",
            "actions": {"entries_allowed": True, "max_action": "reduce_size", "size_scale": 0.85}
        },
        generated_at=datetime(2026, 5, 5, 21, 0, tzinfo=UTC),
        current_champion_path="/var/lib/ai-trading-bot/models/current.joblib",
    )

    assert report["promotion_ready"] is True
    assert report["status"] == "ready_for_approval"
    assert all(report["gates"].values())
    assert report["manifest"]["reason"] == "OK"
    assert report["rollback"]["current_champion_path"] == (
        "/var/lib/ai-trading-bot/models/current.joblib"
    )


def test_promotion_report_blocks_bad_replay_and_missing_shadow(tmp_path: Path) -> None:
    model = tmp_path / "candidate.joblib"
    manifest = _write_model(model)

    report = promotion_pipeline.build_promotion_report(
        model_path=model,
        manifest_path=manifest,
        full_replay=_replay_payload(expectancy_bps=-1.0),
        tail_replay=_replay_payload(),
        recent_replay=_replay_payload(),
        shadow_report={},
        live_cost_model={
            "generated_at": "2026-05-05T20:00:00Z",
            "status": {"available": True, "status": "ready", "breach_count": 0},
        },
        runtime_decay_controls={
            "generated_at": "2026-05-05T20:00:00Z",
            "actions": {"entries_allowed": True, "max_action": "normal"},
        },
        generated_at=datetime(2026, 5, 5, 21, 0, tzinfo=UTC),
    )

    assert report["promotion_ready"] is False
    assert report["status"] == "blocked"
    assert report["gates"]["full_replay_positive"] is False
    assert report["gates"]["shadow_telemetry_acceptable"] is False


def test_promotion_report_blocks_warming_live_cost_and_stale_evidence(
    tmp_path: Path,
) -> None:
    model = tmp_path / "candidate.joblib"
    manifest = _write_model(model)

    report = promotion_pipeline.build_promotion_report(
        model_path=model,
        manifest_path=manifest,
        full_replay=_replay_payload(),
        tail_replay=_replay_payload(),
        recent_replay=_replay_payload(),
        shadow_report=_shadow_payload(),
        live_cost_model={
            "generated_at": "2026-05-01T20:00:00Z",
            "status": {"available": True, "status": "warming_up", "breach_count": 0},
        },
        runtime_decay_controls={
            "generated_at": "2026-05-05T20:00:00Z",
            "actions": {"entries_allowed": True, "max_action": "normal"},
        },
        generated_at=datetime(2026, 5, 5, 21, 0, tzinfo=UTC),
        max_evidence_age_hours=24.0,
    )

    assert report["promotion_ready"] is False
    assert report["gates"]["live_cost_model_acceptable"] is False
    assert report["gates"]["evidence_fresh"] is False
    assert report["live_cost_model"]["status_ready"] is False
    assert report["evidence_freshness"]["live_cost_model"]["reason"] == "evidence_stale"


def test_promotion_report_rejects_replay_governance_schema_for_replay_evidence(
    tmp_path: Path,
) -> None:
    model = tmp_path / "candidate.joblib"
    manifest = _write_model(model)

    report = promotion_pipeline.build_promotion_report(
        model_path=model,
        manifest_path=manifest,
        full_replay={
            "status": "ok",
            "now": "2026-05-05T20:00:00Z",
            "replay": {"ts": "2026-05-05T20:00:00Z", "fill_events": 10},
        },
        tail_replay=_replay_payload(),
        recent_replay=_replay_payload(),
        shadow_report=_shadow_payload(),
        live_cost_model={
            "generated_at": "2026-05-05T20:00:00Z",
            "status": {"available": True, "status": "ready", "breach_count": 0},
        },
        runtime_decay_controls={
            "generated_at": "2026-05-05T20:00:00Z",
            "actions": {"entries_allowed": True, "max_action": "normal"},
        },
        generated_at=datetime(2026, 5, 5, 21, 0, tzinfo=UTC),
    )

    assert report["promotion_ready"] is False
    assert report["replay"]["full"]["reason"] == "unsupported_replay_schema"
    assert report["gates"]["full_replay_positive"] is False


def test_promotion_pipeline_cli_writes_report_and_returns_blocked(tmp_path: Path) -> None:
    model = tmp_path / "candidate.joblib"
    manifest = _write_model(model)
    replay = tmp_path / "full_replay.json"
    shadow = tmp_path / "shadow.json"
    live_cost = tmp_path / "live_cost.json"
    decay = tmp_path / "decay.json"
    output = tmp_path / "promotion_report.json"
    replay.write_text(json.dumps(_replay_payload(expectancy_bps=-2.0)), encoding="utf-8")
    shadow.write_text(json.dumps(_shadow_payload()), encoding="utf-8")
    live_cost.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-05T20:00:00Z",
                "status": {"available": True, "status": "ready", "breach_count": 0},
            }
        ),
        encoding="utf-8",
    )
    decay.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-05T20:00:00Z",
                "actions": {"entries_allowed": True, "max_action": "normal"},
            }
        ),
        encoding="utf-8",
    )

    exit_code = promotion_pipeline.main(
        [
            "--model-path",
            str(model),
            "--manifest-path",
            str(manifest),
            "--full-replay-json",
            str(replay),
            "--shadow-report-json",
            str(shadow),
            "--live-cost-model-json",
            str(live_cost),
            "--runtime-decay-controls-json",
            str(decay),
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "model_promotion_report"
    assert payload["promotion_ready"] is False
