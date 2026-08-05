from __future__ import annotations

import json
from pathlib import Path

from ai_trading.model_registry import ModelRegistry
from ai_trading.tools.research_candidate_registry import (
    register_research_candidates,
)


def _write_tournament(tmp_path: Path) -> Path:
    model = tmp_path / "candidate.joblib"
    manifest = tmp_path / "candidate.manifest.json"
    model.write_bytes(b"candidate-model")
    manifest.write_text("{}", encoding="utf-8")
    report = {
        "artifact_type": "multi_horizon_research_report",
        "generated_at": "2026-08-03T20:15:00Z",
        "config": {"symbols": "AAPL,MSFT"},
        "replay_config": {},
        "successive_halving": {"eta": 2},
        "promotion_authority": False,
        "live_money_authority": False,
        "candidates": [
            {
                "model_name": "one_bar",
                "model_type": "logistic",
                "model_path": str(model),
                "manifest_path": str(manifest),
                "horizon_bars": 1,
                "label_objective": "risk_adjusted",
                "governance_status": "shadow",
                "promotion_authority": False,
                "live_money_authority": False,
                "dataset": {"dataset_hash": "dataset-001"},
                "walk_forward": {
                    "aggregate": {
                        "mean_post_cost_net_edge_bps": 2.5,
                        "profitable_fold_ratio": 0.8,
                        "stability_score": 0.7,
                        "trades": 300,
                        "evidence_qualified": True,
                    }
                },
            }
        ],
    }
    path = tmp_path / "tournament.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_register_research_candidates_is_idempotent_and_shadow_only(
    tmp_path: Path,
) -> None:
    report_path = _write_tournament(tmp_path)
    registry_dir = tmp_path / "registry"

    first = register_research_candidates(
        report_path=report_path,
        registry_dir=registry_dir,
        output_json=tmp_path / "first.json",
    )
    second = register_research_candidates(
        report_path=report_path,
        registry_dir=registry_dir,
        output_json=tmp_path / "second.json",
    )

    assert first["registered"][0]["created"] is True
    assert second["registered"][0]["created"] is False
    registry = ModelRegistry(registry_dir)
    assert len(registry.model_index) == 1
    entry = next(iter(registry.model_index.values()))
    assert entry["governance"]["status"] == "shadow"
    assert entry["governance"]["metrics"] == {
        "mean_post_cost_net_edge_bps": 2.5,
        "profitable_fold_ratio": 0.8,
        "stability_score": 0.7,
        "trades": 300,
        "evidence_qualified": True,
        "generated_at": "2026-08-03T20:15:00Z",
    }
    assert entry["governance"]["promotion_authority"] is False
    assert first["promotion_authority"] is False
    assert first["live_money_authority"] is False


def test_repeat_registration_repairs_same_shadow_candidate_metrics(
    tmp_path: Path,
) -> None:
    report_path = _write_tournament(tmp_path)
    registry_dir = tmp_path / "registry"
    register_research_candidates(
        report_path=report_path,
        registry_dir=registry_dir,
        output_json=tmp_path / "first.json",
    )
    registry = ModelRegistry(registry_dir)
    model_id = next(iter(registry.model_index))
    registry.model_index[model_id]["governance"]["metrics"] = {
        "mean_post_cost_net_edge_bps": None,
        "trades": None,
    }
    registry.model_index[model_id]["metadata"]["metrics"] = {
        "mean_post_cost_net_edge_bps": None,
        "trades": None,
    }
    registry._save_index()  # noqa: SLF001

    result = register_research_candidates(
        report_path=report_path,
        registry_dir=registry_dir,
        output_json=tmp_path / "repaired.json",
    )

    assert result["registered"] == [
        {"candidate": "one_bar", "model_id": model_id, "created": False}
    ]
    repaired = ModelRegistry(registry_dir).model_index[model_id]
    assert repaired["governance"]["metrics"]["mean_post_cost_net_edge_bps"] == 2.5
    assert repaired["governance"]["metrics"]["trades"] == 300
    assert repaired["governance"]["metrics"]["generated_at"] == (
        "2026-08-03T20:15:00Z"
    )
    assert repaired["metadata"]["metrics"] == repaired["governance"]["metrics"]


def test_candidate_generated_at_overrides_tournament_timestamp(tmp_path: Path) -> None:
    report_path = _write_tournament(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["candidates"][0]["generated_at"] = "2026-08-03T21:30:00Z"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    register_research_candidates(
        report_path=report_path,
        registry_dir=tmp_path / "registry",
        output_json=tmp_path / "registered.json",
    )

    registry = ModelRegistry(tmp_path / "registry")
    entry = next(iter(registry.model_index.values()))
    assert entry["governance"]["metrics"]["generated_at"] == (
        "2026-08-03T21:30:00Z"
    )


def test_repeat_registration_never_refreshes_production_metrics(
    tmp_path: Path,
) -> None:
    report_path = _write_tournament(tmp_path)
    registry_dir = tmp_path / "registry"
    register_research_candidates(
        report_path=report_path,
        registry_dir=registry_dir,
        output_json=tmp_path / "first.json",
    )
    registry = ModelRegistry(registry_dir)
    model_id = next(iter(registry.model_index))
    registry.update_governance_status(model_id, "production")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["candidates"][0]["walk_forward"]["aggregate"][
        "mean_post_cost_net_edge_bps"
    ] = 99.0
    report_path.write_text(json.dumps(report), encoding="utf-8")

    register_research_candidates(
        report_path=report_path,
        registry_dir=registry_dir,
        output_json=tmp_path / "repeat.json",
    )

    production = ModelRegistry(registry_dir).model_index[model_id]
    assert production["governance"]["status"] == "production"
    assert production["governance"]["metrics"][
        "mean_post_cost_net_edge_bps"
    ] == 2.5


def test_blocked_accelerator_is_governed_no_candidate_report(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "training_accelerator_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "artifact_type": "training_accelerator_report",
                "generated_at": "2026-08-04T10:00:00Z",
                "status": "blocked",
                "blocked_reasons": ["required_live_cost_model_unusable"],
                "input_signature": "blocked-signature",
                "promotion_authority": False,
                "runtime_authority": False,
                "live_money_authority": False,
            }
        ),
        encoding="utf-8",
    )
    registry_dir = tmp_path / "registry"

    result = register_research_candidates(
        report_path=report_path,
        registry_dir=registry_dir,
        output_json=tmp_path / "registered.json",
    )

    assert result["registered"] == []
    assert result["skipped"] == [
        {
            "candidate": "training_accelerator_report",
            "reason": "source_blocked:required_live_cost_model_unusable",
        }
    ]
    assert result["promotion_authority"] is False
    assert result["runtime_authority"] is False
    assert result["live_money_authority"] is False
    assert ModelRegistry(registry_dir).model_index == {}
