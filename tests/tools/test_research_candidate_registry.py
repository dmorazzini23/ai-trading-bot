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
                    "mean_post_cost_net_edge_bps": 2.5,
                    "profitable_fold_ratio": 0.8,
                    "stability_score": 0.7,
                    "trades": 300,
                    "evidence_qualified": True,
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
    assert entry["governance"]["promotion_authority"] is False
    assert first["promotion_authority"] is False
    assert first["live_money_authority"] is False
