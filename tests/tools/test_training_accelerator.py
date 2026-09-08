from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ai_trading.tools import training_accelerator
from ai_trading.tools.regime_champion_models import build_regime_champion_report


def test_cli_resolves_manifest_before_legacy_directory_preflight(tmp_path, monkeypatch):
    governed = tmp_path / "governed"
    governed.mkdir()
    monkeypatch.setattr(training_accelerator, "_resolve_training_input", lambda args: (governed, {"quality_passed": True}))
    def run(args):
        assert args.data_dir == governed
        assert args.require_validated_data
        return {"status": "no_valid_candidates", "path": str(tmp_path / "report.json")}
    monkeypatch.setattr(training_accelerator, "run_training_accelerator", run)
    assert training_accelerator.main(["--data-dir", str(tmp_path / "missing_legacy"), "--acquisition-manifest-json", str(tmp_path / "acquisition.json"), "--require-validated-data"]) == 0


def test_cli_invalid_manifest_cannot_fall_back_to_existing_csvs(tmp_path, monkeypatch):
    def invalid(args):
        raise ValueError("dataset content hash mismatch")
    monkeypatch.setattr(training_accelerator, "_resolve_training_input", invalid)
    monkeypatch.setattr(training_accelerator, "run_training_accelerator", lambda args: pytest.fail("must not train"))
    assert training_accelerator.main(["--data-dir", str(tmp_path), "--acquisition-manifest-json", str(tmp_path / "bad.json"), "--require-validated-data"]) == 2


def test_real_training_pipeline_candidate_reaches_registry_selector(tmp_path: Path) -> None:
    import numpy as np
    import pandas as pd

    bars = tmp_path / "bars"
    bars.mkdir()
    x = np.linspace(0, 100, 1600)
    close = 100 + 2 * np.sin(x) + 0.3 * np.sin(0.3 * x)
    pd.DataFrame({
        "timestamp": pd.date_range("2026-08-03T13:30:00Z", periods=len(x), freq="min"),
        "open": close, "high": close + 0.1, "low": close - 0.1,
        "close": close, "volume": 12000 + 500 * np.cos(x),
    }).to_csv(bars / "AAPL.csv", index=False)
    report = training_accelerator.run_training_accelerator(argparse.Namespace(
        cadence="daily", data_dir=bars, symbols="AAPL", output_dir=tmp_path / "out",
        training_cache_dir=tmp_path / "cache", model_type="logistic", model_types="logistic",
        horizons="1", label_objectives="net_markout", lead_horizon_bars=1,
        max_candidates=1, screening_folds=2, walk_forward_folds=2,
        fee_bps=0.0, slippage_bps=0.0, plan_only=False,
        max_replay_candidates=0, research_experiments=False,
    ))
    assert report["ranked_candidate_count"] == 1
    candidate = report["candidates"][0]
    assert candidate["model_id"]
    assert list((tmp_path / "out" / "multi_horizon" / "models").glob("*.joblib"))
    selection = build_regime_champion_report(candidates=report)
    assert len(selection["decisions"]) == 1
    decision = selection["decisions"][0]
    assert decision["candidate_model_id"] == candidate["model_id"]
    assert decision["samples"] == candidate["sample_count"]
    assert candidate["sample_count"] > 0
    assert "candidate_model_id_missing" not in decision["reasons"]


def test_scheduled_training_blocks_unverified_input(tmp_path: Path, monkeypatch) -> None:
    def unexpected_training(args):
        raise AssertionError("unverified data must not reach fitting")

    monkeypatch.setattr(training_accelerator, "run_multi_horizon_pipeline", unexpected_training)
    report = training_accelerator.run_training_accelerator(argparse.Namespace(
        cadence="daily", data_dir=tmp_path, symbols="AAPL",
        output_dir=tmp_path / "out", training_cache_dir=tmp_path / "cache",
        model_type="logistic", plan_only=False, max_replay_candidates=None,
        require_validated_data=True,
    ))
    assert report["blocked_reasons"] == ["training_data_completeness_unverified"]
    selection = build_regime_champion_report(candidates=report)
    assert selection["decisions"] == []
    assert selection["reason"] == "no_candidate_records"


def test_training_accelerator_plan_writes_report(tmp_path: Path) -> None:
    report = training_accelerator.run_training_accelerator(
        argparse.Namespace(
            cadence="daily",
            data_dir=tmp_path / "bars",
            symbols="AAPL,AMZN",
            output_dir=tmp_path / "out",
            training_cache_dir=tmp_path / "cache",
            horizons="",
            label_objectives="",
            lead_horizon_bars=0,
            model_type="logistic",
            plan_only=True,
            max_replay_candidates=None,
        )
    )

    assert report["status"] == "planned"
    assert report["promotion_authority"] is False
    assert report["input_signature"]
    assert report["timing"]["duration_seconds"] >= 0.0
    payload = json.loads((tmp_path / "out" / "training_accelerator_report.json").read_text(encoding="utf-8"))
    assert payload["config"]["training_cache_dir"] == str(tmp_path / "cache")
    manifest = json.loads(Path(payload["input_manifest"]).read_text(encoding="utf-8"))
    assert manifest["inputs"]["data_dir"]["exists"] is False
    assert payload["cache"]["hit"] is False


def test_training_accelerator_rejects_symbols_outside_governed_universe(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="not governed: GOOGL"):
        training_accelerator.run_training_accelerator(
            argparse.Namespace(
                cadence="daily",
                data_dir=tmp_path,
                symbols="AAPL,GOOGL",
                output_dir=tmp_path / "out",
                training_cache_dir=tmp_path / "cache",
                model_type="logistic",
                plan_only=True,
                max_replay_candidates=None,
            )
        )


def test_training_accelerator_blocks_required_unusable_live_cost(
    tmp_path: Path,
) -> None:
    live_cost = tmp_path / "live_cost.json"
    live_cost.write_text(
        json.dumps(
            {
                "status": {
                    "available": True,
                    "status": "warming_up",
                }
            }
        ),
        encoding="utf-8",
    )

    report = training_accelerator.run_training_accelerator(
        argparse.Namespace(
            cadence="daily",
            data_dir=tmp_path,
            symbols="AAPL,AMZN,MSFT",
            output_dir=tmp_path / "out",
            training_cache_dir=tmp_path / "cache",
            model_type="logistic",
            live_cost_model_json=live_cost,
            use_live_cost_model=True,
            plan_only=False,
            max_replay_candidates=None,
        )
    )

    assert report["status"] == "blocked"
    assert report["blocked_reasons"] == ["required_live_cost_model_unusable"]
    assert report["live_cost_usability"]["reason"] == "not_ready"
    assert report["promotion_authority"] is False
    assert report["runtime_authority"] is False
    assert report["live_money_authority"] is False
    persisted = json.loads(
        (tmp_path / "out" / "training_accelerator_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["status"] == "blocked"
    assert persisted["runtime_authority"] is False
    assert persisted["live_money_authority"] is False


def test_training_accelerator_research_fallback_does_not_use_unready_live_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live_cost = tmp_path / "live_cost.json"
    live_cost.write_text(
        json.dumps({"status": {"available": False, "status": "unavailable"}}),
        encoding="utf-8",
    )
    calls: list[argparse.Namespace] = []

    def _fake_pipeline(args: argparse.Namespace) -> dict[str, Any]:
        calls.append(args)
        return {"ranked_candidates": [], "lead_candidates": []}

    monkeypatch.setattr(training_accelerator, "run_multi_horizon_pipeline", _fake_pipeline)
    report = training_accelerator.run_training_accelerator(
        argparse.Namespace(
            cadence="daily",
            data_dir=tmp_path,
            symbols="AAPL,AMZN,MSFT",
            output_dir=tmp_path / "out",
            training_cache_dir=tmp_path / "cache",
            model_type="logistic",
            live_cost_model_json=live_cost,
            use_live_cost_model=True,
            research_cost_fallback=True,
            plan_only=False,
            max_replay_candidates=None,
        )
    )

    assert report["status"] == "no_valid_candidates"
    assert report["cost_evidence"]["fallback_active"] is True
    assert report["cost_evidence"]["promotion_authority"] is False
    assert calls[0].use_live_cost_model is False


def test_training_accelerator_ingests_validated_shadow_manifest(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "shadow_manifest.json"
    shadow_jsonl = tmp_path / "shadow.jsonl"
    shadow_jsonl.write_text('{"evidence_type":"shadow_counterfactual"}\n', encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "artifact_type": "shadow_markout_replay_input_manifest",
                "evidence_type": "shadow_counterfactual",
                "evidence_partition": "shadow",
                "research_only": True,
                "row_count": 12,
                "content_sha256": hashlib.sha256(shadow_jsonl.read_bytes()).hexdigest(),
                "output_jsonl": str(shadow_jsonl),
                "fill_based_evidence": False,
                "promotion_eligible": False,
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
            }
        ),
        encoding="utf-8",
    )

    report = training_accelerator.run_training_accelerator(
        argparse.Namespace(
            cadence="daily",
            data_dir=tmp_path,
            symbols="AAPL,AMZN,MSFT",
            output_dir=tmp_path / "out",
            training_cache_dir=tmp_path / "cache",
            model_type="logistic",
            shadow_markout_manifest_json=manifest,
            shadow_markout_jsonl=shadow_jsonl,
            plan_only=True,
            max_replay_candidates=None,
        )
    )

    evidence = report["shadow_markout_evidence"]
    assert evidence["usable"] is True
    assert evidence["row_count"] == 12
    assert evidence["training_ingestion_enabled"] is True
    assert report["shadow_markout_selection"]["source"] == "current_run"


def test_training_accelerator_falls_back_to_latest_verified_shadow_manifest(
    tmp_path: Path,
) -> None:
    fallback_jsonl = tmp_path / "latest.jsonl"
    fallback_jsonl.write_text("{}\n", encoding="utf-8")
    fallback_manifest = tmp_path / "latest_manifest.json"
    fallback_manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "artifact_type": "shadow_markout_replay_input_manifest",
                "evidence_type": "shadow_counterfactual",
                "evidence_partition": "shadow",
                "research_only": True,
                "row_count": 1,
                "content_sha256": hashlib.sha256(
                    fallback_jsonl.read_bytes()
                ).hexdigest(),
                "output_jsonl": str(fallback_jsonl),
                "fill_based_evidence": False,
                "promotion_eligible": False,
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
            }
        ),
        encoding="utf-8",
    )

    report = training_accelerator.run_training_accelerator(
        argparse.Namespace(
            cadence="daily",
            data_dir=tmp_path,
            symbols="AAPL,AMZN,MSFT",
            output_dir=tmp_path / "out",
            training_cache_dir=tmp_path / "cache",
            model_type="logistic",
            shadow_markout_manifest_json=tmp_path / "missing_manifest.json",
            shadow_markout_jsonl=tmp_path / "missing.jsonl",
            shadow_markout_fallback_manifest_json=fallback_manifest,
            shadow_markout_fallback_jsonl=fallback_jsonl,
            plan_only=True,
            max_replay_candidates=None,
        )
    )

    assert report["shadow_markout_selection"]["fallback_used"] is True
    assert report["shadow_markout_evidence"]["usable"] is True


def test_training_accelerator_invokes_multi_horizon_with_cache(tmp_path: Path, monkeypatch) -> None:
    calls: list[argparse.Namespace] = []

    def _fake_pipeline(args: argparse.Namespace) -> dict[str, Any]:
        calls.append(args)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "multi_horizon_research_report.json").write_text("{}", encoding="utf-8")
        return {"ranked_candidates": [{
            "model_path": "m", "model_name": "challenger",
            "development_eligible": False,
            "walk_forward": {"aggregate": {"trades": 42, "mean_post_cost_net_edge_bps": -2.0}},
        }], "lead_candidates": [{"model_path": "m"}]}

    monkeypatch.setattr(training_accelerator, "run_multi_horizon_pipeline", _fake_pipeline)

    report = training_accelerator.run_training_accelerator(
        argparse.Namespace(
            cadence="weekly",
            data_dir=tmp_path,
            symbols="AAPL",
            timestamp_col="timestamp",
            output_dir=tmp_path / "out",
            training_cache_dir=tmp_path / "cache",
            horizons="1,3",
            label_objectives="risk_adjusted",
            lead_horizon_bars=3,
            model_prefix="candidate",
            model_type="logistic",
            fee_bps=0.0,
            slippage_bps=0.0,
            live_cost_model_json=None,
            use_live_cost_model=None,
            min_net_edge_bps=0.0,
            train_fraction=0.7,
            edge_global_threshold=0.66,
            random_state=1,
            replay_confidence_threshold=0.66,
            replay_entry_score_threshold=0.05,
            min_hold_bars=3,
            max_hold_bars=45,
            stop_loss_bps=20.0,
            take_profit_bps=50.0,
            trailing_stop_bps=15.0,
            max_replay_candidates=3,
            plan_only=False,
        )
    )

    assert report["status"] == "complete"
    assert report["promotion_authority"] is False
    assert report["cache"]["hit"] is False
    assert report["cache"]["miss_reason"] == "no_success_state"
    assert report["timing"]["pipeline_duration_seconds"] >= 0.0
    assert calls[0].training_cache is True
    assert calls[0].training_cache_dir == tmp_path / "cache"
    assert calls[0].horizons == "1,3"
    assert calls[0].max_replay_candidates == 3
    selection = build_regime_champion_report(candidates=report)
    decision = selection["decisions"][0]
    assert decision["candidate_model_id"] == "challenger"
    assert decision["samples"] == 42
    assert "development_evidence_not_qualified" in decision["reasons"]
    assert "candidate_model_id_missing" not in decision["reasons"]

    skipped = training_accelerator.run_training_accelerator(
        argparse.Namespace(
            cadence="weekly",
            data_dir=tmp_path,
            symbols="AAPL",
            timestamp_col="timestamp",
            output_dir=tmp_path / "out",
            training_cache_dir=tmp_path / "cache",
            horizons="1,3",
            label_objectives="risk_adjusted",
            lead_horizon_bars=3,
            model_prefix="candidate",
            model_type="logistic",
            fee_bps=0.0,
            slippage_bps=0.0,
            live_cost_model_json=None,
            use_live_cost_model=None,
            min_net_edge_bps=0.0,
            train_fraction=0.7,
            edge_global_threshold=0.66,
            random_state=1,
            replay_confidence_threshold=0.66,
            replay_entry_score_threshold=0.05,
            min_hold_bars=3,
            max_hold_bars=45,
            stop_loss_bps=20.0,
            take_profit_bps=50.0,
            trailing_stop_bps=15.0,
            max_replay_candidates=3,
            plan_only=False,
        )
    )

    assert skipped["status"] == "skipped_unchanged"
    assert skipped["cache"]["hit"] is True
    assert skipped["cache"]["previous_report_exists"] is True
    assert skipped["cache"]["hit_reason"] == "unchanged_successful_signature"
    assert skipped["ranked_candidate_count"] == 1
    assert skipped["candidates"] == report["candidates"]
    assert len(calls) == 1

    Path(str(skipped["previous_report_path"])).unlink()
    rerun = training_accelerator.run_training_accelerator(
        argparse.Namespace(
            cadence="weekly",
            data_dir=tmp_path,
            symbols="AAPL",
            timestamp_col="timestamp",
            output_dir=tmp_path / "out",
            training_cache_dir=tmp_path / "cache",
            horizons="1,3",
            label_objectives="risk_adjusted",
            lead_horizon_bars=3,
            model_prefix="candidate",
            model_type="logistic",
            fee_bps=0.0,
            slippage_bps=0.0,
            live_cost_model_json=None,
            use_live_cost_model=None,
            min_net_edge_bps=0.0,
            train_fraction=0.7,
            edge_global_threshold=0.66,
            random_state=1,
            replay_confidence_threshold=0.66,
            replay_entry_score_threshold=0.05,
            min_hold_bars=3,
            max_hold_bars=45,
            stop_loss_bps=20.0,
            take_profit_bps=50.0,
            trailing_stop_bps=15.0,
            max_replay_candidates=3,
            plan_only=False,
        )
    )

    assert rerun["status"] == "complete"
    assert rerun["cache"]["hit"] is False
    assert rerun["cache"]["miss_reason"] == "previous_report_missing"
    assert len(calls) == 2


def test_training_accelerator_manifest_hash_changes_when_file_content_changes(tmp_path: Path) -> None:
    data_dir = tmp_path / "bars"
    data_dir.mkdir()
    csv_path = data_dir / "AAPL.csv"
    csv_path.write_text("timestamp,close\n2026-01-02T14:30:00Z,100\n", encoding="utf-8")
    args = argparse.Namespace(
        data_dir=data_dir,
        live_cost_model_json=None,
        output_dir=tmp_path / "out",
    )
    config = {"training_cache_dir": str(tmp_path / "cache")}

    first = training_accelerator._accelerator_manifest(args, config)  # noqa: SLF001
    csv_path.write_text("timestamp,close\n2026-01-02T14:30:00Z,101\n", encoding="utf-8")
    second = training_accelerator._accelerator_manifest(args, config)  # noqa: SLF001

    first_file = first["inputs"]["data_dir"]["files"][0]
    second_file = second["inputs"]["data_dir"]["files"][0]
    assert first_file["sha256"] != second_file["sha256"]
    assert training_accelerator._stable_signature(first) != training_accelerator._stable_signature(second)  # noqa: SLF001


def test_training_accelerator_manifest_tracks_code_and_feature_contract(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        data_dir=tmp_path / "bars",
        live_cost_model_json=None,
        output_dir=tmp_path / "out",
    )
    config = {"training_cache_dir": str(tmp_path / "cache")}

    manifest = training_accelerator._accelerator_manifest(args, config)  # noqa: SLF001

    assert manifest["feature_contract"]["columns"]
    assert len(manifest["feature_contract"]["contract_sha256"]) == 64
    assert manifest["implementation"]["training_accelerator"]["sha256"]
    assert manifest["implementation"]["replay_aligned_trainer"]["sha256"]
    assert manifest["implementation"]["model_selection"]["sha256"]


def test_force_retrain_repeats_shadow_holdout_and_records_ledger(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = {"count": 0}

    def _fake_pipeline(args: argparse.Namespace) -> dict[str, Any]:
        calls["count"] += 1
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "multi_horizon_research_report.json").write_text(
            "{}", encoding="utf-8"
        )
        return {
            "ranked_candidates": [{"model_path": "m"}],
            "lead_candidates": [{"model_path": "m"}],
            "holdout_confirmation": {
                "status": "passed",
                "consumed": True,
                "promotion_authority": False,
                "live_money_authority": False,
            },
        }

    monkeypatch.setattr(
        training_accelerator, "run_multi_horizon_pipeline", _fake_pipeline
    )
    data_dir = tmp_path / "bars"
    data_dir.mkdir()
    args = training_accelerator._build_parser().parse_args(  # noqa: SLF001
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--training-cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    first = training_accelerator.run_training_accelerator(args)
    second = training_accelerator.run_training_accelerator(args)
    args.force_retrain = True
    forced = training_accelerator.run_training_accelerator(args)

    assert first["status"] == "complete"
    assert second["status"] == "skipped_unchanged"
    assert forced["status"] == "complete"
    assert forced["cache"]["forced_repeat_holdout"] is True
    assert calls["count"] == 2
    ledger = json.loads(
        Path(forced["holdout_ledger_path"]).read_text(encoding="utf-8")
    )
    assert ledger["forced_repeat"] is True
    assert ledger["promotion_authority"] is False
    assert ledger["live_money_authority"] is False
