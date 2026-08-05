from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from ai_trading.tools import training_accelerator


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


def test_training_accelerator_invokes_multi_horizon_with_cache(tmp_path: Path, monkeypatch) -> None:
    calls: list[argparse.Namespace] = []

    def _fake_pipeline(args: argparse.Namespace) -> dict[str, Any]:
        calls.append(args)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "multi_horizon_research_report.json").write_text("{}", encoding="utf-8")
        return {"ranked_candidates": [{"model_path": "m"}], "lead_candidates": [{"model_path": "m"}]}

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
