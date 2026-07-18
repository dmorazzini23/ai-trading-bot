from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_trading.tools import multi_horizon_research_pipeline as pipeline


def test_multi_horizon_pipeline_ranks_candidates_and_keeps_lead_horizon(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[int, str]] = []

    def _fake_train(args: argparse.Namespace) -> dict[str, Any]:
        calls.append((int(args.horizon_bars), str(args.label_objective)))
        model_path = tmp_path / f"model_h{args.horizon_bars}_{args.label_objective}.joblib"
        model_path.write_text("model", encoding="utf-8")
        return {
            "model_path": str(model_path),
            "manifest_path": f"{model_path}.manifest.json",
            "report_path": str(tmp_path / "report.json"),
            "dataset": {"rows": 100, "symbols": 2},
            "validation": {"roc_auc": 0.50 + (int(args.horizon_bars) / 100.0)},
            "threshold_sweep": [{"confidence_threshold": 0.66}],
            "threshold_sweep_by_regime": {"midday": [{"confidence_threshold": 0.66}]},
            "walk_forward": {
                "aggregate": {
                    "evidence_qualified": int(args.horizon_bars) == 1,
                    "mean_post_cost_net_edge_bps": (
                        6.0 if int(args.horizon_bars) == 1 else 20.0
                    ),
                    "profitable_fold_ratio": (
                        0.8 if int(args.horizon_bars) == 1 else 0.4
                    ),
                    "stability_score": 0.75,
                    "trades": 300,
                }
            },
        }

    def _fake_replay(argv: list[str]) -> dict[str, Any]:
        model_arg = argv[argv.index("--model-path") + 1]
        expectancy = 15.0 if "_h15_" in model_arg else -5.0
        return {
            "aggregate": {
                "total_trades": 50,
                "win_rate": 0.55,
                "profit_factor": 1.2,
                "expectancy_bps": expectancy,
                "net_pnl_bps": expectancy * 50.0,
                "orders_submitted": 50,
                "fill_events": 50,
                "violation_count": 0,
            },
            "candidate_quality": {
                "overall": {"mean_net_markout_bps": expectancy},
                "best_symbols": [{"symbol": "AAPL", "mean_net_markout_bps": expectancy}],
                "by_session_regime": [{"session_regime": "midday", "mean_net_markout_bps": expectancy}],
            },
        }

    monkeypatch.setattr(pipeline, "train_replay_aligned_model", _fake_train)
    monkeypatch.setattr(pipeline, "run_replay", _fake_replay)

    report = pipeline.run_multi_horizon_pipeline(
        argparse.Namespace(
            data_dir=tmp_path,
            symbols="AAPL,AMZN",
            timestamp_col="timestamp",
            output_dir=tmp_path / "out",
            horizons="1,15",
            label_objectives="net_markout,risk_adjusted",
            lead_horizon_bars=15,
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
            training_cache=True,
            training_cache_dir=tmp_path / "cache",
            replay_confidence_threshold=0.66,
            replay_entry_score_threshold=0.05,
            min_hold_bars=3,
            max_hold_bars=45,
            stop_loss_bps=20.0,
            take_profit_bps=50.0,
            trailing_stop_bps=15.0,
            max_replay_candidates=0,
        )
    )

    assert len(calls) == 4
    assert all(call[0] in {1, 15} for call in calls)
    assert report["ranked_candidates"][0]["horizon_bars"] == 1
    assert report["config"]["training_cache"] is True
    assert report["replay_config"]["confidence_threshold"] == 0.66
    assert report["replay_config"]["max_hold_bars"] == 45
    assert report["lead_candidates"]
    assert report["one_bar_challengers"]
    assert report["governance_status"] == "shadow"
    assert report["promotion_authority"] is False
    assert report["live_money_authority"] is False
    assert (tmp_path / "out" / "multi_horizon_research_report.json").is_file()
    replay_outputs = [
        Path(candidate["replay_output"])
        for candidate in report["ranked_candidates"]
        if candidate.get("replay_output")
    ]
    assert replay_outputs
    assert all(path.is_file() for path in replay_outputs)
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["artifacts"]["output_json"] == str(path)
        for path in replay_outputs
    )


def test_multi_horizon_pipeline_replays_only_top_training_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    replayed: list[str] = []

    def _fake_train(args: argparse.Namespace) -> dict[str, Any]:
        model_path = tmp_path / f"model_h{args.horizon_bars}_{args.label_objective}.joblib"
        model_path.write_text("model", encoding="utf-8")
        return {
            "model_path": str(model_path),
            "manifest_path": f"{model_path}.manifest.json",
            "report_path": str(tmp_path / "report.json"),
            "dataset": {"rows": 100, "symbols": 2},
            "validation": {"roc_auc": 0.50 + (int(args.horizon_bars) / 100.0)},
            "threshold_sweep": [{"confidence_threshold": 0.66}],
            "walk_forward": {
                "aggregate": {
                    "evidence_qualified": int(args.horizon_bars) in {1, 3},
                    "mean_post_cost_net_edge_bps": {
                        1: 9.0,
                        3: 7.0,
                        5: 30.0,
                        15: 40.0,
                    }[int(args.horizon_bars)],
                    "profitable_fold_ratio": 0.8,
                    "stability_score": 0.7,
                    "trades": 300,
                }
            },
        }

    def _fake_replay(argv: list[str]) -> dict[str, Any]:
        model_arg = argv[argv.index("--model-path") + 1]
        replayed.append(model_arg)
        return {
            "aggregate": {
                "total_trades": 10,
                "win_rate": 0.6,
                "profit_factor": 1.5,
                "expectancy_bps": 5.0,
            }
        }

    monkeypatch.setattr(pipeline, "train_replay_aligned_model", _fake_train)
    monkeypatch.setattr(pipeline, "run_replay", _fake_replay)

    report = pipeline.run_multi_horizon_pipeline(
        argparse.Namespace(
            data_dir=tmp_path,
            symbols="AAPL,AMZN",
            timestamp_col="timestamp",
            output_dir=tmp_path / "out",
            horizons="1,3,5,15",
            label_objectives="risk_adjusted",
            lead_horizon_bars=15,
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
            training_cache=True,
            training_cache_dir=tmp_path / "cache",
            replay_confidence_threshold=0.66,
            replay_entry_score_threshold=0.05,
            min_hold_bars=3,
            max_hold_bars=45,
            stop_loss_bps=20.0,
            take_profit_bps=50.0,
            trailing_stop_bps=15.0,
            max_replay_candidates=2,
        )
    )

    assert len(replayed) == 2
    assert any("_h1_" in path for path in replayed)
    assert any("_h3_" in path for path in replayed)
    assert not any("_h15_" in path for path in replayed)
    assert report["replay_selection"]["strategy"] == "successive_halving_top_n"
    assert report["replay_selection"]["trained_candidate_count"] == 4
    assert report["replay_selection"]["replayed_candidate_count"] == 2
    assert report["replay_selection"]["skipped_candidate_count"] == 2
