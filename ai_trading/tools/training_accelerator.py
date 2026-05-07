"""Run cached, bounded replay-aligned training research."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.tools.multi_horizon_research_pipeline import run_multi_horizon_pipeline


def _env_text(name: str, default: str = "") -> str:
    return str(get_env(name, default, cast=str, resolve_aliases=False) or default).strip()


def _default_output_dir(cadence: str) -> Path:
    root = resolve_runtime_artifact_path(
        "runtime/training_accelerator",
        default_relative="runtime/training_accelerator",
        for_write=True,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root / str(cadence or "daily").lower() / stamp


def _cadence_defaults(cadence: str) -> tuple[str, str, int, int]:
    normalized = str(cadence or "daily").strip().lower()
    if normalized == "daily":
        return "1,15", "risk_adjusted", 15, 2
    if normalized == "weekly":
        return "1,3,5,15", "net_markout,spread_adjusted,risk_adjusted,mae_mfe", 15, 6
    return "1,3,5,15", "net_markout,spread_adjusted,risk_adjusted,mae_mfe", 15, 4


def run_training_accelerator(args: argparse.Namespace) -> dict[str, Any]:
    horizons, objectives, lead, replay_top_n = _cadence_defaults(str(args.cadence))
    output_dir = Path(args.output_dir or _default_output_dir(str(args.cadence)))
    output_dir.mkdir(parents=True, exist_ok=True)
    training_cache_dir = Path(args.training_cache_dir) if args.training_cache_dir else output_dir / "feature_cache"
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "artifact_type": "training_accelerator_report",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "cadence": str(args.cadence),
        "status": "planned" if args.plan_only else "running",
        "promotion_authority": False,
        "config": {
            "data_dir": str(args.data_dir),
            "symbols": str(args.symbols or ""),
            "horizons": str(args.horizons or horizons),
            "label_objectives": str(args.label_objectives or objectives),
            "lead_horizon_bars": int(args.lead_horizon_bars or lead),
            "model_type": str(args.model_type),
            "training_cache_dir": str(training_cache_dir),
            "max_replay_candidates": int(getattr(args, "max_replay_candidates", None) or replay_top_n),
        },
    }
    if args.plan_only:
        report["status"] = "planned"
    else:
        pipeline_args = argparse.Namespace(
            data_dir=Path(args.data_dir),
            symbols=str(args.symbols or ""),
            timestamp_col=str(args.timestamp_col),
            output_dir=output_dir / "multi_horizon",
            horizons=str(args.horizons or horizons),
            label_objectives=str(args.label_objectives or objectives),
            lead_horizon_bars=int(args.lead_horizon_bars or lead),
            model_prefix=str(args.model_prefix),
            model_type=str(args.model_type),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
            live_cost_model_json=args.live_cost_model_json,
            use_live_cost_model=args.use_live_cost_model,
            min_net_edge_bps=float(args.min_net_edge_bps),
            train_fraction=float(args.train_fraction),
            edge_global_threshold=float(args.edge_global_threshold),
            random_state=int(args.random_state),
            replay_confidence_threshold=float(args.replay_confidence_threshold),
            replay_entry_score_threshold=float(args.replay_entry_score_threshold),
            min_hold_bars=int(args.min_hold_bars),
            max_hold_bars=int(args.max_hold_bars),
            stop_loss_bps=float(args.stop_loss_bps),
            take_profit_bps=float(args.take_profit_bps),
            trailing_stop_bps=float(args.trailing_stop_bps),
            training_cache=True,
            training_cache_dir=training_cache_dir,
            max_replay_candidates=int(getattr(args, "max_replay_candidates", None) or replay_top_n),
        )
        pipeline_report = run_multi_horizon_pipeline(pipeline_args)
        report["status"] = (
            "complete" if pipeline_report.get("ranked_candidates") else "no_valid_candidates"
        )
        report["multi_horizon_report"] = str(
            output_dir / "multi_horizon" / "multi_horizon_research_report.json"
        )
        report["ranked_candidate_count"] = len(pipeline_report.get("ranked_candidates", []))
        report["lead_candidate_count"] = len(pipeline_report.get("lead_candidates", []))
    output_path = output_dir / "training_accelerator_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest = resolve_runtime_artifact_path(
        f"runtime/training_accelerator_{args.cadence}_latest.json",
        default_relative=f"runtime/training_accelerator_{args.cadence}_latest.json",
        for_write=True,
    )
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["path"] = str(output_path)
    return report


def _build_parser() -> argparse.ArgumentParser:
    horizons, objectives, lead, replay_top_n = _cadence_defaults("daily")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence", choices=("daily", "weekly", "monthly"), default="daily")
    default_data_dir = _env_text("AI_TRADING_RESEARCH_DATA_DIR", "")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(default_data_dir) if default_data_dir else None,
    )
    parser.add_argument("--symbols", default=_env_text("AI_TRADING_CANARY_SYMBOLS", "AAPL,AMZN"))
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--horizons", default=horizons)
    parser.add_argument("--label-objectives", default=objectives)
    parser.add_argument("--lead-horizon-bars", type=int, default=lead)
    parser.add_argument("--model-prefix", default="accelerated_replay_aligned")
    parser.add_argument("--model-type", choices=("logistic", "random_forest", "hist_gradient"), default="logistic")
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--use-live-cost-model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-net-edge-bps", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--edge-global-threshold", type=float, default=0.66)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--replay-confidence-threshold", type=float, default=0.66)
    parser.add_argument("--replay-entry-score-threshold", type=float, default=0.05)
    parser.add_argument("--min-hold-bars", type=int, default=3)
    parser.add_argument("--max-hold-bars", type=int, default=45)
    parser.add_argument("--stop-loss-bps", type=float, default=20.0)
    parser.add_argument("--take-profit-bps", type=float, default=50.0)
    parser.add_argument("--trailing-stop-bps", type=float, default=15.0)
    parser.add_argument("--training-cache-dir", type=Path, default=None)
    parser.add_argument("--max-replay-candidates", type=int, default=replay_top_n)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.plan_only and (args.data_dir is None or not Path(args.data_dir).exists()):
        sys.stderr.write("training accelerator requires --data-dir for non-plan runs\n")
        return 2
    report = run_training_accelerator(args)
    sys.stdout.write(json.dumps({"status": report["status"], "path": report["path"]}, sort_keys=True) + "\n")
    return 0 if report["status"] in {"planned", "complete", "no_valid_candidates"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
