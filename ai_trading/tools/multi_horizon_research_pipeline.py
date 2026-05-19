"""Train and evaluate replay-aligned candidates across multiple horizons."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.logging import get_logger
from ai_trading.tools.offline_replay import run_replay
from ai_trading.tools.train_replay_aligned_model import train_replay_aligned_model

logger = get_logger(__name__)


def _parse_int_list(value: str, *, default: tuple[int, ...]) -> list[int]:
    raw = str(value or "").strip()
    if not raw:
        return list(default)
    parsed: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parsed.append(max(1, int(token)))
    return sorted(set(parsed)) or list(default)


def _parse_objectives(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return ["net_markout"]
    return [token.strip() for token in raw.split(",") if token.strip()]


def _slim_replay_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = payload.get("aggregate")
    candidate_quality = payload.get("candidate_quality")
    out: dict[str, Any] = {}
    if isinstance(aggregate, Mapping):
        for key in (
            "total_trades",
            "win_rate",
            "profit_factor",
            "expectancy_bps",
            "net_pnl_bps",
            "orders_submitted",
            "fill_events",
            "violation_count",
        ):
            out[key] = aggregate.get(key)
    if isinstance(candidate_quality, Mapping):
        overall = candidate_quality.get("overall")
        if isinstance(overall, Mapping):
            out["candidate_quality_overall"] = dict(overall)
        for key in ("best_symbols", "worst_symbols", "by_session_regime", "by_session_segment"):
            value = candidate_quality.get(key)
            if isinstance(value, list):
                out[key] = value[:10]
    return out


def _write_replay_payload(path: Path, payload: Mapping[str, Any]) -> None:
    serializable = dict(payload)
    artifacts = serializable.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts["output_json"] = str(path)
    else:
        serializable["artifacts"] = {"output_json": str(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")


def _candidate_rank_key(record: Mapping[str, Any]) -> tuple[float, float, int]:
    replay = record.get("replay")
    validation = record.get("validation")
    expectancy = 0.0
    auc = 0.0
    trades = 0
    if isinstance(replay, Mapping):
        raw_expectancy = replay.get("expectancy_bps")
        raw_trades = replay.get("total_trades")
        expectancy = float(raw_expectancy) if raw_expectancy is not None else 0.0
        trades = int(raw_trades) if raw_trades is not None else 0
    if isinstance(validation, Mapping):
        raw_auc = validation.get("roc_auc")
        auc = float(raw_auc) if raw_auc is not None else 0.0
    return (expectancy, auc, trades)


def _candidate_training_rank_key(record: Mapping[str, Any]) -> tuple[float, int, float]:
    validation = record.get("validation")
    threshold_sweep = record.get("threshold_sweep")
    auc = 0.0
    horizon = int(record.get("horizon_bars", 0) or 0)
    sweep_edge = 0.0
    if isinstance(validation, Mapping):
        raw_auc = validation.get("roc_auc")
        auc = float(raw_auc) if raw_auc is not None else 0.0
    if isinstance(threshold_sweep, list):
        for row in threshold_sweep:
            if not isinstance(row, Mapping):
                continue
            for key in ("net_edge_bps", "mean_net_markout_bps", "expectancy_bps"):
                raw = row.get(key)
                if raw is None:
                    continue
                try:
                    sweep_edge = max(sweep_edge, float(raw))
                except (TypeError, ValueError):
                    continue
    return (auc, horizon, sweep_edge)


def run_multi_horizon_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    horizons = _parse_int_list(str(args.horizons), default=(1, 3, 5, 15))
    objectives = _parse_objectives(str(args.label_objectives))
    candidates: list[dict[str, Any]] = []
    replay_errors: list[dict[str, Any]] = []
    for objective in objectives:
        for horizon in horizons:
            model_name = f"{args.model_prefix}_h{horizon}_{objective}"
            record: dict[str, Any] = {
                "horizon_bars": int(horizon),
                "label_objective": objective,
                "model_name": model_name,
            }
            try:
                training_report = train_replay_aligned_model(
                    argparse.Namespace(
                        data_dir=Path(args.data_dir),
                        symbols=str(args.symbols or ""),
                        timestamp_col=str(args.timestamp_col),
                        output_dir=model_dir,
                        model_name=model_name,
                        model_type=str(args.model_type),
                        horizon_bars=int(horizon),
                        label_objective=objective,
                        fee_bps=float(args.fee_bps),
                        slippage_bps=float(args.slippage_bps),
                        live_cost_model_json=getattr(args, "live_cost_model_json", None),
                        use_live_cost_model=getattr(args, "use_live_cost_model", None),
                        min_net_edge_bps=float(args.min_net_edge_bps),
                        train_fraction=float(args.train_fraction),
                        edge_global_threshold=getattr(args, "edge_global_threshold", None),
                        random_state=int(args.random_state) + int(horizon),
                        training_cache=getattr(args, "training_cache", None),
                        training_cache_dir=getattr(args, "training_cache_dir", None),
                    )
                )
                model_path = str(training_report["model_path"])
                record.update(
                    {
                        "model_path": model_path,
                        "manifest_path": training_report.get("manifest_path"),
                        "training_report_path": training_report.get("report_path"),
                        "dataset": training_report.get("dataset"),
                        "validation": training_report.get("validation"),
                        "threshold_sweep": training_report.get("threshold_sweep", [])[:10],
                        "threshold_sweep_by_regime": training_report.get("threshold_sweep_by_regime"),
                        "feature_importance": training_report.get("feature_importance", [])[:25],
                        "live_cost_model": training_report.get("live_cost_model"),
                        "replay_status": "pending_selection",
                    }
                )
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            candidates.append(record)
    valid_trained = [record for record in candidates if "error" not in record]
    max_replay_candidates = int(getattr(args, "max_replay_candidates", 0) or 0)
    if max_replay_candidates <= 0:
        replay_selected = list(valid_trained)
    else:
        replay_selected = sorted(
            valid_trained,
            key=_candidate_training_rank_key,
            reverse=True,
        )[:max(1, max_replay_candidates)]
    selected_ids = {
        (int(record.get("horizon_bars", 0) or 0), str(record.get("label_objective") or ""))
        for record in replay_selected
    }
    for record in valid_trained:
        record_id = (int(record.get("horizon_bars", 0) or 0), str(record.get("label_objective") or ""))
        if record_id not in selected_ids:
            record["replay_status"] = "skipped_successive_halving"
            continue
        model_path = str(record.get("model_path") or "")
        model_name = str(record.get("model_name") or f"candidate_h{record_id[0]}_{record_id[1]}")
        replay_path = output_dir / f"{model_name}_replay.json"
        replay_argv = [
            "--data-dir",
            str(args.data_dir),
            "--symbols",
            str(args.symbols or ""),
            "--simulation-mode",
            "--use-model-score",
            "--model-path",
            model_path,
            "--confidence-threshold",
            str(args.replay_confidence_threshold),
            "--entry-score-threshold",
            str(args.replay_entry_score_threshold),
            "--min-hold-bars",
            str(args.min_hold_bars),
            "--max-hold-bars",
            str(args.max_hold_bars),
            "--stop-loss-bps",
            str(args.stop_loss_bps),
            "--take-profit-bps",
            str(args.take_profit_bps),
            "--trailing-stop-bps",
            str(args.trailing_stop_bps),
            "--fee-bps",
            str(args.fee_bps),
            "--slippage-bps",
            str(args.slippage_bps),
            "--output-json",
            str(replay_path),
        ]
        if getattr(args, "live_cost_model_json", None) is not None:
            replay_argv.extend(["--live-cost-model-json", str(args.live_cost_model_json)])
        try:
            replay_payload = run_replay(replay_argv)
            _write_replay_payload(replay_path, replay_payload)
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            record["replay_status"] = "error"
            record["replay_error"] = {"type": type(exc).__name__, "message": str(exc)}
            replay_errors.append(dict(record["replay_error"]) | {"model_name": model_name})
            continue
        record.update(
            {
                "replay_status": "complete",
                "replay_output": str(replay_path),
                "replay": _slim_replay_summary(replay_payload),
            }
        )
    ranked = sorted(
        [
            record
            for record in candidates
            if "error" not in record and str(record.get("replay_status") or "") == "complete"
        ],
        key=_candidate_rank_key,
        reverse=True,
    )
    report = {
        "schema_version": "1.0.0",
        "artifact_type": "multi_horizon_research_report",
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "data_dir": str(args.data_dir),
            "symbols": str(args.symbols or ""),
            "horizons": horizons,
            "label_objectives": objectives,
            "lead_horizon_bars": int(args.lead_horizon_bars),
            "live_cost_model_json": (
                str(args.live_cost_model_json) if args.live_cost_model_json else None
            ),
            "training_cache": getattr(args, "training_cache", None),
            "training_cache_dir": (
                str(args.training_cache_dir) if getattr(args, "training_cache_dir", None) else None
            ),
            "max_replay_candidates": max_replay_candidates,
        },
        "replay_config": {
            "confidence_threshold": float(args.replay_confidence_threshold),
            "entry_score_threshold": float(args.replay_entry_score_threshold),
            "min_hold_bars": int(args.min_hold_bars),
            "max_hold_bars": int(args.max_hold_bars),
            "stop_loss_bps": float(args.stop_loss_bps),
            "take_profit_bps": float(args.take_profit_bps),
            "trailing_stop_bps": float(args.trailing_stop_bps),
            "fee_bps": float(args.fee_bps),
            "slippage_bps": float(args.slippage_bps),
        },
        "candidates": candidates,
        "replay_selection": {
            "strategy": "successive_halving_top_n" if max_replay_candidates > 0 else "all_candidates",
            "max_replay_candidates": max_replay_candidates,
            "trained_candidate_count": len(valid_trained),
            "replayed_candidate_count": len(
                [
                    record
                    for record in valid_trained
                    if str(record.get("replay_status") or "") == "complete"
                ]
            ),
            "skipped_candidate_count": len(
                [
                    record
                    for record in valid_trained
                    if str(record.get("replay_status") or "") == "skipped_successive_halving"
                ]
            ),
            "errors": replay_errors,
        },
        "ranked_candidates": ranked,
        "lead_candidates": [
            record
            for record in ranked
            if int(record.get("horizon_bars", 0)) == int(args.lead_horizon_bars)
        ],
        "recommendation": (
            "evaluate_ranked_candidates_in_shadow_only"
            if ranked
            else "no_valid_candidates"
        ),
    }
    output_path = output_dir / "multi_horizon_research_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "MULTI_HORIZON_RESEARCH_REPORT_WRITTEN",
        extra={"path": str(output_path), "candidates": len(candidates)},
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--timestamp-col", type=str, default="timestamp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", type=str, default="1,3,5,15")
    parser.add_argument(
        "--label-objectives",
        type=str,
        default="net_markout,risk_adjusted",
        help="Comma-separated objectives: net_markout, spread_adjusted, risk_adjusted, mae_mfe.",
    )
    parser.add_argument("--lead-horizon-bars", type=int, default=15)
    parser.add_argument("--model-prefix", type=str, default="replay_aligned")
    parser.add_argument("--model-type", choices=("logistic", "random_forest", "hist_gradient"), default="logistic")
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--use-live-cost-model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-net-edge-bps", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--edge-global-threshold", type=float, default=0.66)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--training-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--training-cache-dir", type=Path, default=None)
    parser.add_argument("--max-replay-candidates", type=int, default=0)
    parser.add_argument("--replay-confidence-threshold", type=float, default=0.66)
    parser.add_argument("--replay-entry-score-threshold", type=float, default=0.05)
    parser.add_argument("--min-hold-bars", type=int, default=3)
    parser.add_argument("--max-hold-bars", type=int, default=45)
    parser.add_argument("--stop-loss-bps", type=float, default=20.0)
    parser.add_argument("--take-profit-bps", type=float, default=50.0)
    parser.add_argument("--trailing-stop-bps", type=float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_multi_horizon_pipeline(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
