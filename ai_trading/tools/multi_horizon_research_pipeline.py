"""Train and evaluate replay-aligned candidates across multiple horizons."""

from __future__ import annotations

import argparse
import json
import math
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


def _parse_text_list(value: str, *, default: tuple[str, ...]) -> list[str]:
    parsed = [
        token.strip().lower()
        for token in str(value or "").split(",")
        if token.strip()
    ]
    return sorted(set(parsed)) or list(default)


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


def _walk_forward_aggregate(record: Mapping[str, Any]) -> Mapping[str, Any]:
    walk_forward = record.get("walk_forward")
    if not isinstance(walk_forward, Mapping):
        return {}
    aggregate = walk_forward.get("aggregate")
    return aggregate if isinstance(aggregate, Mapping) else {}


def _candidate_rank_key(
    record: Mapping[str, Any],
) -> tuple[int, float, float, float, int, float]:
    replay = record.get("replay")
    walk_forward = _walk_forward_aggregate(record)
    replay_expectancy = 0.0
    trades = 0
    if isinstance(replay, Mapping):
        raw_expectancy = replay.get("expectancy_bps")
        raw_trades = replay.get("total_trades")
        replay_expectancy = float(raw_expectancy) if raw_expectancy is not None else 0.0
        trades = int(raw_trades) if raw_trades is not None else 0
    qualified = int(bool(walk_forward.get("evidence_qualified")))
    mean_edge = float(walk_forward.get("mean_post_cost_net_edge_bps") or -1e12)
    profitable_ratio = float(walk_forward.get("profitable_fold_ratio") or 0.0)
    stability = float(walk_forward.get("stability_score") or 0.0)
    support = int(walk_forward.get("trades") or 0)
    return (
        qualified,
        mean_edge,
        profitable_ratio,
        stability,
        support,
        replay_expectancy + (trades * 1e-12),
    )


def _candidate_training_rank_key(
    record: Mapping[str, Any],
) -> tuple[int, float, float, float, int, float]:
    walk_forward = _walk_forward_aggregate(record)
    threshold_sweep = record.get("threshold_sweep")
    sweep_edge = 0.0
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
    return (
        int(bool(walk_forward.get("evidence_qualified"))),
        float(walk_forward.get("mean_post_cost_net_edge_bps") or -1e12),
        float(walk_forward.get("profitable_fold_ratio") or 0.0),
        float(walk_forward.get("stability_score") or 0.0),
        int(walk_forward.get("trades") or 0),
        sweep_edge,
    )


def _development_eligible(record: Mapping[str, Any]) -> bool:
    aggregate = _walk_forward_aggregate(record)
    return bool(
        aggregate.get("evidence_qualified")
        and int(aggregate.get("trades") or 0) > 0
        and float(aggregate.get("mean_post_cost_net_edge_bps") or 0.0) > 0.0
    )


def run_multi_horizon_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    horizons = _parse_int_list(str(args.horizons), default=(1, 3, 5, 15))
    objectives = _parse_objectives(str(args.label_objectives))
    model_types = _parse_text_list(
        str(getattr(args, "model_types", "") or ""),
        default=(str(args.model_type),),
    )
    allowed_model_types = {"logistic", "random_forest", "hist_gradient"}
    invalid_model_types = sorted(set(model_types) - allowed_model_types)
    if invalid_model_types:
        raise ValueError(
            "Unsupported model types: " + ",".join(invalid_model_types)
        )
    candidate_specs = [
        (model_type, objective, horizon)
        for model_type in model_types
        for objective in objectives
        for horizon in horizons
    ]
    max_candidates = max(
        1, int(getattr(args, "max_candidates", len(candidate_specs)) or len(candidate_specs))
    )
    candidate_specs = candidate_specs[:max_candidates]
    candidates: list[dict[str, Any]] = []
    replay_errors: list[dict[str, Any]] = []
    total_folds = max(2, int(getattr(args, "walk_forward_folds", 5) or 5))
    screening_folds = max(
        1,
        min(
            total_folds,
            int(getattr(args, "screening_folds", min(2, total_folds)) or 1),
        ),
    )
    halving_eta = max(2, int(getattr(args, "halving_eta", 2) or 2))

    def _training_args(
        record: Mapping[str, Any],
        *,
        evaluation_folds: int,
        evaluate_holdout: bool,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            data_dir=Path(args.data_dir),
            acquisition_manifest_json=getattr(
                args, "acquisition_manifest_json", None
            ),
            symbols=str(args.symbols or ""),
            timestamp_col=str(args.timestamp_col),
            output_dir=model_dir,
            model_name=str(record["model_name"]),
            model_type=str(record["model_type"]),
            horizon_bars=int(record["horizon_bars"]),
            label_objective=str(record["label_objective"]),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
            live_cost_model_json=getattr(args, "live_cost_model_json", None),
            use_live_cost_model=getattr(args, "use_live_cost_model", None),
            min_net_edge_bps=float(args.min_net_edge_bps),
            train_fraction=float(args.train_fraction),
            walk_forward_folds=total_folds,
            evaluation_folds=evaluation_folds,
            walk_forward_embargo_bars=int(
                getattr(args, "walk_forward_embargo_bars", 1) or 1
            ),
            walk_forward_embargo_percent=float(
                getattr(args, "walk_forward_embargo_percent", 0.0) or 0.0
            ),
            walk_forward_min_trades=int(
                getattr(args, "walk_forward_min_trades", 250) or 250
            ),
            walk_forward_min_profitable_fold_ratio=float(
                getattr(args, "walk_forward_min_profitable_fold_ratio", 0.60)
                or 0.60
            ),
            walk_forward_min_mean_net_edge_bps=float(
                getattr(args, "walk_forward_min_mean_net_edge_bps", 0.0) or 0.0
            ),
            walk_forward_min_ranking_separation_bps=float(
                getattr(args, "walk_forward_min_ranking_separation_bps", 0.0)
                or 0.0
            ),
            edge_global_threshold=getattr(args, "edge_global_threshold", None),
            nested_validation_fraction=float(
                getattr(args, "nested_validation_fraction", 0.20) or 0.20
            ),
            nested_min_support=int(
                getattr(args, "nested_min_support", 25) or 25
            ),
            evaluate_holdout=evaluate_holdout,
            random_state=int(args.random_state)
            + int(record["horizon_bars"])
            + sum(ord(char) for char in str(record["model_type"])),
            training_cache=getattr(args, "training_cache", None),
            training_cache_dir=getattr(args, "training_cache_dir", None),
        )

    def _merge_training_report(
        record: dict[str, Any],
        training_report: Mapping[str, Any],
    ) -> None:
        record.update(
            {
                "model_path": training_report.get("model_path"),
                "manifest_path": training_report.get("manifest_path"),
                "training_report_path": training_report.get("report_path"),
                "dataset": training_report.get("dataset"),
                "validation": training_report.get("validation"),
                "threshold_sweep": list(training_report.get("threshold_sweep", []))[:10],
                "threshold_sweep_by_regime": training_report.get(
                    "threshold_sweep_by_regime"
                ),
                "walk_forward": training_report.get("walk_forward"),
                "market_regime_policy": training_report.get("market_regime_policy"),
                "feature_importance": list(
                    training_report.get("feature_importance", [])
                )[:25],
                "live_cost_model": training_report.get("live_cost_model"),
                "holdout_evaluation": training_report.get("holdout_evaluation"),
            }
        )

    multiple_families = len(model_types) > 1
    for model_type, objective, horizon in candidate_specs:
        family_suffix = f"_{model_type}" if multiple_families else ""
        record: dict[str, Any] = {
            "horizon_bars": int(horizon),
            "label_objective": objective,
            "model_type": model_type,
            "model_name": f"{args.model_prefix}_h{horizon}_{objective}{family_suffix}",
            "governance_status": "shadow",
            "promotion_authority": False,
            "live_money_authority": False,
            "replay_status": "pending_selection",
        }
        try:
            screening_report = train_replay_aligned_model(
                _training_args(
                    record,
                    evaluation_folds=screening_folds,
                    evaluate_holdout=False,
                )
            )
            _merge_training_report(record, screening_report)
            record["screening"] = {
                "status": "complete",
                "evaluation_folds": screening_folds,
                "walk_forward": screening_report.get("walk_forward"),
            }
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            record["screening"] = {"status": "error"}
        candidates.append(record)

    screened = [record for record in candidates if "error" not in record]
    survivor_count = max(1, int(math.ceil(len(screened) / halving_eta)))
    survivors = sorted(
        screened,
        key=_candidate_training_rank_key,
        reverse=True,
    )[:survivor_count]
    survivor_names = {str(record["model_name"]) for record in survivors}
    for record in candidates:
        if str(record["model_name"]) not in survivor_names and "error" not in record:
            record["halving_status"] = "eliminated_after_screening"
            record["replay_status"] = "skipped_successive_halving"
    full_evaluated: list[dict[str, Any]] = []
    for record in survivors:
        try:
            full_report = train_replay_aligned_model(
                _training_args(
                    record,
                    evaluation_folds=total_folds,
                    evaluate_holdout=False,
                )
            )
            _merge_training_report(record, full_report)
            record["halving_status"] = "full_evaluation_complete"
            record["full_evaluation_folds"] = total_folds
            full_evaluated.append(record)
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            record["full_evaluation_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            record["halving_status"] = "full_evaluation_error"

    development_ranked = sorted(
        full_evaluated,
        key=_candidate_training_rank_key,
        reverse=True,
    )
    for record in development_ranked:
        aggregate = _walk_forward_aggregate(record)
        record["development_eligible"] = _development_eligible(record)
        record["development_eligibility_reasons"] = list(
            aggregate.get("qualification_reasons") or []
        )
    eligible_ranked = [
        record for record in development_ranked if record["development_eligible"]
    ]
    winner = eligible_ranked[0] if eligible_ranked else None
    holdout_confirmation: dict[str, Any] = {
        "status": "not_run",
        "winner_model_name": None,
        "fallback_attempted": False,
        "consumed": False,
        "reason": "no_development_eligible_candidate",
        "promotion_authority": False,
        "live_money_authority": False,
    }
    if winner is not None:
        holdout_confirmation["winner_model_name"] = winner["model_name"]
        try:
            confirmation_report = train_replay_aligned_model(
                _training_args(
                    winner,
                    evaluation_folds=total_folds,
                    evaluate_holdout=True,
                )
            )
            _merge_training_report(winner, confirmation_report)
            holdout_payload = confirmation_report.get("holdout_evaluation")
            metrics = (
                holdout_payload.get("metrics", {})
                if isinstance(holdout_payload, Mapping)
                else {}
            )
            holdout_passed = bool(
                isinstance(holdout_payload, Mapping)
                and holdout_payload.get("consumed") is True
                and int(metrics.get("trades", 0) or 0) > 0
                and float(metrics.get("mean_post_cost_net_edge_bps") or 0.0) > 0.0
            )
            holdout_confirmation.update(
                {
                    "status": "passed" if holdout_passed else "failed",
                    "consumed": True,
                    "reason": None,
                    "metrics": dict(metrics),
                }
            )
            winner["holdout_confirmation_status"] = holdout_confirmation["status"]
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            holdout_confirmation.update(
                {
                    "status": "error",
                    "consumed": True,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            winner["holdout_confirmation_status"] = "error"

    replay_selected = [winner] if winner is not None else []
    for record in replay_selected:
        record_id = (
            int(record.get("horizon_bars", 0) or 0),
            str(record.get("label_objective") or ""),
        )
        model_path = str(record.get("model_path") or "")
        model_name = str(record.get("model_name") or f"candidate_h{record_id[0]}_{record_id[1]}")
        walk_forward = record.get("walk_forward")
        selected_threshold = (
            walk_forward.get("selected_threshold", {})
            if isinstance(walk_forward, Mapping)
            else {}
        )
        replay_confidence_threshold = float(
            selected_threshold.get(
                "confidence_threshold", args.replay_confidence_threshold
            )
        )
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
            str(replay_confidence_threshold),
            "--entry-score-threshold",
            "0.0",
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
                "replay_score_semantics": {
                    "semantics": "positive_class_probability_rank",
                    "cutoff_source": "development_only_percentile_selection",
                    "frozen_probability_cutoff": replay_confidence_threshold,
                    "entry_score_threshold": 0.0,
                },
            }
        )
    ranked = list(development_ranked)
    valid_trained = list(screened)
    max_replay_candidates = int(getattr(args, "max_replay_candidates", 0) or 0)
    report = {
        "schema_version": "2.0.0",
        "artifact_type": "multi_horizon_research_report",
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "data_dir": str(args.data_dir),
            "symbols": str(args.symbols or ""),
            "horizons": horizons,
            "label_objectives": objectives,
            "model_types": model_types,
            "max_candidates": max_candidates,
            "lead_horizon_bars": int(args.lead_horizon_bars),
            "live_cost_model_json": (
                str(args.live_cost_model_json) if args.live_cost_model_json else None
            ),
            "training_cache": getattr(args, "training_cache", None),
            "training_cache_dir": (
                str(args.training_cache_dir) if getattr(args, "training_cache_dir", None) else None
            ),
            "max_replay_candidates": max_replay_candidates,
            "walk_forward_folds": int(
                getattr(args, "walk_forward_folds", 5) or 5
            ),
            "walk_forward_embargo_bars": int(
                getattr(args, "walk_forward_embargo_bars", 1) or 1
            ),
            "walk_forward_embargo_percent": float(
                getattr(args, "walk_forward_embargo_percent", 0.0) or 0.0
            ),
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
            "strategy": "one_winner_after_successive_halving",
            "max_replay_candidates": max_replay_candidates,
            "trained_candidate_count": len(valid_trained),
            "replayed_candidate_count": int(
                winner is not None
                and str(winner.get("replay_status") or "") == "complete"
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
        "successive_halving": {
            "strategy": "fixed_outer_fold_budget",
            "screening_folds": screening_folds,
            "full_evaluation_folds": total_folds,
            "eta": halving_eta,
            "screened_candidate_count": len(screened),
            "survivor_count": len(survivors),
            "fully_evaluated_count": len(full_evaluated),
            "eliminated_model_names": sorted(
                str(record["model_name"])
                for record in candidates
                if record.get("halving_status") == "eliminated_after_screening"
            ),
        },
        "holdout_confirmation": holdout_confirmation,
        "ranked_candidates": ranked,
        "lead_candidates": [
            record
            for record in candidates
            if int(record.get("horizon_bars", 0)) == int(args.lead_horizon_bars)
            and "error" not in record
        ],
        "one_bar_challengers": [
            record for record in candidates if int(record.get("horizon_bars", 0)) == 1
        ],
        "governance_status": "shadow",
        "promotion_authority": False,
        "live_money_authority": False,
        "recommendation": (
            "shadow_winner_holdout_confirmed"
            if holdout_confirmation.get("status") == "passed"
            else "no_confirmed_candidate"
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
    parser.add_argument("--acquisition-manifest-json", type=Path, default=None)
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
    parser.add_argument(
        "--model-types",
        type=str,
        default="",
        help="Optional comma-separated model-family grid; --model-type remains the fallback.",
    )
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--screening-folds", type=int, default=2)
    parser.add_argument("--halving-eta", type=int, default=2)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--use-live-cost-model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-net-edge-bps", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--walk-forward-folds", type=int, default=5)
    parser.add_argument("--walk-forward-embargo-bars", type=int, default=1)
    parser.add_argument("--walk-forward-embargo-percent", type=float, default=0.0)
    parser.add_argument("--nested-validation-fraction", type=float, default=0.20)
    parser.add_argument("--nested-min-support", type=int, default=25)
    parser.add_argument("--walk-forward-min-trades", type=int, default=250)
    parser.add_argument(
        "--walk-forward-min-profitable-fold-ratio", type=float, default=0.60
    )
    parser.add_argument("--walk-forward-min-mean-net-edge-bps", type=float, default=0.0)
    parser.add_argument(
        "--walk-forward-min-ranking-separation-bps", type=float, default=0.0
    )
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
