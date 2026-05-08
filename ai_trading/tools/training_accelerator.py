"""Run cached, bounded replay-aligned training research."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
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


def _artifact_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_manifest(path: Path | None, *, exclude_paths: tuple[Path, ...] = ()) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    resolved = path.expanduser()
    excludes = tuple(item.expanduser().resolve(strict=False) for item in exclude_paths)
    payload: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    try:
        if resolved.is_file():
            stat = resolved.stat()
            payload.update(
                {
                    "kind": "file",
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "sha256": _file_sha256(resolved),
                }
            )
        elif resolved.is_dir():
            files = []
            for item in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
                item_resolved = item.resolve(strict=False)
                if any(item_resolved.is_relative_to(excluded) for excluded in excludes):
                    continue
                stat = item.stat()
                files.append(
                    {
                        "path": str(item.relative_to(resolved)),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                        "sha256": _file_sha256(item),
                    }
                )
            payload.update({"kind": "directory", "files": files})
    except OSError as exc:
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return payload


def _stable_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _accelerator_manifest(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    live_cost_model_json = getattr(args, "live_cost_model_json", None)
    data_excludes: list[Path] = [Path(config["training_cache_dir"])]
    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        data_excludes.append(Path(output_dir))
    if args.data_dir is not None:
        data_root = Path(args.data_dir)
        data_excludes.extend([data_root / "runtime", data_root / "runtime_data"])
    return {
        "schema_version": "1.0.0",
        "artifact_type": "training_accelerator_input_manifest",
        "config": config,
        "inputs": {
            "data_dir": _path_manifest(
                Path(args.data_dir) if args.data_dir is not None else None,
                exclude_paths=tuple(data_excludes),
            ),
            "live_cost_model_json": _path_manifest(
                Path(live_cost_model_json) if live_cost_model_json is not None else None
            ),
        },
    }


def _read_success_state(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_training_accelerator(args: argparse.Namespace) -> dict[str, Any]:
    started_at = _artifact_timestamp()
    started_perf = perf_counter()
    horizons, objectives, lead, replay_top_n = _cadence_defaults(str(args.cadence))
    output_dir = Path(args.output_dir or _default_output_dir(str(args.cadence)))
    output_dir.mkdir(parents=True, exist_ok=True)
    training_cache_dir = Path(args.training_cache_dir) if args.training_cache_dir else output_dir / "feature_cache"
    config = {
        "data_dir": str(args.data_dir),
        "symbols": str(getattr(args, "symbols", "") or ""),
        "timestamp_col": str(getattr(args, "timestamp_col", "timestamp")),
        "horizons": str(getattr(args, "horizons", "") or horizons),
        "label_objectives": str(getattr(args, "label_objectives", "") or objectives),
        "lead_horizon_bars": int(getattr(args, "lead_horizon_bars", 0) or lead),
        "model_prefix": str(getattr(args, "model_prefix", "accelerated_replay_aligned")),
        "model_type": str(args.model_type),
        "fee_bps": float(getattr(args, "fee_bps", 1.0)),
        "slippage_bps": float(getattr(args, "slippage_bps", 2.0)),
        "live_cost_model_json": (
            str(getattr(args, "live_cost_model_json", None))
            if getattr(args, "live_cost_model_json", None) is not None
            else None
        ),
        "use_live_cost_model": getattr(args, "use_live_cost_model", None),
        "min_net_edge_bps": float(getattr(args, "min_net_edge_bps", 0.0)),
        "train_fraction": float(getattr(args, "train_fraction", 0.70)),
        "edge_global_threshold": float(getattr(args, "edge_global_threshold", 0.66)),
        "random_state": int(getattr(args, "random_state", 42)),
        "replay_confidence_threshold": float(getattr(args, "replay_confidence_threshold", 0.66)),
        "replay_entry_score_threshold": float(getattr(args, "replay_entry_score_threshold", 0.05)),
        "min_hold_bars": int(getattr(args, "min_hold_bars", 3)),
        "max_hold_bars": int(getattr(args, "max_hold_bars", 45)),
        "stop_loss_bps": float(getattr(args, "stop_loss_bps", 20.0)),
        "take_profit_bps": float(getattr(args, "take_profit_bps", 50.0)),
        "trailing_stop_bps": float(getattr(args, "trailing_stop_bps", 15.0)),
        "training_cache_dir": str(training_cache_dir),
        "max_replay_candidates": int(getattr(args, "max_replay_candidates", None) or replay_top_n),
    }
    input_manifest = _accelerator_manifest(args, config)
    input_signature = _stable_signature(input_manifest)
    manifest_path = output_dir / "training_accelerator_input_manifest.json"
    _write_json(manifest_path, input_manifest | {"signature": input_signature})
    success_state_path = training_cache_dir / "training_accelerator_success_signature.json"
    previous_state = _read_success_state(success_state_path)
    previous_signature = (
        str(previous_state.get("input_signature"))
        if isinstance(previous_state, dict) and previous_state.get("input_signature")
        else None
    )
    previous_status = (
        str(previous_state.get("status"))
        if isinstance(previous_state, dict) and previous_state.get("status")
        else None
    )
    previous_report_path = (
        Path(str(previous_state.get("report_path"))).expanduser()
        if isinstance(previous_state, dict) and previous_state.get("report_path")
        else None
    )
    previous_report_exists = bool(previous_report_path is not None and previous_report_path.exists())
    cache_hit = bool(
        not args.plan_only
        and previous_signature == input_signature
        and previous_status == "complete"
        and previous_report_exists
    )
    if previous_state is None:
        miss_reason = "no_success_state"
    elif previous_signature != input_signature:
        miss_reason = "signature_changed"
    elif previous_status != "complete":
        miss_reason = "previous_status_not_successful"
    elif not previous_report_exists:
        miss_reason = "previous_report_missing"
    else:
        miss_reason = None
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "artifact_type": "training_accelerator_report",
        "generated_at": started_at,
        "cadence": str(args.cadence),
        "status": "planned" if args.plan_only else "running",
        "promotion_authority": False,
        "config": config,
        "input_manifest": str(manifest_path),
        "input_signature": input_signature,
        "cache": {
            "success_state_path": str(success_state_path),
            "hit": cache_hit,
            "hit_reason": "unchanged_successful_signature" if cache_hit else None,
            "miss_reason": None if cache_hit or args.plan_only else miss_reason,
            "previous_signature": previous_signature,
            "previous_status": previous_status,
            "previous_report_exists": previous_report_exists,
        },
        "timing": {"started_at": started_at},
    }
    if args.plan_only:
        report["status"] = "planned"
    elif cache_hit:
        report["status"] = "skipped_unchanged"
        report["previous_report_path"] = previous_state.get("report_path") if previous_state else None
        report["ranked_candidate_count"] = int(previous_state.get("ranked_candidate_count", 0)) if previous_state else 0
        report["lead_candidate_count"] = int(previous_state.get("lead_candidate_count", 0)) if previous_state else 0
    else:
        pipeline_started = perf_counter()
        pipeline_args = argparse.Namespace(
            data_dir=Path(args.data_dir),
            symbols=str(config["symbols"]),
            timestamp_col=str(config["timestamp_col"]),
            output_dir=output_dir / "multi_horizon",
            horizons=str(config["horizons"]),
            label_objectives=str(config["label_objectives"]),
            lead_horizon_bars=int(config["lead_horizon_bars"]),
            model_prefix=str(config["model_prefix"]),
            model_type=str(config["model_type"]),
            fee_bps=float(config["fee_bps"]),
            slippage_bps=float(config["slippage_bps"]),
            live_cost_model_json=getattr(args, "live_cost_model_json", None),
            use_live_cost_model=config["use_live_cost_model"],
            min_net_edge_bps=float(config["min_net_edge_bps"]),
            train_fraction=float(config["train_fraction"]),
            edge_global_threshold=float(config["edge_global_threshold"]),
            random_state=int(config["random_state"]),
            replay_confidence_threshold=float(config["replay_confidence_threshold"]),
            replay_entry_score_threshold=float(config["replay_entry_score_threshold"]),
            min_hold_bars=int(config["min_hold_bars"]),
            max_hold_bars=int(config["max_hold_bars"]),
            stop_loss_bps=float(config["stop_loss_bps"]),
            take_profit_bps=float(config["take_profit_bps"]),
            trailing_stop_bps=float(config["trailing_stop_bps"]),
            training_cache=True,
            training_cache_dir=training_cache_dir,
            max_replay_candidates=int(config["max_replay_candidates"]),
        )
        pipeline_report = run_multi_horizon_pipeline(pipeline_args)
        report["timing"]["pipeline_duration_seconds"] = round(perf_counter() - pipeline_started, 6)
        report["status"] = (
            "complete" if pipeline_report.get("ranked_candidates") else "no_valid_candidates"
        )
        report["multi_horizon_report"] = str(
            output_dir / "multi_horizon" / "multi_horizon_research_report.json"
        )
        report["ranked_candidate_count"] = len(pipeline_report.get("ranked_candidates", []))
        report["lead_candidate_count"] = len(pipeline_report.get("lead_candidates", []))
    completed_at = _artifact_timestamp()
    report["timing"]["completed_at"] = completed_at
    report["timing"]["duration_seconds"] = round(perf_counter() - started_perf, 6)
    output_path = output_dir / "training_accelerator_report.json"
    _write_json(output_path, report)
    if report["status"] == "complete":
        _write_json(
            success_state_path,
            {
                "schema_version": "1.0.0",
                "artifact_type": "training_accelerator_success_signature",
                "generated_at": completed_at,
                "input_signature": input_signature,
                "input_manifest": str(manifest_path),
                "report_path": str(output_path),
                "status": report["status"],
                "ranked_candidate_count": int(report.get("ranked_candidate_count", 0)),
                "lead_candidate_count": int(report.get("lead_candidate_count", 0)),
            },
        )
    latest = resolve_runtime_artifact_path(
        f"runtime/training_accelerator_{args.cadence}_latest.json",
        default_relative=f"runtime/training_accelerator_{args.cadence}_latest.json",
        for_write=True,
    )
    _write_json(latest, report)
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
    return 0 if report["status"] in {"planned", "complete", "no_valid_candidates", "skipped_unchanged"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
