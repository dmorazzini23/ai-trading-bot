"""Register research tournament artifacts as governed shadow candidates."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ai_trading.model_registry import ModelRegistry
from ai_trading.runtime.atomic_io import atomic_write_text


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"research report is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("research report must be a JSON object")
    return payload


def _false_authority(payload: Mapping[str, Any], *, context: str) -> None:
    if payload.get("promotion_authority") is not False:
        raise ValueError(f"{context} must explicitly disable promotion authority")
    if payload.get("live_money_authority") is not False:
        raise ValueError(f"{context} must explicitly disable live-money authority")


def _resolve_tournament(
    report_path: Path,
) -> tuple[dict[str, Any], str]:
    report = _read_object(report_path)
    if report.get("artifact_type") == "training_accelerator_report":
        _false_authority(report, context="training accelerator report")
        signature = str(report.get("input_signature") or "").strip()
        if str(report.get("status") or "").strip().lower() == "blocked":
            tournament = dict(report)
            tournament["candidates"] = []
        else:
            nested_path = str(report.get("multi_horizon_report") or "").strip()
            if not nested_path:
                raise ValueError("training accelerator report has no completed tournament")
            tournament = _read_object(Path(nested_path).expanduser())
    elif report.get("artifact_type") == "multi_horizon_research_report":
        tournament = report
        signature = ""
    else:
        raise ValueError("unsupported research report artifact type")
    _false_authority(tournament, context="tournament report")
    if not signature:
        stable = {
            "config": tournament.get("config"),
            "replay_config": tournament.get("replay_config"),
            "successive_halving": tournament.get("successive_halving"),
        }
        signature = hashlib.sha256(
            json.dumps(
                stable,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
    return tournament, signature


def _walk_forward_metrics(
    candidate: Mapping[str, Any],
    *,
    fallback_generated_at: Any = None,
) -> dict[str, Any]:
    """Project the canonical walk-forward aggregate into registry evidence."""

    walk_forward = candidate.get("walk_forward")
    walk_forward_payload = walk_forward if isinstance(walk_forward, Mapping) else {}
    aggregate = walk_forward_payload.get("aggregate")
    source = aggregate if isinstance(aggregate, Mapping) else walk_forward_payload
    generated_at = candidate.get("generated_at") or fallback_generated_at
    metrics = {
        "mean_post_cost_net_edge_bps": source.get("mean_post_cost_net_edge_bps"),
        "profitable_fold_ratio": source.get("profitable_fold_ratio"),
        "stability_score": source.get("stability_score"),
        "trades": source.get("trades"),
        "evidence_qualified": bool(source.get("evidence_qualified")),
    }
    if str(generated_at or "").strip():
        metrics["generated_at"] = str(generated_at).strip()
    return metrics


def register_research_candidates(
    *,
    report_path: Path,
    registry_dir: Path,
    output_json: Path,
    stale_days: int = 30,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Register valid artifacts idempotently and retire only stale dominated shadows."""

    tournament, experiment_signature = _resolve_tournament(report_path)
    registry = ModelRegistry(registry_dir)
    config = tournament.get("config")
    tournament_config = config if isinstance(config, Mapping) else {}
    raw_candidates = tournament.get("candidates")
    candidates = raw_candidates if isinstance(raw_candidates, list) else []
    registered: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    if str(tournament.get("status") or "").strip().lower() == "blocked":
        blocked_reasons = tournament.get("blocked_reasons")
        reasons = (
            [str(reason) for reason in blocked_reasons if str(reason).strip()]
            if isinstance(blocked_reasons, list)
            else []
        )
        skipped.append(
            {
                "candidate": "training_accelerator_report",
                "reason": "source_blocked:" + (",".join(reasons) or "unspecified"),
            }
        )
    retired: set[str] = set()
    generated_at = (now or datetime.now(UTC)).astimezone(UTC)
    stale_before = generated_at - timedelta(days=max(1, int(stale_days)))

    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            skipped.append({"candidate": str(index), "reason": "invalid_candidate"})
            continue
        name = str(raw.get("model_name") or index)
        try:
            _false_authority(raw, context=f"candidate {name}")
        except ValueError as exc:
            skipped.append({"candidate": name, "reason": str(exc)})
            continue
        if str(raw.get("governance_status") or "") != "shadow":
            skipped.append({"candidate": name, "reason": "not_shadow"})
            continue
        model_path = Path(str(raw.get("model_path") or "")).expanduser()
        manifest_path = Path(str(raw.get("manifest_path") or "")).expanduser()
        if not model_path.is_file() or not manifest_path.is_file():
            skipped.append({"candidate": name, "reason": "artifact_missing"})
            continue
        dataset = raw.get("dataset")
        dataset_payload = dataset if isinstance(dataset, Mapping) else {}
        dataset_hash = str(dataset_payload.get("dataset_hash") or "").strip()
        if not dataset_hash:
            skipped.append({"candidate": name, "reason": "dataset_hash_missing"})
            continue
        metrics = _walk_forward_metrics(
            raw,
            fallback_generated_at=tournament.get("generated_at"),
        )
        comparable_scope = {
            "symbols": str(tournament_config.get("symbols") or ""),
            "horizon_bars": int(raw.get("horizon_bars") or 0),
            "label_objective": str(raw.get("label_objective") or ""),
            "model_type": str(raw.get("model_type") or ""),
        }
        model_id, created = registry.register_shadow_candidate(
            {
                "artifact_path": str(model_path.resolve()),
                "manifest_path": str(manifest_path.resolve()),
            },
            "replay_aligned_markout",
            str(raw.get("model_type") or "unknown"),
            dataset_fingerprint=dataset_hash,
            experiment_signature=experiment_signature,
            comparable_scope=comparable_scope,
            metrics=metrics,
            metadata={
                "artifact_path": str(model_path.resolve()),
                "manifest_path": str(manifest_path.resolve()),
                "research_report_path": str(report_path.resolve()),
                "model_name": name,
                "holdout_confirmation_status": raw.get(
                    "holdout_confirmation_status"
                ),
            },
            tags=("research_tournament",),
        )
        registered.append(
            {"candidate": name, "model_id": model_id, "created": bool(created)}
        )
        try:
            retired.update(
                registry.retire_dominated_shadow_candidates(
                    model_id,
                    primary_metric="mean_post_cost_net_edge_bps",
                    stale_before=stale_before,
                )
            )
        except ValueError:
            # Missing/non-finite primary metrics are retained for more evidence.
            pass

    payload = {
        "schema_version": "1.0.0",
        "artifact_type": "research_candidate_registry_report",
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "source_report": str(report_path.resolve()),
        "registry_dir": str(registry_dir.resolve()),
        "experiment_signature": experiment_signature,
        "registered": registered,
        "skipped": skipped,
        "retired_shadow_model_ids": sorted(retired),
        "governance_status": "shadow",
        "promotion_authority": False,
        "runtime_authority": False,
        "live_money_authority": False,
        "manual_production_promotion_required": True,
    }
    atomic_write_text(
        output_json,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-report-json", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--stale-days", type=int, default=30)
    args = parser.parse_args(argv)
    register_research_candidates(
        report_path=args.research_report_json,
        registry_dir=args.registry_dir,
        output_json=args.output_json,
        stale_days=args.stale_days,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
