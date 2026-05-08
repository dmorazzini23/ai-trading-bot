"""Record and evaluate model registry artifacts with manual promotion gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

_ARTIFACT_TYPE = "model_registry"
_EVALUATION_ARTIFACT_TYPE = "model_registry_evaluation"
_SCHEMA_VERSION = "1.0.0"
_DEFAULT_OUTPUT_DIR = "runtime/research_reports/model_registry"
_ALLOWED_ROLES = {"champion", "challenger"}
_OK_STATUSES = {"ok", "pass", "passed", "ready", "success", "complete", "completed"}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _read_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    token = str(value).strip()
    if not token:
        return None
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(token)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _metric_value(metrics: Mapping[str, Any], primary_metric: str) -> float | None:
    if primary_metric in metrics:
        return _finite_float(metrics.get(primary_metric))
    summary = metrics.get("summary")
    if isinstance(summary, Mapping):
        return _finite_float(summary.get(primary_metric))
    return None


def _evidence_timestamp(metrics: Mapping[str, Any], fallback: datetime) -> datetime:
    for key in ("generated_at", "evaluated_at", "as_of", "timestamp", "created_at"):
        parsed = _parse_timestamp(metrics.get(key))
        if parsed is not None:
            return parsed
    return fallback


def _staleness_gate(
    *,
    evidence_at: datetime,
    generated_at: datetime,
    max_age_hours: float,
) -> dict[str, Any]:
    age_hours = max(0.0, (generated_at - evidence_at).total_seconds() / 3600.0)
    future_dated = evidence_at > generated_at
    ok = bool(age_hours <= max_age_hours and not future_dated)
    return {
        "ok": ok,
        "evidence_at": _iso(evidence_at),
        "age_hours": age_hours,
        "max_age_hours": float(max_age_hours),
        "reason": "ok" if ok else ("evidence_future_dated" if future_dated else "evidence_stale"),
    }


def _registry_models(registry: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    raw = (registry or {}).get("models")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _active_champion(models: list[dict[str, Any]]) -> dict[str, Any] | None:
    champions = [
        model
        for model in models
        if str(model.get("role") or "").lower() == "champion"
        and str(model.get("status") or "").lower() not in {"blocked", "failed"}
    ]
    if not champions:
        return None
    return champions[-1]


def _replace_model(models: list[dict[str, Any]], entry: dict[str, Any]) -> list[dict[str, Any]]:
    model_id = str(entry.get("model_id") or "")
    retained = [model for model in models if str(model.get("model_id") or "") != model_id]
    retained.append(entry)
    return retained


def build_model_registration(
    *,
    model_id: str,
    role: str,
    model_path: Path,
    metrics: Mapping[str, Any] | None = None,
    previous_registry: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
    max_evidence_age_hours: float = 96.0,
    manual_approval_id: str | None = None,
    rollback_command: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Return a registry artifact after adding or updating one model entry."""

    generated = generated_at.astimezone(UTC) if generated_at else _utc_now()
    normalized_role = str(role or "").strip().lower()
    metric_payload = dict(metrics or {})
    evidence_at = _evidence_timestamp(metric_payload, generated)
    freshness = _staleness_gate(
        evidence_at=evidence_at,
        generated_at=generated,
        max_age_hours=max_evidence_age_hours,
    )
    previous_models = _registry_models(previous_registry)
    previous_champion = _active_champion(previous_models)
    blocked_reasons: list[str] = []
    if not model_id.strip():
        blocked_reasons.append("model_id_required")
    if normalized_role not in _ALLOWED_ROLES:
        blocked_reasons.append("unsupported_role")
    if not freshness["ok"]:
        blocked_reasons.append(str(freshness["reason"]))
    if normalized_role == "champion" and not str(manual_approval_id or "").strip():
        blocked_reasons.append("manual_approval_required")

    artifact_sha256 = _sha256_file(model_path.expanduser())
    if artifact_sha256 is None:
        blocked_reasons.append("model_artifact_missing")

    entry_status = "blocked" if blocked_reasons else "registered"
    entry = {
        "model_id": model_id.strip(),
        "role": normalized_role if normalized_role in _ALLOWED_ROLES else "unsupported",
        "status": entry_status,
        "registered_at": _iso(generated),
        "model_path": str(model_path.expanduser()),
        "artifact_sha256": artifact_sha256,
        "metrics": metric_payload,
        "metrics_hash": _canonical_hash(metric_payload),
        "evidence_freshness": freshness,
        "manual_approval": {
            "required_for_champion": True,
            "approval_id": str(manual_approval_id or "").strip() or None,
            "boundary": "registry records approval; it does not copy, symlink, or deploy models",
        },
        "rollback": {
            "previous_champion_model_id": (previous_champion or {}).get("model_id"),
            "previous_champion_path": (previous_champion or {}).get("model_path"),
            "command": rollback_command
            or "restore previous AI_TRADING_MODEL_PATH and restart ai-trading.service",
        },
        "notes": str(notes or ""),
    }

    models = previous_models if blocked_reasons else _replace_model(previous_models, entry)
    if blocked_reasons:
        models = previous_models + [entry]
    champion = _active_champion(models)
    challengers = [
        model
        for model in models
        if str(model.get("role") or "").lower() == "challenger"
        and str(model.get("status") or "").lower() not in {"blocked", "failed"}
    ]
    status = "blocked" if blocked_reasons else "registered"
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "generated_at": _iso(generated),
        "status": status,
        "blocked_reasons": blocked_reasons,
        "promotion_authority": False,
        "manual_promotion_required": True,
        "champion": champion,
        "challengers": challengers,
        "models": models,
        "registered_model": entry,
        "rollback": entry["rollback"],
    }


def _model_freshness(
    model: Mapping[str, Any],
    *,
    generated_at: datetime,
    max_evidence_age_hours: float,
) -> dict[str, Any]:
    metrics = model.get("metrics")
    metric_payload = metrics if isinstance(metrics, Mapping) else {}
    fallback = _parse_timestamp(model.get("registered_at")) or generated_at
    return _staleness_gate(
        evidence_at=_evidence_timestamp(metric_payload, fallback),
        generated_at=generated_at,
        max_age_hours=max_evidence_age_hours,
    )


def build_model_evaluation(
    *,
    registry: Mapping[str, Any],
    challenger_id: str | None = None,
    primary_metric: str = "net_edge_bps",
    min_delta: float = 0.0,
    generated_at: datetime | None = None,
    max_evidence_age_hours: float = 96.0,
) -> dict[str, Any]:
    """Compare champion and challenger records without promotion authority."""

    generated = generated_at.astimezone(UTC) if generated_at else _utc_now()
    models = _registry_models(registry)
    champion = _active_champion(models)
    requested = str(challenger_id or "").strip()
    challengers = [
        model
        for model in models
        if str(model.get("role") or "").lower() == "challenger"
        and str(model.get("status") or "").lower() not in {"blocked", "failed"}
        and (not requested or str(model.get("model_id") or "") == requested)
    ]
    challenger = challengers[-1] if challengers else None
    blocked_reasons: list[str] = []
    if champion is None:
        blocked_reasons.append("champion_missing")
    if challenger is None:
        blocked_reasons.append("challenger_missing")

    champion_freshness = (
        _model_freshness(
            champion,
            generated_at=generated,
            max_evidence_age_hours=max_evidence_age_hours,
        )
        if champion is not None
        else {"ok": False, "reason": "champion_missing"}
    )
    challenger_freshness = (
        _model_freshness(
            challenger,
            generated_at=generated,
            max_evidence_age_hours=max_evidence_age_hours,
        )
        if challenger is not None
        else {"ok": False, "reason": "challenger_missing"}
    )
    if champion is not None and not champion_freshness["ok"]:
        blocked_reasons.append("champion_evidence_stale")
    if challenger is not None and not challenger_freshness["ok"]:
        blocked_reasons.append("challenger_evidence_stale")

    champion_metric = (
        _metric_value(
            champion.get("metrics") if isinstance(champion.get("metrics"), Mapping) else {},
            primary_metric,
        )
        if champion is not None
        else None
    )
    challenger_metric = (
        _metric_value(
            challenger.get("metrics") if isinstance(challenger.get("metrics"), Mapping) else {},
            primary_metric,
        )
        if challenger is not None
        else None
    )
    if champion is not None and champion_metric is None:
        blocked_reasons.append("champion_primary_metric_missing")
    if challenger is not None and challenger_metric is None:
        blocked_reasons.append("challenger_primary_metric_missing")

    delta = (
        float(challenger_metric - champion_metric)
        if challenger_metric is not None and champion_metric is not None
        else None
    )
    beats_champion = delta is not None and delta >= float(min_delta)
    status = "blocked" if blocked_reasons else "evaluated"
    recommendation = "blocked"
    if not blocked_reasons:
        recommendation = "manual_review_for_promotion" if beats_champion else "keep_champion"
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": _EVALUATION_ARTIFACT_TYPE,
        "generated_at": _iso(generated),
        "status": status,
        "blocked_reasons": blocked_reasons,
        "promotion_authority": False,
        "manual_promotion_required": bool(beats_champion),
        "primary_metric": primary_metric,
        "min_delta": float(min_delta),
        "champion": champion,
        "challenger": challenger,
        "metrics": {
            "champion": champion_metric,
            "challenger": challenger_metric,
            "delta": delta,
            "beats_champion": bool(beats_champion),
        },
        "freshness": {
            "champion": champion_freshness,
            "challenger": challenger_freshness,
        },
        "recommendation": recommendation,
        "promotion_boundary": "evaluation is advisory and cannot promote or deploy a model",
        "rollback": {
            "current_champion_model_id": (champion or {}).get("model_id"),
            "current_champion_path": (champion or {}).get("model_path"),
        },
    }


def _default_paths(kind: str, generated_at: datetime, output_dir: Path) -> tuple[Path, Path]:
    dated = output_dir / f"{kind}_{_stamp(generated_at)}.json"
    latest = output_dir / f"{kind}_latest.json"
    return dated, latest


def _resolve_output_dir(raw: str | None) -> Path:
    return resolve_runtime_artifact_path(
        raw or _DEFAULT_OUTPUT_DIR,
        default_relative=_DEFAULT_OUTPUT_DIR,
        for_write=True,
    )


def _cli_generated_at(raw: str | None) -> datetime:
    parsed = _parse_timestamp(raw)
    return parsed if parsed is not None else _utc_now()


def _write_outputs(
    *,
    payload: dict[str, Any],
    kind: str,
    generated_at: datetime,
    output_dir: Path,
    output_json: Path | None,
    latest_json: Path | None,
) -> tuple[Path, Path]:
    default_output, default_latest = _default_paths(kind, generated_at, output_dir)
    dated = output_json or default_output
    latest = latest_json or default_latest
    payload.setdefault("paths", {})
    payload["paths"].update({"dated": str(dated), "latest": str(latest)})
    _write_json(dated, payload)
    _write_json(latest, payload)
    return dated, latest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register", help="Register a champion or challenger.")
    register.add_argument("--model-id", required=True)
    register.add_argument("--role", choices=sorted(_ALLOWED_ROLES), required=True)
    register.add_argument("--model-path", type=Path, required=True)
    register.add_argument("--metrics-json", type=Path, default=None)
    register.add_argument("--registry-json", type=Path, default=None)
    register.add_argument("--manual-approval-id", default="")
    register.add_argument("--rollback-command", default="")
    register.add_argument("--notes", default="")
    register.add_argument("--max-evidence-age-hours", type=float, default=96.0)
    register.add_argument("--generated-at", default="")
    register.add_argument("--output-dir", default=None)
    register.add_argument("--output-json", type=Path, default=None)
    register.add_argument("--latest-json", type=Path, default=None)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate challenger against champion.")
    evaluate.add_argument("--registry-json", type=Path, required=True)
    evaluate.add_argument("--challenger-id", default="")
    evaluate.add_argument("--primary-metric", default="net_edge_bps")
    evaluate.add_argument("--min-delta", type=float, default=0.0)
    evaluate.add_argument("--max-evidence-age-hours", type=float, default=96.0)
    evaluate.add_argument("--generated-at", default="")
    evaluate.add_argument("--output-dir", default=None)
    evaluate.add_argument("--output-json", type=Path, default=None)
    evaluate.add_argument("--latest-json", type=Path, default=None)

    args = parser.parse_args(argv)
    generated = _cli_generated_at(args.generated_at)
    output_dir = _resolve_output_dir(args.output_dir)
    if args.command == "register":
        payload = build_model_registration(
            model_id=args.model_id,
            role=args.role,
            model_path=args.model_path,
            metrics=_read_json_mapping(args.metrics_json),
            previous_registry=_read_json_mapping(args.registry_json),
            generated_at=generated,
            max_evidence_age_hours=args.max_evidence_age_hours,
            manual_approval_id=str(args.manual_approval_id or "").strip() or None,
            rollback_command=str(args.rollback_command or "").strip() or None,
            notes=args.notes,
        )
        dated, latest = _write_outputs(
            payload=payload,
            kind=_ARTIFACT_TYPE,
            generated_at=generated,
            output_dir=output_dir,
            output_json=args.output_json,
            latest_json=args.latest_json,
        )
    else:
        payload = build_model_evaluation(
            registry=_read_json_mapping(args.registry_json),
            challenger_id=str(args.challenger_id or "").strip() or None,
            primary_metric=args.primary_metric,
            min_delta=args.min_delta,
            generated_at=generated,
            max_evidence_age_hours=args.max_evidence_age_hours,
        )
        dated, latest = _write_outputs(
            payload=payload,
            kind=_EVALUATION_ARTIFACT_TYPE,
            generated_at=generated,
            output_dir=output_dir,
            output_json=args.output_json,
            latest_json=args.latest_json,
        )
    sys.stdout.write(
        json.dumps(
            {
                "status": payload["status"],
                "output_json": str(dated),
                "latest_json": str(latest),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0 if payload["status"] != "blocked" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
