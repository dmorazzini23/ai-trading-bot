"""Convert Hugging Face discoveries into manual research-intake candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.tools.experiment_ledger import build_experiment_ledger

_ARTIFACT_TYPE = "huggingface_candidate_intake"
_SCHEMA_VERSION = "1.0.0"
_DEFAULT_OUTPUT_DIR = "runtime/research_reports/huggingface"
_DEFAULT_ALLOWED_LICENSES = ("apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc-by-4.0", "cc0-1.0")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _compact_date(value: str) -> str:
    return str(value or "").replace("-", "") or _utc_now().strftime("%Y%m%d")


def _read_json(path: Path | None) -> dict[str, Any]:
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


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _csv(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    values = [raw] if isinstance(raw, str) else list(raw)
    parts: list[str] = []
    for value in values:
        for part in str(value or "").split(","):
            token = part.strip()
            if token:
                parts.append(token)
    return parts


def _default_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path(
        _DEFAULT_OUTPUT_DIR,
        default_relative=_DEFAULT_OUTPUT_DIR,
        for_write=True,
    )
    return (
        root / f"hf_candidate_intake_{_compact_date(report_date)}.json",
        root.parent / "latest" / "hf_candidate_intake_latest.json",
    )


def _default_discovery_path() -> Path:
    return resolve_runtime_artifact_path(
        "runtime/research_reports/latest/hf_discovery_latest.json",
        default_relative="runtime/research_reports/latest/hf_discovery_latest.json",
    )


def _candidate_rows(discovery: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = discovery.get("candidates")
    if not isinstance(raw, list):
        return []
    return [dict(row) for row in raw if isinstance(row, Mapping)]


def _selected_ids(candidate_ids: Sequence[str] | None) -> set[str]:
    return {token for token in _csv(candidate_ids or [])}


def _license(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("license") or "").strip().lower()


def _card_present(candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate.get("card_present")
        or candidate.get("model_card_present")
        or candidate.get("dataset_card_present")
    )


def _intake_gate(
    candidate: Mapping[str, Any],
    *,
    allowed_licenses: set[str],
    allow_gated: bool,
    require_license: bool,
    require_card: bool,
) -> tuple[str, list[str], str]:
    reasons: list[str] = []
    license_value = _license(candidate)
    if require_license and not license_value:
        reasons.append("missing_license")
    elif license_value and allowed_licenses and license_value not in allowed_licenses:
        reasons.append("license_not_allowed")
    if bool(candidate.get("gated")) and not allow_gated:
        reasons.append("gated_access_not_allowed")
    if bool(candidate.get("private")):
        reasons.append("private_repo_not_allowed")
    if require_card and not _card_present(candidate):
        reasons.append("missing_model_or_dataset_card")
    if bool(candidate.get("runtime_authority")):
        reasons.append("unexpected_runtime_authority")
    recommended = str(candidate.get("recommended_use") or "inspect")
    if recommended == "ignore":
        reasons.append("discovery_recommended_ignore")
    intended_use = "offline_experiment" if recommended in {"offline_experiment", "candidate_baseline"} else "inspect"
    return ("blocked" if reasons else "ready_for_manual_review", reasons, intended_use)


def _intake_candidate(
    candidate: Mapping[str, Any],
    *,
    source_hash: str,
    allowed_licenses: set[str],
    allow_gated: bool,
    require_license: bool,
    require_card: bool,
) -> dict[str, Any]:
    status, reasons, intended_use = _intake_gate(
        candidate,
        allowed_licenses=allowed_licenses,
        allow_gated=allow_gated,
        require_license=require_license,
        require_card=require_card,
    )
    return {
        "repo_id": str(candidate.get("repo_id") or candidate.get("hf_id") or ""),
        "hf_id": str(candidate.get("hf_id") or candidate.get("repo_id") or ""),
        "resource_type": str(candidate.get("resource_type") or candidate.get("repo_type") or "model"),
        "license": _license(candidate) or None,
        "gated": bool(candidate.get("gated")),
        "private": bool(candidate.get("private")),
        "card_present": _card_present(candidate),
        "source_artifact_hash": source_hash,
        "status": status,
        "blocked_reasons": reasons,
        "research_fit": dict(candidate.get("research_fit")) if isinstance(candidate.get("research_fit"), Mapping) else {},
        "intake": {
            "intended_use": intended_use,
            "materialization_allowed": bool(status == "ready_for_manual_review" and intended_use in {"offline_experiment", "candidate_baseline"}),
            "runtime_use_allowed": False,
            "local_validation_required": True,
            "manual_review_required": True,
        },
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
    }


def build_huggingface_candidate_intake(
    *,
    report_date: str,
    discovery: Mapping[str, Any],
    candidate_ids: Sequence[str] | None = None,
    allowed_licenses: Sequence[str] | None = None,
    allow_gated: bool = False,
    require_license: bool = True,
    require_card: bool = False,
    previous_ledger: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
    notes: str = "",
) -> dict[str, Any]:
    generated = generated_at.astimezone(UTC) if generated_at else _utc_now()
    source_hash = _canonical_hash(discovery)
    selected = _selected_ids(candidate_ids)
    rows = _candidate_rows(discovery)
    if selected:
        rows = [
            row
            for row in rows
            if str(row.get("repo_id") or row.get("hf_id") or "") in selected
        ]
    license_set = {item.lower() for item in (allowed_licenses or _DEFAULT_ALLOWED_LICENSES)}
    candidates = [
        _intake_candidate(
            row,
            source_hash=source_hash,
            allowed_licenses=license_set,
            allow_gated=allow_gated,
            require_license=require_license,
            require_card=require_card,
        )
        for row in rows
    ]
    ready = [row for row in candidates if row["status"] == "ready_for_manual_review"]
    blocked = [row for row in candidates if row["status"] == "blocked"]
    status = "ready_for_manual_review" if ready else "empty" if not candidates else "blocked"
    blocked_reasons = sorted(
        {str(reason) for row in blocked for reason in list(row.get("blocked_reasons") or [])}
    )
    ledger = build_experiment_ledger(
        run_id=f"hf-intake-{_compact_date(report_date)}",
        workflow="huggingface_candidate_intake",
        status="success" if ready else "blocked" if blocked else "dry-run",
        conclusion=(
            f"{len(ready)} Hugging Face candidates ready for manual offline research review"
            if ready
            else "No Hugging Face candidates ready for manual review"
        ),
        config={
            "source_artifact_hash": source_hash,
            "candidate_ids": sorted(selected),
            "accepted_candidate_ids": [row["repo_id"] for row in ready],
            "runtime_authority": False,
        },
        previous_ledger=previous_ledger,
        generated_at=generated,
        reported_complete=bool(ready),
        notes=notes,
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "report_date": str(report_date),
        "generated_at": _iso(generated),
        "status": status,
        "blocked_reasons": blocked_reasons,
        "summary": {
            "selected": len(candidates),
            "ready": len(ready),
            "blocked": len(blocked),
            "accepted_for_offline_experiment": sum(
                1 for row in ready if row["intake"]["intended_use"] == "offline_experiment"
            ),
        },
        "policy": {
            "allowed_licenses": sorted(license_set),
            "allow_gated": bool(allow_gated),
            "require_license": bool(require_license),
            "require_card": bool(require_card),
            "manual_review_required": True,
        },
        "source": {
            "artifact_type": discovery.get("artifact_type"),
            "source_artifact_hash": source_hash,
        },
        "candidates": candidates,
        "experiment_ledger_entry": ledger["latest_run"],
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
        "provider_authority": False,
        "research_only": True,
        "operator_action": "review_ready_candidates_for_offline_experiment" if ready else "no_hf_intake_action",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--discovery-json", type=Path, default=None)
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--candidate-ids", default="")
    parser.add_argument("--allowed-licenses", default=",".join(_DEFAULT_ALLOWED_LICENSES))
    parser.add_argument("--allow-gated", action="store_true")
    parser.add_argument("--no-require-license", action="store_true")
    parser.add_argument("--require-card", action="store_true")
    parser.add_argument("--ledger-json", type=Path, default=None)
    parser.add_argument("--ledger-output-json", type=Path, default=None)
    parser.add_argument("--ledger-latest-json", type=Path, default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)

    discovery_path = args.discovery_json or _default_discovery_path()
    report = build_huggingface_candidate_intake(
        report_date=str(args.report_date),
        discovery=_read_json(discovery_path),
        candidate_ids=[*list(args.candidate_id or []), str(args.candidate_ids or "")],
        allowed_licenses=_csv(args.allowed_licenses),
        allow_gated=bool(args.allow_gated),
        require_license=not bool(args.no_require_license),
        require_card=bool(args.require_card),
        previous_ledger=_read_json(args.ledger_json),
        notes=str(args.notes or ""),
    )
    default_output, default_latest = _default_paths(str(args.report_date))
    output = args.output_json or default_output
    latest = args.latest_json or default_latest
    report.setdefault("paths", {})
    report["paths"].update({"dated": str(output), "latest": str(latest)})
    _write_json(output, report)
    _write_json(latest, report)
    if args.ledger_output_json or args.ledger_latest_json:
        ledger_payload = build_experiment_ledger(
            run_id=report["experiment_ledger_entry"]["run_id"],
            workflow=report["experiment_ledger_entry"]["workflow"],
            status=report["experiment_ledger_entry"]["status"],
            conclusion=report["experiment_ledger_entry"]["conclusion"],
            config=report["experiment_ledger_entry"]["config"],
            previous_ledger=_read_json(args.ledger_json),
            reported_complete=bool(report["experiment_ledger_entry"]["reported_complete"]),
            notes=str(args.notes or ""),
        )
        if args.ledger_output_json:
            _write_json(args.ledger_output_json, ledger_payload)
        if args.ledger_latest_json:
            _write_json(args.ledger_latest_json, ledger_payload)
    sys.stdout.write(
        json.dumps(
            {
                "status": report["status"],
                "ready": report["summary"]["ready"],
                "blocked": report["summary"]["blocked"],
                "output_json": str(output),
                "latest_json": str(latest),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
