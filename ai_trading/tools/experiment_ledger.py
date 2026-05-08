"""Record research experiment outcomes with completion-status guardrails."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

_ARTIFACT_TYPE = "experiment_ledger"
_SCHEMA_VERSION = "1.0.0"
_DEFAULT_OUTPUT_DIR = "runtime/research_reports/experiment_ledger"
_ALLOWED_STATUSES = {"success", "failed", "blocked", "dry-run"}


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


def _input_hashes(paths: list[Path]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for path in paths:
        expanded = path.expanduser()
        digest = _sha256_file(expanded)
        entry: dict[str, Any] = {
            "path": str(expanded),
            "exists": expanded.is_file(),
            "sha256": digest,
        }
        if expanded.is_file():
            try:
                entry["size_bytes"] = expanded.stat().st_size
            except OSError:
                entry["size_bytes"] = None
        inputs.append(entry)
    return inputs


def _ledger_runs(ledger: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    raw = (ledger or {}).get("runs")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def build_experiment_ledger(
    *,
    run_id: str,
    workflow: str,
    status: str,
    conclusion: str,
    input_paths: list[Path] | None = None,
    config: Mapping[str, Any] | None = None,
    previous_ledger: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
    reported_complete: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    """Return a ledger artifact after recording one research run."""

    generated = generated_at.astimezone(UTC) if generated_at else _utc_now()
    normalized_status = str(status or "").strip().lower()
    blocked_reasons: list[str] = []
    if not run_id.strip():
        blocked_reasons.append("run_id_required")
    if normalized_status not in _ALLOWED_STATUSES:
        blocked_reasons.append("unsupported_status")
    if normalized_status == "failed" and reported_complete:
        blocked_reasons.append("failed_run_cannot_report_complete")
    if normalized_status in {"blocked", "dry-run"} and reported_complete:
        blocked_reasons.append(f"{normalized_status}_run_cannot_report_complete")
    if normalized_status == "success" and not str(conclusion or "").strip():
        blocked_reasons.append("success_conclusion_required")

    config_payload = dict(config or {})
    complete = bool(reported_complete and normalized_status == "success" and not blocked_reasons)
    run = {
        "run_id": run_id.strip(),
        "workflow": str(workflow or "").strip(),
        "status": normalized_status if normalized_status in _ALLOWED_STATUSES else "unsupported",
        "recorded_at": _iso(generated),
        "reported_complete": complete,
        "completion_guard": "ok" if complete or normalized_status != "success" else "not_reported",
        "conclusion": str(conclusion or "").strip(),
        "inputs": _input_hashes(list(input_paths or [])),
        "config": config_payload,
        "config_hash": _canonical_hash(config_payload),
        "notes": str(notes or ""),
    }
    if blocked_reasons:
        run["completion_guard"] = "blocked"
        run["reported_complete"] = False
        run["blocked_reasons"] = blocked_reasons

    runs = _ledger_runs(previous_ledger)
    runs = [existing for existing in runs if str(existing.get("run_id") or "") != run["run_id"]]
    runs.append(run)
    summary = {
        "total_runs": len(runs),
        "success": sum(1 for item in runs if item.get("status") == "success"),
        "failed": sum(1 for item in runs if item.get("status") == "failed"),
        "blocked": sum(1 for item in runs if item.get("status") == "blocked"),
        "dry_run": sum(1 for item in runs if item.get("status") == "dry-run"),
        "reported_complete": sum(1 for item in runs if item.get("reported_complete") is True),
    }
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "generated_at": _iso(generated),
        "status": "blocked" if blocked_reasons else "recorded",
        "blocked_reasons": blocked_reasons,
        "summary": summary,
        "latest_run": run,
        "runs": runs,
        "completion_policy": {
            "complete_status": "success",
            "failed_reported_complete_allowed": False,
            "dry_run_reported_complete_allowed": False,
            "blocked_reported_complete_allowed": False,
        },
    }


def _default_paths(generated_at: datetime, output_dir: Path) -> tuple[Path, Path]:
    return (
        output_dir / f"{_ARTIFACT_TYPE}_{_stamp(generated_at)}.json",
        output_dir / f"{_ARTIFACT_TYPE}_latest.json",
    )


def _resolve_output_dir(raw: str | None) -> Path:
    return resolve_runtime_artifact_path(
        raw or _DEFAULT_OUTPUT_DIR,
        default_relative=_DEFAULT_OUTPUT_DIR,
        for_write=True,
    )


def _write_outputs(
    *,
    payload: dict[str, Any],
    generated_at: datetime,
    output_dir: Path,
    output_json: Path | None,
    latest_json: Path | None,
) -> tuple[Path, Path]:
    default_output, default_latest = _default_paths(generated_at, output_dir)
    dated = output_json or default_output
    latest = latest_json or default_latest
    payload.setdefault("paths", {})
    payload["paths"].update({"dated": str(dated), "latest": str(latest)})
    _write_json(dated, payload)
    _write_json(latest, payload)
    return dated, latest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--status", choices=sorted(_ALLOWED_STATUSES), required=True)
    parser.add_argument("--conclusion", default="")
    parser.add_argument("--input-path", action="append", type=Path, default=[])
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--ledger-json", type=Path, default=None)
    parser.add_argument("--reported-complete", action="store_true")
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)

    generated = _utc_now()
    payload = build_experiment_ledger(
        run_id=args.run_id,
        workflow=args.workflow,
        status=args.status,
        conclusion=args.conclusion,
        input_paths=list(args.input_path or []),
        config=_read_json_mapping(args.config_json),
        previous_ledger=_read_json_mapping(args.ledger_json),
        generated_at=generated,
        reported_complete=bool(args.reported_complete),
        notes=args.notes,
    )
    dated, latest = _write_outputs(
        payload=payload,
        generated_at=generated,
        output_dir=_resolve_output_dir(args.output_dir),
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
