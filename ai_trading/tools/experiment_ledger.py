"""Record research experiment outcomes with completion-status guardrails."""

from __future__ import annotations

import argparse
import hashlib
import fcntl
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.runtime.atomic_io import atomic_write_text

_ARTIFACT_TYPE = "experiment_ledger"
_SCHEMA_VERSION = "1.0.0"
_DEFAULT_OUTPUT_DIR = "runtime/research_reports/experiment_ledger"
_ALLOWED_STATUSES = {"success", "failed", "blocked", "dry-run"}


def register_campaign(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze a research budget and future holdout before viewing outcomes."""
    if int(contract['max_trials']) != 1 or len(contract['hypotheses']) != 1:
        raise ValueError('this campaign permits exactly one fixed hypothesis')
    if not str(contract['development_end']) < str(contract['holdout_start']) <= str(contract['holdout_end']):
        raise ValueError('development and holdout overlap')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            state = json.loads(path.read_text())
            if state['contract'] != dict(contract):
                raise ValueError('campaign contract is immutable')
            return dict(state)
        state = {'contract': dict(contract), 'contract_hash': _canonical_hash(contract), 'trials': [], 'registered_at': _iso(_utc_now())}
        atomic_write_text(path, json.dumps(state, indent=2) + '\n')
        return dict(state)


def amend_campaign_feed(path: Path, *, feed: str, reason: str) -> None:
    """Allow an audited feed repair before any outcomes, preserving the budget."""
    if feed != 'sip' or not reason.strip():
        raise ValueError('only a documented SIP data repair is supported')
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text())
        if state['trials'] or state['contract']['feed'] != 'iex':
            raise ValueError('feed amendment requires an unevaluated IEX campaign')
        state.setdefault('amendments', []).append({'previous_contract': dict(state['contract']), 'reason': reason, 'changed_at': _iso(_utc_now())})
        state['contract']['feed'] = feed
        state['contract_hash'] = _canonical_hash(state['contract'])
        atomic_write_text(path, json.dumps(state, indent=2) + '\n')


def claim_campaign_trial(path: Path, *, hypothesis_id: str, evidence_signature: str, evaluation_start: str, evaluation_end: str, quality_passed: bool) -> dict[str, Any]:
    """Reserve budget before evaluation, blocking retries even after a crash."""
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text())
        contract = state['contract']
        if state['contract_hash'] != _canonical_hash(contract):
            raise ValueError('campaign contract hash mismatch')
        if quality_passed is not True:
            raise ValueError('data quality unverified')
        if hypothesis_id not in contract['hypotheses']:
            raise ValueError('unregistered hypothesis')
        if not contract['development_start'] <= evaluation_start <= evaluation_end <= contract['development_end']:
            raise ValueError('evaluation outside frozen development interval')
        if not evidence_signature:
            raise ValueError('missing evidence signature')
        if len(state['trials']) >= contract['max_trials']:
            raise ValueError('campaign budget exhausted')
        state['trials'].append({'hypothesis_id': hypothesis_id, 'evidence_signature': evidence_signature, 'evaluation_start': evaluation_start, 'evaluation_end': evaluation_end, 'status': 'claimed', 'claimed_at': _iso(_utc_now())})
        atomic_write_text(path, json.dumps(state, indent=2) + '\n')
        return dict(state)


def finish_campaign_trial(path: Path, *, evidence_signature: str, decision: str, report_path: Path) -> None:
    """Attach an immutable outcome to the reserved trial without refunding budget."""
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text())
        trial = state['trials'][-1]
        if trial['evidence_signature'] != evidence_signature or trial['status'] != 'claimed':
            raise ValueError('trial does not match an unfinished claim')
        trial.update(status=decision, report_path=str(report_path), report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(), finished_at=_iso(_utc_now()))
        atomic_write_text(path, json.dumps(state, indent=2) + '\n')


def experiment_identity(contract: Mapping[str, Any]) -> str:
    """Identify a hypothesis independently of new data or output locations."""
    return _canonical_hash(contract)


def experiment_permission(
    ledger: Mapping[str, Any], *, experiment_id: str, evidence_signature: str,
    max_failures: int = 2,
) -> dict[str, Any]:
    state = ledger.get("experiments", {}).get(experiment_id, {})
    if int(state.get("failed_evaluations", 0)) >= max(1, max_failures):
        return {"allowed": False, "reason": "experiment_retired", "state": state}
    if evidence_signature in state.get("evaluated_signatures", []):
        return {"allowed": False, "reason": "evidence_already_evaluated", "state": state}
    return {"allowed": True, "reason": "new_evidence", "state": state}


def read_experiment_state(path: Path) -> dict[str, Any]:
    """Fail closed on corrupt retirement state rather than silently restarting."""
    if not path.exists():
        return {"experiments": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("experiments"), dict):
        raise ValueError(f"Invalid experiment state: {path}")
    return payload


def record_research_experiments(
    path: Path, *, evidence_signature: str, outcomes: list[dict[str, Any]],
    max_failures: int = 2,
) -> dict[str, Any]:
    """Record each independent evidence set once, persisting failures across data refreshes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = read_experiment_state(path)
        for outcome in outcomes:
            contract = outcome["contract"]
            key = experiment_identity(contract)
            permission = experiment_permission(
                ledger, experiment_id=key, evidence_signature=evidence_signature,
                max_failures=max_failures,
            )
            if not permission["allowed"] or outcome.get("conclusive") is not True:
                continue
            previous = permission["state"]
            failures = int(previous.get("failed_evaluations", 0)) + int(outcome.get("accepted") is not True)
            ledger["experiments"][key] = {
                "contract": contract,
                "failed_evaluations": failures,
                "evaluated_signatures": [*previous.get("evaluated_signatures", []), evidence_signature],
                "status": "retired" if failures >= max(1, max_failures) else "active",
                "latest_outcome": outcome,
                "updated_at": _iso(_utc_now()),
            }
        ledger.update({"artifact_type": "research_experiment_state", "schema_version": "1.0.0"})
        atomic_write_text(path, json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return ledger


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
