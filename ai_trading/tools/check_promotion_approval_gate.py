from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.governance.paths import resolve_governance_base_path
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _lineage_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    parsed = str(value).strip()
    return parsed or None


def _parse_iso(value: Any) -> datetime | None:
    text = _lineage_text(value)
    if text is None:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for raw in lines:
        text = str(raw or "").strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _resolve_governance_path(value: str | None) -> Path:
    configured = str(value or resolve_governance_base_path())
    return Path(
        resolve_runtime_artifact_path(
            configured,
            default_relative=str(resolve_governance_base_path()),
            for_write=False,
        )
    )


def _resolve_max_age_hours(value: float | None) -> float:
    if value is not None:
        parsed = float(value)
    else:
        parsed = float(
            get_env(
                "AI_TRADING_INSTITUTIONAL_PROMOTION_APPROVAL_MAX_AGE_HOURS",
                get_env("AI_TRADING_PROMOTION_APPROVAL_MAX_AGE_HOURS", 168.0, cast=float),
                cast=float,
            )
        )
    if not (parsed == parsed):  # NaN
        parsed = 168.0
    return max(1.0, min(parsed, 24.0 * 365.0))


def _resolve_max_future_skew_seconds() -> float:
    try:
        parsed = float(
            get_env(
                "AI_TRADING_INSTITUTIONAL_PROMOTION_APPROVAL_MAX_FUTURE_SKEW_SECONDS",
                get_env("AI_TRADING_PROMOTION_APPROVAL_MAX_FUTURE_SKEW_SECONDS", 300.0, cast=float),
                cast=float,
            )
        )
    except (TypeError, ValueError):
        parsed = 300.0
    if not (parsed == parsed):  # NaN
        parsed = 300.0
    return max(0.0, min(parsed, 24.0 * 3600.0))


def _target_filters(
    *,
    strategy: str | None = None,
    model_id: str | None = None,
    release_tag: str | None = None,
    target_commit: str | None = None,
) -> dict[str, str]:
    filters: dict[str, str] = {}
    for key, value in {
        "strategy": strategy,
        "model_id": model_id,
        "release_tag": release_tag,
        "target_commit": target_commit,
    }.items():
        text = _lineage_text(value)
        if text is not None:
            filters[key] = text
    return filters


def _row_matches_targets(row: dict[str, Any], filters: dict[str, str]) -> bool:
    aliases = {
        "strategy": ("strategy", "strategy_id"),
        "model_id": ("model_id", "model", "candidate_model_id"),
        "release_tag": ("release_tag", "tag"),
        "target_commit": ("target_commit", "commit", "sha", "target_sha"),
    }
    for key, expected in filters.items():
        candidates = aliases[key]
        actual = None
        for candidate in candidates:
            actual = _lineage_text(row.get(candidate))
            if actual is not None:
                break
        if actual != expected:
            return False
    return True


def evaluate_promotion_approval_gate(
    *,
    governance_path: str | None = None,
    max_age_hours: float | None = None,
    strategy: str | None = None,
    model_id: str | None = None,
    release_tag: str | None = None,
    target_commit: str | None = None,
) -> dict[str, Any]:
    base = _resolve_governance_path(governance_path)
    approvals_path = base / "promotion_approvals.jsonl"
    promotion_events_path = base / "promotion_events.jsonl"
    approvals = _read_jsonl(approvals_path)
    events = _read_jsonl(promotion_events_path)
    resolved_max_age = _resolve_max_age_hours(max_age_hours)
    max_future_skew_seconds = _resolve_max_future_skew_seconds()
    now = datetime.now(UTC)
    filters = _target_filters(
        strategy=strategy,
        model_id=model_id,
        release_tag=release_tag,
        target_commit=target_commit,
    )

    if not approvals:
        return {
            "ok": False,
            "reason": "approval_records_missing",
            "governance_path": str(base),
            "approvals_path": str(approvals_path),
            "promotion_events_path": str(promotion_events_path),
        }

    if filters:
        approvals = [row for row in approvals if _row_matches_targets(row, filters)]
        events = [row for row in events if _row_matches_targets(row, filters)]
        if not approvals:
            return {
                "ok": False,
                "reason": "target_approval_records_missing",
                "governance_path": str(base),
                "target": filters,
            }
    approvals_by_id = {
        str(row.get("approval_id")): row
        for row in approvals
        if _lineage_text(row.get("approval_id")) is not None
    }

    latest_approval = approvals[-1]
    approval_to_check = latest_approval

    if events:
        latest_event = events[-1]
        if bool(latest_event.get("force", False)):
            return {
                "ok": False,
                "reason": "forced_promotion_disallowed",
                "governance_path": str(base),
                "latest_event": latest_event,
            }
        approval_id = _lineage_text(latest_event.get("approval_id"))
        if approval_id is None:
            return {
                "ok": False,
                "reason": "promotion_event_missing_approval_id",
                "governance_path": str(base),
                "latest_event": latest_event,
            }
        matched = approvals_by_id.get(approval_id)
        if not isinstance(matched, dict):
            return {
                "ok": False,
                "reason": "approval_id_not_found",
                "governance_path": str(base),
                "latest_event": latest_event,
            }
        approval_to_check = matched

    decision = str(approval_to_check.get("decision") or "").strip().lower()
    if decision != "approved":
        return {
            "ok": False,
            "reason": "latest_approval_not_approved",
            "governance_path": str(base),
            "approval": approval_to_check,
        }

    approved_at = _parse_iso(approval_to_check.get("ts"))
    if approved_at is None:
        return {
            "ok": False,
            "reason": "approval_timestamp_invalid",
            "governance_path": str(base),
            "approval": approval_to_check,
        }
    age_hours = max((now - approved_at).total_seconds() / 3600.0, 0.0)
    future_skew_seconds = max((approved_at - now).total_seconds(), 0.0)
    if future_skew_seconds > max_future_skew_seconds:
        return {
            "ok": False,
            "reason": "approval_future_dated",
            "governance_path": str(base),
            "future_skew_seconds": future_skew_seconds,
            "max_future_skew_seconds": max_future_skew_seconds,
            "target": filters,
            "approval": approval_to_check,
        }
    if age_hours > resolved_max_age:
        return {
            "ok": False,
            "reason": "approval_stale",
            "governance_path": str(base),
            "age_hours": age_hours,
            "max_age_hours": resolved_max_age,
            "approval": approval_to_check,
        }

    return {
        "ok": True,
        "reason": "approval_fresh",
        "governance_path": str(base),
        "age_hours": age_hours,
        "max_age_hours": resolved_max_age,
        "future_skew_seconds": future_skew_seconds,
        "max_future_skew_seconds": max_future_skew_seconds,
        "target": filters,
        "approval": approval_to_check,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate promotion approval freshness for deploy gating.",
    )
    parser.add_argument(
        "--governance-path",
        type=str,
        default=None,
        help="Governance artifact directory (default from AI_TRADING_GOVERNANCE_BASE_PATH).",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=None,
        help="Maximum allowed approval age in hours.",
    )
    parser.add_argument("--strategy", type=str, default=None, help="Required strategy id.")
    parser.add_argument("--model-id", type=str, default=None, help="Required model id.")
    parser.add_argument("--release-tag", type=str, default=None, help="Required release tag.")
    parser.add_argument("--target-commit", type=str, default=None, help="Required target commit/SHA.")
    args = parser.parse_args(argv)
    payload = evaluate_promotion_approval_gate(
        governance_path=args.governance_path,
        max_age_hours=args.max_age_hours,
        strategy=args.strategy,
        model_id=args.model_id,
        release_tag=args.release_tag,
        target_commit=args.target_commit,
    )
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
