"""Shared research-reset policy for schedulers and training entrypoints."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

POLICY_PATH = Path(__file__).resolve().parents[2] / "config/research_reset.json"


def reset_policy(path: Path, now: datetime) -> dict[str, Any] | None:
    """Fail closed for broken policy; review date never automatically resumes work."""
    policy = json.loads(path.read_text())
    if not isinstance(policy, dict):
        raise ValueError("invalid research reset policy")
    if policy.get("enabled") is False:
        return None
    start = datetime.fromisoformat(policy["start_date"]).replace(tzinfo=UTC)
    review = datetime.fromisoformat(policy["review_date"]).replace(tzinfo=UTC)
    if policy.get("enabled") is not True or review <= start:
        raise ValueError("invalid research reset policy")
    if policy.get("resume_policy") != "explicit_review_required":
        raise ValueError("reset requires explicit review before resuming search")
    if now < start:
        return None
    return {**policy, "phase": "review_due" if now >= review else "evidence_reset"}


def training_block_reason(now: datetime | None = None) -> str | None:
    """Require an explicit valid release before scheduled/unregistered training."""
    try:
        policy = reset_policy(POLICY_PATH, now or datetime.now(UTC))
    except (OSError, ValueError, KeyError, TypeError):
        return "research_reset_policy_invalid"
    return "research_reset_active" if policy is not None else None
