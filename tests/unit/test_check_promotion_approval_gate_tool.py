from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

from ai_trading.tools.check_promotion_approval_gate import (
    evaluate_promotion_approval_gate,
    main,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def test_approval_gate_passes_with_fresh_approval(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    approvals_path = tmp_path / "promotion_approvals.jsonl"
    _write_jsonl(
        approvals_path,
        [
            {
                "approval_id": "approval-1",
                "ts": now.isoformat(),
                "strategy": "momentum",
                "model_id": "model-1",
                "decision": "approved",
            }
        ],
    )

    payload = evaluate_promotion_approval_gate(
        governance_path=str(tmp_path),
        max_age_hours=24.0,
    )

    assert payload["ok"] is True
    assert payload["reason"] == "approval_fresh"


def test_approval_gate_fails_on_stale_approval(tmp_path: Path) -> None:
    stale = datetime.now(UTC) - timedelta(days=10)
    approvals_path = tmp_path / "promotion_approvals.jsonl"
    _write_jsonl(
        approvals_path,
        [
            {
                "approval_id": "approval-1",
                "ts": stale.isoformat(),
                "strategy": "momentum",
                "model_id": "model-1",
                "decision": "approved",
            }
        ],
    )

    payload = evaluate_promotion_approval_gate(
        governance_path=str(tmp_path),
        max_age_hours=24.0,
    )

    assert payload["ok"] is False
    assert payload["reason"] == "approval_stale"


def test_approval_gate_fails_on_future_dated_approval(tmp_path: Path) -> None:
    future = datetime.now(UTC) + timedelta(minutes=10)
    approvals_path = tmp_path / "promotion_approvals.jsonl"
    _write_jsonl(
        approvals_path,
        [
            {
                "approval_id": "approval-1",
                "ts": future.isoformat(),
                "strategy": "momentum",
                "model_id": "model-1",
                "decision": "approved",
            }
        ],
    )

    payload = evaluate_promotion_approval_gate(
        governance_path=str(tmp_path),
        max_age_hours=24.0,
    )

    assert payload["ok"] is False
    assert payload["reason"] == "approval_future_dated"


def test_approval_gate_fails_when_latest_event_forced(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    _write_jsonl(
        tmp_path / "promotion_approvals.jsonl",
        [
            {
                "approval_id": "approval-1",
                "ts": now.isoformat(),
                "strategy": "momentum",
                "model_id": "model-1",
                "decision": "approved",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "promotion_events.jsonl",
        [
            {
                "ts": now.isoformat(),
                "strategy": "momentum",
                "model_id": "model-1",
                "force": True,
                "approval_id": "approval-1",
            }
        ],
    )

    payload = evaluate_promotion_approval_gate(
        governance_path=str(tmp_path),
        max_age_hours=24.0,
    )
    assert payload["ok"] is False
    assert payload["reason"] == "forced_promotion_disallowed"


def test_approval_gate_scopes_to_requested_target(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    _write_jsonl(
        tmp_path / "promotion_approvals.jsonl",
        [
            {
                "approval_id": "approval-a",
                "ts": now.isoformat(),
                "strategy": "mean-reversion",
                "model_id": "model-a",
                "release_tag": "v1.0.0",
                "target_commit": "aaa",
                "decision": "approved",
            },
            {
                "approval_id": "approval-b",
                "ts": now.isoformat(),
                "strategy": "momentum",
                "model_id": "model-b",
                "release_tag": "v1.0.1",
                "target_commit": "bbb",
                "decision": "approved",
            },
        ],
    )

    payload = evaluate_promotion_approval_gate(
        governance_path=str(tmp_path),
        max_age_hours=24.0,
        strategy="momentum",
        model_id="model-b",
        release_tag="v1.0.1",
        target_commit="bbb",
    )
    missing = evaluate_promotion_approval_gate(
        governance_path=str(tmp_path),
        max_age_hours=24.0,
        strategy="momentum",
        model_id="model-a",
    )

    assert payload["ok"] is True
    assert payload["approval"]["approval_id"] == "approval-b"
    assert missing["ok"] is False
    assert missing["reason"] == "target_approval_records_missing"


def test_approval_gate_cli_returns_nonzero_on_failure(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "promotion_approvals.jsonl", [])
    exit_code = main(
        [
            "--governance-path",
            str(tmp_path),
            "--max-age-hours",
            "24",
        ]
    )
    assert exit_code == 1
