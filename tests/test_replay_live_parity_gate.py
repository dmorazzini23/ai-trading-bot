from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trading.governance.replay_live_parity import (
    summarize_replay_live_parity_gate,
)


def _write_replay_artifact(path: Path, *, ts: datetime, violations_count: int = 0, counterfactual_passed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": ts.isoformat(),
        "rows": 12,
        "orders_submitted": 8,
        "fill_events": 7,
        "violations": [{} for _ in range(int(violations_count))],
        "violations_by_code": {"TEST": int(violations_count)} if violations_count else {},
        "counterfactual": {"passed": bool(counterfactual_passed)},
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_replay_live_parity_gate_passes_with_fresh_clean_replay(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data-root"
    now = datetime.now(UTC)
    artifact = data_root / "runtime" / "replay_outputs" / f"replay_hash_{now.strftime('%Y%m%d')}.json"
    _write_replay_artifact(artifact, ts=now, violations_count=0, counterfactual_passed=True)

    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_root))

    payload = summarize_replay_live_parity_gate(
        oms_lifecycle_parity={
            "enabled": True,
            "available": True,
            "ok": True,
            "total_violations": 0,
        }
    )

    assert payload["ok"] is True
    assert payload["status"] == "pass"
    assert payload["observed"]["replay_fresh"] is True
    assert payload["observed"]["replay_violations_count"] == 0


def test_replay_live_parity_gate_fails_on_stale_replay(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data-root"
    stale_ts = datetime.now(UTC) - timedelta(days=7)
    artifact = data_root / "runtime" / "replay_outputs" / f"replay_hash_{stale_ts.strftime('%Y%m%d')}.json"
    _write_replay_artifact(artifact, ts=stale_ts, violations_count=0, counterfactual_passed=True)

    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_root))
    monkeypatch.setenv("AI_TRADING_REPLAY_LIVE_PARITY_MAX_REPLAY_AGE_HOURS", "24")

    payload = summarize_replay_live_parity_gate(
        oms_lifecycle_parity={
            "enabled": True,
            "available": True,
            "ok": True,
            "total_violations": 0,
        }
    )

    assert payload["ok"] is False
    assert "replay_fresh" in payload["failed_checks"]
    assert payload["status"] == "fail"


def test_replay_live_parity_gate_selects_newest_payload_ts_over_lexical_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-root"
    output_dir = data_root / "runtime" / "replay_outputs"
    now = datetime.now(UTC)
    older_lexically_later = output_dir / "replay_hash_zzzz.json"
    newer_lexically_earlier = output_dir / "replay_hash_aaaa.json"
    _write_replay_artifact(
        older_lexically_later,
        ts=now - timedelta(hours=2),
        violations_count=3,
        counterfactual_passed=False,
    )
    _write_replay_artifact(
        newer_lexically_earlier,
        ts=now - timedelta(hours=1),
        violations_count=0,
        counterfactual_passed=True,
    )

    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_root))

    payload = summarize_replay_live_parity_gate(
        oms_lifecycle_parity={
            "enabled": True,
            "available": True,
            "ok": True,
            "total_violations": 0,
        }
    )

    assert payload["ok"] is True
    assert payload["replay_governance"]["path"] == str(newer_lexically_earlier)
    assert payload["replay_governance"]["violations_count"] == 0
    assert payload["replay_governance"]["ts_source"] == "payload"


def test_replay_live_parity_gate_requires_lifecycle_parity_by_default_outside_pytest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-root"
    now = datetime.now(UTC)
    artifact = data_root / "runtime" / "replay_outputs" / f"replay_hash_{now.strftime('%Y%m%d')}.json"
    _write_replay_artifact(artifact, ts=now, violations_count=0, counterfactual_passed=True)

    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_root))
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv(
        "AI_TRADING_REPLAY_LIVE_PARITY_REQUIRE_OMS_LIFECYCLE_PARITY",
        raising=False,
    )

    payload = summarize_replay_live_parity_gate(
        oms_lifecycle_parity={
            "enabled": True,
            "available": True,
            "ok": False,
            "total_violations": 2,
        }
    )

    assert payload["ok"] is False
    assert "oms_lifecycle_parity_consistent" in payload["failed_checks"]
