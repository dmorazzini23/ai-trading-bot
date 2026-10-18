from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from ai_trading.governance import replay_live_parity
from ai_trading.governance.replay_live_parity import (
    REPLAY_GOVERNANCE_SCHEMA_VERSION,
    summarize_replay_live_parity_gate,
)

_ET = ZoneInfo("America/New_York")
_LIFECYCLE_OK = {
    "enabled": True,
    "available": True,
    "ok": True,
    "total_violations": 0,
}


def _write_replay_artifact(
    path: Path,
    *,
    ts: datetime,
    violations_count: int = 0,
    cap_adjustments_count: int = 0,
    source_fresh: bool = True,
    counterfactual_passed: bool | None = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": REPLAY_GOVERNANCE_SCHEMA_VERSION,
        "policy_hash": "policy-test",
        "ts": ts.isoformat(),
        "rows": 12,
        "orders_submitted": 8,
        "fill_events": 7,
        "violations": [{} for _ in range(int(violations_count))],
        "violations_by_code": {"TEST": int(violations_count)} if violations_count else {},
        "cap_adjustments": [{} for _ in range(int(cap_adjustments_count))],
        "source_data": {
            "fresh": bool(source_fresh),
            "age_hours": 1.0 if source_fresh else 200.0,
            "max_age_hours": 96.0,
        },
    }
    if counterfactual_passed is not None:
        payload["counterfactual"] = {
            "passed": bool(counterfactual_passed),
            "candidate": {
                "sample_count": 71,
                "net_edge_bps": -17.89,
            },
            "required": {
                "min_samples": 250,
                "min_net_edge_bps": 0.0,
            },
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
    assert payload["observed"]["replay_counterfactual_candidate_samples"] == 71
    assert payload["observed"]["replay_counterfactual_required_samples"] == 250
    assert (
        payload["observed"]["replay_counterfactual_candidate_net_edge_bps"]
        == -17.89
    )
    assert (
        payload["observed"]["replay_counterfactual_required_net_edge_bps"]
        == 0.0
    )


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


def test_replay_live_parity_gate_rejects_mtime_only_freshness(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-root"
    artifact = data_root / "runtime" / "replay_outputs" / "replay_hash_missing_ts.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "rows": 12,
                "orders_submitted": 8,
                "fill_events": 7,
                "violations": [],
                "counterfactual": {"passed": True},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
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

    assert payload["ok"] is False
    assert "replay_fresh" in payload["failed_checks"]
    assert payload["replay_governance"]["ts_source"] is None
    assert payload["replay_governance"]["reason"] == "replay_governance_artifact_missing_payload_ts"


def test_replay_live_parity_gate_fails_on_future_dated_replay(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-root"
    future_ts = datetime.now(UTC) + timedelta(minutes=10)
    artifact = data_root / "runtime" / "replay_outputs" / "replay_hash_future.json"
    _write_replay_artifact(artifact, ts=future_ts, violations_count=0, counterfactual_passed=True)

    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_root))
    monkeypatch.setenv("AI_TRADING_REPLAY_LIVE_PARITY_MAX_FUTURE_SKEW_SECONDS", "60")
    monkeypatch.setenv("AI_TRADING_REPLAY_LIVE_PARITY_REQUIRE_FRESH_REPLAY", "0")

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
    assert payload["observed"]["replay_future_dated"] is True


def test_replay_live_parity_gate_requires_explicit_counterfactual_pass(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-root"
    now = datetime.now(UTC)
    artifact = data_root / "runtime" / "replay_outputs" / "replay_hash_missing_cf.json"
    _write_replay_artifact(artifact, ts=now, violations_count=0, counterfactual_passed=None)

    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_root))
    monkeypatch.setenv("AI_TRADING_REPLAY_LIVE_PARITY_REQUIRE_COUNTERFACTUAL_PASSED", "1")

    payload = summarize_replay_live_parity_gate(
        oms_lifecycle_parity={
            "enabled": True,
            "available": True,
            "ok": True,
            "total_violations": 0,
        }
    )

    assert payload["ok"] is False
    assert "replay_counterfactual" in payload["failed_checks"]
    assert payload["observed"]["replay_counterfactual_passed"] is False


def test_replay_live_parity_gate_requires_counterfactual_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-root"
    now = datetime.now(UTC)
    artifact = data_root / "runtime" / "replay_outputs" / "replay_hash_failed_cf.json"
    _write_replay_artifact(artifact, ts=now, violations_count=0, counterfactual_passed=False)

    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_root))

    payload = summarize_replay_live_parity_gate(
        oms_lifecycle_parity={
            "enabled": True,
            "available": True,
            "ok": True,
            "total_violations": 0,
        }
    )

    assert payload["ok"] is False
    assert payload["observed"]["replay_counterfactual_passed"] is False
    assert payload["thresholds"]["require_counterfactual_passed"] is True


def test_replay_live_parity_gate_rejects_legacy_artifact_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-root"
    artifact = data_root / "runtime" / "replay_outputs" / "replay_hash_legacy.json"
    _write_replay_artifact(artifact, ts=datetime.now(UTC))
    raw = json.loads(artifact.read_text(encoding="utf-8"))
    raw.pop("schema_version")
    raw.pop("policy_hash")
    artifact.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_root))

    payload = summarize_replay_live_parity_gate(
        oms_lifecycle_parity={
            "enabled": True,
            "available": True,
            "ok": True,
            "total_violations": 0,
        }
    )

    assert payload["ok"] is False
    assert "replay_contract" in payload["failed_checks"]


def test_replay_live_parity_gate_rejects_policy_hash_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-root"
    artifact = data_root / "runtime" / "replay_outputs" / "replay_hash_policy.json"
    _write_replay_artifact(artifact, ts=datetime.now(UTC))
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_root))

    payload = summarize_replay_live_parity_gate(
        expected_policy_hash="different-policy",
        oms_lifecycle_parity={
            "enabled": True,
            "available": True,
            "ok": True,
            "total_violations": 0,
        },
    )

    assert payload["ok"] is False
    assert "replay_contract" in payload["failed_checks"]


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


def test_replay_live_parity_gate_reparses_only_changed_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-root"
    output_dir = data_root / "runtime" / "replay_outputs"
    now = datetime.now(UTC)
    older = output_dir / "replay_hash_older.json"
    newer = output_dir / "replay_hash_newer.json"
    _write_replay_artifact(older, ts=now - timedelta(hours=3), violations_count=2)
    _write_replay_artifact(newer, ts=now - timedelta(hours=1))
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_root))

    parsed_paths: list[str] = []
    original_read = replay_live_parity._read_replay_governance_artifact

    def _counting_read(path: Path):
        parsed_paths.append(path.name)
        return original_read(path)

    monkeypatch.setattr(
        replay_live_parity,
        "_read_replay_governance_artifact",
        _counting_read,
    )

    first = summarize_replay_live_parity_gate(oms_lifecycle_parity=_LIFECYCLE_OK)
    assert first["replay_governance"]["path"] == str(newer)
    assert sorted(parsed_paths) == ["replay_hash_newer.json", "replay_hash_older.json"]

    parsed_paths.clear()
    second = summarize_replay_live_parity_gate(oms_lifecycle_parity=_LIFECYCLE_OK)
    assert parsed_paths == []
    assert second["replay_governance"] == first["replay_governance"]

    # Rewriting the older artifact with a newer payload ts must re-rank it.
    _write_replay_artifact(older, ts=now - timedelta(minutes=5), violations_count=2)
    bumped_ns = older.stat().st_mtime_ns + 1_000_000_000
    os.utime(older, ns=(bumped_ns, bumped_ns))
    parsed_paths.clear()
    third = summarize_replay_live_parity_gate(oms_lifecycle_parity=_LIFECYCLE_OK)
    assert parsed_paths == ["replay_hash_older.json"]
    assert third["replay_governance"]["path"] == str(older)
    assert third["replay_governance"]["violations_count"] == 2

    # A corrupted artifact drops out of the ranking instead of serving its cached ts.
    older.write_text("{not json", encoding="utf-8")
    fourth = summarize_replay_live_parity_gate(oms_lifecycle_parity=_LIFECYCLE_OK)
    assert fourth["replay_governance"]["path"] == str(newer)


def test_replay_live_parity_gate_reports_cap_adjustments_without_failing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-root"
    now = datetime.now(UTC)
    artifact = data_root / "runtime" / "replay_outputs" / "replay_hash_cap_adjusted.json"
    _write_replay_artifact(
        artifact,
        ts=now,
        violations_count=0,
        cap_adjustments_count=2,
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
    assert "replay_violations" not in payload["failed_checks"]
    assert payload["observed"]["replay_violations_count"] == 0
    assert payload["observed"]["replay_cap_adjustments_count"] == 2
    assert payload["replay_governance"]["cap_adjustments_count"] == 2
    assert "cap_adjustment" not in payload["replay_governance"]["violations_by_code"]


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


def test_replay_live_parity_gate_fails_on_stale_source_data(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-root"
    now = datetime.now(UTC)
    artifact = data_root / "runtime" / "replay_outputs" / "replay_hash_stale_source.json"
    _write_replay_artifact(
        artifact,
        ts=now,
        violations_count=0,
        counterfactual_passed=True,
        source_fresh=False,
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

    assert payload["ok"] is False
    assert "replay_source_fresh" in payload["failed_checks"]
    assert payload["observed"]["replay_source_data_fresh"] is False
    assert payload["observed"]["replay_source_age_hours"] == 200.0


def _gate_at(
    monkeypatch,
    tmp_path: Path,
    *,
    artifact_ts: datetime,
    now: datetime,
) -> dict:
    data_root = tmp_path / "data-root"
    artifact = data_root / "runtime" / "replay_outputs" / "replay_hash_session.json"
    _write_replay_artifact(artifact, ts=artifact_ts.astimezone(UTC))
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_root))
    monkeypatch.setattr(
        replay_live_parity,
        "_utcnow",
        lambda: now.astimezone(UTC),
    )
    return summarize_replay_live_parity_gate(oms_lifecycle_parity=_LIFECYCLE_OK)


@pytest.mark.parametrize(
    ("artifact_ts", "now", "sessions_since", "covered"),
    [
        # Friday-evening artifact, checked Sunday afternoon.
        (
            datetime(2026, 10, 16, 20, 0, tzinfo=_ET),
            datetime(2026, 10, 18, 12, 0, tzinfo=_ET),
            0,
            "2026-10-16",
        ),
        # Live 2026-10-18 case: Saturday 04:59 UTC artifact, 35.1h wall-clock.
        (
            datetime(2026, 10, 17, 4, 59, tzinfo=UTC),
            datetime(2026, 10, 18, 16, 5, tzinfo=UTC),
            0,
            "2026-10-16",
        ),
        # Regular Friday 05:20 ET timer artifact (covers Thursday) on Sunday.
        (
            datetime(2026, 10, 16, 5, 20, tzinfo=_ET),
            datetime(2026, 10, 18, 12, 0, tzinfo=_ET),
            1,
            "2026-10-15",
        ),
    ],
)
def test_replay_fresh_holds_over_weekend_on_exchange_calendar(
    monkeypatch,
    tmp_path: Path,
    artifact_ts: datetime,
    now: datetime,
    sessions_since: int,
    covered: str,
) -> None:
    payload = _gate_at(monkeypatch, tmp_path, artifact_ts=artifact_ts, now=now)

    observed = payload["observed"]
    assert observed["replay_age_hours"] > 24.0
    assert observed["replay_fresh"] is True
    assert payload["checks"]["replay_fresh"] is True
    assert "replay_fresh" not in payload["failed_checks"]
    assert observed["replay_age_basis"] == "exchange_calendar"
    assert observed["replay_session_age_hours"] <= 24.0
    assert observed["replay_non_trading_hours_excluded"] > 0.0
    assert observed["sessions_since_replay"] == sessions_since
    assert observed["replay_covered_session"] == covered
    assert observed["replay_freshness_calendar"] == "XNYS"
    assert payload["thresholds"]["max_replay_age_hours"] == 24.0
    assert payload["thresholds"]["require_fresh_replay"] is True


def test_replay_stale_monday_after_open_with_friday_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # Friday 05:20 ET artifact covers Thursday; Friday's session completed and
    # Monday's refresh is missing, so Monday after the open it is stale.
    payload = _gate_at(
        monkeypatch,
        tmp_path,
        artifact_ts=datetime(2026, 10, 16, 5, 20, tzinfo=_ET),
        now=datetime(2026, 10, 19, 10, 0, tzinfo=_ET),
    )

    observed = payload["observed"]
    assert observed["replay_fresh"] is False
    assert "replay_fresh" in payload["failed_checks"]
    assert observed["sessions_since_replay"] == 1
    assert observed["replay_session_age_hours"] == pytest.approx(28.0 + 2.0 / 3.0)
    assert observed["replay_covered_session"] == "2026-10-15"


def test_replay_fresh_across_market_holiday(monkeypatch, tmp_path: Path) -> None:
    # Wednesday 2026-11-25 artifact; Thursday 2026-11-26 is Thanksgiving.
    payload = _gate_at(
        monkeypatch,
        tmp_path,
        artifact_ts=datetime(2026, 11, 25, 5, 20, tzinfo=_ET),
        now=datetime(2026, 11, 27, 5, 0, tzinfo=_ET),
    )

    observed = payload["observed"]
    assert observed["replay_age_hours"] > 47.0
    assert observed["replay_fresh"] is True
    assert "replay_fresh" not in payload["failed_checks"]
    assert observed["replay_non_trading_hours_excluded"] == pytest.approx(24.0)
    assert observed["sessions_since_replay"] == 1
    assert observed["replay_covered_session"] == "2026-11-24"


def test_replay_stale_when_holiday_span_misses_two_sessions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # Written before Wednesday's close; by Friday's early close (13:00 ET) two
    # completed sessions are missing even though session age is under 24h.
    payload = _gate_at(
        monkeypatch,
        tmp_path,
        artifact_ts=datetime(2026, 11, 25, 15, 0, tzinfo=_ET),
        now=datetime(2026, 11, 27, 13, 30, tzinfo=_ET),
    )

    observed = payload["observed"]
    assert observed["replay_session_age_hours"] < 24.0
    assert observed["sessions_since_replay"] == 2
    assert observed["replay_fresh"] is False
    assert "replay_fresh" in payload["failed_checks"]


@pytest.mark.parametrize(
    ("now", "expected_fresh"),
    [
        # Same threshold as wall-clock age on trading days: fresh at 23h40m...
        (datetime(2026, 10, 14, 5, 0, tzinfo=_ET), True),
        # ...and stale once 24h pass after missing Tuesday's session.
        (datetime(2026, 10, 14, 6, 0, tzinfo=_ET), False),
        (datetime(2026, 10, 15, 12, 0, tzinfo=_ET), False),
    ],
)
def test_replay_weekday_freshness_matches_wall_clock(
    monkeypatch,
    tmp_path: Path,
    now: datetime,
    expected_fresh: bool,
) -> None:
    payload = _gate_at(
        monkeypatch,
        tmp_path,
        artifact_ts=datetime(2026, 10, 13, 5, 20, tzinfo=_ET),
        now=now,
    )

    observed = payload["observed"]
    assert observed["replay_non_trading_hours_excluded"] == 0.0
    assert observed["replay_session_age_hours"] == pytest.approx(
        observed["replay_age_hours"]
    )
    assert observed["replay_fresh"] is expected_fresh
    assert ("replay_fresh" in payload["failed_checks"]) is (not expected_fresh)


def test_replay_session_freshness_honors_max_age_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AI_TRADING_REPLAY_LIVE_PARITY_MAX_REPLAY_AGE_HOURS", "2")
    payload = _gate_at(
        monkeypatch,
        tmp_path,
        artifact_ts=datetime(2026, 10, 16, 20, 0, tzinfo=_ET),
        now=datetime(2026, 10, 18, 12, 0, tzinfo=_ET),
    )

    assert payload["thresholds"]["max_replay_age_hours"] == 2.0
    assert payload["observed"]["replay_session_age_hours"] == pytest.approx(4.0)
    assert payload["observed"]["replay_fresh"] is False
    assert "replay_fresh" in payload["failed_checks"]


def test_replay_freshness_require_override_waives_stale_check(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AI_TRADING_REPLAY_LIVE_PARITY_REQUIRE_FRESH_REPLAY", "0")
    payload = _gate_at(
        monkeypatch,
        tmp_path,
        artifact_ts=datetime(2026, 10, 16, 5, 20, tzinfo=_ET),
        now=datetime(2026, 10, 19, 10, 0, tzinfo=_ET),
    )

    assert payload["observed"]["replay_fresh"] is False
    assert payload["checks"]["replay_fresh"] is True
    assert payload["thresholds"]["require_fresh_replay"] is False


def test_replay_freshness_falls_back_to_wall_clock_without_calendar(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def _calendar_down(_session_date):
        raise RuntimeError("calendar unavailable")

    monkeypatch.setattr(
        replay_live_parity,
        "_exchange_session_close_utc",
        _calendar_down,
    )
    payload = _gate_at(
        monkeypatch,
        tmp_path,
        artifact_ts=datetime(2026, 10, 16, 20, 0, tzinfo=_ET),
        now=datetime(2026, 10, 18, 12, 0, tzinfo=_ET),
    )

    observed = payload["observed"]
    assert observed["replay_age_basis"] == "wall_clock"
    assert observed["sessions_since_replay"] is None
    assert observed["replay_fresh"] is False
    assert "replay_fresh" in payload["failed_checks"]
