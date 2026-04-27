from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import ai_trading.health_payload as health_payload


def test_safe_helpers_return_defaults_for_bad_inputs(monkeypatch) -> None:
    import ai_trading.config.management as config_management

    monkeypatch.setattr(
        config_management,
        "get_env",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("env unavailable")),
    )

    assert health_payload._safe_observe(lambda: (_ for _ in ()).throw(RuntimeError("boom")), {"ok": False}) == {
        "ok": False,
    }
    assert health_payload._timestamp_age_seconds("   ") is None
    assert health_payload._timestamp_age_seconds("2026-04-27T00:00:00Z") is not None
    assert health_payload._timestamp_age_seconds("2026-04-27T00:00:00") is not None
    assert health_payload._safe_nonnegative_int("-5") == 0
    assert health_payload._safe_nonnegative_int("not-an-int") == 0
    assert health_payload._env_bool("BROKEN_BOOL", True) is True
    assert health_payload._env_float("BROKEN_FLOAT", 2.5) == 2.5
    assert health_payload._health_snapshot_ttl_seconds("broken", 999.0) == 300.0


def test_contract_gate_status_reports_required_and_observed_failures() -> None:
    passed = health_payload._build_contract_gate_status(
        {"enabled": True, "ok": True},
        required=True,
        failure_reason="failed",
        attention_flag="flagged",
        action="repair",
    )
    observed = health_payload._build_contract_gate_status(
        {
            "enabled": True,
            "ok": False,
            "reason": "drift detected",
            "total_violations": 3,
        },
        required=False,
        failure_reason="failed",
        attention_flag="flagged",
        action="repair",
    )
    required = health_payload._build_contract_gate_status(
        {
            "enabled": True,
            "ok": False,
            "reason": "still drifting",
            "total_violations": 4,
        },
        required=True,
        failure_reason="failed",
        attention_flag="flagged",
        action="repair",
    )

    assert passed == {
        "enabled": True,
        "required": True,
        "ok": True,
        "status": "passed",
        "failure_observed": False,
    }
    assert observed == {
        "enabled": True,
        "required": False,
        "ok": False,
        "status": "observed_failure",
        "failure_observed": True,
        "reason": "failed",
        "attention_flag": "flagged",
        "action": "repair",
        "detail": "drift detected",
        "total_violations": 3,
    }
    assert required["status"] == "required_failed"
    assert required["detail"] == "still drifting"
    assert required["total_violations"] == 4


def test_cached_background_snapshot_returns_fresh_stale_and_placeholder(monkeypatch) -> None:
    class DeferredThread:
        def __init__(self, *, target: Any, name: str | None = None, daemon: bool | None = None) -> None:
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self) -> None:
            return None

    monkeypatch.setattr(health_payload, "Thread", DeferredThread)
    monkeypatch.setattr(health_payload.pytime, "monotonic", lambda: 100.0)
    health_payload._HEALTH_SNAPSHOT_CACHE.clear()
    health_payload._HEALTH_SNAPSHOT_CACHE["fresh"] = {
        "value": {"ok": True},
        "updated_mono": 95.0,
        "refreshing": False,
    }
    health_payload._HEALTH_SNAPSHOT_CACHE["stale"] = {
        "value": {"ok": False},
        "updated_mono": 1.0,
        "refreshing": False,
    }

    fresh = health_payload._cached_background_snapshot(
        name="fresh",
        ttl_seconds=10.0,
        placeholder={"ok": False},
        builder=lambda: {"ok": True},
    )
    stale = health_payload._cached_background_snapshot(
        name="stale",
        ttl_seconds=10.0,
        placeholder={"ok": False},
        builder=lambda: {"ok": True},
    )
    placeholder = health_payload._cached_background_snapshot(
        name="empty",
        ttl_seconds=10.0,
        placeholder={"ok": False},
        builder=lambda: {"ok": True},
    )

    assert fresh == {"ok": True, "refreshing": False}
    assert stale == {"ok": False, "refreshing": True, "stale": True}
    assert placeholder == {"ok": False, "refreshing": True, "reason": "warming_up"}


def test_database_readiness_normalizes_boolean_store_result_and_closes(monkeypatch) -> None:
    import ai_trading.config.management as config_management
    import ai_trading.oms.event_store as event_store

    calls: dict[str, Any] = {}

    def fake_get_env(name: str, default: Any = "", **_kwargs: Any) -> Any:
        if name == "DATABASE_URL":
            return "sqlite:///health.db"
        if name == "AI_TRADING_OMS_EXPECTED_ALEMBIC_REVISION":
            return "rev-test"
        return default

    class FakeEventStore:
        def __init__(self, *, path: str | None, url: str | None) -> None:
            calls["path"] = path
            calls["url"] = url

        def is_healthy(self, *, expected_revision: str) -> bool:
            calls["expected_revision"] = expected_revision
            return False

        def close(self) -> None:
            calls["closed"] = True
            raise RuntimeError("close is best effort")

    monkeypatch.setattr(health_payload, "_env_bool", lambda _name, default: True)
    monkeypatch.setattr(config_management, "get_env", fake_get_env)
    monkeypatch.setattr(event_store, "EventStore", FakeEventStore)

    payload = health_payload._database_readiness_snapshot()

    assert payload == {
        "ok": False,
        "enabled": True,
        "configured": True,
        "expected_revision": "rev-test",
    }
    assert calls == {
        "path": None,
        "url": "sqlite:///health.db",
        "expected_revision": "rev-test",
        "closed": True,
    }


def test_database_readiness_reports_configured_store_exception(monkeypatch) -> None:
    import ai_trading.config.management as config_management
    import ai_trading.oms.event_store as event_store

    def fake_get_env(name: str, default: Any = "", **_kwargs: Any) -> Any:
        if name == "AI_TRADING_OMS_INTENT_STORE_PATH":
            return "/tmp/intent-store.sqlite"
        if name == "AI_TRADING_OMS_EXPECTED_ALEMBIC_REVISION":
            return "rev-broken"
        return default

    class BrokenEventStore:
        def __init__(self, *, path: str | None, url: str | None) -> None:
            raise OSError(f"cannot open {path or url}")

    monkeypatch.setattr(health_payload, "_env_bool", lambda _name, default: True)
    monkeypatch.setattr(config_management, "get_env", fake_get_env)
    monkeypatch.setattr(event_store, "EventStore", BrokenEventStore)

    payload = health_payload._database_readiness_snapshot()

    assert payload["enabled"] is True
    assert payload["configured"] is True
    assert payload["ok"] is False
    assert payload["connected"] is False
    assert payload["expected_revision"] == "rev-broken"
    assert "cannot open /tmp/intent-store.sqlite" in payload["error"]


def test_oms_invariant_snapshots_preserve_success_and_unavailable_errors(monkeypatch) -> None:
    import ai_trading.config.management as config_management
    import ai_trading.oms.invariants as oms_invariants

    def fake_get_env(name: str, default: Any = "", **_kwargs: Any) -> Any:
        values = {
            "DATABASE_URL": "postgresql://health.example/db",
            "AI_TRADING_OMS_INTENT_STORE_PATH": "/tmp/intents.sqlite",
            "AI_TRADING_HEALTH_OMS_INVARIANTS_LIMIT": 7,
            "AI_TRADING_HEALTH_OMS_LIFECYCLE_PARITY_LIMIT": 11,
        }
        return values.get(name, default)

    def reconciliation_summary(
        *,
        database_url: str | None,
        intent_store_path: str | None,
        limit: int,
    ) -> dict[str, Any]:
        assert database_url == "postgresql://health.example/db"
        assert intent_store_path == "/tmp/intents.sqlite"
        assert limit == 7
        return {"available": True, "ok": False, "total_violations": 2}

    def lifecycle_summary(
        *,
        database_url: str | None,
        intent_store_path: str | None,
        limit: int,
    ) -> dict[str, Any]:
        assert database_url == "postgresql://health.example/db"
        assert intent_store_path == "/tmp/intents.sqlite"
        assert limit == 11
        return {"available": True, "ok": True, "total_violations": 0}

    monkeypatch.setattr(health_payload, "_env_bool", lambda _name, default: True)
    monkeypatch.setattr(config_management, "get_env", fake_get_env)
    monkeypatch.setattr(
        oms_invariants,
        "evaluate_oms_reconciliation_invariants",
        reconciliation_summary,
    )
    monkeypatch.setattr(
        oms_invariants,
        "evaluate_oms_lifecycle_parity_invariants",
        lifecycle_summary,
    )

    reconciliation = health_payload._oms_invariants_snapshot()
    lifecycle = health_payload._oms_lifecycle_parity_snapshot()

    assert reconciliation == {
        "available": True,
        "ok": False,
        "total_violations": 2,
        "enabled": True,
    }
    assert lifecycle == {
        "available": True,
        "ok": True,
        "total_violations": 0,
        "enabled": True,
    }

    monkeypatch.setattr(
        oms_invariants,
        "evaluate_oms_reconciliation_invariants",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("reconciliation unavailable")),
    )
    monkeypatch.setattr(
        oms_invariants,
        "evaluate_oms_lifecycle_parity_invariants",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("lifecycle unavailable")),
    )

    reconciliation_error = health_payload._oms_invariants_snapshot()
    lifecycle_error = health_payload._oms_lifecycle_parity_snapshot()

    assert reconciliation_error["reason"] == "oms_invariants_unavailable"
    assert reconciliation_error["ok"] is False
    assert reconciliation_error["available"] is False
    assert reconciliation_error["error"] == "reconciliation unavailable"
    assert lifecycle_error["reason"] == "oms_lifecycle_parity_unavailable"
    assert lifecycle_error["ok"] is False
    assert lifecycle_error["available"] is False
    assert lifecycle_error["error"] == "lifecycle unavailable"


def test_runtime_performance_snapshot_filters_malformed_rows(tmp_path, monkeypatch) -> None:
    import ai_trading.config.management as config_management

    report_path = tmp_path / "runtime_perf.json"
    report_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T00:00:00Z",
                "source": "unit-test",
                "go_no_go": {"gate_passed": False},
                "oms_event_tca": {
                    "enabled": True,
                    "available": True,
                    "filled_events": 4,
                    "parent_execution_kpis_by_scope": [
                        {"scope": "core"},
                        "bad-row",
                    ],
                    "event_outcomes_by_scope": [
                        {"outcome": "filled"},
                        None,
                    ],
                    "submit_reject_reasons_top": [
                        {"reason": "min_notional"},
                        3,
                    ],
                    "cancel_reasons_top": [
                        {"reason": "timeout"},
                        [],
                    ],
                    "realized_slippage_decomposition": {"arrival_bps": 1.2},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config_management, "get_env", lambda *_args, **_kwargs: str(report_path))
    monkeypatch.setattr(
        health_payload,
        "resolve_runtime_artifact_path",
        lambda configured_path, *, default_relative: Path(configured_path),
    )

    payload = health_payload._runtime_performance_snapshot()

    assert payload["available"] is True
    assert payload["generated_at"] == "2026-04-27T00:00:00Z"
    assert payload["source"] == "unit-test"
    assert payload["go_no_go"] == {"gate_passed": False}
    assert payload["oms_event_tca"]["enabled"] is True
    assert payload["oms_event_tca"]["available"] is True
    assert payload["oms_event_tca"]["parent_execution_kpis_by_scope"] == [{"scope": "core"}]
    assert payload["oms_event_tca"]["event_outcomes_by_scope"] == [{"outcome": "filled"}]
    assert payload["oms_event_tca"]["submit_reject_reasons_top"] == [{"reason": "min_notional"}]
    assert payload["oms_event_tca"]["cancel_reasons_top"] == [{"reason": "timeout"}]
    assert payload["oms_event_tca"]["realized_slippage_decomposition"] == {"arrival_bps": 1.2}


def test_manual_override_snapshot_degrades_to_default_path_on_env_error(tmp_path, monkeypatch) -> None:
    import ai_trading.config.management as config_management

    resolved_path = tmp_path / "missing_toggles.json"
    monkeypatch.setattr(
        config_management,
        "get_env",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("env unavailable")),
    )
    monkeypatch.setattr(
        health_payload,
        "resolve_runtime_artifact_path",
        lambda configured_path, *, default_relative: resolved_path,
    )

    payload = health_payload._manual_override_snapshot()

    assert payload == {
        "available": False,
        "path": str(resolved_path),
        "state": {},
    }


def test_replay_live_parity_gate_cache_bypasses_stale_lifecycle_snapshot(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def direct_snapshot(*, oms_lifecycle_parity: Any = None) -> dict[str, Any]:
        calls["oms_lifecycle_parity"] = oms_lifecycle_parity
        return {"enabled": True, "available": False, "ok": False, "reason": "warming_up"}

    monkeypatch.setattr(health_payload, "_health_snapshot_cache_enabled", lambda: True)
    monkeypatch.setattr(health_payload, "_replay_live_parity_gate_snapshot", direct_snapshot)
    monkeypatch.setattr(
        health_payload,
        "_cached_background_snapshot",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("cache should be bypassed")),
    )

    lifecycle = {"enabled": True, "available": False, "ok": False}
    payload = health_payload._replay_live_parity_gate_snapshot_cached(
        oms_lifecycle_parity=lifecycle,
    )

    assert payload == {
        "enabled": True,
        "available": False,
        "ok": False,
        "reason": "warming_up",
    }
    assert calls["oms_lifecycle_parity"] is lifecycle
