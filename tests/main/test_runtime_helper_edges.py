from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_trading import main


def test_info_log_ttl_resolution_clamps_and_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_env", lambda *_args, **_kwargs: -5)
    assert main._resolve_info_log_ttl_seconds("X", 10.0) == 0.0

    monkeypatch.setattr(main, "get_env", lambda *_args, **_kwargs: 7200)
    assert main._resolve_info_log_ttl_seconds("X", 10.0) == 3600.0

    def _raise(*_args, **_kwargs):
        raise ValueError("bad env")

    monkeypatch.setattr(main, "get_env", _raise)
    assert main._resolve_info_log_ttl_seconds("X", 12.5) == 12.5


def test_should_emit_info_log_uses_monotonic_ttl(monkeypatch) -> None:
    monkeypatch.setattr(main, "_INFO_LOG_TTL_TRACKER", {}, raising=False)
    ticks = iter([10.0, 11.0, 20.0])
    monkeypatch.setattr(main.time, "monotonic", lambda: next(ticks))

    assert main._should_emit_info_log("key", ttl_seconds=5.0) is True
    assert main._should_emit_info_log("key", ttl_seconds=5.0) is False
    assert main._should_emit_info_log("key", ttl_seconds=5.0) is True
    assert main._should_emit_info_log("always", ttl_seconds=0.0) is True


def test_market_close_date_key_business_day_and_catchup(monkeypatch) -> None:
    assert main._previous_business_day(datetime(2026, 1, 5).date()).isoformat() == "2026-01-02"
    assert main._previous_business_day(datetime(2026, 1, 4).date()).isoformat() == "2026-01-02"
    assert main._previous_business_day(datetime(2026, 1, 3).date()).isoformat() == "2026-01-02"
    assert main._previous_business_day(datetime(2026, 1, 7).date()).isoformat() == "2026-01-06"

    monkeypatch.setattr(main, "get_env", lambda *_args, **_kwargs: True)
    assert main._resolve_market_close_training_date_key(
        datetime(2026, 1, 7, 8, 0),
    ) == "2026-01-06"
    assert main._resolve_market_close_training_date_key(
        datetime(2026, 1, 7, 16, 0),
    ) == "2026-01-07"

    monkeypatch.setattr(main, "get_env", lambda *_args, **_kwargs: False)
    assert main._resolve_market_close_training_date_key(datetime(2026, 1, 7, 8, 0)) is None


def test_http_profile_logging_enabled_parse_paths(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_env", lambda *_args, **_kwargs: "yes")
    assert main._http_profile_logging_enabled() is True

    monkeypatch.setattr(main, "get_env", lambda *_args, **_kwargs: "0")
    assert main._http_profile_logging_enabled() is False

    monkeypatch.setattr(main, "get_env", lambda *_args, **_kwargs: 2)
    assert main._http_profile_logging_enabled() is True

    monkeypatch.setattr(main, "get_env", lambda *_args, **_kwargs: object())
    assert main._http_profile_logging_enabled() is True

    def _raise(*_args, **_kwargs):
        raise ValueError("bad")

    monkeypatch.setattr(main, "get_env", _raise)
    monkeypatch.setattr(main, "_managed_env", lambda *_args, **_kwargs: "on")
    assert main._http_profile_logging_enabled() is True


def test_reset_warmup_cooldown_timestamp_handles_cached_state(monkeypatch) -> None:
    monkeypatch.setattr(main, "_STATE_CACHE", None, raising=False)
    main._reset_warmup_cooldown_timestamp()

    state = SimpleNamespace(last_run_at="set")
    monkeypatch.setattr(main, "_STATE_CACHE", state, raising=False)
    main._reset_warmup_cooldown_timestamp()

    assert state.last_run_at is None


def test_execution_phase_snapshot_and_update_failure_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        main.runtime_state,
        "observe_service_status",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert main._current_service_snapshot() == {}
    assert main._current_execution_phase() == "unknown"

    monkeypatch.setattr(main.runtime_state, "observe_service_status", lambda: ["bad"])
    assert main._current_service_snapshot() == {}

    monkeypatch.setattr(
        main.runtime_state,
        "observe_service_status",
        lambda: {"phase": " active ", "status": ""},
    )
    updates: list[dict[str, object]] = []
    monkeypatch.setattr(
        main.runtime_state,
        "update_service_status",
        lambda **kwargs: updates.append(kwargs),
    )

    main._set_execution_phase("runtime", reason="test", cycle_index=5)

    assert updates[-1]["status"] == "warming_up"
    assert updates[-1]["phase"] == "runtime"


@pytest.mark.parametrize(
    ("provider", "broker", "expected"),
    [
        ({"status": "down", "reason": "provider_off"}, {"status": "ready"}, ("degraded", "provider_off")),
        ({"status": "ready"}, {"status": "failed"}, ("degraded", "broker_unreachable")),
        ({"status": "ready", "data_status": "empty"}, {"status": "ready"}, ("degraded", "data_unavailable")),
        ({"status": "ready", "using_backup": True}, {"status": "ready"}, ("degraded", "provider_fallback_active")),
        ({"status": "healthy"}, {"status": "connected"}, ("ready", "runtime_health_ok")),
        (
            {"status": "warming_up", "data_status": "warming_up", "reason": "market_closed"},
            {"status": "connected"},
            ("ready", "market_closed"),
        ),
        ({"status": "unknown"}, {"status": "connected"}, ("warming_up", "provider_status_unknown")),
        ({"status": "ready"}, {"status": "unknown"}, ("warming_up", "broker_status_unknown")),
        ({"status": "warming"}, {"status": "connecting"}, ("warming_up", "runtime_health_pending")),
    ],
)
def test_resolve_active_service_status_branches(provider, broker, expected) -> None:
    assert main._resolve_active_service_status(provider_state=provider, broker_state=broker) == expected


def test_refresh_active_service_status_observes_runtime_state(monkeypatch) -> None:
    monkeypatch.setattr(
        main.runtime_state,
        "observe_data_provider_state",
        lambda: {"status": "healthy", "data_status": "fresh"},
    )
    monkeypatch.setattr(
        main.runtime_state,
        "observe_broker_status",
        lambda: {"status": "connected"},
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(main, "_set_execution_phase", lambda phase, **kwargs: calls.append({"phase": phase, **kwargs}))

    main._refresh_active_service_status(cycle_index=9)

    assert calls == [
        {
            "phase": "active",
            "status": "ready",
            "reason": "runtime_health_ok",
            "cycle_index": 9,
        }
    ]


def test_timestamp_age_seconds_parses_utc_z_and_rejects_bad_values(monkeypatch) -> None:
    fixed_now = datetime(2026, 4, 24, 12, 0, tzinfo=UTC)

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(main, "datetime", _FixedDateTime)

    assert main._timestamp_age_seconds(None) is None
    assert main._timestamp_age_seconds("") is None
    assert main._timestamp_age_seconds("not-a-date") is None
    assert main._timestamp_age_seconds("2026-04-24T11:59:00Z") == pytest.approx(60.0)
    assert main._timestamp_age_seconds("2026-04-24T12:01:00+00:00") == 0.0


def test_truthy_test_mode_and_preflight_enforcement(monkeypatch) -> None:
    env: dict[str, str | None] = {}
    monkeypatch.setattr(main, "_managed_env", lambda name, default=None: env.get(name, default))

    env["FLAG"] = "yes"
    assert main._is_truthy_env("FLAG") is True
    env["FLAG"] = "0"
    assert main._is_truthy_env("FLAG") is False

    env.clear()
    monkeypatch.delitem(main.sys.modules, "pytest", raising=False)
    assert main._is_test_mode() is False
    env["PYTEST_CURRENT_TEST"] = "test_x"
    assert main._is_test_mode() is True

    env.clear()
    env["IMPORT_PREFLIGHT_STRICT"] = "1"
    assert main.should_enforce_strict_import_preflight() is True
    env.clear()
    env["IMPORT_PREFLIGHT_DISABLED"] = "1"
    assert main.should_enforce_strict_import_preflight() is False
    env.clear()
    env["SYSTEMD_COMPAT"] = "1"
    assert main.should_enforce_strict_import_preflight() is False
    env.clear()
    env["PYTEST_RUNNING"] = "1"
    assert main.should_enforce_strict_import_preflight() is False
