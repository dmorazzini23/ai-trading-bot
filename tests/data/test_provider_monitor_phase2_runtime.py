from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Mapping, cast

import pytest

from ai_trading.data import provider_monitor as pm


def _fake_env(values: dict[str, Any]):
    def fake_get_env(name: str, default: Any = None, *, cast=None, **_kwargs: Any) -> Any:
        value = values.get(name, default)
        if isinstance(value, BaseException):
            raise value
        if cast is not None and value is not None:
            return cast(value)
        return value

    return fake_get_env


def test_provider_monitor_env_and_label_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "BROKEN": RuntimeError("bad env"),
        "NONE_VALUE": None,
        "PYTEST_RUNNING": "0",
        "PYTEST_CURRENT_TEST": "",
        "PROVIDER_MONITOR_RELAXED_TEST_MODE": "yes",
    }
    monkeypatch.setattr(pm, "get_env", _fake_env(values))

    assert pm._env_value("BROKEN", "fallback") == "fallback"
    assert pm._env_text("NONE_VALUE", "fallback") == ""
    assert pm._detect_pytest_env() is False
    values["PYTEST_CURRENT_TEST"] = "test-id"
    assert pm._detect_pytest_env() is True
    assert pm._pytest_relaxed_switchovers_enabled() is True

    assert pm.canonical_provider("") == ""
    assert pm.canonical_provider("alpaca_sip") == "alpaca-sip"
    assert pm.canonical_provider("alpaca_custom_feed") == "alpaca-custom-feed"
    assert pm.canonical_provider("custom_provider") == "custom-provider"
    assert pm._normalize_provider("alpaca-yahoo") == "yahoo"
    assert pm._canonical_label("alpaca_iex") == "alpaca-iex"

    assert pm._reason_is_critical(None) is False
    assert pm._reason_is_critical("provider timeout") is True
    assert pm._extract_feed_attempts(None) == []
    assert pm._extract_feed_attempts(
        {
            "primary_provider": "alpaca_iex",
            "provider": "alpaca_iex",
            "fallback_provider": "yfinance",
            "promoted_provider": "custom_feed",
            "used_backup": True,
        }
    ) == ["alpaca-iex", "yahoo", "custom-feed"]


def test_provider_monitor_config_resolvers_use_settings_env_and_clamps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    settings = SimpleNamespace(
        provider_switch_cooldown_sec="42",
        provider_health_passes_required="3",
        provider_max_cooldown_seconds="30",
        provider_switch_quiet_seconds="-5",
        logging_dedupe_ttl_s="11",
        halt_flag_path=str(tmp_path / "settings-halt.flag"),
    )
    monkeypatch.setattr(pm, "get_settings", lambda: settings)
    monkeypatch.setattr(pm, "get_env", _fake_env({}))

    assert pm._resolve_switch_cooldown_seconds() == 42
    assert pm._resolve_health_passes_required() == 3
    assert pm._resolve_max_cooldown() == 60.0
    assert pm._resolve_switch_quiet_seconds() == 0.0
    assert pm._logging_dedupe_ttl() == 11
    assert pm._resolve_halt_flag_path() == str(tmp_path / "settings-halt.flag")

    settings.provider_switch_cooldown_sec = None
    settings.provider_switch_cooldown_seconds = None
    settings.provider_health_passes_required = None
    settings.provider_health_passes = None
    settings.provider_max_cooldown_seconds = None
    settings.provider_switch_quiet_seconds = None
    settings.halt_flag_path = ""
    values = {
        "AI_TRADING_PROVIDER_SWITCH_COOLDOWN_SEC": "7",
        "AI_TRADING_PROVIDER_HEALTH_PASSES_REQUIRED": "2",
        "PROVIDER_MAX_COOLDOWN_SECONDS": "75",
        "PROVIDER_SWITCH_QUIET_SECONDS": "9",
        "AI_TRADING_HALT_FLAG_PATH": str(tmp_path / "env-halt.flag"),
    }
    monkeypatch.setattr(pm, "get_env", _fake_env(values))

    assert pm._resolve_switch_cooldown_seconds() == 7
    assert pm._resolve_health_passes_required() == 2
    assert pm._resolve_max_cooldown() == 75.0
    assert pm._resolve_switch_quiet_seconds() == 9.0
    assert pm._resolve_halt_flag_path() == str(tmp_path / "env-halt.flag")


def test_provider_monitor_gap_threshold_and_feed_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "AI_TRADING_PROVIDER_PRIMARY_RECOVERY_BIAS_ENABLED": "true",
        "AI_TRADING_PROVIDER_PRIMARY_RECOVERY_MIN_PASSES": "0",
        "AI_TRADING_PROVIDER_PRIMARY_RECOVERY_COOLDOWN_SCALE": "nan",
        "HEALTH_RECOVERY_PASSES": "bad",
        "AI_TRADING_SAFE_MODE_HEALTH_PASSES": "5",
        "DATA_PROVIDER_PRIMARY_DWELL_SECONDS": "-9",
        "AI_TRADING_GAP_RATIO_SAFE_MODE": "0.04",
        "AI_TRADING_GAP_MISSING_SAFE_MODE": "0",
        "QUOTE_MAX_AGE_MS": "1500",
        "SAFE_MODE_FAILSOFT_GAP_RATIO": "0.12",
        "ALPACA_EXECUTION_FEED": "sip",
        "DATA_FEED_INTRADAY": "",
        "ALPACA_DATA_FEED": "",
    }
    monkeypatch.setattr(pm, "get_env", _fake_env(values))
    monkeypatch.setattr(pm, "get_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(
        pm,
        "get_execution_feed",
        lambda requested=None: str(requested or "iex").strip().lower(),
    )

    assert pm._resolve_primary_recovery_bias_settings() == (True, 1, 0.5)
    assert pm._primary_recovery_bias_applicable(primary="alpaca_iex", backup="yahoo", active="yahoo") is True
    assert pm._primary_recovery_bias_applicable(primary="yahoo", backup="alpaca", active="alpaca") is False
    assert pm._resolve_safe_mode_recovery_passes() == 5
    assert pm._resolve_primary_dwell_seconds() == 0.0
    assert pm._resolve_gap_ratio_trigger() == 0.04
    assert pm._resolve_gap_missing_trigger() == 1
    assert pm._quote_recovery_age_limit_ms() == 1500.0
    assert pm._failsoft_gap_ratio_limit() == 0.12
    assert pm._normalize_gap_ratio_value(None, 6.5) == 0.065
    assert pm._normalize_gap_ratio_value(-1, 0.5) == 0.5

    pm._prime_gap_ratio_cache(0.04)
    assert pm._current_intraday_feed() == "sip"
    assert pm._intraday_feed_is_sip() is True
    assert pm._gap_ratio_threshold_for_feed("iex") == 0.30
    assert pm._gap_ratio_threshold_for_metadata({"provider_canonical": "alpaca_iex"}) == 0.30
    assert pm._gap_ratio_threshold_for_metadata({"provider_canonical": "alpaca_sip"}) == 0.04
    assert pm._gap_missing_threshold_for_metadata({"provider": "alpaca_iex", "expected": 100}) == 30
    assert pm._compute_feed_gap_ratio_threshold("", 0.02) == 0.02


def test_provider_monitor_halt_flag_and_safe_mode_recovery(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    halt_path = tmp_path / "halt.flag"
    monkeypatch.setattr(pm, "get_settings", lambda: SimpleNamespace(halt_flag_path=str(halt_path)))
    monkeypatch.setattr(pm, "get_env", _fake_env({"TRADING__SAFE_MODE_FAILSOFT": True}))

    pm._write_halt_flag("minute_gap", metadata={"symbol": "AAPL"})
    assert halt_path.exists()
    pm._clear_halt_flag()
    assert not halt_path.exists()
    pm._clear_halt_flag()

    clear_calls: list[str] = []
    monkeypatch.setattr(pm, "_clear_halt_flag", lambda: clear_calls.append("cleared"))
    monkeypatch.setattr(pm, "_quote_recovery_age_limit_ms", lambda: 1000.0)
    monkeypatch.setattr(pm, "_current_intraday_feed", lambda: "sip")
    monkeypatch.setattr(pm, "_gap_ratio_threshold_for_feed", lambda _feed: 0.1)

    pm._SAFE_MODE_ACTIVE = False
    pm._SAFE_MODE_HEALTHY_PASSES = 3
    pm._maybe_clear_safe_mode(
        success=True,
        gap_ratio=None,
        quote_timestamp_present=True,
        quote_age_ms=0,
        disabled_until=None,
    )
    assert pm._SAFE_MODE_HEALTHY_PASSES == 0

    pm._SAFE_MODE_ACTIVE = True
    pm._SAFE_MODE_REASON = "minute_gap"
    pm._SAFE_MODE_DEGRADED_ONLY = True
    pm._SAFE_MODE_RECOVERY_TARGET = 2
    pm._SAFE_MODE_HEALTHY_PASSES = 1
    pm._maybe_clear_safe_mode(
        success=True,
        gap_ratio=0.01,
        quote_timestamp_present=True,
        quote_age_ms=100,
        disabled_until=datetime.now(UTC) + timedelta(minutes=1),
    )
    assert pm._SAFE_MODE_ACTIVE is True
    assert pm._SAFE_MODE_HEALTHY_PASSES == 2

    pm._maybe_clear_safe_mode(
        success=False,
        gap_ratio=0.01,
        quote_timestamp_present=True,
        quote_age_ms=100,
        disabled_until=None,
    )
    assert pm._SAFE_MODE_HEALTHY_PASSES == 0

    pm._maybe_clear_safe_mode(
        success=True,
        gap_ratio=0.01,
        quote_timestamp_present=False,
        quote_age_ms=100,
        disabled_until=None,
    )
    assert pm._SAFE_MODE_HEALTHY_PASSES == 0

    pm._maybe_clear_safe_mode(
        success=True,
        gap_ratio=0.01,
        quote_timestamp_present=True,
        quote_age_ms=1500,
        disabled_until=None,
    )
    assert pm._SAFE_MODE_HEALTHY_PASSES == 0

    pm._maybe_clear_safe_mode(
        success=True,
        gap_ratio=0.5,
        quote_timestamp_present=True,
        quote_age_ms=100,
        disabled_until=None,
    )
    assert pm._SAFE_MODE_HEALTHY_PASSES == 0

    pm._maybe_clear_safe_mode(
        success=True,
        gap_ratio=0.01,
        quote_timestamp_present=True,
        quote_age_ms=100,
        disabled_until=None,
    )
    pm._maybe_clear_safe_mode(
        success=True,
        gap_ratio=0.01,
        quote_timestamp_present=True,
        quote_age_ms=100,
        disabled_until=None,
    )
    assert pm._SAFE_MODE_ACTIVE is False
    assert pm._SAFE_MODE_REASON is None
    assert pm._SAFE_MODE_DEGRADED_ONLY is False
    assert clear_calls == ["cleared"]


def test_provider_monitor_gap_event_classification_and_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    pm._GAP_RATIO_TRIGGER = 0.05
    pm._GAP_MISSING_TRIGGER = 3
    pm._prime_gap_ratio_cache(0.05)
    pm._gap_event_diagnostics = {}

    assert pm._minute_gap_event_is_primary({"provider": "yahoo"}) == (False, "yahoo")
    assert pm._minute_gap_event_is_primary({"provider": "alpaca", "using_fallback_provider": True}) == (False, "alpaca")
    assert pm._safe_iso(datetime(2026, 4, 20, 12, tzinfo=UTC)) == "2026-04-20T12:00:00+00:00"
    assert pm._safe_iso("raw") == "raw"
    assert pm._safe_iso(object()) is None
    assert pm._gap_event_is_severe({"residual_gap": False, "fallback_contiguous": True}) is False
    assert pm._gap_event_is_severe(
        {
            "residual_gap": False,
            "used_backup": True,
            "initial_gap_ratio": 0.20,
            "initial_missing": 0,
            "provider": "alpaca_sip",
        }
    ) is False
    assert pm._gap_event_is_severe(
        {
            "residual_gap": False,
            "used_backup": True,
            "initial_gap_ratio": 0.20,
            "initial_missing": 0,
            "missing_after": 1,
            "provider": "alpaca_sip",
        }
    ) is True
    assert pm._gap_event_is_severe({"residual_gap": True, "gap_ratio": 0.20, "missing_after": 0}) is True
    assert pm._gap_event_is_severe({"residual_gap": True, "gap_ratio": 0.01, "missing_after": 5}) is True
    assert pm._gap_event_is_severe({"residual_gap": True, "gap_ratio": 0.01, "missing_after": 1}) is False

    monkeypatch.setattr(pm, "update_data_provider_state", lambda **_kwargs: None)
    metadata = {
        "gap_ratio": "0.2",
        "missing_after": "4",
        "initial_gap_ratio": "0.3",
        "initial_missing": "6",
        "expected": "20",
        "used_backup": True,
        "window_start": datetime(2026, 4, 20, 12, tzinfo=UTC),
        "window_end": datetime(2026, 4, 20, 12, 1, tzinfo=UTC),
    }
    pm._update_gap_diagnostics("alpaca", metadata, severe=True)
    diag = pm._gap_event_diagnostics["alpaca"]
    assert diag["events"] == 1
    assert diag["total_missing"] == 4
    assert diag["used_backup_events"] == 1
    assert diag["samples"][-1]["initial_missing"] == 6


def test_provider_monitor_record_event_and_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    triggers: list[tuple[str, int, Mapping[str, Any] | None]] = []
    monkeypatch.setattr(
        pm,
        "_trigger_provider_safe_mode",
        lambda reason, *, count, metadata=None: triggers.append((reason, count, metadata)),
    )
    monkeypatch.setattr(pm, "monotonic_time", iter([0.0, 1.0, 2.0, 130.0, 131.0, 132.0]).__next__)
    monkeypatch.setattr(pm, "_HALT_EVENT_WINDOW_SECONDS", 600.0)
    monkeypatch.setattr(pm, "_SAFE_MODE_EVENT_BURST_WINDOW", 120.0)
    monkeypatch.setattr(pm, "_gap_trigger_cooldown_until", 0.0, raising=False)
    monkeypatch.setattr(pm, "_gap_event_diagnostics", {}, raising=False)
    monkeypatch.setattr(pm, "update_data_provider_state", lambda **_kwargs: None)

    bucket: deque[float] = deque()
    pm._record_event(
        bucket,
        threshold=2,
        reason="minute_gap",
        metadata=cast(Any, object()),
    )  # ignored non-mapping
    pm._record_event(
        bucket,
        threshold=2,
        reason="minute_gap",
        metadata={"provider": "yahoo", "primary_feed_gap": False},
    )
    assert triggers == []

    severe = {
        "provider": "alpaca",
        "primary_feed_gap": True,
        "residual_gap": True,
        "gap_ratio": 0.5,
        "missing_after": 10,
        "expected": 20,
    }
    pm._record_event(bucket, threshold=2, reason="minute_gap", metadata=severe)
    pm._record_event(bucket, threshold=2, reason="minute_gap", metadata=severe)
    assert triggers[-1][0:2] == ("minute_gap", 2)
    assert len(bucket) == 0

    context: dict[str, Any] = {}
    assert (
        pm.decide_provider_action(
            {"healthy": False, "using_backup": False},
            cooldown_ok=True,
            consecutive_switches=3,
            policy={"disable_after": 3},
            context=context,
        )
        is pm.ProviderAction.DISABLE
    )
    assert context["stay_logged"] is False

    context = {}
    assert (
        pm.decide_provider_action(
            {"healthy": True, "using_backup": True, "allow_recovery": True},
            cooldown_ok=True,
            consecutive_switches=0,
            policy=SimpleNamespace(prefer_primary=True),
            from_provider="alpaca",
            to_provider="alpaca",
            cooldown=5,
            context=context,
        )
        is pm.ProviderAction.STAY
    )
    assert context["stay_reason"] == "redundant_request"
    assert pm.decide_provider_action(True, cooldown_ok=False, consecutive_switches=0, policy=None) is pm.ProviderAction.STAY
