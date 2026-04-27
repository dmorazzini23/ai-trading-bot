from __future__ import annotations

import asyncio
from collections import UserDict, UserList, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

import pytest

from ai_trading.data import provider_monitor as pm
from ai_trading.data.fallback import concurrency


class FakeAlertManager:
    def __init__(self) -> None:
        self.alerts: list[tuple[Any, Any, str, dict[str, Any] | None]] = []

    def create_alert(self, alert_type: Any, severity: Any, message: str, *, metadata: dict[str, Any] | None = None) -> None:
        self.alerts.append((alert_type, severity, message, metadata))


@pytest.fixture(autouse=True)
def _reset_provider_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    pm.provider_monitor.reset()
    pm._sip_auth_events.clear()
    pm._gap_events.clear()
    pm._gap_event_diagnostics.clear()
    pm._STAY_LOG_TS.clear()
    monkeypatch.setattr(pm, "_last_halt_reason", None)
    monkeypatch.setattr(pm, "_last_halt_ts", 0.0)
    monkeypatch.setattr(pm, "_SAFE_MODE_ACTIVE", False)
    monkeypatch.setattr(pm, "_SAFE_MODE_REASON", None)
    monkeypatch.setattr(pm, "_SAFE_MODE_HEALTHY_PASSES", 0)
    monkeypatch.setattr(pm, "_SAFE_MODE_RECOVERY_TARGET", 1)
    monkeypatch.setattr(pm, "_SAFE_MODE_DEGRADED_ONLY", False)
    monkeypatch.setattr(pm, "_gap_trigger_cooldown_until", 0.0)
    monkeypatch.setattr(pm, "_FIRST_DECISION", False)
    monkeypatch.setattr(pm, "reset_data_provider_state", lambda: None)
    monkeypatch.setattr(pm, "update_data_provider_state", lambda **_kwargs: None)
    yield
    pm.provider_monitor.reset()
    pm._sip_auth_events.clear()
    pm._gap_events.clear()
    pm._gap_event_diagnostics.clear()
    pm._STAY_LOG_TS.clear()


def test_pooling_limit_resolution_and_semaphore_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    class Snapshot:
        limit = 0
        version = 7

    concurrency._invalidate_pooling_snapshot()
    monkeypatch.setattr(concurrency, "_pooling_reload_host_limit", lambda: Snapshot())
    monkeypatch.setattr(concurrency, "_pooling_get_limit_snapshot", None)
    monkeypatch.setattr(concurrency, "_pooling_host_limit", lambda: 9)
    assert concurrency._get_effective_host_limit() == 1
    assert concurrency._POOLING_LIMIT_STATE == (1, 7)

    concurrency._invalidate_pooling_snapshot()
    monkeypatch.setattr(concurrency, "_pooling_reload_host_limit", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(concurrency, "_pooling_get_limit_snapshot", lambda: (5, 11))
    assert concurrency._get_effective_host_limit() == 5
    assert concurrency._POOLING_LIMIT_STATE == (5, 11)

    async def _exercise_semaphore_paths() -> None:
        current_loop = asyncio.get_running_loop()
        stale = asyncio.Semaphore(4)
        setattr(stale, "_ai_trading_host_limit", 4)
        setattr(stale, "_ai_trading_host_limit_version", 1)
        refreshed = asyncio.Semaphore(2)
        setattr(refreshed, "_ai_trading_host_limit", 2)
        setattr(refreshed, "_ai_trading_host_limit_version", 99)
        setattr(refreshed, "_loop", current_loop)

        monkeypatch.setattr(concurrency, "_POOLING_LIMIT_STATE", (2, 99))
        monkeypatch.setattr(concurrency, "_running_under_pytest_worker", lambda: False)
        monkeypatch.setattr(concurrency, "_pooling_get_host_semaphore", lambda: stale)
        monkeypatch.setattr(concurrency, "_pooling_refresh_host_semaphore", lambda loop=None: refreshed)
        assert concurrency._get_host_limit_semaphore() is refreshed
        assert concurrency._POOLING_LIMIT_STATE == (2, 99)

        foreign_loop = asyncio.new_event_loop()
        try:
            wrong_loop = asyncio.Semaphore(3)
            setattr(wrong_loop, "_loop", foreign_loop)
            setattr(wrong_loop, "_ai_trading_host_limit", 3)
            setattr(wrong_loop, "_ai_trading_host_limit_version", 100)
            monkeypatch.setattr(concurrency, "_POOLING_LIMIT_STATE", (3, 100))
            monkeypatch.setattr(concurrency, "_pooling_get_host_semaphore", lambda: wrong_loop)
            monkeypatch.setattr(concurrency, "_pooling_refresh_host_semaphore", lambda loop=None: None)
            assert concurrency._get_host_limit_semaphore() is None
        finally:
            foreign_loop.close()

    asyncio.run(_exercise_semaphore_paths())


def test_scan_rebinds_foreign_loop_primitives_in_nested_containers() -> None:
    @dataclass(frozen=True)
    class FrozenBox:
        lock: asyncio.Lock
        payload: object

    class SlotHolder:
        __slots__ = ("lock",)

        def __init__(self, lock: asyncio.Lock) -> None:
            self.lock = lock

    async def _exercise_scan() -> None:
        loop = asyncio.get_running_loop()
        foreign_loop = asyncio.new_event_loop()
        try:
            lock = asyncio.Lock()
            semaphore = asyncio.Semaphore(2)
            bounded = asyncio.BoundedSemaphore(3)
            for primitive in (lock, semaphore, bounded):
                setattr(primitive, "_loop", foreign_loop)
            holder = SlotHolder(lock)
            mapping = MappingProxyType({"sem": semaphore})
            namespace = SimpleNamespace(items=deque([bounded]), holder=holder)
            box = FrozenBox(lock=lock, payload=(mapping, namespace))

            rebound = concurrency._scan(box, set(), loop)

            assert isinstance(rebound, FrozenBox)
            assert rebound is not box
            assert rebound.lock is not lock
            rebound_mapping, rebound_namespace = rebound.payload
            assert rebound_mapping["sem"] is not semaphore
            assert rebound_namespace.items[0] is not bounded
            assert rebound_namespace.holder.lock is lock
        finally:
            foreign_loop.close()

    asyncio.run(_exercise_scan())


def test_scan_rebinds_mutable_and_immutable_container_variants() -> None:
    async def _exercise_scan() -> None:
        loop = asyncio.get_running_loop()
        foreign_loop = asyncio.new_event_loop()

        def foreign_lock() -> asyncio.Lock:
            lock = asyncio.Lock()
            setattr(lock, "_loop", foreign_loop)
            return lock

        try:
            user_dict = UserDict({"lock": foreign_lock()})
            user_list = UserList([foreign_lock()])
            plain_list = [foreign_lock()]
            plain_set = {foreign_lock()}
            frozen = frozenset({foreign_lock()})
            plain_tuple = (foreign_lock(),)

            original_user_dict_lock = user_dict["lock"]
            original_user_list_lock = user_list[0]
            original_plain_list_lock = plain_list[0]
            assert concurrency._scan(user_dict, set(), loop)["lock"] is not original_user_dict_lock
            assert concurrency._scan(user_list, set(), loop)[0] is not original_user_list_lock
            assert concurrency._scan(plain_list, set(), loop)[0] is not original_plain_list_lock
            original_set_item = next(iter(plain_set))
            assert next(iter(concurrency._scan(plain_set, set(), loop))) is not original_set_item
            original_frozen_item = next(iter(frozen))
            assert next(iter(concurrency._scan(frozen, set(), loop))) is not original_frozen_item
            assert concurrency._scan(plain_tuple, set(), loop)[0] is not plain_tuple[0]
            assert concurrency._scan({"plain": "value"}, set(), loop) == {"plain": "value"}
        finally:
            foreign_loop.close()

    asyncio.run(_exercise_scan())


def test_run_with_concurrency_host_limit_errors_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _worker(symbol: str) -> str:
        if symbol == "BAD":
            raise TimeoutError("provider slow")
        await asyncio.sleep(0)
        return symbol.lower()

    async def _slow_worker(symbol: str) -> str:
        await asyncio.sleep(0.05)
        return symbol

    async def _exercise() -> None:
        concurrency.reset_tracking_state()
        monkeypatch.setattr(concurrency, "_pooling_reload_host_limit", lambda: (2, 1))
        monkeypatch.setattr(concurrency, "_pooling_get_limit_snapshot", None)
        monkeypatch.setattr(concurrency, "_pooling_get_host_semaphore", lambda: None)
        results, succeeded, failed = await concurrency.run_with_concurrency(
            ["A", "BAD", "B", "C"],
            _worker,
            max_concurrency=9,
        )
        assert results == {"A": "a", "BAD": None, "B": "b", "C": "c"}
        assert succeeded == {"A", "B", "C"}
        assert failed == {"BAD"}
        assert concurrency.LAST_RUN_PEAK_SIMULTANEOUS_WORKERS == 2

        results, succeeded, failed = await concurrency.run_with_concurrency(
            ["S1", "S2", "S3"],
            _slow_worker,
            max_concurrency=2,
            timeout_s=0.001,
        )
        assert results == {"S1": None, "S2": None, "S3": None}
        assert succeeded == set()
        assert failed == {"S1", "S2", "S3"}

    asyncio.run(_exercise())


def test_run_with_concurrency_non_pytest_scheduler_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    active = 0
    peak = 0

    async def _worker(symbol: str) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.002)
            if symbol == "ERR":
                raise RuntimeError("bad worker")
            return symbol.lower()
        finally:
            active -= 1

    async def _recovering_worker(symbol: str) -> str:
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise
        return symbol

    async def _exercise() -> None:
        concurrency.reset_tracking_state()
        host_semaphore = asyncio.Semaphore(2)
        setattr(host_semaphore, "_ai_trading_host_limit", 2)
        setattr(host_semaphore, "_ai_trading_host_limit_version", 3)
        monkeypatch.setattr(concurrency, "_running_under_pytest_worker", lambda: False)
        monkeypatch.setattr(concurrency, "_pooling_reload_host_limit", lambda: (2, 3))
        monkeypatch.setattr(concurrency, "_pooling_get_limit_snapshot", None)
        monkeypatch.setattr(concurrency, "_pooling_get_host_semaphore", lambda: host_semaphore)
        monkeypatch.setattr(concurrency, "_pooling_refresh_host_semaphore", lambda loop=None: host_semaphore)

        results, succeeded, failed = await concurrency.run_with_concurrency(
            ["AA", "BB", "ERR", "CC"],
            _worker,
            max_concurrency=4,
        )
        assert results == {"AA": "aa", "BB": "bb", "ERR": None, "CC": "cc"}
        assert succeeded == {"AA", "BB", "CC"}
        assert failed == {"ERR"}
        assert peak <= 2
        assert concurrency._HOST_PERMITS_HELD == 0

        async def _late_success(symbol: str) -> str:
            if concurrency._HOST_PERMITS_HELD >= 0:
                await asyncio.sleep(0.02)
            return f"late-{symbol}"

        results, succeeded, failed = await concurrency.run_with_concurrency(
            ["T1", "T2"],
            _late_success,
            max_concurrency=2,
            timeout_s=0.001,
        )
        assert results == {"T1": "late-T1", "T2": "late-T2"}
        assert succeeded == {"T1", "T2"}
        assert failed == {"T1", "T2"}

        task = asyncio.create_task(
            concurrency.run_with_concurrency(["CANCEL"], _recovering_worker, max_concurrency=1),
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_exercise())


def test_concurrency_config_error_and_recording_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[int] = []
    monkeypatch.setattr(concurrency, "_pooling_record_concurrency", lambda value: recorded.append(value))
    monkeypatch.setattr(
        concurrency,
        "_http_host_limit",
        SimpleNamespace(record_peak=lambda value: recorded.append(value + 10), current_peak=lambda: 0),
    )
    monkeypatch.setattr(concurrency, "PEAK_SIMULTANEOUS_WORKERS", "bad")
    concurrency._update_peak_counters(3)
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS == 3
    assert recorded == [13, 3]

    concurrency._release_host_permit()
    assert concurrency._HOST_PERMITS_HELD == 0
    concurrency._increment_host_permits()
    concurrency._release_host_permit()
    assert concurrency._HOST_PERMITS_HELD == 0

    concurrency._invalidate_pooling_snapshot()
    monkeypatch.setattr(concurrency, "_pooling_reload_host_limit", None)
    monkeypatch.setattr(concurrency, "_pooling_get_limit_snapshot", lambda: (_ for _ in ()).throw(RuntimeError("bad")))
    monkeypatch.setattr(concurrency, "_pooling_host_limit", lambda: (_ for _ in ()).throw(RuntimeError("bad")))
    assert concurrency._get_effective_host_limit() is None

    monkeypatch.setattr(concurrency, "_pooling_get_host_semaphore", lambda: (_ for _ in ()).throw(RuntimeError("bad")))
    assert concurrency._get_host_limit_semaphore() is None
    monkeypatch.setattr(concurrency, "_pooling_get_host_semaphore", lambda: object())
    assert concurrency._get_host_limit_semaphore() is None
    monkeypatch.setattr(concurrency, "_pooling_get_host_semaphore", None)
    assert concurrency._get_host_limit_semaphore() is None

    assert concurrency._normalise_pooling_state(None) is None
    assert concurrency._normalise_pooling_state(("0", "9")) == (1, 9)
    assert concurrency._normalise_pooling_state(("bad", "9")) is None
    assert concurrency._normalise_positive_int(b"4") == 4
    assert concurrency._normalise_positive_int("0") == 1
    assert concurrency._normalise_positive_int(object()) is None

    assert concurrency._should_replace_closure_cell(object(), "value") is False
    assert concurrency._should_replace_closure_cell([1], [2]) is True
    assert concurrency._should_replace_closure_cell(SimpleNamespace(a=1), SimpleNamespace(a=2)) is True


def test_provider_helpers_thresholds_and_gap_severity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pm, "_env_text", lambda name, default="": {"PROVIDER_MONITOR_RELAXED_TEST_MODE": "yes"}.get(name, default))
    assert pm._pytest_relaxed_switchovers_enabled() is True
    assert pm.canonical_provider("alpaca_custom_feed") == "alpaca-custom-feed"
    assert pm.canonical_provider("  ") == ""
    assert pm._reason_is_critical("upstream_unavailable timeout") is True
    assert pm._reason_is_critical("") is False
    assert pm._extract_feed_attempts(
        {
            "primary_provider": "alpaca_iex",
            "provider": "alpaca_iex",
            "fallback_provider": "yfinance",
            "used_backup": True,
        },
    ) == ["alpaca-iex", "yahoo"]
    assert pm._extract_feed_attempts(None) == []
    assert pm.canonical_provider("alpaca_") == "alpaca"

    monkeypatch.setattr(pm, "_GAP_RATIO_TRIGGER", 0.02)
    monkeypatch.setattr(pm, "_GAP_MISSING_TRIGGER", 3)
    pm._prime_gap_ratio_cache(0.02)
    assert pm._gap_ratio_threshold_for_feed("alpaca_iex") == 0.30
    assert pm._gap_ratio_threshold_for_metadata({"provider": "alpaca_sip"}) == 0.02
    assert pm._gap_missing_threshold_for_metadata({"provider": "alpaca_iex", "expected": "20"}) == 6
    assert pm._normalize_gap_ratio_value(None, "12.5") == pytest.approx(0.125)
    assert pm._normalize_gap_ratio_value("-1", None) is None

    severe_primary = {
        "provider": "alpaca",
        "residual_gap": True,
        "gap_ratio": 0.40,
        "missing_after": 8,
        "expected": 20,
        "window_start": datetime(2026, 1, 1, tzinfo=UTC),
        "window_end": datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
    }
    assert pm._minute_gap_event_is_primary(severe_primary) == (True, "alpaca")
    assert pm._gap_event_is_severe(severe_primary) is True
    pm._update_gap_diagnostics("alpaca", severe_primary, severe=True)
    assert pm._gap_event_diagnostics["alpaca"]["events"] == 1
    assert pm._gap_event_diagnostics["alpaca"]["samples"][0]["severe"] is True

    assert pm._minute_gap_event_is_primary({"provider": "yahoo"}) == (False, "yahoo")
    assert pm._gap_event_is_severe({"used_backup": True, "missing_after": 0}) is False


def test_provider_config_resolution_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pm, "get_settings", lambda: SimpleNamespace(provider_switch_cooldown_seconds="12"))
    assert pm._resolve_switch_cooldown_seconds() == 12
    monkeypatch.setattr(pm, "get_settings", lambda: SimpleNamespace(provider_switch_cooldown_sec="bad"))
    monkeypatch.setattr(pm, "get_env", lambda name, default=None, cast=None, **_kwargs: 4 if name == "AI_TRADING_PROVIDER_SWITCH_COOLDOWN_SEC" else default)
    assert pm._resolve_switch_cooldown_seconds() == 4

    monkeypatch.setattr(pm, "get_settings", lambda: SimpleNamespace(provider_health_passes="5"))
    assert pm._resolve_health_passes_required() == 5
    monkeypatch.setattr(pm, "get_settings", lambda: None)
    monkeypatch.setattr(
        pm,
        "get_env",
        lambda name, default=None, cast=None, **_kwargs: {
            "AI_TRADING_PROVIDER_HEALTH_PASSES_REQUIRED": None,
            "HEALTH_RECOVERY_PASSES": 2,
        }.get(name, default),
    )
    assert pm._resolve_health_passes_required() == 2

    monkeypatch.setattr(pm, "_env_value", lambda name, default=None, **_kwargs: "yes" if name == "AI_TRADING_PROVIDER_PRIMARY_RECOVERY_BIAS_ENABLED" else default)
    monkeypatch.setattr(
        pm,
        "get_env",
        lambda name, default=None, cast=None, **_kwargs: {
            "AI_TRADING_PROVIDER_PRIMARY_RECOVERY_MIN_PASSES": "0",
            "AI_TRADING_PROVIDER_PRIMARY_RECOVERY_COOLDOWN_SCALE": float("inf"),
        }.get(name, default),
    )
    assert pm._resolve_primary_recovery_bias_settings() == (True, 1, 0.5)
    assert pm._primary_recovery_bias_applicable(primary="alpaca_sip", backup="yahoo", active="yahoo") is True
    assert pm._primary_recovery_bias_applicable(primary="yahoo", backup="yahoo", active="yahoo") is False
    assert pm._primary_recovery_bias_applicable(primary="alpaca", backup="", active="yahoo") is False
    assert pm._primary_recovery_bias_applicable(primary="alpaca", backup="yahoo", active="alpaca") is False

    monkeypatch.setattr(
        pm,
        "get_env",
        lambda name, default=None, cast=None, **_kwargs: {
            "HEALTH_RECOVERY_PASSES": None,
            "AI_TRADING_SAFE_MODE_HEALTH_PASSES": 3,
        }.get(name, default),
    )
    monkeypatch.setattr(pm, "_env_text", lambda name, default="": "bad" if name == "HEALTH_RECOVERY_PASSES" else "")
    assert pm._resolve_safe_mode_recovery_passes() == 3

    monkeypatch.setattr(pm, "get_env", lambda name, default=None, cast=None, **_kwargs: None)
    monkeypatch.setattr(pm, "_env_text", lambda name, default="": "7.5" if name == "DATA_PROVIDER_PRIMARY_DWELL_SECONDS" else "")
    assert pm._resolve_primary_dwell_seconds() == 7.5

    monkeypatch.setattr(pm, "_env_text", lambda name, default="": "0.11" if name == "AI_TRADING_GAP_RATIO_SAFE_MODE" else "")
    assert pm._resolve_gap_ratio_trigger() == 0.11
    monkeypatch.setattr(pm, "_env_text", lambda name, default="": "6" if name == "AI_TRADING_GAP_MISSING_SAFE_MODE" else "")
    assert pm._resolve_gap_missing_trigger() == 6
    monkeypatch.setattr(pm, "get_env", lambda name, default=None, cast=None, **_kwargs: 123.0 if name == "TRADING__MIN_QUOTE_FRESHNESS_MS" else None)
    assert pm._quote_recovery_age_limit_ms() == 123.0
    monkeypatch.setattr(pm, "get_env", lambda name, default=None, cast=None, **_kwargs: None)
    monkeypatch.setattr(pm, "_env_text", lambda name, default="": "0.2" if name == "AI_TRADING_FAILSOFT_GAP_RATIO" else "")
    assert pm._failsoft_gap_ratio_limit() == 0.2

    monkeypatch.setattr(pm, "get_settings", lambda: SimpleNamespace(provider_max_cooldown_seconds="bad"))
    monkeypatch.setattr(
        pm,
        "get_env",
        lambda name, default=None, cast=None, **_kwargs: {
            "PROVIDER_MAX_COOLDOWN_SECONDS": None,
            "DATA_PROVIDER_MAX_COOLDOWN": "20",
        }.get(name, default),
    )
    monkeypatch.setattr(pm, "_env_value", lambda name, default=None, **_kwargs: None)
    assert pm._resolve_max_cooldown() == 60.0
    monkeypatch.setattr(pm, "get_settings", lambda: SimpleNamespace(provider_switch_quiet_seconds="2.5"))
    assert pm._resolve_switch_quiet_seconds() == 2.5
    monkeypatch.setattr(pm, "get_settings", lambda: None)
    monkeypatch.setattr(pm, "get_env", lambda name, default=None, cast=None, **_kwargs: None)
    monkeypatch.setattr(pm, "_env_value", lambda name, default=None, **_kwargs: "3.5" if name == "PROVIDER_SWITCH_QUIET_SECONDS" else default)
    assert pm._resolve_switch_quiet_seconds() == 3.5
    monkeypatch.setattr(pm, "get_settings", lambda: (_ for _ in ()).throw(RuntimeError("settings")))
    assert pm._logging_dedupe_ttl() == 0


def test_provider_policy_decision_and_monitor_bookkeeping(monkeypatch: pytest.MonkeyPatch) -> None:
    context: dict[str, Any] = {}
    assert pm._policy_lookup(None, "missing", "default") == "default"
    assert pm._policy_lookup({"x": 1}, "x", 0) == 1
    assert pm._policy_lookup(SimpleNamespace(x=2), "x", 0) == 2
    assert pm.decide_provider_action(False, True, 0, None) is pm.ProviderAction.SWITCH
    assert pm.decide_provider_action({"healthy": False, "using_backup": True}, True, 0, None) is pm.ProviderAction.STAY
    assert pm.decide_provider_action(False, True, 3, {"disable_after": 2}) is pm.ProviderAction.DISABLE
    assert (
        pm.decide_provider_action(
            {"is_healthy": True, "using_backup": True, "allow_recovery": True},
            True,
            0,
            {"allow_recovery": True},
            from_provider="alpaca",
            to_provider="alpaca",
            context=context,
        )
        is pm.ProviderAction.STAY
    )
    assert context["stay_reason"] == "redundant_request"

    monkeypatch.setattr(pm, "_env_text", lambda name, default="": "")
    monkeypatch.setattr(pm, "_env_value", lambda name, default=None, **_kwargs: None)
    monitor = pm.ProviderMonitor(cooldown=1, alert_manager=FakeAlertManager(), primary_dwell_seconds=2)
    assert monitor.retry_jitter_range == (10.0, 45.0)
    assert monitor.primary_dwell_seconds == 2.0

    monitor.disabled_until["alpaca-iex"] = datetime.now(UTC)
    monitor._current_switch_cooldowns["alpaca-iex"] = 4.0
    monitor._migrate_provider_state("alpaca-iex", "alpaca_iex")
    assert "alpaca_iex" in monitor.disabled_until
    assert "alpaca_iex" in monitor._current_switch_cooldowns
    monitor._migrate_provider_state("", "alpaca")
    monitor._migrate_provider_state("alpaca_iex", "alpaca_iex")

    monitor.record_health_pass(True)
    monitor._last_switchover_provider = "alpaca"
    monitor.record_health_pass(False, provider="yahoo")
    assert monitor._last_switchover_passes == 0


def test_provider_filesystem_and_logging_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    flag = tmp_path / "nested" / "halt.flag"
    monkeypatch.setattr(pm, "get_settings", lambda: SimpleNamespace(halt_flag_path=str(flag)))
    assert pm._resolve_halt_flag_path() == str(flag)
    pm._write_halt_flag("minute_gap", metadata={"provider": "alpaca"})
    assert flag.exists()
    pm._clear_halt_flag()
    assert not flag.exists()
    pm._clear_halt_flag()

    monkeypatch.setattr(pm, "get_settings", lambda: None)
    monkeypatch.setattr(pm, "_env_value", lambda name, default=None, **_kwargs: str(flag) if name == "AI_TRADING_HALT_FLAG_PATH" else default)
    assert pm._resolve_halt_flag_path() == str(flag)

    monkeypatch.setattr(pm, "get_env", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad env")))
    assert pm._safe_mode_failsoft_enabled() is True
    assert pm._env_value("BROKEN", "fallback") == "fallback"
    assert pm._env_text("BROKEN") == ""

    monkeypatch.setattr(pm, "_env_text", lambda name, default="": "1" if name == "PYTEST_RUNNING" else default)
    quality_events: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        pm,
        "log_data_quality_event",
        lambda event, *, provider, severity, reason, context: quality_events.append((event, provider, severity)),
    )
    pm.activate_data_kill_switch("empty_bars", provider="alpaca", metadata={"gap_ratio": 0.5})
    assert quality_events == [("kill_switch", "alpaca", "warning")]


def test_record_event_triggers_safe_mode_and_suppresses_non_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    triggered: list[tuple[str, int, MappingProxyType[str, Any] | dict[str, Any] | None]] = []
    monkeypatch.setattr(
        pm,
        "_trigger_provider_safe_mode",
        lambda reason, *, count, metadata=None: triggered.append((reason, count, metadata)),
    )

    bucket: deque[float] = deque()
    pm._record_event(bucket, threshold=2, reason="minute_gap", metadata={"provider": "yahoo"})
    assert triggered == []
    assert len(bucket) == 0

    severe = {
        "provider": "alpaca",
        "primary_feed_gap": True,
        "residual_gap": True,
        "gap_ratio": 0.5,
        "missing_after": 9,
        "expected": 10,
    }
    pm._record_event(bucket, threshold=2, reason="minute_gap", metadata=severe)
    pm._record_event(bucket, threshold=2, reason="minute_gap", metadata=severe)
    assert triggered
    reason, count, metadata = triggered[-1]
    assert reason == "minute_gap"
    assert count == 2
    assert metadata is not None
    assert metadata["provider_canonical"] == "alpaca"
    assert len(bucket) == 0


def test_safe_mode_degraded_outage_and_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    alerts = FakeAlertManager()
    monitor = pm.ProviderMonitor(threshold=1, cooldown=0, alert_manager=alerts, max_cooldown=1)
    monkeypatch.setattr(pm, "provider_monitor", monitor)
    monkeypatch.setattr(pm, "_safe_mode_failsoft_enabled", lambda: True)
    monkeypatch.setattr(pm, "_failsoft_gap_ratio_limit", lambda: 0.25)
    halt_writes: list[tuple[str, Mapping[str, Any] | None]] = []
    halt_clears: list[bool] = []
    monkeypatch.setattr(pm, "_write_halt_flag", lambda reason, metadata=None: halt_writes.append((reason, metadata)))
    monkeypatch.setattr(pm, "_clear_halt_flag", lambda: halt_clears.append(True))

    pm._trigger_provider_safe_mode(
        "minute_gap",
        count=3,
        metadata={
            "provider": "alpaca",
            "primary_provider": "alpaca_iex",
            "fallback_provider": "yahoo",
            "used_backup": True,
            "fallback_contiguous": "yes",
            "gap_ratio": 0.10,
        },
    )

    assert pm.is_safe_mode_active() is True
    assert pm.safe_mode_reason() == "minute_gap"
    assert pm.safe_mode_degraded_only() is True
    assert halt_writes == []
    assert alerts.alerts[-1][2] == "Alpaca minute feed outage detected"

    monkeypatch.setattr(pm, "_SAFE_MODE_RECOVERY_TARGET", 2)
    monkeypatch.setattr(pm, "_quote_recovery_age_limit_ms", lambda: 100.0)
    monkeypatch.setattr(pm, "_current_intraday_feed", lambda: "sip")
    monkeypatch.setattr(pm, "_gap_ratio_threshold_for_feed", lambda _feed: 0.20)
    pm._maybe_clear_safe_mode(
        success=False,
        gap_ratio=0.01,
        quote_timestamp_present=True,
        quote_age_ms=1.0,
        disabled_until=None,
    )
    pm._maybe_clear_safe_mode(
        success=True,
        gap_ratio=0.01,
        quote_timestamp_present=True,
        quote_age_ms=150.0,
        disabled_until=None,
    )
    pm._maybe_clear_safe_mode(
        success=True,
        gap_ratio=0.30,
        quote_timestamp_present=True,
        quote_age_ms=1.0,
        disabled_until=None,
    )
    assert pm.is_safe_mode_active() is True

    pm._maybe_clear_safe_mode(
        success=True,
        gap_ratio=0.01,
        quote_timestamp_present=True,
        quote_age_ms=1.0,
        disabled_until=None,
    )
    pm._maybe_clear_safe_mode(
        success=True,
        gap_ratio=0.01,
        quote_timestamp_present=True,
        quote_age_ms=1.0,
        disabled_until=None,
    )
    assert pm.is_safe_mode_active() is False
    assert halt_clears == [True]


def test_provider_failure_disable_recovery_and_success_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    alerts = FakeAlertManager()
    monitor = pm.ProviderMonitor(threshold=2, cooldown=1, alert_manager=alerts, backoff_factor=2, max_cooldown=5)
    callbacks: list[float] = []
    monitor.register_disable_callback("alpaca", lambda duration: callbacks.append(duration.total_seconds()))
    monkeypatch.setattr(pm.random, "uniform", lambda _low, _high: 0.0)

    monitor.record_failure("yahoo", "empty", "no rows")
    assert monitor.is_disabled("yahoo") is True

    exc = TimeoutError("slow")
    monitor.record_failure("alpaca", "timeout", "slow", exception=exc, retry_after=3.5)
    monitor.record_failure("alpaca", "timeout", "slow", exception=exc, retry_after=3.5)
    assert monitor.is_disabled("alpaca") is True
    assert callbacks
    assert alerts.alerts[-1][2] == "Data provider alpaca failure"

    monitor.disabled_until["alpaca"] = datetime.now(UTC) - timedelta(seconds=1)
    monitor.disabled_since["alpaca"] = datetime.now(UTC) - timedelta(seconds=5)
    monitor.outage_start["alpaca"] = datetime.now(UTC) - timedelta(seconds=7)
    assert monitor.is_disabled("alpaca") is False
    assert alerts.alerts[-1][2] == "Data provider alpaca restored"

    monitor.disabled_until["alpaca_sip"] = datetime.now(UTC) + timedelta(seconds=30)
    monitor.disabled_since["alpaca_sip"] = datetime.now(UTC)
    monitor.consecutive_switches_by_provider["alpaca_sip"] = 3
    monitor.record_success("alpaca")
    assert "alpaca_sip" not in monitor.disabled_until
    assert monitor.consecutive_switches == 0


def test_switchover_quiet_period_health_transitions_and_sip_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pm, "_resolve_switch_quiet_seconds", lambda: 0.0)
    monkeypatch.setattr(pm, "_resolve_switch_cooldown_seconds", lambda: 0)
    monkeypatch.setattr(pm, "_resolve_health_passes_required", lambda: 1)
    monkeypatch.setenv("DATA_COOLDOWN_SECONDS", "0")
    monitor = pm.ProviderMonitor(
        threshold=1,
        cooldown=0,
        alert_manager=FakeAlertManager(),
        switchover_threshold=10,
        max_cooldown=1,
        primary_dwell_seconds=0,
    )
    monitor.min_recovery_seconds = 0
    monitor.recovery_passes_required = 1
    monitor.decision_window_seconds = 0

    assert monitor.update_data_health("alpaca", "yahoo", healthy=False, reason="timeout") == "yahoo"
    assert monitor.active_provider("alpaca", "yahoo") == "yahoo"
    assert monitor.update_data_health(
        "alpaca",
        "yahoo",
        healthy=True,
        reason="recovered",
        quote_timestamp_present=True,
        quote_age_ms=1.0,
    ) == "alpaca"

    assert monitor.update_data_health("alpaca", "alpaca_sip", healthy=False, reason="timeout") == "alpaca"

    monitor.disabled_until["alpaca"] = datetime.now(UTC) + timedelta(seconds=10)
    assert monitor.record_switchover("alpaca", "yahoo") is None
    monitor.disabled_until.clear()
    monitor._last_switchover_provider = "alpaca"
    monitor._last_switchover_passes = 0
    monitor.recovery_passes_required = 2
    assert monitor.record_switchover("alpaca", "yahoo") is None

    monkeypatch.setattr(pm, "_resolve_switch_quiet_seconds", lambda: 120.0)
    monitor._last_switchover_provider = None
    monitor.recovery_passes_required = 1
    monitor.switch_quiet_seconds = 120.0
    assert monitor.record_switchover("alpaca", "yahoo") is None
    assert monitor.record_switchover("alpaca", "yahoo") is None
    assert monitor.record_switchover("alpaca", "yahoo") is pm.ProviderAction.DISABLE
