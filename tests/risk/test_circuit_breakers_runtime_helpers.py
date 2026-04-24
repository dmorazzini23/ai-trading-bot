from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from ai_trading.risk import circuit_breakers as cb
from ai_trading.risk.circuit_breakers import (
    CircuitBreakerState,
    DeadMansSwitch,
    DrawdownCircuitBreaker,
    TradingHaltManager,
    VolatilityCircuitBreaker,
)


def test_safe_percentage_formatting_handles_mocks_and_bad_values() -> None:
    named = Mock(name="pct")
    unnamed = Mock()
    drawdown = DrawdownCircuitBreaker(max_drawdown=0.1)
    volatility = VolatilityCircuitBreaker()
    manager = TradingHaltManager()

    assert drawdown._safe_format_percentage(named) == "<Mock: pct>"  # noqa: SLF001
    assert volatility._safe_format_percentage(unnamed).startswith("<Mock:")  # noqa: SLF001
    assert "object" in manager._safe_format_percentage(object())  # noqa: SLF001


def test_drawdown_breaker_invalid_values_callbacks_and_recovery() -> None:
    breaker = DrawdownCircuitBreaker(max_drawdown=0.10, recovery_threshold=0.90)
    events: list[tuple[str, dict]] = []
    breaker.add_callback(lambda event, payload: events.append((event, payload)))
    breaker.add_callback(lambda *_args: (_ for _ in ()).throw(ValueError("callback failed")))

    assert breaker.update_equity(None) is False
    assert breaker.update_equity(-1.0) is False
    assert breaker.update_equity(100_000.0) is True
    assert breaker.update_equity(85_000.0) is False
    assert breaker.state is CircuitBreakerState.OPEN
    assert events[0][0] == "halt"

    assert breaker.update_equity(89_000.0) is False
    assert breaker.update_equity(90_000.0) is True
    assert breaker.state is CircuitBreakerState.CLOSED
    assert events[-1][0] == "reset"

    status = breaker.get_status()
    assert status["trading_allowed"] is True
    assert status["peak_equity"] == 100_000.0


@pytest.mark.parametrize(
    ("volatility", "expected_status", "expected_state", "allowed"),
    [
        (0.20, "NORMAL_OPERATION", CircuitBreakerState.CLOSED, True),
        (0.60, "HIGH_VOLATILITY_REDUCTION", CircuitBreakerState.HALF_OPEN, True),
        (1.20, "EXTREME_VOLATILITY_HALT", CircuitBreakerState.OPEN, False),
    ],
)
def test_volatility_breaker_thresholds(volatility, expected_status, expected_state, allowed) -> None:
    breaker = VolatilityCircuitBreaker(high_vol_threshold=0.5, extreme_vol_threshold=1.0)

    status = breaker.update_volatility(volatility)

    assert status["status"] == expected_status
    assert breaker.state is expected_state
    assert status["trading_allowed"] is allowed
    assert 0.0 <= status["position_size_multiplier"] <= 1.0
    assert breaker.get_status()["state"] == expected_state.value


def test_trading_halt_manager_combines_manual_emergency_daily_and_breaker_reasons() -> None:
    manager = TradingHaltManager()
    manager.max_daily_trades = 1
    manager.max_daily_loss = 0.01
    manager.record_trade(trade_pnl=-0.02)
    manager.manual_halt_trading("ops")
    manager.emergency_stop_all("panic")
    manager.update_volatility(1.5)
    manager.update_equity(100_000.0)
    manager.update_equity(80_000.0)

    status = manager.is_trading_allowed()

    assert status["trading_allowed"] is False
    assert any("Manual halt" in reason for reason in status["reasons"])
    assert any("Emergency stop" in reason for reason in status["reasons"])
    assert any("Drawdown limit" in reason for reason in status["reasons"])
    assert any("Extreme volatility" in reason for reason in status["reasons"])
    assert any("Daily trade limit" in reason for reason in status["reasons"])
    assert any("Daily loss limit" in reason for reason in status["reasons"])

    manager.resume_trading("ok")
    manager.reset_emergency_stop("ok")
    manager.reset_daily_counters()

    assert manager.manual_halt is False
    assert manager.emergency_stop is False
    assert manager.daily_trade_count == 0
    assert manager.daily_loss_amount == 0.0


def test_trading_halt_manager_callbacks_and_comprehensive_status() -> None:
    manager = TradingHaltManager()
    reasons: list[str] = []
    manager.add_emergency_callback(lambda reason: reasons.append(reason))
    manager.add_emergency_callback(lambda _reason: (_ for _ in ()).throw(ValueError("bad callback")))

    manager.emergency_stop_all("panic")
    comprehensive = manager.get_comprehensive_status()

    assert reasons == ["panic"]
    assert comprehensive["manual_controls"]["emergency_stop"] is True
    assert "trading_status" in comprehensive
    assert "daily_limits" in comprehensive


def test_dead_mans_switch_heartbeat_status_and_emergency(monkeypatch) -> None:
    fixed_now = datetime(2026, 4, 24, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(cb, "safe_utcnow", lambda: fixed_now)
    switch = DeadMansSwitch(timeout_seconds=10)
    calls: list[str] = []
    switch.add_emergency_callback(lambda reason: calls.append(reason))
    switch.add_emergency_callback(lambda _reason: (_ for _ in ()).throw(ValueError("bad callback")))

    switch.is_active = True
    switch.last_heartbeat = fixed_now - timedelta(seconds=3)
    switch.heartbeat()

    assert switch.last_heartbeat == fixed_now
    assert switch.get_status()["status"] == "OK"

    switch.last_heartbeat = fixed_now - timedelta(seconds=20)
    assert switch.get_status()["status"] == "TIMEOUT"

    switch._trigger_emergency()  # noqa: SLF001
    assert calls == ["Dead man's switch timeout"]
    assert switch.is_active is False


def test_dead_mans_switch_start_stop_uses_thread_stub(monkeypatch) -> None:
    started: list[str] = []
    joined: list[float] = []

    class _Thread:
        def __init__(self, target, daemon, name):
            self.target = target
            self.daemon = daemon
            self.name = name
            self._alive = False

        def start(self):
            started.append(self.name)
            self._alive = True

        def is_alive(self):
            return self._alive

        def join(self, timeout):
            joined.append(timeout)
            self._alive = False

    monkeypatch.setattr(cb.threading, "Thread", _Thread)
    switch = DeadMansSwitch(timeout_seconds=1)

    switch.start_monitoring()
    switch.start_monitoring()
    assert started == ["DeadMansSwitch"]
    assert switch.is_active is True

    switch.stop_monitoring()
    assert switch.is_active is False
    assert joined == [5.0]


def test_dead_mans_switch_monitoring_loop_triggers_timeout(monkeypatch) -> None:
    now = datetime(2026, 4, 24, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(cb, "safe_utcnow", lambda: now)
    switch = DeadMansSwitch(timeout_seconds=5)
    switch.last_heartbeat = now - timedelta(seconds=10)
    switch.is_active = True
    calls: list[str] = []
    monkeypatch.setattr(switch, "_trigger_emergency", lambda: calls.append("triggered"))

    switch._monitoring_loop()  # noqa: SLF001

    assert calls == ["triggered"]
