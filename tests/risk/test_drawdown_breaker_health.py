"""The drawdown breaker's state is reported in the health payload, informationally."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

import ai_trading.health_payload as health_payload_module
from ai_trading.risk import circuit_breakers
from ai_trading.risk.circuit_breakers import (
    DRAWDOWN_BREAKER_STATE_SCHEMA_VERSION,
    DrawdownCircuitBreaker,
    drawdown_breaker_health_section,
    register_drawdown_breaker_for_health,
)

ACCOUNT = "acct-secret-123"


@pytest.fixture(autouse=True)
def _clear_registry():
    register_drawdown_breaker_for_health(None)
    yield
    register_drawdown_breaker_for_health(None)


def _breaker(path: Path | None = None) -> DrawdownCircuitBreaker:
    return DrawdownCircuitBreaker(max_drawdown=0.08, recovery_threshold=0.8, state_path=path)


def _feed(breaker: DrawdownCircuitBreaker, equity: float) -> None:
    breaker.score_account_snapshot({"equity": equity, "id": ACCOUNT})


def _write_state(path: Path, **overrides: Any) -> None:
    payload: dict[str, Any] = {
        "schema_version": DRAWDOWN_BREAKER_STATE_SCHEMA_VERSION,
        "account_id": ACCOUNT,
        "peak_equity": 100_000.0,
        "peak_set_at": (circuit_breakers.safe_utcnow() - timedelta(days=3)).isoformat(),
        "state": "closed",
        "tripped_at": None,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_snapshot_before_any_reading() -> None:
    snap = _breaker().health_snapshot()
    assert snap["available"] is True
    assert snap["state"] == "closed"
    assert snap["peak_equity"] is None
    assert snap["current_drawdown"] is None
    assert snap["trip_threshold"] == 0.08
    assert snap["persistence"] == "not_configured"
    assert snap["persisted_state_loaded"] is False
    json.dumps(snap)


def test_snapshot_pending_until_first_reading(tmp_path: Path) -> None:
    snap = _breaker(tmp_path / "s.json").health_snapshot()
    assert snap["persistence"] == "pending"
    assert snap["persisted_state_loaded"] is False


def test_snapshot_closed(tmp_path: Path) -> None:
    breaker = _breaker(tmp_path / "s.json")
    _feed(breaker, 100_000.0)
    _feed(breaker, 99_000.0)
    snap = breaker.health_snapshot()
    assert snap["state"] == "closed"
    assert snap["peak_equity"] == 100_000.0
    assert snap["peak_set_at"] is not None
    assert snap["current_drawdown"] == pytest.approx(0.01)
    assert snap["tripped_at"] is None
    assert snap["persistence"] == "failed"
    assert snap["load_failure_reason"] == "missing"


def test_snapshot_open() -> None:
    breaker = _breaker()
    _feed(breaker, 100_000.0)
    _feed(breaker, 90_000.0)
    snap = breaker.health_snapshot()
    assert snap["state"] == "open"
    assert snap["tripped_at"] is not None
    assert snap["current_drawdown"] == pytest.approx(0.10)
    assert snap["reset_threshold"] == pytest.approx(0.064)


def test_snapshot_restored_and_never_leaks_account(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    _write_state(path, state="open", tripped_at=circuit_breakers.safe_utcnow().isoformat())
    breaker = _breaker(path)
    _feed(breaker, 92_500.0)
    snap = breaker.health_snapshot()
    assert snap["state"] == "open"
    assert snap["peak_equity"] == 100_000.0
    assert snap["persistence"] == "restored"
    assert snap["persisted_state_loaded"] is True
    assert snap["load_failure_reason"] is None
    assert ACCOUNT not in json.dumps(snap)


@pytest.mark.parametrize(
    ("content", "reason"),
    [("{not json", "corrupt"), (None, "account_mismatch")],
)
def test_snapshot_load_failure_reason(tmp_path: Path, content: str | None, reason: str) -> None:
    path = tmp_path / "s.json"
    if content is None:
        _write_state(path, account_id="other-account")
    else:
        path.write_text(content, encoding="utf-8")
    breaker = _breaker(path)
    _feed(breaker, 100_000.0)
    snap = breaker.health_snapshot()
    assert snap["persistence"] == "failed"
    assert snap["persisted_state_loaded"] is False
    assert snap["load_failure_reason"] == reason


def test_section_unavailable_when_not_registered() -> None:
    assert drawdown_breaker_health_section() == {"available": False, "reason": "not_initialized"}


def test_payload_carries_section_without_changing_status() -> None:
    baseline = health_payload_module.build_runtime_health_payload()
    assert baseline["drawdown_breaker"] == {"available": False, "reason": "not_initialized"}

    breaker = _breaker()
    register_drawdown_breaker_for_health(breaker)
    _feed(breaker, 100_000.0)
    closed = health_payload_module.build_runtime_health_payload()
    _feed(breaker, 80_000.0)
    opened = health_payload_module.build_runtime_health_payload()

    assert closed["drawdown_breaker"]["state"] == "closed"
    assert opened["drawdown_breaker"]["state"] == "open"
    for key in ("ok", "status", "reason", "attention_flags", "readiness_failures"):
        assert closed.get(key) == baseline.get(key) == opened.get(key)
    assert ACCOUNT not in json.dumps(opened, default=str)


def test_payload_survives_snapshot_raising(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    baseline = health_payload_module.build_runtime_health_payload()

    def _boom(self: DrawdownCircuitBreaker) -> dict[str, Any]:
        raise RuntimeError("boom")

    breaker = _breaker()
    register_drawdown_breaker_for_health(breaker)
    monkeypatch.setattr(DrawdownCircuitBreaker, "health_snapshot", _boom)
    payload = health_payload_module.build_runtime_health_payload()
    assert payload["drawdown_breaker"] == {"available": False, "reason": "snapshot_failed"}
    assert payload["ok"] == baseline["ok"]
    assert payload["status"] == baseline["status"]
    assert any(r.getMessage() == "DRAWDOWN_BREAKER_HEALTH_UNAVAILABLE" for r in caplog.records)


def test_healthz_route_does_not_raise_when_snapshot_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_trading.app import create_app

    def _boom(self: DrawdownCircuitBreaker) -> dict[str, Any]:
        raise RuntimeError("boom")

    breaker = _breaker()  # the registry holds it weakly
    register_drawdown_breaker_for_health(breaker)
    monkeypatch.setattr(DrawdownCircuitBreaker, "health_snapshot", _boom)
    client = create_app().test_client()
    for route in ("/healthz", "/health"):
        resp = client.get(route)
        assert resp.status_code in (200, 503)
        body = resp.get_json()
        assert body["drawdown_breaker"] == {"available": False, "reason": "snapshot_failed"}
