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
    assert snap["state_load"] == "not_configured"
    assert snap["last_reading_at"] is None
    assert snap["last_write_failed"] is False
    assert snap["persisted_state_loaded"] is False
    json.dumps(snap)


def test_snapshot_pending_until_first_reading(tmp_path: Path) -> None:
    snap = _breaker(tmp_path / "s.json").health_snapshot()
    assert snap["state_load"] == "pending"
    assert snap["state"] == "unknown"
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
    assert snap["state_load"] == "fresh"
    assert snap["load_failure_reason"] is None
    assert snap["state"] == "closed"
    assert snap["last_reading_at"] is not None


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
    assert snap["state_load"] == "restored"
    assert snap["persisted_state_loaded"] is True
    assert snap["load_failure_reason"] is None
    assert ACCOUNT not in json.dumps(snap)


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("{not json", "corrupt"),
        (None, "account_mismatch"),
        ("schema", "schema_mismatch"),
        ("peak", "peak_invalid"),
    ],
)
def test_snapshot_load_failure_reason(tmp_path: Path, content: str | None, reason: str) -> None:
    path = tmp_path / "s.json"
    if content is None:
        _write_state(path, account_id="other-account")
    elif content == "schema":
        _write_state(path, schema_version=99)
    elif content == "peak":
        _write_state(path, peak_equity=-5.0)
    else:
        path.write_text(content, encoding="utf-8")
    breaker = _breaker(path)
    _feed(breaker, 100_000.0)
    snap = breaker.health_snapshot()
    assert snap["state_load"] == "failed"
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


def test_healthz_route_matches_healthy_baseline_when_snapshot_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_trading.app import create_app

    breaker = _breaker()  # the registry holds it weakly
    register_drawdown_breaker_for_health(breaker)
    client = create_app().test_client()
    baselines = {route: client.get(route) for route in ("/healthz", "/health")}
    for resp in baselines.values():
        assert resp.get_json()["drawdown_breaker"]["available"] is True

    def _boom(self: DrawdownCircuitBreaker) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(DrawdownCircuitBreaker, "health_snapshot", _boom)
    for route, base in baselines.items():
        resp = client.get(route)
        body = resp.get_json()
        assert resp.status_code == base.status_code
        assert body["status"] == base.get_json()["status"]
        assert body["ok"] == base.get_json()["ok"]
        assert body["drawdown_breaker"] == {"available": False, "reason": "snapshot_failed"}


def test_healthz_route_renders_non_finite_peak_as_null() -> None:
    from ai_trading.app import create_app

    breaker = _breaker()
    register_drawdown_breaker_for_health(breaker)
    breaker.peak_equity = float("inf")
    resp = create_app().test_client().get("/healthz")
    assert resp.get_json()["drawdown_breaker"]["peak_equity"] is None


def test_saved_open_state_is_unknown_until_first_reading(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    _write_state(path, state="open", tripped_at=circuit_breakers.safe_utcnow().isoformat())
    breaker = _breaker(path)
    before = breaker.health_snapshot()
    assert before["state"] == "unknown"
    assert before["state_load"] == "pending"
    _feed(breaker, 92_500.0)
    after = breaker.health_snapshot()
    assert after["state"] == "open"
    assert after["state_load"] == "restored"


def test_unreadable_file_stays_pending_with_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "s.json"
    _write_state(path)

    def _deny(self: Path) -> bytes:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_bytes", _deny)
    breaker = _breaker(path)
    _feed(breaker, 100_000.0)
    snap = breaker.health_snapshot()
    assert snap["state_load"] == "pending"
    assert snap["state"] == "unknown"
    assert snap["load_failure_reason"] == "unreadable"
    assert snap["persisted_state_loaded"] is False


def test_last_write_failed_tracks_write_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = circuit_breakers.atomic_write_text
    fail = {"on": True}

    def flaky(p: Any, content: str, **kwargs: Any) -> Path:
        if fail["on"]:
            raise OSError("disk full")
        return real(p, content, **kwargs)

    monkeypatch.setattr(circuit_breakers, "atomic_write_text", flaky)
    breaker = _breaker(tmp_path / "s.json")
    _feed(breaker, 100_000.0)
    assert breaker.health_snapshot()["last_write_failed"] is True
    fail["on"] = False
    _feed(breaker, 101_000.0)
    assert breaker.health_snapshot()["last_write_failed"] is False


def test_last_reading_at_is_set_by_accepted_readings_only() -> None:
    breaker = _breaker()
    breaker.score_account_snapshot({"equity": None})
    assert breaker.health_snapshot()["last_reading_at"] is None
    _feed(breaker, 100_000.0)
    assert breaker.health_snapshot()["last_reading_at"] == breaker.last_reading_at.isoformat()


def test_health_path_never_reads_the_state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "s.json"
    _write_state(path)
    breaker = _breaker(path)
    register_drawdown_breaker_for_health(breaker)
    calls = {"n": 0}
    real = Path.read_bytes

    def counting(self: Path) -> bytes:
        calls["n"] += 1
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    health_payload_module.build_runtime_health_payload()
    breaker.health_snapshot()
    drawdown_breaker_health_section()
    assert calls["n"] == 0


def test_unavailable_log_is_rate_limited(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from ai_trading.logging import emit_once as emit_once_module

    monkeypatch.setattr(emit_once_module, "_emitted", {})

    def _boom() -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(circuit_breakers, "drawdown_breaker_health_section", _boom)
    for _ in range(3):
        assert health_payload_module._drawdown_breaker_snapshot()["reason"] == "snapshot_failed"
    hits = [r for r in caplog.records if r.getMessage() == "DRAWDOWN_BREAKER_HEALTH_UNAVAILABLE"]
    assert len(hits) == 1


def _build_lazy_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    from ai_trading.core import bot_engine

    monkeypatch.setattr(bot_engine, "_resolve_runtime_artifact_path", lambda rel: tmp_path / "breaker.json")
    lazy = bot_engine.LazyBotContext()
    lazy._ensure_initialized()
    return lazy


def test_lazy_context_registers_its_breaker_for_health(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lazy = _build_lazy_context(monkeypatch, tmp_path)
    breaker = lazy._context.drawdown_circuit_breaker
    assert isinstance(breaker, DrawdownCircuitBreaker)
    section = drawdown_breaker_health_section()
    assert section["available"] is True
    assert section["state"] == "unknown"
    assert section == breaker.health_snapshot()
