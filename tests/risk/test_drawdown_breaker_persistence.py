"""The drawdown breaker's peak and open state survive a restart via its state file."""
from __future__ import annotations

import json
import logging
import math
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.risk import circuit_breakers
from ai_trading.risk.circuit_breakers import (
    DRAWDOWN_BREAKER_STATE_SCHEMA_VERSION,
    CircuitBreakerState,
    DrawdownCircuitBreaker,
)

ACCOUNT = "acct-1"


def _breaker(path: Path | None) -> DrawdownCircuitBreaker:
    # Same construction as LazyBotContext with MAX_DRAWDOWN_THRESHOLD=0.08.
    return DrawdownCircuitBreaker(max_drawdown=0.08, recovery_threshold=0.8, state_path=path)


def _feed(breaker: DrawdownCircuitBreaker, equity: float, account: str | None = ACCOUNT) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"equity": equity}
    if account is not None:
        snapshot["id"] = account
    return breaker.score_account_snapshot(snapshot)


def _load_failures(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == "DRAWDOWN_BREAKER_STATE_LOAD_FAILED"]


def _write_state(path: Path, **overrides: Any) -> None:
    payload: dict[str, Any] = {
        "schema_version": DRAWDOWN_BREAKER_STATE_SCHEMA_VERSION,
        "account_id": ACCOUNT,
        "peak_equity": 100_000.0,
        "peak_set_at": (circuit_breakers.safe_utcnow() - timedelta(days=30)).isoformat(),
        "state": "closed",
        "tripped_at": None,
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class _WriteCounter:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls = 0
        real = circuit_breakers.atomic_write_text

        def counting(path: Any, content: str, **kwargs: Any) -> Path:
            self.calls += 1
            return real(path, content, **kwargs)

        monkeypatch.setattr(circuit_breakers, "atomic_write_text", counting)


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "runtime" / "drawdown_breaker_state.json"


# ---- writes ---------------------------------------------------------------


def test_writes_only_on_new_peak_trip_and_reset(state_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writes = _WriteCounter(monkeypatch)
    breaker = _breaker(state_path)

    _feed(breaker, 100_000.0)  # first peak
    assert writes.calls == 1
    _feed(breaker, 99_000.0)  # drawdown, no change to persist
    _feed(breaker, 99_000.0)
    assert writes.calls == 1
    _feed(breaker, 101_000.0)  # new peak
    assert writes.calls == 2
    _feed(breaker, 92_000.0)  # trip (8.9%)
    assert writes.calls == 3
    assert json.loads(state_path.read_text())["state"] == "open"
    _feed(breaker, 93_000.0)  # still open, nothing changed
    assert writes.calls == 3
    _feed(breaker, 95_000.0)  # 5.9% <= 6.4%: reset
    assert writes.calls == 4
    stored = json.loads(state_path.read_text())
    assert stored["state"] == "closed"
    assert stored["tripped_at"] is None
    assert stored["peak_equity"] == 101_000.0
    assert stored["account_id"] == ACCOUNT
    assert stored["schema_version"] == DRAWDOWN_BREAKER_STATE_SCHEMA_VERSION


def test_no_write_when_restored_state_is_unchanged(state_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_state(state_path)
    writes = _WriteCounter(monkeypatch)
    breaker = _breaker(state_path)

    _feed(breaker, 97_000.0)
    _feed(breaker, 98_000.0)

    assert writes.calls == 0


# ---- restore --------------------------------------------------------------


def test_round_trip_restores_peak_and_peak_time(state_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    first = _breaker(state_path)
    _feed(first, 100_000.0)
    peak_set_at = first.peak_set_at

    second = _breaker(state_path)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        payload = _feed(second, 96_000.0)

    assert payload["peak_equity"] == 100_000.0
    assert payload["current_drawdown"] == pytest.approx(0.04)
    assert second.peak_set_at == peak_set_at
    assert [r for r in caplog.records if r.getMessage() == "DRAWDOWN_BREAKER_STATE_RESTORED"]
    assert not _load_failures(caplog)


def test_drawdown_across_a_restart_trips(state_path: Path) -> None:
    before = _breaker(state_path)
    _feed(before, 100_000.0)
    _feed(before, 95_000.0)  # 5%, closed
    assert before.state is CircuitBreakerState.CLOSED

    after = _breaker(state_path)  # restart
    payload = _feed(after, 91_500.0)  # 8.5% from the pre-restart peak

    assert payload["open"] is True
    assert after.state is CircuitBreakerState.OPEN


def test_restart_does_not_clear_a_tripped_breaker(state_path: Path) -> None:
    before = _breaker(state_path)
    _feed(before, 100_000.0)
    _feed(before, 91_000.0)
    tripped_at = before.halt_timestamp
    assert before.state is CircuitBreakerState.OPEN

    after = _breaker(state_path)
    payload = _feed(after, 93_000.0)  # 7%: above the 6.4% reset level

    assert payload["open"] is True
    assert after.halt_timestamp == tripped_at

    # The existing reset rule still applies after the restart.
    reset = _feed(after, 94_000.0)  # 6% <= 6.4%
    assert reset["open"] is False
    assert json.loads(state_path.read_text())["state"] == "closed"


def test_missing_file_starts_fresh_with_one_warning(state_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    breaker = _breaker(state_path)
    with caplog.at_level(logging.INFO):
        first = _feed(breaker, 81_000.0)
        _feed(breaker, 80_000.0)

    failures = _load_failures(caplog)
    assert len(failures) == 1
    assert failures[0].reason == "missing"
    assert first == {
        "open": False,
        "current_drawdown": 0.0,
        "max_drawdown": 0.08,
        "peak_equity": 81_000.0,
        "equity": 81_000.0,
    }
    assert state_path.exists()


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("{not json", "corrupt"),
        ("[]", "corrupt"),
        (None, "schema_mismatch"),
        ("nan", "peak_invalid"),
        ("zero", "peak_invalid"),
        ("future", "future_timestamp"),
        ("naive", "timestamp_invalid"),
        ("open_without_trip_time", "timestamp_invalid"),
        ("bad_state", "state_invalid"),
        ("other_account", "account_mismatch"),
    ],
)
def test_unusable_file_starts_fresh_and_never_raises(
    state_path: Path, caplog: pytest.LogCaptureFixture, content: str | None, reason: str
) -> None:
    future = (circuit_breakers.safe_utcnow() + timedelta(days=1)).isoformat()
    if content in {"{not json", "[]"}:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(content, encoding="utf-8")
    elif content is None:
        _write_state(state_path, schema_version=99)
    elif content == "nan":
        # json.dumps writes NaN, which json.loads reads back as float('nan').
        _write_state(state_path, peak_equity=math.nan)
    elif content == "zero":
        _write_state(state_path, peak_equity=0)
    elif content == "future":
        _write_state(state_path, peak_set_at=future)
    elif content == "naive":
        _write_state(state_path, peak_set_at="2026-11-01T10:00:00")
    elif content == "open_without_trip_time":
        _write_state(state_path, state="open", tripped_at=None)
    elif content == "bad_state":
        _write_state(state_path, state="tripped")
    else:
        _write_state(state_path, account_id="acct-2", state="open", tripped_at=future)

    breaker = _breaker(state_path)
    with caplog.at_level(logging.INFO):
        payload = _feed(breaker, 50_000.0)

    failures = _load_failures(caplog)
    assert [r.reason for r in failures] == [reason]
    # Fresh start: the first reading is the peak and the breaker is closed.
    assert payload["open"] is False
    assert payload["peak_equity"] == 50_000.0
    stored = json.loads(state_path.read_text())
    assert stored["peak_equity"] == 50_000.0
    assert stored["account_id"] == ACCOUNT


def test_state_without_account_is_accepted_and_gains_the_account(state_path: Path) -> None:
    _write_state(state_path, account_id=None)
    breaker = _breaker(state_path)

    payload = _feed(breaker, 99_000.0)

    assert payload["peak_equity"] == 100_000.0
    assert json.loads(state_path.read_text())["account_id"] == ACCOUNT


def test_unknown_current_account_keeps_the_stored_one(state_path: Path) -> None:
    _write_state(state_path)
    breaker = _breaker(state_path)

    _feed(breaker, 101_000.0, account=None)

    stored = json.loads(state_path.read_text())
    assert stored["peak_equity"] == 101_000.0
    assert stored["account_id"] == ACCOUNT


def test_object_snapshot_account_id_is_read(state_path: Path) -> None:
    breaker = _breaker(state_path)

    breaker.score_account_snapshot(SimpleNamespace(equity="100000", id=ACCOUNT))

    assert json.loads(state_path.read_text())["account_id"] == ACCOUNT


def test_write_failure_is_logged_and_retried(
    state_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = circuit_breakers.atomic_write_text

    def failing(path: Any, content: str, **kwargs: Any) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(circuit_breakers, "atomic_write_text", failing)
    breaker = _breaker(state_path)
    with caplog.at_level(logging.INFO):
        payload = _feed(breaker, 100_000.0)
    assert payload["peak_equity"] == 100_000.0
    assert [r for r in caplog.records if r.getMessage() == "DRAWDOWN_BREAKER_STATE_WRITE_FAILED"]

    monkeypatch.setattr(circuit_breakers, "atomic_write_text", real)
    _feed(breaker, 99_000.0)  # nothing new, but the pending write goes out
    assert json.loads(state_path.read_text())["peak_equity"] == 100_000.0


# ---- no persistence configured -------------------------------------------


def test_no_file_access_without_state_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("file access without state_path")

    monkeypatch.setattr(circuit_breakers, "atomic_write_text", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    breaker = _breaker(None)

    _feed(breaker, 100_000.0)
    _feed(breaker, 91_000.0)
    assert breaker.state is CircuitBreakerState.OPEN
    _feed(breaker, 94_000.0)
    assert breaker.state is CircuitBreakerState.CLOSED
    assert list(workdir.iterdir()) == []


def test_update_equity_directly_also_restores_first(state_path: Path) -> None:
    _write_state(state_path)
    breaker = _breaker(state_path)

    assert breaker.update_equity(91_000.0) is False
    assert breaker.state is CircuitBreakerState.OPEN


def test_runtime_state_path_resolves_under_the_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai_trading.core import bot_engine

    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))

    resolved = bot_engine._resolve_runtime_artifact_path(
        circuit_breakers.DRAWDOWN_BREAKER_STATE_PATH
    )

    assert resolved == (tmp_path / "runtime" / "drawdown_breaker_state.json").resolve()
