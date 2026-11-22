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
        ("huge_int", "corrupt"),
        ("year_one", "corrupt"),
        ("nested", "corrupt"),
        ("not_utf8", "corrupt"),
    ],
)
def test_unusable_file_starts_fresh_and_never_raises(
    state_path: Path, caplog: pytest.LogCaptureFixture, content: str | None, reason: str
) -> None:
    future = (circuit_breakers.safe_utcnow() + timedelta(days=1)).isoformat()
    if content in {"{not json", "[]", "nested"}:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        text = "[" * 100_000 + "]" * 100_000 if content == "nested" else content
        state_path.write_text(text, encoding="utf-8")
    elif content == "not_utf8":
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_bytes(b"\xff\xfe{")
    elif content == "huge_int":
        _write_state(state_path, peak_equity=10**400)  # float() raises OverflowError
    elif content == "year_one":
        # astimezone(UTC) raises OverflowError below datetime.min.
        _write_state(state_path, peak_set_at="0001-01-01T00:00:00+05:00")
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
    monkeypatch.setattr(Path, "read_bytes", forbidden)
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


# ---- follow-ups: retries, merge, restored open state ------------------------


def test_failed_write_at_trip_is_retried_while_still_open(
    state_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    breaker = _breaker(state_path)
    _feed(breaker, 100_000.0)
    real = circuit_breakers.atomic_write_text

    def failing(path: Any, content: str, **kwargs: Any) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(circuit_breakers, "atomic_write_text", failing)
    _feed(breaker, 91_000.0)  # trips; the write fails
    assert json.loads(state_path.read_text())["state"] == "closed"

    monkeypatch.setattr(circuit_breakers, "atomic_write_text", real)
    payload = _feed(breaker, 91_500.0)  # still open (8.5%)
    assert payload["open"] is True
    stored = json.loads(state_path.read_text())
    assert stored["state"] == "open"
    assert stored["tripped_at"] is not None


def _unreadable(monkeypatch: pytest.MonkeyPatch, path: Path, exc: OSError) -> list[int]:
    calls = [0]
    real = Path.read_bytes

    def read_bytes(self: Path) -> bytes:
        if self == path and calls[0] >= 0:
            calls[0] += 1
            raise exc
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return calls


def test_unreadable_file_is_retried_not_overwritten(
    state_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _write_state(state_path, state="open", tripped_at=(circuit_breakers.safe_utcnow() - timedelta(days=1)).isoformat())
    original = state_path.read_text()
    calls = _unreadable(monkeypatch, state_path, PermissionError("denied"))
    breaker = _breaker(state_path)

    with caplog.at_level(logging.INFO):
        first = _feed(breaker, 99_000.0)
        _feed(breaker, 98_000.0)
        _feed(breaker, 105_000.0)  # new in-memory peak: still nothing written

    assert calls[0] == 3  # retried at every reading
    assert state_path.read_text(encoding="utf-8") == original
    failures = _load_failures(caplog)
    assert [r.reason for r in failures] == ["unreadable"]  # rate-limited
    assert first["open"] is False  # in memory only while unreadable


def test_deferred_load_merges_higher_peak_and_open_state(
    state_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tripped_at = (circuit_breakers.safe_utcnow() - timedelta(days=1)).isoformat()
    _write_state(state_path, state="open", tripped_at=tripped_at)  # peak 100k, open
    calls = _unreadable(monkeypatch, state_path, OSError(5, "EIO"))
    breaker = _breaker(state_path)
    _feed(breaker, 90_000.0)
    assert breaker.state is CircuitBreakerState.CLOSED

    calls[0] = -1  # the disk recovers
    with caplog.at_level(logging.INFO):
        payload = _feed(breaker, 92_000.0)  # 8% from the stored 100k peak

    assert payload["open"] is True
    assert payload["peak_equity"] == 100_000.0
    assert [r for r in caplog.records if r.getMessage() == "DRAWDOWN_BREAKER_RESTORED_OPEN"]
    stored = json.loads(state_path.read_text())
    assert stored["state"] == "open" and stored["peak_equity"] == 100_000.0


def test_deferred_load_keeps_a_higher_in_memory_peak(
    state_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_state(state_path)  # peak 100k, closed
    calls = _unreadable(monkeypatch, state_path, PermissionError("denied"))
    breaker = _breaker(state_path)
    _feed(breaker, 110_000.0)
    in_memory_set_at = breaker.peak_set_at

    calls[0] = -1
    _feed(breaker, 109_000.0)

    assert breaker.peak_equity == 110_000.0
    assert breaker.peak_set_at == in_memory_set_at
    assert json.loads(state_path.read_text())["peak_equity"] == 110_000.0


def test_restored_open_logs_a_critical_stop_marker(state_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write_state(state_path, state="open", tripped_at=(circuit_breakers.safe_utcnow() - timedelta(hours=2)).isoformat())
    breaker = _breaker(state_path)

    with caplog.at_level(logging.INFO):
        _feed(breaker, 93_000.0)

    (record,) = [r for r in caplog.records if r.getMessage() == "DRAWDOWN_BREAKER_RESTORED_OPEN"]
    assert record.levelno == logging.CRITICAL
    assert record.reason == "restored_from_state"
    assert record.current_drawdown == pytest.approx(0.07)
    assert record.peak_equity == 100_000.0


def test_restored_open_between_reset_and_trip_level_stays_open(state_path: Path) -> None:
    _write_state(state_path, state="open", tripped_at=(circuit_breakers.safe_utcnow() - timedelta(hours=2)).isoformat())
    breaker = _breaker(state_path)

    assert _feed(breaker, 93_000.0)["open"] is True  # 7%
    assert _feed(breaker, 93_500.0)["open"] is True  # 6.5% > 6.4%


def test_restored_open_with_a_new_high_resets_and_persists(state_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write_state(state_path, state="open", tripped_at=(circuit_breakers.safe_utcnow() - timedelta(hours=2)).isoformat())
    breaker = _breaker(state_path)

    with caplog.at_level(logging.INFO):
        payload = _feed(breaker, 101_000.0)

    assert payload["open"] is False
    assert payload["peak_equity"] == 101_000.0
    assert [r for r in caplog.records if r.getMessage() == "DRAWDOWN_BREAKER_RESET"]
    stored = json.loads(state_path.read_text())
    assert (stored["state"], stored["peak_equity"], stored["tripped_at"]) == ("closed", 101_000.0, None)


@pytest.mark.parametrize("text", ["[" * 100_000 + "]" * 100_000, '{"schema_version": 1, "peak_equity": 1' + "0" * 400 + "}"])
def test_bad_file_cannot_abort_the_cycle_feed(state_path: Path, text: str) -> None:
    from ai_trading.core import bot_engine

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(text, encoding="utf-8")
    runtime = SimpleNamespace(drawdown_circuit_breaker=_breaker(state_path))
    exec_engine = SimpleNamespace(_get_account_snapshot=lambda: SimpleNamespace(equity="80000", id=ACCOUNT))

    payload = bot_engine._score_drawdown_breaker(runtime, exec_engine)

    assert payload["open"] is False
    assert payload["peak_equity"] == 80_000.0
