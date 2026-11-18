from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ai_trading.monitoring import journal_alerts as ja

slack_srv = importlib.import_module("tools.mcp_slack_alerts_server")

T0 = datetime(2026, 11, 18, 15, 0, tzinfo=UTC)


def _ts(offset_s: float = 0) -> str:
    return str(int((T0 + timedelta(seconds=offset_s)).timestamp() * 1_000_000))


def _app(msg: str, level: str = "INFO", name: str = "ai_trading.x", off: float = 0, **extra: Any) -> dict[str, Any]:
    body = {"ts": "t", "level": level, "name": name, "msg": msg, **extra}
    return {
        "_SYSTEMD_UNIT": "ai-trading.service",
        "MESSAGE": json.dumps(body),
        "PRIORITY": "6",
        "__REALTIME_TIMESTAMP": _ts(off),
        "__CURSOR": f"c{off}",
    }


def _sysd(msg: str, off: float = 0) -> dict[str, Any]:
    return {
        "_SYSTEMD_UNIT": "init.scope",
        "UNIT": "ai-trading.service",
        "MESSAGE": msg,
        "PRIORITY": "6",
        "__REALTIME_TIMESTAMP": _ts(off),
        "__CURSOR": f"c{off}",
    }


def _text(message: dict[str, Any]) -> str:
    return json.dumps(message)


# --- alert 1: service lifecycle -------------------------------------------------


def test_manual_restart_is_start_only_not_failure() -> None:
    entries = [
        _sysd("ai-trading.service: Deactivated successfully.", 0),
        _sysd("Stopped ai-trading.service - AI Trading Bot.", 1),
        _sysd("Started ai-trading.service - AI Trading Bot.", 9),
    ]
    events = ja.detect(entries)
    assert [e["kind"] for e in events] == ["start"]
    msg = ja.build_messages(ja.apply_dedupe(events, {}, now=T0)[0])[0]
    assert "started" in _text(msg)
    assert "failed" not in _text(msg).lower().replace("no failure", "")


def test_crash_and_restart_is_critical_failure() -> None:
    entries = [
        _sysd("ai-trading.service: Main process exited, code=killed, status=9/KILL", 0),
        _sysd("ai-trading.service: Failed with result 'signal'.", 1),
        _sysd("ai-trading.service: Scheduled restart job, restart counter is at 3.", 11),
        _sysd("Started ai-trading.service - AI Trading Bot.", 12),
    ]
    events = ja.detect(entries)
    assert [e["kind"] for e in events] == ["failed", "failed", "restart", "start"]
    to_send, _ = ja.apply_dedupe(events, {}, now=T0)
    (msg,) = ja.build_messages(to_send)
    text = _text(msg)
    assert "failed or restarted unexpectedly" in text
    assert "status 9/KILL" in text
    assert "failed with result signal" in text
    assert "scheduled restart #3" in text


def test_clean_exit_is_not_a_failure() -> None:
    events = ja.detect([_sysd("ai-trading.service: Main process exited, code=exited, status=0/SUCCESS", 0)])
    assert events == []


def test_service_failure_cooldown_counts_suppressed() -> None:
    fail = ja.detect([_sysd("ai-trading.service: Failed with result 'exit-code'.", 0)])
    sent, state = ja.apply_dedupe(fail, {}, now=T0)
    assert len(sent) == 1
    sent, state = ja.apply_dedupe(fail, state, now=T0 + timedelta(minutes=2))
    assert sent == []
    assert state["service:failed"]["suppressed"] == 1
    sent, _ = ja.apply_dedupe(fail, state, now=T0 + timedelta(minutes=11))
    assert sent[0]["count"] == 2


def test_start_never_suppressed() -> None:
    start = ja.detect([_sysd("Started ai-trading.service - AI Trading Bot.", 0)])
    _, state = ja.apply_dedupe(start, {}, now=T0)
    sent, _ = ja.apply_dedupe(start, state, now=T0 + timedelta(seconds=30))
    assert len(sent) == 1


# --- alert 2: trading stopped / resumed -----------------------------------------


def test_drawdown_breaker_fields_as_the_breaker_logs_them() -> None:
    events = ja.detect(
        [_app("DRAWDOWN_BREAKER_TRIPPED", "ERROR", current_drawdown=0.0801, peak_equity=81655.0, equity=75114.0)]
    )
    sent, _ = ja.apply_dedupe(events, {}, now=T0)
    text = _text(ja.build_messages(sent)[0])
    assert "drawdown 8.0%" in text and "81,655" in text and "75,114" in text


def test_drawdown_breaker_tripped_and_reset_messages() -> None:
    trip = ja.detect(
        [_app("DRAWDOWN_BREAKER_TRIPPED", "ERROR", drawdown=0.0612, peak=101234.5, equity=95040.0)]
    )
    assert trip[0]["kind"] == "stopped"
    to_send, state = ja.apply_dedupe(trip, {}, now=T0)
    (msg,) = ja.build_messages(to_send)
    text = _text(msg)
    assert "Trading stopped" in text
    assert "drawdown 6.1%" in text
    assert "$101,234.50" in text and "$95,040.00" in text
    # trip markers are not double-reported as new errors
    assert all(e["alert"] == "trading" for e in trip)

    reset = ja.detect([_app("DRAWDOWN_BREAKER_RESET", "INFO", off=3600, drawdown=0.01, peak=101234.5, equity=100000.0)])
    to_send, _ = ja.apply_dedupe(reset, state, now=T0 + timedelta(hours=1))
    (msg,) = ja.build_messages(to_send)
    assert "Trading resumed" in _text(msg)


@pytest.mark.parametrize(
    "marker",
    ["HALT_FLAG_WRITTEN", "DISASTER_HALT_TRIGGERED", "ALERT_HALT_TRADING", "DATA_KILL_SWITCH_ACTIVATED"],
)
def test_runtime_halt_markers_detected(marker: str) -> None:
    events = ja.detect([_app(marker, "WARNING", reason="DISASTER_DRAW_DOWN_12.00%")])
    assert events and events[0]["kind"] == "stopped"
    (msg,) = ja.build_messages(ja.apply_dedupe(events, {}, now=T0)[0])
    assert "DISASTER_DRAW_DOWN_12.00%" in _text(msg)


def test_halt_flag_removed_is_resume() -> None:
    events = ja.detect([_app("HALT_FLAG_REMOVED", "INFO", path="/x")])
    assert events[0]["kind"] == "resumed"


def test_trading_marker_cooldown_and_combined_message() -> None:
    events = ja.detect(
        [
            _app("DISASTER_HALT_TRIGGERED", "ERROR", drawdown=0.2),
            _app("HALT_FLAG_WRITTEN", "WARNING", off=0, reason="DISASTER_DRAW_DOWN_20.00%"),
        ]
    )
    to_send, state = ja.apply_dedupe(events, {}, now=T0)
    assert len(ja.build_messages(to_send)) == 1  # two markers, one message
    again, _ = ja.apply_dedupe(events, state, now=T0 + timedelta(minutes=5))
    assert again == []


# --- alert 2b: position backstop exits --------------------------------------------


def _stop(symbol: str, off: float = 0, **extra: Any) -> dict[str, Any]:
    fields = {"symbol": symbol, "side": "long", "qty": 10, "avg_entry": 200.0, "price": 189.4,
              "loss_pct": -0.053, "threshold": -0.05, "lock_until": "2026-11-19T14:30:00+00:00"}
    fields.update(extra)
    return _app(f"POSITION_STOP_LOSS_BREACH | symbol={symbol}", "WARNING", off=off, **fields)


def test_position_backstop_markers_are_their_own_group() -> None:
    assert not set(ja.POSITION_BACKSTOP_MARKERS) & set(ja.TRADING_STOP_MARKERS)
    events = ja.detect(
        [
            _stop("AAPL"),
            _app("POSITION_MAX_HOLD_BREACH | symbol=MSFT", "WARNING", off=1, symbol="MSFT", side="short",
                 sessions_held=5),
            _app("POSITION_BACKSTOP_EXIT_STALLED | symbol=AAPL", "WARNING", off=2, symbol="AAPL", side="long", qty=10,
                 last_gates=["POSITION_STOP_LOSS_EXIT", "KILL_SWITCH_BLOCK"]),
            _app("POSITION_BACKSTOP_UNARMED | symbol=NVDA", "INFO", off=3, symbol="NVDA", reason="no_basis"),
            # A bare marker (no symbol suffix) is still recognised.
            _app("POSITION_MAX_HOLD_BREACH", "WARNING", off=4, symbol="AMD", side="long", sessions_held=6),
        ]
    )
    assert [(e["alert"], e["kind"], e["key"]) for e in events] == [
        ("position", "exit", "position:POSITION_STOP_LOSS_BREACH:AAPL"),
        ("position", "exit", "position:POSITION_MAX_HOLD_BREACH:MSFT"),
        ("position", "stalled", "position:POSITION_BACKSTOP_EXIT_STALLED:AAPL"),
        ("position", "exit", "position:POSITION_MAX_HOLD_BREACH:AMD"),
    ]
    to_send, _ = ja.apply_dedupe(events, {}, now=T0)
    texts = [_text(message) for message in ja.build_messages(to_send)]
    assert len(texts) == 3  # one message per marker type
    assert "Safety backstop: AAPL long stop-loss exit, -5.3% vs entry" in texts[0]
    assert "Safety backstop: max-hold exit for MSFT, AMD" in texts[1]
    assert "MSFT short max-hold exit, held 5 sessions" in texts[1]
    assert "Safety backstop: AAPL long exit stalled, still holding 10" in texts[2]
    assert "KILL_SWITCH_BLOCK" in texts[2]
    assert "Trading stopped" not in "".join(texts)


def test_position_backstop_dedupes_per_symbol_with_the_trading_cooldown() -> None:
    events = ja.detect([_stop("AAPL"), _stop("MSFT", off=1, loss_pct=-0.061), _stop("AAPL", off=2)])
    to_send, state = ja.apply_dedupe(events, {}, now=T0)
    assert sorted(e["symbol"] for e in to_send) == ["AAPL", "MSFT"]
    assert [e["count"] for e in to_send if e["symbol"] == "AAPL"] == [2]

    within, state = ja.apply_dedupe(ja.detect([_stop("AAPL", off=60)]), state, now=T0 + timedelta(minutes=5))
    assert within == []
    after, _ = ja.apply_dedupe(ja.detect([_stop("AAPL", off=1200)]), state, now=T0 + ja.TRADING_COOLDOWN)
    assert [e["count"] for e in after] == [2]


def test_position_backstop_alert_never_posts_from_tests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> int:
        raise AssertionError("must not post")

    monkeypatch.setattr(slack_srv, "_post_slack_message", boom)
    out = slack_srv.tool_notify_journal_alerts(
        {"dry_run": True, "state_path": str(tmp_path / "inc.json"), "journal_runner": FakeJournal([_stop("AAPL")])}
    )
    assert out["messages"] == 1 and out["sent"] is False
    assert "Safety backstop: AAPL long stop-loss exit" in _text(out["message_preview"][0])


# --- alert 3: new error types ----------------------------------------------------


def test_new_error_alerts_once_then_counts_repeats() -> None:
    err = lambda off: _app("BROKER_CALL_FAILED", "ERROR", "ai_trading.logging", off, age_s=int(off))  # noqa: E731
    events = ja.detect([err(0), err(1), err(2)])
    assert len({e["key"] for e in events}) == 1
    sent, state = ja.apply_dedupe(events, {}, now=T0)
    assert len(sent) == 1 and sent[0]["count"] == 3 and sent[0]["new"]
    (msg,) = ja.build_messages(sent)
    assert "New ERROR: BROKER_CALL_FAILED" in _text(msg)

    sent, state = ja.apply_dedupe(ja.detect([err(60)]), state, now=T0 + timedelta(minutes=1))
    assert sent == []
    sent, state = ja.apply_dedupe(ja.detect([err(70)]), state, now=T0 + timedelta(minutes=30))
    assert sent == []
    sent, _ = ja.apply_dedupe(ja.detect([err(80)]), state, now=T0 + timedelta(minutes=61))
    assert sent[0]["count"] == 3 and not sent[0]["new"]
    (msg,) = ja.build_messages(sent)
    assert "Repeated ERROR" in _text(msg) and "3 since" in _text(msg)


def test_standing_error_repeats_at_most_once_a_day() -> None:
    err = lambda off: _app("MODEL_REGISTRY_STALE", "ERROR", "ai_trading.logging", off, age_s=int(off))  # noqa: E731
    sent, state = ja.apply_dedupe(ja.detect([err(0)]), {}, now=T0)
    assert len(sent) == 1 and sent[0]["new"]
    for hours in (2, 12, 23):
        sent, state = ja.apply_dedupe(ja.detect([err(hours)]), state, now=T0 + timedelta(hours=hours))
        assert sent == []
    sent, _ = ja.apply_dedupe(ja.detect([err(99)]), state, now=T0 + timedelta(hours=24, minutes=1))
    assert len(sent) == 1 and sent[0]["count"] == 4 and not sent[0]["new"]


def test_error_types_with_different_numbers_group_together() -> None:
    a = ja.detect([_app("Order 123 rejected for AAPL", "ERROR")])[0]["key"]
    b = ja.detect([_app("Order 456 rejected for AAPL", "ERROR")])[0]["key"]
    c = ja.detect([_app("Different failure", "CRITICAL")])[0]["key"]
    assert a == b and a != c


def test_warnings_and_info_ignored_and_stderr_priority_error_detected() -> None:
    assert ja.detect([_app("X", "WARNING"), _app("Y", "INFO")]) == []
    raw = {
        "_SYSTEMD_UNIT": "ai-trading.service",
        "MESSAGE": "Traceback boom",
        "PRIORITY": "3",
        "__REALTIME_TIMESTAMP": _ts(),
    }
    events = ja.detect([raw])
    assert events[0]["alert"] == "error"


def test_state_pruned_after_retention() -> None:
    old = {"error:a": {"last_sent": T0.isoformat(), "suppressed": 0, "last_seen": (T0 - timedelta(days=8)).isoformat()}}
    _, state = ja.apply_dedupe([], old, now=T0)
    assert state == {}


# --- journal reading and the sender -------------------------------------------------


class FakeJournal:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> str:
        self.calls.append(list(argv))
        rows = self.rows
        if "-n" in argv:
            rows = rows[-1:]
        else:
            cursor = next((a.split("=", 1)[1] for a in argv if a.startswith("--after-cursor=")), None)
            if cursor:
                idx = [r["__CURSOR"] for r in rows].index(cursor)
                rows = rows[idx + 1 :]
        return "\n".join(json.dumps(r) for r in rows)


def test_first_run_initializes_cursor_without_alerting(tmp_path: Path) -> None:
    journal = FakeJournal([_app("OLD_ERROR", "ERROR", off=0)])
    args = {"state_path": str(tmp_path / "inc.json"), "journal_runner": journal}
    out = slack_srv.tool_notify_journal_alerts(args)
    assert out["reason"] == "journal_cursor_initialized" and out["sent"] is False
    saved = json.loads((tmp_path / "slack_journal_alert_state.json").read_text())
    assert saved["cursor"] == "c0"


def test_send_flow_dedupes_and_advances_cursor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    posted: list[dict[str, Any]] = []
    monkeypatch.setattr(slack_srv, "_post_slack_message", lambda url, payload, timeout_s=5.0: posted.append(payload) or 200)
    journal = FakeJournal([_app("SEED", "INFO", off=0)])
    args = {"state_path": str(tmp_path / "inc.json"), "journal_runner": journal, "webhook_url": "https://hooks.invalid/x"}
    slack_srv.tool_notify_journal_alerts(args)
    journal.rows += [_app("BOOM", "ERROR", off=1), _app("BOOM", "ERROR", off=2)]
    out = slack_srv.tool_notify_journal_alerts(args)
    assert out["sent"] is True and len(posted) == 1
    out = slack_srv.tool_notify_journal_alerts(args)  # nothing new after the cursor
    assert out["sent"] is False and len(posted) == 1
    journal.rows.append(_app("BOOM", "ERROR", off=3))
    out = slack_srv.tool_notify_journal_alerts(args)  # within cool-down
    assert out["sent"] is False and len(posted) == 1


def test_dry_run_sends_nothing_writes_nothing_needs_no_webhook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: Any, **_k: Any) -> int:
        raise AssertionError("must not post")

    monkeypatch.setattr(slack_srv, "_post_slack_message", boom)
    monkeypatch.delenv("AI_TRADING_SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    journal = FakeJournal([_app("DRAWDOWN_BREAKER_TRIPPED", "ERROR", drawdown=0.07)])
    state_dir = tmp_path / "state"
    out = slack_srv.tool_notify_journal_alerts(
        {"dry_run": True, "state_path": str(state_dir / "inc.json"), "journal_runner": journal}
    )
    assert out["reason"] == "dry_run" and out["messages"] == 1 and out["sent"] is False
    assert not state_dir.exists()


def test_unreadable_journal_reports_and_keeps_state(tmp_path: Path) -> None:
    def broken(_argv: Sequence[str]) -> str:
        raise ja.JournalReadError("no access")

    out = slack_srv.tool_notify_journal_alerts(
        {"state_path": str(tmp_path / "inc.json"), "journal_runner": broken}
    )
    assert out["reason"] == "journal_unreadable"
    assert not (tmp_path / "slack_journal_alert_state.json").exists()


def test_position_alerts_group_per_marker_and_keep_per_symbol_keys() -> None:
    events = ja.detect(
        [
            _stop("AAPL"),
            _stop("MSFT", off=1, loss_pct=-0.061),
            _app("POSITION_MAX_HOLD_BREACH | symbol=NVDA", "WARNING", off=2, symbol="NVDA", side="long", sessions_held=5),
        ]
    )
    to_send, state = ja.apply_dedupe(events, {}, now=T0)
    assert sorted(state) == [
        "position:POSITION_MAX_HOLD_BREACH:NVDA",
        "position:POSITION_STOP_LOSS_BREACH:AAPL",
        "position:POSITION_STOP_LOSS_BREACH:MSFT",
    ]
    messages = ja.build_messages(to_send)
    assert len(messages) == 2  # one per marker type
    stop_text = _text(messages[0])
    assert "Safety backstop: stop-loss exit for AAPL, MSFT" in stop_text
    assert "AAPL long stop-loss exit, -5.3% vs entry" in stop_text
    assert "MSFT long stop-loss exit, -6.1% vs entry" in stop_text
    assert "Safety backstop: NVDA long max-hold exit, held 5 sessions" in _text(messages[1])


def test_position_alert_symbol_falls_back_to_the_marker_suffix() -> None:
    (event,) = ja.detect([_app("POSITION_STOP_LOSS_BREACH | symbol=TSLA", "WARNING", loss_pct=-0.07)])
    assert event["symbol"] == "TSLA"
    assert event["key"] == "position:POSITION_STOP_LOSS_BREACH:TSLA"
