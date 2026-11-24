"""Position backstop: stop loss, maximum hold, post-exit lock and position cap."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from ai_trading.config.runtime import TradingConfig
from ai_trading.core import bot_engine, netting_position_backstop as nbp, netting_symbol_cycle
from ai_trading.core.execution_guards import evaluate_execution_approval
from ai_trading.core.netting import NettedTarget, SleeveConfig, SleeveProposal
from ai_trading.core.netting_execution_context import _NON_DISABLEABLE_GATE_NAMES
from ai_trading.core.netting_position_backstop import (
    BackstopStore,
    Episode,
    Lock,
    PositionBackstop,
    PositionBasis,
    evaluate,
    next_session_open,
    sessions_held,
)
from ai_trading.core.netting_submit_prelude import prepare_netting_submit_prelude
from ai_trading.core.netting_symbol_adjustments import apply_symbol_adjustments
from ai_trading.core.netting_symbol_approval import prepare_netting_symbol_approval
from ai_trading.core.netting_symbol_cycle import process_netting_symbol
from ai_trading.core.netting_symbol_prelude import prepare_netting_symbol_prelude
from ai_trading.execution.engine import BrokerSyncResult
from ai_trading.policy.compiler import ExecutionApproval
from ai_trading.utils.time import monotonic_time
from tests.test_netting_submit_prelude import _base_kwargs as _submit_prelude_kwargs
from tests.test_netting_symbol_cycle import _make_net_target, _make_processor

ET = ZoneInfo("America/New_York")
# Wednesday 2026-11-18, 10:00 ET: a regular session.
NOW = datetime(2026, 11, 18, 10, 0, tzinfo=ET).astimezone(UTC)
NEXT_OPEN = datetime(2026, 11, 19, 9, 30, tzinfo=ET).astimezone(UTC)
LOGGER = logging.getLogger("tests.position_backstop")
OPEN_BREAKER = {"open": True, "current_drawdown": 0.09, "max_drawdown": 0.08}


def _et(*args: int) -> datetime:
    return datetime(*args, tzinfo=ET).astimezone(UTC)


def _messages(caplog: pytest.LogCaptureFixture, name: str) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.getMessage().split(" | ", 1)[0] == name]


def test_per_symbol_markers_survive_the_runtime_log_throttle(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from ai_trading.logging import MessageThrottleFilter

    throttle = MessageThrottleFilter(throttle_seconds=5.0)
    with caplog.at_level(logging.INFO):
        backstop = _backstop(tmp_path, positions={"AAPL": 10, "MSFT": 5}, bases={"AAPL": 100.0, "MSFT": 100.0})
        for symbol, qty in (("AAPL", 10), ("MSFT", 5)):
            backstop.evaluate_symbol(symbol=symbol, current_shares=qty, delta_shares=0, price=90.0, bar_ts=NOW)

    breaches = _messages(caplog, "POSITION_STOP_LOSS_BREACH")
    assert [record.getMessage() for record in breaches] == [
        "POSITION_STOP_LOSS_BREACH | symbol=AAPL",
        "POSITION_STOP_LOSS_BREACH | symbol=MSFT",
    ]
    assert all(throttle.filter(record) for record in breaches)


def _episode(opened_at: datetime, side: str = "long") -> Episode:
    return Episode(opened_at=opened_at, opened_at_source="observed", side=side)


# ---- pure evaluation ------------------------------------------------------


def _eval(current: int, price: float, avg: float | None, **kwargs: Any) -> nbp.Evaluation:
    return evaluate(
        current_shares=current,
        delta_shares=kwargs.pop("delta", 0),
        price=price,
        now=kwargs.pop("now", NOW),
        basis=PositionBasis(avg, float(current)) if avg is not None else None,
        episode=kwargs.pop("episode", None),
        lock=kwargs.pop("lock", None),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("current", "price", "avg", "fires"),
    [
        (10, 95.0, 100.0, True),  # exactly -5%
        (10, 95.01, 100.0, False),  # -4.99%
        (10, 19.0, 20.0, True),  # exactly -5% at another scale
        (10, 80.0, 100.0, True),
        (-10, 105.0, 100.0, True),  # short, exactly -5%
        (-10, 104.99, 100.0, False),  # short, -4.99%
        (-10, 95.0, 100.0, False),  # short in profit
        (10, 105.0, 100.0, False),  # long in profit
    ],
)
def test_stop_loss_boundary_is_symmetric(current: int, price: float, avg: float, fires: bool) -> None:
    result = _eval(current, price, avg)

    assert (result.action == "exit") is fires
    if fires:
        assert result.reason == "stop_loss"
        assert result.gate == "POSITION_STOP_LOSS_EXIT"
        assert result.breach is True
        assert result.new_lock is not None
        assert result.new_lock.until == NEXT_OPEN
        assert result.new_lock.side == ("long" if current > 0 else "short")
        assert result.new_lock.entry_price == avg
        assert result.new_lock.trigger_price == price
        assert result.loss_pct == pytest.approx(-abs(price / avg - 1.0))


def test_stop_loss_needs_a_basis_and_a_held_position() -> None:
    assert _eval(10, 50.0, None).action == "none"
    assert _eval(0, 50.0, 100.0, delta=5).action == "none"


@pytest.mark.parametrize("price", [59.0, 141.0])
def test_price_beyond_the_sanity_bound_is_skipped(price: float) -> None:
    result = _eval(10, price, 100.0)

    assert result.action == "none"
    assert result.price_suspect is True


def test_price_suspect_still_allows_the_max_hold_exit() -> None:
    result = _eval(10, 50.0, 100.0, episode=_episode(NOW - timedelta(days=9)))

    assert result.price_suspect is True
    assert result.reason == "max_hold"


def test_stop_loss_wins_when_both_breach() -> None:
    result = _eval(10, 90.0, 100.0, episode=_episode(NOW - timedelta(days=30)))

    assert result.reason == "stop_loss"
    assert result.sessions_held == 5


def test_max_hold_fires_after_five_session_closes() -> None:
    # Opened Monday 2026-11-16 10:00 ET; closes Mon..Fri.
    opened = _et(2026, 11, 16, 10, 0)
    before = _eval(10, 100.0, None, episode=_episode(opened), now=_et(2026, 11, 20, 15, 59))
    after = _eval(10, 100.0, None, episode=_episode(opened), now=_et(2026, 11, 23, 9, 35))

    assert before.action == "none"
    assert before.sessions_held == 4
    assert after.action == "exit"
    assert after.reason == "max_hold"
    assert after.gate == "POSITION_MAX_HOLD_EXIT"
    assert after.sessions_held == 5
    assert after.new_lock is not None
    assert after.new_lock.entry_price is None
    assert after.new_lock.until == _et(2026, 11, 24, 9, 30)


def test_max_hold_reads_the_module_constant_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nbp, "POSITION_MAX_HOLD_SESSIONS", 1)
    opened = _et(2026, 11, 17, 10, 0)

    result = _eval(10, 100.0, None, episode=_episode(opened), now=_et(2026, 11, 18, 9, 35))

    assert result.reason == "max_hold"


# ---- NYSE sessions --------------------------------------------------------


def test_session_count_skips_the_weekend() -> None:
    opened = _et(2026, 11, 13, 10, 0)  # Friday

    assert sessions_held(opened, _et(2026, 11, 13, 15, 59)) == 0
    assert sessions_held(opened, _et(2026, 11, 13, 16, 0)) == 1
    assert sessions_held(opened, _et(2026, 11, 16, 15, 0)) == 1  # Monday, before close
    assert sessions_held(opened, _et(2026, 11, 16, 16, 30)) == 2


def test_session_count_handles_holiday_and_early_close() -> None:
    opened = _et(2026, 11, 25, 15, 0)  # Wednesday before Thanksgiving

    assert sessions_held(opened, _et(2026, 11, 26, 17, 0)) == 1  # Thanksgiving: no session
    assert sessions_held(opened, _et(2026, 11, 27, 12, 59)) == 1
    assert sessions_held(opened, _et(2026, 11, 27, 13, 0)) == 2  # 13:00 early close


def test_session_count_stops_at_the_limit_and_ignores_reverse_time() -> None:
    opened = _et(2025, 1, 6, 10, 0)

    assert sessions_held(opened, _et(2026, 11, 18, 10, 0), limit=5) == 5
    assert sessions_held(NOW, NOW - timedelta(hours=1)) == 0


def test_next_session_open_skips_weekend_and_holiday() -> None:
    assert next_session_open(_et(2026, 11, 25, 14, 0)) == _et(2026, 11, 27, 9, 30)
    assert next_session_open(_et(2026, 11, 20, 10, 0)) == _et(2026, 11, 23, 9, 30)
    assert next_session_open(_et(2026, 11, 18, 8, 0)) == _et(2026, 11, 18, 9, 30)


# ---- episodes -------------------------------------------------------------


def test_first_reconcile_marks_held_positions_first_seen() -> None:
    store = BackstopStore()

    assert store.reconcile({"AAPL": 10, "MSFT": 0}, NOW, previously_seen=None) is True
    assert store.episodes == {"AAPL": Episode(NOW, "first_seen", "long")}


def test_episode_rules_open_add_trim_flip_and_flat() -> None:
    store = BackstopStore()
    store.reconcile({}, NOW, previously_seen=None)
    t1, t2, t3, t4 = (NOW + timedelta(minutes=5 * i) for i in range(1, 5))

    store.reconcile({"AAPL": 10}, t1, previously_seen=set())
    assert store.episodes["AAPL"] == Episode(t1, "observed", "long")
    assert store.reconcile({"AAPL": 25}, t2, previously_seen={"AAPL"}) is False  # add
    assert store.reconcile({"AAPL": 5}, t3, previously_seen={"AAPL"}) is False  # trim
    assert store.episodes["AAPL"].opened_at == t1
    assert store.reconcile({"AAPL": -5}, t4, previously_seen={"AAPL"}) is True  # flip
    assert store.episodes["AAPL"] == Episode(t4, "observed", "short")
    assert store.reconcile({"AAPL": 0}, t4, previously_seen={"AAPL"}) is True  # flat
    assert store.episodes == {}


# ---- the store ------------------------------------------------------------


def _backstop(
    tmp_path: Path,
    *,
    positions: dict[str, Any] | None = None,
    bases: dict[str, float] | None = None,
    unarmed: dict[str, str] | None = None,
    now: datetime = NOW,
    cap: Any = None,
    pending: tuple[str, ...] = (),
    backstop: PositionBackstop | None = None,
) -> PositionBackstop:
    backstop = backstop or PositionBackstop(tmp_path / "runtime" / "position_backstop_state.json", LOGGER)
    held = dict(positions or {})
    backstop.begin_cycle(
        now=now,
        positions=held,
        bases={sym: PositionBasis(avg, float(held.get(sym, 0))) for sym, avg in (bases or {}).items()},
        unarmed=unarmed or {},
        cap_limit=cap,
        pending_symbols=pending,
    )
    return backstop


def test_lock_until_next_open_survives_a_reload_and_then_expires(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    backstop = _backstop(tmp_path, positions={"AAPL": 10}, bases={"AAPL": 100.0})
    step = backstop.evaluate_symbol(symbol="AAPL", current_shares=10, delta_shares=0, price=94.0, bar_ts=NOW)
    assert step.action == "exit"
    payload = json.loads(backstop.path.read_text())
    assert payload["locks"]["AAPL"]["until"] == NEXT_OPEN.isoformat()
    assert payload["locks"]["AAPL"]["reason"] == "stop_loss"
    assert payload["episodes"]["AAPL"]["opened_at_source"] == "first_seen"

    # A restart (new process object) keeps the lock: flat now, so re-entry is blocked.
    reloaded = _backstop(tmp_path, positions={}, now=NOW + timedelta(hours=2))
    blocked = reloaded.evaluate_symbol(symbol="AAPL", current_shares=0, delta_shares=5, price=95.0, bar_ts=NOW)
    assert blocked.action == "reentry_block"
    assert blocked.gate == "POSITION_BACKSTOP_REENTRY_BLOCK"
    idle = reloaded.evaluate_symbol(symbol="AAPL", current_shares=0, delta_shares=0, price=95.0, bar_ts=NOW)
    assert idle.action == "none"

    with caplog.at_level(logging.INFO):
        _backstop(tmp_path, positions={}, now=NEXT_OPEN, backstop=reloaded)
    assert "AAPL" not in reloaded.store.locks
    assert json.loads(reloaded.path.read_text())["locks"] == {}
    (expired,) = _messages(caplog, "POSITION_BACKSTOP_LOCK_EXPIRED")
    assert expired.levelno == logging.INFO
    assert expired.still_held is False
    allowed = reloaded.evaluate_symbol(symbol="AAPL", current_shares=0, delta_shares=5, price=95.0, bar_ts=NOW)
    assert allowed.action == "none"


def test_expired_lock_on_a_still_held_position_is_evaluated_again(tmp_path: Path) -> None:
    backstop = _backstop(tmp_path, positions={"AAPL": 10}, bases={"AAPL": 100.0})
    backstop.evaluate_symbol(symbol="AAPL", current_shares=10, delta_shares=0, price=94.0, bar_ts=NOW)

    _backstop(tmp_path, positions={"AAPL": 10}, bases={"AAPL": 100.0}, now=NEXT_OPEN, backstop=backstop)
    recovered = backstop.evaluate_symbol(
        symbol="AAPL", current_shares=10, delta_shares=0, price=99.0, bar_ts=NEXT_OPEN
    )
    again = backstop.evaluate_symbol(symbol="AAPL", current_shares=10, delta_shares=0, price=94.0, bar_ts=NEXT_OPEN)

    assert recovered.action == "none"
    assert again.action == "exit"
    assert backstop.store.locks["AAPL"].until == _et(2026, 11, 20, 9, 30)


@pytest.mark.parametrize("content", ["{not json", "[]", '{"episodes": []}'])
def test_corrupt_state_file_falls_back_to_an_empty_store(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, content: str
) -> None:
    path = tmp_path / "runtime" / "position_backstop_state.json"
    path.parent.mkdir(parents=True)
    path.write_text(content)

    with caplog.at_level(logging.INFO):
        backstop = PositionBackstop(path, LOGGER)

    assert backstop.store == BackstopStore()
    (failed,) = _messages(caplog, "POSITION_BACKSTOP_STATE_LOAD_FAILED")
    assert failed.levelno == logging.WARNING
    assert failed.reason == "corrupt"


def test_missing_state_file_falls_back_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        backstop = PositionBackstop(tmp_path / "absent.json", LOGGER)

    assert backstop.store == BackstopStore()
    assert _messages(caplog, "POSITION_BACKSTOP_STATE_LOAD_FAILED")[0].reason == "missing"


def test_write_failure_does_not_break_decisions(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(nbp, "atomic_write_text", _boom)
    with caplog.at_level(logging.INFO):
        backstop = _backstop(tmp_path, positions={"AAPL": 10}, bases={"AAPL": 100.0})
        step = backstop.evaluate_symbol(symbol="AAPL", current_shares=10, delta_shares=0, price=94.0, bar_ts=NOW)

    assert step.action == "exit"
    assert "AAPL" in backstop.store.locks
    failures = _messages(caplog, "POSITION_BACKSTOP_STATE_WRITE_FAILED")
    assert failures and all(record.levelno == logging.WARNING for record in failures)
    assert not backstop.path.exists()


def test_writes_only_when_the_store_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[str] = []
    real = nbp.atomic_write_text

    def _counting(path: Any, content: str, **kwargs: Any) -> Any:
        writes.append(content)
        return real(path, content, **kwargs)

    monkeypatch.setattr(nbp, "atomic_write_text", _counting)
    backstop = _backstop(tmp_path, positions={"AAPL": 10})
    for minutes in (5, 10, 15):
        _backstop(tmp_path, positions={"AAPL": 12}, now=NOW + timedelta(minutes=minutes), backstop=backstop)

    assert len(writes) == 1


# ---- logging --------------------------------------------------------------


def test_unarmed_and_price_suspect_log_once_per_symbol_per_session(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        backstop = _backstop(
            tmp_path,
            positions={"AAPL": 10, "MSFT": 5, "NVDA": 3},
            bases={"NVDA": 100.0},
            unarmed={"AAPL": "no_basis", "MSFT": "qty_mismatch", "TSLA": "no_basis"},
        )
        for minutes in (5, 10):
            _backstop(
                tmp_path,
                positions={"AAPL": 10, "MSFT": 5, "NVDA": 3},
                bases={"NVDA": 100.0},
                unarmed={"AAPL": "no_basis", "MSFT": "qty_mismatch"},
                now=NOW + timedelta(minutes=minutes),
                backstop=backstop,
            )
            backstop.evaluate_symbol(symbol="NVDA", current_shares=3, delta_shares=0, price=10.0, bar_ts=NOW)
        _backstop(
            tmp_path,
            positions={"AAPL": 10},
            unarmed={"AAPL": "stale_snapshot"},
            now=NEXT_OPEN,
            backstop=backstop,
        )

    unarmed = [(r.symbol, r.reason) for r in _messages(caplog, "POSITION_BACKSTOP_UNARMED")]
    assert unarmed == [("AAPL", "no_basis"), ("MSFT", "qty_mismatch"), ("AAPL", "stale_snapshot")]
    assert all(r.levelno == logging.INFO for r in _messages(caplog, "POSITION_BACKSTOP_UNARMED"))
    (suspect,) = _messages(caplog, "POSITION_BACKSTOP_PRICE_SUSPECT")
    assert suspect.levelno == logging.WARNING
    assert suspect.symbol == "NVDA"
    assert "NVDA" not in backstop.store.locks


def test_armed_marker_logs_once_per_basis_and_again_on_reopen_or_change(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def armed() -> list[logging.LogRecord]:
        return _messages(caplog, "POSITION_BACKSTOP_ARMED")

    def unarmed() -> list[logging.LogRecord]:
        return _messages(caplog, "POSITION_BACKSTOP_UNARMED")

    with caplog.at_level(logging.INFO):
        backstop = _backstop(tmp_path, positions={"AAPL": 10, "TSLA": -4}, bases={"AAPL": 100.0, "TSLA": 200.0})
        assert [r.symbol for r in armed()] == ["AAPL", "TSLA"]
        first, short = armed()
        assert first.levelno == logging.INFO
        assert first.getMessage() == "POSITION_BACKSTOP_ARMED | symbol=AAPL"
        assert (first.side, first.qty, first.avg_entry) == ("long", 10.0, 100.0)
        assert first.stop_price == pytest.approx(95.0)
        assert short.side == "short"
        assert short.stop_price == pytest.approx(210.0)
        # Following cycles do not repeat it.
        for minutes in (1, 2):
            _backstop(
                tmp_path,
                positions={"AAPL": 10, "TSLA": -4},
                bases={"AAPL": 100.0, "TSLA": 200.0},
                now=NOW + timedelta(minutes=minutes),
                backstop=backstop,
            )
        assert len(armed()) == 2
        # Close and reopen: armed again for that symbol only.
        _backstop(
            tmp_path,
            positions={"TSLA": -4},
            bases={"TSLA": 200.0},
            now=NOW + timedelta(minutes=3),
            backstop=backstop,
        )
        assert len(armed()) == 2
        _backstop(
            tmp_path,
            positions={"AAPL": 10, "TSLA": -4},
            bases={"AAPL": 101.0, "TSLA": 200.0},
            now=NOW + timedelta(minutes=4),
            backstop=backstop,
        )
        assert [r.symbol for r in armed()] == ["AAPL", "TSLA", "AAPL"]
        assert armed()[-1].avg_entry == 101.0
        # An unarmed spell with an unchanged basis logs UNARMED once and no new ARMED.
        for minutes, spell in ((5, True), (6, False), (7, True), (8, False)):
            _backstop(
                tmp_path,
                positions={"AAPL": 10, "TSLA": -4},
                bases={"TSLA": 200.0} if spell else {"AAPL": 101.0, "TSLA": 200.0},
                unarmed={"AAPL": "stale_snapshot"} if spell else None,
                now=NOW + timedelta(minutes=minutes),
                backstop=backstop,
            )
        assert [r.reason for r in unarmed()] == ["stale_snapshot"]
        assert len(armed()) == 3
        # A scale-in changes quantity and entry price: armed again with the new stop.
        _backstop(
            tmp_path,
            positions={"AAPL": 20, "TSLA": -4},
            bases={"AAPL": 110.0, "TSLA": 200.0},
            now=NOW + timedelta(minutes=9),
            backstop=backstop,
        )
        assert [r.symbol for r in armed()] == ["AAPL", "TSLA", "AAPL", "AAPL"]
        assert (armed()[-1].qty, armed()[-1].stop_price) == (20.0, pytest.approx(104.5))
        # A side flip between two cycles logs the short stop.
        _backstop(
            tmp_path,
            positions={"AAPL": -20, "TSLA": -4},
            bases={"AAPL": 120.0, "TSLA": 200.0},
            now=NOW + timedelta(minutes=10),
            backstop=backstop,
        )
        flipped = armed()[-1]
        assert len(armed()) == 5
        assert (flipped.side, flipped.stop_price, flipped.threshold) == ("short", pytest.approx(126.0), -0.05)
        assert flipped.locked is False


def test_breach_markers_carry_the_context(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        backstop = _backstop(tmp_path, positions={"AAPL": -10}, bases={"AAPL": 100.0})
        backstop.evaluate_symbol(symbol="AAPL", current_shares=-10, delta_shares=0, price=106.0, bar_ts=NOW)

    (breach,) = _messages(caplog, "POSITION_STOP_LOSS_BREACH")
    assert breach.levelno == logging.WARNING
    assert (breach.symbol, breach.side, breach.qty) == ("AAPL", "short", -10)
    assert (breach.avg_entry, breach.price) == (100.0, 106.0)
    assert breach.loss_pct == pytest.approx(-0.06)
    assert breach.threshold == -0.05
    assert breach.lock_until == NEXT_OPEN.isoformat()
    assert breach.source == "first_seen"
    assert breach.opened_at == NOW.isoformat()


def test_max_hold_marker(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    backstop = _backstop(tmp_path, positions={"AAPL": 10}, now=_et(2026, 11, 16, 10, 0))
    with caplog.at_level(logging.INFO):
        _backstop(tmp_path, positions={"AAPL": 10}, now=_et(2026, 11, 23, 9, 35), backstop=backstop)
        step = backstop.evaluate_symbol(symbol="AAPL", current_shares=10, delta_shares=0, price=1.0, bar_ts=NOW)

    assert step.gate == "POSITION_MAX_HOLD_EXIT"
    (breach,) = _messages(caplog, "POSITION_MAX_HOLD_BREACH")
    assert breach.levelno == logging.WARNING
    assert breach.sessions_held == 5
    assert breach.threshold == 5
    assert breach.avg_entry is None


def test_exit_stalled_logs_once_per_lock_after_two_more_bars(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    backstop = _backstop(tmp_path, positions={"AAPL": 10}, bases={"AAPL": 100.0})
    bars = [NOW + timedelta(minutes=5 * i) for i in range(5)]
    with caplog.at_level(logging.INFO):
        backstop.evaluate_symbol(symbol="AAPL", current_shares=10, delta_shares=0, price=94.0, bar_ts=bars[0])
        backstop.note_decision("AAPL", ["POSITION_STOP_LOSS_EXIT", "KILL_SWITCH_BLOCK"])
        backstop.evaluate_symbol(symbol="AAPL", current_shares=10, delta_shares=0, price=94.0, bar_ts=bars[1])
        backstop.evaluate_symbol(symbol="AAPL", current_shares=10, delta_shares=0, price=94.0, bar_ts=bars[1])
        assert not _messages(caplog, "POSITION_BACKSTOP_EXIT_STALLED")
        for bar in bars[2:]:
            backstop.evaluate_symbol(symbol="AAPL", current_shares=10, delta_shares=0, price=94.0, bar_ts=bar)

    (stalled,) = _messages(caplog, "POSITION_BACKSTOP_EXIT_STALLED")
    assert stalled.levelno == logging.WARNING
    assert stalled.bars_since_arming == 2
    assert stalled.last_gates == ["POSITION_STOP_LOSS_EXIT", "KILL_SWITCH_BLOCK"]


# ---- entry basis from the broker-sync snapshot -----------------------------


def _sync(*positions: Any, age_s: float = 5.0, positions_fresh: bool = True) -> BrokerSyncResult:
    return BrokerSyncResult(
        open_orders=(),
        positions=tuple(positions),
        open_buy_by_symbol={},
        open_sell_by_symbol={},
        timestamp=float(monotonic_time()) - age_s,
        positions_fresh=positions_fresh,
    )


def _runtime(snapshot: Any) -> Any:
    return SimpleNamespace(execution_engine=SimpleNamespace(_broker_sync=snapshot))


def test_entry_basis_from_a_fresh_matching_snapshot() -> None:
    snapshot = _sync(
        SimpleNamespace(symbol="AAPL", qty="10", side="long", avg_entry_price="101.5"),
        {"symbol": "NVDA", "qty": "-4", "side": SimpleNamespace(value="short"), "avg_entry_price": 50.0},
        SimpleNamespace(symbol="MSFT", qty="7", side="long", avg_entry_price="400"),
        SimpleNamespace(symbol="AMD", qty="3", side="long", avg_entry_price="0"),
        SimpleNamespace(symbol="TSLA", qty="2", side="long", avg_entry_price="nan"),
    )

    bases, unarmed = bot_engine._position_entry_basis(
        _runtime(snapshot), {"AAPL": 10, "NVDA": -4, "MSFT": 5, "AMD": 3, "TSLA": 2, "SPY": 1, "QQQ": 0}
    )

    assert bases == {"AAPL": PositionBasis(101.5, 10.0), "NVDA": PositionBasis(50.0, -4.0)}
    assert unarmed == {"MSFT": "qty_mismatch", "AMD": "no_basis", "TSLA": "no_basis", "SPY": "no_basis"}


@pytest.mark.parametrize(
    "snapshot",
    [
        _sync(SimpleNamespace(symbol="AAPL", qty="10", side="long", avg_entry_price="100"), age_s=601.0),
        _sync(
            SimpleNamespace(symbol="AAPL", qty="10", side="long", avg_entry_price="100"),
            positions_fresh=False,
        ),
        SimpleNamespace(positions=(), positions_fresh=True, timestamp=None),
    ],
)
def test_entry_basis_from_a_stale_snapshot_is_unarmed(snapshot: Any) -> None:
    assert bot_engine._position_entry_basis(_runtime(snapshot), {"AAPL": 10}) == (
        {},
        {"AAPL": "stale_snapshot"},
    )


def test_entry_basis_without_a_snapshot_is_unarmed() -> None:
    assert bot_engine._position_entry_basis(SimpleNamespace(), {"AAPL": 10}) == ({}, {"AAPL": "no_basis"})
    assert bot_engine._position_entry_basis(_runtime(None), {}) == ({}, {})


# ---- process_netting_symbol --------------------------------------------------


def _processor(
    tmp_path: Path,
    *,
    positions: dict[str, int],
    avg_entry: float | None = 11.0,
    unarmed: dict[str, str] | None = None,
    cap: Any = None,
    pending: tuple[str, ...] = (),
    **overrides: Any,
) -> tuple[Any, list[dict[str, Any]], list[int]]:
    processor, records = _make_processor(positions=dict(positions), **overrides)
    processor.now = NOW
    processor.logger = LOGGER
    bases = {"AAPL": avg_entry} if avg_entry is not None and positions.get("AAPL") else {}
    processor.position_backstop = _backstop(
        tmp_path, positions=positions, bases=bases, unarmed=unarmed, cap=cap, pending=pending
    )
    submitted: list[int] = []
    real_submit = processor.execute_submission_func

    def _submit(**kwargs: Any) -> Any:
        submitted.append(int(kwargs["delta_shares"]))
        return real_submit(**kwargs)

    processor.execute_submission_func = _submit
    return processor, records, submitted


def _process(processor: Any, *, target_dollars: float, symbol: str = "AAPL") -> Any:
    net_target = _make_net_target(bar_ts=NOW, target_dollars=target_dollars)
    net_target.symbol = symbol
    for proposal in net_target.proposals:  # fields the real approval step reads
        proposal.expected_edge_bps = 1.0
        proposal.expected_cost_bps = 4.0
        proposal.target_dollars = target_dollars
    return process_netting_symbol(
        processor=processor, symbol=symbol, net_target=net_target, orders_submitted=0
    )


def test_breach_exits_through_the_normal_submission(tmp_path: Path) -> None:
    # Price 10 vs entry 11 (-9.1%); the sleeves want to add, the backstop exits.
    processor, records, submitted = _processor(tmp_path, positions={"AAPL": 10})

    result = _process(processor, target_dollars=200.0)

    assert result.submitted_increment == 1
    assert submitted == [-10]
    assert records[0]["gates"] == ["POSITION_STOP_LOSS_EXIT", "OK_TRADE"]
    assert records[0]["net_target"].target_shares == 0
    snapshot = records[0]["config_snapshot"]["position_backstop"]
    assert snapshot["reason"] == "stop_loss"
    assert snapshot["lock_until"] == NEXT_OPEN.isoformat()
    assert "AAPL" in processor.position_backstop.store.locks


def test_short_breach_buys_to_cover(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Do not depend on the ambient launch profile, which other tests in the
    # full run can leave short-disabled (long-only would flatten the short first).
    monkeypatch.setattr(netting_symbol_cycle, "_short_openings_allowed", lambda _cfg: True)
    processor, records, submitted = _processor(tmp_path, positions={"AAPL": -10}, avg_entry=9.0)

    _process(processor, target_dollars=-100.0)

    assert submitted == [10]
    assert records[0]["gates"][0] == "POSITION_STOP_LOSS_EXIT"


def test_no_breach_leaves_the_decision_unchanged(tmp_path: Path) -> None:
    for positions, target in (({"AAPL": 10}, 0.0), ({"AAPL": 10}, 200.0), ({}, 100.0)):
        processor, records, _ = _processor(tmp_path / str(len(positions)) / str(target), positions=positions, avg_entry=10.2)
        plain, plain_records = _make_processor(positions=dict(positions))
        plain.now = NOW

        assert _process(processor, target_dollars=target) == _process(plain, target_dollars=target)
        assert records[0]["gates"] == plain_records[0]["gates"]
        assert records[0]["config_snapshot"] == plain_records[0]["config_snapshot"]


def test_lock_blocks_reentry_until_the_next_open(tmp_path: Path) -> None:
    processor, _records, _ = _processor(tmp_path, positions={"AAPL": 10})
    _process(processor, target_dollars=0.0)

    flat, flat_records, flat_submitted = _processor(tmp_path, positions={})
    flat.position_backstop = _backstop(tmp_path, positions={}, now=NOW + timedelta(minutes=5))
    _process(flat, target_dollars=100.0)

    assert flat_submitted == []
    assert flat_records[0]["gates"] == ["POSITION_BACKSTOP_REENTRY_BLOCK"]
    assert flat_records[0]["config_snapshot"]["position_backstop"]["action"] == "reentry_block"

    nextday, next_records, next_submitted = _processor(tmp_path, positions={})
    nextday.position_backstop = _backstop(tmp_path, positions={}, now=NEXT_OPEN + timedelta(minutes=5))
    _process(nextday, target_dollars=100.0)

    assert next_submitted == [10]
    assert next_records[0]["gates"] == ["OK_TRADE"]


def test_lock_keeps_forcing_the_exit_while_held(tmp_path: Path) -> None:
    processor, _records, _ = _processor(tmp_path, positions={"AAPL": 10}, kill_switch=True)
    _process(processor, target_dollars=0.0)

    # Price recovered above the stop, entry basis gone: the lock still exits.
    later, records, submitted = _processor(tmp_path, positions={"AAPL": 10}, avg_entry=None)
    later.position_backstop = _backstop(tmp_path, positions={"AAPL": 10}, now=NOW + timedelta(minutes=5))
    later.latest_price = {"AAPL": 12.0}
    _process(later, target_dollars=120.0)

    assert submitted == [-10]
    assert records[0]["gates"] == ["POSITION_STOP_LOSS_EXIT", "OK_TRADE"]
    assert records[0]["config_snapshot"]["position_backstop"]["breach"] is False


def test_kill_switch_still_blocks_the_exit_and_the_lock_persists(tmp_path: Path) -> None:
    processor, records, submitted = _processor(tmp_path, positions={"AAPL": 10}, kill_switch=True)

    _process(processor, target_dollars=0.0)

    assert submitted == []
    assert records[0]["gates"] == ["POSITION_STOP_LOSS_EXIT", "KILL_SWITCH_BLOCK"]
    saved = json.loads(processor.position_backstop.path.read_text())
    assert saved["locks"]["AAPL"]["reason"] == "stop_loss"


def test_open_drawdown_breaker_lets_the_exit_through(tmp_path: Path) -> None:
    processor, records, submitted = _processor(
        tmp_path, positions={"AAPL": 10}, drawdown_breaker=dict(OPEN_BREAKER)
    )

    _process(processor, target_dollars=200.0)

    assert submitted == [-10]
    assert records[0]["gates"] == ["POSITION_STOP_LOSS_EXIT", "OK_TRADE"]


def test_halt_or_missing_price_takes_no_action(tmp_path: Path) -> None:
    halted, halted_records, halted_submitted = _processor(tmp_path / "halt", positions={"AAPL": 10})
    halted.state.halt_trading = True
    halted.state.halt_reason = "HALT_TRADING"
    no_price, price_records, price_submitted = _processor(tmp_path / "price", positions={"AAPL": 10})
    no_price.latest_price = {}

    _process(halted, target_dollars=0.0)
    _process(no_price, target_dollars=0.0)

    assert halted_submitted == price_submitted == []
    assert halted_records[0]["gates"] == ["HALT_TRADING"]
    assert "POSITION_STOP_LOSS_EXIT" not in price_records[0]["gates"]
    assert halted.position_backstop.store.locks == {}
    assert no_price.position_backstop.store.locks == {}


def test_unarmed_position_is_not_exited(tmp_path: Path) -> None:
    processor, records, submitted = _processor(
        tmp_path, positions={"AAPL": 10}, avg_entry=None, unarmed={"AAPL": "no_basis"}
    )

    _process(processor, target_dollars=100.0)

    assert submitted == []  # holding 10 at $10 is the target: nothing to do
    assert "position_backstop" not in records[0]["config_snapshot"]
    assert processor.position_backstop.store.locks == {}


# ---- gates a protective exit passes, and normal exits still meet ------------


class _Quarantine:
    def is_quarantined(self, **_kwargs: Any) -> tuple[bool, str]:
        return True, "SYMBOL_QUARANTINED"


def _approval_policy() -> Any:
    return SimpleNamespace(
        calibration=SimpleNamespace(
            max_ece_normal=1.0, max_brier_normal=1.0, max_ece_stress=1.0, max_brier_stress=1.0, min_samples=10
        ),
        objective=SimpleNamespace(fee_bps=0.0, borrow_bps=0.0),
    )


def _blocking_approval(gate: str) -> Any:
    return lambda policy, candidate: ExecutionApproval(False, 0, -3.0, (gate,))


_BYPASS_CASES: dict[str, dict[str, Any]] = {
    "policy_ablation": {
        "overrides": {
            "policy_disabled_sleeves": {"main"},
            "prepare_symbol_prelude_func": prepare_netting_symbol_prelude,
        },
        "gate": "POLICY_ABLATION_SLEEVE_BLOCK",
    },
    "quarantine_block": {
        "overrides": {
            "quarantine_enabled": True,
            "quarantine_manager": _Quarantine(),
            "quarantine_apply_symbol": True,
            "quarantine_mode": "block",
        },
        "gate": "SYMBOL_QUARANTINED",
    },
    "event_blackout": {
        "overrides": {"event_blackout_enabled": True, "is_near_event_func": lambda *a, **k: True},
        "gate": "EVENT_RISK_BLACKOUT_BLOCK",
    },
    "adaptive_sizing": {
        "overrides": {
            "symbol_adaptive_profiles": {"AAPL": {"scale": 0.05}},
            "apply_symbol_adjustments_func": apply_symbol_adjustments,
        },
        "gate": "SYMBOL_EXPECTANCY_SLIPPAGE_BLOCK",
    },
    "capacity_throttle": {
        "overrides": {
            "capacity_throttle_enabled": True,
            "capacity_spread_soft_bps": 1.0,
            "capacity_spread_hard_bps": 2.0,
            "capacity_min_scale": 0.01,
            "effective_policy": _approval_policy(),
            "evaluate_execution_approval_func": evaluate_execution_approval,
            "approve_execution_candidate_func": lambda policy, candidate: ExecutionApproval(
                True, int(candidate.proposed_delta_shares), 1.0, ()
            ),
            "prepare_symbol_approval_func": prepare_netting_symbol_approval,
        },
        "gate": "CAPACITY_THROTTLE_BLOCK",
    },
}
for _gate in (
    "NET_EDGE_FLOOR_BLOCK",
    "SPREAD_TOO_WIDE_BLOCK",
    "CALIBRATION_BLOCK",
    "ORDER_PACING_CAP_BLOCK",
    "STALE_ORDER_BLOCK",
):
    _BYPASS_CASES[_gate.lower()] = {
        "overrides": {
            "effective_policy": _approval_policy(),
            "evaluate_execution_approval_func": evaluate_execution_approval,
            "approve_execution_candidate_func": _blocking_approval(_gate),
            "prepare_symbol_approval_func": prepare_netting_symbol_approval,
        },
        "gate": _gate,
    }


_OVERRIDE_MARKERS = {"PROTECTIVE_EXIT_OVERRIDE", "CLOSING_EXIT_OVERRIDE"}


def _entry_processor(tmp_path: Path, overrides: dict[str, Any]) -> tuple[Any, list[dict[str, Any]], list[int]]:
    # The test proposals carry 1 bps of edge against 4 bps of cost; keep the
    # cost-aware entry guard out of the way so the gate under test decides.
    return _processor(
        tmp_path, positions={}, ineffective_gate_blocklist={"COST_AWARE_ENTRY_GUARD"}, **overrides
    )


@pytest.mark.parametrize("case", sorted(_BYPASS_CASES))
def test_bypassed_gate_passes_protective_and_normal_exits_and_still_holds_an_entry(
    tmp_path: Path, case: str
) -> None:
    overrides = _BYPASS_CASES[case]["overrides"]
    gate = _BYPASS_CASES[case]["gate"]
    # A one-share position: adaptive and capacity scaling would round its exit to zero.
    normal, normal_records, normal_submitted = _processor(
        tmp_path / "normal", positions={"AAPL": 1}, avg_entry=10.2, **overrides
    )
    protective, records, submitted = _processor(tmp_path / "protective", positions={"AAPL": 1}, **overrides)
    entry, entry_records, entry_submitted = _entry_processor(tmp_path / "entry", overrides)

    _process(normal, target_dollars=0.0)
    _process(protective, target_dollars=0.0)
    entry_result = _process(entry, target_dollars=10.0)  # one share at $10

    # Normal exit (was: blocked exactly as without the backstop): now a pure
    # reduction goes out at full size, original gate name kept, own marker once.
    assert normal_submitted == [-1]
    normal_gates = normal_records[0]["gates"]
    assert gate in normal_gates
    assert normal_gates.count("CLOSING_EXIT_OVERRIDE") == 1
    assert "PROTECTIVE_EXIT_OVERRIDE" not in normal_gates
    assert normal_gates[-1] == "OK_TRADE"
    # Protective exit: unchanged.
    assert submitted == [-1]
    gates = records[0]["gates"]
    assert gates[0] == "POSITION_STOP_LOSS_EXIT"
    assert gate in gates
    assert gates.count("PROTECTIVE_EXIT_OVERRIDE") == 1
    assert "CLOSING_EXIT_OVERRIDE" not in gates
    assert gates[-1] == "OK_TRADE"
    # Entry: still held by the gate, no override.
    assert entry_submitted == []
    assert entry_result.submitted_increment == 0
    assert entry_records[0]["gates"][-1] == gate
    assert not _OVERRIDE_MARKERS & set(entry_records[0]["gates"])


@pytest.mark.parametrize("case", sorted(_BYPASS_CASES))
def test_flip_goes_out_as_its_exit_leg_and_the_entry_leg_stays_gated(tmp_path: Path, case: str) -> None:
    overrides = _BYPASS_CASES[case]["overrides"]
    gate = _BYPASS_CASES[case]["gate"]
    # Short one share, the sleeves want long 10: the reduction to flat passes the
    # gate; the long entry is not part of this order.
    flip, flip_records, flip_submitted = _processor(
        tmp_path / "flip", positions={"AAPL": -1}, avg_entry=10.2, **overrides
    )

    _process(flip, target_dollars=100.0)

    assert flip_submitted == [1]
    flip_gates = flip_records[0]["gates"]
    assert flip_gates[0] == "FLAT_BEFORE_REVERSAL"
    assert gate in flip_gates
    assert flip_gates.count("CLOSING_EXIT_OVERRIDE") == 1
    assert flip_records[0]["net_target"].target_shares == 0
    # The entry leg, from flat on a later bar, meets the gate exactly as any entry.
    entry, entry_records, entry_submitted = _entry_processor(tmp_path / "entry", overrides)
    _process(entry, target_dollars=100.0)
    assert entry_submitted == []
    assert entry_records[0]["gates"][-1] == gate
    assert not _OVERRIDE_MARKERS & set(entry_records[0]["gates"])


@pytest.mark.parametrize(
    "case", ["spread_too_wide_block", "event_blackout", "capacity_throttle", "net_edge_floor_block"]
)
def test_partial_trim_is_gated_like_an_entry_and_the_full_close_passes(tmp_path: Path, case: str) -> None:
    overrides = _BYPASS_CASES[case]["overrides"]
    gate = _BYPASS_CASES[case]["gate"]
    # Long 10 at $10: trimming to 5 leaves a position, so it meets the gate as
    # on main; closing to flat passes it.
    trim, trim_records, trim_submitted = _processor(
        tmp_path / "trim", positions={"AAPL": 10}, avg_entry=10.2, **overrides
    )
    close, close_records, close_submitted = _processor(
        tmp_path / "close", positions={"AAPL": 10}, avg_entry=10.2, **overrides
    )

    trim_result = _process(trim, target_dollars=50.0)
    _process(close, target_dollars=0.0)

    assert trim_submitted == []
    assert trim_result.submitted_increment == 0
    assert trim_records[0]["gates"][-1] == gate
    assert not _OVERRIDE_MARKERS & set(trim_records[0]["gates"])
    assert close_submitted == [-10]
    assert gate in close_records[0]["gates"]
    assert close_records[0]["gates"].count("CLOSING_EXIT_OVERRIDE") == 1
    assert close_records[0]["gates"][-1] == "OK_TRADE"


def test_partial_trim_is_scaled_by_a_soft_throttle_and_the_full_close_is_not(tmp_path: Path) -> None:
    trim, trim_records, trim_submitted = _processor(
        tmp_path / "trim",
        positions={"AAPL": 100},
        avg_entry=10.2,
        **_real_approval(_throttle_policy(factor_hard_ratio=0.99)),
    )
    close, close_records, close_submitted = _processor(
        tmp_path / "close", positions={"AAPL": 100}, avg_entry=10.2, **_real_approval(_throttle_policy())
    )

    _process(trim, target_dollars=500.0)  # 100 -> 50
    _process(close, target_dollars=0.0)

    assert trim_submitted == [-18]  # scaled by the factor soft throttle, as on main
    assert trim_records[0]["gates"] == ["RISK_FACTOR_SOFT_THROTTLE", "OK_TRADE"]
    assert close_submitted == [-100]
    assert close_records[0]["gates"].count("CLOSING_EXIT_OVERRIDE") == 1


@pytest.mark.usefixtures("thin_regime")
def test_partial_trim_meets_the_thin_max_order_block_and_the_full_close_is_clipped(tmp_path: Path) -> None:
    trim, trim_records, trim_submitted = _processor(
        tmp_path / "trim", positions={"AAPL": 1000}, avg_entry=10.2, liq_regime_enabled=True
    )
    close, close_records, close_submitted = _processor(
        tmp_path / "close", positions={"AAPL": 1000}, avg_entry=10.2, liq_regime_enabled=True
    )

    _process(trim, target_dollars=2000.0)  # 1000 -> 200: 615 shares after the cost multiplier
    _process(close, target_dollars=0.0)

    assert trim_submitted == []
    assert trim_records[0]["gates"] == ["LIQ_THIN_COST_MULT_SCALE", "LIQ_THIN_MAX_ORDER_BLOCK"]
    assert close_submitted == [-500]
    assert close_records[0]["gates"][-2:] == ["CLOSING_EXIT_OVERRIDE", "OK_TRADE"]


def test_partial_trim_meets_the_zero_volume_block_and_the_full_close_is_capped(
    tmp_path: Path, real_participation_cap: dict[str, Any]
) -> None:
    trim, trim_records, trim_submitted = _processor(
        tmp_path / "trim", positions={"AAPL": 1000}, avg_entry=10.2, **real_participation_cap
    )
    close, close_records, close_submitted = _processor(
        tmp_path / "close", positions={"AAPL": 1000}, avg_entry=10.2, **real_participation_cap
    )
    trim.latest_liquidity = {}
    close.latest_liquidity = {}

    _process(trim, target_dollars=5000.0)  # 1000 -> 500
    _process(close, target_dollars=0.0)

    assert trim_submitted == []
    assert trim_records[0]["gates"] == ["LIQ_PARTICIPATION_BLOCK"]
    assert close_submitted == [-500]
    assert close_records[0]["gates"][:2] == ["LIQ_PARTICIPATION_BLOCK", "CLOSING_EXIT_OVERRIDE"]


@pytest.mark.usefixtures("thin_regime")
def test_a_clipped_full_close_keeps_the_override_and_the_next_bar_closes_the_rest(tmp_path: Path) -> None:
    overrides = _BYPASS_CASES["spread_too_wide_block"]["overrides"]
    # Bar 1: close 1000 -> the thin cap clips it to 500, and the spread block
    # later in the cycle still does not hold what remains.
    first, first_records, first_submitted = _processor(
        tmp_path / "bar1", positions={"AAPL": 1000}, avg_entry=10.2, liq_regime_enabled=True, **overrides
    )
    _process(first, target_dollars=0.0)
    assert first_submitted == [-500]
    assert first_records[0]["gates"][-4:] == [
        "LIQ_THIN_MAX_ORDER_BLOCK",
        "CLOSING_EXIT_OVERRIDE",
        "SPREAD_TOO_WIDE_BLOCK",
        "OK_TRADE",
    ]
    # Bar 2: still targeting flat, the remaining 500 is again a full close.
    second, second_records, second_submitted = _processor(
        tmp_path / "bar2", positions={"AAPL": 500}, avg_entry=10.2, liq_regime_enabled=True, **overrides
    )
    _process(second, target_dollars=0.0)
    assert second_submitted == [-385]  # 500 / 1.3: the thin cost multiplier still slices
    assert second_records[0]["gates"].count("CLOSING_EXIT_OVERRIDE") == 1
    assert second_records[0]["gates"][-1] == "OK_TRADE"


def test_closing_exit_override_logs_a_per_symbol_marker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    overrides = _BYPASS_CASES["spread_too_wide_block"]["overrides"]
    normal, _records, _submitted = _processor(tmp_path / "normal", positions={"AAPL": 1}, avg_entry=10.2, **overrides)
    protective, _p_records, _p_submitted = _processor(tmp_path / "protective", positions={"AAPL": 1}, **overrides)
    entry, _e_records, _e_submitted = _entry_processor(tmp_path / "entry", overrides)

    with caplog.at_level(logging.INFO):
        _process(normal, target_dollars=0.0)
        _process(protective, target_dollars=0.0)
        _process(entry, target_dollars=10.0)

    (marker,) = _messages(caplog, "CLOSING_EXIT_OVERRIDE")
    assert marker.getMessage() == "CLOSING_EXIT_OVERRIDE | symbol=AAPL"
    assert marker.levelno == logging.INFO
    assert (marker.current_shares, marker.delta_shares, marker.submitted) == (1, -1, True)
    assert "SPREAD_TOO_WIDE_BLOCK" in marker.gates


@pytest.mark.parametrize(
    ("current", "delta", "expected"),
    [
        (10, -10, True),
        (-10, 10, True),
        (1, -1, True),
        (10.9, -10.2, True),  # whole shares: 10 -> flat
        (10, -3, False),  # partial trim
        (-10, 4, False),  # partial cover
        (10, -11, False),  # flip
        (10, 5, False),  # add
        (0, -5, False),  # opening
        (0.7, -0.7, False),  # sub-one-share position counts as flat
        (10, 0, False),
    ],
)
def test_full_close_only_takes_the_position_to_flat(current: float, delta: float, expected: bool) -> None:
    assert nbp.is_full_close(int(current), int(delta)) is expected
    gate = nbp.exit_override_gate(protective_exit=False, current_shares=int(current), delta_shares=int(delta))
    assert gate == ("CLOSING_EXIT_OVERRIDE" if expected else None)
    assert (
        nbp.exit_override_gate(protective_exit=True, current_shares=int(current), delta_shares=int(delta))
        == "PROTECTIVE_EXIT_OVERRIDE"
    )


@pytest.mark.parametrize(
    ("current", "delta", "expected"),
    [(10, -4, "CLOSING_EXIT_OVERRIDE"), (-10, 3, "CLOSING_EXIT_OVERRIDE"), (10, -11, None), (10, 2, None), (0, -1, None)],
)
def test_a_clipped_full_close_keeps_the_override_on_what_remains(
    current: int, delta: int, expected: str | None
) -> None:
    # full_close: the intended order was a full close before a liquidity clip;
    # the remainder keeps the override only while it still just reduces.
    assert (
        nbp.exit_override_gate(protective_exit=False, current_shares=current, delta_shares=delta, full_close=True)
        == expected
    )


@pytest.mark.parametrize("avg_entry", [11.0, 10.2], ids=["protective", "normal"])
def test_event_blackout_is_not_raised_again_by_pretrade_for_an_exit(tmp_path: Path, avg_entry: float) -> None:
    seen: list[bool] = []
    processor, records, submitted = _processor(
        tmp_path,
        positions={"AAPL": 10},
        avg_entry=avg_entry,
        event_blackout_enabled=True,
        is_near_event_func=lambda *a, **k: True,
    )
    real_prelude = processor.prepare_submit_prelude_func

    def _prelude(**kwargs: Any) -> Any:
        seen.append(bool(kwargs["event_risk_near"]))
        return real_prelude(**kwargs)

    processor.prepare_submit_prelude_func = _prelude

    _process(processor, target_dollars=0.0)

    assert seen == [False]
    assert submitted == [-10]
    assert records[0]["config_snapshot"]["event_risk_near"] is True


@pytest.mark.parametrize("avg_entry", [11.0, 10.2], ids=["protective", "normal"])
def test_approval_block_outside_the_override_list_still_holds_an_exit(tmp_path: Path, avg_entry: float) -> None:
    processor, records, submitted = _processor(
        tmp_path,
        positions={"AAPL": 10},
        avg_entry=avg_entry,
        effective_policy=_approval_policy(),
        evaluate_execution_approval_func=evaluate_execution_approval,
        approve_execution_candidate_func=lambda policy, candidate: ExecutionApproval(
            False, 0, -3.0, ("NET_EDGE_FLOOR_BLOCK", "RISK_PORTFOLIO_HARD_BLOCK")
        ),
        prepare_symbol_approval_func=prepare_netting_symbol_approval,
    )

    _process(processor, target_dollars=0.0)

    assert submitted == []
    assert not _OVERRIDE_MARKERS & set(records[0]["gates"])
    assert records[0]["metrics"]["terminal_reason"] == "EXECUTION_APPROVAL_BLOCK"


def _exit_prelude_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs = _submit_prelude_kwargs()
    kwargs.update(
        side="sell",
        delta_shares=-10,
        current_shares=10.0,
        opening_trade=False,
        portfolio_optimizer_enabled=True,
        portfolio_optimizer=object(),
        portfolio_optimizer_allows_trade_func=lambda **_k: (False, {"decision": "reject"}),
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("overrides", "gate"),
    [
        ({}, "PORTFOLIO_OPTIMIZER_REJECT"),
        (
            {
                "portfolio_optimizer": None,
                "portfolio_optimizer_context": {"init_failed": True, "init_fail_open": False},
            },
            "PORTFOLIO_OPTIMIZER_INIT_FAILED",
        ),
        (
            {
                "cfg": SimpleNamespace(seed="seed", execution_mode="live"),
                "portfolio_optimizer_allows_trade_func": lambda **_k: (_ for _ in ()).throw(
                    RuntimeError("solver")
                ),
            },
            "PORTFOLIO_OPTIMIZER_DECISION_ERROR",
        ),
    ],
)
def test_portfolio_optimizer_passes_exits_and_holds_an_entry(overrides: dict[str, Any], gate: str) -> None:
    normal = prepare_netting_submit_prelude(**cast(Any, _exit_prelude_kwargs(**overrides)))
    protective = prepare_netting_submit_prelude(
        **cast(Any, _exit_prelude_kwargs(**overrides)), protective_exit=True
    )
    entry = prepare_netting_submit_prelude(
        **cast(
            Any,
            _exit_prelude_kwargs(
                **{"side": "buy", "delta_shares": 10, "current_shares": 0.0, "opening_trade": True, **overrides}
            ),
        )
    )

    # Normal exit (was: blocked): a pure reduction now passes with its own marker.
    assert normal.blocked_reason is None
    assert normal.execution_intent_context is not None
    assert list(normal.gates_added) == [gate, "CLOSING_EXIT_OVERRIDE"]
    assert protective.blocked_reason is None
    assert protective.execution_intent_context is not None
    assert list(protective.gates_added) == [gate, "PROTECTIVE_EXIT_OVERRIDE"]
    assert entry.blocked_reason == gate
    assert entry.execution_intent_context is None
    assert list(entry.gates_added) == [gate]


@pytest.mark.parametrize(
    ("overrides", "gates"),
    [
        ({"enforce_participation_cap_func": lambda **k: (False, 0.0, "LIQ_PARTICIPATION_BLOCK")}, ["LIQ_PARTICIPATION_BLOCK"]),
        ({"state": SimpleNamespace(halt_trading=False, halt_reason=None, last_order_bar_ts={"AAPL": NOW}, degraded_providers=set())}, ["BAR_DEDUP"]),
    ],
)
def test_exits_still_respect_participation_cap_and_bar_dedup(
    tmp_path: Path, overrides: dict[str, Any], gates: list[str]
) -> None:
    processor, records, submitted = _processor(tmp_path / "protective", positions={"AAPL": 10}, **overrides)
    normal, normal_records, normal_submitted = _processor(
        tmp_path / "normal", positions={"AAPL": 10}, avg_entry=10.2, **overrides
    )

    _process(processor, target_dollars=0.0)
    _process(normal, target_dollars=0.0)

    assert submitted == normal_submitted == []
    assert records[0]["gates"] == ["POSITION_STOP_LOSS_EXIT", *gates]
    assert normal_records[0]["gates"] == gates


def test_backstop_gates_cannot_be_auto_disabled(tmp_path: Path) -> None:
    names = {
        "POSITION_STOP_LOSS_EXIT",
        "POSITION_MAX_HOLD_EXIT",
        "POSITION_BACKSTOP_REENTRY_BLOCK",
        "POSITION_CAP_BLOCK",
    }
    assert names <= _NON_DISABLEABLE_GATE_NAMES
    roots = {bot_engine._gate_root_cause(name) for name in names | {"PROTECTIVE_EXIT_OVERRIDE"}}
    assert len(roots) == 5
    assert bot_engine._dedupe_gate_root_causes(
        ["POSITION_STOP_LOSS_EXIT", "PROTECTIVE_EXIT_OVERRIDE", "POSITION_CAP_BLOCK"]
    ) == ["POSITION_STOP_LOSS_EXIT", "PROTECTIVE_EXIT_OVERRIDE", "POSITION_CAP_BLOCK"]

    blocklist = names | {bot_engine._gate_root_cause(name) for name in names}
    processor, records, submitted = _processor(
        tmp_path / "cap", positions={"MSFT": 5}, cap=1, ineffective_gate_blocklist=set(blocklist)
    )
    _process(processor, target_dollars=100.0)
    assert submitted == []
    assert records[0]["gates"] == ["POSITION_CAP_BLOCK"]

    exiting, exit_records, exit_submitted = _processor(
        tmp_path / "stop", positions={"AAPL": 10}, ineffective_gate_blocklist=set(blocklist)
    )
    _process(exiting, target_dollars=200.0)
    assert exit_submitted == [-10]
    assert exit_records[0]["gates"][0] == "POSITION_STOP_LOSS_EXIT"


# ---- position cap ----------------------------------------------------------


def test_cap_blocks_an_opening_at_the_cap(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    processor, records, submitted = _processor(tmp_path, positions={"MSFT": 5, "NVDA": 2}, cap=2)

    with caplog.at_level(logging.INFO):
        result = _process(processor, target_dollars=100.0)
        _process(processor, target_dollars=100.0)

    assert result.submitted_increment == 0
    assert submitted == []
    assert records[0]["gates"] == ["POSITION_CAP_BLOCK"]
    assert records[0]["config_snapshot"]["position_cap"] == {
        "cap": 2,
        "occupied": 2,
        "positions": 2,
        "pending": 0,
        "cycle_openings": 0,
    }
    blocks = _messages(caplog, "POSITION_CAP_BLOCK")
    assert len(blocks) == 1  # throttled
    assert blocks[0].levelno == logging.INFO


@pytest.mark.parametrize(
    ("current", "target_dollars", "delta"),
    [(5, 100.0, 5), (5, 20.0, -3), (5, 0.0, -5)],  # add, trim, exit
)
def test_cap_never_blocks_adds_or_reductions(
    tmp_path: Path, current: int, target_dollars: float, delta: int
) -> None:
    processor, records, submitted = _processor(
        tmp_path, positions={"AAPL": current, "MSFT": 5, "NVDA": 2}, avg_entry=10.0, cap=1
    )

    _process(processor, target_dollars=target_dollars)

    assert submitted == [delta]
    assert "POSITION_CAP_BLOCK" not in records[0]["gates"]


def test_cap_counts_pending_orders_and_same_cycle_openings(tmp_path: Path) -> None:
    pending, pending_records, pending_submitted = _processor(
        tmp_path / "pending", positions={"MSFT": 5}, cap=2, pending=("NVDA",)
    )
    _process(pending, target_dollars=100.0)
    assert pending_submitted == []
    assert pending_records[0]["config_snapshot"]["position_cap"]["pending"] == 1

    processor, records, submitted = _processor(tmp_path / "cycle", positions={"MSFT": 5}, cap=2)
    processor.latest_price = {"AAPL": 10.0, "AMD": 10.0}
    first = _process(processor, target_dollars=100.0, symbol="AAPL")
    second = _process(processor, target_dollars=100.0, symbol="AMD")

    assert first.submitted_increment == 1
    assert second.submitted_increment == 0
    assert submitted == [10]
    assert records[1]["gates"] == ["POSITION_CAP_BLOCK"]
    assert records[1]["config_snapshot"]["position_cap"]["cycle_openings"] == 1


def test_symbol_with_its_own_pending_order_takes_no_new_slot(tmp_path: Path) -> None:
    processor, _records, submitted = _processor(
        tmp_path, positions={"MSFT": 5}, cap=2, pending=("AAPL",)
    )

    _process(processor, target_dollars=100.0)

    assert submitted == [10]


@pytest.mark.parametrize("cap", [10, None, 0, "x"])
def test_below_the_cap_or_without_a_cap_nothing_changes(tmp_path: Path, cap: Any) -> None:
    processor, records, submitted = _processor(tmp_path, positions={"MSFT": 5, "NVDA": 2}, cap=cap)
    plain, plain_records = _make_processor(positions={"MSFT": 5, "NVDA": 2})
    plain.now = NOW

    assert _process(processor, target_dollars=100.0) == _process(plain, target_dollars=100.0)
    assert submitted == [10]
    assert records[0]["gates"] == plain_records[0]["gates"]
    assert records[0]["config_snapshot"] == plain_records[0]["config_snapshot"]


def test_over_the_cap_logs_once_per_session_and_never_force_closes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    positions = {"AAPL": 10, "MSFT": 5, "NVDA": 2}
    with caplog.at_level(logging.INFO):
        processor, records, submitted = _processor(tmp_path, positions=positions, avg_entry=10.0, cap=2)
        _backstop(tmp_path, positions=positions, cap=2, now=NOW + timedelta(minutes=5), backstop=processor.position_backstop)
        _process(processor, target_dollars=100.0)  # sleeves hold AAPL at $100

    (exceeded,) = _messages(caplog, "POSITION_CAP_EXCEEDED")
    assert exceeded.levelno == logging.WARNING
    assert (exceeded.positions, exceeded.cap) == (3, 2)
    assert submitted == []
    assert records[0]["gates"] == []


# ---- a full netting cycle --------------------------------------------------


class _AllowBreakers:
    def allow(self, dep: str) -> bool:
        return True

    def open_reason(self, dep: str) -> str | None:
        return None

    def record_failure(self, dep: str, error_info: Any) -> None:
        _ = (dep, error_info)

    def record_success(self, dep: str) -> None:
        _ = dep


class _NoBarDedupeDict(dict[str, datetime]):
    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        _ = (key, default)
        return None


_TARGET_DOLLARS = {"AAPL": 1010.0, "MSFT": 1515.0}


def _patch_cycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, list[tuple[str, int, str]]]:
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    now = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1000],
        },
        index=pd.DatetimeIndex([now - timedelta(minutes=1), now], tz=UTC),
    )

    def _force_target(symbol: str, bar_ts: datetime, proposals: Any, disagree_ratio: Any) -> NettedTarget:
        return NettedTarget(
            symbol=str(symbol),
            bar_ts=bar_ts,
            target_dollars=_TARGET_DOLLARS[str(symbol).upper()],
            target_shares=0.0,
            proposals=list(proposals),
            disagreement_ratio=0.0,
        )

    def _proposal(sleeve: Any, symbol: str, bar_ts: datetime, *_args: Any, **_kwargs: Any) -> SleeveProposal:
        return SleeveProposal(
            symbol=str(symbol),
            sleeve=sleeve.name,
            bar_ts=bar_ts,
            target_dollars=_TARGET_DOLLARS[str(symbol).upper()],
            expected_edge_bps=8.0,
            expected_cost_bps=2.0,
            score=0.5,
            confidence=0.8,
        )

    submitted: list[tuple[str, int, str]] = []

    def _fake_submit(runtime_obj: Any, symbol: str, qty: Any, side: str, **_kwargs: Any) -> Any:
        submitted.append((str(symbol), int(qty), str(side)))
        runtime_obj.execution_engine._last_submit_outcome = {"status": "submitted", "symbol": str(symbol), "side": str(side)}
        return SimpleNamespace(id=f"order-{symbol}", status="submitted", filled_avg_price=None)

    for key, value in {
        "AI_TRADING_POLICY_STRICT_CONFIG_GOVERNANCE": "0",
        "AI_TRADING_KILL_SWITCH": "0",
        "AI_TRADING_CANARY_SYMBOLS": "AAPL,MSFT",
        "AI_TRADING_WARMUP_MODE": "0",
        "AI_TRADING_EVENT_DRIVEN_NEW_BAR_ONLY": "0",
        "AI_TRADING_NETTING_CYCLE_SLO_LOG_TTL_SEC": "0",
        "AI_TRADING_DERISK_ON_SLO_BREACH_ENABLED": "0",
        "AI_TRADING_LIQ_REGIME_ENABLED": "0",
        "AI_TRADING_CAPACITY_AWARE_THROTTLE_ENABLED": "0",
        "AI_TRADING_EVENT_RISK_BLACKOUT_ENABLED": "0",
        "AI_TRADING_QUARANTINE_ENABLED": "0",
        "AI_TRADING_PRIMARY_FEED_DERISK_ENABLED": "0",
        "AI_TRADING_PARTICIPATION_CAP_ENABLED": "0",
        "AI_TRADING_ALPHA_DECAY_DEWEIGHT_ENABLED": "0",
        "AI_TRADING_EXECUTION_COST_AWARE_ENTRY_GUARD_ENABLED": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        "ai_trading.data.fetch.get_bars_batch",
        lambda symbols, timeframe, start, end: {str(sym).upper(): df for sym in symbols},
    )
    monkeypatch.setattr("ai_trading.core.netting.net_targets_for_symbol", _force_target)
    monkeypatch.setattr(
        "ai_trading.core.horizons.build_sleeve_configs",
        lambda cfg=None: [
            SleeveConfig(
                name="day", timeframe="1Min", enabled=True, entry_threshold=0.2, exit_threshold=0.1,
                flip_threshold=0.3, reentry_threshold=0.6, deadband_dollars=50.0, deadband_shares=1.0,
                turnover_cap_dollars=0.0, cost_k=1.5, edge_scale_bps=20.0, max_symbol_dollars=10000.0,
                max_gross_dollars=50000.0,
            )
        ],
    )
    monkeypatch.setattr("ai_trading.core.netting.compute_sleeve_proposal", _proposal)
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda cfg: (False, None))
    # These cycles exercise sizing and exits on the paper rule-sampling path.
    monkeypatch.setattr(bot_engine, "rule_paper_sampling_allowed", lambda _cfg: True)
    monkeypatch.setattr(bot_engine, "_dependency_breakers", lambda _state: _AllowBreakers())
    monkeypatch.setattr(
        "ai_trading.oms.ledger.deterministic_client_order_id",
        lambda **kwargs: f"{kwargs['symbol']}-{kwargs['bar_ts']}-{kwargs['side']}-{int(abs(float(kwargs['qty'])))}",
    )
    monkeypatch.setattr("ai_trading.oms.ledger.OrderLedger.seen_client_order_id", lambda self, cid: False)
    monkeypatch.setattr("ai_trading.oms.ledger.OrderLedger.record", lambda self, entry: None)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda _now=None: True)
    monkeypatch.setattr(bot_engine, "retry_idempotent", lambda fn, **_kwargs: fn())
    monkeypatch.setattr(bot_engine, "ensure_data_fetcher", lambda runtime_obj: None)
    monkeypatch.setattr(bot_engine, "compute_current_positions", lambda runtime_obj: {"MSFT": 15})
    monkeypatch.setattr(bot_engine, "check_daily_loss", lambda runtime_obj, state_obj: False)
    monkeypatch.setattr(bot_engine, "check_weekly_loss", lambda runtime_obj, state_obj: False)
    monkeypatch.setattr(bot_engine, "_run_reconciliation_if_due", lambda *args, **kwargs: True)
    monkeypatch.setattr(bot_engine, "_pre_rank_execution_candidates", lambda symbols, runtime=None: list(symbols))
    monkeypatch.setattr(
        bot_engine,
        "approve_execution_candidate",
        lambda policy, candidate: ExecutionApproval(True, int(candidate.proposed_delta_shares), 5.0, tuple()),
    )
    for name in (
        "_run_post_trade_learning_update",
        "_run_tca_cost_calibration",
        "_run_replay_governance",
        "_run_walk_forward_governance",
    ):
        monkeypatch.setattr(bot_engine, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_engine, "_update_rollout_governance_state", lambda *args, **kwargs: {"capital_ramp": {}})
    monkeypatch.setattr(bot_engine, "_resolve_primary_feed_derisk_state", lambda _runtime: {})
    monkeypatch.setattr(bot_engine, "enforce_participation_cap", lambda **kwargs: (True, float(kwargs["order_qty"]), None))
    monkeypatch.setattr(
        bot_engine,
        "_alpha_decay_entry_guard",
        lambda *args, **kwargs: {"blocked": False, "trades_in_window": 0, "start_trades": 0},
    )
    monkeypatch.setattr(bot_engine, "is_near_event", lambda _symbol, days=0: False)
    monkeypatch.setattr(bot_engine, "_tca_stale_block_reason", lambda _now: None)
    monkeypatch.setattr(bot_engine, "safe_validate_pretrade", lambda *args, **kwargs: (True, "OK", {}))
    monkeypatch.setattr(bot_engine, "submit_order", _fake_submit)
    written: list[Any] = []
    monkeypatch.setattr(bot_engine, "_write_decision_record", lambda record, path: written.append(record))
    return written, submitted


def test_netting_cycle_exits_a_stopped_position_and_blocks_nothing_else(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """AAPL is an entry from flat; MSFT is held at 15 and the sleeves want to keep it."""

    written, submitted = _patch_cycle(monkeypatch, tmp_path)
    cfg = TradingConfig.from_env(allow_missing_drawdown=True).update(
        data_contract_enabled=False,
        recon_enabled=False,
        ledger_enabled=False,
        rth_only=False,
        allow_extended=True,
        decision_log_path=None,
    )
    sync = _sync(SimpleNamespace(symbol="MSFT", qty="15", side="long", avg_entry_price="110.0"))
    runtime = SimpleNamespace(
        cfg=cfg,
        tickers=["AAPL", "MSFT"],
        universe_tickers=["AAPL", "MSFT"],
        api=None,
        execution_engine=SimpleNamespace(
            _last_submit_outcome={},
            _get_account_snapshot=lambda: SimpleNamespace(id="acct-1", equity="100000"),
            _broker_sync=sync,
        ),
    )
    state = bot_engine.BotState()
    state.last_order_bar_ts = _NoBarDedupeDict()

    with caplog.at_level(logging.INFO):
        bot_engine._run_netting_cycle(state, runtime, "loop", 0.0)
        first = list(submitted)
        # Same bars again: no new bar, so nothing is evaluated or submitted.
        bot_engine._run_netting_cycle(state, runtime, "loop", 0.0)

    assert state.halt_trading is False
    assert sorted((symbol, side) for symbol, _qty, side in first) == [("AAPL", "buy"), ("MSFT", "sell")]
    assert ("MSFT", 15, "sell") in first
    assert submitted == first
    gates = {str(r.symbol).upper(): list(r.gates or []) for r in written}
    snapshots = {str(r.symbol).upper(): dict(r.config_snapshot or {}) for r in written}
    assert "POSITION_STOP_LOSS_EXIT" in gates["MSFT"]
    assert "OK_TRADE" in gates["MSFT"]
    assert snapshots["MSFT"]["position_backstop"]["reason"] == "stop_loss"
    assert "POSITION_STOP_LOSS_EXIT" not in gates["AAPL"]
    assert "position_backstop" not in snapshots["AAPL"]
    (breach,) = _messages(caplog, "POSITION_STOP_LOSS_BREACH")
    assert breach.symbol == "MSFT"
    assert breach.loss_pct == pytest.approx(101.0 / 110.0 - 1.0)
    saved = json.loads((tmp_path / "runtime" / "position_backstop_state.json").read_text())
    assert saved["locks"]["MSFT"]["reason"] == "stop_loss"
    assert saved["episodes"]["MSFT"]["opened_at_source"] == "first_seen"


# ---- review follow-up: thin-liquidity cap, soft throttles, store, restart --------


@pytest.fixture
def thin_regime(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_trading.risk.liquidity_regime import LiquidityRegime

    monkeypatch.setattr(netting_symbol_cycle, "classify_liquidity_regime", lambda *a, **k: LiquidityRegime.THIN)
    monkeypatch.setenv("AI_TRADING_PARTICIPATION_CAP_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_LIQ_THIN_COST_MULT", "1.3")
    monkeypatch.setenv("AI_TRADING_LIQ_THIN_MAX_ORDER_DOLLARS", "5000")


@pytest.mark.usefixtures("thin_regime")
def test_thin_max_order_clips_exits_and_still_blocks_an_entry(tmp_path: Path) -> None:
    # 1000 shares at $10: the thin cost multiplier makes it 769 shares ($7,690),
    # above the $5,000 thin cap.
    normal, normal_records, normal_submitted = _processor(
        tmp_path / "normal", positions={"AAPL": 1000}, avg_entry=10.2, liq_regime_enabled=True
    )
    protective, records, submitted = _processor(
        tmp_path / "protective", positions={"AAPL": 1000}, liq_regime_enabled=True
    )
    entry, entry_records, entry_submitted = _processor(tmp_path / "entry", positions={}, liq_regime_enabled=True)

    _process(normal, target_dollars=0.0)
    _process(protective, target_dollars=0.0)
    _process(entry, target_dollars=10_000.0)

    # Normal exit (was: blocked): clipped to the cap like a protective exit.
    assert normal_submitted == [-500]
    assert normal_records[0]["gates"] == [
        "LIQ_THIN_COST_MULT_SCALE",
        "LIQ_THIN_MAX_ORDER_BLOCK",
        "CLOSING_EXIT_OVERRIDE",
        "OK_TRADE",
    ]
    assert entry_submitted == []
    assert entry_records[0]["gates"] == ["LIQ_THIN_COST_MULT_SCALE", "LIQ_THIN_MAX_ORDER_BLOCK"]
    assert submitted == [-500]
    assert records[0]["gates"] == [
        "POSITION_STOP_LOSS_EXIT",
        "LIQ_THIN_COST_MULT_SCALE",
        "LIQ_THIN_MAX_ORDER_BLOCK",
        "PROTECTIVE_EXIT_OVERRIDE",
        "OK_TRADE",
    ]
    # The lock finishes the rest on later bars.
    assert "AAPL" in protective.position_backstop.store.locks


@pytest.mark.usefixtures("thin_regime")
def test_flip_is_judged_on_its_exit_leg_by_the_thin_cap(tmp_path: Path) -> None:
    # Nov 19 AAPL: long 1, sleeves want short ~1000. The whole flip ($7,700 after
    # the cost multiplier) used to hit LIQ_THIN_MAX_ORDER_BLOCK, holding the
    # one-share exit; the exit leg alone is far under the cap.
    processor, records, submitted = _processor(
        tmp_path, positions={"AAPL": 1}, avg_entry=10.2, liq_regime_enabled=True
    )

    _process(processor, target_dollars=-10_000.0)

    assert submitted == [-1]
    assert records[0]["gates"][0] == "FLAT_BEFORE_REVERSAL"
    assert "LIQ_THIN_MAX_ORDER_BLOCK" not in records[0]["gates"]
    assert records[0]["gates"][-1] == "OK_TRADE"
    assert records[0]["net_target"].target_shares == 0


@pytest.mark.usefixtures("thin_regime")
def test_thin_max_order_clip_sends_at_least_one_share(tmp_path: Path) -> None:
    # 3 shares at $6,000: the $5,000 cap rounds to zero shares; one goes out.
    processor, records, submitted = _processor(
        tmp_path, positions={"AAPL": 3}, avg_entry=7000.0, liq_regime_enabled=True
    )
    processor.latest_price = {"AAPL": 6000.0}

    _process(processor, target_dollars=0.0)

    assert submitted == [-1]
    assert "LIQ_THIN_MAX_ORDER_BLOCK" in records[0]["gates"]


def _throttle_policy(*, factor_hard_ratio: float = 0.5001) -> Any:
    big = 1e12
    return SimpleNamespace(
        objective=SimpleNamespace(fee_bps=0.0, borrow_bps=0.0, min_expected_net_edge_bps=-1000.0),
        execution=SimpleNamespace(
            stale_order_block_sec=1e9, max_spread_bps=1000.0, min_rolling_volume=0.0, calibration_block_enabled=False
        ),
        risk_budgets=SimpleNamespace(
            symbol_hard_dollars=big, symbol_soft_dollars=big,
            portfolio_hard_gross_dollars=big, portfolio_soft_gross_dollars=big,
            sleeve_hard_dollars=big, sleeve_soft_dollars=big,
            factor_soft_ratio=0.1, factor_hard_ratio=factor_hard_ratio,
        ),
        calibration=SimpleNamespace(
            max_ece_normal=1.0, max_brier_normal=1.0, max_ece_stress=1.0, max_brier_stress=1.0, min_samples=10
        ),
    )


def _real_approval(policy: Any) -> dict[str, Any]:
    from ai_trading.policy.compiler import approve_execution_candidate

    # Exiting 100 shares at $10 leaves a sector share of 0.5 of the book:
    # RISK_FACTOR_SOFT_THROTTLE scales the order to a tenth.
    return {
        "effective_policy": policy,
        "portfolio_current_gross": 2000.0,
        "sector_gross": {"TECH": 1500.0},
        "get_sector_func": lambda symbol: "TECH",
        "evaluate_execution_approval_func": evaluate_execution_approval,
        "approve_execution_candidate_func": approve_execution_candidate,
        "prepare_symbol_approval_func": prepare_netting_symbol_approval,
    }


def test_soft_throttle_keeps_exits_full_and_still_scales_an_entry(tmp_path: Path) -> None:
    overrides = _real_approval(_throttle_policy())
    normal, normal_records, normal_submitted = _processor(
        tmp_path / "normal", positions={"AAPL": 100}, avg_entry=10.2, **overrides
    )
    protective, records, submitted = _processor(tmp_path / "protective", positions={"AAPL": 100}, **overrides)
    entry_overrides = _real_approval(_throttle_policy(factor_hard_ratio=0.99))
    entry, entry_records, entry_submitted = _entry_processor(tmp_path / "entry", entry_overrides)

    _process(normal, target_dollars=0.0)
    _process(protective, target_dollars=0.0)
    _process(entry, target_dollars=1000.0)

    # Normal exit (was: scaled to a tenth): full size, own marker.
    assert normal_submitted == [-100]
    assert "RISK_FACTOR_SOFT_THROTTLE" in normal_records[0]["gates"]
    assert normal_records[0]["gates"].count("CLOSING_EXIT_OVERRIDE") == 1
    # Entry of 100 shares: still scaled by the soft throttle, no override.
    assert len(entry_submitted) == 1 and 0 < entry_submitted[0] < 100
    assert "RISK_FACTOR_SOFT_THROTTLE" in entry_records[0]["gates"]
    assert not _OVERRIDE_MARKERS & set(entry_records[0]["gates"])
    assert submitted == [-100]
    assert "RISK_FACTOR_SOFT_THROTTLE" in records[0]["gates"]
    assert records[0]["gates"].count("PROTECTIVE_EXIT_OVERRIDE") == 1


@pytest.mark.parametrize("avg_entry", [11.0, 10.2], ids=["protective", "normal"])
def test_hard_risk_block_still_holds_an_exit(tmp_path: Path, avg_entry: float) -> None:
    processor, records, submitted = _processor(
        tmp_path,
        positions={"AAPL": 100},
        avg_entry=avg_entry,
        **_real_approval(_throttle_policy(factor_hard_ratio=0.2)),
    )

    _process(processor, target_dollars=0.0)

    assert submitted == []
    assert "RISK_FACTOR_HARD_BLOCK" in records[0]["gates"]
    assert not _OVERRIDE_MARKERS & set(records[0]["gates"])


def test_malformed_row_drops_only_that_row(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "runtime" / "position_backstop_state.json"
    path.parent.mkdir(parents=True)
    good_lock = {
        "until": NEXT_OPEN.isoformat(), "reason": "stop_loss", "armed_at": NOW.isoformat(),
        "entry_price": 100.0, "trigger_price": 94.0, "side": "long",
    }
    path.write_text(json.dumps({
        "episodes": {
            "AAPL": {"opened_at": NOW.isoformat(), "opened_at_source": "observed", "side": "long"},
            "MSFT": {"side": "long"},
            "NVDA": "garbage",
        },
        "locks": {"AAPL": good_lock, "AMD": {**good_lock, "until": "not-a-time"}},
    }))

    with caplog.at_level(logging.INFO):
        backstop = PositionBackstop(path, LOGGER)

    assert set(backstop.store.episodes) == {"AAPL"}
    assert set(backstop.store.locks) == {"AAPL"}
    (dropped,) = _messages(caplog, "POSITION_BACKSTOP_STATE_ROWS_DROPPED")
    assert dropped.levelno == logging.WARNING
    assert dropped.dropped_rows == 3
    assert not _messages(caplog, "POSITION_BACKSTOP_STATE_LOAD_FAILED")
    _backstop(tmp_path, positions={"AAPL": 10}, backstop=backstop)
    saved = json.loads(path.read_text())
    assert set(saved["episodes"]) == {"AAPL"} and set(saved["locks"]) == {"AAPL"}


def test_failed_calendar_lookup_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    day = datetime(2031, 3, 4).date()  # a Tuesday
    nbp._cached_session_bounds.cache_clear()
    real = nbp.market_calendar.session_info
    calls = {"n": 0}

    def _flaky(d: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("calendar unavailable")
        return real(d, **kwargs)

    monkeypatch.setattr(nbp.market_calendar, "session_info", _flaky)
    try:
        assert nbp._session_bounds(day) is None
        bounds = nbp._session_bounds(day)
        assert bounds is not None and bounds[1].astimezone(ET).hour == 16
    finally:
        nbp._cached_session_bounds.cache_clear()


def test_restart_with_a_flipped_side_starts_a_new_episode(tmp_path: Path) -> None:
    opened = NOW - timedelta(days=3)
    first = _backstop(tmp_path, positions={"AAPL": 10, "MSFT": 5}, now=opened)
    assert first.store.episodes["AAPL"].opened_at == opened

    # Restart: AAPL is now short (flipped while down), MSFT still long.
    restarted = _backstop(tmp_path, positions={"AAPL": -4, "MSFT": 7})

    assert restarted.store.episodes["AAPL"] == Episode(NOW, "first_seen", "short")
    assert restarted.store.episodes["MSFT"].opened_at == opened


# ---- owner-approved: zero-volume participation block on a protective exit -----


@pytest.fixture
def real_participation_cap(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from ai_trading.risk.liquidity_regime import enforce_participation_cap

    monkeypatch.setenv("AI_TRADING_PARTICIPATION_CAP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PARTICIPATION_BLOCK_MODE", "block")
    monkeypatch.setenv("AI_TRADING_MAX_PARTICIPATION_PCT", "0.015")
    monkeypatch.setenv("AI_TRADING_LIQ_THIN_COST_MULT", "1.3")
    monkeypatch.setenv("AI_TRADING_LIQ_THIN_MAX_ORDER_DOLLARS", "5000")
    return {"enforce_participation_cap_func": enforce_participation_cap}


@pytest.mark.parametrize("thin", [False, True])
def test_zero_volume_protective_exit_goes_out_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_participation_cap: dict[str, Any], thin: bool
) -> None:
    if thin:  # whatever regime is computed, the cap applies once
        from ai_trading.risk.liquidity_regime import LiquidityRegime

        monkeypatch.setattr(netting_symbol_cycle, "classify_liquidity_regime", lambda *a, **k: LiquidityRegime.THIN)
    processor, records, submitted = _processor(
        tmp_path, positions={"AAPL": 1000}, liq_regime_enabled=thin, **real_participation_cap
    )
    processor.latest_liquidity = {}  # no liquidity entry: rolling volume defaults to 0

    _process(processor, target_dollars=0.0)

    assert submitted == [-500]  # $5,000 cap at $10
    gates = records[0]["gates"]
    assert gates[:3] == ["POSITION_STOP_LOSS_EXIT", "LIQ_PARTICIPATION_BLOCK", "PROTECTIVE_EXIT_OVERRIDE"]
    assert gates[-1] == "OK_TRADE"
    assert "LIQ_THIN_COST_MULT_SCALE" not in gates and "LIQ_THIN_MAX_ORDER_BLOCK" not in gates
    assert "AAPL" in processor.position_backstop.store.locks


def test_zero_volume_protective_exit_sends_at_least_one_share(
    tmp_path: Path, real_participation_cap: dict[str, Any]
) -> None:
    processor, records, submitted = _processor(
        tmp_path, positions={"AAPL": 3}, avg_entry=7000.0, **real_participation_cap
    )
    processor.latest_price = {"AAPL": 6000.0}
    processor.latest_liquidity = {}

    _process(processor, target_dollars=0.0)

    assert submitted == [-1]
    assert "LIQ_PARTICIPATION_BLOCK" in records[0]["gates"]


def test_zero_volume_normal_exit_goes_out_capped_and_an_entry_is_still_blocked(
    tmp_path: Path, real_participation_cap: dict[str, Any]
) -> None:
    normal, normal_records, normal_submitted = _processor(
        tmp_path / "normal", positions={"AAPL": 1000}, avg_entry=10.2, **real_participation_cap
    )
    entry, entry_records, entry_submitted = _processor(tmp_path / "entry", positions={}, **real_participation_cap)
    normal.latest_liquidity = {}
    entry.latest_liquidity = {}

    _process(normal, target_dollars=0.0)
    entry_result = _process(entry, target_dollars=10_000.0)

    # Normal exit (was: blocked): capped at $5,000 like a protective exit.
    assert normal_submitted == [-500]
    assert normal_records[0]["gates"][:2] == ["LIQ_PARTICIPATION_BLOCK", "CLOSING_EXIT_OVERRIDE"]
    assert normal_records[0]["gates"][-1] == "OK_TRADE"
    assert entry_submitted == []
    assert entry_result.submitted_increment == 0
    assert entry_records[0]["gates"] == ["LIQ_PARTICIPATION_BLOCK"]


@pytest.mark.parametrize("avg_entry", [11.0, 10.2], ids=["protective", "normal"])
@pytest.mark.parametrize(("mode", "expected"), [("block", []), ("scale", [-25])])
def test_positive_volume_participation_cap_still_applies_to_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_participation_cap: dict[str, Any],
    mode: str,
    expected: list[int],
    avg_entry: float,
) -> None:
    monkeypatch.setenv("AI_TRADING_PARTICIPATION_BLOCK_MODE", mode)
    # 1000 shares of volume at 1.5% allows 15 shares; the exit is 100.
    processor, records, submitted = _processor(
        tmp_path, positions={"AAPL": 100}, avg_entry=avg_entry, **real_participation_cap
    )

    _process(processor, target_dollars=0.0)

    assert submitted == expected
    assert not _OVERRIDE_MARKERS & set(records[0]["gates"])
    if mode == "block":
        backstop_gates = ["POSITION_STOP_LOSS_EXIT"] if avg_entry == 11.0 else []
        assert records[0]["gates"] == [*backstop_gates, "LIQ_PARTICIPATION_BLOCK"]
    else:
        assert "LIQ_REGIME_THIN_SCALE" in records[0]["gates"]


# ---- ATTACK tier: approval upsizing a trim to flat must not make it a full close ----


def _attack_policy() -> Any:
    big = 1e12
    return SimpleNamespace(
        objective=SimpleNamespace(fee_bps=0.0, borrow_bps=0.0, min_expected_net_edge_bps=-1000.0),
        execution=SimpleNamespace(
            stale_order_block_sec=1e9, max_spread_bps=1000.0, min_rolling_volume=0.0, calibration_block_enabled=False
        ),
        risk_budgets=SimpleNamespace(
            symbol_hard_dollars=big, symbol_soft_dollars=big,
            portfolio_hard_gross_dollars=big, portfolio_soft_gross_dollars=big,
            sleeve_hard_dollars=big, sleeve_soft_dollars=big,
            factor_soft_ratio=10.0, factor_hard_ratio=20.0,
        ),
        calibration=SimpleNamespace(
            max_ece_normal=1.0, max_brier_normal=1.0, max_ece_stress=1.0, max_brier_stress=1.0, min_samples=10
        ),
        # Relaxed ATTACK sizing: x1.05, so a 96-share trim of 100 becomes 101, clipped to 100.
        safety=SimpleNamespace(
            attack_size_multiplier=1.15,
            attack_relax_reject_rate_pct=2.0,
            attack_relax_size_multiplier=1.05,
            attack_degrade_reject_rate_pct=8.0,
            attack_degrade_size_multiplier=1.0,
        ),
    )


def _attack_processor(tmp_path: Path, current: int, **overrides: Any) -> tuple[Any, list[dict[str, Any]], list[int]]:
    from ai_trading.policy.compiler import approve_execution_candidate
    from ai_trading.risk.liquidity_regime import LiquidityFeatures

    processor, records, submitted = _processor(
        tmp_path,
        positions={"AAPL": current},
        avg_entry=10.2,
        effective_policy=_attack_policy(),
        evaluate_execution_approval_func=evaluate_execution_approval,
        approve_execution_candidate_func=approve_execution_candidate,
        prepare_symbol_approval_func=prepare_netting_symbol_approval,
        clip_sell_qty_to_available_position_func=lambda **k: (
            min(int(k["requested_qty"]), abs(int(k["current_shares"]))),
            None,
        ),
        latest_liquidity={"AAPL": LiquidityFeatures(rolling_volume=1e6, spread_bps=5.0, volatility_proxy=0.0)},
        **overrides,
    )
    processor.state.operational_safety_tier = "attack"
    return processor, records, submitted


_ATTACK_CAPACITY = {
    "capacity_throttle_enabled": True,
    "capacity_spread_soft_bps": 1.0,
    "capacity_spread_hard_bps": 2.0,
    "capacity_min_scale": 0.5,
}


@pytest.mark.parametrize(
    ("current", "target_shares", "expected"),
    [(100, 4, [-50]), (-100, -3, [50])],
    ids=["long_trim_96", "short_cover_97"],
)
def test_attack_upsized_trim_is_still_capacity_throttled_as_on_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, current: int, target_shares: int, expected: list[int]
) -> None:
    monkeypatch.setattr(netting_symbol_cycle, "_short_openings_allowed", lambda _cfg: True)
    processor, records, submitted = _attack_processor(tmp_path, current, **_ATTACK_CAPACITY)

    _process(processor, target_dollars=target_shares * 10.0)

    # Main: ATTACK upsizes the trim to 101/100, the close-only clip makes it
    # exactly flat, and the capacity throttle still halves it.
    assert submitted == expected
    assert "CAPACITY_THROTTLE_SCALE" in records[0]["gates"]
    assert "SAFETY_TIER_ATTACK_SCALE" in records[0]["gates"]
    assert not _OVERRIDE_MARKERS & set(records[0]["gates"])


@pytest.mark.parametrize(("current", "expected"), [(100, [-100]), (-100, [100])], ids=["long", "short"])
def test_attack_full_close_still_passes_the_capacity_throttle(
    tmp_path: Path, current: int, expected: list[int]
) -> None:
    processor, records, submitted = _attack_processor(tmp_path, current, **_ATTACK_CAPACITY)

    _process(processor, target_dollars=0.0)

    assert submitted == expected
    assert "CAPACITY_THROTTLE_SCALE" in records[0]["gates"]
    assert records[0]["gates"].count("CLOSING_EXIT_OVERRIDE") == 1


def _real_optimizer_prelude(**kwargs: Any) -> Any:
    # The real submit prelude with an optimizer that rejects, fed the cycle's order.
    keys = ("side", "delta_shares", "current_shares", "opening_trade", "protective_exit", "full_close_exit")
    return prepare_netting_submit_prelude(**_exit_prelude_kwargs(**{k: kwargs[k] for k in keys if k in kwargs}))


@pytest.mark.parametrize(
    ("current", "target_shares", "expected_full_close", "submitted_expected"),
    [
        (100, 4, False, []),  # trim 96, upsized to flat: rejected as on main
        (-100, -3, False, []),  # cover 97, upsized to flat: rejected as on main
        (100, 0, True, [-100]),  # full close: passes
        (-100, 0, True, [100]),
    ],
    ids=["long_trim_96", "short_cover_97", "long_close", "short_close"],
)
def test_attack_optimizer_reject_holds_an_upsized_trim_and_passes_a_full_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current: int,
    target_shares: int,
    expected_full_close: bool,
    submitted_expected: list[int],
) -> None:
    monkeypatch.setattr(netting_symbol_cycle, "_short_openings_allowed", lambda _cfg: True)
    seen: list[tuple[int, bool]] = []

    def _prelude(**kwargs: Any) -> Any:
        seen.append((int(kwargs["delta_shares"]), bool(kwargs["full_close_exit"])))
        return _real_optimizer_prelude(**kwargs)

    processor, records, submitted = _attack_processor(tmp_path, current, prepare_submit_prelude_func=_prelude)

    _process(processor, target_dollars=target_shares * 10.0)

    # The order reaching the optimizer is flat-sized either way; only the
    # decision taken on the intended order tells them apart.
    assert seen == [(-current, expected_full_close)]
    assert submitted == submitted_expected
    gates = records[0]["gates"]
    assert "PORTFOLIO_OPTIMIZER_REJECT" in gates
    if expected_full_close:
        assert gates.count("CLOSING_EXIT_OVERRIDE") == 1
    else:
        assert not _OVERRIDE_MARKERS & set(gates)
        assert records[0]["metrics"]["terminal_reason"] == "PORTFOLIO_OPTIMIZER_REJECT"


def test_netting_cycle_without_qualified_strategy_still_takes_backstop_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no qualified strategy the entry is refused; the stop-loss exit is not."""

    written, submitted = _patch_cycle(monkeypatch, tmp_path)
    monkeypatch.setattr(bot_engine, "rule_paper_sampling_allowed", lambda _cfg: False)
    cfg = TradingConfig.from_env(allow_missing_drawdown=True).update(
        data_contract_enabled=False,
        recon_enabled=False,
        ledger_enabled=False,
        rth_only=False,
        allow_extended=True,
        decision_log_path=None,
    )
    sync = _sync(SimpleNamespace(symbol="MSFT", qty="15", side="long", avg_entry_price="110.0"))
    runtime = SimpleNamespace(
        cfg=cfg,
        tickers=["AAPL", "MSFT"],
        universe_tickers=["AAPL", "MSFT"],
        api=None,
        execution_engine=SimpleNamespace(
            _last_submit_outcome={},
            _get_account_snapshot=lambda: SimpleNamespace(id="acct-1", equity="100000"),
            _broker_sync=sync,
        ),
    )
    state = bot_engine.BotState()
    state.last_order_bar_ts = _NoBarDedupeDict()

    bot_engine._run_netting_cycle(state, runtime, "loop", 0.0)

    assert submitted == [("MSFT", 15, "sell")]
    gates = {str(r.symbol).upper(): list(r.gates or []) for r in written}
    assert "POSITION_STOP_LOSS_EXIT" in gates["MSFT"]
    assert "NO_QUALIFIED_STRATEGY" in gates["AAPL"]
