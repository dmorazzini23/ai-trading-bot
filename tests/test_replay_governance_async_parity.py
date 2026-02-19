from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.core import bot_engine


class _State:
    last_replay_run_date = None


def test_replay_governance_uses_async_parity_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    monkeypatch.setattr(bot_engine, "_replay_schedule_due", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        bot_engine,
        "_load_replay_bars",
        lambda **_kwargs: [
            {
                "symbol": "AAPL",
                "ts": "2026-02-18T22:00:00+00:00",
                "close": 189.5,
                "side": "buy",
                "qty": 1,
                "client_order_id": "aapl-1",
            },
            {
                "symbol": "AAPL",
                "ts": "2026-02-18T22:01:00+00:00",
                "close": 189.8,
                "side": "buy",
                "qty": 1,
                "client_order_id": "aapl-2",
            },
        ],
    )
    monkeypatch.setenv("AI_TRADING_REPLAY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_REPLAY_SEED", "123")
    monkeypatch.setenv("AI_TRADING_REPLAY_SIMULATE_FILLS", "1")
    monkeypatch.setenv("AI_TRADING_REPLAY_ENFORCE_OMS_GATES", "1")

    state = _State()
    bot_engine._run_replay_governance(state, now=now, market_open_now=False)

    out_path = tmp_path / "replay_hash_20260218.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["simulate_fills"] is True
    assert payload["seed"] == 123
    assert payload["orders_submitted"] >= 1
    assert "violations" in payload
    assert state.last_replay_run_date == now.date()


def test_replay_governance_enforced_invariants_raise(
    monkeypatch,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    monkeypatch.setattr(bot_engine, "_replay_schedule_due", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        bot_engine,
        "_load_replay_bars",
        lambda **_kwargs: [
            {
                "symbol": "MSFT",
                "ts": "2026-02-18T22:00:00+00:00",
                "close": 405.0,
                "side": "buy",
                "qty": 1,
                "client_order_id": "dup-intent",
            },
            {
                "symbol": "MSFT",
                "ts": "2026-02-18T22:01:00+00:00",
                "close": 405.2,
                "side": "buy",
                "qty": 1,
                "client_order_id": "dup-intent",
            },
        ],
    )
    monkeypatch.setenv("AI_TRADING_REPLAY_SIMULATE_FILLS", "1")
    monkeypatch.setenv("AI_TRADING_REPLAY_ENFORCE_OMS_GATES", "1")

    state = _State()
    try:
        bot_engine._run_replay_governance(state, now=now, market_open_now=False)
    except RuntimeError as exc:
        assert "REPLAY_GOVERNANCE_INVARIANTS_FAILED" in str(exc)
    else:  # pragma: no cover - explicit safety
        raise AssertionError("Expected replay invariants to fail")
