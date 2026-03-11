from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trading.core import bot_engine


class _State:
    last_walk_forward_run_date = None


def _write_tca(path: Path, *, rows: int) -> None:
    base_ts = datetime(2026, 2, 1, 15, 0, tzinfo=UTC)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(rows):
            payload = {
                "ts": (base_ts + timedelta(minutes=idx)).isoformat(),
                "is_bps": float(idx % 5),
                "qty": 1.0,
                "fill_price": 100.0,
            }
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


def test_walk_forward_skips_leakage_when_no_folds(monkeypatch, tmp_path: Path) -> None:
    tca_path = tmp_path / "runtime" / "tca_records.jsonl"
    _write_tca(tca_path, rows=12)
    monkeypatch.setattr(bot_engine, "_walk_forward_schedule_due", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(bot_engine, "_resolved_tca_path", lambda: tca_path)
    monkeypatch.setenv("AI_TRADING_WF_OUTPUT_DIR", str(tmp_path / "wf"))
    monkeypatch.setenv("AI_TRADING_LEAKAGE_GUARDS_ENABLED", "1")

    import ai_trading.research.leakage_tests as leakage_tests_module
    import ai_trading.research.walk_forward as walk_forward_module

    called = {"leakage": 0}

    monkeypatch.setattr(
        walk_forward_module,
        "run_walk_forward",
        lambda frame, score_fn, config: {"fold_count": 0},
    )

    def _fake_leakage(**_kwargs):
        called["leakage"] += 1

    monkeypatch.setattr(leakage_tests_module, "run_leakage_guards", _fake_leakage)

    state = _State()
    now = datetime(2026, 2, 20, 23, 0, tzinfo=UTC)
    bot_engine._run_walk_forward_governance(state, now=now, market_open_now=False)

    assert called["leakage"] == 0
    assert state.last_walk_forward_run_date == now.date()


def test_walk_forward_uses_explicit_horizon_days_for_leakage(monkeypatch, tmp_path: Path) -> None:
    tca_path = tmp_path / "runtime" / "tca_records.jsonl"
    _write_tca(tca_path, rows=15)
    monkeypatch.setattr(bot_engine, "_walk_forward_schedule_due", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(bot_engine, "_resolved_tca_path", lambda: tca_path)
    monkeypatch.setenv("AI_TRADING_WF_OUTPUT_DIR", str(tmp_path / "wf"))
    monkeypatch.setenv("AI_TRADING_LEAKAGE_GUARDS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_WF_LEAKAGE_MIN_ROWS", "1")
    monkeypatch.setenv("AI_TRADING_WF_HORIZON_DAYS", "3")
    monkeypatch.setenv("AI_TRADING_WF_EMBARGO_DAYS", "2")

    import ai_trading.research.leakage_tests as leakage_tests_module
    import ai_trading.research.walk_forward as walk_forward_module

    captured: dict[str, int] = {}

    monkeypatch.setattr(
        walk_forward_module,
        "run_walk_forward",
        lambda frame, score_fn, config: {"fold_count": 2},
    )

    def _fake_leakage(**kwargs):
        captured["horizon_days"] = int(kwargs["horizon_days"])
        captured["embargo_days"] = int(kwargs["embargo_days"])

    monkeypatch.setattr(leakage_tests_module, "run_leakage_guards", _fake_leakage)

    state = _State()
    now = datetime(2026, 2, 21, 23, 0, tzinfo=UTC)
    bot_engine._run_walk_forward_governance(state, now=now, market_open_now=False)

    assert captured["horizon_days"] == 3
    assert captured["embargo_days"] == 2
