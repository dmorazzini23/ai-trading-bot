from __future__ import annotations

from ai_trading.meta_learning import core as meta_core


def test_synthetic_bootstrap_blocked_without_override(monkeypatch, tmp_path) -> None:
    trade_log = tmp_path / "trades.csv"
    trade_log.write_text("timestamp,symbol,side,entry_price,exit_price,quantity,pnl,signal_tags\n", encoding="utf-8")
    call_counter = {"append_calls": 0}

    monkeypatch.setenv("AI_TRADING_META_LEARNING_ALLOW_SYNTHETIC_BOOTSTRAP", "0")
    monkeypatch.setenv("PYTEST_RUNNING", "0")
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setattr(
        meta_core,
        "_append_synthetic_trades_to_log",
        lambda *_a, **_k: call_counter.__setitem__("append_calls", call_counter["append_calls"] + 1),
    )

    meta_core._attempt_synthetic_data_generation(str(trade_log), min_samples=20)

    assert call_counter["append_calls"] == 0


def test_synthetic_bootstrap_allowed_with_explicit_override(monkeypatch, tmp_path) -> None:
    trade_log = tmp_path / "trades.csv"
    trade_log.write_text("timestamp,symbol,side,entry_price,exit_price,quantity,pnl,signal_tags\n", encoding="utf-8")
    call_counter = {"append_calls": 0}

    monkeypatch.setenv("AI_TRADING_META_LEARNING_ALLOW_SYNTHETIC_BOOTSTRAP", "1")
    monkeypatch.setenv("PYTEST_RUNNING", "0")
    monkeypatch.setattr(
        meta_core,
        "_append_synthetic_trades_to_log",
        lambda *_a, **_k: call_counter.__setitem__("append_calls", call_counter["append_calls"] + 1),
    )

    meta_core._attempt_synthetic_data_generation(str(trade_log), min_samples=20)

    assert call_counter["append_calls"] == 1
