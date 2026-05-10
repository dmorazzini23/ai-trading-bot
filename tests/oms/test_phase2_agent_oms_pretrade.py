from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.oms import pretrade
from ai_trading.oms.pretrade import OrderIntent, SlidingWindowRateLimiter


RTH_BAR = datetime(2026, 4, 21, 14, 30, tzinfo=UTC)


class _Ledger:
    positions = {"AAPL": 0.0}
    gross_notional = 0.0
    intraday_var = 0.0
    intraday_cvar = 0.0
    current_drawdown = 0.0
    daily_loss_pct = 0.0
    daily_loss_abs = 0.0
    execution_drift_bps = 0.0
    reject_rate_pct = 0.0

    @staticmethod
    def seen_client_order_id(_value: str) -> bool:
        return False


def _cfg(**updates: Any) -> SimpleNamespace:
    values = {
        "max_order_dollars": 0.0,
        "max_order_shares": 0,
        "price_collar_pct": 0.20,
        "max_symbol_notional": 0.0,
        "max_gross_notional": 0.0,
        "max_sector_notional": 0.0,
        "max_factor_exposure": 0.0,
        "intraday_var_limit": 0.0,
        "intraday_cvar_limit": 0.0,
        "intraday_drawdown_limit": 0.0,
        "quote_max_age_ms": 0,
        "daily_loss_limit_pct": 0.0,
        "daily_loss_limit_abs": 0.0,
        "rth_only": True,
        "allow_extended": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _intent(**updates: Any) -> OrderIntent:
    values = {
        "symbol": "AAPL",
        "side": "buy",
        "qty": 10,
        "notional": 1_000.0,
        "limit_price": 100.0,
        "bar_ts": RTH_BAR,
        "client_order_id": "cid-agent-oms",
        "last_price": 100.0,
        "mid": 100.0,
        "bid": 99.95,
        "ask": 100.05,
        "spread": 0.10,
        "submit_quote_source": "broker_nbbo",
        "quote_quality_ok": True,
        "liquidity_bucket": "NORMAL",
    }
    values.update(updates)
    return OrderIntent(**values)


def test_in_memory_cancel_loop_blocks_until_configured_future_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SlidingWindowRateLimiter, "_now", staticmethod(lambda: 1_000.0))
    limiter = SlidingWindowRateLimiter(
        cancel_loop_max_without_fill=2,
        cancel_loop_block_bars=2,
    )

    limiter.record_cancel("aapl", bar_ts=RTH_BAR, filled=False)
    limiter.record_cancel("AAPL", bar_ts=RTH_BAR, filled=False)

    same_bar = limiter.allow_order("AAPL", RTH_BAR)
    next_bar = limiter.allow_order("AAPL", RTH_BAR + timedelta(minutes=1))
    release_bar = limiter.allow_order("AAPL", RTH_BAR + timedelta(minutes=2))

    assert same_bar[0] is False
    assert same_bar[1] == "CANCEL_LOOP_BLOCK"
    assert same_bar[2]["symbol"] == "AAPL"
    assert next_bar[0] is False
    assert release_bar == (True, None, {})


def test_durable_rate_limiter_persists_reserved_order_slots(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "pretrade-rate.sqlite"
    monkeypatch.setattr(SlidingWindowRateLimiter, "_now", staticmethod(lambda: 1_000.0))
    first = SlidingWindowRateLimiter(global_orders_per_min=1, state_path=state_path)

    assert first.allow_and_record_order(" aapl ", RTH_BAR) == (True, None, {})

    monkeypatch.setattr(SlidingWindowRateLimiter, "_now", staticmethod(lambda: 1_001.0))
    second = SlidingWindowRateLimiter(global_orders_per_min=1, state_path=state_path)

    allowed, reason, details = second.allow_order("AAPL", RTH_BAR)
    assert allowed is False
    assert reason == "RATE_THROTTLE_BLOCK"
    assert details == {"scope": "global", "limit": 1}


def test_json_env_dict_accepts_valid_json_and_falls_back_on_bad_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "VALID": '{"aapl": "7.5", "bad": "not-a-number"}',
        "EMPTY": "",
        "BAD": "{",
    }
    monkeypatch.setattr(pretrade, "get_env", lambda name, default=None, **_kwargs: values.get(name, default))

    assert pretrade._json_env_dict("VALID", {"MSFT": 1.0}) == {"AAPL": 7.5}
    assert pretrade._json_env_dict("EMPTY", {"MSFT": 1.0}) == {"MSFT": 1.0}
    assert pretrade._json_env_dict("BAD", {"MSFT": 1.0}) == {"MSFT": 1.0}


def test_safe_validate_pretrade_fails_closed_on_unexpected_limiter_error() -> None:
    class BrokenLimiter:
        def allow_and_record_order(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str | None, dict[str, Any]]:
            raise RuntimeError("rate store offline")

    allowed, reason, details = pretrade.safe_validate_pretrade(
        _intent(),
        cfg=_cfg(),
        ledger=_Ledger(),
        rate_limiter=BrokenLimiter(),  # type: ignore[arg-type]
    )

    assert allowed is False
    assert reason == "PRETRADE_VALIDATION_ERROR"
    assert details["symbol"] == "AAPL"
    assert details["fail_closed"] is True


@pytest.mark.parametrize("qty", [0, -1])
def test_validate_pretrade_blocks_non_positive_qty(qty: int) -> None:
    allowed, reason, details = pretrade.validate_pretrade(
        _intent(qty=qty),
        cfg=_cfg(),
        ledger=_Ledger(),
        rate_limiter=SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100),
    )

    assert allowed is False
    assert reason == "INVALID_QTY_BLOCK"
    assert details["qty"] == qty


def test_stale_decay_artifact_blocks_live_canary_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PRETRADE_RUNTIME_DECAY_GATE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_RUNTIME_DECAY_CONTROLS_MAX_AGE_MINUTES", "1")
    monkeypatch.setenv("AI_TRADING_PRETRADE_LIVE_COST_MODEL_GATE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setattr(
        pretrade,
        "_runtime_decay_controls_payload",
        lambda: {
            "artifact_type": "runtime_decay_controls",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "status": {"available": True},
            "actions": {"entries_allowed": True, "max_action": "normal"},
        },
    )
    monkeypatch.setattr(pretrade, "_runtime_decay_controls_usable", lambda _payload: False)

    allowed, reason, details = pretrade.validate_pretrade(
        _intent(opening_trade=True),
        cfg=_cfg(execution_mode="live", launch_profile="live_canary"),
        ledger=_Ledger(),
        rate_limiter=SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100),
    )

    assert allowed is False
    assert reason == "RUNTIME_DECAY_ARTIFACT_STALE_BLOCK"
    assert details["opening_trade"] is True


def test_stale_live_cost_artifact_blocks_live_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PRETRADE_RUNTIME_DECAY_GATE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_RUNTIME_DECAY_CONTROLS_MAX_AGE_MINUTES", "0")
    monkeypatch.setenv("AI_TRADING_PRETRADE_LIVE_COST_MODEL_GATE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PRETRADE_LIVE_COST_MODEL_MAX_AGE_MINUTES", "1")
    monkeypatch.setattr(pretrade, "_runtime_decay_controls_payload", lambda: None)
    monkeypatch.setattr(
        pretrade,
        "_live_cost_model_payload",
        lambda: {
            "artifact_type": "live_cost_model",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "status": {"available": True, "status": "ready"},
            "by_symbol_side_session": [],
        },
    )
    monkeypatch.setattr(pretrade, "_live_cost_model_usable", lambda _payload: False)

    allowed, reason, details = pretrade.validate_pretrade(
        _intent(opening_trade=True),
        cfg=_cfg(execution_mode="live", launch_profile="live_trade"),
        ledger=_Ledger(),
        rate_limiter=SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100),
    )

    assert allowed is False
    assert reason == "LIVE_COST_ARTIFACT_STALE_BLOCK"
    assert details["opening_trade"] is True


def test_missing_live_cost_artifact_blocks_live_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PRETRADE_RUNTIME_DECAY_GATE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PRETRADE_LIVE_COST_MODEL_GATE_ENABLED", "1")
    monkeypatch.setattr(pretrade, "_runtime_decay_controls_payload", lambda: None)
    monkeypatch.setattr(pretrade, "_live_cost_model_payload", lambda: None)

    allowed, reason, details = pretrade.validate_pretrade(
        _intent(opening_trade=True),
        cfg=_cfg(execution_mode="live", launch_profile="live_canary"),
        ledger=_Ledger(),
        rate_limiter=SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100),
    )

    assert allowed is False
    assert reason == "LIVE_COST_ARTIFACT_MISSING_BLOCK"
    assert details["opening_trade"] is True


def test_canonical_daily_loss_limit_blocks_pretrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PRETRADE_RUNTIME_DECAY_GATE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_PRETRADE_LIVE_COST_MODEL_GATE_ENABLED", "0")
    ledger = _Ledger()
    ledger.daily_loss_pct = 0.06

    allowed, reason, details = pretrade.validate_pretrade(
        _intent(opening_trade=True),
        cfg=_cfg(daily_loss_limit=0.05, daily_loss_limit_pct=0.0),
        ledger=ledger,
        rate_limiter=SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100),
    )

    assert allowed is False
    assert reason == "DAILY_RISK_BUDGET_BLOCK"
    assert details["daily_loss_pct"] == pytest.approx(0.06)
    assert details["max_daily_loss_pct"] == pytest.approx(0.05)
