from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.oms.pretrade import (
    OrderIntent,
    SlidingWindowRateLimiter,
    safe_validate_pretrade,
    validate_pretrade,
)


RTH_BAR = datetime(2026, 4, 21, 14, 30, tzinfo=UTC)


class _Ledger:
    positions = {"AAPL": 0.0}
    gross_notional = 0.0
    sector_notional_map = {"TECH": 0.0}
    factor_exposure_map = {"MOMENTUM": 0.0}
    intraday_var = 0.0
    intraday_cvar = 0.0
    current_drawdown = 0.0
    daily_loss_pct = 0.0
    daily_loss_abs = 0.0
    execution_drift_bps = 0.0
    reject_rate_pct = 0.0

    def __init__(self, **updates: Any) -> None:
        for key, value in updates.items():
            setattr(self, key, value)

    @staticmethod
    def seen_client_order_id(_value: str) -> bool:
        return False


def _cfg(**updates: Any) -> SimpleNamespace:
    base = {
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
    base.update(updates)
    return SimpleNamespace(**base)


def _intent(**updates: Any) -> OrderIntent:
    base = {
        "symbol": "AAPL",
        "side": "buy",
        "qty": 10,
        "notional": 1_000.0,
        "limit_price": 100.0,
        "bar_ts": RTH_BAR,
        "client_order_id": "cid-pretrade",
        "last_price": 100.0,
        "mid": 100.0,
        "bid": 99.95,
        "ask": 100.05,
        "spread": 0.10,
        "submit_quote_source": "broker_nbbo",
        "quote_quality_ok": True,
        "liquidity_bucket": "NORMAL",
    }
    base.update(updates)
    return OrderIntent(**base)


def _limiter(**updates: Any) -> SlidingWindowRateLimiter:
    base = {"global_orders_per_min": 100, "per_symbol_orders_per_min": 100}
    base.update(updates)
    return SlidingWindowRateLimiter(**base)


def _validate(
    intent: OrderIntent,
    *,
    cfg: SimpleNamespace | None = None,
    ledger: Any | None = None,
    rate_limiter: SlidingWindowRateLimiter | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    return validate_pretrade(
        intent,
        cfg=cfg or _cfg(),
        ledger=ledger if ledger is not None else _Ledger(),
        rate_limiter=rate_limiter or _limiter(),
    )


@pytest.mark.parametrize(
    ("intent_updates", "reason", "detail_key"),
    [
        (
            {"kill_switch_active": True, "kill_switch_reason": "manual_halt"},
            "KILL_SWITCH_BLOCK",
            "reason",
        ),
        (
            {"broker_ready": False, "broker_ready_reason": "BROKER_AUTH_BLOCK", "broker_cooldown_remaining_sec": 7.25},
            "BROKER_AUTH_BLOCK",
            "auth_forbidden_retry_after_sec",
        ),
        ({"bid": 101.0, "ask": 100.0}, "QUOTE_SANITY_BLOCK", "bid"),
        ({"quote_age_ms": 2_500.0}, "STALE_QUOTE_BLOCK", "quote_age_ms"),
        (
            {"opening_trade": True, "require_realtime_nbbo": True, "submit_quote_source": "delayed_iex"},
            "NBBO_REQUIRED_OPENING_SKIP",
            "quote_source",
        ),
        ({"event_risk": True, "event_type": "earnings"}, "EVENT_RISK_BLACKOUT_BLOCK", "event_type"),
        ({"quote_quality_ok": False}, "DERISK_DATA_QUALITY_BLOCK", "quote_quality_ok"),
        ({"reject_rate_pct": 9.0}, "DERISK_SLO_BREACH_BLOCK", "reject_rate_pct"),
        ({"execution_drift_bps": 80.0}, "DERISK_SLO_BREACH_BLOCK", "execution_drift_bps"),
        ({"expected_slippage_bps": 50.0}, "SLIPPAGE_CEILING_BLOCK", "expected_slippage_bps"),
        ({"expected_tca_bps": 80.0}, "TCA_GATE_BLOCK", "expected_tca_bps"),
        ({"fill_quality_score": 0.10}, "FILL_QUALITY_GATE_BLOCK", "fill_quality_score"),
    ],
)
def test_pretrade_blocks_extended_intent_controls(
    intent_updates: dict[str, Any],
    reason: str,
    detail_key: str,
) -> None:
    cfg_updates = {"quote_max_age_ms": 1_000} if reason == "STALE_QUOTE_BLOCK" else {}

    allowed, actual_reason, details = _validate(
        _intent(**intent_updates),
        cfg=_cfg(**cfg_updates),
    )

    assert allowed is False
    assert actual_reason == reason
    assert detail_key in details


@pytest.mark.parametrize(
    ("cfg_updates", "ledger", "reason", "detail_key"),
    [
        (
            {"max_symbol_notional": 1_500.0},
            _Ledger(positions={"AAPL": 10.0}),
            "SYMBOL_NOTIONAL_BLOCK",
            "projected_symbol_notional",
        ),
        (
            {"max_gross_notional": 1_500.0},
            _Ledger(positions={"AAPL": 5.0}, gross_notional=1_400.0),
            "GROSS_NOTIONAL_BLOCK",
            "projected_gross_notional",
        ),
        (
            {"max_sector_notional": 1_250.0},
            _Ledger(sector_notional_map={"TECH": 400.0}),
            "SECTOR_CONCENTRATION_BLOCK",
            "projected_sector_notional",
        ),
        (
            {"max_factor_exposure": 0.20},
            _Ledger(factor_exposure_map={"MOMENTUM": 0.15}),
            "FACTOR_CONCENTRATION_BLOCK",
            "projected_factor_exposure",
        ),
        (
            {"intraday_var_limit": 0.05},
            _Ledger(intraday_var=0.07),
            "INTRADAY_VAR_BLOCK",
            "intraday_var",
        ),
        (
            {"intraday_cvar_limit": 0.05},
            _Ledger(intraday_cvar=0.08),
            "INTRADAY_CVAR_BLOCK",
            "intraday_cvar",
        ),
        (
            {"intraday_drawdown_limit": 0.05},
            _Ledger(current_drawdown=0.06),
            "INTRADAY_DRAWDOWN_BLOCK",
            "intraday_drawdown",
        ),
        (
            {"daily_loss_limit_pct": 0.03},
            _Ledger(daily_loss_pct=0.03),
            "DAILY_RISK_BUDGET_BLOCK",
            "daily_loss_pct",
        ),
        (
            {"daily_loss_limit_abs": 500.0},
            _Ledger(daily_loss_abs=500.0),
            "DAILY_RISK_BUDGET_BLOCK",
            "daily_loss_abs",
        ),
    ],
)
def test_pretrade_blocks_projected_portfolio_and_intraday_limits(
    cfg_updates: dict[str, Any],
    ledger: _Ledger,
    reason: str,
    detail_key: str,
) -> None:
    allowed, actual_reason, details = _validate(
        _intent(sector="TECH", factor_name="MOMENTUM", factor_exposure=0.10),
        cfg=_cfg(**cfg_updates),
        ledger=ledger,
    )

    assert allowed is False
    assert actual_reason == reason
    assert detail_key in details


@pytest.mark.parametrize(
    ("intent_updates", "reason", "scope"),
    [
        ({"avg_daily_volume": 100.0, "qty": 10}, "PARTICIPATION_CAP_BLOCK", "adv"),
        ({"minute_volume": 30.0, "qty": 10}, "PARTICIPATION_CAP_BLOCK", "minute"),
    ],
)
def test_pretrade_blocks_participation_caps(
    intent_updates: dict[str, Any],
    reason: str,
    scope: str,
) -> None:
    allowed, actual_reason, details = _validate(_intent(**intent_updates))

    assert allowed is False
    assert actual_reason == reason
    assert details["scope"] == scope


def test_pretrade_blocks_custom_event_blackout_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_EVENT_BLACKOUT_WINDOWS_ET", "10:00-11:00")

    allowed, reason, details = _validate(_intent())

    assert allowed is False
    assert reason == "EVENT_RISK_BLACKOUT_BLOCK"
    assert details["reason"] == "custom_window:10:00-11:00"


def test_pretrade_records_fingerprint_after_success_and_blocks_duplicate() -> None:
    ledger = _Ledger()
    intent = _intent(client_order_id="cid-one")

    first = _validate(intent, ledger=ledger)
    second = _validate(intent, ledger=ledger)

    assert first == (True, "OK", {})
    assert second[0] is False
    assert second[1] == "DUPLICATE_ORDER_BLOCK"
    assert second[2]["fingerprint"][0] == "AAPL"


def test_safe_validate_pretrade_fail_open_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenLimiter:
        def allow_order(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str | None, dict[str, Any]]:
            raise RuntimeError("limiter offline")

    monkeypatch.setenv("AI_TRADING_PRETRADE_FAIL_CLOSED", "0")

    allowed, reason, details = safe_validate_pretrade(
        _intent(),
        cfg=_cfg(),
        ledger=_Ledger(),
        rate_limiter=_BrokenLimiter(),  # type: ignore[arg-type]
    )

    assert allowed is True
    assert reason == "PRETRADE_VALIDATION_FAIL_OPEN"
    assert details["fail_closed"] is False


def test_pretrade_quote_age_from_timestamp_blocks_stale_quote() -> None:
    quote_ts = datetime.now(UTC) - timedelta(seconds=3)

    allowed, reason, details = _validate(
        _intent(quote_age_ms=None, quote_ts=quote_ts),
        cfg=_cfg(quote_max_age_ms=1_000),
    )

    assert allowed is False
    assert reason == "STALE_QUOTE_BLOCK"
    assert details["quote_ts"] == quote_ts.isoformat()
