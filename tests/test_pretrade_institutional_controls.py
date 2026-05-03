from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_trading.oms.pretrade import (
    OrderIntent,
    SlidingWindowRateLimiter,
    safe_validate_pretrade,
    validate_pretrade,
)


class _ExposureLedger:
    def __init__(
        self,
        *,
        symbol_qty: dict[str, float] | None = None,
        gross_notional: float = 0.0,
        sector_notional: dict[str, float] | None = None,
        factor_exposure: dict[str, float] | None = None,
        intraday_var: float = 0.0,
        intraday_cvar: float = 0.0,
        current_drawdown: float = 0.0,
        daily_loss_pct: float = 0.0,
        daily_loss_abs: float = 0.0,
        reject_rate_pct: float = 0.0,
        execution_drift_bps: float = 0.0,
    ) -> None:
        self._symbol_qty = {str(k).upper(): float(v) for k, v in (symbol_qty or {}).items()}
        self._gross_notional = float(gross_notional)
        self._sector_notional = {str(k): float(v) for k, v in (sector_notional or {}).items()}
        self._factor_exposure = {str(k): float(v) for k, v in (factor_exposure or {}).items()}
        self._intraday_var = float(intraday_var)
        self._intraday_cvar = float(intraday_cvar)
        self._current_drawdown = float(current_drawdown)
        self._daily_loss_pct = float(daily_loss_pct)
        self._daily_loss_abs = float(daily_loss_abs)
        self._reject_rate_pct = float(reject_rate_pct)
        self._execution_drift_bps = float(execution_drift_bps)

    @staticmethod
    def seen_client_order_id(_value: str) -> bool:
        return False

    def position_qty(self, symbol: str) -> float:
        return float(self._symbol_qty.get(str(symbol).upper(), 0.0))

    def gross_notional(self) -> float:
        return float(self._gross_notional)

    def sector_notional(self, sector: str) -> float:
        return float(self._sector_notional.get(str(sector), 0.0))

    def factor_exposure(self, factor_name: str) -> float:
        return float(self._factor_exposure.get(str(factor_name), 0.0))

    def var_95(self) -> float:
        return float(self._intraday_var)

    def cvar_95(self) -> float:
        return float(self._intraday_cvar)

    def current_drawdown(self) -> float:
        return float(self._current_drawdown)

    def daily_loss_pct(self) -> float:
        return float(self._daily_loss_pct)

    def daily_loss_abs(self) -> float:
        return float(self._daily_loss_abs)

    def reject_rate_pct(self) -> float:
        return float(self._reject_rate_pct)

    def execution_drift_bps(self) -> float:
        return float(self._execution_drift_bps)


def _intent(
    *,
    symbol: str = "AAPL",
    side: str = "buy",
    qty: int = 1,
    price: float = 100.0,
    **kwargs,
) -> OrderIntent:
    bar_ts = kwargs.pop("bar_ts", datetime.now(UTC))
    return OrderIntent(
        symbol=symbol,
        side=side,
        qty=qty,
        notional=abs(float(qty) * float(price)),
        limit_price=price,
        bar_ts=bar_ts,
        client_order_id=f"{symbol.lower()}-{side}-{qty}",
        last_price=price,
        mid=price,
        **kwargs,
    )


def _write_live_cost_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_type": "live_cost_model",
                "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "status": {"available": True, "status": "ready"},
                "by_symbol_side_session": rows,
            }
        ),
        encoding="utf-8",
    )


def test_pretrade_blocks_projected_symbol_notional() -> None:
    cfg = SimpleNamespace(
        max_order_dollars=0.0,
        max_order_shares=0,
        price_collar_pct=0.10,
        max_symbol_notional=1000.0,
        max_gross_notional=0.0,
    )
    ledger = _ExposureLedger(symbol_qty={"AAPL": 8}, gross_notional=1200.0)
    intent = _intent(symbol="AAPL", side="buy", qty=5, price=100.0)
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "SYMBOL_NOTIONAL_BLOCK"
    assert details["symbol"] == "AAPL"
    assert details["projected_symbol_notional"] == pytest.approx(1300.0)


def test_pretrade_blocks_projected_gross_notional() -> None:
    cfg = SimpleNamespace(
        max_order_dollars=0.0,
        max_order_shares=0,
        price_collar_pct=0.10,
        max_symbol_notional=0.0,
        max_gross_notional=5000.0,
    )
    ledger = _ExposureLedger(symbol_qty={"AAPL": 2}, gross_notional=4900.0)
    intent = _intent(symbol="AAPL", side="buy", qty=1, price=200.0)
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "GROSS_NOTIONAL_BLOCK"
    assert details["projected_gross_notional"] == pytest.approx(5100.0)
    assert details["max_gross_notional"] == pytest.approx(5000.0)


def test_safe_validate_pretrade_fail_closed_blocks_on_gateway_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PRETRADE_FAIL_CLOSED", "1")
    intent = _intent(symbol="MSFT", side="buy", qty=1, price=300.0)
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)

    allowed, reason, details = safe_validate_pretrade(
        intent,
        cfg=cfg,
        ledger=_ExposureLedger(),
        rate_limiter=None,  # type: ignore[arg-type]
    )

    assert allowed is False
    assert reason == "PRETRADE_VALIDATION_ERROR"
    assert details["fail_closed"] is True


def test_safe_validate_pretrade_fail_open_allows_on_gateway_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PRETRADE_FAIL_CLOSED", "0")
    intent = _intent(symbol="MSFT", side="buy", qty=1, price=300.0)
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)

    allowed, reason, details = safe_validate_pretrade(
        intent,
        cfg=cfg,
        ledger=_ExposureLedger(),
        rate_limiter=None,  # type: ignore[arg-type]
    )

    assert allowed is True
    assert reason == "PRETRADE_VALIDATION_FAIL_OPEN"
    assert details["fail_closed"] is False


def test_pretrade_blocks_symbol_slippage_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_EXEC_SLIPPAGE_CEILING_BPS_BY_BUCKET", '{"NORMAL": 10}')
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger()
    intent = _intent(
        symbol="AAPL",
        side="buy",
        qty=10,
        price=100.0,
        expected_slippage_bps=15.0,
        liquidity_bucket="NORMAL",
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "SLIPPAGE_CEILING_BLOCK"
    assert details["ceiling_bps"] == pytest.approx(10.0)


def test_pretrade_blocks_default_hotspot_symbol_session_from_derived_spread() -> None:
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger()
    intent = _intent(
        symbol="BA",
        side="buy",
        qty=10,
        price=100.0,
        spread=0.09,
        bar_ts=datetime(2026, 4, 16, 17, 0, tzinfo=UTC),
        liquidity_bucket="NORMAL",
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "SLIPPAGE_CEILING_BLOCK"
    assert details["symbol"] == "BA"
    assert details["session_regime"] == "midday"
    assert details["expected_slippage_source"] == "derived_from_spread"
    assert details["expected_slippage_bps"] == pytest.approx(9.0)
    assert details["ceiling_bps"] == pytest.approx(6.0)


def test_pretrade_blocks_participation_adv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_EXEC_MAX_PARTICIPATION_PCT_ADV", "0.05")
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger()
    intent = _intent(symbol="MSFT", side="buy", qty=600, price=100.0, avg_daily_volume=10000.0)
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "PARTICIPATION_CAP_BLOCK"
    assert details["scope"] == "adv"


def test_pretrade_blocks_tca_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_EXEC_TCA_GATE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_TCA_MAX_EXPECTED_BPS", "20")
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger()
    intent = _intent(symbol="NVDA", side="buy", qty=10, price=100.0, expected_tca_bps=25.0)
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, _details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "TCA_GATE_BLOCK"


def test_pretrade_blocks_intraday_var_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_INTRADAY_VAR_LIMIT", "0.02")
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger(intraday_var=0.03)
    intent = _intent(symbol="AMZN", side="buy", qty=5, price=100.0)
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, _details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "INTRADAY_VAR_BLOCK"


def test_pretrade_derisk_blocks_on_data_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_DERISK_ON_DATA_DEGRADED", "1")
    monkeypatch.setenv("AI_TRADING_DERISK_MODE", "block")
    monkeypatch.setenv("AI_TRADING_DATA_DEGRADED", "1")
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger()
    intent = _intent(symbol="TSLA", side="buy", qty=1, price=100.0)
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "DERISK_DATA_QUALITY_BLOCK"
    assert details["data_degraded"] is True


def test_pretrade_blocks_daily_risk_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_DAILY_RISK_BUDGET_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DAILY_LOSS_LIMIT_PCT", "0.02")
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger(daily_loss_pct=0.03)
    intent = _intent(symbol="SPY", side="buy", qty=1, price=100.0)
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "DAILY_RISK_BUDGET_BLOCK"
    assert details["daily_loss_pct"] == pytest.approx(0.03)


def test_pretrade_blocks_event_risk_blackout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_EVENT_RISK_BLACKOUT_ENABLED", "1")
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger()
    intent = _intent(
        symbol="NVDA",
        side="buy",
        qty=2,
        price=100.0,
        event_risk=True,
        event_type="earnings",
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "EVENT_RISK_BLACKOUT_BLOCK"
    assert details["event_type"] == "earnings"


def test_pretrade_blocks_on_slo_derisk_breach(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_DERISK_ON_SLO_BREACH_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DERISK_SLO_MODE", "block")
    monkeypatch.setenv("AI_TRADING_DERISK_SLO_MAX_EXEC_DRIFT_BPS", "20")
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger(execution_drift_bps=25.0)
    intent = _intent(symbol="AAPL", side="buy", qty=1, price=100.0)
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "DERISK_SLO_BREACH_BLOCK"
    assert details["execution_drift_bps"] == pytest.approx(25.0)


def test_pretrade_blocks_kill_switch_at_final_boundary() -> None:
    cfg = SimpleNamespace(
        max_order_dollars=0.0,
        max_order_shares=0,
        price_collar_pct=0.10,
        kill_switch=False,
    )
    ledger = _ExposureLedger()
    intent = _intent(
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        bar_ts=datetime(2026, 4, 20, 14, 0, tzinfo=UTC),
        kill_switch_active=True,
        kill_switch_reason="operator_toggle",
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "KILL_SWITCH_BLOCK"
    assert details["reason"] == "operator_toggle"


def test_pretrade_blocks_broker_readiness_at_final_boundary() -> None:
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger()
    intent = _intent(
        symbol="MSFT",
        side="buy",
        qty=1,
        price=100.0,
        bar_ts=datetime(2026, 4, 20, 14, 0, tzinfo=UTC),
        broker_ready=False,
        broker_ready_reason="AUTH_BROKER_HALT_FORBIDDEN_COOLDOWN",
        broker_cooldown_remaining_sec=12.345,
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "AUTH_BROKER_HALT_FORBIDDEN_COOLDOWN"
    assert details["auth_forbidden_retry_after_sec"] == pytest.approx(12.345, abs=1e-3)


def test_pretrade_blocks_outside_market_hours_at_final_boundary() -> None:
    cfg = SimpleNamespace(
        max_order_dollars=0.0,
        max_order_shares=0,
        price_collar_pct=0.10,
        rth_only=True,
        allow_extended=False,
    )
    ledger = _ExposureLedger()
    intent = _intent(
        symbol="NVDA",
        side="buy",
        qty=1,
        price=100.0,
        bar_ts=datetime(2026, 4, 19, 14, 0, tzinfo=UTC),
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "MARKET_HOURS_BLOCK"
    assert details["rth_only"] is True


def test_pretrade_blocks_stale_quote_at_final_boundary() -> None:
    cfg = SimpleNamespace(
        max_order_dollars=0.0,
        max_order_shares=0,
        price_collar_pct=0.10,
        quote_max_age_ms=500,
    )
    ledger = _ExposureLedger()
    intent = _intent(
        symbol="TSLA",
        side="buy",
        qty=1,
        price=100.0,
        bar_ts=datetime(2026, 4, 20, 14, 0, tzinfo=UTC),
        submit_quote_source="broker_nbbo",
        quote_age_ms=750.0,
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "STALE_QUOTE_BLOCK"
    assert details["max_quote_age_ms"] == 500


def test_pretrade_blocks_symbol_specific_wide_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_PRETRADE_EXECUTION_QUALITY_GATE_ENABLED", "1")
    monkeypatch.setenv(
        "AI_TRADING_EXEC_MAX_SPREAD_BPS_BY_SYMBOL",
        json.dumps({"MSFT": 12.0}),
    )
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger()
    intent = _intent(
        symbol="MSFT",
        side="buy",
        qty=1,
        price=100.0,
        bar_ts=datetime(2026, 4, 20, 14, 0, tzinfo=UTC),
        bid=99.90,
        ask=100.10,
        submit_quote_source="broker_nbbo",
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "EXECUTION_QUALITY_SPREAD_BLOCK"
    assert details["block_reason"] == "spread_bps_too_wide"
    assert details["symbol"] == "MSFT"
    assert details["spread_bps"] == pytest.approx(20.0)
    assert details["max_spread_bps"] == 12.0


def test_pretrade_execution_quality_gate_allows_reducing_existing_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PRETRADE_EXECUTION_QUALITY_GATE_ENABLED", "1")
    monkeypatch.setenv(
        "AI_TRADING_EXEC_MAX_SPREAD_BPS_BY_SYMBOL",
        json.dumps({"AAPL": 1.0}),
    )
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger(symbol_qty={"AAPL": 5.0})
    intent = _intent(
        symbol="AAPL",
        side="sell",
        qty=2,
        price=100.0,
        bar_ts=datetime(2026, 4, 20, 14, 0, tzinfo=UTC),
        bid=99.90,
        ask=100.10,
        submit_quote_source="broker_nbbo",
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is True
    assert reason == "OK"
    assert details == {}


def test_pretrade_blocks_symbol_session_quote_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PRETRADE_EXECUTION_QUALITY_GATE_ENABLED", "1")
    monkeypatch.setenv(
        "AI_TRADING_EXEC_MAX_QUOTE_AGE_MS_BY_SYMBOL_SESSION",
        json.dumps({"MSFT:OPENING": 450.0}),
    )
    cfg = SimpleNamespace(
        max_order_dollars=0.0,
        max_order_shares=0,
        price_collar_pct=0.10,
        quote_max_age_ms=2000,
    )
    ledger = _ExposureLedger()
    intent = _intent(
        symbol="MSFT",
        side="buy",
        qty=1,
        price=100.0,
        bar_ts=datetime(2026, 4, 20, 14, 0, tzinfo=UTC),
        quote_age_ms=600.0,
        submit_quote_source="broker_nbbo",
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "EXECUTION_QUALITY_STALE_QUOTE_BLOCK"
    assert details["block_reason"] == "quote_age_too_stale"
    assert details["threshold_source"] == "MSFT:OPENING"
    assert details["quote_age_ms"] == 600.0
    assert details["max_quote_age_ms"] == 450.0


def test_pretrade_live_cost_model_blocks_above_adaptive_spread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "live_cost_model_latest.json"
    _write_live_cost_artifact(
        artifact,
        [
            {
                "symbol": "MSFT",
                "side": "buy",
                "session_regime": "opening",
                "sample_count": 8,
                "sufficient_samples": True,
                "p90_spread_bps": 10.0,
                "p90_total_cost_bps": 5.0,
            }
        ],
    )
    monkeypatch.setenv("AI_TRADING_LIVE_COST_MODEL_PATH", str(artifact))
    monkeypatch.setenv("AI_TRADING_PRETRADE_LIVE_COST_MODEL_SPREAD_MULTIPLIER", "1.0")
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger()
    intent = _intent(
        symbol="MSFT",
        side="buy",
        qty=1,
        price=100.0,
        bar_ts=datetime(2026, 4, 20, 14, 0, tzinfo=UTC),
        bid=99.925,
        ask=100.075,
        submit_quote_source="broker_nbbo",
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "EXECUTION_QUALITY_SPREAD_BLOCK"
    assert details["block_reason"] == "spread_bps_too_wide"
    assert details["threshold_source"] == "LIVE_COST_MODEL:MSFT:buy:opening"
    assert details["max_spread_bps"] == 10.0
    assert details["live_cost_sample_count"] == 8


def test_pretrade_live_cost_model_blocks_expensive_symbol_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "live_cost_model_latest.json"
    _write_live_cost_artifact(
        artifact,
        [
            {
                "symbol": "MSFT",
                "side": "buy",
                "session_regime": "opening",
                "sample_count": 12,
                "sufficient_samples": True,
                "mean_spread_bps": 2.0,
                "p90_total_cost_bps": 31.0,
                "mean_total_cost_bps": 28.0,
            }
        ],
    )
    monkeypatch.setenv("AI_TRADING_LIVE_COST_MODEL_PATH", str(artifact))
    monkeypatch.setenv("AI_TRADING_PRETRADE_LIVE_COST_MODEL_MAX_TOTAL_COST_BPS", "25")
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger()
    intent = _intent(
        symbol="MSFT",
        side="buy",
        qty=1,
        price=100.0,
        bar_ts=datetime(2026, 4, 20, 14, 0, tzinfo=UTC),
        bid=99.99,
        ask=100.01,
        submit_quote_source="broker_nbbo",
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "EXECUTION_QUALITY_LIVE_COST_BLOCK"
    assert details["block_reason"] == "live_cost_too_high"
    assert details["p90_total_cost_bps"] == 31.0
    assert details["max_total_cost_bps"] == 25.0
    assert details["sample_count"] == 12


def test_pretrade_live_cost_model_feeds_slippage_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "live_cost_model_latest.json"
    _write_live_cost_artifact(
        artifact,
        [
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "sample_count": 9,
                "sufficient_samples": True,
                "p90_adverse_slippage_bps": 15.0,
                "p90_total_cost_bps": 5.0,
            }
        ],
    )
    monkeypatch.setenv("AI_TRADING_LIVE_COST_MODEL_PATH", str(artifact))
    monkeypatch.setenv(
        "AI_TRADING_EXEC_SLIPPAGE_CEILING_BPS_BY_BUCKET",
        json.dumps({"NORMAL": 10.0}),
    )
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger()
    intent = _intent(
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        bar_ts=datetime(2026, 4, 20, 16, 0, tzinfo=UTC),
        liquidity_bucket="NORMAL",
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "SLIPPAGE_CEILING_BLOCK"
    assert details["expected_slippage_bps"] == 15.0
    assert details["expected_slippage_source"] == (
        "LIVE_COST_MODEL:AAPL:buy:midday:p90_adverse_slippage_bps"
    )


def test_pretrade_explicit_slippage_wins_over_live_cost_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "live_cost_model_latest.json"
    _write_live_cost_artifact(
        artifact,
        [
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "sample_count": 9,
                "sufficient_samples": True,
                "p90_adverse_slippage_bps": 50.0,
                "p90_total_cost_bps": 5.0,
            }
        ],
    )
    monkeypatch.setenv("AI_TRADING_LIVE_COST_MODEL_PATH", str(artifact))
    monkeypatch.setenv(
        "AI_TRADING_EXEC_SLIPPAGE_CEILING_BPS_BY_BUCKET",
        json.dumps({"NORMAL": 10.0}),
    )
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger()
    intent = _intent(
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        bar_ts=datetime(2026, 4, 20, 16, 0, tzinfo=UTC),
        liquidity_bucket="NORMAL",
        expected_slippage_bps=5.0,
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is True
    assert reason == "OK"


def test_pretrade_symbol_universe_gate_blocks_opening_but_allows_reduction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "symbol_universe_scorecard_latest.json"
    artifact.write_text(
        json.dumps(
            {
                "artifact_type": "symbol_universe_scorecard",
                "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "status": {"available": True, "status": "ready"},
                "symbols": [
                    {
                        "symbol": "MSFT",
                        "effective_mode": "disabled",
                        "sample_count": 30,
                        "persistence_count": 2,
                        "reasons": ["p90_total_cost_bps_disable"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_PRETRADE_SYMBOL_UNIVERSE_GATE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_SYMBOL_UNIVERSE_SCORECARD_PATH", str(artifact))
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(
        _intent(
            symbol="MSFT",
            side="buy",
            qty=1,
            price=100.0,
            bar_ts=datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
        ),
        cfg=cfg,
        ledger=_ExposureLedger(),
        rate_limiter=limiter,
    )

    assert allowed is False
    assert reason == "SYMBOL_UNIVERSE_MODE_BLOCK"
    assert details["mode"] == "disabled"

    allowed, reason, details = validate_pretrade(
        _intent(
            symbol="MSFT",
            side="sell",
            qty=1,
            price=100.0,
            bar_ts=datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
        ),
        cfg=cfg,
        ledger=_ExposureLedger(symbol_qty={"MSFT": 2.0}),
        rate_limiter=limiter,
    )

    assert allowed is True
    assert reason == "OK"
    assert details == {}
    assert details == {}


def test_pretrade_blocks_opening_without_realtime_nbbo() -> None:
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)
    ledger = _ExposureLedger()
    intent = _intent(
        symbol="SPY",
        side="buy",
        qty=1,
        price=100.0,
        bar_ts=datetime(2026, 4, 20, 14, 0, tzinfo=UTC),
        opening_trade=True,
        require_realtime_nbbo=True,
        submit_quote_source="last_trade",
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "NBBO_REQUIRED_OPENING_SKIP"
    assert details["opening_trade"] is True
