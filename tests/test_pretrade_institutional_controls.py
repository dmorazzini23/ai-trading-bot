from __future__ import annotations

from datetime import UTC, datetime
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
    ) -> None:
        self._symbol_qty = {str(k).upper(): float(v) for k, v in (symbol_qty or {}).items()}
        self._gross_notional = float(gross_notional)
        self._sector_notional = {str(k): float(v) for k, v in (sector_notional or {}).items()}
        self._factor_exposure = {str(k): float(v) for k, v in (factor_exposure or {}).items()}
        self._intraday_var = float(intraday_var)
        self._intraday_cvar = float(intraday_cvar)
        self._current_drawdown = float(current_drawdown)

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


def _intent(
    *,
    symbol: str = "AAPL",
    side: str = "buy",
    qty: int = 1,
    price: float = 100.0,
    **kwargs,
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=side,
        qty=qty,
        notional=abs(float(qty) * float(price)),
        limit_price=price,
        bar_ts=datetime.now(UTC),
        client_order_id=f"{symbol.lower()}-{side}-{qty}",
        last_price=price,
        mid=price,
        **kwargs,
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
