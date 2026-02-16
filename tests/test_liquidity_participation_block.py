from __future__ import annotations

from ai_trading.risk.liquidity_regime import enforce_participation_cap


def test_liquidity_participation_block_mode_blocks() -> None:
    allowed, qty, reason = enforce_participation_cap(
        order_qty=1000,
        rolling_volume=10000,
        max_participation_pct=0.05,  # cap 500
        mode="block",
        scale_min=0.25,
    )
    assert allowed is False
    assert qty == 1000
    assert reason == "LIQ_PARTICIPATION_BLOCK"


def test_liquidity_participation_scale_mode_scales() -> None:
    allowed, qty, reason = enforce_participation_cap(
        order_qty=1000,
        rolling_volume=10000,
        max_participation_pct=0.05,  # cap 500
        mode="scale",
        scale_min=0.25,
    )
    assert allowed is True
    assert abs(qty) <= 1000
    assert reason == "LIQ_REGIME_THIN_SCALE"
