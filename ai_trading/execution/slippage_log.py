"""
Lightweight slippage/cost logging with per-symbol EWMA feedback.

Designed to be import-light and safe in constrained environments.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import csv
import math
from typing import Optional

from ai_trading.paths import SLIPPAGE_LOG_PATH
from ai_trading.logging import get_logger

logger = get_logger(__name__)


@dataclass
class _EwmaState:
    value: float
    count: int


_EWMA_BPS: dict[str, _EwmaState] = {}
_ALPHA: float = 0.2  # default smoothing factor


def _to_bps(expected: float | None, fill: float | None, side: str | None) -> Optional[float]:
    try:
        e = float(expected) if expected is not None else None
        f = float(fill) if fill is not None else None
    except (TypeError, ValueError):
        return None
    if e is None or f is None or e <= 0:
        return None
    s = str(side or "").lower()
    # Positive bps is favorable; convert to cost bps where higher means worse
    if s.startswith("buy"):
        slippage = (f - e) / e
    else:  # sell or unknown
        slippage = (e - f) / e
    return float(slippage * 10000.0)


def _update_ewma(symbol: str, bps: float, alpha: float | None = None) -> float:
    a = float(_ALPHA if alpha is None else alpha)
    st = _EWMA_BPS.get(symbol)
    if st is None:
        st = _EwmaState(value=bps, count=1)
    else:
        st.value = a * bps + (1.0 - a) * st.value
        st.count += 1
    _EWMA_BPS[symbol] = st
    return st.value


def record_fill(
    *,
    symbol: str,
    side: str,
    qty: int | float,
    expected_price: float | None,
    fill_price: float | None,
    timestamp: datetime | None = None,
) -> None:
    """Append a slippage observation and update EWMA in memory.

    Writes a compact CSV row to :data:`SLIPPAGE_LOG_PATH` and updates the in‑memory
    EWMA used for cost estimation. Failures are swallowed to avoid affecting
    trading paths.
    """
    ts = (timestamp or datetime.now(UTC)).isoformat()
    try:
        bps = _to_bps(expected_price, fill_price, side)
        if bps is None or not math.isfinite(bps):
            return
        _update_ewma(symbol, abs(bps))
        # Best-effort append; file may be on read-only FS in some environments
        try:
            SLIPPAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with SLIPPAGE_LOG_PATH.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerow([ts, symbol, side, int(qty or 0), f"{expected_price}", f"{fill_price}", f"{bps:.4f}"])
        except Exception:
            # Logging to disk is best-effort only.
            pass
    except Exception:
        logger.debug("SLIPPAGE_LOG_FAILED", exc_info=True)


def get_ewma_cost_bps(symbol: str, default: float = 2.0) -> float:
    """Return the current EWMA cost bps for ``symbol``.

    Defaults to ``default`` when no observations exist yet.
    """
    st = _EWMA_BPS.get(symbol)
    try:
        return float(abs(st.value)) if st is not None else float(default)
    except Exception:
        return float(default)

