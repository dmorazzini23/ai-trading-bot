"""Diagnostic paper-order sampling gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping

from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.runtime.atomic_io import atomic_write_text


_STATE_LOCK = RLock()


@dataclass(frozen=True)
class PaperSamplingDecision:
    enabled: bool
    allowed: bool
    qty: int
    reason: str
    details: dict[str, Any]


def _state_path() -> Path:
    return resolve_runtime_artifact_path(
        "runtime/paper_sampling_state_latest.json",
        default_relative="runtime/paper_sampling_state_latest.json",
        for_write=True,
    )


def _today_key(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(UTC).date().isoformat()


def _load_state(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(dict(state), sort_keys=True, separators=(",", ":")),
    )


def _allowed_symbols(cfg: Any) -> set[str]:
    raw_symbols = getattr(cfg, "paper_sampling_allowed_symbols", ("AAPL", "AMZN"))
    candidates: Iterable[Any]
    if isinstance(raw_symbols, str):
        candidates = raw_symbols.split(",")
    else:
        candidates = raw_symbols or ()
    symbols = {str(symbol).strip().upper() for symbol in candidates if str(symbol).strip()}
    return symbols or {"AAPL", "AMZN"}


def _is_paper_sampling_active(cfg: Any) -> tuple[bool, str | None]:
    if not bool(getattr(cfg, "paper_sampling_enabled", False)):
        return False, None
    execution_mode = str(getattr(cfg, "execution_mode", "sim") or "sim").strip().lower()
    paper_mode = bool(getattr(cfg, "paper", False))
    base_url = str(getattr(cfg, "alpaca_base_url", "") or "").strip().lower()
    launch_profile = str(
        getattr(
            cfg,
            "launch_profile",
            get_env("AI_TRADING_LAUNCH_PROFILE", "", cast=str),
        )
        or ""
    ).strip().lower()
    if execution_mode != "paper":
        return False, "execution_mode_not_paper"
    if not paper_mode or "paper" not in base_url:
        return False, "live_money_endpoint"
    if launch_profile == "live_canary" or launch_profile.startswith("live_"):
        return False, "live_launch_profile"
    return True, None


def evaluate_paper_sampling_order(
    cfg: Any,
    *,
    symbol: str,
    side: str,
    qty: int,
    price: float,
    consumes_daily_slot: bool = True,
) -> PaperSamplingDecision:
    """Apply diagnostic paper-sampling narrowing without bypassing hard gates."""

    active, inactive_reason = _is_paper_sampling_active(cfg)
    if not active:
        return PaperSamplingDecision(
            enabled=bool(getattr(cfg, "paper_sampling_enabled", False)),
            allowed=inactive_reason is None,
            qty=int(max(0, qty)),
            reason=inactive_reason or "paper_sampling_disabled",
            details={"reason": inactive_reason} if inactive_reason else {},
        )

    symbol_key = str(symbol).strip().upper()
    side_key = str(side).strip().lower()
    details: dict[str, Any] = {
        "symbol": symbol_key,
        "side": side_key,
        "mode": "paper_sampling",
        "consumes_daily_slot": bool(consumes_daily_slot),
    }
    if not bool(consumes_daily_slot):
        requested_qty = int(max(0, qty))
        details.update({"requested_qty": requested_qty, "adjusted_qty": requested_qty})
        return PaperSamplingDecision(True, True, requested_qty, "OK", details)
    if symbol_key not in _allowed_symbols(cfg):
        details["allowed_symbols"] = sorted(_allowed_symbols(cfg))
        return PaperSamplingDecision(True, False, 0, "PAPER_SAMPLING_SYMBOL_BLOCK", details)
    if side_key in {"short", "sell_short", "sell-short", "sell short"}:
        return PaperSamplingDecision(True, False, 0, "PAPER_SAMPLING_SHORT_BLOCK", details)

    requested_qty = int(max(0, qty))
    price_value = float(price)
    max_notional = float(getattr(cfg, "paper_sampling_max_notional_per_order", 250.0) or 250.0)
    if not math.isfinite(price_value) or price_value <= 0.0 or requested_qty <= 0:
        details.update({"qty": requested_qty, "price": price})
        return PaperSamplingDecision(True, False, 0, "PAPER_SAMPLING_INPUT_BLOCK", details)

    if not math.isfinite(max_notional) or max_notional <= 0.0:
        details.update({"qty": requested_qty, "price": price_value, "max_notional_per_order": max_notional})
        return PaperSamplingDecision(True, False, 0, "PAPER_SAMPLING_INPUT_BLOCK", details)
    if price_value > max_notional:
        details.update(
            {
                "requested_qty": requested_qty,
                "adjusted_qty": 0,
                "price": price_value,
                "max_notional_per_order": max_notional,
            }
        )
        return PaperSamplingDecision(True, False, 0, "PAPER_SAMPLING_MAX_NOTIONAL_BLOCK", details)

    max_qty = int(math.floor(max_notional / price_value))
    adjusted_qty = min(requested_qty, max_qty)
    details.update(
        {
            "requested_qty": requested_qty,
            "adjusted_qty": adjusted_qty,
            "price": price_value,
            "max_notional_per_order": max_notional,
            "one_share_fallback": bool(max_notional < price_value and adjusted_qty == 1),
        }
    )
    return PaperSamplingDecision(True, True, adjusted_qty, "OK", details)


def reserve_paper_sampling_order(
    cfg: Any,
    *,
    symbol: str,
    side: str,
    qty: int,
    price: float,
    now: datetime | None = None,
    consumes_daily_slot: bool = True,
) -> PaperSamplingDecision:
    """Reserve a diagnostic paper-sampling daily slot after upstream gates pass."""

    decision = evaluate_paper_sampling_order(
        cfg,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        consumes_daily_slot=consumes_daily_slot,
    )
    if not decision.enabled or not decision.allowed:
        return decision
    if not bool(consumes_daily_slot):
        return decision

    max_trades = int(getattr(cfg, "paper_sampling_max_trades_per_day", 2) or 2)
    today = _today_key(now)
    path = _state_path()
    with _STATE_LOCK:
        state = _load_state(path)
        state_date = str(state.get("date") or "")
        count = int(state.get("count", 0) or 0) if state_date == today else 0
        if count >= max_trades:
            details = dict(decision.details)
            details.update({"date": today, "count": count, "max_trades_per_day": max_trades})
            return PaperSamplingDecision(
                True,
                False,
                decision.qty,
                "PAPER_SAMPLING_DAILY_CAP_BLOCK",
                details,
            )
        state = {
            "artifact_type": "paper_sampling_state",
            "date": today,
            "count": count + 1,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _write_state(path, state)
    details = dict(decision.details)
    details.update({"date": today, "count": count + 1, "max_trades_per_day": max_trades})
    return PaperSamplingDecision(True, True, decision.qty, "OK", details)


__all__ = [
    "PaperSamplingDecision",
    "evaluate_paper_sampling_order",
    "reserve_paper_sampling_order",
]
