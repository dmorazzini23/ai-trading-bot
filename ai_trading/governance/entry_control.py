"""Canonical replay-readiness control for exposure-increasing orders."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

REPLAY_LIVE_ENTRY_CONTROL_ATTR = "replay_live_entry_control"
REPLAY_LIVE_PARITY_GATE_ATTR = "replay_live_parity_gate"
REPLAY_LIVE_PARITY_GATE_FAILED = "REPLAY_LIVE_PARITY_GATE_FAILED"

_LIVE_EXECUTION_MODES = frozenset({"live", "live_canary", "canary"})
_PAPER_SIM_EXECUTION_MODES = frozenset({"paper", "sim", "simulation"})


def normalize_entry_control_execution_mode(value: Any) -> str:
    """Return the stable execution-mode token used by replay entry control."""

    token = str(value or "paper").strip().lower()
    if token == "simulation":
        return "sim"
    return token or "paper"


def build_replay_live_entry_control(
    *,
    gate: Mapping[str, Any] | None,
    required: bool,
    execution_mode: Any,
) -> dict[str, Any]:
    """Build the mode-aware execution contract for a replay gate snapshot."""

    gate_snapshot = dict(gate or {})
    mode = normalize_entry_control_execution_mode(execution_mode)
    enabled = bool(gate_snapshot.get("enabled", False))
    available = bool(gate_snapshot.get("available", False))
    gate_ok = bool(gate_snapshot.get("ok", False))
    required_failure = bool(
        required and (not enabled or not available or not gate_ok)
    )
    live_mode = mode in _LIVE_EXECUTION_MODES
    monitor_only = bool(required_failure and not live_mode)
    block_exposure_increasing = bool(required_failure and live_mode)

    if block_exposure_increasing:
        status = "blocked"
        enforcement = "fail_closed"
        reason: str | None = REPLAY_LIVE_PARITY_GATE_FAILED
    elif monitor_only:
        status = "monitor_only"
        enforcement = "monitor_only"
        reason = str(gate_snapshot.get("reason") or "replay_live_parity_gate_failed")
    else:
        status = "ready" if enabled else "disabled"
        enforcement = "allow"
        reason = None

    return {
        "version": 1,
        "status": status,
        "reason": reason,
        "execution_mode": mode,
        "live_mode": live_mode,
        "paper_or_sim_mode": mode in _PAPER_SIM_EXECUTION_MODES,
        "required": bool(required),
        "gate_enabled": enabled,
        "gate_available": available,
        "gate_ok": gate_ok,
        "gate_status": gate_snapshot.get("status"),
        "gate_reason": gate_snapshot.get("reason"),
        "failed_checks": list(gate_snapshot.get("failed_checks") or ()),
        "enforcement": enforcement,
        "monitor_only": monitor_only,
        "block_exposure_increasing": block_exposure_increasing,
        "exposure_increasing_allowed": not block_exposure_increasing,
        "reductions_allowed": True,
    }


def evaluate_replay_live_order(
    *,
    control: Mapping[str, Any] | None,
    side: Any,
    requested_quantity: Any,
    current_position_quantity: Any,
    closing_position: bool,
) -> dict[str, Any]:
    """Return a replay-control decision, clipping reductions before zero-crossing."""

    snapshot = dict(control or {})
    try:
        requested = int(requested_quantity)
    except (TypeError, ValueError, OverflowError):
        requested = 0

    base = {
        "control_status": str(snapshot.get("status") or "unavailable"),
        "execution_mode": normalize_entry_control_execution_mode(
            snapshot.get("execution_mode")
        ),
        "monitor_only": bool(snapshot.get("monitor_only", False)),
        "closing_position": bool(closing_position),
        "requested_quantity": requested,
        "effective_quantity": requested,
        "clamped": False,
        "reduction": False,
        "gate_reason": snapshot.get("gate_reason"),
        "failed_checks": list(snapshot.get("failed_checks") or ()),
    }
    if not bool(snapshot.get("block_exposure_increasing", False)):
        return base | {"allowed": True, "reason": None}

    side_token = str(getattr(side, "value", side) or "").strip().lower()
    if side_token in {"buy_to_cover", "buy-to-cover", "buy to cover"}:
        side_token = "cover"
    if side_token in {"buy", "cover"}:
        direction = 1
    elif side_token in {"sell", "sell_short", "short"}:
        direction = -1
    else:
        return base | {
            "allowed": False,
            "reason": REPLAY_LIVE_PARITY_GATE_FAILED,
            "block_detail": "invalid_side",
        }

    try:
        position = float(current_position_quantity)
    except (TypeError, ValueError, OverflowError):
        position = math.nan
    if not math.isfinite(position):
        return base | {
            "allowed": False,
            "reason": REPLAY_LIVE_PARITY_GATE_FAILED,
            "block_detail": "position_unavailable",
        }
    if requested <= 0:
        return base | {
            "allowed": False,
            "reason": REPLAY_LIVE_PARITY_GATE_FAILED,
            "block_detail": "quantity_unavailable",
            "current_position_quantity": position,
        }

    reduces_position = bool(
        (position > 0.0 and direction < 0)
        or (position < 0.0 and direction > 0)
    )
    if not reduces_position:
        return base | {
            "allowed": False,
            "reason": REPLAY_LIVE_PARITY_GATE_FAILED,
            "block_detail": "exposure_increasing_order",
            "current_position_quantity": position,
        }

    maximum_reduction = int(math.floor(abs(position)))
    effective_quantity = min(requested, maximum_reduction)
    if effective_quantity <= 0:
        return base | {
            "allowed": False,
            "reason": REPLAY_LIVE_PARITY_GATE_FAILED,
            "block_detail": "no_non_flipping_reduction_quantity",
            "current_position_quantity": position,
        }
    return base | {
        "allowed": True,
        "reason": None,
        "block_detail": None,
        "reduction": True,
        "current_position_quantity": position,
        "maximum_reduction_quantity": maximum_reduction,
        "effective_quantity": effective_quantity,
        "clamped": effective_quantity != requested,
    }
