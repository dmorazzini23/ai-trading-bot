"""Rollout governance helpers for paper burn-in and phased live capital ramp."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping


def _as_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(raw: Any, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        value = int(default)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return int(value)


def _as_float(raw: Any, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(default)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return float(value)


def _normalize_mode(raw: Any) -> str:
    mode = str(raw or "").strip().lower()
    if mode in {"live", "paper", "sim"}:
        return mode
    return "sim"


def _parse_ramp_phases(raw: Any) -> tuple[float, ...]:
    default_phases = (0.25, 0.50, 0.75, 1.00)
    if raw in (None, ""):
        return default_phases
    parsed: list[float] = []
    for token in str(raw).split(","):
        item = str(token).strip()
        if not item:
            continue
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        value = max(0.05, min(value, 1.0))
        parsed.append(value)
    if not parsed:
        return default_phases
    deduped: list[float] = []
    for value in parsed:
        if value not in deduped:
            deduped.append(value)
    if deduped[-1] < 1.0:
        deduped.append(1.0)
    return tuple(deduped)


@dataclass(frozen=True)
class BurnInPolicy:
    enabled: bool
    min_paper_cycles: int
    min_paper_days: int
    require_policy_hash_stable: bool
    require_config_hash_stable: bool


@dataclass(frozen=True)
class CapitalRampPolicy:
    enabled: bool
    phases: tuple[float, ...]
    min_cycles_per_phase: int
    max_pacing_hit_rate_pct: float
    max_pending_oldest_age_sec: float
    max_calibration_ece: float
    max_calibration_brier: float
    downgrade_on_breach: bool


@dataclass
class RolloutState:
    burn_in_paper_cycles: int = 0
    burn_in_paper_days: tuple[str, ...] = ()
    burn_in_policy_hash: str = ""
    burn_in_config_hash: str = ""
    burn_in_reset_count: int = 0
    burn_in_last_reset_reason: str = ""
    ramp_phase_index: int = 0
    ramp_phase_cycles: int = 0
    ramp_multiplier: float = 1.0
    ramp_last_transition: str = ""
    updated_at: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RolloutState:
        days_raw = payload.get("burn_in_paper_days")
        days: tuple[str, ...] = ()
        if isinstance(days_raw, list):
            days = tuple(str(day).strip() for day in days_raw if str(day).strip())
        elif isinstance(days_raw, tuple):
            days = tuple(str(day).strip() for day in days_raw if str(day).strip())
        return cls(
            burn_in_paper_cycles=_as_int(payload.get("burn_in_paper_cycles"), 0, min_value=0),
            burn_in_paper_days=days,
            burn_in_policy_hash=str(payload.get("burn_in_policy_hash", "") or ""),
            burn_in_config_hash=str(payload.get("burn_in_config_hash", "") or ""),
            burn_in_reset_count=_as_int(payload.get("burn_in_reset_count"), 0, min_value=0),
            burn_in_last_reset_reason=str(payload.get("burn_in_last_reset_reason", "") or ""),
            ramp_phase_index=_as_int(payload.get("ramp_phase_index"), 0, min_value=0),
            ramp_phase_cycles=_as_int(payload.get("ramp_phase_cycles"), 0, min_value=0),
            ramp_multiplier=_as_float(payload.get("ramp_multiplier"), 1.0, min_value=0.05, max_value=1.0),
            ramp_last_transition=str(payload.get("ramp_last_transition", "") or ""),
            updated_at=str(payload.get("updated_at", "") or ""),
        )


def load_rollout_state(path: Path) -> RolloutState:
    if not path.exists() or not path.is_file():
        return RolloutState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return RolloutState()
    if not isinstance(payload, Mapping):
        return RolloutState()
    return RolloutState.from_mapping(payload)


def save_rollout_state(path: Path, state: RolloutState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def build_burn_in_policy(env: Mapping[str, Any]) -> BurnInPolicy:
    return BurnInPolicy(
        enabled=_as_bool(env.get("AI_TRADING_BURN_IN_ENABLED"), True),
        min_paper_cycles=_as_int(env.get("AI_TRADING_BURN_IN_MIN_PAPER_CYCLES"), 120, min_value=1, max_value=1_000_000),
        min_paper_days=_as_int(env.get("AI_TRADING_BURN_IN_MIN_PAPER_DAYS"), 3, min_value=1, max_value=365),
        require_policy_hash_stable=_as_bool(env.get("AI_TRADING_BURN_IN_REQUIRE_POLICY_HASH"), True),
        require_config_hash_stable=_as_bool(env.get("AI_TRADING_BURN_IN_REQUIRE_CONFIG_HASH"), True),
    )


def build_capital_ramp_policy(env: Mapping[str, Any]) -> CapitalRampPolicy:
    return CapitalRampPolicy(
        enabled=_as_bool(env.get("AI_TRADING_CAPITAL_RAMP_ENABLED"), True),
        phases=_parse_ramp_phases(env.get("AI_TRADING_CAPITAL_RAMP_PHASES")),
        min_cycles_per_phase=_as_int(
            env.get("AI_TRADING_CAPITAL_RAMP_MIN_CYCLES_PER_PHASE"),
            90,
            min_value=1,
            max_value=100_000,
        ),
        max_pacing_hit_rate_pct=_as_float(
            env.get("AI_TRADING_CAPITAL_RAMP_MAX_PACING_HIT_RATE_PCT"),
            30.0,
            min_value=0.0,
            max_value=100.0,
        ),
        max_pending_oldest_age_sec=_as_float(
            env.get("AI_TRADING_CAPITAL_RAMP_MAX_PENDING_OLDEST_AGE_SEC"),
            240.0,
            min_value=0.0,
            max_value=86_400.0,
        ),
        max_calibration_ece=_as_float(
            env.get("AI_TRADING_CAPITAL_RAMP_MAX_CALIBRATION_ECE"),
            0.15,
            min_value=0.0,
            max_value=1.0,
        ),
        max_calibration_brier=_as_float(
            env.get("AI_TRADING_CAPITAL_RAMP_MAX_CALIBRATION_BRIER"),
            0.35,
            min_value=0.0,
            max_value=1.0,
        ),
        downgrade_on_breach=_as_bool(
            env.get("AI_TRADING_CAPITAL_RAMP_DOWNGRADE_ON_BREACH"),
            True,
        ),
    )


def _evaluate_burn_in(
    *,
    state: RolloutState,
    policy: BurnInPolicy,
    mode: str,
    policy_hash: str,
    config_hash: str,
    today: date,
) -> tuple[RolloutState, bool, str]:
    if not policy.enabled:
        return state, True, ""

    next_state = RolloutState.from_mapping(asdict(state))
    if mode == "paper":
        policy_drift = (
            policy.require_policy_hash_stable
            and bool(next_state.burn_in_policy_hash)
            and next_state.burn_in_policy_hash != policy_hash
        )
        config_drift = (
            policy.require_config_hash_stable
            and bool(next_state.burn_in_config_hash)
            and next_state.burn_in_config_hash != config_hash
        )
        if policy_drift or config_drift:
            reasons: list[str] = []
            if policy_drift:
                reasons.append("policy_hash_changed")
            if config_drift:
                reasons.append("config_hash_changed")
            next_state.burn_in_paper_cycles = 0
            next_state.burn_in_paper_days = ()
            next_state.burn_in_reset_count = int(next_state.burn_in_reset_count) + 1
            next_state.burn_in_last_reset_reason = ",".join(reasons)
        next_state.burn_in_policy_hash = str(policy_hash)
        next_state.burn_in_config_hash = str(config_hash)
        next_state.burn_in_paper_cycles = int(next_state.burn_in_paper_cycles) + 1
        day_set = {day for day in next_state.burn_in_paper_days if day}
        day_set.add(today.isoformat())
        next_state.burn_in_paper_days = tuple(sorted(day_set))

    enough_cycles = int(next_state.burn_in_paper_cycles) >= int(policy.min_paper_cycles)
    enough_days = len(next_state.burn_in_paper_days) >= int(policy.min_paper_days)
    ready = bool(enough_cycles and enough_days)
    reason = ""
    if mode == "live":
        if policy.require_policy_hash_stable and next_state.burn_in_policy_hash != policy_hash:
            ready = False
            reason = "BURN_IN_POLICY_HASH_MISMATCH"
        elif policy.require_config_hash_stable and next_state.burn_in_config_hash != config_hash:
            ready = False
            reason = "BURN_IN_CONFIG_HASH_MISMATCH"
        elif not ready:
            reason = "BURN_IN_MIN_SAMPLE_NOT_MET"
    return next_state, ready, reason


def _evaluate_ramp(
    *,
    state: RolloutState,
    policy: CapitalRampPolicy,
    mode: str,
    telemetry: Mapping[str, Any],
) -> tuple[RolloutState, dict[str, Any]]:
    next_state = RolloutState.from_mapping(asdict(state))
    phases = policy.phases or (1.0,)
    phase_index = _as_int(next_state.ramp_phase_index, 0, min_value=0, max_value=len(phases) - 1)
    phase_cycles = _as_int(next_state.ramp_phase_cycles, 0, min_value=0)
    transition = ""
    breached = False

    if not policy.enabled:
        next_state.ramp_phase_index = 0
        next_state.ramp_phase_cycles = 0
        next_state.ramp_multiplier = 1.0
        next_state.ramp_last_transition = ""
        return next_state, {
            "enabled": False,
            "phase_index": 0,
            "phase_value": 1.0,
            "phase_cycles": 0,
            "multiplier": 1.0,
            "breached": False,
            "transition": "",
        }

    if mode == "live":
        pacing_hit_rate = _as_float(telemetry.get("order_pacing_cap_hit_rate_pct"), 0.0, min_value=0.0)
        pending_oldest_age = _as_float(telemetry.get("pending_oldest_age_sec"), 0.0, min_value=0.0)
        calibration_ece = _as_float(telemetry.get("live_calibration_ece"), 0.0, min_value=0.0)
        calibration_brier = _as_float(telemetry.get("live_calibration_brier"), 0.0, min_value=0.0)

        breached = bool(
            pacing_hit_rate >= policy.max_pacing_hit_rate_pct
            or pending_oldest_age >= policy.max_pending_oldest_age_sec
            or calibration_ece >= policy.max_calibration_ece
            or calibration_brier >= policy.max_calibration_brier
        )

        if breached and policy.downgrade_on_breach and phase_index > 0:
            phase_index -= 1
            phase_cycles = 0
            transition = "downgrade_on_breach"
        else:
            phase_cycles += 1
            if (
                not breached
                and phase_cycles >= policy.min_cycles_per_phase
                and phase_index < len(phases) - 1
            ):
                phase_index += 1
                phase_cycles = 0
                transition = "upgrade"
    else:
        phase_cycles = 0

    multiplier = float(phases[phase_index]) if mode == "live" else 1.0
    next_state.ramp_phase_index = int(phase_index)
    next_state.ramp_phase_cycles = int(phase_cycles)
    next_state.ramp_multiplier = float(max(0.05, min(multiplier, 1.0)))
    next_state.ramp_last_transition = transition
    return next_state, {
        "enabled": True,
        "phase_index": int(phase_index),
        "phase_value": float(phases[phase_index]),
        "phase_cycles": int(phase_cycles),
        "multiplier": float(next_state.ramp_multiplier),
        "breached": bool(breached),
        "transition": str(transition),
    }


def apply_rollout_policies(
    *,
    state: RolloutState,
    burn_in: BurnInPolicy,
    ramp: CapitalRampPolicy,
    execution_mode: str,
    policy_hash: str,
    config_hash: str,
    today: date,
    telemetry: Mapping[str, Any] | None = None,
) -> tuple[RolloutState, dict[str, Any]]:
    mode = _normalize_mode(execution_mode)
    telemetry_payload = telemetry if isinstance(telemetry, Mapping) else {}

    burn_state, burn_ready, burn_reason = _evaluate_burn_in(
        state=state,
        policy=burn_in,
        mode=mode,
        policy_hash=str(policy_hash or ""),
        config_hash=str(config_hash or ""),
        today=today,
    )
    ramp_state, ramp_summary = _evaluate_ramp(
        state=burn_state,
        policy=ramp,
        mode=mode,
        telemetry=telemetry_payload,
    )
    ramp_state.updated_at = datetime.now(UTC).isoformat()

    summary = {
        "execution_mode": mode,
        "burn_in_enabled": bool(burn_in.enabled),
        "burn_in_ready": bool(burn_ready),
        "burn_in_reason": str(burn_reason or ""),
        "burn_in_paper_cycles": int(ramp_state.burn_in_paper_cycles),
        "burn_in_paper_days": len(ramp_state.burn_in_paper_days),
        "burn_in_policy_hash": str(ramp_state.burn_in_policy_hash),
        "burn_in_config_hash": str(ramp_state.burn_in_config_hash),
        "burn_in_reset_count": int(ramp_state.burn_in_reset_count),
        "burn_in_last_reset_reason": str(ramp_state.burn_in_last_reset_reason or ""),
        "capital_ramp": dict(ramp_summary),
        "updated_at": str(ramp_state.updated_at),
    }
    return ramp_state, summary

