"""Policy compiler and runtime governance helpers.

This module centralizes policy compilation from canonical ``AI_TRADING_*`` keys
into a single immutable ``EffectivePolicy`` payload used by runtime decision and
execution gates.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from ai_trading.config.management import merged_env_snapshot


CANONICAL_MODE_KEY = "AI_TRADING_TRADING_MODE"

CANONICAL_KNOB_KEYS: dict[str, str] = {
    "kelly_fraction": "AI_TRADING_KELLY_FRACTION",
    "conf_threshold": "AI_TRADING_CONF_THRESHOLD",
    "daily_loss_limit": "AI_TRADING_DAILY_LOSS_LIMIT",
    "max_position_size": "AI_TRADING_SIGNAL_MAX_POSITION_SIZE",
    "capital_cap": "AI_TRADING_CAPITAL_CAP",
    "take_profit_factor": "AI_TRADING_TAKE_PROFIT_FACTOR",
}

OPTIONAL_KNOB_KEYS: dict[str, str] = {
    "buy_threshold": "AI_TRADING_BUY_THRESHOLD",
    "min_confidence": "AI_TRADING_MIN_CONFIDENCE",
    "kelly_fraction_max": "AI_TRADING_KELLY_FRACTION_MAX",
    "signal_confirmation_bars": "AI_TRADING_SIGNAL_CONFIRMATION_BARS",
}

REMOVED_LEGACY_KEYS: frozenset[str] = frozenset(
    {
        "TRADING_MODE",
        "TRADING_MODE_PRECEDENCE",
        "TRADING_MODE_ADAPTIVE_ENABLED",
        "KELLY_FRACTION",
        "CONF_THRESHOLD",
        "DAILY_LOSS_LIMIT",
        "MAX_POSITION_SIZE",
        "AI_TRADING_MAX_POSITION_SIZE",
        "CAPITAL_CAP",
        "TAKE_PROFIT_FACTOR",
        "BUY_THRESHOLD",
        "MIN_CONFIDENCE",
        "KELLY_FRACTION_MAX",
        "SIGNAL_CONFIRMATION_BARS",
    }
)

KNOWN_POLICY_KEYS: frozenset[str] = frozenset(
    {
        CANONICAL_MODE_KEY,
        *CANONICAL_KNOB_KEYS.values(),
        *OPTIONAL_KNOB_KEYS.values(),
        "AI_TRADING_POLICY_STRICT_CONFIG_GOVERNANCE",
        "AI_TRADING_POLICY_UNKNOWN_KEY_FAIL",
        "AI_TRADING_POLICY_FEE_BPS",
        "AI_TRADING_POLICY_BORROW_BPS",
        "AI_TRADING_POLICY_MIN_NET_EDGE_BPS",
        "AI_TRADING_POLICY_MAX_DRAWDOWN_PCT",
        "AI_TRADING_POLICY_MAX_EXPOSURE_PCT",
        "AI_TRADING_POLICY_PORTFOLIO_HARD_GROSS_DOLLARS",
        "AI_TRADING_POLICY_PORTFOLIO_SOFT_GROSS_DOLLARS",
        "AI_TRADING_POLICY_SLEEVE_HARD_DOLLARS",
        "AI_TRADING_POLICY_SLEEVE_SOFT_DOLLARS",
        "AI_TRADING_POLICY_SYMBOL_HARD_DOLLARS",
        "AI_TRADING_POLICY_SYMBOL_SOFT_DOLLARS",
        "AI_TRADING_POLICY_FACTOR_HARD_RATIO",
        "AI_TRADING_POLICY_FACTOR_SOFT_RATIO",
        "AI_TRADING_POLICY_EXEC_MAX_SPREAD_BPS",
        "AI_TRADING_POLICY_EXEC_MIN_ROLLING_VOLUME",
        "AI_TRADING_POLICY_EXEC_STALE_ORDER_BLOCK_SEC",
        "AI_TRADING_POLICY_EXEC_CALIBRATION_BLOCK",
        "AI_TRADING_POLICY_PROMOTION_MIN_OOS_SAMPLES",
        "AI_TRADING_POLICY_PROMOTION_MIN_OOS_NET_BPS",
        "AI_TRADING_POLICY_REPLAY_MIN_SAMPLES",
        "AI_TRADING_POLICY_REPLAY_NET_TOLERANCE_BPS",
        "AI_TRADING_POLICY_REPLAY_DRAWDOWN_TOLERANCE_PCT",
        "AI_TRADING_POLICY_SAFE_PENDING_AGE_SEC",
        "AI_TRADING_POLICY_SAFE_PACING_HIT_RATE_PCT",
        "AI_TRADING_POLICY_SAFE_ECE",
        "AI_TRADING_POLICY_SAFE_BRIER",
        "AI_TRADING_POLICY_ATTACK_MIN_NET_EDGE_BPS",
        "AI_TRADING_POLICY_ATTACK_REQUIRE_ZERO_PENDING",
        "AI_TRADING_POLICY_ATTACK_SIZE_MULTIPLIER",
        "AI_TRADING_POLICY_ATTACK_RELAX_REJECT_RATE_PCT",
        "AI_TRADING_POLICY_ATTACK_RELAX_SIZE_MULTIPLIER",
        "AI_TRADING_POLICY_ATTACK_DEGRADE_REJECT_RATE_PCT",
        "AI_TRADING_POLICY_ATTACK_DEGRADE_SIZE_MULTIPLIER",
    }
)


MODE_DEFAULTS: dict[str, dict[str, float]] = {
    "conservative": {
        "kelly_fraction": 0.25,
        "conf_threshold": 0.85,
        "daily_loss_limit": 0.03,
        "max_position_size": 5000.0,
        "capital_cap": 0.20,
        "take_profit_factor": 1.5,
    },
    "balanced": {
        "kelly_fraction": 0.60,
        "conf_threshold": 0.75,
        "daily_loss_limit": 0.05,
        "max_position_size": 8000.0,
        "capital_cap": 0.25,
        "take_profit_factor": 1.8,
    },
    "aggressive": {
        "kelly_fraction": 0.75,
        "conf_threshold": 0.65,
        "daily_loss_limit": 0.08,
        "max_position_size": 12000.0,
        "capital_cap": 0.30,
        "take_profit_factor": 2.5,
    },
}


class PolicyConfigError(RuntimeError):
    """Raised when effective-policy compilation fails strict governance checks."""


class SafetyTier(str, Enum):
    """Operational runtime tier."""

    SAFE = "safe"
    NORMAL = "normal"
    ATTACK = "attack"


@dataclass(frozen=True)
class ObjectivePolicy:
    objective_name: str
    min_expected_net_edge_bps: float
    fee_bps: float
    borrow_bps: float
    max_drawdown_pct: float
    max_exposure_pct: float


@dataclass(frozen=True)
class RiskBudgetPolicy:
    portfolio_hard_gross_dollars: float
    portfolio_soft_gross_dollars: float
    sleeve_hard_dollars: float
    sleeve_soft_dollars: float
    symbol_hard_dollars: float
    symbol_soft_dollars: float
    factor_hard_ratio: float
    factor_soft_ratio: float


@dataclass(frozen=True)
class CalibrationPolicy:
    max_ece_normal: float
    max_ece_stress: float
    max_brier_normal: float
    max_brier_stress: float
    min_samples: int


@dataclass(frozen=True)
class ExecutionPolicy:
    max_spread_bps: float
    min_rolling_volume: float
    stale_order_block_sec: float
    calibration_block_enabled: bool


@dataclass(frozen=True)
class GovernancePolicy:
    promotion_min_oos_samples: int
    promotion_min_oos_net_bps: float
    replay_min_samples: int
    replay_net_tolerance_bps: float
    replay_drawdown_tolerance_pct: float


@dataclass(frozen=True)
class SafetyPolicy:
    safe_pending_age_sec: float
    safe_pacing_hit_rate_pct: float
    safe_ece: float
    safe_brier: float
    attack_min_net_edge_bps: float
    attack_require_zero_pending: bool
    attack_size_multiplier: float
    attack_relax_reject_rate_pct: float
    attack_relax_size_multiplier: float
    attack_degrade_reject_rate_pct: float
    attack_degrade_size_multiplier: float


@dataclass(frozen=True)
class EffectivePolicy:
    """Immutable runtime policy payload."""

    compiled_at: str
    trading_mode: str
    knobs: tuple[tuple[str, Any], ...]
    objective: ObjectivePolicy
    risk_budgets: RiskBudgetPolicy
    calibration: CalibrationPolicy
    execution: ExecutionPolicy
    governance: GovernancePolicy
    safety: SafetyPolicy
    source_env: tuple[tuple[str, str], ...]
    policy_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "compiled_at": self.compiled_at,
            "trading_mode": self.trading_mode,
            "knobs": {k: v for k, v in self.knobs},
            "objective": self.objective.__dict__,
            "risk_budgets": self.risk_budgets.__dict__,
            "calibration": self.calibration.__dict__,
            "execution": self.execution.__dict__,
            "governance": self.governance.__dict__,
            "safety": self.safety.__dict__,
            "source_env": {k: v for k, v in self.source_env},
            "policy_hash": self.policy_hash,
        }

    def hash_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("policy_hash", None)
        return payload


@dataclass(frozen=True)
class ExecutionCandidate:
    """Candidate proposed by model/netting layer before execution approval."""

    symbol: str
    side: str
    proposed_delta_shares: int
    current_shares: int
    price: float
    expected_edge_bps: float
    expected_cost_bps: float
    confidence: float
    spread_bps: float
    rolling_volume: float
    pending_oldest_age_sec: float
    pacing_headroom: int
    stale_orders_present: bool
    calibration_ok: bool
    portfolio_post_gross_dollars: float
    sleeve_post_notional_dollars: float
    factor_post_ratio: float
    reject_rate_pct: float = 0.0
    safety_tier: SafetyTier = SafetyTier.NORMAL


@dataclass(frozen=True)
class ExecutionApproval:
    allowed: bool
    adjusted_delta_shares: int
    expected_net_edge_bps: float
    reasons: tuple[str, ...]


def _env_map(env: Mapping[str, str] | None = None) -> dict[str, str]:
    if env is None:
        return merged_env_snapshot()
    return {str(k): str(v) for k, v in env.items() if v is not None}


def _read_env_file(path: str | None) -> dict[str, str]:
    """Best-effort dotenv parser for policy startup diff diagnostics."""

    target = str(path or "").strip()
    if not target:
        return {}
    src = Path(target).expanduser()
    if not src.exists() or not src.is_file():
        return {}
    parsed: dict[str, str] = {}
    try:
        for raw_line in src.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            env_key = str(key).strip()
            if not env_key:
                continue
            value = str(raw_value).strip()
            if (
                len(value) >= 2
                and ((value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")))
            ):
                value = value[1:-1]
            parsed[env_key] = value
    except Exception:
        return {}
    return parsed


def _as_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(raw: Any, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return float(value)


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


def _normalize_mode(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value in MODE_DEFAULTS:
        return value
    return "balanced"


def _select_mode(cfg: Any, env_map: Mapping[str, str]) -> str:
    env_mode = env_map.get(CANONICAL_MODE_KEY)
    if env_mode not in (None, ""):
        return _normalize_mode(env_mode)
    return _normalize_mode(getattr(cfg, "trading_mode", "balanced"))


def _enforce_removed_keys(env_map: Mapping[str, str], *, strict: bool) -> None:
    violations = [key for key in REMOVED_LEGACY_KEYS if env_map.get(key) not in (None, "")]
    if not violations:
        return
    message = (
        "Legacy policy keys are removed; use canonical AI_TRADING_* keys only. "
        f"Found: {', '.join(sorted(violations))}"
    )
    if strict:
        raise PolicyConfigError(message)


def _enforce_unknown_policy_keys(env_map: Mapping[str, str], *, strict: bool) -> None:
    unknown = [
        key
        for key in env_map
        if key.startswith("AI_TRADING_POLICY_") and key not in KNOWN_POLICY_KEYS
    ]
    if not unknown:
        return
    message = f"Unknown AI_TRADING_POLICY_* keys: {', '.join(sorted(unknown))}"
    if strict:
        raise PolicyConfigError(message)


def _pull_knob(
    *,
    cfg: Any,
    env_map: Mapping[str, str],
    field: str,
    canonical_key: str,
    default_value: float,
    min_value: float,
    max_value: float,
) -> float:
    if canonical_key in env_map and env_map.get(canonical_key) not in (None, ""):
        return _as_float(env_map.get(canonical_key), default_value, min_value=min_value, max_value=max_value)
    return _as_float(getattr(cfg, field, default_value), default_value, min_value=min_value, max_value=max_value)


def compile_effective_policy(cfg: Any, env: Mapping[str, str] | None = None) -> EffectivePolicy:
    """Compile immutable policy from canonical env + runtime config."""

    env_values = _env_map(env)
    strict_governance = _as_bool(env_values.get("AI_TRADING_POLICY_STRICT_CONFIG_GOVERNANCE"), True)
    unknown_key_fail = _as_bool(env_values.get("AI_TRADING_POLICY_UNKNOWN_KEY_FAIL"), True)
    _enforce_removed_keys(env_values, strict=strict_governance)
    _enforce_unknown_policy_keys(env_values, strict=(strict_governance and unknown_key_fail))

    mode = _select_mode(cfg, env_values)
    base = MODE_DEFAULTS[mode]

    knobs: dict[str, Any] = {
        "kelly_fraction": _pull_knob(
            cfg=cfg,
            env_map=env_values,
            field="kelly_fraction",
            canonical_key=CANONICAL_KNOB_KEYS["kelly_fraction"],
            default_value=base["kelly_fraction"],
            min_value=0.0,
            max_value=1.0,
        ),
        "conf_threshold": _pull_knob(
            cfg=cfg,
            env_map=env_values,
            field="conf_threshold",
            canonical_key=CANONICAL_KNOB_KEYS["conf_threshold"],
            default_value=base["conf_threshold"],
            min_value=0.0,
            max_value=1.0,
        ),
        "daily_loss_limit": _pull_knob(
            cfg=cfg,
            env_map=env_values,
            field="daily_loss_limit",
            canonical_key=CANONICAL_KNOB_KEYS["daily_loss_limit"],
            default_value=base["daily_loss_limit"],
            min_value=0.0,
            max_value=1.0,
        ),
        "max_position_size": _pull_knob(
            cfg=cfg,
            env_map=env_values,
            field="max_position_size",
            canonical_key=CANONICAL_KNOB_KEYS["max_position_size"],
            default_value=base["max_position_size"],
            min_value=1.0,
            max_value=5_000_000.0,
        ),
        "capital_cap": _pull_knob(
            cfg=cfg,
            env_map=env_values,
            field="capital_cap",
            canonical_key=CANONICAL_KNOB_KEYS["capital_cap"],
            default_value=base["capital_cap"],
            min_value=0.0,
            max_value=1.0,
        ),
        "take_profit_factor": _pull_knob(
            cfg=cfg,
            env_map=env_values,
            field="take_profit_factor",
            canonical_key=CANONICAL_KNOB_KEYS["take_profit_factor"],
            default_value=base["take_profit_factor"],
            min_value=0.1,
            max_value=20.0,
        ),
        "buy_threshold": _as_float(
            env_values.get(OPTIONAL_KNOB_KEYS["buy_threshold"], getattr(cfg, "buy_threshold", 0.2)),
            0.2,
            min_value=0.0,
            max_value=1.0,
        ),
        "min_confidence": _as_float(
            env_values.get(OPTIONAL_KNOB_KEYS["min_confidence"], getattr(cfg, "min_confidence", 0.5)),
            0.5,
            min_value=0.0,
            max_value=1.0,
        ),
        "kelly_fraction_max": _as_float(
            env_values.get(OPTIONAL_KNOB_KEYS["kelly_fraction_max"], getattr(cfg, "kelly_fraction_max", 1.0)),
            1.0,
            min_value=0.0,
            max_value=1.0,
        ),
        "signal_confirmation_bars": _as_int(
            env_values.get(
                OPTIONAL_KNOB_KEYS["signal_confirmation_bars"],
                getattr(cfg, "signal_confirmation_bars", 1),
            ),
            1,
            min_value=1,
            max_value=20,
        ),
    }

    objective = ObjectivePolicy(
        objective_name="expected_net_edge_after_fees_slippage_borrow",
        min_expected_net_edge_bps=_as_float(
            env_values.get("AI_TRADING_POLICY_MIN_NET_EDGE_BPS"),
            2.0,
            min_value=0.0,
            max_value=1_000.0,
        ),
        fee_bps=_as_float(
            env_values.get("AI_TRADING_POLICY_FEE_BPS"),
            0.4,
            min_value=0.0,
            max_value=100.0,
        ),
        borrow_bps=_as_float(
            env_values.get("AI_TRADING_POLICY_BORROW_BPS"),
            0.2,
            min_value=0.0,
            max_value=100.0,
        ),
        max_drawdown_pct=_as_float(
            env_values.get("AI_TRADING_POLICY_MAX_DRAWDOWN_PCT", getattr(cfg, "max_drawdown_threshold", 0.2)),
            0.2,
            min_value=0.0,
            max_value=1.0,
        ),
        max_exposure_pct=_as_float(
            env_values.get("AI_TRADING_POLICY_MAX_EXPOSURE_PCT", getattr(cfg, "capital_cap", 0.25)),
            0.25,
            min_value=0.0,
            max_value=1.0,
        ),
    )

    risk_budgets = RiskBudgetPolicy(
        portfolio_hard_gross_dollars=_as_float(
            env_values.get("AI_TRADING_POLICY_PORTFOLIO_HARD_GROSS_DOLLARS", getattr(cfg, "global_max_gross_dollars", 150_000.0)),
            150_000.0,
            min_value=0.0,
            max_value=100_000_000.0,
        ),
        portfolio_soft_gross_dollars=_as_float(
            env_values.get("AI_TRADING_POLICY_PORTFOLIO_SOFT_GROSS_DOLLARS", getattr(cfg, "global_max_gross_dollars", 150_000.0) * 0.85),
            127_500.0,
            min_value=0.0,
            max_value=100_000_000.0,
        ),
        sleeve_hard_dollars=_as_float(
            env_values.get("AI_TRADING_POLICY_SLEEVE_HARD_DOLLARS", 60_000.0),
            60_000.0,
            min_value=0.0,
            max_value=100_000_000.0,
        ),
        sleeve_soft_dollars=_as_float(
            env_values.get("AI_TRADING_POLICY_SLEEVE_SOFT_DOLLARS", 45_000.0),
            45_000.0,
            min_value=0.0,
            max_value=100_000_000.0,
        ),
        symbol_hard_dollars=_as_float(
            env_values.get("AI_TRADING_POLICY_SYMBOL_HARD_DOLLARS", getattr(cfg, "global_max_symbol_dollars", 25_000.0)),
            25_000.0,
            min_value=0.0,
            max_value=10_000_000.0,
        ),
        symbol_soft_dollars=_as_float(
            env_values.get("AI_TRADING_POLICY_SYMBOL_SOFT_DOLLARS", getattr(cfg, "global_max_symbol_dollars", 25_000.0) * 0.75),
            18_750.0,
            min_value=0.0,
            max_value=10_000_000.0,
        ),
        factor_hard_ratio=_as_float(
            env_values.get("AI_TRADING_POLICY_FACTOR_HARD_RATIO", 0.40),
            0.40,
            min_value=0.0,
            max_value=1.0,
        ),
        factor_soft_ratio=_as_float(
            env_values.get("AI_TRADING_POLICY_FACTOR_SOFT_RATIO", 0.30),
            0.30,
            min_value=0.0,
            max_value=1.0,
        ),
    )

    calibration = CalibrationPolicy(
        max_ece_normal=_as_float(env_values.get("AI_TRADING_POLICY_SAFE_ECE"), 0.12, min_value=0.0, max_value=1.0),
        max_ece_stress=_as_float(env_values.get("AI_TRADING_POLICY_SAFE_ECE"), 0.16, min_value=0.0, max_value=1.0),
        max_brier_normal=_as_float(env_values.get("AI_TRADING_POLICY_SAFE_BRIER"), 0.35, min_value=0.0, max_value=1.0),
        max_brier_stress=_as_float(env_values.get("AI_TRADING_POLICY_SAFE_BRIER"), 0.45, min_value=0.0, max_value=1.0),
        min_samples=_as_int(env_values.get("AI_TRADING_POLICY_PROMOTION_MIN_OOS_SAMPLES"), 30, min_value=1, max_value=100000),
    )

    execution = ExecutionPolicy(
        max_spread_bps=_as_float(
            env_values.get("AI_TRADING_POLICY_EXEC_MAX_SPREAD_BPS"),
            30.0,
            min_value=0.0,
            max_value=10_000.0,
        ),
        min_rolling_volume=_as_float(
            env_values.get("AI_TRADING_POLICY_EXEC_MIN_ROLLING_VOLUME"),
            100.0,
            min_value=0.0,
            max_value=10_000_000.0,
        ),
        stale_order_block_sec=_as_float(
            env_values.get("AI_TRADING_POLICY_EXEC_STALE_ORDER_BLOCK_SEC"),
            180.0,
            min_value=0.0,
            max_value=86_400.0,
        ),
        calibration_block_enabled=_as_bool(
            env_values.get("AI_TRADING_POLICY_EXEC_CALIBRATION_BLOCK"),
            True,
        ),
    )

    governance = GovernancePolicy(
        promotion_min_oos_samples=_as_int(
            env_values.get("AI_TRADING_POLICY_PROMOTION_MIN_OOS_SAMPLES"),
            40,
            min_value=1,
            max_value=1_000_000,
        ),
        promotion_min_oos_net_bps=_as_float(
            env_values.get("AI_TRADING_POLICY_PROMOTION_MIN_OOS_NET_BPS"),
            0.0,
            min_value=-1_000.0,
            max_value=1_000.0,
        ),
        replay_min_samples=_as_int(
            env_values.get("AI_TRADING_POLICY_REPLAY_MIN_SAMPLES"),
            100,
            min_value=1,
            max_value=1_000_000,
        ),
        replay_net_tolerance_bps=_as_float(
            env_values.get("AI_TRADING_POLICY_REPLAY_NET_TOLERANCE_BPS"),
            2.0,
            min_value=0.0,
            max_value=1_000.0,
        ),
        replay_drawdown_tolerance_pct=_as_float(
            env_values.get("AI_TRADING_POLICY_REPLAY_DRAWDOWN_TOLERANCE_PCT"),
            0.01,
            min_value=0.0,
            max_value=1.0,
        ),
    )

    safety = SafetyPolicy(
        safe_pending_age_sec=_as_float(
            env_values.get("AI_TRADING_POLICY_SAFE_PENDING_AGE_SEC"),
            180.0,
            min_value=0.0,
            max_value=86_400.0,
        ),
        safe_pacing_hit_rate_pct=_as_float(
            env_values.get("AI_TRADING_POLICY_SAFE_PACING_HIT_RATE_PCT"),
            20.0,
            min_value=0.0,
            max_value=100.0,
        ),
        safe_ece=_as_float(
            env_values.get("AI_TRADING_POLICY_SAFE_ECE"),
            0.15,
            min_value=0.0,
            max_value=1.0,
        ),
        safe_brier=_as_float(
            env_values.get("AI_TRADING_POLICY_SAFE_BRIER"),
            0.40,
            min_value=0.0,
            max_value=1.0,
        ),
        attack_min_net_edge_bps=_as_float(
            env_values.get("AI_TRADING_POLICY_ATTACK_MIN_NET_EDGE_BPS"),
            8.0,
            min_value=0.0,
            max_value=10_000.0,
        ),
        attack_require_zero_pending=_as_bool(
            env_values.get("AI_TRADING_POLICY_ATTACK_REQUIRE_ZERO_PENDING"),
            True,
        ),
        attack_size_multiplier=_as_float(
            env_values.get("AI_TRADING_POLICY_ATTACK_SIZE_MULTIPLIER"),
            1.15,
            min_value=1.0,
            max_value=3.0,
        ),
        attack_relax_reject_rate_pct=_as_float(
            env_values.get("AI_TRADING_POLICY_ATTACK_RELAX_REJECT_RATE_PCT"),
            2.0,
            min_value=0.0,
            max_value=100.0,
        ),
        attack_relax_size_multiplier=_as_float(
            env_values.get("AI_TRADING_POLICY_ATTACK_RELAX_SIZE_MULTIPLIER"),
            1.05,
            min_value=1.0,
            max_value=3.0,
        ),
        attack_degrade_reject_rate_pct=_as_float(
            env_values.get("AI_TRADING_POLICY_ATTACK_DEGRADE_REJECT_RATE_PCT"),
            8.0,
            min_value=0.0,
            max_value=100.0,
        ),
        attack_degrade_size_multiplier=_as_float(
            env_values.get("AI_TRADING_POLICY_ATTACK_DEGRADE_SIZE_MULTIPLIER"),
            0.85,
            min_value=0.25,
            max_value=3.0,
        ),
    )

    source_pairs = sorted(
        (key, value)
        for key, value in env_values.items()
        if key in KNOWN_POLICY_KEYS and value not in (None, "")
    )
    policy = EffectivePolicy(
        compiled_at=datetime.now(UTC).isoformat(),
        trading_mode=mode,
        knobs=tuple(sorted(knobs.items(), key=lambda item: item[0])),
        objective=objective,
        risk_budgets=risk_budgets,
        calibration=calibration,
        execution=execution,
        governance=governance,
        safety=safety,
        source_env=tuple(source_pairs),
    )
    digest = hashlib.sha256(
        json.dumps(policy.hash_payload(), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return replace(policy, policy_hash=digest)


def startup_policy_diff(
    policy: EffectivePolicy,
    env: Mapping[str, str] | None = None,
    *,
    env_file: str | None = None,
) -> list[dict[str, Any]]:
    """Return startup diff of provided env values vs compiled effective policy."""

    env_values = _env_map(env)
    env_file_path = env_file
    if env_file_path in (None, ""):
        env_file_path = str(env_values.get("ENV_LOADED_FROM", "") or "").strip() or ".env"
    file_values = _read_env_file(env_file_path)
    compare_values = file_values if file_values else env_values
    source = "env_file" if file_values else "process_env"
    compiled = policy.to_dict().get("knobs", {})
    diffs: list[dict[str, Any]] = []
    for logical_key, canonical_key in {**CANONICAL_KNOB_KEYS, **OPTIONAL_KNOB_KEYS}.items():
        raw = compare_values.get(canonical_key)
        if raw in (None, ""):
            continue
        effective = compiled.get(logical_key)
        if str(effective) != str(raw):
            diffs.append(
                {
                    "key": canonical_key,
                    "provided": raw,
                    "effective": effective,
                    "source": source,
                }
            )
    raw_mode = compare_values.get(CANONICAL_MODE_KEY)
    if raw_mode not in (None, "") and _normalize_mode(raw_mode) != policy.trading_mode:
        diffs.append(
            {
                "key": CANONICAL_MODE_KEY,
                "provided": raw_mode,
                "effective": policy.trading_mode,
                "source": source,
            }
        )
    return diffs


def compute_expected_net_edge_bps(
    expected_edge_bps: float,
    expected_cost_bps: float,
    *,
    fee_bps: float = 0.0,
    borrow_bps: float = 0.0,
) -> float:
    """Return expected net edge after direct costs."""

    edge = max(0.0, float(expected_edge_bps))
    cost = max(0.0, float(expected_cost_bps))
    fee = max(0.0, float(fee_bps))
    borrow = max(0.0, float(borrow_bps))
    return edge - cost - fee - borrow


def _scale_qty(delta: int, scale: float) -> int:
    if delta == 0:
        return 0
    scaled = int(round(float(delta) * max(0.0, scale)))
    if scaled == 0:
        return 0
    if delta > 0:
        return max(1, scaled)
    return min(-1, scaled)


def approve_execution_candidate(policy: EffectivePolicy, candidate: ExecutionCandidate) -> ExecutionApproval:
    """Approve/reject model-proposed execution candidate under effective policy."""

    reasons: list[str] = []
    qty = int(candidate.proposed_delta_shares)
    net_edge_bps = compute_expected_net_edge_bps(
        candidate.expected_edge_bps,
        candidate.expected_cost_bps,
        fee_bps=policy.objective.fee_bps,
        borrow_bps=policy.objective.borrow_bps,
    )
    if qty == 0:
        return ExecutionApproval(False, 0, net_edge_bps, ("ZERO_QTY",))

    if net_edge_bps < policy.objective.min_expected_net_edge_bps:
        reasons.append("NET_EDGE_FLOOR_BLOCK")

    if candidate.pacing_headroom <= 0:
        reasons.append("ORDER_PACING_CAP_BLOCK")

    if candidate.stale_orders_present and candidate.pending_oldest_age_sec >= policy.execution.stale_order_block_sec:
        reasons.append("STALE_ORDER_BLOCK")

    if candidate.spread_bps > policy.execution.max_spread_bps:
        reasons.append("SPREAD_TOO_WIDE_BLOCK")

    if candidate.rolling_volume < policy.execution.min_rolling_volume:
        proposed_abs = abs(candidate.current_shares + qty)
        current_abs = abs(candidate.current_shares)
        if proposed_abs > current_abs:
            reasons.append("LIQUIDITY_VOLUME_BLOCK")

    if policy.execution.calibration_block_enabled and not candidate.calibration_ok:
        reasons.append("CALIBRATION_BLOCK")

    proposed_notional = abs(float(candidate.current_shares + qty) * float(candidate.price))
    if proposed_notional > policy.risk_budgets.symbol_hard_dollars:
        reasons.append("RISK_SYMBOL_HARD_BLOCK")
    elif proposed_notional > policy.risk_budgets.symbol_soft_dollars:
        soft_scale = policy.risk_budgets.symbol_soft_dollars / max(proposed_notional, 1e-9)
        qty = _scale_qty(qty, soft_scale)
        reasons.append("RISK_SYMBOL_SOFT_THROTTLE")

    if candidate.portfolio_post_gross_dollars > policy.risk_budgets.portfolio_hard_gross_dollars:
        reasons.append("RISK_PORTFOLIO_HARD_BLOCK")
    elif candidate.portfolio_post_gross_dollars > policy.risk_budgets.portfolio_soft_gross_dollars:
        soft_scale = policy.risk_budgets.portfolio_soft_gross_dollars / max(candidate.portfolio_post_gross_dollars, 1e-9)
        qty = _scale_qty(qty, soft_scale)
        reasons.append("RISK_PORTFOLIO_SOFT_THROTTLE")

    if candidate.sleeve_post_notional_dollars > policy.risk_budgets.sleeve_hard_dollars:
        reasons.append("RISK_SLEEVE_HARD_BLOCK")
    elif candidate.sleeve_post_notional_dollars > policy.risk_budgets.sleeve_soft_dollars:
        soft_scale = policy.risk_budgets.sleeve_soft_dollars / max(candidate.sleeve_post_notional_dollars, 1e-9)
        qty = _scale_qty(qty, soft_scale)
        reasons.append("RISK_SLEEVE_SOFT_THROTTLE")

    if candidate.factor_post_ratio > policy.risk_budgets.factor_hard_ratio:
        reasons.append("RISK_FACTOR_HARD_BLOCK")
    elif candidate.factor_post_ratio > policy.risk_budgets.factor_soft_ratio:
        span = max(policy.risk_budgets.factor_hard_ratio - policy.risk_budgets.factor_soft_ratio, 1e-9)
        progress = (candidate.factor_post_ratio - policy.risk_budgets.factor_soft_ratio) / span
        soft_scale = max(0.1, 1.0 - progress)
        qty = _scale_qty(qty, soft_scale)
        reasons.append("RISK_FACTOR_SOFT_THROTTLE")

    if candidate.safety_tier is SafetyTier.SAFE:
        proposed_abs = abs(candidate.current_shares + qty)
        current_abs = abs(candidate.current_shares)
        if proposed_abs > current_abs:
            reasons.append("SAFETY_TIER_SAFE_BLOCK")
    elif candidate.safety_tier is SafetyTier.ATTACK and qty != 0:
        attack_scale = float(policy.safety.attack_size_multiplier)
        reject_rate = max(0.0, float(candidate.reject_rate_pct))
        relax_threshold = max(0.0, float(policy.safety.attack_relax_reject_rate_pct))
        relax_scale = max(1.0, float(policy.safety.attack_relax_size_multiplier))
        degrade_threshold = max(
            0.0,
            float(policy.safety.attack_degrade_reject_rate_pct),
        )
        degrade_scale = max(0.25, min(float(policy.safety.attack_degrade_size_multiplier), attack_scale))
        attack_scale_reason: str | None = None
        if reject_rate <= relax_threshold:
            bounded_relax_scale = min(attack_scale, relax_scale)
            if bounded_relax_scale < attack_scale - 1e-9:
                attack_scale = bounded_relax_scale
                attack_scale_reason = "SAFETY_TIER_ATTACK_SCALE_RELAXED"
        elif reject_rate >= degrade_threshold:
            bounded_degrade_scale = max(0.25, min(attack_scale, degrade_scale))
            if bounded_degrade_scale < attack_scale - 1e-9:
                attack_scale = bounded_degrade_scale
                attack_scale_reason = "SAFETY_TIER_ATTACK_SCALE_DEGRADED"
        qty_before_attack_scale = int(qty)
        qty = _scale_qty(qty, attack_scale)
        if qty != qty_before_attack_scale:
            reasons.append("SAFETY_TIER_ATTACK_SCALE")
            if attack_scale_reason:
                reasons.append(attack_scale_reason)

    if qty == 0 and "ZERO_QTY" not in reasons:
        reasons.append("ZERO_QTY")
    hard_blocks = [r for r in reasons if r.endswith("_BLOCK")]
    if hard_blocks:
        return ExecutionApproval(False, 0, net_edge_bps, tuple(reasons))
    if qty == 0:
        return ExecutionApproval(False, 0, net_edge_bps, tuple(reasons))
    return ExecutionApproval(True, qty, net_edge_bps, tuple(reasons))


def resolve_operational_safety_tier(
    policy: EffectivePolicy,
    telemetry: Mapping[str, Any],
    *,
    previous: SafetyTier = SafetyTier.NORMAL,
) -> tuple[SafetyTier, tuple[str, ...]]:
    """Resolve runtime safety tier with automatic downgrade triggers."""

    reasons: list[str] = []
    pending_oldest_age = _as_float(telemetry.get("pending_oldest_age_sec"), 0.0, min_value=0.0)
    pacing_hit_rate = _as_float(telemetry.get("order_pacing_cap_hit_rate_pct"), 0.0, min_value=0.0)
    ece = _as_float(telemetry.get("live_calibration_ece"), 0.0, min_value=0.0)
    brier = _as_float(telemetry.get("live_calibration_brier"), 0.0, min_value=0.0)
    pending_count = _as_int(telemetry.get("pending_orders_count"), 0, min_value=0)
    net_edge = _as_float(telemetry.get("expected_net_edge_bps"), 0.0)

    if pending_oldest_age >= policy.safety.safe_pending_age_sec:
        reasons.append("SAFE_TRIGGER_PENDING_AGE")
    if pacing_hit_rate >= policy.safety.safe_pacing_hit_rate_pct:
        reasons.append("SAFE_TRIGGER_PACING")
    if ece >= policy.safety.safe_ece:
        reasons.append("SAFE_TRIGGER_ECE")
    if brier >= policy.safety.safe_brier:
        reasons.append("SAFE_TRIGGER_BRIER")
    if _as_bool(telemetry.get("kill_switch"), False):
        reasons.append("SAFE_TRIGGER_KILL_SWITCH")
    if reasons:
        return (SafetyTier.SAFE, tuple(reasons))

    attack_reasons: list[str] = []
    if net_edge >= policy.safety.attack_min_net_edge_bps:
        attack_reasons.append("ATTACK_NET_EDGE_OK")
    if ece < policy.safety.safe_ece and brier < policy.safety.safe_brier:
        attack_reasons.append("ATTACK_CALIBRATION_OK")
    if (not policy.safety.attack_require_zero_pending) or pending_count == 0:
        attack_reasons.append("ATTACK_PENDING_OK")
    if len(attack_reasons) == 3:
        return (SafetyTier.ATTACK, tuple(attack_reasons))

    if previous is SafetyTier.SAFE and pending_oldest_age > 0:
        # Hysteresis: avoid flapping out of SAFE until stale/pending state clears.
        return (SafetyTier.SAFE, ("SAFE_HYSTERESIS",))
    return (SafetyTier.NORMAL, ("NORMAL_TIER",))


def decompose_tca_components(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Decompose slippage into spread, impact, and timing components.

    Includes optional symbol and hour-of-day cost profiles so runtime gating can
    downweight pockets with persistently poor realized execution.
    """

    spread_vals: list[float] = []
    impact_vals: list[float] = []
    timing_vals: list[float] = []
    total_vals: list[float] = []
    symbol_totals: dict[str, list[float]] = {}
    hour_totals: dict[str, list[float]] = {}
    for row in records:
        try:
            total = abs(float(row.get("is_bps", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(total):
            continue
        try:
            spread = abs(float(row.get("spread_paid_bps", 0.0) or 0.0))
        except (TypeError, ValueError):
            spread = 0.0
        try:
            latency_ms = float(row.get("fill_latency_ms", 0.0) or 0.0)
        except (TypeError, ValueError):
            latency_ms = 0.0
        latency_penalty = max(0.0, min(25.0, latency_ms / 80.0))
        impact = max(0.0, total - spread - latency_penalty)
        spread_vals.append(spread)
        impact_vals.append(impact)
        timing_vals.append(latency_penalty)
        total_vals.append(total)
        symbol_raw = str(row.get("symbol", "") or "").strip().upper()
        if symbol_raw:
            symbol_totals.setdefault(symbol_raw, []).append(total)
        ts_raw = row.get("ts")
        if ts_raw in (None, ""):
            benchmark_raw = row.get("benchmark")
            if isinstance(benchmark_raw, Mapping):
                ts_raw = benchmark_raw.get("submit_ts") or benchmark_raw.get("decision_ts")
        if ts_raw not in (None, ""):
            ts_text = str(ts_raw).strip()
            if ts_text.endswith("Z"):
                ts_text = f"{ts_text[:-1]}+00:00"
            try:
                parsed_ts = datetime.fromisoformat(ts_text)
            except ValueError:
                parsed_ts = None
            if parsed_ts is not None:
                if parsed_ts.tzinfo is None:
                    parsed_ts = parsed_ts.replace(tzinfo=UTC)
                hour_key = f"{parsed_ts.astimezone(UTC).hour:02d}"
                hour_totals.setdefault(hour_key, []).append(total)
    if not total_vals:
        return {
            "sample_count": 0.0,
            "spread_bps": 0.0,
            "impact_bps": 0.0,
            "timing_bps": 0.0,
            "total_bps": 0.0,
            "by_symbol_total_bps": {},
            "by_hour_total_bps": {},
        }

    symbol_profile: dict[str, dict[str, float]] = {}
    for symbol, values in symbol_totals.items():
        if not values:
            continue
        symbol_profile[symbol] = {
            "median_bps": float(median(values)),
            "samples": float(len(values)),
        }

    hour_profile: dict[str, dict[str, float]] = {}
    for hour_key, values in hour_totals.items():
        if not values:
            continue
        hour_profile[hour_key] = {
            "median_bps": float(median(values)),
            "samples": float(len(values)),
        }

    return {
        "sample_count": float(len(total_vals)),
        "spread_bps": float(median(spread_vals)),
        "impact_bps": float(median(impact_vals)),
        "timing_bps": float(median(timing_vals)),
        "total_bps": float(median(total_vals)),
        "by_symbol_total_bps": symbol_profile,
        "by_hour_total_bps": hour_profile,
    }


def evaluate_counterfactual_non_regression(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    min_samples: int,
    net_tolerance_bps: float,
    drawdown_tolerance_pct: float,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate non-regression constraint for policy replay promotion."""

    base_samples = _as_int(baseline.get("sample_count"), 0, min_value=0)
    cand_samples = _as_int(candidate.get("sample_count"), 0, min_value=0)
    base_net = _as_float(baseline.get("net_edge_bps"), 0.0)
    cand_net = _as_float(candidate.get("net_edge_bps"), 0.0)
    base_dd = _as_float(baseline.get("max_drawdown_pct"), 0.0, min_value=0.0)
    cand_dd = _as_float(candidate.get("max_drawdown_pct"), 0.0, min_value=0.0)

    checks = {
        "sample_size": cand_samples >= min_samples,
        "net_edge_non_regression": cand_net + net_tolerance_bps >= base_net,
        "drawdown_non_regression": cand_dd <= (base_dd + drawdown_tolerance_pct),
    }
    return (
        all(checks.values()),
        {
            "checks": checks,
            "baseline": {
                "sample_count": base_samples,
                "net_edge_bps": base_net,
                "max_drawdown_pct": base_dd,
            },
            "candidate": {
                "sample_count": cand_samples,
                "net_edge_bps": cand_net,
                "max_drawdown_pct": cand_dd,
            },
            "required": {
                "min_samples": int(min_samples),
                "net_tolerance_bps": float(net_tolerance_bps),
                "drawdown_tolerance_pct": float(drawdown_tolerance_pct),
            },
        },
    )
