"""Early symbol-level prelude orchestration for the live netting cycle."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Collection, Mapping

from ai_trading.config.management import get_env


@dataclass(frozen=True, slots=True)
class NettingSymbolPreludeResult:
    delta_shares: int
    target_shares: int
    target_dollars: float
    gates_added: tuple[str, ...]
    snapshot_updates: dict[str, Any]
    blocked_reason: str | None


def prepare_netting_symbol_prelude(
    *,
    state: Any,
    symbol: str,
    now: datetime,
    current_shares: int,
    delta_shares: int,
    price: float,
    net_target: Any,
    policy_disabled_sleeves: set[str],
    policy_rollback_disabled_slices: Collection[str],
    sleeve_configs_map: Mapping[str, Any],
    candidate_expected_net_edge: Mapping[str, Any],
    alpha_time_stop_enabled: bool,
    alpha_time_stop_sec: float,
    alpha_time_stop_max_expected_edge_bps: float,
    opportunity_quality_enabled: bool,
    opportunity_allowed_symbols: set[str],
    opportunity_openings_only: bool,
    opportunity_quality_by_symbol: Mapping[str, Any],
    opportunity_quality_gate: Mapping[str, Any],
    opportunity_top_quantile: float,
    alpha_time_decay_enabled: bool,
    alpha_stale_signal_sec: float,
    live_execution_mode: bool,
    burn_in_live_ready: bool,
    burn_in_live_reason: str,
    ramp_live_multiplier: float,
    ramp_summary: Mapping[str, Any],
    gates: list[str],
    position_opened_at_func: Callable[..., Any],
    exit_policy_pressure_context_func: Callable[..., Mapping[str, Any]],
) -> NettingSymbolPreludeResult:
    gates_added: list[str] = []
    snapshot_updates: dict[str, Any] = {}
    delta_shares_value = int(delta_shares)
    target_shares = int(current_shares + delta_shares_value)
    target_dollars = float(target_shares * price)

    if policy_disabled_sleeves:
        disabled_for_symbol = sorted(
            {
                str(proposal.sleeve).strip().lower()
                for proposal in net_target.proposals
                if str(proposal.sleeve).strip().lower() in policy_disabled_sleeves
            }
        )
        if disabled_for_symbol:
            gates_added.append("POLICY_ABLATION_SLEEVE_BLOCK")
            snapshot_updates["policy_rollback"] = {
                "disabled_sleeves_for_symbol": disabled_for_symbol,
                "disabled_slices": sorted(policy_rollback_disabled_slices),
            }
            return NettingSymbolPreludeResult(
                delta_shares=delta_shares_value,
                target_shares=target_shares,
                target_dollars=target_dollars,
                gates_added=tuple(gates_added),
                snapshot_updates=snapshot_updates,
                blocked_reason="POLICY_ABLATION_SLEEVE_BLOCK",
            )

    snapshot_updates["sleeve_configs"] = {
        proposal.sleeve: {
            "thresholds": {
                "entry": float(getattr(sleeve_configs_map.get(proposal.sleeve), "entry_threshold", 0.0)),
                "exit": float(getattr(sleeve_configs_map.get(proposal.sleeve), "exit_threshold", 0.0)),
                "flip": float(getattr(sleeve_configs_map.get(proposal.sleeve), "flip_threshold", 0.0)),
                "reentry": float(getattr(sleeve_configs_map.get(proposal.sleeve), "reentry_threshold", 0.0)),
            },
            "deadband_dollars": float(
                getattr(sleeve_configs_map.get(proposal.sleeve), "deadband_dollars", 0.0)
            ),
            "cost_k": float(getattr(sleeve_configs_map.get(proposal.sleeve), "cost_k", 0.0)),
            "edge_scale_bps": float(
                getattr(sleeve_configs_map.get(proposal.sleeve), "edge_scale_bps", 0.0)
            ),
            "turnover_cap_dollars": float(
                getattr(sleeve_configs_map.get(proposal.sleeve), "turnover_cap_dollars", 0.0)
            ),
        }
        for proposal in net_target.proposals
    }

    signal_age_seconds = max(
        0.0,
        (now - net_target.bar_ts).total_seconds() if isinstance(net_target.bar_ts, datetime) else 0.0,
    )
    snapshot_updates["signal_age_seconds"] = float(signal_age_seconds)

    if alpha_time_stop_enabled and alpha_time_stop_sec > 0.0 and current_shares != 0:
        opened_at = position_opened_at_func(state, symbol)
        if isinstance(opened_at, datetime):
            position_age_seconds = max(0.0, (now - opened_at).total_seconds())
            expected_edge_now = float(candidate_expected_net_edge.get(symbol, 0.0) or 0.0)
            side_for_exit_policy = "long" if current_shares > 0 else "short"
            exit_policy_context = exit_policy_pressure_context_func(
                state,
                symbol=symbol,
                regime=str(getattr(state, "current_regime", "sideways") or "sideways"),
                side=side_for_exit_policy,
                now=now,
                position_age_seconds=float(position_age_seconds),
                expected_edge_bps=float(expected_edge_now),
            )
            effective_time_stop_sec = float(alpha_time_stop_sec)
            pressure_score = float(exit_policy_context.get("pressure_score", 0.0) or 0.0)
            pressure_min = max(
                0.0,
                min(
                    1.0,
                    float(get_env("AI_TRADING_EXEC_EXIT_POLICY_PRESSURE_MIN", 0.55, cast=float)),
                ),
            )
            suggested_age_seconds = exit_policy_context.get("suggested_max_age_seconds")
            try:
                suggested_age = float(suggested_age_seconds) if suggested_age_seconds is not None else None
            except (TypeError, ValueError):
                suggested_age = None
            if bool(exit_policy_context.get("active", False)) and suggested_age is not None and pressure_score >= pressure_min:
                effective_time_stop_sec = min(float(effective_time_stop_sec), max(60.0, float(suggested_age)))
            exit_policy_edge_buffer_bps = max(
                0.0,
                float(get_env("AI_TRADING_EXEC_EXIT_POLICY_EDGE_BUFFER_BPS", 0.5, cast=float)),
            )
            if position_age_seconds >= float(effective_time_stop_sec):
                if expected_edge_now <= (
                    float(alpha_time_stop_max_expected_edge_bps) + float(exit_policy_edge_buffer_bps)
                ):
                    target_shares = 0
                    target_dollars = 0.0
                    delta_shares_value = -current_shares
                    if bool(exit_policy_context.get("active", False)) and pressure_score >= pressure_min:
                        if "EXIT_POLICY_HAZARD_TIME_STOP" not in gates and "EXIT_POLICY_HAZARD_TIME_STOP" not in gates_added:
                            gates_added.append("EXIT_POLICY_HAZARD_TIME_STOP")
                    elif "ALPHA_TIME_STOP_EXIT" not in gates and "ALPHA_TIME_STOP_EXIT" not in gates_added:
                        gates_added.append("ALPHA_TIME_STOP_EXIT")
                    snapshot_updates["alpha_time_stop"] = {
                        "enabled": True,
                        "position_age_seconds": float(position_age_seconds),
                        "time_stop_sec": float(alpha_time_stop_sec),
                        "effective_time_stop_sec": float(effective_time_stop_sec),
                        "expected_edge_bps": float(expected_edge_now),
                        "max_expected_edge_bps": float(alpha_time_stop_max_expected_edge_bps),
                        "exit_policy": dict(exit_policy_context),
                    }

    post_trade_shares = int(current_shares + delta_shares_value)
    expanding_exposure = abs(post_trade_shares) > abs(current_shares)
    if (
        expanding_exposure
        and opportunity_quality_enabled
        and opportunity_allowed_symbols
        and symbol not in opportunity_allowed_symbols
        and (not opportunity_openings_only or current_shares == 0)
    ):
        gates_added.append("OPPORTUNITY_QUALITY_QUANTILE_BLOCK")
        snapshot_updates["opportunity_quality"] = {
            "score": float(opportunity_quality_by_symbol.get(symbol, 0.0) or 0.0),
            "threshold": opportunity_quality_gate.get("threshold"),
            "top_quantile": float(opportunity_top_quantile),
            "allowed_symbols_count": int(len(opportunity_allowed_symbols)),
            "openings_only": bool(opportunity_openings_only),
        }
        return NettingSymbolPreludeResult(
            delta_shares=delta_shares_value,
            target_shares=target_shares,
            target_dollars=target_dollars,
            gates_added=tuple(gates_added),
            snapshot_updates=snapshot_updates,
            blocked_reason="OPPORTUNITY_QUALITY_QUANTILE_BLOCK",
        )

    if (
        expanding_exposure
        and alpha_time_decay_enabled
        and alpha_stale_signal_sec > 0.0
        and signal_age_seconds > float(alpha_stale_signal_sec)
    ):
        gates_added.append("STALE_SIGNAL_BLOCK")
        snapshot_updates["stale_signal"] = {
            "signal_age_seconds": float(signal_age_seconds),
            "stale_signal_sec": float(alpha_stale_signal_sec),
        }
        return NettingSymbolPreludeResult(
            delta_shares=delta_shares_value,
            target_shares=target_shares,
            target_dollars=target_dollars,
            gates_added=tuple(gates_added),
            snapshot_updates=snapshot_updates,
            blocked_reason="STALE_SIGNAL_BLOCK",
        )

    if live_execution_mode and not burn_in_live_ready and expanding_exposure:
        blocked_reason = burn_in_live_reason or "PAPER_BURN_IN_BLOCK"
        gates_added.append(blocked_reason)
        snapshot_updates["rollout"] = {
            "burn_in_ready": burn_in_live_ready,
            "burn_in_reason": burn_in_live_reason,
            "capital_ramp": dict(ramp_summary),
        }
        return NettingSymbolPreludeResult(
            delta_shares=delta_shares_value,
            target_shares=target_shares,
            target_dollars=target_dollars,
            gates_added=tuple(gates_added),
            snapshot_updates=snapshot_updates,
            blocked_reason=blocked_reason,
        )

    if live_execution_mode and expanding_exposure and ramp_live_multiplier < 0.999:
        scaled_qty = int(round(float(delta_shares_value) * ramp_live_multiplier))
        if scaled_qty == 0:
            scaled_qty = 1 if delta_shares_value > 0 else -1
        if abs(scaled_qty) > abs(delta_shares_value):
            scaled_qty = int(delta_shares_value)
        if scaled_qty != int(delta_shares_value):
            delta_shares_value = int(scaled_qty)
            target_shares = int(current_shares + delta_shares_value)
            target_dollars = float(target_shares * price)
            gates_added.append("CAPITAL_RAMP_SCALE")
            snapshot_updates["capital_ramp"] = {
                "multiplier": ramp_live_multiplier,
                "phase_index": ramp_summary.get("phase_index"),
                "phase_cycles": ramp_summary.get("phase_cycles"),
                "breached": ramp_summary.get("breached"),
                "transition": ramp_summary.get("transition"),
            }

    return NettingSymbolPreludeResult(
        delta_shares=delta_shares_value,
        target_shares=target_shares,
        target_dollars=target_dollars,
        gates_added=tuple(gates_added),
        snapshot_updates=snapshot_updates,
        blocked_reason=None,
    )


__all__ = ["NettingSymbolPreludeResult", "prepare_netting_symbol_prelude"]
