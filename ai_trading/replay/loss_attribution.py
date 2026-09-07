"""Accounting attribution for the same-source capped governance replay.

The headline metric is a mean of fill markouts, not realized portfolio P&L.
This module deliberately does not infer exits or causal fill effects from it.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def build_loss_attribution(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile capped-minus-uncapped mean markout and measure size effects.

Valid only for identical recorded decisions with caps as the changed policy.
Order-set selection is measured first at baseline fill outcomes; subsequent
fill mix/timing and execution-price effects reconcile the observed result.
"""
    base = list(baseline_summary.get("markout_observations", []))
    cand = list(candidate_summary.get("markout_observations", []))
    base_orders = {str(row["client_order_id"]): row for row in baseline.get("orders", [])}
    cand_orders = {str(row["client_order_id"]): row for row in candidate.get("orders", [])}
    retained = [row for row in base if str(row["client_order_id"]) in cand_orders]
    delta = float(candidate_summary["net_edge_bps"]) - float(baseline_summary["net_edge_bps"])
    cap_selection = _mean(retained, "net_edge_bps") - _mean(base, "net_edge_bps")
    fill_effect = _mean(cand, "gross_edge_bps") - _mean(retained, "gross_edge_bps")
    cost_effect = _mean(retained, "execution_drag_bps") - _mean(cand, "execution_drag_bps")
    accounted = cap_selection + fill_effect + cost_effect

    base_pnl = 0.0
    resized_pnl = 0.0
    base_notional = 0.0
    resized_notional = 0.0
    for row in base:
        identity = str(row["client_order_id"])
        original_qty = float(base_orders.get(identity, {}).get("qty", 0.0))
        new_qty = float(cand_orders.get(identity, {}).get("qty", 0.0))
        ratio = new_qty / original_qty if original_qty > 0.0 else 0.0
        notional = float(row["fill_qty"]) * float(row["fill_price"])
        pnl = notional * float(row["net_edge_bps"]) / 10_000.0
        base_pnl += pnl
        resized_pnl += pnl * ratio
        base_notional += notional
        resized_notional += notional * ratio
    cand_notional = sum(float(row["fill_qty"]) * float(row["fill_price"]) for row in cand)
    cand_pnl = sum(
        float(row["fill_qty"]) * float(row["fill_price"]) * float(row["net_edge_bps"]) / 10_000.0
        for row in cand
    )
    by_symbol: dict[str, float] = defaultdict(float)
    for rows, sign in ((base, -1.0), (cand, 1.0)):
        for row in rows:
            by_symbol[str(row["symbol"])] += sign * float(row["net_edge_bps"]) / len(rows)
    adjustments = list(candidate.get("cap_adjustments", []))
    complete = len(base) == baseline_summary["sample_count"] and len(cand) == candidate_summary["sample_count"]
    return {
        "schema_version": "1.0.0",
        "method": "sequential_accounting_same_decisions_caps_only",
        "metric": "mean_next_observation_fill_markout_bps",
        "status": "complete" if complete and base and cand else "insufficient_markout_evidence",
        "baseline_net_edge_bps": baseline_summary["net_edge_bps"],
        "candidate_net_edge_bps": candidate_summary["net_edge_bps"],
        "candidate_minus_baseline_bps": delta,
        "components": {
            "entry_selection": {"contribution_bps": 0.0, "reason": "identical_recorded_decisions"},
            "sizing_caps": {
                "contribution_bps": cap_selection,
                "reason": "cap_retained_order_set_at_baseline_fill_outcomes",
                "quantity_only_headline_effect_bps": 0.0,
            },
            "fills": {"contribution_bps": fill_effect, "reason": "gross_markout_fill_mix_and_timing_including_rng_interactions"},
            "costs": {"contribution_bps": cost_effect, "reason": "exact_reference_to_fill_price_drag_including_return_denominator_effect"},
            "exits": {"contribution_bps": 0.0, "status": "not_identifiable", "reason": "headline_uses_markouts_not_linked_entry_exit_trades"},
        },
        "residual_bps": delta - accounted,
        "reconciled": complete and abs(delta - accounted) < 1e-8,
        "cap_effect": {
            "adjustment_count": len(adjustments),
            "fully_blocked_count": sum(float(row.get("adjusted_qty", 0.0)) <= 0.0 for row in adjustments),
            "removed_requested_notional": sum(
                max(0.0, float(row["requested_qty"]) - float(row["adjusted_qty"])) * float(row["close"])
                for row in adjustments
            ),
            "total_policy_delta_bps": delta,
            "interpretation": "caps_only_policy_contrast_includes_downstream_fill_rng_interactions_not_direct_sizing_causality",
        },
        "quantity_weighted_markout": {
            "baseline_dollars": base_pnl,
            "baseline_fills_resized_to_candidate_orders_dollars": resized_pnl,
            "direct_size_effect_holding_baseline_fills_dollars": resized_pnl - base_pnl,
            "candidate_dollars": cand_pnl,
            "remaining_fill_and_price_effect_dollars": cand_pnl - resized_pnl,
            "baseline_bps": base_pnl / base_notional * 10_000.0 if base_notional else None,
            "resized_baseline_bps": resized_pnl / resized_notional * 10_000.0 if resized_notional else None,
            "candidate_bps": cand_pnl / cand_notional * 10_000.0 if cand_notional else None,
            "is_realized_pnl": False,
        },
        "symbol_contributions_bps": dict(sorted(by_symbol.items(), key=lambda item: item[1])),
        "evidence": {
            "baseline_markouts": len(base), "candidate_markouts": len(cand),
            "baseline_markouts_retained_by_caps": len(retained),
            "baseline_orders": len(base_orders), "candidate_orders": len(cand_orders),
        },
        "limitations": [
            "Sequential accounting depends on component ordering and is not independent causal attribution.",
            "No linked exits, queue position, market impact or separately observed fees are available.",
            "Quantity resizing holds baseline fills fixed and excludes downstream position and fill feedback.",
        ],
    }
