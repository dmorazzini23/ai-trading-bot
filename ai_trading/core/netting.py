"""Netting and proposal logic for multi-horizon sleeves."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Iterable


@dataclass(slots=True)
class SleeveConfig:
    name: str
    timeframe: str
    enabled: bool
    entry_threshold: float
    exit_threshold: float
    flip_threshold: float
    reentry_threshold: float
    deadband_dollars: float
    deadband_shares: float
    turnover_cap_dollars: float
    cost_k: float
    edge_scale_bps: float
    max_symbol_dollars: float
    max_gross_dollars: float


@dataclass(slots=True)
class SleeveProposal:
    symbol: str
    sleeve: str
    bar_ts: datetime
    target_dollars: float
    expected_edge_bps: float
    expected_cost_bps: float
    score: float
    confidence: float
    blocked: bool = False
    reason_code: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NettedTarget:
    symbol: str
    bar_ts: datetime
    target_dollars: float
    target_shares: float
    reasons: list[str] = field(default_factory=list)
    proposals: list[SleeveProposal] = field(default_factory=list)
    disagreement_ratio: float | None = None
    blocked: bool = False


@dataclass(slots=True)
class DecisionRecord:
    symbol: str
    bar_ts: datetime
    sleeves: list[SleeveProposal]
    net_target: NettedTarget
    gates: list[str] = field(default_factory=list)
    order: dict[str, Any] | None = None
    fills: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    tca: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        sleeves = [asdict(s) for s in self.sleeves]
        for sleeve in sleeves:
            bar_ts = sleeve.get("bar_ts")
            if isinstance(bar_ts, datetime):
                sleeve["bar_ts"] = bar_ts.isoformat()
        net_target = asdict(self.net_target)
        if isinstance(net_target.get("bar_ts"), datetime):
            net_target["bar_ts"] = net_target["bar_ts"].isoformat()
        return {
            "symbol": self.symbol,
            "bar_ts": self.bar_ts.isoformat(),
            "sleeves": sleeves,
            "net_target": net_target,
            "gates": list(self.gates),
            "order": self.order,
            "fills": list(self.fills),
            "metrics": dict(self.metrics),
            "config_snapshot": dict(self.config_snapshot),
            "tca": dict(self.tca) if isinstance(self.tca, dict) else self.tca,
        }


def compute_sleeve_target_dollars(
    cfg: SleeveConfig,
    score: float,
    current_pos_dollars: float,
) -> float:
    """Compute raw sleeve target dollars with hysteresis thresholds."""
    score_abs = abs(score)
    if current_pos_dollars == 0.0 and score_abs < cfg.entry_threshold:
        return 0.0
    desired = 0.0
    if score > 0:
        desired = cfg.max_symbol_dollars * min(score_abs, 1.0)
    elif score < 0:
        desired = -cfg.max_symbol_dollars * min(score_abs, 1.0)
    if current_pos_dollars != 0.0:
        if (current_pos_dollars > 0 and score > 0) or (current_pos_dollars < 0 and score < 0):
            if score_abs < cfg.exit_threshold:
                return current_pos_dollars
        else:
            if score_abs < cfg.flip_threshold:
                return 0.0
    return desired


def apply_hysteresis(
    cfg: SleeveConfig,
    target_dollars: float,
    current_pos_dollars: float,
) -> float:
    """Apply deadband hysteresis on the target."""
    if abs(target_dollars - current_pos_dollars) < cfg.deadband_dollars:
        return current_pos_dollars
    return target_dollars


def estimate_cost_bps(
    price: float,
    spread: float | None,
    vol: float | None,
    size_dollars: float,
    volume: float | None = None,
) -> float:
    if price <= 0:
        return 0.0
    spread_bps = 5.0
    if spread is not None and spread > 0:
        spread_bps = min(100.0, (spread / price) * 10000.0)
    vol_bps = 5.0
    if vol is not None and vol > 0:
        vol_bps = min(100.0, vol * 10000.0)
    size_bps = 0.0
    if volume and volume > 0 and size_dollars > 0:
        size_bps = min(100.0, (size_dollars / max(volume * price, 1.0)) * 10000.0)
    return spread_bps + 0.5 * vol_bps + size_bps


def apply_cost_gate(expected_edge_bps: float, expected_cost_bps: float, cost_k: float) -> bool:
    """Return True when cost gate blocks the trade."""
    return expected_edge_bps < (cost_k * expected_cost_bps)


def apply_turnover_gate(projected_turnover: float, cap: float) -> bool:
    """Return True when turnover cap blocks the trade."""
    if cap <= 0:
        return False
    return projected_turnover > cap


def compute_sleeve_proposal(
    cfg: SleeveConfig,
    symbol: str,
    bar_ts: datetime,
    score: float,
    confidence: float,
    current_pos: float,
    price: float,
    spread: float | None,
    vol: float | None,
    volume: float | None = None,
) -> SleeveProposal:
    current_pos_dollars = float(current_pos) * float(price or 0.0)
    target_dollars = compute_sleeve_target_dollars(cfg, score, current_pos_dollars)
    target_dollars = apply_hysteresis(cfg, target_dollars, current_pos_dollars)
    size_dollars = abs(target_dollars - current_pos_dollars)
    expected_edge_bps = abs(score) * cfg.edge_scale_bps
    expected_cost_bps = estimate_cost_bps(price, spread, vol, size_dollars, volume)
    blocked = False
    reason = None
    if apply_cost_gate(expected_edge_bps, expected_cost_bps, cfg.cost_k):
        blocked = True
        reason = "COST_GATE"
    return SleeveProposal(
        symbol=symbol,
        sleeve=cfg.name,
        bar_ts=bar_ts,
        target_dollars=target_dollars if not blocked else current_pos_dollars,
        expected_edge_bps=expected_edge_bps,
        expected_cost_bps=expected_cost_bps,
        score=score,
        confidence=confidence,
        blocked=blocked,
        reason_code=reason,
        debug={
            "entry_threshold": cfg.entry_threshold,
            "exit_threshold": cfg.exit_threshold,
            "flip_threshold": cfg.flip_threshold,
            "deadband_dollars": cfg.deadband_dollars,
            "deadband_shares": cfg.deadband_shares,
            "size_dollars": size_dollars,
        },
    )


def apply_disagreement_damping(
    net_target: float,
    proposals: Iterable[SleeveProposal],
    threshold: float,
) -> tuple[float, float | None, bool]:
    targets = [abs(p.target_dollars) for p in proposals]
    total = sum(targets)
    if total <= 0:
        return net_target, None, False
    ratio = abs(net_target) / total if total else 1.0
    if ratio < threshold:
        return net_target * ratio, ratio, True
    return net_target, ratio, False


def net_targets_for_symbol(
    symbol: str,
    bar_ts: datetime,
    proposals: list[SleeveProposal],
    disagree_threshold: float,
) -> NettedTarget:
    net = sum(p.target_dollars for p in proposals)
    adjusted, ratio, damped = apply_disagreement_damping(net, proposals, disagree_threshold)
    reasons: list[str] = []
    if damped:
        reasons.append("DISAGREEMENT_DAMPING")
    return NettedTarget(
        symbol=symbol,
        bar_ts=bar_ts,
        target_dollars=adjusted,
        target_shares=0.0,
        reasons=reasons,
        proposals=proposals,
        disagreement_ratio=ratio,
    )


def apply_global_caps(
    targets: dict[str, NettedTarget],
    max_symbol_dollars: float,
    max_gross_dollars: float,
    max_net_dollars: float,
) -> list[str]:
    reasons: list[str] = []
    for target in targets.values():
        if max_symbol_dollars > 0 and abs(target.target_dollars) > max_symbol_dollars:
            target.target_dollars = max(-max_symbol_dollars, min(max_symbol_dollars, target.target_dollars))
            target.reasons.append("RISK_CAP_SYMBOL")
            reasons.append("RISK_CAP_SYMBOL")
    gross = sum(abs(t.target_dollars) for t in targets.values())
    net = sum(t.target_dollars for t in targets.values())
    scale = 1.0
    if max_gross_dollars > 0 and gross > max_gross_dollars:
        scale = min(scale, max_gross_dollars / gross)
        reasons.append("RISK_CAP_PORTFOLIO")
    if max_net_dollars > 0 and abs(net) > max_net_dollars:
        scale = min(scale, max_net_dollars / abs(net))
        reasons.append("RISK_CAP_PORTFOLIO")
    if scale < 1.0:
        for target in targets.values():
            target.target_dollars *= scale
            target.reasons.append("RISK_CAP_PORTFOLIO")
    return reasons
