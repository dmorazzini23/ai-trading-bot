"""Build a human-readable trading-day attribution report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_jsonl(
    path: Path | None,
    *,
    report_date: str | None = None,
    max_rows: int = 200_000,
) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                if report_date and not _date_match(parsed, report_date):
                    continue
                rows.append(parsed)
                if len(rows) >= max(1, int(max_rows)):
                    rows.pop(0)
    return rows


def _date_match(row: Mapping[str, Any], report_date: str) -> bool:
    ts = str(row.get("ts") or row.get("timestamp") or row.get("decision_ts") or "")
    return ts.startswith(report_date)


def _status(payload: Mapping[str, Any], default: str = "missing") -> str:
    raw = payload.get("status")
    if isinstance(raw, Mapping):
        return str(raw.get("status") or default)
    return str(raw or default)


def build_trading_day_report(
    *,
    report_date: str,
    order_intents: Sequence[Mapping[str, Any]],
    fills: Sequence[Mapping[str, Any]],
    shadow_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
    live_cost_model: Mapping[str, Any],
    symbol_scorecard: Mapping[str, Any],
    regime_entry_throttle: Mapping[str, Any] | None = None,
    expected_edge_calibration: Mapping[str, Any] | None = None,
    execution_capture: Mapping[str, Any] | None = None,
    counterfactual_execution: Mapping[str, Any] | None = None,
    portfolio_edge: Mapping[str, Any] | None = None,
    decision_receipts: Mapping[str, Any] | None = None,
    model_registry: Mapping[str, Any] | None = None,
    pretrade_risk_verifier: Mapping[str, Any] | None = None,
    post_trade_surveillance: Mapping[str, Any] | None = None,
    experiment_ledger: Mapping[str, Any] | None = None,
    walk_forward_capital: Mapping[str, Any] | None = None,
    order_type_optimizer: Mapping[str, Any] | None = None,
    regime_champions: Mapping[str, Any] | None = None,
    adversarial_failure: Mapping[str, Any] | None = None,
    drift_monitor: Mapping[str, Any] | None = None,
    operator_control_plane: Mapping[str, Any] | None = None,
    huggingface_discovery: Mapping[str, Any] | None = None,
    huggingface_candidate_intake: Mapping[str, Any] | None = None,
    huggingface_cache_materialization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    intents = [row for row in order_intents if _date_match(row, report_date)]
    fill_rows = [row for row in fills if _date_match(row, report_date)]
    shadows = [row for row in shadow_rows if _date_match(row, report_date)]
    gates = [row for row in gate_rows if _date_match(row, report_date)]
    rejected = [row for row in gates if str(row.get("action") or row.get("status") or "").lower() in {"reject", "rejected", "blocked"}]
    reject_reasons = Counter(str(row.get("reason") or row.get("gate") or "unknown") for row in rejected)
    symbol_trade_flow: dict[str, Counter[str]] = defaultdict(Counter)
    for row in intents:
        symbol = str(row.get("symbol") or "").upper() or "UNKNOWN"
        symbol_trade_flow[symbol]["desired"] += 1
        if str(row.get("status") or "").upper() in {"SUBMITTED", "FILLED"}:
            symbol_trade_flow[symbol]["submitted"] += 1
    for row in rejected:
        symbol = str(row.get("symbol") or "").upper() or "UNKNOWN"
        symbol_trade_flow[symbol]["rejected"] += 1
    symbol_pnl: dict[str, float] = defaultdict(float)
    symbol_realized_edge_bps: dict[str, list[float]] = defaultdict(list)
    symbol_expected_edge_bps: dict[str, list[float]] = defaultdict(list)
    symbol_slippage_bps: dict[str, list[float]] = defaultdict(list)
    for row in fill_rows:
        symbol = str(row.get("symbol") or "").upper() or "UNKNOWN"
        symbol_trade_flow[symbol]["fills"] += 1
        pnl = row.get("pnl") if row.get("pnl") is not None else row.get("realized_pnl")
        try:
            symbol_pnl[symbol] += float(pnl or 0.0)
        except (TypeError, ValueError):
            pass
        for source_key, target in (
            ("realized_net_edge_bps", symbol_realized_edge_bps),
            ("expected_net_edge_bps", symbol_expected_edge_bps),
            ("slippage_bps", symbol_slippage_bps),
        ):
            try:
                target[symbol].append(float(row.get(source_key) or 0.0))
            except (TypeError, ValueError):
                pass
    side_semantics: Counter[str] = Counter()
    for row in list(intents) + list(gates) + list(fill_rows):
        side = str(row.get("side") or row.get("order_side") or row.get("intended_side") or "").lower()
        reason = str(row.get("reason") or row.get("block_reason") or row.get("detail") or "").lower()
        position_intent = str(row.get("position_intent") or row.get("intent") or "").lower()
        closing = bool(row.get("closing_position") or row.get("reduce_only"))
        if side in {"sell_short", "short", "sellshort"} and (
            "long_only" in reason or "short" in reason
        ):
            side_semantics["sell_short_blocked"] += 1
        elif side in {"sell_short", "short", "sellshort"}:
            side_semantics["sell_short"] += 1
        elif side == "sell" and (closing or "close" in position_intent):
            side_semantics["sell_to_close"] += 1
        elif side == "sell":
            side_semantics["sell_to_reduce"] += 1
        elif side in {"buy", "buy_to_cover", "cover"} and "cover" in position_intent:
            side_semantics["cover_short"] += 1
        elif side == "buy":
            side_semantics["buy_to_open"] += 1
    missed_symbols = Counter(
        str(row.get("symbol") or "").upper() or "UNKNOWN"
        for row in shadows
        if bool(row.get("challenger_would_trade")) and not bool(row.get("champion_would_trade"))
    )

    def _mean_by_symbol(values: Mapping[str, list[float]]) -> dict[str, float]:
        return {
            symbol: float(sum(rows) / len(rows))
            for symbol, rows in sorted(values.items())
            if rows
        }

    def _mean_all(values: Mapping[str, list[float]]) -> float | None:
        rows = [item for symbol_rows in values.values() for item in symbol_rows]
        return float(sum(rows) / len(rows)) if rows else None

    desired_count = len(intents)
    submitted_count = len(
        [row for row in intents if str(row.get("status") or "").upper() in {"SUBMITTED", "FILLED"}]
    )
    rejected_count = len(rejected)
    fill_count = len(fill_rows)
    live_status_payload = live_cost_model.get("status", {})
    live_status = _status(
        live_status_payload if isinstance(live_status_payload, Mapping) else live_cost_model
    )
    calibration_status = _status(expected_edge_calibration or {})
    report = {
        "schema_version": "1.0.0",
        "artifact_type": "trading_day_report",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "desired_trades": {"count": desired_count},
        "submitted_trades": {"count": submitted_count},
        "rejected_trades": {
            "count": rejected_count,
            "reasons": dict(reject_reasons),
        },
        "realized_fills": {"count": fill_count},
        "slippage_spread_cost": {
            "live_cost_status": live_cost_model.get("status", {}),
            "summary": live_cost_model.get("summary", {}),
        },
        "estimated_edge_vs_realized": {
            "shadow_rows": len(shadows),
            "fill_rows": len(fill_rows),
        },
        "symbol_contribution": dict(sorted(symbol_pnl.items())),
        "symbol_trade_flow": {
            symbol: {
                "desired": int(counts.get("desired", 0)),
                "submitted": int(counts.get("submitted", 0)),
                "rejected": int(counts.get("rejected", 0)),
                "fills": int(counts.get("fills", 0)),
            }
            for symbol, counts in sorted(symbol_trade_flow.items())
        },
        "symbol_realized_edge_bps": _mean_by_symbol(symbol_realized_edge_bps),
        "symbol_expected_edge_bps": _mean_by_symbol(symbol_expected_edge_bps),
        "symbol_slippage_bps": _mean_by_symbol(symbol_slippage_bps),
        "edge_quality": {
            "mean_realized_edge_bps": _mean_all(symbol_realized_edge_bps),
            "mean_expected_edge_bps": _mean_all(symbol_expected_edge_bps),
            "mean_slippage_bps": _mean_all(symbol_slippage_bps),
        },
        "expected_edge_calibration": dict(expected_edge_calibration or {}),
        "execution_capture_diagnosis": (
            dict(expected_edge_calibration.get("execution_capture_diagnosis", {}))
            if isinstance(expected_edge_calibration, Mapping)
            else {}
        ),
        "long_only_side_semantics": {
            "counts": dict(sorted(side_semantics.items())),
            "sell_to_close_allowed": True,
            "open_short_blocked_field": "sell_short_blocked",
        },
        "gate_effectiveness": {"rejected_by_gate": dict(reject_reasons)},
        "missed_opportunities": {
            "shadow_only_count": int(sum(missed_symbols.values())),
            "symbols": dict(sorted(missed_symbols.items())),
        },
        "symbol_scorecard": {
            "summary": symbol_scorecard.get("summary", {}),
            "symbols": symbol_scorecard.get("symbols", []),
        },
        "regime_entry_throttle": dict(regime_entry_throttle or {}),
        "execution_capture": dict(execution_capture or {}),
        "counterfactual_execution": dict(counterfactual_execution or {}),
        "portfolio_edge_control": dict(portfolio_edge or {}),
        "decision_receipts": dict(decision_receipts or {}),
        "model_registry": dict(model_registry or {}),
        "pretrade_risk_control_verifier": dict(pretrade_risk_verifier or {}),
        "post_trade_surveillance": dict(post_trade_surveillance or {}),
        "experiment_ledger": dict(experiment_ledger or {}),
        "walk_forward_capital_simulation": dict(walk_forward_capital or {}),
        "order_type_optimizer": dict(order_type_optimizer or {}),
        "regime_champion_models": dict(regime_champions or {}),
        "adversarial_failure_simulation": dict(adversarial_failure or {}),
        "model_data_drift_monitor": dict(drift_monitor or {}),
        "operator_control_plane": dict(operator_control_plane or {}),
        "huggingface_research": {
            "discovery": dict(huggingface_discovery or {}),
            "candidate_intake": dict(huggingface_candidate_intake or {}),
            "cache_materialization": dict(huggingface_cache_materialization or {}),
            "research_only": True,
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
        },
        "next_session_recommendation": "review_live_capital_readiness_before_live_trading",
    }
    report["health_report_summary"] = {
        "desired": desired_count,
        "submitted": submitted_count,
        "rejected": rejected_count,
        "fills": fill_count,
        "live_cost_status": live_status,
        "expected_edge_calibration_status": calibration_status,
        "model_registry_status": _status(model_registry or {}),
        "pretrade_risk_status": _status(pretrade_risk_verifier or {}),
        "post_trade_surveillance_status": _status(post_trade_surveillance or {}),
        "experiment_ledger_status": _status(experiment_ledger or {}),
        "walk_forward_capital_status": _status(walk_forward_capital or {}),
        "order_type_optimizer_status": _status(order_type_optimizer or {}),
        "regime_champion_status": _status(regime_champions or {}),
        "adversarial_failure_status": _status(adversarial_failure or {}),
        "drift_monitor_status": _status(drift_monitor or {}),
        "operator_control_plane_status": _status(operator_control_plane or {}),
        "huggingface_research_status": _status(huggingface_discovery or {}),
        "huggingface_intake_status": _status(huggingface_candidate_intake or {}),
        "top_reject_reasons": dict(reject_reasons.most_common(5)),
        "symbols_with_activity": sorted(report["symbol_trade_flow"]),
    }
    report["openclaw_summary"] = {
        "service": "ai-trading-research",
        "severity": "info" if rejected_count == 0 else "warning",
        "summary": (
            f"trading_day desired={desired_count} submitted={submitted_count} "
            f"rejected={rejected_count} fills={fill_count}"
        ),
        "suggested_action": "review rejects and live-capital readiness before next session",
        "details": report["health_report_summary"],
    }
    return report


def _default_report_paths(report_date: str) -> tuple[Path, Path, Path]:
    root = resolve_runtime_artifact_path(
        "runtime/reports",
        default_relative="runtime/reports",
        for_write=True,
    )
    compact = report_date.replace("-", "")
    return (
        root / f"trading_day_{compact}.json",
        root / "trading_day_latest.json",
        root / "trading_day_latest.md",
    )


def _markdown(report: Mapping[str, Any]) -> str:
    throttle = report.get("regime_entry_throttle")
    throttle_actions = {}
    if isinstance(throttle, Mapping):
        actions = throttle.get("actions")
        throttle_actions = dict(actions) if isinstance(actions, Mapping) else {}
    return "\n".join(
        [
            f"# Trading Day {report.get('report_date')}",
            "",
            f"- Desired trades: `{report.get('desired_trades', {}).get('count', 0)}`",
            f"- Submitted trades: `{report.get('submitted_trades', {}).get('count', 0)}`",
            f"- Rejected trades: `{report.get('rejected_trades', {}).get('count', 0)}`",
            f"- Realized fills: `{report.get('realized_fills', {}).get('count', 0)}`",
            f"- Regime entry throttle: `{throttle_actions or {}}`",
            f"- Expected-edge calibration: `{report.get('expected_edge_calibration', {}).get('status', 'missing')}`",
            f"- Execution capture: `{report.get('execution_capture', {}).get('status', 'missing')}`",
            f"- Counterfactual execution: `{report.get('counterfactual_execution', {}).get('status', 'missing')}`",
            f"- Portfolio edge: `{report.get('portfolio_edge_control', {}).get('output', 'missing')}`",
            f"- Decision receipts: `{report.get('decision_receipts', {}).get('status', 'missing')}`",
            f"- Model registry: `{report.get('model_registry', {}).get('status', 'missing')}`",
            f"- Pre-trade risk verifier: `{report.get('pretrade_risk_control_verifier', {}).get('status', 'missing')}`",
            f"- Post-trade surveillance: `{report.get('post_trade_surveillance', {}).get('status', 'missing')}`",
            f"- Experiment ledger: `{report.get('experiment_ledger', {}).get('status', 'missing')}`",
            f"- Walk-forward capital: `{report.get('walk_forward_capital_simulation', {}).get('status', 'missing')}`",
            f"- Order-type optimizer: `{report.get('order_type_optimizer', {}).get('status', 'missing')}`",
            f"- Regime champions: `{report.get('regime_champion_models', {}).get('status', 'missing')}`",
            f"- Adversarial simulation: `{report.get('adversarial_failure_simulation', {}).get('status', 'missing')}`",
            f"- Drift monitor: `{report.get('model_data_drift_monitor', {}).get('status', 'missing')}`",
            f"- Operator control plane: `{report.get('operator_control_plane', {}).get('status', 'missing')}`",
            f"- Hugging Face research: `{report.get('huggingface_research', {}).get('discovery', {}).get('status', 'missing')}`",
            f"- Health/report summary: `{report.get('health_report_summary', {}).get('desired', 0)}` desired, "
            f"`{report.get('health_report_summary', {}).get('fills', 0)}` fills",
            f"- Next session: `{report.get('next_session_recommendation')}`",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--order-intents-jsonl", type=Path, default=None)
    parser.add_argument("--fills-jsonl", type=Path, default=None)
    parser.add_argument("--shadow-jsonl", type=Path, default=None)
    parser.add_argument("--gate-jsonl", type=Path, default=None)
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--symbol-scorecard-json", type=Path, default=None)
    parser.add_argument("--regime-entry-throttle-json", type=Path, default=None)
    parser.add_argument("--expected-edge-calibration-json", type=Path, default=None)
    parser.add_argument("--execution-capture-json", type=Path, default=None)
    parser.add_argument("--counterfactual-execution-json", type=Path, default=None)
    parser.add_argument("--portfolio-edge-json", type=Path, default=None)
    parser.add_argument("--decision-receipts-json", type=Path, default=None)
    parser.add_argument("--model-registry-json", type=Path, default=None)
    parser.add_argument("--pretrade-risk-json", type=Path, default=None)
    parser.add_argument("--post-trade-surveillance-json", type=Path, default=None)
    parser.add_argument("--experiment-ledger-json", type=Path, default=None)
    parser.add_argument("--walk-forward-capital-json", type=Path, default=None)
    parser.add_argument("--order-type-optimizer-json", type=Path, default=None)
    parser.add_argument("--regime-champions-json", type=Path, default=None)
    parser.add_argument("--adversarial-failure-json", type=Path, default=None)
    parser.add_argument("--drift-monitor-json", type=Path, default=None)
    parser.add_argument("--operator-control-plane-json", type=Path, default=None)
    parser.add_argument("--huggingface-discovery-json", type=Path, default=None)
    parser.add_argument("--huggingface-candidate-intake-json", type=Path, default=None)
    parser.add_argument("--huggingface-cache-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    parser.add_argument("--latest-md", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, latest_json, latest_md = _default_report_paths(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    latest_md = args.latest_md or latest_md
    report = build_trading_day_report(
        report_date=str(args.report_date),
        order_intents=_read_jsonl(args.order_intents_jsonl, report_date=str(args.report_date)),
        fills=_read_jsonl(args.fills_jsonl, report_date=str(args.report_date)),
        shadow_rows=_read_jsonl(args.shadow_jsonl, report_date=str(args.report_date)),
        gate_rows=_read_jsonl(args.gate_jsonl, report_date=str(args.report_date)),
        live_cost_model=_read_json(args.live_cost_model_json),
        symbol_scorecard=_read_json(args.symbol_scorecard_json),
        regime_entry_throttle=_read_json(args.regime_entry_throttle_json),
        expected_edge_calibration=_read_json(args.expected_edge_calibration_json),
        execution_capture=_read_json(args.execution_capture_json),
        counterfactual_execution=_read_json(args.counterfactual_execution_json),
        portfolio_edge=_read_json(args.portfolio_edge_json),
        decision_receipts=_read_json(args.decision_receipts_json),
        model_registry=_read_json(args.model_registry_json),
        pretrade_risk_verifier=_read_json(args.pretrade_risk_json),
        post_trade_surveillance=_read_json(args.post_trade_surveillance_json),
        experiment_ledger=_read_json(args.experiment_ledger_json),
        walk_forward_capital=_read_json(args.walk_forward_capital_json),
        order_type_optimizer=_read_json(args.order_type_optimizer_json),
        regime_champions=_read_json(args.regime_champions_json),
        adversarial_failure=_read_json(args.adversarial_failure_json),
        drift_monitor=_read_json(args.drift_monitor_json),
        operator_control_plane=_read_json(args.operator_control_plane_json),
        huggingface_discovery=_read_json(args.huggingface_discovery_json),
        huggingface_candidate_intake=_read_json(args.huggingface_candidate_intake_json),
        huggingface_cache_materialization=_read_json(args.huggingface_cache_json),
    )
    for path, content in (
        (output_json, json.dumps(report, indent=2, sort_keys=True) + "\n"),
        (latest_json, json.dumps(report, indent=2, sort_keys=True) + "\n"),
        (latest_md, _markdown(report)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": "written"}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
