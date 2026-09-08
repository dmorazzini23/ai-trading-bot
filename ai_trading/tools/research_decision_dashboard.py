"""Source-backed research decisions and fixed-budget portfolio comparisons."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.candidate_abstention_review import review_candidate


def candidate_review(row: dict[str, Any]) -> dict[str, Any]:
    walk = row.get("walk_forward") or {}
    folds = walk.get("folds", [])
    funnel = []
    for fold in folds:
        source = fold.get("opportunity_funnel") or {}
        total, finite, selected = (source.get(key) for key in ("outer_test_rows", "finite_scores", "selected_by_frozen_threshold"))
        valid = all(isinstance(x, int) and not isinstance(x, bool) for x in (total, finite, selected)) and 0 <= selected <= finite <= total
        funnel.append({"fold": fold.get("fold_index"), "status": "reconciled" if valid else "missing_or_inconsistent", "opportunities": total, "invalid_scores": total - finite if valid else None, "threshold_rejections": finite - selected if valid else None, "selected": selected, "reason": source.get("reason"), "scope": "model_selection_only"})
    comparisons = walk.get("controlled_comparisons") or {}
    variants = {item["name"]: item for item in comparisons.get("variants", [])}
    baseline_rows = []
    candidate_folds = {f["fold_index"]: f for f in variants.get("candidate", {}).get("folds", [])}
    common_contract = comparisons.get("scope") == "development_outer_folds_only" and comparisons.get("selection_scope") == "nested_inner_validation_only" and comparisons.get("metric") == "equal_notional_markout_per_common_opportunity_not_portfolio_return"
    for name in ("cash", "always_long", "momentum"):
        baseline = variants.get(name, {})
        base_folds = {f["fold_index"]: f for f in baseline.get("folds", [])}
        matched = common_contract and bool(candidate_folds) and len(candidate_folds) == len(variants["candidate"]["folds"]) and len(base_folds) == len(baseline.get("folds", [])) and candidate_folds.keys() == base_folds.keys() and all(candidate_folds[k]["opportunities"] == base_folds[k]["opportunities"] for k in candidate_folds)
        deltas = [candidate_folds[k]["net_edge_per_opportunity_bps"] - base_folds[k]["net_edge_per_opportunity_bps"] for k in candidate_folds] if matched else []
        baseline_rows.append({"baseline": name, "status": "paired" if matched else "missing_or_unpaired", "folds": len(deltas), "candidate_minus_baseline_bps": float(np.mean(deltas)) if deltas else None, "improving_fold_ratio": float(np.mean(np.asarray(deltas) > 0)) if deltas else None, "metric": "net_edge_per_common_opportunity_not_portfolio_return"})
    return {"model_id": row.get("model_name"), "development_eligible": row.get("development_eligible") is True, "opportunity_funnel": funnel, "execution_funnel": {"status": "separate_evidence_required", "reason": "training_opportunities_are_not_correlated_live_orders"}, "baselines": baseline_rows, "baseline_protocol": {"scope": comparisons.get("scope"), "selection_scope": comparisons.get("selection_scope")}, "decision": review_candidate(row), "training_report_path": row.get("training_report_path"), "replay_status": row.get("replay_status")}


def evaluate_portfolio(frame: pd.DataFrame, *, candidate: str, costs: tuple[float, ...] = (0, 3, 6, 10)) -> dict[str, Any]:
    """Use explicit aligned daily equity returns, never pool opportunity markouts."""
    required = {"session", "strategy", "gross_return", "turnover", "gross_exposure"}
    if not required <= set(frame.columns):
        raise ValueError("portfolio input requires session,strategy,gross_return,turnover,gross_exposure")
    frame = frame.copy()
    frame["session"] = pd.to_datetime(frame["session"], utc=True, errors="raise")
    if frame.empty or frame["session"].isna().any() or (frame["session"] != frame["session"].dt.normalize()).any() or frame["strategy"].isna().any() or frame["strategy"].astype(str).str.strip().eq("").any() or frame.duplicated(["session", "strategy"]).any():
        raise ValueError("empty, missing strategy, or duplicate session/strategy rows")
    numeric = frame[["gross_return", "turnover", "gross_exposure"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all() or (numeric["turnover"] < 0).any() or ((numeric["gross_exposure"] < 0) | (numeric["gross_exposure"] > 1)).any() or (numeric["gross_return"] <= -1).any():
        raise ValueError("invalid returns, turnover, or unlevered exposure")
    frame[numeric.columns] = numeric
    gross = frame.pivot(index="session", columns="strategy", values="gross_return").sort_index()
    if candidate not in gross or len(gross.columns) < 2 or gross.isna().any().any():
        raise ValueError("candidate plus existing strategies need identical explicit session coverage")
    turnover = frame.pivot(index="session", columns="strategy", values="turnover").reindex_like(gross)
    exposure = frame.pivot(index="session", columns="strategy", values="gross_exposure").reindex_like(gross)
    existing = [name for name in gross if name != candidate]
    def metrics(series: pd.Series) -> dict[str, float]:
        if (series <= -1).any():
            raise ValueError("scenario exhausts portfolio capital")
        equity = np.r_[1.0, np.cumprod(1 + series.to_numpy())]
        return {"total_return": float(equity[-1] - 1), "mean_daily_return": float(series.mean()), "max_drawdown": float(np.max(1 - equity / np.maximum.accumulate(equity)))}
    scenarios = []
    for cost in costs:
        if not np.isfinite(cost) or cost < 0:
            raise ValueError("cost scenarios must be finite and nonnegative")
        net = gross - turnover * cost / 10000
        if (net <= -1).any().any():
            raise ValueError("scenario exhausts a strategy's allocated capital")
        def funded_return(names: list[str]) -> pd.Series:
            equity = (1 + net[names]).cumprod().mean(axis=1)
            return equity / equity.shift(1, fill_value=1) - 1
        before, after = funded_return(existing), funded_return(list(net.columns))
        correlation = net[candidate].corr(before) if net[candidate].std() > 0 and before.std() > 0 else None
        scenarios.append({"cost_bps_per_traded_notional": cost, "before": metrics(before), "after": metrics(after), "mean_daily_incremental_return": float((after - before).mean()), "candidate_existing_correlation": float(correlation) if correlation is not None and np.isfinite(correlation) else None, "joint_loss_fraction": float(((net[candidate] < 0) & (before < 0)).mean())})
    return {"status": "evaluated" if len(gross) >= 20 else "insufficient_sessions", "minimum_sessions": 20, "sessions": len(gross), "candidate": candidate, "existing": existing, "allocation": "equal_initial_capital_independent_sleeves_no_cross_sleeve_rebalancing", "mean_existing_sleeve_turnover": float(turnover[existing].mean(axis=1).mean()), "mean_all_sleeve_turnover": float(turnover.mean(axis=1).mean()), "mean_sleeve_gross_exposure": float(exposure.mean(axis=1).mean()), "concurrent_exposure_fraction": float(((exposure[candidate] > 0) & (exposure[existing].sum(axis=1) > 0)).mean()), "scenarios": scenarios, "limitations": ["Turnover includes each sleeve's entry, exit and rebalancing trades per sleeve NAV.", "Same-day exposure overlap does not measure intraday overlap or liquidity capacity.", "Initial allocations are diagnostic; no weight optimization or promotion authority."], "promotion_authority": False}


def render(report: dict[str, Any]) -> str:
    rows = []
    for candidate in report["candidates"]:
        funnel_text = "\n".join(f"Fold {f['fold']}: {f['opportunities']} opportunities; {f['invalid_scores']} invalid scores; {f['threshold_rejections']} threshold rejections; {f['selected']} selected" for f in candidate["opportunity_funnel"])
        baseline_text = "\n".join(f"{b['baseline']}: {b['status']}; candidate advantage {b['candidate_minus_baseline_bps']} bps" for b in candidate["baselines"])
        cells = [candidate["model_id"], candidate["decision"]["status"], funnel_text, baseline_text]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in cells) + "</tr>")
    return "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Research decisions</title><style>body{font:16px system-ui;margin:2rem;max-width:1400px;background:#101820;color:#e9eef2}table{border-collapse:collapse;width:100%}td,th{border:1px solid #53616c;padding:12px;text-align:left;vertical-align:top}td{overflow-wrap:anywhere}pre{white-space:pre-wrap}input{padding:10px;margin:12px 0}</style><h1>Research decisions</h1><p>Development evidence only. Missing evidence stays visible; this report grants no trading authority.</p><label>Filter candidates <input id='filter' type='search'></label><table><thead><tr><th>Candidate</th><th>Decision</th><th>Opportunity funnel</th><th>Independent baselines</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table><h2>Portfolio evidence</h2><pre>" + html.escape(json.dumps(report["portfolio"], indent=2)) + "</pre><h2>Experiment lifecycle</h2><pre>" + html.escape(json.dumps(report["experiment_lifecycle"], indent=2)) + "</pre><h2>Sources</h2><pre>" + html.escape(json.dumps(report["sources"], indent=2)) + "</pre><script>document.getElementById('filter').addEventListener('input',e=>{document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()))});</script></html>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--portfolio-returns", type=Path)
    parser.add_argument("--portfolio-candidate")
    parser.add_argument("--portfolio-manifest", type=Path)
    parser.add_argument("--paper-review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = []
    def read(path: Path) -> dict[str, Any]:
        if not path.exists():
            sources.append({"path": str(path), "missing": True})
            return {}
        raw = path.read_bytes()
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("research report must be a JSON object")
        return value
    training = read(args.training_report)
    if training.get("status") == "skipped_unchanged" and training.get("previous_report_path"):
        training = read(Path(training["previous_report_path"]))
    pipeline = read(Path(training["multi_horizon_report"])) if training.get("multi_horizon_report") else training
    portfolio = {"status": "missing_aligned_daily_return_evidence"}
    if args.portfolio_returns is not None and args.portfolio_returns.exists():
        digest = hashlib.sha256(args.portfolio_returns.read_bytes()).hexdigest()
        sources.append({"path": str(args.portfolio_returns), "sha256": digest})
        manifest = read(args.portfolio_manifest) if args.portfolio_manifest else {}
        if manifest.get("sha256") != digest or manifest.get("evidence_partition") != "out_of_sample" or manifest.get("return_basis") != "daily_equity_fraction" or manifest.get("turnover_basis") != "traded_notional_over_sleeve_nav":
            raise ValueError("portfolio evidence requires a matching out-of-sample equity-return manifest")
        portfolio = evaluate_portfolio(pd.read_csv(args.portfolio_returns), candidate=args.portfolio_candidate or "candidate")
    report = {"generated_at": datetime.now(UTC).isoformat(), "status": "reported" if pipeline.get("candidates") else "evidence_pending", "candidates": [candidate_review(row) for row in pipeline.get("candidates", [])], "experiment_lifecycle": pipeline.get("experiment_lifecycle", {}), "portfolio": portfolio, "sources": sources, "promotion_authority": False}
    report["html_path"] = str(args.output.with_suffix(".html"))
    if args.paper_review is not None:
        paper = read(args.paper_review)
        audit = paper.get("session_audit", {})
        report["execution_diagnostics"] = {"status": paper.get("status", "missing"), "session_date": paper.get("session_date"), "accepted_fills": audit.get("accepted_unique_fills"), "rejection_counts": audit.get("counts", {}), "completion_gaps": paper.get("completion_gaps", []), "cap_selection": paper.get("cap_selection", {}), "scope": "observed_session_and_replay_diagnostics_not_the_training_population"}
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    page = render(report)
    if "execution_diagnostics" in report:
        page = page.replace("<h2>Portfolio evidence</h2>", "<h2>Execution diagnostics</h2><pre>" + html.escape(json.dumps(report["execution_diagnostics"], indent=2)) + "</pre><h2>Portfolio evidence</h2>")
    atomic_write_text(args.output.with_suffix(".html"), page)
    get_logger(__name__).info("RESEARCH_DECISION_DASHBOARD_WRITTEN", extra={"output": str(args.output)})


if __name__ == "__main__":
    main()
