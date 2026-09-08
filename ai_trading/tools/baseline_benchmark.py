"""Evaluate fixed baselines on saved development folds without fitting models."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.data.training_provenance import validate_training_provenance
from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.train_replay_aligned_model import _resolve_training_input, _summarize_controlled_comparisons, build_training_dataset


def benchmark(dataset: pd.DataFrame, candidate: dict[str, Any]) -> dict[str, Any]:
    rows, stress = [], []
    previous_end = None
    for fold in candidate["walk_forward"]["folds"]:
        start, end = pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])
        if previous_end is not None and start <= previous_end:
            raise ValueError("development test folds overlap")
        previous_end = end
        test = dataset.loc[dataset["timestamp"].between(start, end)]
        if len(test) != fold["test_rows"] or test.empty:
            raise ValueError("reconstructed fold does not match saved opportunity coverage")
        fields = test[["gross_long_bps", "net_long_bps", "round_trip_cost_bps", "sma_spread"]].to_numpy(dtype=float)
        if not np.isfinite(fields).all():
            raise ValueError("nonfinite benchmark input")
        if not np.allclose(test["gross_long_bps"] - test["round_trip_cost_bps"], test["net_long_bps"]):
            raise ValueError("net/gross cost contract mismatch")
        index = fold["fold_index"]
        total = float(fold["total_post_cost_net_edge_bps"])
        rows.append({"name": "candidate", "fold_index": index, "opportunities": len(test), "trades": int(fold["selected_candidates"]), "total_net_edge_bps": total, "net_edge_per_opportunity_bps": total / len(test)})
        for name, mask in {"cash": np.zeros(len(test), dtype=bool), "always_long": np.ones(len(test), dtype=bool), "momentum": test["sma_spread"].to_numpy() > 0}.items():
            gross = test["gross_long_bps"].to_numpy()[mask]
            net = test["net_long_bps"].to_numpy()[mask]
            rows.append({"name": name, "fold_index": index, "opportunities": len(test), "trades": int(mask.sum()), "total_net_edge_bps": float(net.sum()), "mean_net_edge_bps": float(net.mean()) if len(net) else None, "net_edge_per_opportunity_bps": float(net.sum() / len(test))})
            for cost in (0, 3, 6, 10):
                stress.append({"name": name, "fold_index": index, "round_trip_cost_bps": cost, "net_edge_per_opportunity_bps": float((gross - cost).sum() / len(test)), "break_even_round_trip_cost_bps": float(gross.mean()) if len(gross) else None})
    comparisons = _summarize_controlled_comparisons(rows)
    decisions = []
    for variant in comparisons["variants"]:
        if variant["name"] == "candidate":
            continue
        qualified = variant["name"] != "cash" and len(variant["folds"]) >= 5 and variant["net_edge_per_opportunity_bps"] > 0 and variant["profitable_fold_ratio"] >= .6
        decisions.append({"name": variant["name"], "decision": "reference_only" if variant["name"] == "cash" else "eligible_for_further_research" if qualified else "fails_benchmark_criterion", "net_edge_per_opportunity_bps": variant["net_edge_per_opportunity_bps"], "profitable_fold_ratio": variant["profitable_fold_ratio"]})
    return {"controlled_comparisons": comparisons, "cost_scenarios": stress, "decisions": decisions, "models_fitted": 0, "holdout_evaluated": False, "metric": "overlapping_equal_notional_opportunity_markouts_not_equity_returns", "promotion_authority": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-report", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.pipeline_report.read_bytes()
    pipeline = json.loads(raw)
    candidate = next(row for row in pipeline["candidates"] if row["model_name"] == args.model_id)
    config = pipeline["config"]
    costs = candidate["walk_forward"]["folds"][0]["cost_model"]
    if costs.get("live_cost_model_source_sha256") or costs.get("version") != "static_fee_spread_slippage_v1" or any(fold["cost_model"] != costs for fold in candidate["walk_forward"]["folds"]):
        raise ValueError("this benchmark requires the saved static cost contract")
    if candidate.get("dataset", {}).get("shadow_markout_evidence", {}).get("matched_rows", 0):
        raise ValueError("shadow label overrides require their original evidence before benchmarking")
    if any(fold.get("chronological_non_overlap") is not True or fold.get("label_purge_ok") is not True for fold in candidate["walk_forward"]["folds"]):
        raise ValueError("saved folds must verify chronological separation and label purging")
    protocol = {"generated_at": datetime.now(UTC).isoformat(), "source_sha256": hashlib.sha256(raw).hexdigest(), "model_id": args.model_id, "hypothesis": "Fixed cash, always-long and positive-SMA-spread momentum controls establish net-edge benchmarks on identical development opportunities", "acceptance": "For non-cash baselines: positive net edge per opportunity across at least five folds and at least 60 percent profitable folds", "stopping_rule": "Report this fixed comparison once; no parameter tuning or candidate revival", "folds": [{k: row[k] for k in ("fold_index", "test_start", "test_end", "test_rows")} for row in candidate["walk_forward"]["folds"]], "cost_contract": costs, "models_fitted": 0, "holdout_evaluated": False}
    atomic_write_text(args.output_dir / "protocol.json", json.dumps(protocol, indent=2) + "\n")
    data_dir, provenance = _resolve_training_input(argparse.Namespace(acquisition_manifest_json=args.acquisition_manifest, symbols=config["symbols"], timestamp_col="timestamp"))
    if provenance.get("quality_passed") is not True or provenance.get("dataset_hash") != candidate["acquisition"].get("dataset_hash"):
        raise ValueError("governed dataset differs from saved candidate evidence")
    quality = validate_training_provenance(data_dir, symbols=config["symbols"].split(","), dataset_identity=provenance["dataset_identity"], min_sessions=20, max_missing_ratio=.02, timestamp_col="timestamp")
    if quality["quality_passed"] is not True:
        raise ValueError("dataset representativeness failed")
    dataset = build_training_dataset(data_dir=data_dir, symbols=config["symbols"], horizon_bars=int(candidate["horizon_bars"]), label_objective=candidate["label_objective"], fee_bps=float(costs["fee_bps"]), slippage_bps=float(costs["slippage_bps"]), use_training_cache=False)
    if dataset.attrs.get("quality_report", {}).get("unexpected_invalid_rate", 0) > .02:
        raise ValueError("reconstructed dataset quality failed")
    report = benchmark(dataset, candidate)
    report.update(protocol=protocol, dataset_hash=provenance["dataset_hash"], quality_passed=True)
    atomic_write_text(args.output_dir / "baseline_benchmark.json", json.dumps(report, indent=2) + "\n")
    derived = copy.deepcopy(pipeline)
    derived["candidates"] = [copy.deepcopy(candidate)]
    derived["candidates"][0]["walk_forward"]["controlled_comparisons"] = report["controlled_comparisons"]
    derived["benchmark_source"] = str(args.output_dir / "baseline_benchmark.json")
    atomic_write_text(args.output_dir / "pipeline_with_baselines.json", json.dumps(derived, indent=2) + "\n")
    get_logger(__name__).info("BASELINE_BENCHMARK_COMPLETE", extra={"decisions": report["decisions"]})


if __name__ == "__main__":
    main()
