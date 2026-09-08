"""Explain a candidate's stopping decision from development-only diagnostics."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def review_candidate(record: Mapping[str, Any]) -> dict[str, Any]:
    walk = record.get("walk_forward") or {}
    folds = []
    reasons: Counter[str] = Counter()
    for fold in walk.get("folds", []):
        diagnostic = fold.get("abstention_diagnostics") or {}
        trials = diagnostic.get("tested_percentiles", []) if diagnostic.get("scope") == "inner_validation_only" else []
        reasons.update(reason for trial in trials for reason in trial.get("rejection_reasons", []))
        folds.append({"fold_index": fold.get("fold_index"), "threshold_status": diagnostic.get("status"), "threshold_trials": len(trials), "best_development_net_bps": max((trial["mean_net_markout_bps"] for trial in trials if trial.get("mean_net_markout_bps") is not None), default=None), "best_development_gross_bps": max((trial["mean_gross_markout_bps"] for trial in trials if trial.get("mean_gross_markout_bps") is not None), default=None), "selected_candidates": fold.get("selected_candidates"), "confidence_threshold": fold.get("confidence_threshold"), "entry_score_threshold": fold.get("entry_score_threshold")})
    rejected = bool(folds) and all(row["threshold_trials"] and row["threshold_status"] == "abstained" and row["best_development_net_bps"] is not None and row["best_development_net_bps"] <= 0 for row in folds)
    return {"status": "retired_candidate" if rejected else "review_required", "reason": "all_development_thresholds_fail_net_edge" if rejected else "inspect_development_evidence", "folds": folds, "threshold_rejection_counts": dict(reasons), "causal_chain": ["inner_validation_threshold_rejection", "abstention_thresholds", "zero_selected_samples", "regime_abstention", "registry_ineligible"] if rejected else [], "bounded_correction": "none_evidence_does_not_support_lowering_thresholds" if rejected else "not_selected", "stop_rule": "Do not rerun this candidate on the same evidence; a changed hypothesis or new evidence requires its own acceptance criterion.", "holdout_used_for_decision": False, "promotion_authority": False}
