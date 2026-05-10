"""Benchmark Hugging Face sentiment candidates against local sentiment fixtures.

This tool is research-only. It never downloads Hugging Face artifacts, never
loads candidates in runtime, and never grants model, promotion, provider, or
live-money authority. Candidate scores must come from an explicit offline
prediction artifact produced outside live runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

_ARTIFACT_TYPE = "huggingface_sentiment_benchmark"
_SCHEMA_VERSION = "1.0.0"
_DEFAULT_OUTPUT_DIR = "runtime/research_reports/huggingface"
_DEFAULT_SAMPLES: tuple[dict[str, Any], ...] = (
    {
        "id": "positive_earnings",
        "text": "AAPL reported stronger than expected revenue, margin expansion, and upbeat guidance.",
        "expected_label": "positive",
        "symbol": "AAPL",
        "source": "fixture",
    },
    {
        "id": "negative_downgrade",
        "text": "AMZN shares fell after analysts cut estimates and warned about slowing cloud demand.",
        "expected_label": "negative",
        "symbol": "AMZN",
        "source": "fixture",
    },
    {
        "id": "neutral_schedule",
        "text": "MSFT will hold its annual shareholder meeting next month according to a company notice.",
        "expected_label": "neutral",
        "symbol": "MSFT",
        "source": "fixture",
    },
    {
        "id": "negative_regulatory",
        "text": "The company faces a new regulatory investigation and warned legal costs may rise.",
        "expected_label": "negative",
        "symbol": "UNKNOWN",
        "source": "fixture",
    },
    {
        "id": "positive_buyback",
        "text": "The board authorized a larger buyback after free cash flow beat expectations.",
        "expected_label": "positive",
        "symbol": "UNKNOWN",
        "source": "fixture",
    },
    {
        "id": "positive_price_target",
        "text": "Analysts raised the price target after management reported accelerating enterprise demand.",
        "expected_label": "positive",
        "symbol": "MSFT",
        "source": "fixture",
    },
    {
        "id": "positive_upgrade",
        "text": "The stock advanced after a major bank upgraded shares to buy and cited margin upside.",
        "expected_label": "positive",
        "symbol": "AAPL",
        "source": "fixture",
    },
    {
        "id": "positive_contract_win",
        "text": "The company won a multiyear cloud contract expected to lift revenue next year.",
        "expected_label": "positive",
        "symbol": "AMZN",
        "source": "fixture",
    },
    {
        "id": "positive_guidance_raise",
        "text": "Executives raised full year guidance and said demand remains resilient.",
        "expected_label": "positive",
        "symbol": "UNKNOWN",
        "source": "fixture",
    },
    {
        "id": "negative_earnings_miss",
        "text": "Shares dropped after earnings missed expectations and operating margin contracted.",
        "expected_label": "negative",
        "symbol": "AAPL",
        "source": "fixture",
    },
    {
        "id": "negative_downgrade_debt",
        "text": "Credit analysts downgraded the company and warned that debt costs may pressure cash flow.",
        "expected_label": "negative",
        "symbol": "UNKNOWN",
        "source": "fixture",
    },
    {
        "id": "negative_outage",
        "text": "A service outage disrupted customer operations and management declined to update guidance.",
        "expected_label": "negative",
        "symbol": "MSFT",
        "source": "fixture",
    },
    {
        "id": "negative_antitrust",
        "text": "Regulators opened an antitrust probe and the company faces potential fines.",
        "expected_label": "negative",
        "symbol": "AMZN",
        "source": "fixture",
    },
    {
        "id": "neutral_dividend_date",
        "text": "The company announced the record date for its regular quarterly dividend.",
        "expected_label": "neutral",
        "symbol": "AAPL",
        "source": "fixture",
    },
    {
        "id": "neutral_board_appointment",
        "text": "The board appointed a new independent director effective next month.",
        "expected_label": "neutral",
        "symbol": "MSFT",
        "source": "fixture",
    },
    {
        "id": "neutral_product_event",
        "text": "The company scheduled a product presentation for analysts and media next week.",
        "expected_label": "neutral",
        "symbol": "AMZN",
        "source": "fixture",
    },
    {
        "id": "neutral_filing",
        "text": "The company filed its annual report with no change to previously reported results.",
        "expected_label": "neutral",
        "symbol": "UNKNOWN",
        "source": "fixture",
    },
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _compact_date(value: str) -> str:
    return str(value or "").replace("-", "") or _utc_now().strftime("%Y%m%d")


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path(
        _DEFAULT_OUTPUT_DIR,
        default_relative=_DEFAULT_OUTPUT_DIR,
        for_write=True,
    )
    return (
        root / f"hf_sentiment_benchmark_{_compact_date(report_date)}.json",
        root.parent / "latest" / "hf_sentiment_benchmark_latest.json",
    )


def _default_intake_path() -> Path:
    return resolve_runtime_artifact_path(
        "runtime/research_reports/latest/hf_candidate_intake_latest.json",
        default_relative="runtime/research_reports/latest/hf_candidate_intake_latest.json",
    )


def _samples_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("samples")
    if not isinstance(raw, list):
        raw = payload.get("fixtures")
    if not isinstance(raw, list):
        return [dict(row) for row in _DEFAULT_SAMPLES]
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            continue
        text = str(row.get("text") or "").strip()
        expected = _normalize_label(row.get("expected_label") or row.get("label"))
        if not text or expected not in {"positive", "negative", "neutral"}:
            continue
        samples.append(
            {
                "id": str(row.get("id") or f"sample_{index + 1}"),
                "text": text,
                "expected_label": expected,
                "symbol": str(row.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN",
                "source": str(row.get("source") or "fixture"),
            }
        )
    return samples or [dict(row) for row in _DEFAULT_SAMPLES]


def _candidate_rows(intake: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = intake.get("candidates")
    if not isinstance(raw, list):
        return []
    return [dict(row) for row in raw if isinstance(row, Mapping)]


def _ready_candidates(intake: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _candidate_rows(intake):
        if str(row.get("status") or "") != "ready_for_manual_review":
            continue
        rows.append(row)
    return rows


def _normalize_label(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in {"pos", "positive", "bull", "bullish", "1"}:
        return "positive"
    if token in {"neg", "negative", "bear", "bearish", "-1"}:
        return "negative"
    return "neutral"


def _label_from_score(score: float | None, *, threshold: float) -> str | None:
    if score is None or not math.isfinite(score):
        return None
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _current_sentiment_prediction(text: str) -> dict[str, Any]:
    from ai_trading.config.management import (
        clear_runtime_env_override,
        reload_env,
        set_runtime_env_override,
    )

    runtime_env = Path("/run/ai-trading-bot/ai-trading-runtime.env")
    if runtime_env.exists():
        reload_env(path=runtime_env, override=False)
    from ai_trading.analysis import sentiment

    started = time.perf_counter()
    set_runtime_env_override("AI_TRADING_HF_SENTIMENT_BENCHMARK_MODE", "1")
    try:
        payload = sentiment.analyze_text(text)
    finally:
        clear_runtime_env_override("AI_TRADING_HF_SENTIMENT_BENCHMARK_MODE")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    pos = _safe_float(payload.get("pos")) or 0.0
    neg = _safe_float(payload.get("neg")) or 0.0
    neu = _safe_float(payload.get("neu")) or 0.0
    return {
        "available": bool(payload.get("available")),
        "score": float(pos - neg),
        "confidence": float(max(pos, neg, neu)),
        "latency_ms": elapsed_ms,
        "raw": {"pos": pos, "neg": neg, "neu": neu},
    }


def _prediction_maps(payload: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    raw = payload.get("predictions")
    if not isinstance(raw, list):
        raw = payload.get("candidates")
    output: dict[str, dict[str, dict[str, Any]]] = {}
    if not isinstance(raw, list):
        return output
    for candidate in raw:
        if not isinstance(candidate, Mapping):
            continue
        repo_id = str(candidate.get("repo_id") or candidate.get("hf_id") or "").strip()
        if not repo_id:
            continue
        sample_rows = candidate.get("samples")
        if not isinstance(sample_rows, list):
            sample_rows = candidate.get("predictions")
        by_sample: dict[str, dict[str, Any]] = {}
        if isinstance(sample_rows, list):
            for row in sample_rows:
                if not isinstance(row, Mapping):
                    continue
                sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
                if not sample_id:
                    continue
                by_sample[sample_id] = dict(row)
        output[repo_id] = by_sample
    return output


def _score_prediction(row: Mapping[str, Any], *, threshold: float) -> dict[str, Any]:
    label = row.get("label") or row.get("predicted_label")
    score = _safe_float(row.get("score"))
    if score is None:
        pos = _safe_float(row.get("pos"))
        neg = _safe_float(row.get("neg"))
        if pos is not None and neg is not None:
            score = float(pos - neg)
    normalized_label = _normalize_label(label) if label not in (None, "") else _label_from_score(score, threshold=threshold)
    confidence = _safe_float(row.get("confidence"))
    if confidence is None:
        values = [
            value
            for value in (_safe_float(row.get("pos")), _safe_float(row.get("neg")), _safe_float(row.get("neu")))
            if value is not None
        ]
        confidence = max(values) if values else abs(score or 0.0)
    return {
        "score": score,
        "label": normalized_label,
        "confidence": confidence,
        "latency_ms": _safe_float(row.get("latency_ms")),
        "available": normalized_label is not None,
    }


def _evaluate_predictions(
    *,
    name: str,
    samples: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    threshold: float,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    correct = 0
    covered = 0
    confidence_values: list[float] = []
    latency_values: list[float] = []
    for sample in samples:
        sample_id = str(sample.get("id") or "")
        expected = _normalize_label(sample.get("expected_label"))
        raw_prediction = predictions.get(sample_id)
        if raw_prediction is None:
            rows.append(
                {
                    "sample_id": sample_id,
                    "expected_label": expected,
                    "predicted_label": None,
                    "correct": False,
                    "available": False,
                    "reason": unavailable_reason or "missing_prediction",
                }
            )
            continue
        scored = _score_prediction(raw_prediction, threshold=threshold)
        predicted = scored["label"]
        available = bool(scored["available"])
        is_correct = bool(available and predicted == expected)
        if available:
            covered += 1
            if scored["confidence"] is not None:
                confidence_values.append(float(scored["confidence"]))
            if scored["latency_ms"] is not None:
                latency_values.append(float(scored["latency_ms"]))
        if is_correct:
            correct += 1
        rows.append(
            {
                "sample_id": sample_id,
                "expected_label": expected,
                "predicted_label": predicted,
                "score": scored["score"],
                "confidence": scored["confidence"],
                "latency_ms": scored["latency_ms"],
                "correct": is_correct,
                "available": available,
            }
        )
    total = len(samples)
    coverage = covered / total if total else 0.0
    accuracy = correct / covered if covered else None
    mean_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else None
    mean_latency = sum(latency_values) / len(latency_values) if latency_values else None
    status = "evaluated" if covered else "not_evaluated"
    return {
        "name": name,
        "status": status,
        "sample_count": total,
        "covered_samples": covered,
        "coverage": round(coverage, 4),
        "accuracy_on_covered": round(accuracy, 4) if accuracy is not None else None,
        "mean_confidence": round(mean_confidence, 4) if mean_confidence is not None else None,
        "mean_latency_ms": round(mean_latency, 3) if mean_latency is not None else None,
        "correct": correct,
        "rows": rows,
    }


def _current_predictions(
    samples: Sequence[Mapping[str, Any]],
    analyzer: Callable[[str], Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    predictions: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for sample in samples:
        sample_id = str(sample.get("id") or "")
        try:
            payload = dict(analyzer(str(sample.get("text") or "")))
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"{sample_id}:{type(exc).__name__}")
            continue
        predictions[sample_id] = payload
    return predictions, errors


def _candidate_recommendation(result: Mapping[str, Any], *, current_accuracy: float | None) -> str:
    coverage = _safe_float(result.get("coverage")) or 0.0
    accuracy = _safe_float(result.get("accuracy_on_covered"))
    if coverage <= 0.0:
        return "materialize_or_generate_offline_predictions"
    if accuracy is None:
        return "collect_more_eval_samples"
    if current_accuracy is not None and accuracy <= current_accuracy:
        return "do_not_prioritize"
    if coverage >= 0.8 and accuracy >= 0.7:
        return "consider_deeper_weekend_testing"
    return "collect_more_eval_samples"


def _is_prediction_ready_model(candidate: Mapping[str, Any]) -> bool:
    intended_use = str(candidate.get("intended_use") or "").strip()
    return (
        str(candidate.get("resource_type") or "").strip() == "model"
        and intended_use in {"offline_experiment", "candidate_baseline"}
    )


def _operator_action(
    *,
    deeper_candidates: Sequence[str],
    prediction_ready_models_missing: int,
    dataset_or_inspect_remaining: int,
    evaluated_candidates: int,
) -> str:
    if deeper_candidates:
        return "review_deeper_weekend_testing_candidates"
    if prediction_ready_models_missing:
        return "generate_offline_predictions_for_ready_hf_model_candidates"
    if dataset_or_inspect_remaining:
        return "review_remaining_dataset_or_inspect_candidates"
    if evaluated_candidates:
        return "review_benchmark_results_current_path_not_beaten"
    return "no_hf_sentiment_benchmark_action"


def build_huggingface_sentiment_benchmark(
    *,
    report_date: str,
    intake: Mapping[str, Any],
    samples_payload: Mapping[str, Any] | None = None,
    candidate_predictions_payload: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
    threshold: float = 0.15,
    current_analyzer: Callable[[str], Mapping[str, Any]] = _current_sentiment_prediction,
) -> dict[str, Any]:
    generated = generated_at.astimezone(UTC) if generated_at else _utc_now()
    samples = _samples_from_payload(samples_payload or {})
    candidate_predictions = _prediction_maps(candidate_predictions_payload or {})
    ready_candidates = _ready_candidates(intake)

    current_raw, current_errors = _current_predictions(samples, current_analyzer)
    current = _evaluate_predictions(
        name="current_sentiment_path",
        samples=samples,
        predictions=current_raw,
        threshold=threshold,
        unavailable_reason="current_sentiment_unavailable",
    )
    current["errors"] = current_errors
    current_accuracy = _safe_float(current.get("accuracy_on_covered"))

    candidates: list[dict[str, Any]] = []
    for candidate in ready_candidates:
        repo_id = str(candidate.get("repo_id") or candidate.get("hf_id") or "")
        predictions = candidate_predictions.get(repo_id, {})
        result = _evaluate_predictions(
            name=repo_id,
            samples=samples,
            predictions=predictions,
            threshold=threshold,
            unavailable_reason="offline_predictions_not_provided",
        )
        result.update(
            {
                "repo_id": repo_id,
                "resource_type": candidate.get("resource_type"),
                "license": candidate.get("license"),
                "intended_use": (candidate.get("intake") or {}).get("intended_use")
                if isinstance(candidate.get("intake"), Mapping)
                else None,
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
                "provider_authority": False,
                "recommendation": _candidate_recommendation(
                    result,
                    current_accuracy=current_accuracy,
                ),
            }
        )
        candidates.append(result)

    evaluated_candidates = [row for row in candidates if row["covered_samples"] > 0]
    prediction_ready_missing = [
        row
        for row in candidates
        if row["covered_samples"] == 0 and _is_prediction_ready_model(row)
    ]
    dataset_or_inspect_remaining = [
        row
        for row in candidates
        if row["covered_samples"] == 0 and not _is_prediction_ready_model(row)
    ]
    deeper = [
        row["repo_id"]
        for row in candidates
        if row.get("recommendation") == "consider_deeper_weekend_testing"
    ]
    status = "evaluated" if current["covered_samples"] or evaluated_candidates else "planned"
    if current_errors and not current["covered_samples"] and not evaluated_candidates and not ready_candidates:
        status = "blocked"
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "report_date": str(report_date),
        "generated_at": _iso(generated),
        "status": status,
        "summary": {
            "sample_count": len(samples),
            "ready_candidates": len(ready_candidates),
            "evaluated_candidates": len(evaluated_candidates),
            "candidates_needing_predictions": len(prediction_ready_missing),
            "dataset_or_inspect_candidates_remaining": len(dataset_or_inspect_remaining),
            "current_path_coverage": current["coverage"],
            "current_path_accuracy_on_covered": current["accuracy_on_covered"],
            "deeper_weekend_testing_candidates": deeper,
        },
        "samples": [
            {
                "id": row["id"],
                "expected_label": row["expected_label"],
                "symbol": row["symbol"],
                "source": row["source"],
            }
            for row in samples
        ],
        "current_sentiment_path": current,
        "candidate_results": candidates,
        "policy": {
            "metadata_only": True,
            "downloads_performed": False,
            "candidate_runtime_loading": False,
            "threshold": float(threshold),
        },
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
        "provider_authority": False,
        "research_only": True,
        "operator_action": _operator_action(
            deeper_candidates=deeper,
            prediction_ready_models_missing=len(prediction_ready_missing),
            dataset_or_inspect_remaining=len(dataset_or_inspect_remaining),
            evaluated_candidates=len(evaluated_candidates),
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--intake-json", type=Path, default=None)
    parser.add_argument("--samples-json", type=Path, default=None)
    parser.add_argument("--candidate-predictions-json", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = build_huggingface_sentiment_benchmark(
        report_date=str(args.report_date),
        intake=_read_json(args.intake_json or _default_intake_path()),
        samples_payload=_read_json(args.samples_json),
        candidate_predictions_payload=_read_json(args.candidate_predictions_json),
        threshold=float(args.threshold),
    )
    default_output, default_latest = _default_paths(str(args.report_date))
    output = args.output_json or default_output
    latest = args.latest_json or default_latest
    report.setdefault("paths", {})
    report["paths"].update({"dated": str(output), "latest": str(latest)})
    _write_json(output, report)
    _write_json(latest, report)
    sys.stdout.write(
        json.dumps(
            {
                "status": report["status"],
                "sample_count": report["summary"]["sample_count"],
                "ready_candidates": report["summary"]["ready_candidates"],
                "evaluated_candidates": report["summary"]["evaluated_candidates"],
                "output_json": str(output),
                "latest_json": str(latest),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
