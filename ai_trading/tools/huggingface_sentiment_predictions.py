"""Generate offline Hugging Face sentiment predictions for research benchmarks.

This tool is research-only. It may download/cache explicitly approved Hugging
Face models when downloads are enabled, but generated predictions never grant
runtime, promotion, provider, or live-money authority.
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

from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.tools import huggingface_sentiment_benchmark as benchmark

_ARTIFACT_TYPE = "huggingface_sentiment_predictions"
_SCHEMA_VERSION = "1.0.0"
_DEFAULT_OUTPUT_DIR = "runtime/research_reports/huggingface"
_DEFAULT_CACHE_DIR = "runtime/research_reports/huggingface/cache"

PipelineFactory = Callable[[str, Path, bool, str], Callable[[str], Any]]


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


def _truthy(raw: Any, *, default: bool = False) -> bool:
    if raw is None:
        return default
    token = str(raw).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return default


def _default_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path(
        _DEFAULT_OUTPUT_DIR,
        default_relative=_DEFAULT_OUTPUT_DIR,
        for_write=True,
    )
    return (
        root / f"hf_sentiment_predictions_{_compact_date(report_date)}.json",
        root.parent / "latest" / "hf_sentiment_predictions_latest.json",
    )


def _default_intake_path() -> Path:
    return resolve_runtime_artifact_path(
        "runtime/research_reports/latest/hf_candidate_intake_latest.json",
        default_relative="runtime/research_reports/latest/hf_candidate_intake_latest.json",
    )


def _default_cache_dir(raw: str | Path | None = None) -> Path:
    value = str(
        raw
        or get_env("AI_TRADING_HF_CACHE_DIR", _DEFAULT_CACHE_DIR, cast=str, resolve_aliases=False)
        or _DEFAULT_CACHE_DIR
    )
    return resolve_runtime_artifact_path(value, default_relative=_DEFAULT_CACHE_DIR, for_write=True)


def _candidate_repo_id(row: Mapping[str, Any]) -> str:
    return str(row.get("repo_id") or row.get("hf_id") or "").strip()


def _candidate_revision(row: Mapping[str, Any]) -> str:
    revision = str(row.get("sha") or row.get("revision") or "").strip()
    return "" if revision.lower() in {"", "main", "master", "latest"} else revision


def _approved_candidates(
    intake: Mapping[str, Any],
    *,
    candidate_ids: Sequence[str] | None = None,
    max_candidates: int = 3,
) -> list[dict[str, Any]]:
    wanted = {item.strip() for item in candidate_ids or [] if item.strip()}
    raw = intake.get("candidates")
    if not isinstance(raw, list):
        return []
    approved: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        repo_id = _candidate_repo_id(item)
        intake_details = item.get("intake")
        if wanted and repo_id not in wanted:
            continue
        if (
            item.get("status") == "ready_for_manual_review"
            and item.get("resource_type", "model") == "model"
            and isinstance(intake_details, Mapping)
            and bool(intake_details.get("materialization_allowed"))
            and not bool(intake_details.get("runtime_use_allowed"))
        ):
            approved.append(dict(item))
    return approved[: max(0, int(max_candidates))]


def _default_pipeline_factory(
    repo_id: str,
    cache_dir: Path,
    allow_downloads: bool,
    revision: str,
) -> Callable[[str], Any]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise RuntimeError("transformers_not_installed") from exc
    model = AutoModelForSequenceClassification.from_pretrained(
        repo_id,
        cache_dir=str(cache_dir),
        local_files_only=not allow_downloads,
        revision=revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        cache_dir=str(cache_dir),
        local_files_only=not allow_downloads,
        revision=revision,
    )
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        top_k=None,
        truncation=True,
    )


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


def _label_bucket(label: Any) -> str | None:
    token = str(label or "").strip().lower().replace("_", "-")
    if token in {"positive", "pos", "bullish", "bull"} or "positive" in token or "bullish" in token:
        return "pos"
    if token in {"negative", "neg", "bearish", "bear"} or "negative" in token or "bearish" in token:
        return "neg"
    if token in {"neutral", "neu"} or "neutral" in token:
        return "neu"
    return None


def _flatten_pipeline_output(output: Any) -> list[Mapping[str, Any]]:
    if isinstance(output, Mapping):
        return [output]
    if not isinstance(output, list):
        return []
    if output and isinstance(output[0], list):
        output = output[0]
    return [dict(row) for row in output if isinstance(row, Mapping)]


def _prediction_from_output(output: Any) -> dict[str, Any]:
    rows = _flatten_pipeline_output(output)
    scores = {"pos": 0.0, "neg": 0.0, "neu": 0.0}
    raw_labels: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("label") or "")
        score = _safe_float(row.get("score"))
        if score is None:
            continue
        bucket = _label_bucket(label)
        raw_labels.append({"label": label, "score": score, "bucket": bucket})
        if bucket is not None:
            scores[bucket] = max(scores[bucket], score)
    confidence = max(scores.values()) if any(value > 0.0 for value in scores.values()) else None
    label = "neutral"
    if confidence is not None:
        label = max(scores, key=scores.get)
        label = {"pos": "positive", "neg": "negative", "neu": "neutral"}[label]
    return {
        "label": label if confidence is not None else None,
        "score": float(scores["pos"] - scores["neg"]),
        "confidence": confidence,
        "pos": scores["pos"],
        "neg": scores["neg"],
        "neu": scores["neu"],
        "available": confidence is not None,
        "raw_labels": raw_labels,
    }


def _predict_samples(
    *,
    repo_id: str,
    revision: str,
    samples: Sequence[Mapping[str, Any]],
    cache_dir: Path,
    allow_downloads: bool,
    pipeline_factory: PipelineFactory,
) -> dict[str, Any]:
    started_model = time.perf_counter()
    try:
        classifier = pipeline_factory(repo_id, cache_dir, allow_downloads, revision)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "repo_id": repo_id,
            "revision": revision,
            "status": "blocked",
            "blocked_reasons": [f"pipeline_unavailable:{type(exc).__name__}"],
            "samples": [],
            "model_load_ms": round((time.perf_counter() - started_model) * 1000.0, 3),
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
            "provider_authority": False,
        }
    predictions: list[dict[str, Any]] = []
    failures = 0
    for sample in samples:
        sample_id = str(sample.get("id") or "")
        started = time.perf_counter()
        try:
            output = classifier(str(sample.get("text") or ""))
            prediction = _prediction_from_output(output)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            failures += 1
            prediction = {
                "label": None,
                "score": None,
                "confidence": None,
                "available": False,
                "error": type(exc).__name__,
            }
        prediction.update(
            {
                "sample_id": sample_id,
                "expected_label": benchmark._normalize_label(sample.get("expected_label")),
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        )
        predictions.append(prediction)
    covered = sum(1 for row in predictions if row.get("available"))
    status = "evaluated" if covered else "blocked"
    blocked_reasons = ["all_predictions_unavailable"] if not covered else []
    return {
        "repo_id": repo_id,
        "revision": revision,
        "status": status,
        "blocked_reasons": blocked_reasons,
        "sample_count": len(samples),
        "covered_samples": covered,
        "prediction_failures": failures,
        "model_load_ms": round((time.perf_counter() - started_model) * 1000.0, 3),
        "samples": predictions,
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
        "provider_authority": False,
    }


def build_huggingface_sentiment_predictions(
    *,
    report_date: str,
    intake: Mapping[str, Any],
    samples_payload: Mapping[str, Any] | None = None,
    cache_dir: Path,
    allow_downloads: bool = False,
    max_candidates: int = 3,
    candidate_ids: Sequence[str] | None = None,
    pipeline_factory: PipelineFactory = _default_pipeline_factory,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated = generated_at.astimezone(UTC) if generated_at else _utc_now()
    samples = benchmark._samples_from_payload(samples_payload or {})
    candidates = _approved_candidates(intake, candidate_ids=candidate_ids, max_candidates=max_candidates)
    blocked_reasons: list[str] = []
    if not allow_downloads:
        blocked_reasons.append("hf_downloads_disabled")
    predictions: list[dict[str, Any]] = []
    for candidate in candidates:
        repo_id = _candidate_repo_id(candidate)
        revision = _candidate_revision(candidate)
        if not allow_downloads:
            predictions.append(
                {
                    "repo_id": repo_id,
                    "revision": revision or None,
                    "status": "blocked",
                    "blocked_reasons": ["hf_downloads_disabled"],
                    "samples": [],
                    "runtime_authority": False,
                    "promotion_authority": False,
                    "live_money_authority": False,
                    "provider_authority": False,
                }
            )
            continue
        if not revision:
            predictions.append(
                {
                    "repo_id": repo_id,
                    "revision": None,
                    "status": "blocked",
                    "blocked_reasons": ["hf_revision_unpinned"],
                    "samples": [],
                    "runtime_authority": False,
                    "promotion_authority": False,
                    "live_money_authority": False,
                    "provider_authority": False,
                }
            )
            continue
        predictions.append(
            _predict_samples(
                repo_id=repo_id,
                revision=revision,
                samples=samples,
                cache_dir=cache_dir,
                allow_downloads=allow_downloads,
                pipeline_factory=pipeline_factory,
            )
        )
    evaluated = sum(1 for row in predictions if row.get("status") == "evaluated")
    blocked = sum(1 for row in predictions if row.get("status") == "blocked")
    status = "empty" if not candidates else "evaluated" if evaluated else "blocked"
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "report_date": str(report_date),
        "generated_at": _iso(generated),
        "status": status,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "cache_dir": str(cache_dir),
        "summary": {
            "selected_candidates": len(candidates),
            "evaluated_candidates": evaluated,
            "blocked_candidates": blocked,
            "sample_count": len(samples),
            "downloads_enabled": bool(allow_downloads),
        },
        "predictions": predictions,
        "policy": {
            "research_only": True,
            "downloads_require_explicit_opt_in": True,
            "candidate_runtime_loading": False,
        },
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
        "provider_authority": False,
        "research_only": True,
        "operator_action": "run_hf_sentiment_benchmark_with_predictions" if evaluated else "review_hf_prediction_blockers",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--intake-json", type=Path, default=None)
    parser.add_argument("--samples-json", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    allow_downloads = bool(args.allow_downloads) or _truthy(
        get_env("AI_TRADING_HF_ALLOW_DOWNLOADS", "0", cast=str, resolve_aliases=False)
    )
    report = build_huggingface_sentiment_predictions(
        report_date=str(args.report_date),
        intake=_read_json(args.intake_json or _default_intake_path()),
        samples_payload=_read_json(args.samples_json),
        cache_dir=_default_cache_dir(args.cache_dir),
        allow_downloads=allow_downloads,
        max_candidates=int(args.max_candidates),
        candidate_ids=list(args.candidate_id or []),
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
                "selected_candidates": report["summary"]["selected_candidates"],
                "evaluated_candidates": report["summary"]["evaluated_candidates"],
                "blocked_candidates": report["summary"]["blocked_candidates"],
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
