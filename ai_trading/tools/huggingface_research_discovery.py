"""Discover Hugging Face models and datasets as research-only candidates.

This tool intentionally creates advisory research artifacts only. Hugging Face
metadata, cards, likes, downloads, or tags must never grant model-promotion,
provider, execution, canary, or live-capital authority.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.config.management import get_env
from ai_trading.config.managed_secrets import hydrate_managed_secrets
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

_ARTIFACT_TYPE = "huggingface_research_discovery"
_SCHEMA_VERSION = "1.0.0"
_DEFAULT_OUTPUT_DIR = "runtime/research_reports/huggingface"
_DEFAULT_QUERIES = (
    "finance time series",
    "stock market",
    "tabular anomaly detection",
    "drift detection",
    "calibration",
)
_FINANCE_TERMS = {
    "finance",
    "financial",
    "stock",
    "stocks",
    "market",
    "trading",
    "equity",
    "portfolio",
    "sentiment",
    "time-series",
    "timeseries",
}
_USEFUL_TERMS = {
    "tabular",
    "classification",
    "regression",
    "anomaly",
    "forecast",
    "forecasting",
    "drift",
    "calibration",
    "sentiment",
}
_RISKY_TERMS = {
    "future",
    "leakage",
    "label-leak",
    "lookahead",
    "synthetic",
    "backtest-only",
}
_PERMISSIVE_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc0-1.0",
    "openrail",
    "bigscience-openrail-m",
}


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
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    return _truthy(get_env(name, None, cast=str, resolve_aliases=False), default=default)


def _env_int(name: str, default: int) -> int:
    raw = get_env(name, None, cast=str, resolve_aliases=False)
    if raw in (None, ""):
        return default
    try:
        return max(0, int(str(raw).strip()))
    except (TypeError, ValueError):
        return default


def _csv(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    parts: list[str] = []
    values = [raw] if isinstance(raw, str) else list(raw)
    for value in values:
        for part in str(value or "").split(","):
            token = part.strip()
            if token:
                parts.append(token)
    return parts


def _default_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path(
        _DEFAULT_OUTPUT_DIR,
        default_relative=_DEFAULT_OUTPUT_DIR,
        for_write=True,
    )
    return (
        root / f"hf_discovery_{_compact_date(report_date)}.json",
        root.parent / "latest" / "hf_discovery_latest.json",
    )


def _extract_license(item: Mapping[str, Any]) -> str | None:
    card_data = item.get("cardData")
    if isinstance(card_data, Mapping):
        raw_license = card_data.get("license")
        if isinstance(raw_license, list) and raw_license:
            return str(raw_license[0]).strip().lower() or None
        if raw_license not in (None, ""):
            return str(raw_license).strip().lower()
    for tag in _csv(item.get("tags") if isinstance(item.get("tags"), list) else []):
        if tag.lower().startswith("license:"):
            return tag.split(":", 1)[1].strip().lower() or None
    return None


def _has_card(item: Mapping[str, Any]) -> bool:
    if isinstance(item.get("cardData"), Mapping):
        return True
    if item.get("has_card") is not None:
        return bool(item.get("has_card"))
    if item.get("card") is not None:
        return bool(item.get("card"))
    return False


def _repo_id(item: Mapping[str, Any]) -> str:
    for key in ("id", "modelId", "datasetId", "repo_id", "repoId"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _resource_type(item: Mapping[str, Any], fallback: str) -> str:
    raw = str(item.get("repo_type") or item.get("resource_type") or fallback or "model").lower()
    if raw in {"models", "model"}:
        return "model"
    if raw in {"datasets", "dataset"}:
        return "dataset"
    return raw


def _downloads(item: Mapping[str, Any]) -> int:
    for key in ("downloads", "downloadsAllTime"):
        try:
            return max(0, int(item.get(key) or 0))
        except (TypeError, ValueError):
            continue
    return 0


def _likes(item: Mapping[str, Any]) -> int:
    try:
        return max(0, int(item.get("likes") or 0))
    except (TypeError, ValueError):
        return 0


def _score_candidate(item: Mapping[str, Any], *, min_downloads: int) -> dict[str, Any]:
    repo = _repo_id(item).lower()
    tags = {tag.lower() for tag in _csv(item.get("tags") if isinstance(item.get("tags"), list) else [])}
    pipeline = str(item.get("pipeline_tag") or item.get("task") or "").strip().lower()
    searchable = set(tags)
    searchable.update(part for part in repo.replace("/", " ").replace("-", " ").split() if part)
    if pipeline:
        searchable.add(pipeline)
    license_value = _extract_license(item)
    has_card = _has_card(item)
    gated = bool(item.get("gated") not in (False, None, "", "false", "False"))
    downloads = _downloads(item)

    score = 0.0
    reasons: list[str] = []
    if searchable & _FINANCE_TERMS:
        score += 0.35
        reasons.append("finance_relevance")
    if searchable & _USEFUL_TERMS:
        score += 0.25
        reasons.append("useful_ml_task")
    if has_card:
        score += 0.15
        reasons.append("card_present")
    if license_value:
        score += 0.1
        reasons.append("license_present")
    if downloads >= max(0, min_downloads):
        score += 0.1
        reasons.append("download_threshold_met")
    if gated:
        score -= 0.15
        reasons.append("gated_access")
    if searchable & _RISKY_TERMS:
        score -= 0.2
        reasons.append("possible_leakage_or_synthetic_risk")
    score = max(0.0, min(1.0, score))

    leakage_risk = "high" if searchable & _RISKY_TERMS else "medium" if "stock" in searchable else "low"
    license_risk = "high"
    if license_value in _PERMISSIVE_LICENSES:
        license_risk = "low"
    elif license_value:
        license_risk = "medium"
    documentation_quality = "high" if has_card and license_value else "medium" if has_card else "low"
    usefulness = "high" if score >= 0.65 else "medium" if score >= 0.35 else "low"
    recommended_use = "ignore"
    if usefulness == "high" and license_risk != "high" and leakage_risk != "high" and not gated:
        recommended_use = "offline_experiment"
    elif usefulness in {"high", "medium"}:
        recommended_use = "inspect"
    return {
        "score": round(score, 4),
        "reasons": reasons,
        "finance_relevance": "high" if searchable & _FINANCE_TERMS else "medium" if score >= 0.35 else "low",
        "leakage_risk": leakage_risk,
        "license_risk": license_risk,
        "documentation_quality": documentation_quality,
        "usefulness": usefulness,
        "recommended_use": recommended_use,
    }


def _normalize_candidate(
    item: Mapping[str, Any],
    *,
    resource_type: str,
    min_downloads: int,
) -> dict[str, Any]:
    repo_id = _repo_id(item)
    normalized_type = _resource_type(item, resource_type)
    score = _score_candidate(item, min_downloads=min_downloads)
    candidate = {
        "repo_id": repo_id,
        "hf_id": repo_id,
        "repo_type": normalized_type,
        "resource_type": normalized_type,
        "author": str(item.get("author") or repo_id.split("/", 1)[0] if repo_id else ""),
        "pipeline_tag": item.get("pipeline_tag") or item.get("task"),
        "tags": _csv(item.get("tags") if isinstance(item.get("tags"), list) else []),
        "license": _extract_license(item),
        "downloads": _downloads(item),
        "likes": _likes(item),
        "card_present": _has_card(item),
        "model_card_present": bool(_has_card(item) and normalized_type == "model"),
        "dataset_card_present": bool(_has_card(item) and normalized_type == "dataset"),
        "gated": bool(item.get("gated") not in (False, None, "", "false", "False")),
        "private": bool(item.get("private", False)),
        "last_modified": item.get("lastModified") or item.get("last_modified") or item.get("updatedAt"),
        "sha": item.get("sha"),
        "revision": item.get("sha") or item.get("revision") or "main",
        "research_fit": score,
        "finance_relevance": score["finance_relevance"],
        "leakage_risk": score["leakage_risk"],
        "license_risk": score["license_risk"],
        "documentation_quality": score["documentation_quality"],
        "usefulness": score["usefulness"],
        "recommended_use": score["recommended_use"],
        "risk_notes": score["reasons"],
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
    }
    return candidate


def _offline_items(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("candidates")
    if not isinstance(raw, list):
        raw = payload.get("results")
    if not isinstance(raw, list):
        raw = payload.get("models")
    rows: list[dict[str, Any]] = []
    if isinstance(raw, list):
        rows.extend(dict(item) for item in raw if isinstance(item, Mapping))
    datasets = payload.get("datasets")
    if isinstance(datasets, list):
        rows.extend({**dict(item), "repo_type": "dataset"} for item in datasets if isinstance(item, Mapping))
    return rows


def _resolve_token() -> tuple[str, str | None]:
    secret_key = str(
        get_env("AI_TRADING_HF_TOKEN_SECRET_NAME", "", cast=str, resolve_aliases=False) or ""
    ).strip()
    hydration_error: str | None = None
    if secret_key:
        try:
            hydrate_managed_secrets(required_keys=(secret_key,))
        except RuntimeError as exc:
            hydration_error = f"{type(exc).__name__}: {exc}"
    token = str(
        get_env(secret_key, "", cast=str, resolve_aliases=False) if secret_key else ""
    ).strip()
    if not token:
        token = str(
            get_env("AI_TRADING_HF_TOKEN", "", cast=str, resolve_aliases=False)
            or get_env("HF_TOKEN", "", cast=str, resolve_aliases=False)
            or ""
        ).strip()
    return token, hydration_error


def _fetch_hf_json(url: str, token: str, timeout_s: float) -> Any:
    headers = {"User-Agent": "ai-trading-bot-hf-research/1.0"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url=url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # nosec B310 - HF metadata API
        return json.loads(response.read().decode("utf-8"))


def _hub_url(resource_type: str, query: str, limit: int) -> str:
    endpoint = "datasets" if resource_type == "dataset" else "models"
    params = urllib.parse.urlencode(
        {"search": query, "limit": max(1, int(limit)), "sort": "downloads", "direction": "-1"}
    )
    return f"https://huggingface.co/api/{endpoint}?{params}"


def _api_items(
    *,
    queries: Sequence[str],
    resource_types: Sequence[str],
    max_results: int,
    token: str,
    timeout_s: float,
    fetch_json: Callable[[str, str, float], Any] = _fetch_hf_json,
) -> tuple[list[dict[str, Any]], list[str]]:
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    per_query_limit = max(1, min(max_results, 100))
    for query in queries:
        for resource_type in resource_types:
            try:
                payload = fetch_json(_hub_url(resource_type, query, per_query_limit), token, timeout_s)
            except (
                OSError,
                TimeoutError,
                ValueError,
                urllib.error.URLError,
                urllib.error.HTTPError,
            ) as exc:
                errors.append(f"{resource_type}:{query}:{type(exc).__name__}")
                continue
            if isinstance(payload, list):
                for row in payload:
                    if isinstance(row, Mapping):
                        items.append({**dict(row), "repo_type": resource_type})
            elif isinstance(payload, Mapping):
                for row in _offline_items(payload):
                    items.append({**row, "repo_type": resource_type})
    return items, errors


def build_huggingface_research_discovery(
    *,
    report_date: str,
    queries: Sequence[str] | None = None,
    resource_types: Sequence[str] | None = None,
    offline_results: Mapping[str, Any] | None = None,
    enabled: bool = False,
    use_hf_api: bool = False,
    include_gated: bool = False,
    max_results: int = 25,
    min_downloads: int = 0,
    generated_at: datetime | None = None,
    timeout_s: float = 5.0,
    fetch_json: Callable[[str, str, float], Any] = _fetch_hf_json,
) -> dict[str, Any]:
    generated = generated_at.astimezone(UTC) if generated_at else _utc_now()
    query_list = _csv(queries or []) or list(_DEFAULT_QUERIES)
    normalized_types = [
        "dataset" if str(item).lower() in {"dataset", "datasets"} else "model"
        for item in (resource_types or ("model", "dataset"))
        if str(item).lower() not in {"all", ""}
    ]
    if not normalized_types:
        normalized_types = ["model", "dataset"]

    blocked_reasons: list[str] = []
    errors: list[str] = []
    token = ""
    hydration_error: str | None = None
    raw_items: list[dict[str, Any]] = []
    if offline_results:
        raw_items.extend(_offline_items(offline_results))
    elif not enabled:
        blocked_reasons.append("hf_research_disabled")
    elif use_hf_api:
        token, hydration_error = _resolve_token()
        if hydration_error:
            errors.append("managed_secret_hydration_failed")
        raw_items, api_errors = _api_items(
            queries=query_list,
            resource_types=normalized_types,
            max_results=max_results,
            token=token,
            timeout_s=timeout_s,
            fetch_json=fetch_json,
        )
        errors.extend(api_errors)
    else:
        blocked_reasons.append("hf_api_not_enabled")

    seen: set[tuple[str, str]] = set()
    candidates: list[dict[str, Any]] = []
    for item in raw_items:
        resource_type = _resource_type(item, "model")
        if resource_type not in set(normalized_types):
            continue
        candidate = _normalize_candidate(
            item,
            resource_type=resource_type,
            min_downloads=min_downloads,
        )
        key = (candidate["repo_id"], candidate["resource_type"])
        if not candidate["repo_id"] or key in seen:
            continue
        seen.add(key)
        if candidate["gated"] and not include_gated:
            candidate["recommended_use"] = "inspect"
            candidate["blocked_reasons"] = ["gated_access_not_allowed"]
        candidates.append(candidate)
    candidates.sort(
        key=lambda row: (
            0 if row.get("blocked_reasons") else 1,
            float(row.get("research_fit", {}).get("score", 0.0)),
            int(row.get("downloads") or 0),
            int(row.get("likes") or 0),
        ),
        reverse=True,
    )
    candidates = candidates[: max(0, int(max_results))]
    accepted = [
        row for row in candidates if str(row.get("recommended_use")) in {"offline_experiment", "candidate_baseline"}
    ]
    status = "discovered" if candidates else "empty"
    if blocked_reasons and not candidates:
        status = "disabled" if "hf_research_disabled" in blocked_reasons else "blocked"
    elif errors and not candidates:
        status = "unavailable"
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "report_date": str(report_date),
        "generated_at": _iso(generated),
        "status": status,
        "blocked_reasons": blocked_reasons,
        "errors": errors,
        "query": {
            "queries": query_list,
            "resource_types": normalized_types,
            "include_gated": bool(include_gated),
            "max_results": int(max_results),
            "min_downloads": int(min_downloads),
            "metadata_only": True,
            "use_hf_api": bool(use_hf_api),
        },
        "summary": {
            "raw_results": len(raw_items),
            "candidate_count": len(candidates),
            "accepted_for_offline_experiment": len(accepted),
            "blocked_candidate_count": sum(1 for row in candidates if row.get("blocked_reasons")),
            "gated_count": sum(1 for row in candidates if row.get("gated")),
            "model_count": sum(1 for row in candidates if row.get("resource_type") == "model"),
            "dataset_count": sum(1 for row in candidates if row.get("resource_type") == "dataset"),
            "top_blocker_reasons": sorted(
                {
                    str(reason)
                    for row in candidates
                    for reason in list(row.get("blocked_reasons") or [])
                }
            ),
        },
        "candidates": candidates,
        "token": {"configured": bool(token), "hydration_error": bool(hydration_error)},
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
        "provider_authority": False,
        "research_only": True,
        "operator_action": "review_candidates_for_offline_intake" if accepted else "no_hf_action_required",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--resource-type", action="append", default=[])
    parser.add_argument("--offline-results-json", type=Path, default=None)
    parser.add_argument("--use-hf-api", action="store_true")
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--include-gated", action="store_true")
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument("--min-downloads", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = build_huggingface_research_discovery(
        report_date=str(args.report_date),
        queries=list(args.query or []) or _csv(
            get_env("AI_TRADING_HF_QUERIES", "", cast=str, resolve_aliases=False)
        ),
        resource_types=list(args.resource_type or []) or ["model", "dataset"],
        offline_results=_read_json(args.offline_results_json),
        enabled=bool(args.enabled) or _env_bool("AI_TRADING_HF_RESEARCH_ENABLED", False),
        use_hf_api=bool(args.use_hf_api),
        include_gated=bool(args.include_gated),
        max_results=int(args.max_results if args.max_results is not None else _env_int("AI_TRADING_HF_MAX_RESULTS", 25)),
        min_downloads=int(args.min_downloads),
        timeout_s=float(args.timeout_s),
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
                "candidate_count": report["summary"]["candidate_count"],
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
