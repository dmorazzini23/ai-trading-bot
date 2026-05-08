"""Build a research-only order-type optimization artifact."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_trading.config.launch_profiles import launch_profile_payload, resolve_launch_profile
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _parse_ts(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_jsonl(path: Path | None, *, max_rows: int = 100_000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "path": str(path) if path is not None else None,
        "exists": bool(path is not None and path.exists()),
        "rows_read": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
    }
    if path is None or not path.exists():
        return [], stats
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        stats["read_error"] = True
        return [], stats
    with handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            stats["rows_read"] = int(stats["rows_read"]) + 1
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                stats["invalid_rows"] = int(stats["invalid_rows"]) + 1
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
                stats["valid_rows"] = int(stats["valid_rows"]) + 1
            else:
                stats["invalid_rows"] = int(stats["invalid_rows"]) + 1
            if len(rows) > max(1, int(max_rows)):
                rows.pop(0)
    return rows, stats


def _freshness(payload: Mapping[str, Any], *, max_age_hours: float, now: datetime) -> dict[str, Any]:
    generated = _parse_ts(payload.get("generated_at") or payload.get("as_of") or payload.get("timestamp"))
    age_hours = None
    if generated is not None:
        age_hours = max(0.0, (now - generated).total_seconds() / 3600.0)
    fresh = bool(generated is not None and generated <= now and age_hours is not None and age_hours <= max_age_hours)
    return {
        "fresh": fresh,
        "generated_at": generated.isoformat().replace("+00:00", "Z") if generated else None,
        "age_hours": age_hours,
        "max_age_hours": float(max_age_hours),
    }


def _symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("symbol") or row.get("ticker") or "UNKNOWN").strip().upper() or "UNKNOWN"


def _side(row: Mapping[str, Any]) -> str:
    token = str(row.get("side") or row.get("order_side") or "buy").strip().lower().replace("-", "_")
    if token in {"short", "sellshort", "sell_short"}:
        return "sell_short"
    if token in {"buy", "sell"}:
        return token
    return "unknown"


def _session(row: Mapping[str, Any]) -> str:
    return str(row.get("session_regime") or row.get("session") or "unknown").strip().lower() or "unknown"


def _volatility(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("volatility_bucket") or row.get("vol_bucket") or "").strip().lower()
    if explicit:
        return explicit
    spread = _to_float(row.get("spread_bps") or row.get("decision_spread_bps"))
    if spread is None:
        return "unknown"
    if spread <= 5.0:
        return "tight_spread"
    if spread <= 20.0:
        return "normal_spread"
    return "wide_spread"


def _order_options(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("order_type_options") or row.get("candidate_order_types")
    if isinstance(raw, list):
        options = [str(value).strip().lower().replace("-", "_") for value in raw if str(value).strip()]
    else:
        options = []
    current = str(row.get("order_type") or row.get("current_order_type") or "").strip().lower().replace("-", "_")
    if current:
        options.append(current)
    if not options:
        options = ["market", "limit", "marketable_limit"]
    return sorted(set(options))


def _cost_rows(live_cost_model: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = live_cost_model.get("by_symbol_side_session_order_type_volatility")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, Mapping)]
    fallback = live_cost_model.get("by_symbol_side_session")
    if isinstance(fallback, list):
        return [row for row in fallback if isinstance(row, Mapping)]
    return []


def _score_row(row: Mapping[str, Any]) -> float | None:
    for key in ("p90_total_cost_bps", "mean_total_cost_bps"):
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _matching_costs(
    *,
    cost_rows: list[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    option: str,
) -> list[Mapping[str, Any]]:
    symbol = _symbol(candidate)
    side = _side(candidate)
    session = _session(candidate)
    vol = _volatility(candidate)
    exact = [
        row
        for row in cost_rows
        if str(row.get("symbol") or "").upper() == symbol
        and str(row.get("side") or "").lower() == side
        and str(row.get("session_regime") or "").lower() == session
        and str(row.get("order_type") or "").lower() == option
        and str(row.get("volatility_bucket") or "").lower() in {vol, "", "unknown"}
    ]
    if exact:
        return exact
    return [
        row
        for row in cost_rows
        if str(row.get("symbol") or "").upper() == symbol
        and str(row.get("side") or "").lower() == side
        and str(row.get("session_regime") or "").lower() == session
        and str(row.get("order_type") or "").lower() == option
    ]


def build_order_type_optimizer(
    *,
    candidates: Sequence[Mapping[str, Any]],
    live_cost_model: Mapping[str, Any],
    max_cost_model_age_hours: float = 24.0,
    launch_profile_name: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Recommend order types for shadow research only."""

    generated = (now or datetime.now(UTC)).astimezone(UTC)
    profile = resolve_launch_profile(launch_profile_name)
    freshness = _freshness(live_cost_model, max_age_hours=max_cost_model_age_hours, now=generated)
    status_payload = live_cost_model.get("status") if isinstance(live_cost_model.get("status"), Mapping) else {}
    reasons: list[str] = []
    if not live_cost_model:
        reasons.append("live_cost_model_missing")
    elif not freshness["fresh"]:
        reasons.append("live_cost_model_stale")
    if status_payload and str(status_payload.get("status") or "").lower() not in {"ready", "ok"}:
        reasons.append("live_cost_model_not_ready")

    recommendations: list[dict[str, Any]] = []
    blocked_candidates: list[dict[str, Any]] = []
    rows = _cost_rows(live_cost_model)
    recommendations_enabled = not reasons
    if recommendations_enabled:
        for index, candidate in enumerate(candidates):
            candidate_reasons: list[str] = []
            symbol = _symbol(candidate)
            side = _side(candidate)
            if profile.allowed_symbols and symbol not in profile.allowed_symbols:
                candidate_reasons.append("symbol_not_allowed_by_launch_profile")
            if side == "sell_short" and not profile.shorts_allowed:
                candidate_reasons.append("shorts_not_allowed_by_launch_profile")
            quote_age = _to_float(candidate.get("quote_age_ms") or candidate.get("decision_quote_age_ms"))
            spread = _to_float(candidate.get("spread_bps") or candidate.get("decision_spread_bps"))
            if profile.max_quote_age_ms is not None and quote_age is not None and quote_age > profile.max_quote_age_ms:
                candidate_reasons.append("quote_age_exceeds_launch_profile")
            if profile.max_spread_bps is not None and spread is not None and spread > profile.max_spread_bps:
                candidate_reasons.append("spread_exceeds_launch_profile")
            scored: list[dict[str, Any]] = []
            for option in _order_options(candidate):
                matches = _matching_costs(cost_rows=rows, candidate=candidate, option=option)
                if not matches:
                    continue
                best = min(matches, key=lambda row: _score_row(row) if _score_row(row) is not None else math.inf)
                score = _score_row(best)
                if score is None:
                    continue
                scored.append(
                    {
                        "order_type": option,
                        "expected_cost_bps": score,
                        "sample_count": int(_to_float(best.get("sample_count")) or 0.0),
                        "sufficient_samples": bool(best.get("sufficient_samples", True)),
                    }
                )
            if candidate_reasons or not scored:
                blocked_candidates.append(
                    {
                        "index": index,
                        "symbol": symbol,
                        "side": side,
                        "blocked_reasons": candidate_reasons or ["no_matching_cost_model_bucket"],
                    }
                )
                continue
            best = min(scored, key=lambda item: (float(item["expected_cost_bps"]), str(item["order_type"])))
            recommendations.append(
                {
                    "index": index,
                    "symbol": symbol,
                    "side": side,
                    "session_regime": _session(candidate),
                    "volatility_bucket": _volatility(candidate),
                    "recommended_order_type": best["order_type"],
                    "expected_cost_bps": best["expected_cost_bps"],
                    "alternatives": scored,
                    "shadow_only": True,
                }
            )

    status = "blocked" if reasons else ("ready" if recommendations else "no_recommendations")
    return {
        "schema_version": "1.0.0",
        "artifact_type": "order_type_optimizer",
        "generated_at": generated.isoformat().replace("+00:00", "Z"),
        "mode": "research_shadow",
        "live_enabled": False,
        "status": status,
        "reasons": reasons,
        "recommendations_enabled": recommendations_enabled,
        "launch_profile": launch_profile_payload(profile),
        "freshness": {"live_cost_model": freshness},
        "summary": {
            "candidate_count": len(candidates),
            "recommendation_count": len(recommendations),
            "blocked_candidate_count": len(blocked_candidates),
        },
        "recommendations": recommendations,
        "blocked_candidates": blocked_candidates,
        "operator_note": "Research/shadow artifact only; stale data blocks recommendations and no live routing is changed.",
    }


def _default_path(path_value: str, *, for_write: bool = False) -> Path:
    return resolve_runtime_artifact_path(path_value, default_relative=path_value, for_write=for_write)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-jsonl", type=Path, default=None)
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-cost-model-age-hours", type=float, default=24.0)
    parser.add_argument("--launch-profile", default=None)
    args = parser.parse_args(argv)

    candidates, diagnostics = _read_jsonl(args.candidates_jsonl)
    live_cost_model = _read_json(
        args.live_cost_model_json or _default_path("runtime/live_cost_model_latest.json")
    )
    report = build_order_type_optimizer(
        candidates=candidates,
        live_cost_model=live_cost_model,
        max_cost_model_age_hours=max(0.0, float(args.max_cost_model_age_hours)),
        launch_profile_name=args.launch_profile,
    )
    report["sources"] = {"candidates_jsonl": diagnostics}
    output = args.output_json or _default_path("runtime/order_type_optimizer_latest.json", for_write=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output), "status": report["status"]}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
