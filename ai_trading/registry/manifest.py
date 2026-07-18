"""Model manifest validation helpers for governance artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, cast

from ai_trading.models.contracts import normalize_bar_timeframe


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
MARKET_REGIME_POLICY_SCHEMA_VERSION = "market_regime_policy.v1"
MARKET_REGIME_CLASSIFIER_ID = "day_sleeve_past_only_v1"


class ManifestValidationError(ValueError):
    """Raised when manifest metadata violates required schema constraints."""


@dataclass(frozen=True, slots=True)
class MarketRegimePolicyDecision:
    """Fail-closed decision produced by the shared live/replay evaluator."""

    allowed: bool
    declared: bool
    market_regime: str
    reason: str
    evidence_identity: str | None = None
    evidence_sha256: str | None = None


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{field} must be a mapping")
    return value


def _require_non_empty_str(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ManifestValidationError(f"{field} must be non-empty")
    return text


def _require_positive_int(value: Any, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"{field} must be an integer") from exc
    if parsed <= 0:
        raise ManifestValidationError(f"{field} must be > 0")
    return parsed


def _require_non_negative_int(value: Any, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise ManifestValidationError(f"{field} must be >= 0")
    return parsed


def _require_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ManifestValidationError(f"{field} must be boolean-like")


def _require_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ManifestValidationError(f"{field} must be finite")
    return parsed


def _optional_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    return _require_float(value, field=field)


def _require_float_range(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float:
    parsed = _require_float(value, field=field)
    if parsed < minimum or parsed > maximum:
        raise ManifestValidationError(f"{field} must be between {minimum} and {maximum}")
    return parsed


def _require_hash(value: Any, *, field: str) -> str:
    text = _require_non_empty_str(value, field=field).lower()
    if not _HEX64_RE.fullmatch(text):
        raise ManifestValidationError(f"{field} must be a 64-char lowercase hex digest")
    return text


def _require_utc_timestamp(value: Any, *, field: str) -> datetime:
    text = _require_non_empty_str(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestValidationError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ManifestValidationError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _policy_regime_metrics(
    row: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    support = _require_non_negative_int(row.get("support"), field=f"{field}.support")
    net_edge_bps = _optional_float(
        row.get("mean_post_cost_net_edge_bps"),
        field=f"{field}.mean_post_cost_net_edge_bps",
    )
    ranking = _require_mapping(
        row.get("ranking_separation"),
        field=f"{field}.ranking_separation",
    )
    ranking_separation_bps = _optional_float(
        ranking.get("high_minus_low_net_edge_bps"),
        field=f"{field}.ranking_separation.high_minus_low_net_edge_bps",
    )
    profitable_fold_ratio = _require_float_range(
        row.get("profitable_fold_ratio"),
        field=f"{field}.profitable_fold_ratio",
        minimum=0.0,
        maximum=1.0,
    )
    evidence_qualified = _require_bool(
        row.get("evidence_qualified"),
        field=f"{field}.evidence_qualified",
    )
    return {
        "support": support,
        "mean_post_cost_net_edge_bps": net_edge_bps,
        "ranking_separation_bps": ranking_separation_bps,
        "profitable_fold_ratio": profitable_fold_ratio,
        "evidence_qualified": evidence_qualified,
    }


def derive_market_regime_policy(
    walk_forward: Mapping[str, Any],
    *,
    generated_at: datetime | str,
    evidence_identity: str | None = None,
    evidence_sha256: str | None = None,
    max_age_days: int = 14,
) -> dict[str, Any]:
    """Derive an abstention-only shadow policy from governed walk-forward evidence."""

    source = _require_mapping(walk_forward, field="walk_forward")
    classifier = _require_non_empty_str(
        source.get("market_regime_classifier"),
        field="walk_forward.market_regime_classifier",
    )
    if classifier != MARKET_REGIME_CLASSIFIER_ID:
        raise ManifestValidationError(
            "walk_forward.market_regime_classifier is incompatible with serving"
        )
    aggregate = _require_mapping(source.get("aggregate"), field="walk_forward.aggregate")
    candidate_qualified = _require_bool(
        aggregate.get("evidence_qualified"),
        field="walk_forward.aggregate.evidence_qualified",
    )
    config = _require_mapping(source.get("config"), field="walk_forward.config")
    by_regime = _require_mapping(
        source.get("by_market_regime"),
        field="walk_forward.by_market_regime",
    )
    if not by_regime:
        raise ManifestValidationError("walk_forward.by_market_regime must be non-empty")

    min_total_support = _require_positive_int(
        config.get("min_trades"),
        field="walk_forward.config.min_trades",
    )
    min_support = max(25, int(math.ceil(min_total_support / len(by_regime))))
    min_edge_bps = max(
        0.0,
        _require_float(
            config.get("min_mean_net_edge_bps", 0.0),
            field="walk_forward.config.min_mean_net_edge_bps",
        ),
    )
    min_ranking_separation_bps = max(
        0.0,
        _require_float(
            config.get("min_ranking_separation_bps", 0.0),
            field="walk_forward.config.min_ranking_separation_bps",
        ),
    )
    min_profitable_fold_ratio = _require_float_range(
        config.get("min_profitable_fold_ratio"),
        field="walk_forward.config.min_profitable_fold_ratio",
        minimum=0.0,
        maximum=1.0,
    )
    generated = (
        generated_at.astimezone(UTC)
        if isinstance(generated_at, datetime) and generated_at.tzinfo is not None
        else _require_utc_timestamp(generated_at, field="generated_at")
    )
    if generated.tzinfo is None:
        raise ManifestValidationError("generated_at must include a timezone")
    generated_text = generated.astimezone(UTC).isoformat()
    canonical_evidence = json.dumps(source, sort_keys=True, separators=(",", ":"), default=str)
    resolved_hash = (
        _require_hash(evidence_sha256, field="evidence_sha256")
        if evidence_sha256 is not None
        else hashlib.sha256(canonical_evidence.encode("utf-8")).hexdigest()
    )
    resolved_identity = (
        _require_non_empty_str(evidence_identity, field="evidence_identity")
        if evidence_identity is not None
        else f"walk_forward:{resolved_hash[:16]}"
    )

    regimes: dict[str, dict[str, Any]] = {}
    allowed: list[str] = []
    abstained: list[str] = []
    for raw_regime, raw_row in sorted(by_regime.items(), key=lambda item: str(item[0])):
        regime = _require_non_empty_str(
            raw_regime,
            field="walk_forward.by_market_regime.key",
        ).lower()
        metrics = _policy_regime_metrics(
            _require_mapping(
                raw_row,
                field=f"walk_forward.by_market_regime.{regime}",
            ),
            field=f"walk_forward.by_market_regime.{regime}",
        )
        reasons: list[str] = []
        if not candidate_qualified:
            reasons.append("candidate_evidence_unqualified")
        if not metrics["evidence_qualified"]:
            reasons.append("regime_evidence_unqualified")
        if int(metrics["support"]) < min_support:
            reasons.append("insufficient_support")
        if (
            metrics["mean_post_cost_net_edge_bps"] is None
            or float(metrics["mean_post_cost_net_edge_bps"]) <= min_edge_bps
        ):
            reasons.append("nonpositive_or_below_minimum_net_edge")
        if (
            metrics["ranking_separation_bps"] is None
            or float(metrics["ranking_separation_bps"])
            <= min_ranking_separation_bps
        ):
            reasons.append("nonpositive_or_below_minimum_ranking_separation")
        if float(metrics["profitable_fold_ratio"]) < min_profitable_fold_ratio:
            reasons.append("unstable_profitable_folds")
        disposition = "observe" if not reasons else "abstain"
        (allowed if disposition == "observe" else abstained).append(regime)
        regimes[regime] = {
            **metrics,
            "disposition": disposition,
            "reasons": reasons,
        }

    policy = {
        "schema_version": MARKET_REGIME_POLICY_SCHEMA_VERSION,
        "market_regime_classifier": MARKET_REGIME_CLASSIFIER_ID,
        "governance_status": "shadow",
        "promotion_authority": False,
        "generated_at": generated_text,
        "max_age_days": _require_positive_int(max_age_days, field="max_age_days"),
        "evidence": {
            "identity": resolved_identity,
            "sha256": resolved_hash,
            "generated_at": generated_text,
            "candidate_evidence_qualified": candidate_qualified,
        },
        "criteria": {
            "min_support": min_support,
            "min_mean_post_cost_net_edge_bps": min_edge_bps,
            "min_ranking_separation_bps": min_ranking_separation_bps,
            "min_profitable_fold_ratio": min_profitable_fold_ratio,
        },
        "allowed_regimes": allowed,
        "abstained_regimes": abstained,
        "regimes": regimes,
    }
    return validate_market_regime_policy(policy)


def validate_market_regime_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a declared policy and recompute its evidence-backed dispositions."""

    source = _require_mapping(payload, field="market_regime_policy")
    schema_version = _require_non_empty_str(
        source.get("schema_version"),
        field="market_regime_policy.schema_version",
    )
    if schema_version != MARKET_REGIME_POLICY_SCHEMA_VERSION:
        raise ManifestValidationError("market_regime_policy.schema_version is unsupported")
    classifier = _require_non_empty_str(
        source.get("market_regime_classifier"),
        field="market_regime_policy.market_regime_classifier",
    )
    if classifier != MARKET_REGIME_CLASSIFIER_ID:
        raise ManifestValidationError(
            "market_regime_policy.market_regime_classifier is incompatible with serving"
        )
    governance_status = _require_non_empty_str(
        source.get("governance_status"),
        field="market_regime_policy.governance_status",
    ).lower()
    if governance_status != "shadow":
        raise ManifestValidationError("market_regime_policy.governance_status must be shadow")
    if _require_bool(
        source.get("promotion_authority"),
        field="market_regime_policy.promotion_authority",
    ):
        raise ManifestValidationError(
            "market_regime_policy.promotion_authority must be false"
        )
    generated_at = _require_utc_timestamp(
        source.get("generated_at"),
        field="market_regime_policy.generated_at",
    )
    max_age_days = _require_positive_int(
        source.get("max_age_days"),
        field="market_regime_policy.max_age_days",
    )
    evidence = _require_mapping(
        source.get("evidence"),
        field="market_regime_policy.evidence",
    )
    evidence_identity = _require_non_empty_str(
        evidence.get("identity"),
        field="market_regime_policy.evidence.identity",
    )
    evidence_sha256 = _require_hash(
        evidence.get("sha256"),
        field="market_regime_policy.evidence.sha256",
    )
    evidence_generated = _require_utc_timestamp(
        evidence.get("generated_at"),
        field="market_regime_policy.evidence.generated_at",
    )
    if evidence_generated != generated_at:
        raise ManifestValidationError(
            "market_regime_policy evidence and policy timestamps must match"
        )
    candidate_qualified = _require_bool(
        evidence.get("candidate_evidence_qualified"),
        field="market_regime_policy.evidence.candidate_evidence_qualified",
    )
    criteria = _require_mapping(
        source.get("criteria"),
        field="market_regime_policy.criteria",
    )
    min_support = _require_positive_int(
        criteria.get("min_support"),
        field="market_regime_policy.criteria.min_support",
    )
    min_edge_bps = max(
        0.0,
        _require_float(
            criteria.get("min_mean_post_cost_net_edge_bps"),
            field="market_regime_policy.criteria.min_mean_post_cost_net_edge_bps",
        ),
    )
    min_ranking_bps = max(
        0.0,
        _require_float(
            criteria.get("min_ranking_separation_bps"),
            field="market_regime_policy.criteria.min_ranking_separation_bps",
        ),
    )
    min_profitable_ratio = _require_float_range(
        criteria.get("min_profitable_fold_ratio"),
        field="market_regime_policy.criteria.min_profitable_fold_ratio",
        minimum=0.0,
        maximum=1.0,
    )
    regimes_raw = _require_mapping(
        source.get("regimes"),
        field="market_regime_policy.regimes",
    )
    if not regimes_raw:
        raise ManifestValidationError("market_regime_policy.regimes must be non-empty")

    regimes: dict[str, dict[str, Any]] = {}
    expected_allowed: list[str] = []
    expected_abstained: list[str] = []
    for raw_regime, raw_row in sorted(regimes_raw.items(), key=lambda item: str(item[0])):
        regime = _require_non_empty_str(
            raw_regime,
            field="market_regime_policy.regimes.key",
        ).lower()
        row = _require_mapping(
            raw_row,
            field=f"market_regime_policy.regimes.{regime}",
        )
        support = _require_non_negative_int(
            row.get("support"),
            field=f"market_regime_policy.regimes.{regime}.support",
        )
        edge = _optional_float(
            row.get("mean_post_cost_net_edge_bps"),
            field=f"market_regime_policy.regimes.{regime}.mean_post_cost_net_edge_bps",
        )
        ranking = _optional_float(
            row.get("ranking_separation_bps"),
            field=f"market_regime_policy.regimes.{regime}.ranking_separation_bps",
        )
        profitable_ratio = _require_float_range(
            row.get("profitable_fold_ratio"),
            field=f"market_regime_policy.regimes.{regime}.profitable_fold_ratio",
            minimum=0.0,
            maximum=1.0,
        )
        regime_qualified = _require_bool(
            row.get("evidence_qualified"),
            field=f"market_regime_policy.regimes.{regime}.evidence_qualified",
        )
        qualified = (
            candidate_qualified
            and regime_qualified
            and support >= min_support
            and edge is not None
            and edge > min_edge_bps
            and ranking is not None
            and ranking > min_ranking_bps
            and profitable_ratio >= min_profitable_ratio
        )
        expected_disposition = "observe" if qualified else "abstain"
        declared_disposition = _require_non_empty_str(
            row.get("disposition"),
            field=f"market_regime_policy.regimes.{regime}.disposition",
        ).lower()
        if declared_disposition != expected_disposition:
            raise ManifestValidationError(
                f"market_regime_policy.regimes.{regime}.disposition contradicts evidence"
            )
        reasons_raw = row.get("reasons", [])
        if not isinstance(reasons_raw, (list, tuple)):
            raise ManifestValidationError(
                f"market_regime_policy.regimes.{regime}.reasons must be a sequence"
            )
        normalized_row = {
            "support": support,
            "mean_post_cost_net_edge_bps": edge,
            "ranking_separation_bps": ranking,
            "profitable_fold_ratio": profitable_ratio,
            "evidence_qualified": regime_qualified,
            "disposition": expected_disposition,
            "reasons": [
                _require_non_empty_str(
                    reason,
                    field=f"market_regime_policy.regimes.{regime}.reasons[]",
                )
                for reason in reasons_raw
            ],
        }
        regimes[regime] = normalized_row
        (expected_allowed if qualified else expected_abstained).append(regime)

    allowed_raw = source.get("allowed_regimes")
    abstained_raw = source.get("abstained_regimes")
    if not isinstance(allowed_raw, (list, tuple)) or not isinstance(
        abstained_raw, (list, tuple)
    ):
        raise ManifestValidationError(
            "market_regime_policy allowed/abstained regimes must be sequences"
        )
    allowed = sorted(
        _require_non_empty_str(
            regime,
            field="market_regime_policy.allowed_regimes[]",
        ).lower()
        for regime in allowed_raw
    )
    abstained = sorted(
        _require_non_empty_str(
            regime,
            field="market_regime_policy.abstained_regimes[]",
        ).lower()
        for regime in abstained_raw
    )
    if allowed != expected_allowed or abstained != expected_abstained:
        raise ManifestValidationError(
            "market_regime_policy allowed/abstained regimes contradict evidence"
        )
    return {
        "schema_version": schema_version,
        "market_regime_classifier": classifier,
        "governance_status": governance_status,
        "promotion_authority": False,
        "generated_at": generated_at.isoformat(),
        "max_age_days": max_age_days,
        "evidence": {
            "identity": evidence_identity,
            "sha256": evidence_sha256,
            "generated_at": evidence_generated.isoformat(),
            "candidate_evidence_qualified": candidate_qualified,
        },
        "criteria": {
            "min_support": min_support,
            "min_mean_post_cost_net_edge_bps": min_edge_bps,
            "min_ranking_separation_bps": min_ranking_bps,
            "min_profitable_fold_ratio": min_profitable_ratio,
        },
        "allowed_regimes": allowed,
        "abstained_regimes": abstained,
        "regimes": regimes,
    }


def evaluate_market_regime_policy(
    policy: Mapping[str, Any] | None,
    *,
    market_regime: Any,
    now: datetime | None = None,
) -> MarketRegimePolicyDecision:
    """Evaluate an optional policy; any declared invalid policy abstains."""

    regime = str(market_regime or "").strip().lower() or "unknown"
    if policy is None:
        return MarketRegimePolicyDecision(
            allowed=True,
            declared=False,
            market_regime=regime,
            reason="legacy_policy_absent",
        )
    try:
        validated = validate_market_regime_policy(policy)
    except (ManifestValidationError, TypeError, ValueError):
        return MarketRegimePolicyDecision(
            allowed=False,
            declared=True,
            market_regime=regime,
            reason="policy_invalid",
        )
    evidence = cast(Mapping[str, Any], validated["evidence"])
    identity = str(evidence["identity"])
    sha256 = str(evidence["sha256"])
    evaluated_at = (now or datetime.now(UTC)).astimezone(UTC)
    generated_at = _require_utc_timestamp(
        validated["generated_at"],
        field="market_regime_policy.generated_at",
    )
    if generated_at > evaluated_at:
        return MarketRegimePolicyDecision(
            allowed=False,
            declared=True,
            market_regime=regime,
            reason="policy_from_future",
            evidence_identity=identity,
            evidence_sha256=sha256,
        )
    if evaluated_at - generated_at > timedelta(days=int(validated["max_age_days"])):
        return MarketRegimePolicyDecision(
            allowed=False,
            declared=True,
            market_regime=regime,
            reason="policy_stale",
            evidence_identity=identity,
            evidence_sha256=sha256,
        )
    regimes = cast(Mapping[str, Any], validated["regimes"])
    row = regimes.get(regime)
    if not isinstance(row, Mapping):
        return MarketRegimePolicyDecision(
            allowed=False,
            declared=True,
            market_regime=regime,
            reason="market_regime_unknown",
            evidence_identity=identity,
            evidence_sha256=sha256,
        )
    allowed = str(row.get("disposition") or "") == "observe"
    return MarketRegimePolicyDecision(
        allowed=allowed,
        declared=True,
        market_regime=regime,
        reason="regime_observe" if allowed else "regime_abstain",
        evidence_identity=identity,
        evidence_sha256=sha256,
    )


def validate_manifest_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize model manifest metadata payloads."""

    source = _require_mapping(payload, field="manifest_metadata")
    strategy = _require_non_empty_str(source.get("strategy"), field="strategy")
    rows = _require_positive_int(source.get("rows"), field="rows")
    lookback_days = _require_positive_int(source.get("lookback_days"), field="lookback_days")
    horizon_days = _require_positive_int(source.get("horizon_days"), field="horizon_days")
    embargo_days = _require_positive_int(source.get("embargo_days"), field="embargo_days")
    default_threshold = _require_float_range(
        source.get("default_threshold"),
        field="default_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    selected_threshold = _require_float_range(
        source.get("selected_threshold", default_threshold),
        field="selected_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    cost_floor_bps = _require_float_range(
        source.get("cost_floor_bps"),
        field="cost_floor_bps",
        minimum=0.0,
        maximum=10_000.0,
    )
    cost_model_version = _require_non_empty_str(
        source.get("cost_model_version"), field="cost_model_version"
    )
    feature_hash = _require_hash(source.get("feature_hash"), field="feature_hash")
    feature_contract_hash = str(source.get("feature_contract_hash") or "").strip()
    if feature_contract_hash:
        feature_contract_hash = _require_hash(
            feature_contract_hash,
            field="feature_contract_hash",
        )
    feature_contract_version = str(source.get("feature_contract_version") or "").strip()
    training_bar_timeframe = normalize_bar_timeframe(source.get("training_bar_timeframe"))
    required_bar_timeframe = normalize_bar_timeframe(source.get("required_bar_timeframe"))
    if not training_bar_timeframe:
        raise ManifestValidationError("training_bar_timeframe must be non-empty")
    if not required_bar_timeframe:
        raise ManifestValidationError("required_bar_timeframe must be non-empty")
    dataset_fingerprint = _require_hash(
        source.get("dataset_fingerprint"), field="dataset_fingerprint"
    )

    symbols_raw = source.get("symbols")
    if not isinstance(symbols_raw, (list, tuple)) or not symbols_raw:
        raise ManifestValidationError("symbols must be a non-empty sequence")
    symbols = [_require_non_empty_str(item, field="symbols[]") for item in symbols_raw]

    features_raw = source.get("feature_columns")
    if not isinstance(features_raw, (list, tuple)) or not features_raw:
        raise ManifestValidationError("feature_columns must be a non-empty sequence")
    feature_columns = [
        _require_non_empty_str(item, field="feature_columns[]") for item in features_raw
    ]

    thresholds_by_regime = _require_mapping(
        source.get("thresholds_by_regime"), field="thresholds_by_regime"
    )
    normalized_thresholds = {
        _require_non_empty_str(regime, field="thresholds_by_regime.key"): _require_float_range(
            threshold,
            field=f"thresholds_by_regime.{regime}",
            minimum=0.0,
            maximum=1.0,
        )
        for regime, threshold in thresholds_by_regime.items()
    }

    data_sources = _require_mapping(source.get("data_sources"), field="data_sources")
    normalized_sources = {
        "daily_source": _require_non_empty_str(
            data_sources.get("daily_source"), field="data_sources.daily_source"
        ),
        "minute_source": _require_non_empty_str(
            data_sources.get("minute_source"), field="data_sources.minute_source"
        ),
        "data_provenance": _require_non_empty_str(
            data_sources.get("data_provenance"), field="data_sources.data_provenance"
        ),
        "alpaca_data_feed": _require_non_empty_str(
            data_sources.get("alpaca_data_feed"), field="data_sources.alpaca_data_feed"
        ),
    }

    sensitivity = _require_mapping(source.get("sensitivity_sweep"), field="sensitivity_sweep")
    normalized_sensitivity = {
        "enabled": _require_bool(sensitivity.get("enabled"), field="sensitivity_sweep.enabled"),
        "gate": _require_bool(sensitivity.get("gate"), field="sensitivity_sweep.gate"),
        "summary": dict(
            _require_mapping(
                sensitivity.get("summary", {}),
                field="sensitivity_sweep.summary",
            )
        ),
    }

    normalized_regime_policy: dict[str, Any] | None = None
    if "market_regime_policy" in source:
        normalized_regime_policy = validate_market_regime_policy(
            _require_mapping(
                source.get("market_regime_policy"),
                field="market_regime_policy",
            )
        )

    normalized = {
        "strategy": strategy,
        "symbols": symbols,
        "rows": rows,
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "embargo_days": embargo_days,
        "feature_columns": feature_columns,
        "feature_hash": feature_hash,
        "feature_contract_version": feature_contract_version,
        "feature_contract_hash": feature_contract_hash,
        "training_bar_timeframe": training_bar_timeframe,
        "required_bar_timeframe": required_bar_timeframe,
        "default_threshold": default_threshold,
        "selected_threshold": selected_threshold,
        "thresholds_by_regime": normalized_thresholds,
        "cost_floor_bps": cost_floor_bps,
        "cost_model_version": cost_model_version,
        "data_sources": normalized_sources,
        "dataset_fingerprint": dataset_fingerprint,
        "sensitivity_sweep": normalized_sensitivity,
    }
    if normalized_regime_policy is not None:
        normalized["market_regime_policy"] = normalized_regime_policy
    return normalized
