"""
Model promotion pipeline for shadow-to-production governance.

Manages the promotion process from shadow testing to production deployment
with performance validation and safety checks.
"""
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
import hashlib
import json
import math
import uuid
from ai_trading.logging import get_logger
from ai_trading.governance.paths import resolve_governance_base_path
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, cast
from ..model_registry import ModelRegistry
from ai_trading.config.management import get_env
from ai_trading.research.institutional_validation import (
    compute_risk_adjusted_scorecard,
    run_monte_carlo_trade_sequence_stress,
    run_purged_walk_forward_validation,
    run_regime_split_validation,
)
from ai_trading.utils.lazy_imports import load_pandas
logger = get_logger(__name__)

@dataclass
class PromotionCriteria:
    """Criteria for model promotion from shadow to production."""
    min_shadow_sessions: int = 5
    max_turnover_ratio: float = 1.5
    min_live_sharpe: float = 0.5
    max_drift_psi: float = 0.25
    min_shadow_days: int = 3
    max_drawdown_threshold: float = 0.05
    min_trade_count: int = 10
    min_live_sortino: float = 0.5
    min_live_calmar: float = 0.4
    max_tail_loss_95: float = 0.03
    max_risk_of_ruin: float = 0.20
    min_purged_walk_forward_pass_ratio: float = 0.60
    min_monte_carlo_p05_bps: float = -20.0
    min_regime_pass_ratio: float = 0.60
    require_tca_gate: bool = True
    max_reject_rate: float = 0.03
    max_execution_drift_bps: float = 25.0
    challenger_significance_alpha: float = 0.05
    min_challenger_uplift_bps: float = 0.5
    min_net_expectancy_bps: float = 0.0
    max_live_calibration_ece: float = 0.10
    max_live_calibration_brier: float = 0.30
    min_calibration_samples: int = 50
    challenger_sequential_min_samples: int = 20
    challenger_sequential_required_passes: int = 3
    control_band_drawdown: float = 0.08
    control_band_reject_rate: float = 0.05
    control_band_execution_drift_bps: float = 35.0
    control_band_drift_psi: float = 0.30
    control_band_live_calibration_ece: float = 0.15
    control_band_live_calibration_brier: float = 0.35

@dataclass
class PromotionMetrics:
    """Metrics collected during shadow testing."""
    sessions_completed: int = 0
    total_trades: int = 0
    turnover_ratio: float = 0.0
    live_sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    drift_psi: float = 0.0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    live_sortino_ratio: float = 0.0
    live_calmar_ratio: float = 0.0
    tail_loss_95: float = 0.0
    risk_of_ruin: float = 0.0
    purged_walk_forward_pass_ratio: float = 0.0
    monte_carlo_p05_bps: float = 0.0
    regime_pass_ratio: float = 0.0
    tca_gate_passed: bool = False
    reject_rate: float = 0.0
    execution_drift_bps: float = 0.0
    challenger_uplift_bps: float = 0.0
    challenger_p_value: float = 1.0
    gross_expectancy_bps: float = 0.0
    avg_cost_bps: float = 0.0
    net_expectancy_bps: float = 0.0
    live_calibration_ece: float = 1.0
    live_calibration_brier: float = 1.0
    calibration_samples: int = 0
    challenger_eval_samples: int = 0
    challenger_sequential_passes: int = 0
    last_updated: datetime | None = None

class ModelPromotion:
    """
    Model promotion manager for shadow-to-production workflow.

    Handles shadow testing, performance validation, and automatic
    promotion based on defined criteria.
    """

    def __init__(self, model_registry: ModelRegistry | None=None, criteria: PromotionCriteria | None=None, base_path: str | None=None):
        """
        Initialize model promotion manager.

        Args:
            model_registry: Model registry instance
            criteria: Promotion criteria
            base_path: Base path for governance artifacts
        """
        self.registry = model_registry or ModelRegistry()
        self.criteria = criteria or PromotionCriteria()
        self.base_path = resolve_governance_base_path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(f'{__name__}.{self.__class__.__name__}')
        self.active_dir = self.base_path / 'active'
        self.active_dir.mkdir(exist_ok=True)
        self._event_store: Any | None = None
        self._event_store_init_failed = False

    def _governance_event_path(self, filename: str) -> Path:
        return cast(Path, self.base_path / str(filename))

    def _append_jsonl_event(
        self,
        *,
        filename: str,
        payload: dict[str, Any],
        error_event: str,
    ) -> str | None:
        path = self._governance_event_path(filename)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
            return str(path)
        except OSError as exc:
            self.logger.error(
                error_event,
                extra={"path": str(path), "error": str(exc)},
            )
            return None

    def _snapshot_model_governance(self, model_id: str | None) -> dict[str, Any]:
        if model_id in (None, ""):
            return {}
        info = self.registry.model_index.get(str(model_id))
        if not isinstance(info, dict):
            return {}
        governance = info.get("governance")
        return dict(governance) if isinstance(governance, dict) else {}

    def _restore_model_governance(self, model_id: str | None, governance: dict[str, Any]) -> None:
        if model_id in (None, ""):
            return
        info = self.registry.model_index.get(str(model_id))
        if not isinstance(info, dict):
            return
        restored = dict(governance)
        model_dir = Path(info.get("path", self.registry.models_dir / str(model_id)))
        meta_payload = self.registry._read_metadata_file(model_dir) or {**info}
        info["governance"] = restored
        meta_payload["governance"] = restored
        self.registry.model_index[str(model_id)] = info
        self.registry._write_metadata_file(model_dir, meta_payload)
        self.registry._save_index()

    def _read_jsonl_tail(self, *, filename: str, limit: int = 20) -> list[dict[str, Any]]:
        path = self._governance_event_path(filename)
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        rows: list[dict[str, Any]] = []
        for raw in lines[-max(1, int(limit)):]:
            text = str(raw or "").strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
        return rows

    def _promotion_requires_approval(self) -> bool:
        try:
            return bool(get_env("AI_TRADING_PROMOTION_REQUIRE_APPROVAL", False, cast=bool))
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            return False

    def _promotion_approval_max_age_hours(self) -> float:
        try:
            value = float(get_env("AI_TRADING_PROMOTION_APPROVAL_MAX_AGE_HOURS", 168.0, cast=float))
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            value = 168.0
        if not math.isfinite(value):
            value = 168.0
        return max(1.0, min(value, 24.0 * 365.0))

    def _promotion_approval_future_skew_seconds(self) -> float:
        try:
            value = float(get_env("AI_TRADING_PROMOTION_APPROVAL_FUTURE_SKEW_SECONDS", 300.0, cast=float))
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            value = 300.0
        if not math.isfinite(value):
            value = 300.0
        return max(0.0, min(value, 24.0 * 3600.0))

    def _shadow_metrics_max_age_hours(self) -> float:
        try:
            value = float(get_env("AI_TRADING_PROMOTION_SHADOW_METRICS_MAX_AGE_HOURS", 24.0, cast=float))
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            value = 24.0
        if not math.isfinite(value):
            value = 24.0
        return max(1.0, min(value, 24.0 * 365.0))

    def _shadow_metrics_future_skew_seconds(self) -> float:
        try:
            value = float(get_env("AI_TRADING_PROMOTION_SHADOW_METRICS_FUTURE_SKEW_SECONDS", 300.0, cast=float))
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            value = 300.0
        if not math.isfinite(value):
            value = 300.0
        return max(0.0, min(value, 24.0 * 3600.0))

    def _shadow_metrics_freshness(self, metrics: PromotionMetrics) -> tuple[bool, dict[str, Any]]:
        max_age_hours = self._shadow_metrics_max_age_hours()
        future_skew_seconds = self._shadow_metrics_future_skew_seconds()
        if metrics.last_updated is None:
            return (
                False,
                {
                    "reason": "missing_last_updated",
                    "max_age_hours": max_age_hours,
                    "future_skew_seconds": future_skew_seconds,
                },
            )
        last_updated = metrics.last_updated
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=UTC)
        else:
            last_updated = last_updated.astimezone(UTC)
        now = datetime.now(UTC)
        delta_seconds = (now - last_updated).total_seconds()
        if delta_seconds < -future_skew_seconds:
            return (
                False,
                {
                    "reason": "future_last_updated",
                    "last_updated": last_updated.isoformat(),
                    "future_seconds": abs(delta_seconds),
                    "future_skew_seconds": future_skew_seconds,
                    "max_age_hours": max_age_hours,
                },
            )
        age_hours = max(delta_seconds, 0.0) / 3600.0
        if age_hours > max_age_hours:
            return (
                False,
                {
                    "reason": "stale_last_updated",
                    "last_updated": last_updated.isoformat(),
                    "age_hours": age_hours,
                    "max_age_hours": max_age_hours,
                    "future_skew_seconds": future_skew_seconds,
                },
            )
        return (
            True,
            {
                "last_updated": last_updated.isoformat(),
                "age_hours": age_hours,
                "max_age_hours": max_age_hours,
                "future_skew_seconds": future_skew_seconds,
            },
        )

    def _governance_event_store_enabled(self) -> bool:
        try:
            return bool(get_env("AI_TRADING_GOVERNANCE_EVENT_STORE_ENABLED", True, cast=bool))
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            return True

    def _resolve_event_store(self) -> Any | None:
        if not self._governance_event_store_enabled():
            return None
        if self._event_store is not None:
            return self._event_store
        if self._event_store_init_failed:
            return None
        try:
            from ai_trading.oms.event_store import EventStore

            self._event_store = EventStore()
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
            self._event_store_init_failed = True
            self.logger.warning(
                "GOVERNANCE_EVENT_STORE_INIT_FAILED",
                extra={"error": str(exc)},
            )
            return None
        return self._event_store

    @staticmethod
    def _json_default(value: Any) -> str:
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            try:
                return str(isoformat())
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                return str(value)
        return str(value)

    @staticmethod
    def _lineage_text(value: Any) -> str | None:
        if value in (None, ""):
            return None
        parsed = str(value).strip()
        return parsed or None

    @staticmethod
    def _explicit_bool_pass(value: Any) -> bool:
        if value is True:
            return True
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "pass", "passed"}
        return False

    def _extract_model_lineage(self, model_id: str | None) -> dict[str, Any]:
        resolved_model_id = str(model_id or "").strip()
        if not resolved_model_id:
            return {}
        model_info = dict(self.registry.model_index.get(resolved_model_id) or {})
        metadata_payload = dict(model_info.get("metadata", {}) or {})
        combined_meta: dict[str, Any] = {}
        try:
            _model_obj, loaded_meta = self.registry.load_model(
                resolved_model_id,
                verify_dataset_hash=False,
            )
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            loaded_meta = {}
        if isinstance(loaded_meta, dict):
            combined_meta.update(loaded_meta)
        governance_payload = dict(combined_meta.get("governance", {}) or {})

        def _first_text(*candidates: Any) -> str | None:
            for candidate in candidates:
                parsed = self._lineage_text(candidate)
                if parsed is not None:
                    return parsed
            return None

        model_hash = _first_text(
            model_info.get("model_hash"),
            combined_meta.get("model_hash"),
            metadata_payload.get("model_hash"),
        )
        lineage: dict[str, Any] = {
            "model_id": resolved_model_id,
        }
        strategy = _first_text(model_info.get("strategy"), combined_meta.get("strategy"))
        if strategy is not None:
            lineage["strategy"] = strategy
        registered_at = _first_text(
            model_info.get("registered_at"),
            combined_meta.get("registered_at"),
        )
        if registered_at is not None:
            lineage["registered_at"] = registered_at
        governance_status = _first_text(
            governance_payload.get("status"),
            model_info.get("governance", {}).get("status") if isinstance(model_info.get("governance"), dict) else None,
        )
        if governance_status is not None:
            lineage["governance_status"] = governance_status
        dataset_hash = _first_text(
            combined_meta.get("dataset_hash"),
            metadata_payload.get("dataset_hash"),
            combined_meta.get("dataset_fingerprint"),
            metadata_payload.get("dataset_fingerprint"),
            model_info.get("dataset_fingerprint"),
        )
        if dataset_hash is not None:
            lineage["dataset_hash"] = dataset_hash
        feature_version = _first_text(
            combined_meta.get("feature_version"),
            metadata_payload.get("feature_version"),
            combined_meta.get("feature_set_version"),
            metadata_payload.get("feature_set_version"),
            combined_meta.get("features_version"),
            metadata_payload.get("features_version"),
        )
        if feature_version is not None:
            lineage["feature_version"] = feature_version
        model_artifact_hash = _first_text(
            combined_meta.get("model_artifact_hash"),
            metadata_payload.get("model_artifact_hash"),
            combined_meta.get("artifact_hash"),
            metadata_payload.get("artifact_hash"),
            combined_meta.get("hash"),
            metadata_payload.get("hash"),
            model_hash,
        )
        if model_artifact_hash is not None:
            lineage["model_artifact_hash"] = model_artifact_hash
        if model_hash is not None:
            lineage["model_hash"] = model_hash
        policy_hash = _first_text(
            combined_meta.get("policy_hash"),
            metadata_payload.get("policy_hash"),
            combined_meta.get("effective_policy_hash"),
            metadata_payload.get("effective_policy_hash"),
            governance_payload.get("policy_hash"),
        )
        if policy_hash is not None:
            lineage["policy_hash"] = policy_hash
        config_snapshot_hash = _first_text(
            combined_meta.get("config_snapshot_hash"),
            metadata_payload.get("config_snapshot_hash"),
        )
        if config_snapshot_hash is not None:
            lineage["config_snapshot_hash"] = config_snapshot_hash
        return lineage

    def _append_governance_audit_event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        primary_lineage: dict[str, Any] | None = None,
        related_lineages: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        store = self._resolve_event_store()
        if store is None:
            return False
        event_payload = dict(payload)
        if primary_lineage:
            event_payload["lineage"] = dict(primary_lineage)
        if related_lineages:
            filtered_related = {
                str(key): dict(value)
                for key, value in related_lineages.items()
                if isinstance(value, dict) and value
            }
            if filtered_related:
                event_payload["related_lineages"] = filtered_related
        anchor_lineage = primary_lineage or {}
        if not anchor_lineage and related_lineages:
            for candidate in related_lineages.values():
                if isinstance(candidate, dict) and candidate:
                    anchor_lineage = candidate
                    break
        try:
            material = {
                "event_type": str(event_type or "").strip().upper(),
                "payload": event_payload,
            }
            idempotency_key = hashlib.sha256(
                json.dumps(material, sort_keys=True, default=self._json_default).encode("utf-8")
            ).hexdigest()
            return bool(
                store.append_oms_event_payload(
                    event_type=str(event_type or "").strip().upper() or "RECONCILE_UPDATE",
                    event_source="governance",
                    idempotency_key=idempotency_key,
                    event_ts=str(event_payload.get("ts") or datetime.now(UTC).isoformat()),
                    policy_hash=(
                        str(anchor_lineage.get("policy_hash"))
                        if anchor_lineage.get("policy_hash") not in (None, "")
                        else None
                    ),
                    model_hash=(
                        str(
                            anchor_lineage.get("model_hash")
                            or anchor_lineage.get("model_artifact_hash")
                        )
                        if (
                            anchor_lineage.get("model_hash")
                            or anchor_lineage.get("model_artifact_hash")
                        )
                        not in (None, "")
                        else None
                    ),
                    payload=event_payload,
                )
            )
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
            self.logger.warning(
                "GOVERNANCE_EVENT_APPEND_FAILED",
                extra={"event_type": str(event_type), "error": str(exc)},
            )
            return False

    def record_promotion_approval(
        self,
        *,
        strategy: str,
        model_id: str,
        approver: str,
        decision: str = "approved",
        note: str | None = None,
        ticket: str | None = None,
    ) -> str | None:
        """Append an explicit promotion approval/rejection checkpoint."""

        decision_token = str(decision or "").strip().lower()
        if decision_token not in {"approved", "rejected"}:
            raise ValueError("decision must be 'approved' or 'rejected'")
        ts = datetime.now(UTC).isoformat()
        payload: dict[str, Any] = {
            "approval_id": str(uuid.uuid4()),
            "ts": ts,
            "strategy": str(strategy),
            "model_id": str(model_id),
            "approver": str(approver),
            "decision": decision_token,
            "max_age_hours": float(self._promotion_approval_max_age_hours()),
        }
        note_text = str(note or "").strip()
        if note_text:
            payload["note"] = note_text
        ticket_text = str(ticket or "").strip()
        if ticket_text:
            payload["ticket"] = ticket_text
        output_path = self._append_jsonl_event(
            filename="promotion_approvals.jsonl",
            payload=payload,
            error_event="PROMOTION_APPROVAL_WRITE_FAILED",
        )
        self._append_governance_audit_event(
            event_type="GOVERNANCE_APPROVAL_RECORDED",
            payload=payload,
            primary_lineage=self._extract_model_lineage(model_id),
        )
        return output_path

    def _latest_promotion_approval(
        self,
        *,
        strategy: str,
        model_id: str,
    ) -> dict[str, Any] | None:
        rows = self._read_jsonl_tail(filename="promotion_approvals.jsonl", limit=500)
        strategy_token = str(strategy or "").strip()
        model_token = str(model_id or "").strip()
        for row in reversed(rows):
            if str(row.get("strategy") or "").strip() != strategy_token:
                continue
            if str(row.get("model_id") or "").strip() != model_token:
                continue
            return row
        return None

    def list_recent_promotion_approvals(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent promotion approval checkpoints."""

        return self._read_jsonl_tail(
            filename="promotion_approvals.jsonl",
            limit=max(1, int(limit)),
        )

    def list_recent_rollback_audit(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent rollback lineage/audit events."""

        return self._read_jsonl_tail(
            filename="rollback_audit.jsonl",
            limit=max(1, int(limit)),
        )

    def list_recent_champion_challenger_scorecards(
        self,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return recent champion/challenger scorecard summaries."""

        return self._read_jsonl_tail(
            filename="champion_challenger_scorecards.jsonl",
            limit=max(1, int(limit)),
        )

    def _live_kpi_breach_state_path(self) -> Path:
        try:
            raw = str(
                get_env(
                    "AI_TRADING_PROMOTION_LIVE_KPI_BREACH_STATE_PATH",
                    "",
                    cast=str,
                )
                or ""
            ).strip()
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            raw = ""
        if not raw:
            return cast(Path, self.base_path / "live_kpi_breach_state.json")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.base_path / path
        return path

    def _load_live_kpi_breach_state(self) -> dict[str, Any]:
        path = self._live_kpi_breach_state_path()
        if not path.exists():
            return {"version": 1, "strategies": {}}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.logger.warning(
                "LIVE_KPI_BREACH_STATE_LOAD_FAILED",
                extra={"path": str(path)},
            )
            return {"version": 1, "strategies": {}}
        if not isinstance(payload, dict):
            return {"version": 1, "strategies": {}}
        strategies = payload.get("strategies")
        if not isinstance(strategies, dict):
            payload["strategies"] = {}
        payload["version"] = 1
        return payload

    def _write_live_kpi_breach_state(self, payload: dict[str, Any]) -> None:
        path = self._live_kpi_breach_state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(f"{path.suffix}.tmp")
            tmp_path.write_text(
                json.dumps(payload, sort_keys=True, indent=2, default=self._json_default),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        except OSError as exc:
            self.logger.warning(
                "LIVE_KPI_BREACH_STATE_WRITE_FAILED",
                extra={"path": str(path), "error": str(exc)},
            )

    def _required_live_kpi_breach_count(self) -> int:
        try:
            value = int(
                get_env(
                    "AI_TRADING_PROMOTION_LIVE_KPI_BREACH_CONSECUTIVE_REQUIRED",
                    1,
                    cast=int,
                )
            )
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            value = 1
        return max(1, min(value, 30))

    def _update_live_kpi_breach_window(
        self,
        *,
        strategy: str,
        model_id: str | None,
        breaches: dict[str, Any],
        allow_rollback: bool,
    ) -> dict[str, Any]:
        state = self._load_live_kpi_breach_state()
        strategies = state.setdefault("strategies", {})
        if not isinstance(strategies, dict):
            strategies = {}
            state["strategies"] = strategies
        strategy_key = str(strategy)
        prior = strategies.get(strategy_key)
        if not isinstance(prior, dict):
            prior = {}
        failed_kpis = sorted(str(key) for key in breaches)
        signature_payload = {
            "model_id": str(model_id or ""),
            "failed_kpis": failed_kpis,
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        prior_model_id = str(prior.get("model_id") or "")
        prior_signature = str(prior.get("last_breach_signature") or "")
        if prior_model_id != str(model_id or ""):
            prior_count = 0
        else:
            try:
                prior_count = int(prior.get("consecutive_breach_count", 0) or 0)
            except (TypeError, ValueError):
                prior_count = 0
        duplicate_rollback_eval = (
            bool(allow_rollback)
            and prior_signature == signature
            and str(prior.get("last_status") or "") == "pending"
            and not bool(prior.get("last_allow_rollback", True))
            and prior_count > 0
        )
        breach_count = prior_count if duplicate_rollback_eval else prior_count + 1
        required = self._required_live_kpi_breach_count()
        now = datetime.now(UTC).isoformat()
        window = {
            "model_id": str(model_id or ""),
            "consecutive_breach_count": int(max(0, breach_count)),
            "required_consecutive_breaches": int(required),
            "failed_kpis": failed_kpis,
            "breaches": dict(breaches),
            "first_breach_ts": prior.get("first_breach_ts") if prior_count > 0 else now,
            "last_breach_ts": now,
            "last_breach_signature": signature,
            "last_status": "pending",
            "last_allow_rollback": bool(allow_rollback),
        }
        strategies[strategy_key] = window
        state["updated_at"] = now
        self._write_live_kpi_breach_state(state)
        return dict(window)

    def _clear_live_kpi_breach_window(
        self,
        *,
        strategy: str,
        model_id: str | None,
    ) -> None:
        state = self._load_live_kpi_breach_state()
        strategies = state.get("strategies")
        if not isinstance(strategies, dict):
            return
        prior = strategies.get(str(strategy))
        if not isinstance(prior, dict):
            return
        if str(prior.get("model_id") or "") != str(model_id or ""):
            return
        strategies.pop(str(strategy), None)
        state["updated_at"] = datetime.now(UTC).isoformat()
        self._write_live_kpi_breach_state(state)

    def _mark_live_kpi_window_status(
        self,
        *,
        strategy: str,
        status: str,
    ) -> None:
        state = self._load_live_kpi_breach_state()
        strategies = state.get("strategies")
        if not isinstance(strategies, dict):
            return
        window = strategies.get(str(strategy))
        if not isinstance(window, dict):
            return
        window["last_status"] = str(status)
        window["updated_at"] = datetime.now(UTC).isoformat()
        state["updated_at"] = window["updated_at"]
        self._write_live_kpi_breach_state(state)

    def _record_rollback_audit(
        self,
        *,
        strategy: str,
        status: str,
        reason: str,
        from_model_id: str | None = None,
        to_model_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "strategy": str(strategy),
            "status": str(status),
            "reason": str(reason),
            "from_model_id": str(from_model_id) if from_model_id not in (None, "") else None,
            "to_model_id": str(to_model_id) if to_model_id not in (None, "") else None,
        }
        if extra:
            for key, value in extra.items():
                payload[str(key)] = value
        output_path = self._append_jsonl_event(
            filename="rollback_audit.jsonl",
            payload=payload,
            error_event="ROLLBACK_AUDIT_WRITE_FAILED",
        )
        self._append_governance_audit_event(
            event_type="MODEL_ROLLBACK_RECORDED",
            payload=payload,
            primary_lineage=self._extract_model_lineage(from_model_id),
            related_lineages={
                "rollback_target": self._extract_model_lineage(to_model_id),
            },
        )
        return output_path

    @staticmethod
    def _coerce_returns(values: Any) -> list[float]:
        if not isinstance(values, list | tuple):
            return []
        cleaned: list[float] = []
        for raw in values:
            try:
                parsed = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed):
                cleaned.append(parsed)
        return cleaned

    def _derive_institutional_validation_metrics(
        self,
        session_stats: dict[str, Any],
    ) -> dict[str, Any]:
        auto_validation_enabled = bool(
            get_env("AI_TRADING_MODEL_GOVERNANCE_AUTO_VALIDATION_ENABLED", True, cast=bool)
        )
        if not auto_validation_enabled:
            return {}

        needs_pwf = "purged_walk_forward_pass_ratio" not in session_stats
        needs_regime = "regime_pass_ratio" not in session_stats
        needs_calibration = (
            "live_calibration_ece" not in session_stats
            or "live_calibration_brier" not in session_stats
            or "calibration_samples" not in session_stats
        )
        if not (needs_pwf or needs_regime or needs_calibration):
            return {}

        returns = self._coerce_returns(session_stats.get("returns"))
        if not returns:
            return {}

        pd = load_pandas()
        frame_source = session_stats.get("validation_frame")
        frame = None
        if hasattr(frame_source, "columns") and hasattr(frame_source, "copy"):
            try:
                frame = frame_source.copy()
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                frame = None
        elif isinstance(frame_source, list | tuple):
            try:
                frame = pd.DataFrame(list(frame_source))
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                frame = None
        elif isinstance(frame_source, dict):
            try:
                frame = pd.DataFrame(frame_source)
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                frame = None

        if frame is None or frame.empty:
            regimes_raw = session_stats.get("regimes")
            regimes: list[str]
            if isinstance(regimes_raw, list):
                regimes = []
                for idx in range(len(returns)):
                    token = ""
                    if idx < len(regimes_raw):
                        token = str(regimes_raw[idx] or "").strip()
                    regimes.append(token or "unknown")
            else:
                regimes = ["unknown"] * len(returns)
            frame = pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        end=datetime.now(UTC),
                        periods=len(returns),
                        freq="min",
                    ),
                    "ret": returns,
                    "regime": regimes,
                }
            )
        else:
            if "ret" not in frame.columns:
                for candidate_col in ("return", "returns", "post_cost_return", "pnl"):
                    if candidate_col in frame.columns:
                        frame["ret"] = frame[candidate_col]
                        break
            if "ret" not in frame.columns:
                sized_returns = returns[: len(frame)]
                if len(sized_returns) < len(frame):
                    sized_returns = sized_returns + ([0.0] * (len(frame) - len(sized_returns)))
                frame["ret"] = sized_returns
            if "timestamp" not in frame.columns:
                frame["timestamp"] = pd.date_range(
                    end=datetime.now(UTC),
                    periods=len(frame),
                    freq="min",
                )
            if "regime" not in frame.columns:
                frame["regime"] = "unknown"

        derived: dict[str, Any] = {}
        if needs_pwf:
            try:
                pwf_report = run_purged_walk_forward_validation(
                    frame,
                    return_col="ret",
                    timestamp_col="timestamp",
                    n_splits=max(
                        1,
                        int(get_env("AI_TRADING_MODEL_GOVERNANCE_PWF_SPLITS", 5, cast=int)),
                    ),
                    embargo_pct=max(
                        0.0,
                        float(
                            get_env(
                                "AI_TRADING_MODEL_GOVERNANCE_PWF_EMBARGO_PCT",
                                0.01,
                                cast=float,
                            )
                        ),
                    ),
                    purge_pct=max(
                        0.0,
                        float(
                            get_env(
                                "AI_TRADING_MODEL_GOVERNANCE_PWF_PURGE_PCT",
                                0.02,
                                cast=float,
                            )
                        ),
                    ),
                    min_fold_samples=max(
                        5,
                        int(
                            get_env(
                                "AI_TRADING_MODEL_GOVERNANCE_PWF_MIN_FOLD_SAMPLES",
                                20,
                                cast=int,
                            )
                        ),
                    ),
                )
                derived["purged_walk_forward_pass_ratio"] = float(
                    pwf_report.get("pass_ratio", 0.0) or 0.0
                )
                derived["purged_walk_forward_report"] = pwf_report
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                self.logger.debug("PROMOTION_PWF_AUTOVALIDATION_FAILED", exc_info=True)

        if needs_regime:
            try:
                regime_report = run_regime_split_validation(
                    frame,
                    regime_col="regime",
                    return_col="ret",
                    min_samples=max(
                        5,
                        int(
                            get_env(
                                "AI_TRADING_MODEL_GOVERNANCE_REGIME_MIN_SAMPLES",
                                30,
                                cast=int,
                            )
                        ),
                    ),
                    min_expectancy_bps=float(
                        get_env(
                            "AI_TRADING_MODEL_GOVERNANCE_REGIME_MIN_EXPECTANCY_BPS",
                            0.0,
                            cast=float,
                        )
                    ),
                    min_hit_rate=float(
                        get_env(
                            "AI_TRADING_MODEL_GOVERNANCE_REGIME_MIN_HIT_RATE",
                            0.45,
                            cast=float,
                        )
                    ),
                )
                derived["regime_pass_ratio"] = float(regime_report.get("pass_ratio", 0.0) or 0.0)
                derived["regime_split_report"] = regime_report
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                self.logger.debug("PROMOTION_REGIME_AUTOVALIDATION_FAILED", exc_info=True)

        if needs_calibration:
            try:
                probability_col: str | None = None
                label_col: str | None = None
                for candidate in (
                    "predicted_prob",
                    "prediction_probability",
                    "probability",
                    "prob",
                    "p_up",
                    "confidence",
                ):
                    if candidate in frame.columns:
                        probability_col = candidate
                        break
                for candidate in ("label", "target", "outcome", "y", "realized_label"):
                    if candidate in frame.columns:
                        label_col = candidate
                        break

                if probability_col is not None and label_col is not None:
                    probs: list[float] = []
                    labels: list[float] = []
                    for prob_raw, label_raw in zip(
                        frame[probability_col].tolist(),
                        frame[label_col].tolist(),
                        strict=False,
                    ):
                        try:
                            prob_val = float(prob_raw)
                            label_val = float(label_raw)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(prob_val) or not math.isfinite(label_val):
                            continue
                        prob_val = max(0.0, min(1.0, prob_val))
                        label_val = 1.0 if label_val > 0.5 else 0.0
                        probs.append(prob_val)
                        labels.append(label_val)
                    if probs and labels:
                        sample_count = len(probs)
                        brier = sum((p - y) ** 2 for p, y in zip(probs, labels, strict=False)) / float(
                            sample_count
                        )
                        bins = 10
                        ece = 0.0
                        for idx in range(bins):
                            lo = idx / float(bins)
                            hi = (idx + 1) / float(bins)
                            if idx == bins - 1:
                                members = [
                                    (p, y)
                                    for p, y in zip(probs, labels, strict=False)
                                    if lo <= p <= hi
                                ]
                            else:
                                members = [
                                    (p, y)
                                    for p, y in zip(probs, labels, strict=False)
                                    if lo <= p < hi
                                ]
                            if not members:
                                continue
                            prob_mean = sum(p for p, _ in members) / float(len(members))
                            acc_mean = sum(y for _, y in members) / float(len(members))
                            ece += (len(members) / float(sample_count)) * abs(prob_mean - acc_mean)

                        derived["live_calibration_ece"] = float(max(0.0, ece))
                        derived["live_calibration_brier"] = float(max(0.0, brier))
                        derived["calibration_samples"] = int(sample_count)
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                self.logger.debug("PROMOTION_CALIBRATION_AUTOVALIDATION_FAILED", exc_info=True)

        return derived

    def start_shadow_testing(self, model_id: str, benchmark_model_id: str | None=None) -> bool:
        """
        Start shadow testing for a model.

        Args:
            model_id: Model ID to put in shadow mode
            benchmark_model_id: Optional benchmark model for comparison

        Returns:
            True if shadow testing started successfully
        """
        _ = benchmark_model_id  # Reserved for future benchmark-vs-shadow comparisons.
        try:
            model_info = self.registry.model_index.get(model_id)
            if model_info is None:
                raise ValueError(f'Model {model_id} not found')
            strategy = model_info['strategy']
            shadow_models = self.registry.get_shadow_models(strategy)
            if shadow_models:
                self.logger.warning(f'Strategy {strategy} already has shadow models: {[m[0] for m in shadow_models]}')
            self.registry.update_governance_status(model_id, 'shadow')
            shadow_metrics = PromotionMetrics(last_updated=datetime.now(UTC))
            self._save_shadow_metrics(model_id, shadow_metrics)
            self.logger.info(f'Started shadow testing for model {model_id} (strategy: {strategy})')
            return True
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error starting shadow testing for {model_id}: {e}')
            return False

    def update_shadow_metrics(self, model_id: str, session_stats: dict[str, Any]) -> None:
        """
        Update shadow testing metrics.

        Args:
            model_id: Model ID in shadow testing
            session_stats: Statistics from latest trading session
        """
        try:
            session_payload = dict(session_stats or {})
            session_payload.update(
                self._derive_institutional_validation_metrics(session_payload)
            )
            current_metrics = self._load_shadow_metrics(model_id)
            if current_metrics is None:
                current_metrics = PromotionMetrics()
            current_metrics.sessions_completed += 1
            current_metrics.total_trades += int(session_payload.get('trade_count', 0) or 0)
            current_metrics.last_updated = datetime.now(UTC)
            alpha = 0.1
            new_turnover = float(session_payload.get('turnover_ratio', 0.0) or 0.0)
            current_metrics.turnover_ratio = alpha * new_turnover + (1 - alpha) * current_metrics.turnover_ratio
            new_sharpe = float(session_payload.get('sharpe_ratio', 0.0) or 0.0)
            current_metrics.live_sharpe_ratio = alpha * new_sharpe + (1 - alpha) * current_metrics.live_sharpe_ratio
            new_drawdown = float(session_payload.get('max_drawdown', 0.0) or 0.0)
            current_metrics.max_drawdown = max(current_metrics.max_drawdown, new_drawdown)
            new_psi = float(session_payload.get('drift_psi', 0.0) or 0.0)
            current_metrics.drift_psi = alpha * new_psi + (1 - alpha) * current_metrics.drift_psi
            new_latency = float(session_payload.get('avg_latency_ms', 0.0) or 0.0)
            current_metrics.avg_latency_ms = alpha * new_latency + (1 - alpha) * current_metrics.avg_latency_ms
            new_error_rate = float(session_payload.get('error_rate', 0.0) or 0.0)
            current_metrics.error_rate = alpha * new_error_rate + (1 - alpha) * current_metrics.error_rate

            returns = session_payload.get("returns")
            if isinstance(returns, list) and returns:
                scorecard = compute_risk_adjusted_scorecard(returns)
                current_metrics.live_sharpe_ratio = alpha * float(scorecard["sharpe_ratio"]) + (
                    1 - alpha
                ) * current_metrics.live_sharpe_ratio
                current_metrics.live_sortino_ratio = alpha * float(scorecard["sortino_ratio"]) + (
                    1 - alpha
                ) * current_metrics.live_sortino_ratio
                current_metrics.live_calmar_ratio = alpha * float(scorecard["calmar_ratio"]) + (
                    1 - alpha
                ) * current_metrics.live_calmar_ratio
                current_metrics.tail_loss_95 = alpha * float(scorecard["tail_loss_95"]) + (
                    1 - alpha
                ) * current_metrics.tail_loss_95
                current_metrics.risk_of_ruin = alpha * float(scorecard["risk_of_ruin"]) + (
                    1 - alpha
                ) * current_metrics.risk_of_ruin
                current_metrics.max_drawdown = max(
                    current_metrics.max_drawdown,
                    float(scorecard["max_drawdown"]),
                )
                monte = run_monte_carlo_trade_sequence_stress(returns, trials=300, seed=42)
                current_metrics.monte_carlo_p05_bps = alpha * float(monte["p05_return_bps"]) + (
                    1 - alpha
                ) * current_metrics.monte_carlo_p05_bps
                coerced_returns = self._coerce_returns(returns)
                gross_expectancy_bps = (
                    float(mean(coerced_returns) * 10000.0) if coerced_returns else 0.0
                )
                avg_cost_bps = float(
                    session_payload.get(
                        "avg_cost_bps",
                        session_payload.get(
                            "realized_slippage_bps",
                            session_payload.get("expected_cost_bps", current_metrics.avg_cost_bps),
                        ),
                    )
                    or 0.0
                )
                if not math.isfinite(avg_cost_bps):
                    avg_cost_bps = 0.0
                net_expectancy_bps = gross_expectancy_bps - avg_cost_bps
                current_metrics.gross_expectancy_bps = alpha * gross_expectancy_bps + (
                    1 - alpha
                ) * current_metrics.gross_expectancy_bps
                current_metrics.avg_cost_bps = alpha * max(0.0, avg_cost_bps) + (
                    1 - alpha
                ) * current_metrics.avg_cost_bps
                current_metrics.net_expectancy_bps = alpha * net_expectancy_bps + (
                    1 - alpha
                ) * current_metrics.net_expectancy_bps

            current_metrics.live_sortino_ratio = alpha * float(
                session_payload.get("sortino_ratio", current_metrics.live_sortino_ratio) or 0.0
            ) + (1 - alpha) * current_metrics.live_sortino_ratio
            current_metrics.live_calmar_ratio = alpha * float(
                session_payload.get("calmar_ratio", current_metrics.live_calmar_ratio) or 0.0
            ) + (1 - alpha) * current_metrics.live_calmar_ratio
            current_metrics.tail_loss_95 = alpha * float(
                session_payload.get("tail_loss_95", current_metrics.tail_loss_95) or 0.0
            ) + (1 - alpha) * current_metrics.tail_loss_95
            current_metrics.risk_of_ruin = alpha * float(
                session_payload.get("risk_of_ruin", current_metrics.risk_of_ruin) or 0.0
            ) + (1 - alpha) * current_metrics.risk_of_ruin
            current_metrics.purged_walk_forward_pass_ratio = alpha * float(
                session_payload.get(
                    "purged_walk_forward_pass_ratio",
                    current_metrics.purged_walk_forward_pass_ratio,
                )
                or 0.0
            ) + (1 - alpha) * current_metrics.purged_walk_forward_pass_ratio
            current_metrics.regime_pass_ratio = alpha * float(
                session_payload.get("regime_pass_ratio", current_metrics.regime_pass_ratio) or 0.0
            ) + (1 - alpha) * current_metrics.regime_pass_ratio
            current_metrics.monte_carlo_p05_bps = alpha * float(
                session_payload.get("monte_carlo_p05_bps", current_metrics.monte_carlo_p05_bps) or 0.0
            ) + (1 - alpha) * current_metrics.monte_carlo_p05_bps
            current_metrics.tca_gate_passed = self._explicit_bool_pass(
                session_payload.get("tca_gate_passed")
            )
            current_metrics.reject_rate = alpha * float(
                session_payload.get("reject_rate", current_metrics.reject_rate) or 0.0
            ) + (1 - alpha) * current_metrics.reject_rate
            current_metrics.execution_drift_bps = alpha * float(
                session_payload.get("execution_drift_bps", current_metrics.execution_drift_bps) or 0.0
            ) + (1 - alpha) * current_metrics.execution_drift_bps
            calibration_ece = float(
                session_payload.get("live_calibration_ece", current_metrics.live_calibration_ece)
                or current_metrics.live_calibration_ece
            )
            calibration_brier = float(
                session_payload.get("live_calibration_brier", current_metrics.live_calibration_brier)
                or current_metrics.live_calibration_brier
            )
            calibration_samples = int(
                session_payload.get("calibration_samples", current_metrics.calibration_samples)
                or current_metrics.calibration_samples
            )
            if math.isfinite(calibration_ece):
                current_metrics.live_calibration_ece = alpha * max(0.0, calibration_ece) + (
                    1 - alpha
                ) * current_metrics.live_calibration_ece
            if math.isfinite(calibration_brier):
                current_metrics.live_calibration_brier = alpha * max(0.0, calibration_brier) + (
                    1 - alpha
                ) * current_metrics.live_calibration_brier
            current_metrics.calibration_samples = max(0, calibration_samples)

            challenger_returns = session_payload.get("challenger_returns")
            champion_returns = session_payload.get("champion_returns")
            if isinstance(challenger_returns, list) and isinstance(champion_returns, list):
                stat_sig = self.evaluate_challenger_significance(
                    challenger_returns,
                    champion_returns,
                    alpha=self.criteria.challenger_significance_alpha,
                )
                current_metrics.challenger_uplift_bps = float(stat_sig["uplift_bps"])
                current_metrics.challenger_p_value = float(stat_sig["p_value"])
                challenger_sample_count = min(len(challenger_returns), len(champion_returns))
                current_metrics.challenger_eval_samples = int(max(challenger_sample_count, 0))
                sequential_pass = (
                    challenger_sample_count >= int(self.criteria.challenger_sequential_min_samples)
                    and float(stat_sig["uplift_bps"]) >= float(self.criteria.min_challenger_uplift_bps)
                    and float(stat_sig["p_value"]) <= float(self.criteria.challenger_significance_alpha)
                )
                if sequential_pass:
                    current_metrics.challenger_sequential_passes = int(
                        current_metrics.challenger_sequential_passes + 1
                    )
                else:
                    current_metrics.challenger_sequential_passes = 0

            self._save_shadow_metrics(model_id, current_metrics)
            metrics_dict = {
                'sessions_completed': current_metrics.sessions_completed,
                'total_trades': current_metrics.total_trades,
                'turnover_ratio': current_metrics.turnover_ratio,
                'live_sharpe_ratio': current_metrics.live_sharpe_ratio,
                'live_sortino_ratio': current_metrics.live_sortino_ratio,
                'live_calmar_ratio': current_metrics.live_calmar_ratio,
                'max_drawdown': current_metrics.max_drawdown,
                'tail_loss_95': current_metrics.tail_loss_95,
                'risk_of_ruin': current_metrics.risk_of_ruin,
                'drift_psi': current_metrics.drift_psi,
                'purged_walk_forward_pass_ratio': current_metrics.purged_walk_forward_pass_ratio,
                'monte_carlo_p05_bps': current_metrics.monte_carlo_p05_bps,
                'regime_pass_ratio': current_metrics.regime_pass_ratio,
                'tca_gate_passed': current_metrics.tca_gate_passed,
                'reject_rate': current_metrics.reject_rate,
                'execution_drift_bps': current_metrics.execution_drift_bps,
                'gross_expectancy_bps': current_metrics.gross_expectancy_bps,
                'avg_cost_bps': current_metrics.avg_cost_bps,
                'net_expectancy_bps': current_metrics.net_expectancy_bps,
                'live_calibration_ece': current_metrics.live_calibration_ece,
                'live_calibration_brier': current_metrics.live_calibration_brier,
                'calibration_samples': current_metrics.calibration_samples,
                'challenger_uplift_bps': current_metrics.challenger_uplift_bps,
                'challenger_p_value': current_metrics.challenger_p_value,
                'challenger_eval_samples': current_metrics.challenger_eval_samples,
                'challenger_sequential_passes': current_metrics.challenger_sequential_passes,
                'last_updated': current_metrics.last_updated.isoformat() if current_metrics.last_updated else None,
            }
            self.registry.update_governance_status(model_id, 'shadow', metrics_dict)
            self.logger.debug(f'Updated shadow metrics for model {model_id}: {current_metrics.sessions_completed} sessions')
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error updating shadow metrics for {model_id}: {e}')

    @staticmethod
    def evaluate_challenger_significance(
        challenger_returns: list[float],
        champion_returns: list[float],
        *,
        alpha: float = 0.05,
    ) -> dict[str, float | bool]:
        """Estimate challenger uplift significance using normal approximation."""

        challenger = [float(x) for x in challenger_returns if isinstance(x, int | float)]
        champion = [float(x) for x in champion_returns if isinstance(x, int | float)]
        if len(challenger) < 2 or len(champion) < 2:
            return {"uplift_bps": 0.0, "p_value": 1.0, "significant": False}

        mean_diff = mean(challenger) - mean(champion)
        var_ch = pstdev(challenger) ** 2 if len(challenger) > 1 else 0.0
        var_cp = pstdev(champion) ** 2 if len(champion) > 1 else 0.0
        denom = math.sqrt((var_ch / len(challenger)) + (var_cp / len(champion)))
        if denom <= 0:
            p_value = 0.0 if mean_diff > 0 else 1.0
        else:
            z = mean_diff / denom
            p_value = math.erfc(abs(z) / math.sqrt(2.0))
        uplift_bps = float(mean_diff * 10000.0)
        return {
            "uplift_bps": uplift_bps,
            "p_value": float(max(0.0, min(1.0, p_value))),
            "significant": bool(p_value <= max(0.0, min(1.0, alpha))),
        }

    def check_promotion_eligibility(self, model_id: str) -> tuple[bool, dict[str, Any]]:
        """
        Check if model is eligible for promotion.

        Args:
            model_id: Model ID to check

        Returns:
            Tuple of (eligible, evaluation_details)
        """
        try:
            metrics = self._load_shadow_metrics(model_id)
            if metrics is None:
                return (False, {'error': 'No shadow metrics found'})
            metrics_fresh, metrics_freshness = self._shadow_metrics_freshness(metrics)
            model_data = self.registry.load_model(model_id, verify_dataset_hash=False)[1]
            governance = model_data.get('governance', {})
            shadow_start = governance.get('shadow_start_time')
            if shadow_start:
                shadow_start_dt = datetime.fromisoformat(shadow_start.replace('Z', '+00:00'))
                days_in_shadow = (datetime.now(UTC) - shadow_start_dt).days
            else:
                days_in_shadow = 0
            policy_min_oos_samples = max(
                1,
                int(
                    get_env(
                        "AI_TRADING_POLICY_PROMOTION_MIN_OOS_SAMPLES",
                        self.criteria.challenger_sequential_min_samples,
                        cast=int,
                    )
                ),
            )
            policy_min_oos_net_bps = float(
                get_env(
                    "AI_TRADING_POLICY_PROMOTION_MIN_OOS_NET_BPS",
                    self.criteria.min_net_expectancy_bps,
                    cast=float,
                )
            )
            checks = {
                'shadow_metrics_freshness': metrics_fresh,
                'min_sessions': metrics.sessions_completed >= self.criteria.min_shadow_sessions,
                'min_days': days_in_shadow >= self.criteria.min_shadow_days,
                'min_trades': metrics.total_trades >= self.criteria.min_trade_count,
                'turnover_check': metrics.turnover_ratio <= self.criteria.max_turnover_ratio,
                'sharpe_check': metrics.live_sharpe_ratio >= self.criteria.min_live_sharpe,
                'sortino_check': metrics.live_sortino_ratio >= self.criteria.min_live_sortino,
                'calmar_check': metrics.live_calmar_ratio >= self.criteria.min_live_calmar,
                'drawdown_check': metrics.max_drawdown <= self.criteria.max_drawdown_threshold,
                'tail_loss_check': metrics.tail_loss_95 <= self.criteria.max_tail_loss_95,
                'risk_of_ruin_check': metrics.risk_of_ruin <= self.criteria.max_risk_of_ruin,
                'drift_check': metrics.drift_psi <= self.criteria.max_drift_psi,
                'purged_walk_forward_check': (
                    metrics.purged_walk_forward_pass_ratio >= self.criteria.min_purged_walk_forward_pass_ratio
                ),
                'monte_carlo_check': metrics.monte_carlo_p05_bps >= self.criteria.min_monte_carlo_p05_bps,
                'regime_split_check': metrics.regime_pass_ratio >= self.criteria.min_regime_pass_ratio,
                'tca_gate_check': (not self.criteria.require_tca_gate) or metrics.tca_gate_passed,
                'reject_rate_check': metrics.reject_rate <= self.criteria.max_reject_rate,
                'execution_drift_check': (
                    metrics.execution_drift_bps <= self.criteria.max_execution_drift_bps
                ),
                'net_expectancy_check': (
                    metrics.net_expectancy_bps >= self.criteria.min_net_expectancy_bps
                ),
                'calibration_ece_check': (
                    metrics.calibration_samples >= self.criteria.min_calibration_samples
                    and metrics.live_calibration_ece <= self.criteria.max_live_calibration_ece
                ),
                'calibration_brier_check': (
                    metrics.calibration_samples >= self.criteria.min_calibration_samples
                    and metrics.live_calibration_brier <= self.criteria.max_live_calibration_brier
                ),
                'challenger_significance_check': (
                    metrics.challenger_uplift_bps >= self.criteria.min_challenger_uplift_bps
                    and metrics.challenger_p_value <= self.criteria.challenger_significance_alpha
                ),
                'challenger_sequential_check': (
                    metrics.challenger_sequential_passes
                    >= self.criteria.challenger_sequential_required_passes
                ),
                'policy_oos_samples_check': (
                    metrics.challenger_eval_samples >= policy_min_oos_samples
                ),
                'policy_oos_net_check': (
                    metrics.net_expectancy_bps >= policy_min_oos_net_bps
                ),
            }
            eligible = all(checks.values())
            evaluation = {
                'eligible': eligible,
                'checks': checks,
                'metrics': {
                    'sessions_completed': metrics.sessions_completed,
                    'days_in_shadow': days_in_shadow,
                    'total_trades': metrics.total_trades,
                    'turnover_ratio': metrics.turnover_ratio,
                    'live_sharpe_ratio': metrics.live_sharpe_ratio,
                    'live_sortino_ratio': metrics.live_sortino_ratio,
                    'live_calmar_ratio': metrics.live_calmar_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'tail_loss_95': metrics.tail_loss_95,
                    'risk_of_ruin': metrics.risk_of_ruin,
                    'drift_psi': metrics.drift_psi,
                    'purged_walk_forward_pass_ratio': metrics.purged_walk_forward_pass_ratio,
                    'monte_carlo_p05_bps': metrics.monte_carlo_p05_bps,
                    'regime_pass_ratio': metrics.regime_pass_ratio,
                    'tca_gate_passed': metrics.tca_gate_passed,
                    'reject_rate': metrics.reject_rate,
                    'execution_drift_bps': metrics.execution_drift_bps,
                    'gross_expectancy_bps': metrics.gross_expectancy_bps,
                    'avg_cost_bps': metrics.avg_cost_bps,
                    'net_expectancy_bps': metrics.net_expectancy_bps,
                    'live_calibration_ece': metrics.live_calibration_ece,
                    'live_calibration_brier': metrics.live_calibration_brier,
                    'calibration_samples': metrics.calibration_samples,
                    'challenger_uplift_bps': metrics.challenger_uplift_bps,
                    'challenger_p_value': metrics.challenger_p_value,
                    'challenger_eval_samples': metrics.challenger_eval_samples,
                    'challenger_sequential_passes': metrics.challenger_sequential_passes,
                    'policy_min_oos_samples': policy_min_oos_samples,
                    'policy_min_oos_net_bps': policy_min_oos_net_bps,
                    'last_updated': metrics_freshness.get('last_updated'),
                    'freshness': metrics_freshness,
                },
                'criteria': {
                    'min_sessions': self.criteria.min_shadow_sessions,
                    'min_days': self.criteria.min_shadow_days,
                    'min_trades': self.criteria.min_trade_count,
                    'max_turnover': self.criteria.max_turnover_ratio,
                    'min_sharpe': self.criteria.min_live_sharpe,
                    'min_sortino': self.criteria.min_live_sortino,
                    'min_calmar': self.criteria.min_live_calmar,
                    'max_drawdown': self.criteria.max_drawdown_threshold,
                    'max_tail_loss_95': self.criteria.max_tail_loss_95,
                    'max_risk_of_ruin': self.criteria.max_risk_of_ruin,
                    'max_drift': self.criteria.max_drift_psi,
                    'min_purged_walk_forward_pass_ratio': self.criteria.min_purged_walk_forward_pass_ratio,
                    'min_monte_carlo_p05_bps': self.criteria.min_monte_carlo_p05_bps,
                    'min_regime_pass_ratio': self.criteria.min_regime_pass_ratio,
                    'max_reject_rate': self.criteria.max_reject_rate,
                    'max_execution_drift_bps': self.criteria.max_execution_drift_bps,
                    'min_net_expectancy_bps': self.criteria.min_net_expectancy_bps,
                    'max_live_calibration_ece': self.criteria.max_live_calibration_ece,
                    'max_live_calibration_brier': self.criteria.max_live_calibration_brier,
                    'min_calibration_samples': self.criteria.min_calibration_samples,
                    'challenger_significance_alpha': self.criteria.challenger_significance_alpha,
                    'min_challenger_uplift_bps': self.criteria.min_challenger_uplift_bps,
                    'challenger_sequential_min_samples': self.criteria.challenger_sequential_min_samples,
                    'challenger_sequential_required_passes': self.criteria.challenger_sequential_required_passes,
                    'policy_min_oos_samples': policy_min_oos_samples,
                    'policy_min_oos_net_bps': policy_min_oos_net_bps,
                },
            }
            return (eligible, evaluation)
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error checking promotion eligibility for {model_id}: {e}')
            return (False, {'error': str(e)})

    def promote_to_production(self, model_id: str, force: bool=False) -> bool:
        """
        Promote model to production.

        Args:
            model_id: Model ID to promote
            force: Force promotion without eligibility check

        Returns:
            True if promotion successful
        """
        try:
            model_info = self.registry.model_index.get(model_id)
            if model_info is None:
                raise ValueError(f'Model {model_id} not found')
            strategy = model_info['strategy']
            model_lineage = self._extract_model_lineage(model_id)
            if not force:
                eligible, details = self.check_promotion_eligibility(model_id)
                if not eligible:
                    self.logger.warning(f'Model {model_id} not eligible for promotion: {details}')
                    self._append_governance_audit_event(
                        event_type="MODEL_PROMOTION_BLOCKED",
                        payload={
                            "ts": datetime.now(UTC).isoformat(),
                            "strategy": str(strategy),
                            "model_id": str(model_id),
                            "force": bool(force),
                            "reason": "eligibility_failed",
                            "details": details,
                        },
                        primary_lineage=model_lineage,
                    )
                    return False
            approval_payload: dict[str, Any] | None = None
            if (not force) and self._promotion_requires_approval():
                approval_payload = self._latest_promotion_approval(
                    strategy=str(strategy),
                    model_id=str(model_id),
                )
                if not isinstance(approval_payload, dict):
                    self.logger.warning(
                        "PROMOTION_APPROVAL_REQUIRED_MISSING",
                        extra={"strategy": strategy, "model_id": model_id},
                    )
                    self._append_governance_audit_event(
                        event_type="MODEL_PROMOTION_BLOCKED",
                        payload={
                            "ts": datetime.now(UTC).isoformat(),
                            "strategy": str(strategy),
                            "model_id": str(model_id),
                            "force": bool(force),
                            "reason": "approval_missing",
                        },
                        primary_lineage=model_lineage,
                    )
                    return False
                decision_token = str(approval_payload.get("decision") or "").strip().lower()
                if decision_token != "approved":
                    self.logger.warning(
                        "PROMOTION_APPROVAL_REQUIRED_REJECTED",
                        extra={
                            "strategy": strategy,
                            "model_id": model_id,
                            "decision": decision_token or "unknown",
                        },
                    )
                    self._append_governance_audit_event(
                        event_type="MODEL_PROMOTION_BLOCKED",
                        payload={
                            "ts": datetime.now(UTC).isoformat(),
                            "strategy": str(strategy),
                            "model_id": str(model_id),
                            "force": bool(force),
                            "reason": "approval_rejected",
                            "approval": dict(approval_payload),
                        },
                        primary_lineage=model_lineage,
                    )
                    return False
                approved_ts_raw = str(approval_payload.get("ts") or "").strip()
                approved_ts: datetime | None = None
                if approved_ts_raw:
                    approved_ts_text = (
                        f"{approved_ts_raw[:-1]}+00:00"
                        if approved_ts_raw.endswith("Z")
                        else approved_ts_raw
                    )
                    try:
                        approved_ts = datetime.fromisoformat(approved_ts_text)
                    except ValueError:
                        approved_ts = None
                if approved_ts is None:
                    self.logger.warning(
                        "PROMOTION_APPROVAL_REQUIRED_TIMESTAMP_INVALID",
                        extra={"strategy": strategy, "model_id": model_id},
                    )
                    self._append_governance_audit_event(
                        event_type="MODEL_PROMOTION_BLOCKED",
                        payload={
                            "ts": datetime.now(UTC).isoformat(),
                            "strategy": str(strategy),
                            "model_id": str(model_id),
                            "force": bool(force),
                            "reason": "approval_timestamp_invalid",
                            "approval": dict(approval_payload),
                        },
                        primary_lineage=model_lineage,
                    )
                    return False
                if approved_ts is not None:
                    if approved_ts.tzinfo is None:
                        approved_ts = approved_ts.replace(tzinfo=UTC)
                    else:
                        approved_ts = approved_ts.astimezone(UTC)
                    now = datetime.now(UTC)
                    max_age_hours = self._promotion_approval_max_age_hours()
                    future_skew_seconds = self._promotion_approval_future_skew_seconds()
                    approval_age_seconds = (now - approved_ts).total_seconds()
                    if approval_age_seconds < -future_skew_seconds:
                        future_seconds = abs(approval_age_seconds)
                        self.logger.warning(
                            "PROMOTION_APPROVAL_REQUIRED_TIMESTAMP_FUTURE",
                            extra={
                                "strategy": strategy,
                                "model_id": model_id,
                                "future_seconds": future_seconds,
                                "future_skew_seconds": future_skew_seconds,
                            },
                        )
                        self._append_governance_audit_event(
                            event_type="MODEL_PROMOTION_BLOCKED",
                            payload={
                                "ts": now.isoformat(),
                                "strategy": str(strategy),
                                "model_id": str(model_id),
                                "force": bool(force),
                                "reason": "approval_timestamp_future",
                                "approval": dict(approval_payload),
                                "future_seconds": future_seconds,
                                "future_skew_seconds": future_skew_seconds,
                            },
                            primary_lineage=model_lineage,
                        )
                        return False
                    age_hours = max(approval_age_seconds, 0.0) / 3600.0
                    if age_hours > max_age_hours:
                        self.logger.warning(
                            "PROMOTION_APPROVAL_REQUIRED_STALE",
                            extra={
                                "strategy": strategy,
                                "model_id": model_id,
                                "age_hours": age_hours,
                                "max_age_hours": max_age_hours,
                            },
                        )
                        self._append_governance_audit_event(
                            event_type="MODEL_PROMOTION_BLOCKED",
                            payload={
                                "ts": datetime.now(UTC).isoformat(),
                                "strategy": str(strategy),
                                "model_id": str(model_id),
                                "force": bool(force),
                                "reason": "approval_stale",
                                "approval": dict(approval_payload),
                                "age_hours": age_hours,
                                "max_age_hours": max_age_hours,
                            },
                            primary_lineage=model_lineage,
                        )
                        return False
            current_production = self.registry.get_production_model(strategy)
            promoted_at = datetime.now(UTC).isoformat()
            previous_model_id: str | None = None
            previous_model_lineage: dict[str, Any] = {}
            governance_snapshots: dict[str, dict[str, Any]] = {
                str(model_id): self._snapshot_model_governance(model_id)
            }
            if current_production:
                old_model_id, _ = current_production
                previous_model_id = old_model_id
                previous_model_lineage = self._extract_model_lineage(old_model_id)
                governance_snapshots[str(old_model_id)] = self._snapshot_model_governance(old_model_id)
            previous_active_path = self.get_active_model_path(strategy)
            try:
                self._create_active_symlink(strategy, model_id)
            except OSError as exc:
                self.logger.error(
                    "PROMOTION_ACTIVE_SYMLINK_FAILED",
                    extra={
                        "strategy": str(strategy),
                        "model_id": str(model_id),
                        "error": str(exc),
                    },
                )
                self._append_governance_audit_event(
                    event_type="MODEL_PROMOTION_BLOCKED",
                    payload={
                        "ts": datetime.now(UTC).isoformat(),
                        "strategy": str(strategy),
                        "model_id": str(model_id),
                        "force": bool(force),
                        "reason": "active_symlink_failed",
                        "error": str(exc),
                    },
                    primary_lineage=model_lineage,
                    related_lineages={"previous_production": previous_model_lineage},
                )
                return False
            try:
                if current_production:
                    old_model_id, _ = current_production
                    if old_model_id != model_id:
                        self.registry.update_governance_status(
                            old_model_id,
                            'challenger',
                            {
                                'demoted_at': promoted_at,
                                'replaced_by': model_id,
                            },
                        )
                        self.logger.info(f'Demoted previous production model {old_model_id} for strategy {strategy}')
                self.registry.update_governance_status(
                    model_id,
                    'production',
                    {
                        'promoted_at': promoted_at,
                        'previous_production_model_id': previous_model_id,
                        'promotion_approval_id': (
                            str(approval_payload.get("approval_id"))
                            if isinstance(approval_payload, dict)
                            and approval_payload.get("approval_id") not in (None, "")
                            else None
                        ),
                        'promotion_approved_by': (
                            str(approval_payload.get("approver"))
                            if isinstance(approval_payload, dict)
                            and approval_payload.get("approver") not in (None, "")
                            else None
                        ),
                        'promotion_approval_ts': (
                            str(approval_payload.get("ts"))
                            if isinstance(approval_payload, dict)
                            and approval_payload.get("ts") not in (None, "")
                            else None
                        ),
                    },
                )
            except (OSError, ValueError, TypeError) as exc:
                for restore_model_id, governance in governance_snapshots.items():
                    self._restore_model_governance(restore_model_id, governance)
                self._restore_active_symlink(strategy, previous_active_path)
                self.logger.error(
                    "PROMOTION_REGISTRY_STATUS_FAILED",
                    extra={
                        "strategy": str(strategy),
                        "model_id": str(model_id),
                        "error": str(exc),
                    },
                )
                self._append_governance_audit_event(
                    event_type="MODEL_PROMOTION_BLOCKED",
                    payload={
                        "ts": datetime.now(UTC).isoformat(),
                        "strategy": str(strategy),
                        "model_id": str(model_id),
                        "force": bool(force),
                        "reason": "registry_status_failed",
                        "error": str(exc),
                    },
                    primary_lineage=model_lineage,
                    related_lineages={"previous_production": previous_model_lineage},
                )
                return False
            self._append_jsonl_event(
                filename="promotion_events.jsonl",
                payload={
                    "ts": promoted_at,
                    "strategy": str(strategy),
                    "model_id": str(model_id),
                    "previous_model_id": (
                        str(previous_model_id) if previous_model_id not in (None, "") else None
                    ),
                    "force": bool(force),
                    "approval_id": (
                        str(approval_payload.get("approval_id"))
                        if isinstance(approval_payload, dict)
                        and approval_payload.get("approval_id") not in (None, "")
                        else None
                    ),
                    "approval": dict(approval_payload) if isinstance(approval_payload, dict) else None,
                    "lineage": model_lineage,
                    "previous_model_lineage": previous_model_lineage or None,
                },
                error_event="PROMOTION_EVENT_WRITE_FAILED",
            )
            self._append_governance_audit_event(
                event_type="MODEL_PROMOTED",
                payload={
                    "ts": promoted_at,
                    "strategy": str(strategy),
                    "model_id": str(model_id),
                    "previous_model_id": (
                        str(previous_model_id) if previous_model_id not in (None, "") else None
                    ),
                    "force": bool(force),
                    "approval": dict(approval_payload) if isinstance(approval_payload, dict) else None,
                },
                primary_lineage=model_lineage,
                related_lineages={"previous_production": previous_model_lineage},
            )
            self.logger.info(f'Promoted model {model_id} to production for strategy {strategy}')
            return True
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error promoting model {model_id}: {e}')
            return False

    def record_challenger_evaluation(
        self,
        *,
        strategy: str,
        champion_model_id: str,
        challenger_model_id: str,
        metrics: dict[str, Any],
    ) -> str | None:
        """Append a challenger-vs-champion evaluation record for governance review."""

        eval_path = self.base_path / "challenger_evaluations.jsonl"
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "strategy": str(strategy),
            "champion_model_id": str(champion_model_id),
            "challenger_model_id": str(challenger_model_id),
            "metrics": dict(metrics),
        }
        scorecard_payload: dict[str, Any] = {
            "ts": payload["ts"],
            "strategy": str(strategy),
            "champion_model_id": str(champion_model_id),
            "challenger_model_id": str(challenger_model_id),
        }
        challenger_returns = metrics.get("challenger_returns")
        champion_returns = metrics.get("champion_returns")
        if isinstance(challenger_returns, list) and isinstance(champion_returns, list):
            significance = self.evaluate_challenger_significance(
                challenger_returns,
                champion_returns,
                alpha=self.criteria.challenger_significance_alpha,
            )
            sample_count = min(len(challenger_returns), len(champion_returns))
            sequential_pass = (
                sample_count >= int(self.criteria.challenger_sequential_min_samples)
                and float(significance.get("uplift_bps", 0.0)) >= float(self.criteria.min_challenger_uplift_bps)
                and float(significance.get("p_value", 1.0)) <= float(self.criteria.challenger_significance_alpha)
            )
            payload["significance"] = significance
            payload["sequential_gate"] = {
                "sample_count": int(max(sample_count, 0)),
                "min_samples": int(self.criteria.challenger_sequential_min_samples),
                "required_consecutive_passes": int(self.criteria.challenger_sequential_required_passes),
                "session_pass": bool(sequential_pass),
            }
            scorecard_payload["uplift_bps"] = float(significance.get("uplift_bps", 0.0))
            scorecard_payload["p_value"] = float(significance.get("p_value", 1.0))
            scorecard_payload["sample_count"] = int(max(sample_count, 0))
            scorecard_payload["session_pass"] = bool(sequential_pass)
            scorecard_payload["required_uplift_bps"] = float(
                self.criteria.min_challenger_uplift_bps
            )
            scorecard_payload["required_alpha"] = float(
                self.criteria.challenger_significance_alpha
            )
        else:
            scorecard_payload["uplift_bps"] = float(
                metrics.get("challenger_uplift_bps", 0.0) or 0.0
            )
            scorecard_payload["p_value"] = float(
                metrics.get("challenger_p_value", 1.0) or 1.0
            )
        scorecard_payload["net_expectancy_bps_champion"] = float(
            metrics.get("champion_net_expectancy_bps", 0.0) or 0.0
        )
        scorecard_payload["net_expectancy_bps_challenger"] = float(
            metrics.get("challenger_net_expectancy_bps", 0.0) or 0.0
        )
        scorecard_payload["reject_rate_champion"] = float(
            metrics.get("champion_reject_rate", 0.0) or 0.0
        )
        scorecard_payload["reject_rate_challenger"] = float(
            metrics.get("challenger_reject_rate", 0.0) or 0.0
        )
        scorecard_payload["execution_drift_bps_champion"] = float(
            metrics.get("champion_execution_drift_bps", 0.0) or 0.0
        )
        scorecard_payload["execution_drift_bps_challenger"] = float(
            metrics.get("challenger_execution_drift_bps", 0.0) or 0.0
        )
        scorecard_payload["delta_net_expectancy_bps"] = float(
            scorecard_payload["net_expectancy_bps_challenger"]
            - scorecard_payload["net_expectancy_bps_champion"]
        )
        scorecard_payload["delta_reject_rate"] = float(
            scorecard_payload["reject_rate_challenger"]
            - scorecard_payload["reject_rate_champion"]
        )
        scorecard_payload["delta_execution_drift_bps"] = float(
            scorecard_payload["execution_drift_bps_challenger"]
            - scorecard_payload["execution_drift_bps_champion"]
        )
        challenger_lineage = self._extract_model_lineage(challenger_model_id)
        champion_lineage = self._extract_model_lineage(champion_model_id)
        try:
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            with eval_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
            self._append_jsonl_event(
                filename="champion_challenger_scorecards.jsonl",
                payload=scorecard_payload,
                error_event="CHAMPION_CHALLENGER_SCORECARD_WRITE_FAILED",
            )
            self._append_governance_audit_event(
                event_type="CHALLENGER_EVALUATION_RECORDED",
                payload=payload,
                primary_lineage=challenger_lineage,
                related_lineages={"champion": champion_lineage},
            )
            self._append_governance_audit_event(
                event_type="CHAMPION_CHALLENGER_SCORECARD_RECORDED",
                payload=scorecard_payload,
                primary_lineage=challenger_lineage,
                related_lineages={"champion": champion_lineage},
            )
            return str(eval_path)
        except OSError as exc:
            self.logger.error(
                "CHALLENGER_EVALUATION_WRITE_FAILED",
                extra={"path": str(eval_path), "error": str(exc)},
            )
            return None

    def rollback_to_previous_production(
        self,
        *,
        strategy: str,
        reason: str,
        force: bool = True,
    ) -> bool:
        """Rollback strategy production model to the previous production model."""

        current = self.registry.get_production_model(strategy)
        if current is None:
            self.logger.warning("ROLLBACK_SKIPPED_NO_PRODUCTION_MODEL", extra={"strategy": strategy})
            self._record_rollback_audit(
                strategy=strategy,
                status="skipped_no_production_model",
                reason=reason,
            )
            return False
        current_model_id, _current_meta = current
        governance = dict(_current_meta.get("governance", {}))
        previous_model_id = str(
            governance.get("previous_production_model_id")
            or governance.get("previous_champion_model_id")
            or ""
        ).strip()
        if not previous_model_id:
            self.logger.warning(
                "ROLLBACK_SKIPPED_NO_PREVIOUS_MODEL",
                extra={"strategy": strategy, "current_model_id": current_model_id},
            )
            self._record_rollback_audit(
                strategy=strategy,
                status="skipped_no_previous_model",
                reason=reason,
                from_model_id=current_model_id,
            )
            return False
        if previous_model_id not in self.registry.model_index:
            self.logger.warning(
                "ROLLBACK_SKIPPED_PREVIOUS_MODEL_MISSING",
                extra={
                    "strategy": strategy,
                    "current_model_id": current_model_id,
                    "previous_model_id": previous_model_id,
                },
            )
            self._record_rollback_audit(
                strategy=strategy,
                status="skipped_previous_model_missing",
                reason=reason,
                from_model_id=current_model_id,
                to_model_id=previous_model_id,
            )
            return False

        promoted = self.promote_to_production(previous_model_id, force=force)
        if not promoted:
            self._record_rollback_audit(
                strategy=strategy,
                status="rollback_failed",
                reason=reason,
                from_model_id=current_model_id,
                to_model_id=previous_model_id,
            )
            return False

        rolled_back_at = datetime.now(UTC).isoformat()
        self.registry.update_governance_status(
            previous_model_id,
            "production",
            {
                "rollback_from_model_id": current_model_id,
                "rollback_reason": reason,
                "rollback_at": rolled_back_at,
            },
        )
        self.registry.update_governance_status(
            current_model_id,
            "challenger",
            {
                "rolled_back_to_model_id": previous_model_id,
                "rollback_reason": reason,
                "rollback_at": rolled_back_at,
            },
        )
        self.logger.warning(
            "MODEL_ROLLBACK_COMPLETED",
            extra={
                "strategy": strategy,
                "from_model_id": current_model_id,
                "to_model_id": previous_model_id,
                "reason": reason,
            },
        )
        self._record_rollback_audit(
            strategy=strategy,
            status="rolled_back",
            reason=reason,
            from_model_id=current_model_id,
            to_model_id=previous_model_id,
        )
        return True

    def evaluate_live_kpis_and_maybe_rollback(
        self,
        *,
        strategy: str,
        live_kpis: dict[str, Any],
        force: bool = True,
        allow_rollback: bool = True,
    ) -> dict[str, Any]:
        """Rollback production when live KPI control bands are persistently breached."""

        breaches: dict[str, Any] = {}
        drawdown = float(live_kpis.get("max_drawdown", 0.0) or 0.0)
        reject_rate = float(live_kpis.get("reject_rate", 0.0) or 0.0)
        execution_drift_bps = float(live_kpis.get("execution_drift_bps", 0.0) or 0.0)
        drift_psi = float(live_kpis.get("drift_psi", 0.0) or 0.0)
        live_calibration_ece = float(live_kpis.get("live_calibration_ece", 0.0) or 0.0)
        live_calibration_brier = float(live_kpis.get("live_calibration_brier", 0.0) or 0.0)

        if drawdown > float(self.criteria.control_band_drawdown):
            breaches["max_drawdown"] = drawdown
        if reject_rate > float(self.criteria.control_band_reject_rate):
            breaches["reject_rate"] = reject_rate
        if execution_drift_bps > float(self.criteria.control_band_execution_drift_bps):
            breaches["execution_drift_bps"] = execution_drift_bps
        if drift_psi > float(self.criteria.control_band_drift_psi):
            breaches["drift_psi"] = drift_psi
        if live_calibration_ece > float(self.criteria.control_band_live_calibration_ece):
            breaches["live_calibration_ece"] = live_calibration_ece
        if live_calibration_brier > float(self.criteria.control_band_live_calibration_brier):
            breaches["live_calibration_brier"] = live_calibration_brier

        current = self.registry.get_production_model(strategy)
        current_model_id = current[0] if current is not None else None
        result: dict[str, Any] = {
            "strategy": strategy,
            "model_id": current_model_id,
            "breached": bool(breaches),
            "breaches": breaches,
            "failed_kpis": sorted(str(key) for key in breaches),
            "triggered": False,
        }
        if not breaches:
            self._clear_live_kpi_breach_window(
                strategy=strategy,
                model_id=current_model_id,
            )
            return result
        breach_window = self._update_live_kpi_breach_window(
            strategy=strategy,
            model_id=current_model_id,
            breaches=breaches,
            allow_rollback=allow_rollback,
        )
        breach_count = int(breach_window.get("consecutive_breach_count", 0) or 0)
        required_breaches = int(breach_window.get("required_consecutive_breaches", 1) or 1)
        failed_kpis = list(breach_window.get("failed_kpis", result["failed_kpis"]))
        audit_extra = {
            "breaches": dict(breaches),
            "failed_kpis": failed_kpis,
            "consecutive_breach_count": breach_count,
            "required_consecutive_breaches": required_breaches,
            "current_model_id": current_model_id,
        }
        result["consecutive_breach_count"] = breach_count
        result["required_consecutive_breaches"] = required_breaches
        if not allow_rollback:
            result["status"] = "pending"
            self._record_rollback_audit(
                strategy=strategy,
                status="pending",
                reason="live_kpi_control_band_breach",
                from_model_id=current_model_id,
                extra=audit_extra,
            )
            return result
        if breach_count < required_breaches:
            result["status"] = "pending"
            self._record_rollback_audit(
                strategy=strategy,
                status="pending",
                reason="live_kpi_control_band_breach",
                from_model_id=current_model_id,
                extra=audit_extra,
            )
            return result

        rollback_enabled = False
        try:
            rollback_enabled = bool(
                get_env("AI_TRADING_PROMOTION_AUTO_ROLLBACK_ON_CONTROL_BAND", False, cast=bool)
            )
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            rollback_enabled = False
        if not rollback_enabled:
            result["status"] = "dry_run_disabled"
            self._mark_live_kpi_window_status(
                strategy=strategy,
                status="dry_run_disabled",
            )
            self._record_rollback_audit(
                strategy=strategy,
                status="dry_run_disabled",
                reason="live_kpi_control_band_breach",
                from_model_id=current_model_id,
                extra=audit_extra,
            )
            return result

        rolled_back = self.rollback_to_previous_production(
            strategy=strategy,
            reason="live_kpi_control_band_breach",
            force=force,
        )
        result["triggered"] = bool(rolled_back)
        result["status"] = "rolled_back" if rolled_back else "rollback_failed"
        if rolled_back:
            self._mark_live_kpi_window_status(strategy=strategy, status="rolled_back")
            self._record_rollback_audit(
                strategy=strategy,
                status="rolled_back",
                reason="live_kpi_control_band_breach",
                from_model_id=current_model_id,
                extra=audit_extra,
            )
            return result

        if current_model_id:
            demoted_at = datetime.now(UTC).isoformat()
            self.registry.update_governance_status(
                current_model_id,
                "challenger",
                {
                    "demoted_at": demoted_at,
                    "demotion_reason": "live_kpi_control_band_breach",
                    "live_kpi_breach_count": breach_count,
                    "live_kpi_failed_kpis": failed_kpis,
                },
            )
            result["triggered"] = True
            result["status"] = "demoted_no_rollback_target"
            self._mark_live_kpi_window_status(
                strategy=strategy,
                status="demoted_no_rollback_target",
            )
            self._record_rollback_audit(
                strategy=strategy,
                status="demoted_no_rollback_target",
                reason="live_kpi_control_band_breach",
                from_model_id=current_model_id,
                extra=audit_extra,
            )
        else:
            self._mark_live_kpi_window_status(strategy=strategy, status="rollback_failed")
        return result

    def _save_shadow_metrics(self, model_id: str, metrics: PromotionMetrics) -> None:
        """Save shadow metrics to disk."""
        metrics_file = self.base_path / f'{model_id}_shadow_metrics.json'
        metrics_dict = {
            'sessions_completed': metrics.sessions_completed,
            'total_trades': metrics.total_trades,
            'turnover_ratio': metrics.turnover_ratio,
            'live_sharpe_ratio': metrics.live_sharpe_ratio,
            'live_sortino_ratio': metrics.live_sortino_ratio,
            'live_calmar_ratio': metrics.live_calmar_ratio,
            'max_drawdown': metrics.max_drawdown,
            'tail_loss_95': metrics.tail_loss_95,
            'risk_of_ruin': metrics.risk_of_ruin,
            'drift_psi': metrics.drift_psi,
            'avg_latency_ms': metrics.avg_latency_ms,
            'error_rate': metrics.error_rate,
            'purged_walk_forward_pass_ratio': metrics.purged_walk_forward_pass_ratio,
            'monte_carlo_p05_bps': metrics.monte_carlo_p05_bps,
            'regime_pass_ratio': metrics.regime_pass_ratio,
            'tca_gate_passed': metrics.tca_gate_passed,
            'reject_rate': metrics.reject_rate,
            'execution_drift_bps': metrics.execution_drift_bps,
            'gross_expectancy_bps': metrics.gross_expectancy_bps,
            'avg_cost_bps': metrics.avg_cost_bps,
            'net_expectancy_bps': metrics.net_expectancy_bps,
            'live_calibration_ece': metrics.live_calibration_ece,
            'live_calibration_brier': metrics.live_calibration_brier,
            'calibration_samples': metrics.calibration_samples,
            'challenger_uplift_bps': metrics.challenger_uplift_bps,
            'challenger_p_value': metrics.challenger_p_value,
            'challenger_eval_samples': metrics.challenger_eval_samples,
            'challenger_sequential_passes': metrics.challenger_sequential_passes,
            'last_updated': metrics.last_updated.isoformat() if metrics.last_updated else None,
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

    def _load_shadow_metrics(self, model_id: str) -> PromotionMetrics | None:
        """Load shadow metrics from disk."""
        metrics_file = self.base_path / f'{model_id}_shadow_metrics.json'
        if not metrics_file.exists():
            return None
        try:
            with open(metrics_file) as f:
                data = json.load(f)
            last_updated = None
            if data.get('last_updated'):
                last_updated = datetime.fromisoformat(data['last_updated'].replace('Z', '+00:00'))
            return PromotionMetrics(
                sessions_completed=data.get('sessions_completed', 0),
                total_trades=data.get('total_trades', 0),
                turnover_ratio=data.get('turnover_ratio', 0.0),
                live_sharpe_ratio=data.get('live_sharpe_ratio', 0.0),
                max_drawdown=data.get('max_drawdown', 0.0),
                drift_psi=data.get('drift_psi', 0.0),
                avg_latency_ms=data.get('avg_latency_ms', 0.0),
                error_rate=data.get('error_rate', 0.0),
                live_sortino_ratio=data.get('live_sortino_ratio', 0.0),
                live_calmar_ratio=data.get('live_calmar_ratio', 0.0),
                tail_loss_95=data.get('tail_loss_95', 0.0),
                risk_of_ruin=data.get('risk_of_ruin', 0.0),
                purged_walk_forward_pass_ratio=data.get('purged_walk_forward_pass_ratio', 0.0),
                monte_carlo_p05_bps=data.get('monte_carlo_p05_bps', 0.0),
                regime_pass_ratio=data.get('regime_pass_ratio', 0.0),
                tca_gate_passed=self._explicit_bool_pass(data.get('tca_gate_passed')),
                reject_rate=data.get('reject_rate', 0.0),
                execution_drift_bps=data.get('execution_drift_bps', 0.0),
                gross_expectancy_bps=data.get('gross_expectancy_bps', 0.0),
                avg_cost_bps=data.get('avg_cost_bps', 0.0),
                net_expectancy_bps=data.get('net_expectancy_bps', 0.0),
                live_calibration_ece=data.get('live_calibration_ece', 1.0),
                live_calibration_brier=data.get('live_calibration_brier', 1.0),
                calibration_samples=data.get('calibration_samples', 0),
                challenger_uplift_bps=data.get('challenger_uplift_bps', 0.0),
                challenger_p_value=data.get('challenger_p_value', 1.0),
                challenger_eval_samples=data.get('challenger_eval_samples', 0),
                challenger_sequential_passes=data.get('challenger_sequential_passes', 0),
                last_updated=last_updated,
            )
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error loading shadow metrics for {model_id}: {e}')
            return None

    def _create_active_symlink(self, strategy: str, model_id: str) -> None:
        """Create symlink to active production model."""
        tmp_path: Path | None = None
        try:
            model_info = self.registry.model_index[model_id]
            model_path = Path(model_info['path'])
            symlink_path = self.active_dir / f'{strategy}_active'
            tmp_path = self.active_dir / f'.{strategy}_active.{uuid.uuid4().hex}.tmp'
            tmp_path.symlink_to(model_path.absolute())
            tmp_path.replace(symlink_path)
            if not symlink_path.is_symlink() or symlink_path.resolve() != model_path.absolute().resolve():
                raise OSError("active symlink target verification failed")
            self.logger.debug(f'Created active symlink for {strategy}: {symlink_path} -> {model_path}')
        except OSError as e:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self.logger.error(f'Error creating active symlink for {strategy}: {e}')
            raise
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error creating active symlink for {strategy}: {e}')

    def _restore_active_symlink(self, strategy: str, previous_active_path: str | None) -> None:
        """Restore active symlink after a failed registry status commit."""
        symlink_path = self.active_dir / f'{strategy}_active'
        tmp_path: Path | None = None
        try:
            if previous_active_path:
                tmp_path = self.active_dir / f'.{strategy}_active.restore.{uuid.uuid4().hex}.tmp'
                tmp_path.symlink_to(Path(previous_active_path).absolute())
                tmp_path.replace(symlink_path)
            elif symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
        except OSError as e:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self.logger.error(f'Error restoring active symlink for {strategy}: {e}')

    def _remove_active_symlink(self, strategy: str) -> None:
        """Remove active symlink for strategy."""
        try:
            symlink_path = self.active_dir / f'{strategy}_active'
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
                self.logger.debug(f'Removed active symlink for {strategy}')
        except (OSError, ValueError, TypeError) as e:
            self.logger.error(f'Error removing active symlink for {strategy}: {e}')

    def get_active_model_path(self, strategy: str) -> str | None:
        """Get path to active production model."""
        symlink_path = self.active_dir / f'{strategy}_active'
        if symlink_path.exists() and symlink_path.is_symlink():
            return str(symlink_path.resolve())
        return None

    def list_shadow_models(self) -> list[dict[str, Any]]:
        """List all models currently in shadow testing."""
        shadow_models = []
        for model_id, info in self.registry.model_index.items():
            if not info.get('active', True):
                continue
            try:
                metadata_file = Path(info['path']) / 'meta.json'
                with open(metadata_file) as f:
                    metadata = json.load(f)
                governance = metadata.get('governance', {})
                if governance.get('status') == 'shadow':
                    shadow_metrics = self._load_shadow_metrics(model_id)
                    shadow_info = {'model_id': model_id, 'strategy': info['strategy'], 'shadow_start': governance.get('shadow_start_time'), 'metrics': shadow_metrics}
                    eligible, details = self.check_promotion_eligibility(model_id)
                    shadow_info['promotion_eligible'] = eligible
                    shadow_info['promotion_details'] = details
                    shadow_models.append(shadow_info)
            except (ValueError, TypeError) as e:
                self.logger.debug(f'Error checking shadow status for model {model_id}: {e}')
                continue
        return shadow_models
_global_promotion_manager: ModelPromotion | None = None

def get_promotion_manager() -> ModelPromotion:
    """Get or create global promotion manager."""
    global _global_promotion_manager
    if _global_promotion_manager is None:
        _global_promotion_manager = ModelPromotion()
    return _global_promotion_manager
