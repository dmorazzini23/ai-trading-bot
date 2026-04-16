"""Decision-event emission helpers for immutable OMS audit persistence."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from threading import RLock
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger

from .event_types import DecisionEvent

logger = get_logger(__name__)

_STORE_LOCK = RLock()
_EVENT_STORE: Any | None = None
_EVENT_STORE_INIT_FAILED = False


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _jsonable_default(value: Any) -> str:
    return str(value)


def _idempotency_key(payload: Mapping[str, Any]) -> str:
    order_payload = _as_mapping(payload.get("order"))
    metrics_payload = _as_mapping(payload.get("metrics"))
    net_target_payload = _as_mapping(payload.get("net_target"))
    material = {
        "schema_version": str(payload.get("schema_version") or ""),
        "symbol": str(payload.get("symbol") or "").strip().upper(),
        "bar_ts": str(payload.get("bar_ts") or ""),
        "gates": list(payload.get("gates") or []),
        "order": {
            "side": str(order_payload.get("side") or "").strip().lower(),
            "quantity": order_payload.get("quantity"),
            "qty": order_payload.get("qty"),
            "price": order_payload.get("price"),
            "strategy_id": order_payload.get("strategy_id"),
            "client_order_id": order_payload.get("client_order_id"),
            "id": order_payload.get("id"),
        },
        "metrics": {
            "confidence": metrics_payload.get("confidence"),
            "expected_edge_bps": metrics_payload.get("expected_edge_bps"),
            "expected_net_edge_bps": metrics_payload.get("expected_net_edge_bps"),
        },
        "net_target": {
            "target_dollars": net_target_payload.get("target_dollars"),
            "target_shares": net_target_payload.get("target_shares"),
        },
    }
    encoded = json.dumps(material, sort_keys=True, default=_jsonable_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _decision_action(payload: Mapping[str, Any]) -> str:
    explicit = str(payload.get("decision_action") or "").strip().upper()
    if explicit in {"BUY", "SELL", "HOLD", "REDUCE", "EXIT"}:
        return explicit

    order_payload = _as_mapping(payload.get("order"))
    side = str(order_payload.get("side") or "").strip().lower()
    if side == "buy":
        return "BUY"
    if side == "sell":
        return "SELL"

    gates_raw = payload.get("gates")
    gates = (
        [str(item).strip().upper() for item in gates_raw if str(item).strip()]
        if isinstance(gates_raw, list)
        else []
    )
    if "OK_TRADE" in gates:
        return "BUY"
    return "HOLD"


def _strategy_id(payload: Mapping[str, Any]) -> str | None:
    order_payload = _as_mapping(payload.get("order"))
    metrics_payload = _as_mapping(payload.get("metrics"))
    for candidate in (
        order_payload.get("strategy_id"),
        order_payload.get("strategy"),
        metrics_payload.get("strategy_id"),
        metrics_payload.get("strategy"),
    ):
        if candidate in (None, ""):
            continue
        return str(candidate)
    return None


def _confidence(payload: Mapping[str, Any]) -> float | None:
    metrics_payload = _as_mapping(payload.get("metrics"))
    for candidate in (
        payload.get("confidence"),
        metrics_payload.get("confidence"),
    ):
        try:
            if candidate is not None:
                return float(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _expected_edge_bps(payload: Mapping[str, Any]) -> float | None:
    metrics_payload = _as_mapping(payload.get("metrics"))
    tca_payload = _as_mapping(payload.get("tca"))
    for candidate in (
        payload.get("expected_edge_bps"),
        metrics_payload.get("expected_net_edge_bps"),
        metrics_payload.get("expected_edge_bps"),
        tca_payload.get("expected_net_edge_bps"),
    ):
        try:
            if candidate is not None:
                return float(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _config_hash(payload: Mapping[str, Any]) -> str | None:
    snapshot = _as_mapping(payload.get("config_snapshot"))
    value = snapshot.get("config_snapshot_hash")
    if value in (None, ""):
        return None
    return str(value)


def _policy_hash(payload: Mapping[str, Any]) -> str | None:
    snapshot = _as_mapping(payload.get("config_snapshot"))
    value = snapshot.get("effective_policy_hash")
    if value in (None, ""):
        return None
    return str(value)


def _model_hash(payload: Mapping[str, Any]) -> str | None:
    for container in (_as_mapping(payload.get("metrics")), _as_mapping(payload.get("config_snapshot"))):
        for candidate_key in ("model_hash", "model_version_hash"):
            value = container.get(candidate_key)
            if value not in (None, ""):
                return str(value)
    return None


def _lineage_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    parsed = str(value).strip()
    return parsed or None


def _lineage_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics_payload = _as_mapping(payload.get("metrics"))
    snapshot_payload = _as_mapping(payload.get("config_snapshot"))
    order_payload = _as_mapping(payload.get("order"))

    policy_hash = (
        _lineage_text(snapshot_payload.get("effective_policy_hash"))
        or _lineage_text(payload.get("policy_hash"))
    )
    config_snapshot_hash = (
        _lineage_text(snapshot_payload.get("config_snapshot_hash"))
        or _lineage_text(payload.get("config_snapshot_hash"))
    )
    model_id = (
        _lineage_text(payload.get("model_id"))
        or _lineage_text(metrics_payload.get("model_id"))
        or _lineage_text(order_payload.get("model_id"))
    )
    model_version = (
        _lineage_text(payload.get("model_version"))
        or _lineage_text(metrics_payload.get("model_version"))
        or _lineage_text(order_payload.get("model_version"))
    )
    dataset_hash = (
        _lineage_text(payload.get("dataset_hash"))
        or _lineage_text(metrics_payload.get("dataset_hash"))
        or _lineage_text(snapshot_payload.get("dataset_hash"))
        or _lineage_text(snapshot_payload.get("dataset_fingerprint"))
    )
    feature_version = (
        _lineage_text(payload.get("feature_version"))
        or _lineage_text(metrics_payload.get("feature_version"))
        or _lineage_text(snapshot_payload.get("feature_version"))
        or _lineage_text(snapshot_payload.get("feature_set_version"))
    )
    model_artifact_hash = (
        _lineage_text(payload.get("model_artifact_hash"))
        or _lineage_text(metrics_payload.get("model_artifact_hash"))
        or _lineage_text(snapshot_payload.get("model_artifact_hash"))
        or _lineage_text(metrics_payload.get("model_hash"))
        or _lineage_text(snapshot_payload.get("model_hash"))
        or _lineage_text(metrics_payload.get("model_version_hash"))
    )
    decision_trace_id = (
        _lineage_text(payload.get("decision_trace_id"))
        or _lineage_text(order_payload.get("decision_trace_id"))
    )
    if decision_trace_id is None:
        symbol = str(payload.get("symbol") or "").strip().upper()
        bar_ts = str(payload.get("bar_ts") or "").strip()
        order_id = _lineage_text(order_payload.get("id"))
        client_order_id = _lineage_text(order_payload.get("client_order_id"))
        trace_material = "|".join(
            (
                symbol,
                bar_ts,
                order_id or "",
                client_order_id or "",
            )
        )
        if trace_material.replace("|", ""):
            decision_trace_id = hashlib.sha1(
                trace_material.encode("utf-8")
            ).hexdigest()[:24]

    lineage: dict[str, Any] = {}
    if policy_hash is not None:
        lineage["policy_hash"] = policy_hash
    if config_snapshot_hash is not None:
        lineage["config_snapshot_hash"] = config_snapshot_hash
    if model_id is not None:
        lineage["model_id"] = model_id
    if model_version is not None:
        lineage["model_version"] = model_version
    if dataset_hash is not None:
        lineage["dataset_hash"] = dataset_hash
    if feature_version is not None:
        lineage["feature_version"] = feature_version
    if model_artifact_hash is not None:
        lineage["model_artifact_hash"] = model_artifact_hash
    if decision_trace_id is not None:
        lineage["decision_trace_id"] = decision_trace_id
    return lineage


def _context(payload: Mapping[str, Any]) -> dict[str, Any]:
    order_payload = _as_mapping(payload.get("order"))
    net_target_payload = _as_mapping(payload.get("net_target"))
    gates_raw = payload.get("gates")
    return {
        "bar_ts": payload.get("bar_ts"),
        "gates": list(gates_raw) if isinstance(gates_raw, list) else [],
        "order": {
            "id": order_payload.get("id"),
            "client_order_id": order_payload.get("client_order_id"),
            "side": order_payload.get("side"),
            "qty": order_payload.get("qty"),
            "quantity": order_payload.get("quantity"),
            "price": order_payload.get("price"),
        },
        "net_target": {
            "target_dollars": net_target_payload.get("target_dollars"),
            "target_shares": net_target_payload.get("target_shares"),
        },
        "lineage": _lineage_context(payload),
    }


def _features(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics_payload = _as_mapping(payload.get("metrics"))
    features: dict[str, Any] = {}
    for key in (
        "score",
        "edge_proxy_bps",
        "expected_edge_bps",
        "expected_net_edge_bps",
        "realized_is_bps",
        "realized_net_edge_bps",
    ):
        if key in metrics_payload:
            features[key] = metrics_payload.get(key)
    return features


def _resolve_event_store() -> Any | None:
    global _EVENT_STORE
    global _EVENT_STORE_INIT_FAILED
    if not bool(get_env("AI_TRADING_DECISION_EVENT_STORE_ENABLED", True, cast=bool)):
        return None
    if _EVENT_STORE_INIT_FAILED:
        return None
    if _EVENT_STORE is not None:
        return _EVENT_STORE

    with _STORE_LOCK:
        if _EVENT_STORE is not None:
            return _EVENT_STORE
        if _EVENT_STORE_INIT_FAILED:
            return None
        try:
            from ai_trading.oms.event_store import EventStore

            _EVENT_STORE = EventStore()
        except Exception as exc:
            _EVENT_STORE_INIT_FAILED = True
            logger.warning(
                "DECISION_EVENT_STORE_INIT_FAILED",
                extra={"error": str(exc)},
            )
            return None
        return _EVENT_STORE


def reset_decision_event_store_cache() -> None:
    """Reset cached EventStore instance (used by tests)."""

    global _EVENT_STORE
    global _EVENT_STORE_INIT_FAILED
    with _STORE_LOCK:
        if _EVENT_STORE is not None:
            try:
                _EVENT_STORE.close()
            except Exception:
                pass
        _EVENT_STORE = None
        _EVENT_STORE_INIT_FAILED = False


def emit_decision_event_from_payload(
    payload: Mapping[str, Any],
    *,
    event_source: str = "decision_record",
) -> dict[str, Any]:
    """Persist immutable decision/OMS audit events from a normalized decision payload."""

    store = _resolve_event_store()
    symbol = str(payload.get("symbol") or "").strip().upper()
    if store is None or not symbol:
        return {"persisted": False}

    decision = DecisionEvent(
        symbol=symbol,
        decision_action=_decision_action(payload),  # type: ignore[arg-type]
        decision_source=str(event_source or "decision_record"),
        idempotency_key=_idempotency_key(payload),
        strategy_id=_strategy_id(payload),
        confidence=_confidence(payload),
        expected_edge_bps=_expected_edge_bps(payload),
        policy_hash=_policy_hash(payload),
        model_hash=_model_hash(payload),
        config_hash=_config_hash(payload),
        features=_features(payload),
        context=_context(payload),
        decision_ts=str(payload.get("bar_ts") or ""),
    ).normalized()
    try:
        decision_persisted = bool(store.append_decision_event(decision))
    except Exception as exc:
        logger.warning(
            "DECISION_EVENT_APPEND_FAILED",
            extra={"symbol": symbol, "error": str(exc)},
        )
        return {"persisted": False, "error": str(exc)}

    order_payload = _as_mapping(payload.get("order"))
    intent_id: str | None = None
    for key in ("id", "intent_id", "client_order_id"):
        value = order_payload.get(key)
        if value not in (None, ""):
            intent_id = str(value)
            break

    oms_key = hashlib.sha256(
        f"DECISION_EMITTED|{decision.idempotency_key}".encode("utf-8")
    ).hexdigest()
    try:
        decision_context = dict(decision.context or {})
        decision_lineage = _as_mapping(decision_context.get("lineage"))
        oms_persisted = bool(
            store.append_oms_event_payload(
                event_type="DECISION_EMITTED",
                event_source=str(event_source or "decision_record"),
                idempotency_key=oms_key,
                intent_id=intent_id,
                event_ts=decision.decision_ts,
                policy_hash=decision.policy_hash,
                model_hash=decision.model_hash,
                payload={
                    "decision_uuid": decision.decision_uuid,
                    "decision_action": decision.decision_action,
                    "symbol": decision.symbol,
                    "strategy_id": decision.strategy_id,
                    "confidence": decision.confidence,
                    "expected_edge_bps": decision.expected_edge_bps,
                    "lineage": decision_lineage,
                },
            )
        )
    except Exception as exc:
        logger.warning(
            "DECISION_OMS_EVENT_APPEND_FAILED",
            extra={"symbol": symbol, "error": str(exc)},
        )
        oms_persisted = False

    return {
        "persisted": bool(decision_persisted),
        "oms_event_persisted": bool(oms_persisted),
        "symbol": decision.symbol,
        "decision_uuid": decision.decision_uuid,
        "idempotency_key": decision.idempotency_key,
    }


__all__ = ["emit_decision_event_from_payload", "reset_decision_event_store_cache"]
