from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import json
import time as pytime
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Mapping

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.config.management import get_env
from ai_trading.governance.paths import resolve_governance_base_path
from ai_trading.settings import get_backup_data_provider
from ai_trading.telemetry import runtime_state
from ai_trading.governance.replay_live_parity import summarize_replay_live_parity_gate

_HEALTH_SNAPSHOT_CACHE_LOCK = Lock()
_HEALTH_SNAPSHOT_CACHE: dict[str, dict[str, Any]] = {}


def _safe_observe(observer: Callable[[], Any], default: Any) -> Any:
    try:
        return observer()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return default


def _timestamp_age_seconds(raw_value: Any) -> float | None:
    if raw_value in (None, ""):
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return max((datetime.now(UTC) - parsed).total_seconds(), 0.0)


def _safe_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _normalized_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _dedupe_flags(flags: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw_flag in flags:
        flag = str(raw_flag or "").strip()
        if not flag or flag in seen:
            continue
        seen.add(flag)
        out.append(flag)
    return out


def _build_runtime_attention_flags(
    *,
    provider_state: Mapping[str, Any],
    broker_state: Mapping[str, Any],
    service_state: Mapping[str, Any],
    database_readiness: Mapping[str, Any] | None = None,
    oms_invariants: Mapping[str, Any] | None = None,
    oms_lifecycle_parity: Mapping[str, Any] | None = None,
    replay_live_parity_gate: Mapping[str, Any] | None = None,
    require_database_ready: bool = False,
    require_oms_invariants: bool = False,
    require_oms_lifecycle_parity: bool = False,
    require_replay_live_parity_gate: bool = False,
    include_optional_contract_failures: bool = False,
) -> list[str]:
    flags: list[str] = []
    provider_reason_normalized = _normalized_token(provider_state.get("reason"))
    service_reason_normalized = _normalized_token(service_state.get("reason"))
    service_status_normalized = _normalized_token(service_state.get("status"))
    market_closed_mode = (
        provider_reason_normalized == "market_closed"
        or service_reason_normalized == "market_closed"
    )
    open_orders_count = _safe_nonnegative_int(broker_state.get("open_orders_count"))
    positions_count = _safe_nonnegative_int(broker_state.get("positions_count"))

    if market_closed_mode and positions_count > 0:
        flags.append("market_closed_non_flat_positions")
    if market_closed_mode and open_orders_count > 0:
        flags.append("market_closed_open_orders")
    if bool(provider_state.get("using_backup")):
        flags.append("provider_backup_active")
    if bool(provider_state.get("safe_mode")):
        flags.append("provider_safe_mode")
    if service_status_normalized in {"degraded", "failed", "error", "halted", "stopped"}:
        flags.append("service_degraded")
    if service_status_normalized in {"halted", "stopped"} or any(
        token in service_reason_normalized for token in ("halt", "hard_stop")
    ):
        flags.append("service_halt_active")
    if service_reason_normalized in {
        "trade_updates_stream_failed",
        "trade_updates_stream_exited",
    }:
        flags.append("trade_updates_stream_degraded")
    if (
        service_reason_normalized == "replay_live_parity_gate_failed"
        or (
            isinstance(replay_live_parity_gate, Mapping)
            and (
                include_optional_contract_failures
                or require_replay_live_parity_gate
            )
            and replay_live_parity_gate.get("enabled", False)
            and not bool(replay_live_parity_gate.get("ok"))
        )
    ):
        flags.append("replay_live_parity_gate_failed")
    if (
        isinstance(database_readiness, Mapping)
        and (
            include_optional_contract_failures
            or require_database_ready
        )
        and bool(database_readiness.get("configured"))
        and not bool(database_readiness.get("ok"))
    ):
        flags.append("database_unhealthy")
    if (
        isinstance(oms_invariants, Mapping)
        and (
            include_optional_contract_failures
            or require_oms_invariants
        )
        and oms_invariants.get("enabled", False)
        and not bool(oms_invariants.get("ok"))
    ):
        flags.append("oms_invariants_failed")
    if (
        isinstance(oms_lifecycle_parity, Mapping)
        and (
            include_optional_contract_failures
            or require_oms_lifecycle_parity
        )
        and oms_lifecycle_parity.get("enabled", False)
        and not bool(oms_lifecycle_parity.get("ok"))
    ):
        flags.append("oms_lifecycle_parity_failed")

    return _dedupe_flags(flags)


def _env_bool(name: str, default: bool) -> bool:
    try:
        from ai_trading.config.management import get_env

        return bool(get_env(name, default, cast=bool))
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return bool(default)


def _default_fail_closed_outside_tests() -> bool:
    return not bool(
        str(get_env("PYTEST_CURRENT_TEST", "", cast=str) or "").strip()
        or bool(get_env("PYTEST_RUNNING", False, cast=bool))
    )


def _env_float(name: str, default: float) -> float:
    try:
        from ai_trading.config.management import get_env

        raw = get_env(name, default, cast=float)
        return float(raw if raw is not None else default)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return float(default)


def _health_snapshot_cache_enabled() -> bool:
    if bool(
        str(get_env("PYTEST_CURRENT_TEST", "", cast=str) or "").strip()
        or bool(get_env("PYTEST_RUNNING", False, cast=bool))
    ):
        return False
    return _env_bool("AI_TRADING_HEALTH_ASYNC_CACHE_ENABLED", True)


def _health_snapshot_ttl_seconds(name: str, default: float) -> float:
    env_name = f"AI_TRADING_HEALTH_{name.upper()}_TTL_SEC"
    return max(0.0, min(_env_float(env_name, default), 300.0))


def _cached_background_snapshot(
    *,
    name: str,
    ttl_seconds: float,
    placeholder: Mapping[str, Any],
    builder: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    now_mono = float(pytime.monotonic())
    cached_value: dict[str, Any] | None = None
    should_refresh = False

    with _HEALTH_SNAPSHOT_CACHE_LOCK:
        entry = _HEALTH_SNAPSHOT_CACHE.setdefault(name, {})
        cached_raw = entry.get("value")
        updated_mono = float(entry.get("updated_mono", 0.0) or 0.0)
        refreshing = bool(entry.get("refreshing", False))

        if isinstance(cached_raw, dict):
            cached_value = dict(cached_raw)
            if ttl_seconds > 0.0 and (now_mono - updated_mono) <= ttl_seconds:
                cached_value["refreshing"] = False
                return cached_value

        if not refreshing:
            entry["refreshing"] = True
            should_refresh = True

    if should_refresh:
        def _refresh() -> None:
            try:
                snapshot = dict(builder())
            except AI_TRADING_FALLBACK_EXCEPTIONS as exc:  # pragma: no cover - defensive
                snapshot = {
                    **dict(placeholder),
                    "ok": False,
                    "available": False,
                    "error": str(exc),
                }
            with _HEALTH_SNAPSHOT_CACHE_LOCK:
                entry = _HEALTH_SNAPSHOT_CACHE.setdefault(name, {})
                entry["value"] = snapshot
                entry["updated_mono"] = float(pytime.monotonic())
                entry["refreshing"] = False

        Thread(
            target=_refresh,
            name=f"health-snapshot-{name}",
            daemon=True,
        ).start()

    if cached_value is not None:
        cached_value["refreshing"] = True
        cached_value["stale"] = True
        return cached_value

    payload = dict(placeholder)
    payload["refreshing"] = True
    payload.setdefault("reason", "warming_up")
    return payload


def _market_is_closed_now() -> bool:
    try:
        from ai_trading.utils.base import is_market_open as _is_market_open_base

        return not bool(_is_market_open_base())
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return False


def _model_liveness_snapshot() -> dict[str, Any]:
    try:
        from ai_trading.monitoring.model_liveness import (
            get_model_liveness_snapshot,
        )

        snapshot = get_model_liveness_snapshot()
        if isinstance(snapshot, dict):
            return snapshot
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        pass
    return {}


def _database_readiness_snapshot() -> dict[str, Any]:
    enabled = _env_bool("AI_TRADING_HEALTH_DB_READINESS_ENABLED", True)
    if not enabled:
        return {"enabled": False}

    try:
        from ai_trading.config.management import get_env

        configured_database_url = str(
            get_env("DATABASE_URL", "", cast=str, resolve_aliases=False) or ""
        ).strip()
        configured_store_path = str(
            get_env(
                "AI_TRADING_OMS_INTENT_STORE_PATH",
                "",
                cast=str,
                resolve_aliases=False,
            )
            or ""
        ).strip()
        expected_revision = str(
            get_env(
                "AI_TRADING_OMS_EXPECTED_ALEMBIC_REVISION",
                "20260414_0001",
                cast=str,
                resolve_aliases=False,
            )
            or ""
        ).strip()
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        return {"enabled": True, "configured": False, "ok": False, "error": str(exc)}

    if not configured_database_url and not configured_store_path:
        return {
            "enabled": True,
            "configured": False,
            "ok": True,
            "reason": "database_not_configured",
        }

    store: Any | None = None
    try:
        from ai_trading.oms.event_store import EventStore

        store = EventStore(
            path=(configured_store_path or None),
            url=(configured_database_url or None),
        )
        payload_raw = store.is_healthy(expected_revision=expected_revision)
        payload = (
            dict(payload_raw)
            if isinstance(payload_raw, Mapping)
            else {"ok": bool(payload_raw)}
        )
        payload["enabled"] = True
        payload["configured"] = True
        payload["expected_revision"] = expected_revision
        return payload
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        return {
            "enabled": True,
            "configured": True,
            "ok": False,
            "connected": False,
            "error": str(exc),
            "expected_revision": expected_revision,
        }
    finally:
        if store is not None:
            try:
                store.close()
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                pass


def _database_readiness_snapshot_cached() -> dict[str, Any]:
    if not _health_snapshot_cache_enabled():
        return _database_readiness_snapshot()
    return _cached_background_snapshot(
        name="database_readiness",
        ttl_seconds=_health_snapshot_ttl_seconds("db_readiness", 15.0),
        placeholder={
            "enabled": True,
            "configured": True,
            "ok": False,
            "connected": False,
            "reason": "warming_up",
        },
        builder=_database_readiness_snapshot,
    )


def _oms_invariants_snapshot() -> dict[str, Any]:
    enabled = _env_bool(
        "AI_TRADING_HEALTH_OMS_INVARIANTS_ENABLED",
        _default_fail_closed_outside_tests(),
    )
    if not enabled:
        return {"enabled": False}
    try:
        from ai_trading.config.management import get_env
        from ai_trading.oms.invariants import evaluate_oms_reconciliation_invariants

        configured_database_url = str(
            get_env("DATABASE_URL", "", cast=str, resolve_aliases=False) or ""
        ).strip()
        configured_store_path = str(
            get_env(
                "AI_TRADING_OMS_INTENT_STORE_PATH",
                "",
                cast=str,
                resolve_aliases=False,
            )
            or ""
        ).strip()
        summary = evaluate_oms_reconciliation_invariants(
            database_url=(configured_database_url or None),
            intent_store_path=(configured_store_path or None),
            limit=int(
                get_env(
                    "AI_TRADING_HEALTH_OMS_INVARIANTS_LIMIT",
                    5000,
                    cast=int,
                    resolve_aliases=False,
                )
            ),
        )
        payload = dict(summary)
        payload["enabled"] = True
        return payload
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        return {"enabled": True, "available": False, "ok": False, "error": str(exc)}


def _oms_invariants_snapshot_cached() -> dict[str, Any]:
    if not _health_snapshot_cache_enabled():
        return _oms_invariants_snapshot()
    return _cached_background_snapshot(
        name="oms_invariants",
        ttl_seconds=_health_snapshot_ttl_seconds("oms_invariants", 15.0),
        placeholder={
            "enabled": True,
            "available": False,
            "ok": False,
            "reason": "warming_up",
            "total_violations": 0,
        },
        builder=_oms_invariants_snapshot,
    )


def _oms_lifecycle_parity_snapshot() -> dict[str, Any]:
    enabled = _env_bool(
        "AI_TRADING_HEALTH_OMS_LIFECYCLE_PARITY_ENABLED",
        _default_fail_closed_outside_tests(),
    )
    if not enabled:
        return {"enabled": False}
    try:
        from ai_trading.config.management import get_env
        from ai_trading.oms.invariants import evaluate_oms_lifecycle_parity_invariants

        configured_database_url = str(
            get_env("DATABASE_URL", "", cast=str, resolve_aliases=False) or ""
        ).strip()
        configured_store_path = str(
            get_env(
                "AI_TRADING_OMS_INTENT_STORE_PATH",
                "",
                cast=str,
                resolve_aliases=False,
            )
            or ""
        ).strip()
        summary = evaluate_oms_lifecycle_parity_invariants(
            database_url=(configured_database_url or None),
            intent_store_path=(configured_store_path or None),
            limit=int(
                get_env(
                    "AI_TRADING_HEALTH_OMS_LIFECYCLE_PARITY_LIMIT",
                    5000,
                    cast=int,
                    resolve_aliases=False,
                )
            ),
        )
        payload = dict(summary)
        payload["enabled"] = True
        return payload
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        return {"enabled": True, "available": False, "ok": False, "error": str(exc)}


def _oms_lifecycle_parity_snapshot_cached() -> dict[str, Any]:
    if not _health_snapshot_cache_enabled():
        return _oms_lifecycle_parity_snapshot()
    return _cached_background_snapshot(
        name="oms_lifecycle_parity",
        ttl_seconds=_health_snapshot_ttl_seconds("oms_lifecycle_parity", 15.0),
        placeholder={
            "enabled": True,
            "available": False,
            "ok": False,
            "reason": "warming_up",
            "total_violations": 0,
        },
        builder=_oms_lifecycle_parity_snapshot,
    )


def _replay_live_parity_gate_snapshot(
    *,
    oms_lifecycle_parity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        return summarize_replay_live_parity_gate(
            oms_lifecycle_parity=oms_lifecycle_parity,
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        return {"enabled": True, "available": False, "ok": False, "error": str(exc)}


def _replay_live_parity_gate_snapshot_cached(
    *,
    oms_lifecycle_parity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not _health_snapshot_cache_enabled():
        return _replay_live_parity_gate_snapshot(
            oms_lifecycle_parity=oms_lifecycle_parity,
        )
    return _cached_background_snapshot(
        name="replay_live_parity_gate",
        ttl_seconds=_health_snapshot_ttl_seconds("replay_live_parity_gate", 15.0),
        placeholder={
            "enabled": True,
            "available": False,
            "ok": False,
            "status": "degraded",
            "reason": "warming_up",
            "failed_checks": [],
            "checks": {},
            "observed": {},
            "thresholds": {},
        },
        builder=lambda: _replay_live_parity_gate_snapshot(
            oms_lifecycle_parity=oms_lifecycle_parity,
        ),
    )


def _read_json_mapping_artifact(
    *,
    configured_path: str,
    default_relative: str,
) -> tuple[dict[str, Any], Path]:
    resolved = resolve_runtime_artifact_path(
        configured_path or default_relative,
        default_relative=default_relative,
    )
    if not resolved.exists():
        return ({}, resolved)
    try:
        parsed = json.loads(resolved.read_text(encoding="utf-8"))
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return ({}, resolved)
    if isinstance(parsed, dict):
        return (dict(parsed), resolved)
    return ({}, resolved)


def _governance_base_path() -> Path:
    return resolve_governance_base_path()


def _read_jsonl_tail(path: Path, *, limit: int = 20) -> list[dict[str, Any]]:
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


def _governance_snapshot() -> dict[str, Any]:
    base = _governance_base_path()
    approvals = _read_jsonl_tail(base / "promotion_approvals.jsonl", limit=20)
    scorecards = _read_jsonl_tail(
        base / "champion_challenger_scorecards.jsonl",
        limit=20,
    )
    rollback_audit = _read_jsonl_tail(base / "rollback_audit.jsonl", limit=20)
    promotion_events = _read_jsonl_tail(base / "promotion_events.jsonl", limit=20)
    return {
        "base_path": str(base),
        "latest_promotion_approval": approvals[-1] if approvals else None,
        "latest_champion_challenger_scorecard": (
            scorecards[-1] if scorecards else None
        ),
        "latest_rollback_audit": rollback_audit[-1] if rollback_audit else None,
        "latest_promotion_event": promotion_events[-1] if promotion_events else None,
        "recent_promotion_approvals": approvals[-5:],
        "recent_champion_challenger_scorecards": scorecards[-5:],
        "recent_rollback_audit": rollback_audit[-5:],
    }


def _runtime_performance_snapshot() -> dict[str, Any]:
    try:
        from ai_trading.config.management import get_env

        latest_path = str(
            get_env(
                "AI_TRADING_RUNTIME_PERF_REPORT_LATEST_PATH",
                "runtime/runtime_performance_report_latest.json",
                cast=str,
                resolve_aliases=False,
            )
            or "runtime/runtime_performance_report_latest.json"
        ).strip()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        latest_path = "runtime/runtime_performance_report_latest.json"
    payload, resolved = _read_json_mapping_artifact(
        configured_path=latest_path,
        default_relative="runtime/runtime_performance_report_latest.json",
    )
    go_no_go_raw = payload.get("go_no_go")
    go_no_go = dict(go_no_go_raw) if isinstance(go_no_go_raw, Mapping) else {}
    oms_event_tca_raw = payload.get("oms_event_tca")
    oms_event_tca = (
        dict(oms_event_tca_raw)
        if isinstance(oms_event_tca_raw, Mapping)
        else {}
    )
    parent_scope_rows_raw = oms_event_tca.get("parent_execution_kpis_by_scope")
    parent_scope_rows = (
        [dict(row) for row in parent_scope_rows_raw if isinstance(row, Mapping)]
        if isinstance(parent_scope_rows_raw, list)
        else []
    )
    outcomes_scope_rows_raw = oms_event_tca.get("event_outcomes_by_scope")
    outcomes_scope_rows = (
        [dict(row) for row in outcomes_scope_rows_raw if isinstance(row, Mapping)]
        if isinstance(outcomes_scope_rows_raw, list)
        else []
    )
    reject_reasons_raw = oms_event_tca.get("submit_reject_reasons_top")
    reject_reasons = (
        [dict(row) for row in reject_reasons_raw if isinstance(row, Mapping)]
        if isinstance(reject_reasons_raw, list)
        else []
    )
    cancel_reasons_raw = oms_event_tca.get("cancel_reasons_top")
    cancel_reasons = (
        [dict(row) for row in cancel_reasons_raw if isinstance(row, Mapping)]
        if isinstance(cancel_reasons_raw, list)
        else []
    )
    slippage_decomposition_raw = oms_event_tca.get("realized_slippage_decomposition")
    slippage_decomposition = (
        dict(slippage_decomposition_raw)
        if isinstance(slippage_decomposition_raw, Mapping)
        else {}
    )
    return {
        "available": bool(payload),
        "path": str(resolved),
        "go_no_go": go_no_go,
        "generated_at": payload.get("generated_at"),
        "source": payload.get("source"),
        "oms_event_tca": {
            "enabled": bool(oms_event_tca.get("enabled", bool(oms_event_tca))),
            "available": bool(
                oms_event_tca.get("available", bool(oms_event_tca))
            ),
            "filled_events": oms_event_tca.get("filled_events"),
            "submit_reject_rate_pct": oms_event_tca.get(
                "submit_reject_rate_pct"
            ),
            "cancel_to_submit_ack_rate_pct": oms_event_tca.get(
                "cancel_to_submit_ack_rate_pct"
            ),
            "reject_cancel_rate_pct": oms_event_tca.get(
                "reject_cancel_rate_pct"
            ),
            "p90_slippage_bps": oms_event_tca.get("p90_slippage_bps"),
            "parent_execution_summary_events": oms_event_tca.get(
                "parent_execution_summary_events"
            ),
            "parent_execution_kpis_by_scope": parent_scope_rows[:5],
            "event_outcomes_by_scope": outcomes_scope_rows[:5],
            "submit_reject_reasons_top": reject_reasons[:5],
            "cancel_reasons_top": cancel_reasons[:5],
            "realized_slippage_decomposition": slippage_decomposition,
        },
    }


def _manual_override_snapshot() -> dict[str, Any]:
    try:
        from ai_trading.config.management import get_env

        toggle_path = str(
            get_env(
                "AI_TRADING_POLICY_RUNTIME_TOGGLES_PATH",
                "runtime/policy_runtime_toggles.json",
                cast=str,
                resolve_aliases=False,
            )
            or "runtime/policy_runtime_toggles.json"
        ).strip()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        toggle_path = "runtime/policy_runtime_toggles.json"
    payload, resolved = _read_json_mapping_artifact(
        configured_path=toggle_path,
        default_relative="runtime/policy_runtime_toggles.json",
    )
    return {
        "available": bool(payload),
        "path": str(resolved),
        "state": payload,
    }


def build_alpaca_health_payload(
    context: Mapping[str, Any] | None = None,
    *,
    enrich_from_runtime_env: bool = True,
) -> dict[str, Any]:
    """Build a normalized Alpaca diagnostic section for health payloads."""

    payload: dict[str, Any] = {
        "sdk_ok": False,
        "initialized": False,
        "client_attached": False,
        "has_key": False,
        "has_secret": False,
        "base_url": "",
        "paper": False,
        "shadow_mode": False,
    }
    if isinstance(context, Mapping):
        for key, value in context.items():
            if key not in payload:
                continue
            if isinstance(payload[key], bool):
                payload[key] = bool(value)
            elif value is not None:
                payload[key] = value

    if enrich_from_runtime_env:
        try:
            from ai_trading.utils.env import alpaca_credential_status, get_alpaca_base_url

            has_key, has_secret = alpaca_credential_status()
            payload["has_key"] = bool(payload["has_key"] or has_key)
            payload["has_secret"] = bool(payload["has_secret"] or has_secret)
            if not payload.get("base_url"):
                payload["base_url"] = str(get_alpaca_base_url() or "")
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            pass

        try:
            from ai_trading.alpaca_api import ALPACA_AVAILABLE as alpaca_sdk_ok

            payload["sdk_ok"] = bool(payload["sdk_ok"] or alpaca_sdk_ok)
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            pass

    if payload.get("base_url"):
        payload["paper"] = bool(payload["paper"] or ("paper" in str(payload["base_url"]).lower()))

    return payload


def build_runtime_health_payload(
    *,
    service_name: str = "ai-trading",
    force_ok_for_pytest: bool = False,
    healthy_status_mode: str = "service",
    ok_mode: str = "strict",
) -> dict[str, Any]:
    """Build a normalized runtime health payload shared by API/health surfaces."""

    provider_state: dict[str, Any] = _safe_observe(
        runtime_state.observe_data_provider_state,
        {},
    )
    if not provider_state.get("backup"):
        provider_state = dict(provider_state)
        provider_state["backup"] = get_backup_data_provider()
    broker_state: dict[str, Any] = _safe_observe(runtime_state.observe_broker_status, {})
    service_state: dict[str, Any] = _safe_observe(
        runtime_state.observe_service_status,
        {"status": "unknown"},
    )
    quote_state: dict[str, Any] = _safe_observe(runtime_state.observe_quote_status, {})
    model_liveness = _model_liveness_snapshot()
    database_readiness = _database_readiness_snapshot_cached()
    oms_invariants = _oms_invariants_snapshot_cached()
    oms_lifecycle_parity = _oms_lifecycle_parity_snapshot_cached()
    replay_live_parity_gate = _replay_live_parity_gate_snapshot_cached(
        oms_lifecycle_parity=oms_lifecycle_parity,
    )

    raw_provider_status = provider_state.get("status")
    provider_status = raw_provider_status or (
        "degraded" if provider_state.get("using_backup") else "unknown"
    )
    provider_status_normalized = str(provider_status or "").strip().lower()
    data_status = provider_state.get("data_status")
    data_status_normalized = str(data_status or "").strip().lower()
    gap_ratio_recent = provider_state.get("gap_ratio_recent")
    gap_ratio_pct: float | None = None
    if gap_ratio_recent is not None:
        try:
            gap_ratio_pct = float(gap_ratio_recent) * 100.0
        except (TypeError, ValueError):
            gap_ratio_pct = None

    provider_payload: dict[str, Any] = {
        "status": provider_status,
        "reason": provider_state.get("reason"),
        "reason_code": provider_state.get("reason_code"),
        "reason_detail": provider_state.get("reason_detail"),
        "http_code": provider_state.get("http_code"),
        "using_backup": bool(provider_state.get("using_backup")),
        "active": provider_state.get("active"),
        "primary": provider_state.get("primary"),
        "backup": provider_state.get("backup"),
        "consecutive_failures": provider_state.get("consecutive_failures"),
        "last_error_at": provider_state.get("last_error_at"),
        "cooldown_seconds_remaining": provider_state.get("cooldown_sec"),
        "gap_ratio_recent": gap_ratio_recent,
        "gap_ratio_pct": gap_ratio_pct,
        "quote_fresh_ms": provider_state.get("quote_fresh_ms"),
        "safe_mode": bool(provider_state.get("safe_mode")),
        "data_status": data_status,
    }
    broker_connected_raw = broker_state.get("connected")
    broker_status = broker_state.get("status")
    if not broker_status:
        if broker_connected_raw is None:
            broker_status = "unknown"
        else:
            broker_status = "reachable" if bool(broker_connected_raw) else "unreachable"
    broker_status_normalized = str(broker_status or "").strip().lower()
    broker_connected: bool | None
    if broker_connected_raw is None:
        broker_connected = None
    else:
        broker_connected = bool(broker_connected_raw)
    broker_payload = {
        "status": broker_status,
        "connected": broker_connected,
        "latency_ms": broker_state.get("latency_ms"),
        "last_error": broker_state.get("last_error"),
        "last_order_ack_ms": broker_state.get("last_order_ack_ms"),
        "open_orders_count": broker_state.get("open_orders_count"),
        "positions_count": broker_state.get("positions_count"),
    }

    service_status = service_state.get("status", "unknown")
    service_status_normalized = _normalized_token(service_status)
    service_reason = service_state.get("reason")
    provider_reason_normalized = str(provider_state.get("reason") or "").strip().lower()
    service_reason_normalized = str(service_reason or "").strip().lower()
    service_phase_normalized = str(service_state.get("phase") or "").strip().lower()
    service_phase_age_s = _timestamp_age_seconds(service_state.get("phase_since"))
    service_payload = {
        "status": service_status,
        "reason": service_reason,
        "phase": service_state.get("phase"),
        "phase_since": service_state.get("phase_since"),
        "cycle_index": service_state.get("cycle_index"),
        "updated": service_state.get("updated"),
    }
    market_closed_mode = (
        provider_reason_normalized == "market_closed"
        or service_reason_normalized == "market_closed"
    )
    open_orders_count = _safe_nonnegative_int(broker_state.get("open_orders_count"))
    positions_count = _safe_nonnegative_int(broker_state.get("positions_count"))
    provider_disabled = provider_status_normalized in {"down", "disabled", "failed", "unreachable"}
    provider_unknown = provider_status_normalized in {"", "unknown"}
    broker_down = broker_status_normalized in {"unreachable", "down", "failed"}
    broker_degraded = broker_status_normalized in {"degraded"}
    broker_unknown = broker_status_normalized in {"", "unknown"}
    data_degraded = data_status_normalized in {"empty", "degraded"}
    service_degraded = service_status_normalized in {
        "degraded",
        "failed",
        "error",
        "halted",
        "stopped",
    }

    degraded = provider_disabled or provider_payload.get("using_backup") or (
        provider_status_normalized not in {"healthy", "ready"}
    )
    if broker_down or broker_degraded:
        degraded = True
    if provider_unknown or broker_unknown:
        degraded = True
    if data_degraded:
        degraded = True
    if service_degraded:
        degraded = True

    provider_healthy = provider_status_normalized in {"healthy", "ready"} and not data_degraded
    broker_healthy = broker_status_normalized in {"reachable", "ready", "connected"}
    overall_ok = provider_healthy and broker_healthy
    if str(ok_mode).strip().lower() == "connectivity":
        provider_connectivity_ok = (
            provider_status_normalized in {"healthy", "ready", "degraded"}
            and not data_degraded
        )
        broker_connectivity_ok = broker_status_normalized in {"reachable", "ready", "connected"}
        overall_ok = provider_connectivity_ok and broker_connectivity_ok
    offhours_market_closed_ready = (
        market_closed_mode
        and broker_healthy
        and not broker_down
        and not broker_degraded
        and not service_degraded
        and not data_degraded
        and not provider_disabled
        and not bool(provider_payload.get("using_backup"))
    )
    warmup_fast_path_enabled = _env_bool(
        "AI_TRADING_HEALTH_WARMUP_MARKET_CLOSED_FASTPATH_ENABLED",
        True,
    )
    warmup_fast_path_max_age_s = max(
        30.0,
        min(
            _env_float(
                "AI_TRADING_HEALTH_WARMUP_MARKET_CLOSED_FASTPATH_MAX_AGE_SEC",
                420.0,
            ),
            3600.0,
        ),
    )
    warmup_market_closed_ready = (
        warmup_fast_path_enabled
        and service_phase_normalized == "warmup"
        and service_reason_normalized == "warmup_cycle"
        and (market_closed_mode or _market_is_closed_now())
        and not provider_disabled
        and not data_degraded
        and not bool(provider_payload.get("using_backup"))
        and not broker_down
        and not broker_degraded
        and not service_degraded
        and (broker_healthy or broker_unknown)
        and (
            service_phase_age_s is None
            or float(service_phase_age_s) <= float(warmup_fast_path_max_age_s)
        )
    )
    if offhours_market_closed_ready:
        overall_ok = True
        degraded = False
    elif warmup_market_closed_ready:
        overall_ok = True
        degraded = False
    if force_ok_for_pytest:
        overall_ok = True
    require_database_ready = _env_bool("AI_TRADING_HEALTH_REQUIRE_DB_READY", False)
    database_configured = bool(database_readiness.get("configured"))
    database_ok = bool(database_readiness.get("ok"))
    if require_database_ready and database_configured and not database_ok:
        overall_ok = False
        degraded = True
    require_oms_invariants = _env_bool(
        "AI_TRADING_HEALTH_REQUIRE_OMS_INVARIANTS",
        _default_fail_closed_outside_tests(),
    )
    if require_oms_invariants and oms_invariants.get("enabled", False) and not bool(
        oms_invariants.get("ok")
    ):
        overall_ok = False
        degraded = True
    require_oms_lifecycle_parity = _env_bool(
        "AI_TRADING_HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY",
        _default_fail_closed_outside_tests(),
    )
    if (
        require_oms_lifecycle_parity
        and oms_lifecycle_parity.get("enabled", False)
        and not bool(oms_lifecycle_parity.get("ok"))
    ):
        overall_ok = False
        degraded = True
    require_replay_live_parity_gate = _env_bool(
        "AI_TRADING_HEALTH_REQUIRE_REPLAY_LIVE_PARITY_GATE",
        _default_fail_closed_outside_tests(),
    )
    if (
        require_replay_live_parity_gate
        and replay_live_parity_gate.get("enabled", False)
        and not bool(replay_live_parity_gate.get("ok"))
    ):
        overall_ok = False
        degraded = True
    if service_degraded:
        overall_ok = False
    if not overall_ok:
        degraded = True

    attention_flags = _build_runtime_attention_flags(
        provider_state=provider_state,
        broker_state=broker_state,
        service_state=service_state,
        database_readiness=database_readiness,
        oms_invariants=oms_invariants,
        oms_lifecycle_parity=oms_lifecycle_parity,
        replay_live_parity_gate=replay_live_parity_gate,
        require_database_ready=require_database_ready,
        require_oms_invariants=require_oms_invariants,
        require_oms_lifecycle_parity=require_oms_lifecycle_parity,
        require_replay_live_parity_gate=require_replay_live_parity_gate,
    )

    if offhours_market_closed_ready:
        resolved_status = "healthy"
    elif warmup_market_closed_ready:
        resolved_status = "healthy"
    elif degraded:
        resolved_status = "degraded"
    elif healthy_status_mode == "healthy":
        resolved_status = "healthy"
    else:
        resolved_status = service_status

    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "ok": overall_ok,
        "timestamp": timestamp,
        "service": service_name,
        "status": resolved_status,
        "service_state": service_payload,
        "data_provider": provider_payload,
        "broker": broker_payload,
        "broker_connectivity": broker_payload,
        "fallback_active": bool(provider_payload.get("using_backup")),
        "quotes_status": quote_state,
        "primary_data_provider": provider_payload,
        "gap_ratio_recent": provider_payload.get("gap_ratio_recent"),
        "gap_ratio_pct": gap_ratio_pct,
        "quote_fresh_ms": provider_payload.get("quote_fresh_ms"),
        "safe_mode": provider_payload.get("safe_mode"),
        "provider_state": provider_state,
        "cooldown_seconds_remaining": provider_payload.get("cooldown_seconds_remaining"),
        "data_status": data_status,
        "model_liveness": model_liveness,
        "database": database_readiness,
        "oms_invariants": oms_invariants,
        "oms_lifecycle_parity": oms_lifecycle_parity,
        "replay_live_parity_gate": replay_live_parity_gate,
        "attention_flags": attention_flags,
    }
    if offhours_market_closed_ready:
        payload["reason"] = "market_closed"
    elif warmup_market_closed_ready:
        payload["reason"] = "market_closed"
    elif service_reason:
        payload.setdefault("reason", service_reason)
    degrade_reason = provider_payload.get("reason")
    if degraded and degrade_reason:
        payload.setdefault("reason", degrade_reason)
    if degraded and provider_payload.get("http_code") is not None:
        payload.setdefault("http_code", provider_payload.get("http_code"))
    if data_degraded and not payload.get("reason"):
        payload["reason"] = "data_unavailable"
    if broker_down and not payload.get("reason"):
        payload["reason"] = broker_state.get("last_error") or "broker_unreachable"
    if provider_unknown and not payload.get("reason"):
        payload["reason"] = "provider_status_unknown"
    if broker_unknown and not payload.get("reason"):
        payload["reason"] = "broker_status_unknown"
    if service_degraded and not payload.get("reason"):
        payload["reason"] = "service_degraded"
    if require_database_ready and database_configured and not database_ok and not payload.get("reason"):
        payload["reason"] = "database_unhealthy"
    if (
        require_oms_invariants
        and oms_invariants.get("enabled", False)
        and not bool(oms_invariants.get("ok"))
        and not payload.get("reason")
    ):
        payload["reason"] = "oms_invariants_failed"
    if (
        require_oms_lifecycle_parity
        and oms_lifecycle_parity.get("enabled", False)
        and not bool(oms_lifecycle_parity.get("ok"))
        and not payload.get("reason")
    ):
        payload["reason"] = "oms_lifecycle_parity_failed"
    if (
        require_replay_live_parity_gate
        and replay_live_parity_gate.get("enabled", False)
        and not bool(replay_live_parity_gate.get("ok"))
        and not payload.get("reason")
    ):
        payload["reason"] = "replay_live_parity_gate_failed"
    if force_ok_for_pytest:
        payload["ok"] = True
        payload.setdefault("status", payload.get("status") or "healthy")
    return payload


def build_control_plane_snapshot(
    *,
    service_name: str = "ai-trading",
) -> dict[str, Any]:
    """Build an operator-facing control-plane snapshot from runtime state."""

    provider_state: dict[str, Any] = _safe_observe(
        runtime_state.observe_data_provider_state,
        {},
    )
    if not provider_state.get("backup"):
        provider_state = dict(provider_state)
        provider_state["backup"] = get_backup_data_provider()
    broker_state: dict[str, Any] = _safe_observe(runtime_state.observe_broker_status, {})
    service_state: dict[str, Any] = _safe_observe(runtime_state.observe_service_status, {})
    quote_state: dict[str, Any] = _safe_observe(runtime_state.observe_quote_status, {})
    model_liveness = _model_liveness_snapshot()
    database_readiness = _database_readiness_snapshot()
    oms_invariants = _oms_invariants_snapshot()
    oms_lifecycle_parity = _oms_lifecycle_parity_snapshot()
    replay_live_parity_gate = _replay_live_parity_gate_snapshot(
        oms_lifecycle_parity=oms_lifecycle_parity,
    )
    runtime_performance = _runtime_performance_snapshot()
    manual_overrides = _manual_override_snapshot()
    governance = _governance_snapshot()

    go_no_go_raw = runtime_performance.get("go_no_go")
    go_no_go = dict(go_no_go_raw) if isinstance(go_no_go_raw, Mapping) else {}
    go_no_go_observed_raw = go_no_go.get("observed")
    go_no_go_observed = (
        dict(go_no_go_observed_raw)
        if isinstance(go_no_go_observed_raw, Mapping)
        else {}
    )
    runtime_oms_event_tca_raw = runtime_performance.get("oms_event_tca")
    runtime_oms_event_tca = (
        dict(runtime_oms_event_tca_raw)
        if isinstance(runtime_oms_event_tca_raw, Mapping)
        else {}
    )
    parent_scope_rows_raw = runtime_oms_event_tca.get(
        "parent_execution_kpis_by_scope"
    )
    parent_scope_rows = (
        [dict(row) for row in parent_scope_rows_raw if isinstance(row, Mapping)]
        if isinstance(parent_scope_rows_raw, list)
        else []
    )
    outcomes_scope_rows_raw = runtime_oms_event_tca.get("event_outcomes_by_scope")
    outcomes_scope_rows = (
        [dict(row) for row in outcomes_scope_rows_raw if isinstance(row, Mapping)]
        if isinstance(outcomes_scope_rows_raw, list)
        else []
    )
    reject_reasons_raw = runtime_oms_event_tca.get("submit_reject_reasons_top")
    reject_reasons = (
        [dict(row) for row in reject_reasons_raw if isinstance(row, Mapping)]
        if isinstance(reject_reasons_raw, list)
        else []
    )
    cancel_reasons_raw = runtime_oms_event_tca.get("cancel_reasons_top")
    cancel_reasons = (
        [dict(row) for row in cancel_reasons_raw if isinstance(row, Mapping)]
        if isinstance(cancel_reasons_raw, list)
        else []
    )
    slippage_decomposition_raw = runtime_oms_event_tca.get(
        "realized_slippage_decomposition"
    )
    slippage_decomposition = (
        dict(slippage_decomposition_raw)
        if isinstance(slippage_decomposition_raw, Mapping)
        else {}
    )

    attention_flags = _build_runtime_attention_flags(
        provider_state=provider_state,
        broker_state=broker_state,
        service_state=service_state,
        database_readiness=database_readiness,
        oms_invariants=oms_invariants,
        oms_lifecycle_parity=oms_lifecycle_parity,
        replay_live_parity_gate=replay_live_parity_gate,
        include_optional_contract_failures=True,
    )

    return {
        "service": service_name,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "rollout": {
            "phase": service_state.get("phase"),
            "phase_since": service_state.get("phase_since"),
            "cycle_index": service_state.get("cycle_index"),
            "status": service_state.get("status"),
            "reason": service_state.get("reason"),
        },
        "service_state": service_state,
        "broker_health": broker_state,
        "attention_flags": attention_flags,
        "data_provider": provider_state,
        "quotes": quote_state,
        "positions": {
            "reconciliation_available": go_no_go_observed.get(
                "open_position_reconciliation_available"
            ),
            "reconciliation_consistent": go_no_go_observed.get(
                "open_position_reconciliation_consistent"
            ),
            "reconciliation_ratio": go_no_go_observed.get(
                "open_position_reconciliation_ratio"
            ),
            "mismatch_count": go_no_go_observed.get(
                "open_position_reconciliation_mismatch_count"
            ),
            "max_abs_delta_qty": go_no_go_observed.get(
                "open_position_reconciliation_max_abs_delta_qty"
            ),
            "broker_position_snapshots": runtime_performance.get(
                "broker_open_position_snapshots",
                {},
            ),
        },
        "open_orders": {
            "available": runtime_performance.get("available"),
            "source": runtime_performance.get("source"),
        },
        "execution_quality": {
            "oms_event_tca_available": runtime_oms_event_tca.get("available"),
            "submit_reject_rate_pct": runtime_oms_event_tca.get(
                "submit_reject_rate_pct"
            ),
            "cancel_to_submit_ack_rate_pct": runtime_oms_event_tca.get(
                "cancel_to_submit_ack_rate_pct"
            ),
            "reject_cancel_rate_pct": runtime_oms_event_tca.get(
                "reject_cancel_rate_pct"
            ),
            "p90_slippage_bps": runtime_oms_event_tca.get("p90_slippage_bps"),
            "parent_execution_summary_events": runtime_oms_event_tca.get(
                "parent_execution_summary_events"
            ),
            "parent_execution_kpis_by_scope": parent_scope_rows[:3],
            "event_outcomes_by_scope": outcomes_scope_rows[:5],
            "submit_reject_reasons_top": reject_reasons[:5],
            "cancel_reasons_top": cancel_reasons[:5],
            "realized_slippage_decomposition": slippage_decomposition,
            "parent_retry_per_order": go_no_go_observed.get(
                "event_tca_parent_retry_per_order"
            ),
            "parent_failed_slices_per_order": go_no_go_observed.get(
                "event_tca_parent_failed_slices_per_order"
            ),
            "parent_avg_success_ratio": go_no_go_observed.get(
                "event_tca_parent_avg_success_ratio"
            ),
            "parent_avg_arrival_slippage_bps": go_no_go_observed.get(
                "event_tca_parent_avg_arrival_slippage_bps"
            ),
            "parent_execution_consistent": go_no_go_observed.get(
                "event_tca_parent_execution_consistent"
            ),
            "parent_scope_threshold_breach_count": go_no_go_observed.get(
                "event_tca_parent_scope_threshold_breach_count"
            ),
        },
        "circuit_breakers": {
            "go_no_go_gate_passed": go_no_go.get("gate_passed"),
            "failed_checks": go_no_go.get("failed_checks"),
        },
        "liveness": model_liveness,
        "database": database_readiness,
        "oms_invariants": oms_invariants,
        "oms_lifecycle_parity": oms_lifecycle_parity,
        "replay_live_parity_gate": replay_live_parity_gate,
        "runtime_performance": runtime_performance,
        "manual_overrides": manual_overrides,
        "governance": governance,
    }


def build_service_health_payload(
    *,
    service_name: str = "ai-trading",
    force_ok_for_pytest: bool = False,
    healthy_status_mode: str = "service",
    ok_mode: str = "connectivity",
    env_error: Any | None = None,
    alpaca_context: Mapping[str, Any] | None = None,
    enrich_alpaca_from_runtime_env: bool = True,
) -> dict[str, Any]:
    """Build canonical API/health-service payload shared by all health entrypoints."""

    payload = build_runtime_health_payload(
        service_name=service_name,
        force_ok_for_pytest=force_ok_for_pytest,
        healthy_status_mode=healthy_status_mode,
        ok_mode=ok_mode,
    )
    payload["alpaca"] = build_alpaca_health_payload(
        alpaca_context,
        enrich_from_runtime_env=enrich_alpaca_from_runtime_env,
    )

    env_error_text = str(env_error or "").strip()
    if env_error_text and not payload.get("reason"):
        payload["reason"] = env_error_text

    if force_ok_for_pytest:
        payload["ok"] = True
        payload.setdefault("status", payload.get("status") or "healthy")
    return payload


def build_api_health_payload(
    *,
    service_name: str = "ai-trading",
    force_ok_for_pytest: bool = False,
    env_error: Any | None = None,
) -> dict[str, Any]:
    """Build the canonical `/health` payload used by API entrypoints."""

    errors: list[str] = []
    alpaca_import_ok = True
    sdk_ok = False
    trading_client: Any = None
    key = secret = None
    base_url = ""
    paper = False
    shadow = False

    try:
        from ai_trading.alpaca_api import ALPACA_AVAILABLE as alpaca_sdk_available
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:  # pragma: no cover - defensive
        alpaca_import_ok = False
        errors.append(str(exc) or exc.__class__.__name__)
        sdk_ok = False
    else:
        sdk_ok = bool(alpaca_sdk_available)

    if alpaca_import_ok:
        try:
            from ai_trading.core.bot_engine import _resolve_alpaca_env
            from ai_trading.core.bot_engine import trading_client as _trading_client

            trading_client = _trading_client
            key, secret, base_url = _resolve_alpaca_env()
            base_url = str(base_url or "")
            paper = bool(base_url and "paper" in base_url.lower())
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:  # pragma: no cover - defensive
            errors.append(str(exc) or exc.__class__.__name__)
            trading_client, key, secret, base_url, paper = (None, None, None, "", False)

    try:
        from ai_trading.config.management import is_shadow_mode

        shadow = bool(is_shadow_mode())
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:  # pragma: no cover - defensive
        errors.append(str(exc) or exc.__class__.__name__)
        shadow = False

    if (not alpaca_import_ok) or (not key) or (not secret):
        shadow = False

    payload = build_service_health_payload(
        service_name=service_name,
        force_ok_for_pytest=force_ok_for_pytest,
        healthy_status_mode="service",
        ok_mode="connectivity",
        env_error=env_error,
        alpaca_context={
            "sdk_ok": sdk_ok,
            "initialized": bool(trading_client),
            "client_attached": bool(trading_client),
            "has_key": bool(key),
            "has_secret": bool(secret),
            "base_url": base_url,
            "paper": paper,
            "shadow_mode": shadow,
        },
        enrich_alpaca_from_runtime_env=False,
    )
    env_error_text = str(env_error or "").strip()
    if env_error_text:
        errors.append(env_error_text)

    if errors:
        payload["ok"] = False
        payload["status"] = "degraded"
        payload["error"] = "; ".join(dict.fromkeys(errors))
    elif force_ok_for_pytest:
        payload["ok"] = True
        payload.setdefault("status", payload.get("status") or "healthy")
    return payload


def build_canonical_healthz_payload(
    *,
    service_name: str = "ai-trading",
    force_ok_for_pytest: bool = False,
    healthy_status_mode: str = "healthy",
    ok_mode: str = "connectivity",
    env_error: Any | None = None,
    alpaca_context: Mapping[str, Any] | None = None,
    enrich_alpaca_from_runtime_env: bool = True,
    error: str | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical ``/healthz`` payload shared across runtime entrypoints."""

    payload = build_service_health_payload(
        service_name=service_name,
        force_ok_for_pytest=force_ok_for_pytest,
        healthy_status_mode=healthy_status_mode,
        ok_mode=ok_mode,
        env_error=env_error,
        alpaca_context=alpaca_context,
        enrich_alpaca_from_runtime_env=enrich_alpaca_from_runtime_env,
    )
    error_text = str(error or "").strip()
    if error_text:
        payload["ok"] = False
        payload["status"] = "degraded"
        payload["error"] = error_text
    if extras:
        payload.update(dict(extras))
    return payload


def build_health_exception_payload(
    exc: Exception,
    *,
    service_name: str = "ai-trading",
) -> dict[str, Any]:
    """Build a normalized degraded payload for health handler exceptions."""

    return {
        "ok": False,
        "status": "degraded",
        "service": service_name,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "error": str(exc),
    }


def build_health_json_response(
    payload: Mapping[str, Any],
    status: int,
    *,
    jsonify_fn: Callable[[Mapping[str, Any]], Any],
) -> Any:
    """Build a robust Flask-compatible JSON response for health handlers."""

    response = jsonify_fn(dict(payload))
    if response is None or isinstance(response, Mapping):
        return dict(payload) if status == 200 else (dict(payload), status)
    if not (
        callable(getattr(response, "get_data", None))
        or callable(getattr(response, "get_json", None))
        or hasattr(response, "status_code")
    ):
        return dict(payload) if status == 200 else (dict(payload), status)
    try:
        response.status_code = status
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        pass
    return response


def register_healthz_routes(
    app: Any,
    *,
    payload_builder: Callable[[], Mapping[str, Any]],
    response_builder: Callable[[dict[str, Any], int], Any],
    service_name: str = "ai-trading",
    routes: tuple[str, ...] = ("/healthz",),
    methods: tuple[str, ...] = ("GET",),
    logger: Any | None = None,
    error_event: str = "HEALTH_CHECK_FAILED",
) -> None:
    """Register canonical health routes on ``app`` using shared runtime logic."""

    def _healthz_handler() -> Any:
        try:
            payload = dict(payload_builder())
            return response_builder(payload, 200)
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
            if logger is not None:
                try:
                    logger.exception(error_event, exc_info=exc)
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    pass
            fallback_payload = build_health_exception_payload(exc, service_name=service_name)
            return response_builder(fallback_payload, 500)

    for route in routes:
        decorator = app.route(route, methods=list(methods))
        decorator(_healthz_handler)


__all__ = [
    "build_runtime_health_payload",
    "build_control_plane_snapshot",
    "build_alpaca_health_payload",
    "build_service_health_payload",
    "build_api_health_payload",
    "build_canonical_healthz_payload",
    "build_health_exception_payload",
    "build_health_json_response",
    "register_healthz_routes",
]
