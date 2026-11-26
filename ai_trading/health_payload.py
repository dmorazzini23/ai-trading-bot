from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import json
import math
import time as pytime
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Mapping

from ai_trading.runtime import paper_sampling
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.config.launch_profiles import (
    launch_profile_payload,
    provider_authority_allows,
    resolve_launch_profile,
)
from ai_trading.config.management import get_env
from ai_trading.logging import logger
from ai_trading.logging.emit_once import emit_once
from ai_trading.governance.paths import resolve_governance_base_path
from ai_trading.settings import get_backup_data_provider
from ai_trading.telemetry import runtime_state
from ai_trading.telemetry.trade_updates_stream import observe_trade_updates_stream_state
from ai_trading.governance.replay_live_parity import summarize_replay_live_parity_gate

try:
    from sqlalchemy.exc import SQLAlchemyError as _SQLAlchemyError
except ImportError:  # pragma: no cover - sqlalchemy is a runtime dependency
    _SQLALCHEMY_FALLBACK_EXC: tuple[type[Exception], ...] = ()
else:
    _SQLALCHEMY_FALLBACK_EXC = (_SQLAlchemyError,)

_HEALTH_FALLBACK_EXC: tuple[type[Exception], ...] = (
    AI_TRADING_FALLBACK_EXCEPTIONS + _SQLALCHEMY_FALLBACK_EXC
)

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
    except (TypeError, ValueError, OverflowError):
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
    observe_oms_invariants: bool = False,
    observe_oms_lifecycle_parity: bool = False,
    trade_updates_stream: Mapping[str, Any] | None = None,
    decision_loop: Mapping[str, Any] | None = None,
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
    if broker_state.get("fresh") is False:
        flags.append("broker_snapshot_stale")
    if broker_state.get("open_orders_fresh") is False:
        flags.append("broker_open_orders_unknown")
    if broker_state.get("positions_fresh") is False:
        flags.append("broker_positions_unknown")
    if service_status_normalized in {"degraded", "failed", "error", "halted", "stopped"}:
        flags.append("service_degraded")
    if service_status_normalized in {"halted", "stopped"} or any(
        token in service_reason_normalized for token in ("halt", "hard_stop")
    ):
        flags.append("service_halt_active")
    if service_reason_normalized in {
        "trade_updates_stream_failed",
        "trade_updates_stream_exited",
    } or (
        isinstance(trade_updates_stream, Mapping)
        and bool(trade_updates_stream.get("degraded"))
    ):
        flags.append("trade_updates_stream_degraded")
    if (
        service_reason_normalized == "replay_live_parity_gate_failed"
        or (
            isinstance(replay_live_parity_gate, Mapping)
            and (
                (
                    require_replay_live_parity_gate
                    and (
                        not bool(replay_live_parity_gate.get("enabled", False))
                        or not bool(replay_live_parity_gate.get("available", False))
                        or not bool(replay_live_parity_gate.get("ok"))
                    )
                )
                or (
                    include_optional_contract_failures
                    and bool(replay_live_parity_gate.get("enabled", False))
                    and (
                        not bool(replay_live_parity_gate.get("available", False))
                        or not bool(replay_live_parity_gate.get("ok"))
                    )
                )
            )
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
            or observe_oms_invariants
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
            or observe_oms_lifecycle_parity
        )
        and oms_lifecycle_parity.get("enabled", False)
        and not bool(oms_lifecycle_parity.get("ok"))
    ):
        flags.append("oms_lifecycle_parity_failed")
    # Non-fatal: the decision-loop watchdog is alert-only and never changes ok/status.
    if isinstance(decision_loop, Mapping) and bool(decision_loop.get("stalled")):
        flags.append("decision_loop_stalled")

    return _dedupe_flags(flags)


def _build_contract_gate_status(
    snapshot: Mapping[str, Any],
    *,
    required: bool,
    failure_reason: str,
    attention_flag: str,
    action: str,
) -> dict[str, Any]:
    enabled = bool(snapshot.get("enabled", False))
    ok = bool(snapshot.get("ok", True)) if enabled else True
    failure_observed = enabled and not ok
    status = "disabled"
    if enabled and failure_observed and required:
        status = "required_failed"
    elif enabled and failure_observed:
        status = "observed_failure"
    elif enabled:
        status = "passed"

    gate: dict[str, Any] = {
        "enabled": enabled,
        "required": bool(required),
        "ok": ok,
        "status": status,
        "failure_observed": failure_observed,
    }
    if failure_observed:
        gate["reason"] = failure_reason
        gate["attention_flag"] = attention_flag
        gate["action"] = action
        if snapshot.get("reason"):
            gate["detail"] = snapshot.get("reason")
        if snapshot.get("total_violations") is not None:
            gate["total_violations"] = snapshot.get("total_violations")
    return gate


# The replay gate's ``replay_counterfactual`` check is an economics test (enough
# capped samples with a positive net edge), not a replay/live parity test.
REPLAY_ECONOMICS_CHECK = "replay_counterfactual"
REPLAY_PARITY_ATTENTION_FLAG = "replay_live_parity_gate_failed"


def _finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _replay_economics_summary(
    gate: Mapping[str, Any],
    *,
    withheld_attention_flag: str | None,
) -> dict[str, Any]:
    """Describe the replay counterfactual (economics) check for display only.

    Never changes the gate's ``ok`` or ``failed_checks``.
    """

    observed_raw = gate.get("observed")
    observed = observed_raw if isinstance(observed_raw, Mapping) else {}
    thresholds_raw = gate.get("thresholds")
    thresholds = thresholds_raw if isinstance(thresholds_raw, Mapping) else {}
    governance_raw = gate.get("replay_governance")
    governance = governance_raw if isinstance(governance_raw, Mapping) else {}
    counterfactual_raw = governance.get("counterfactual")
    counterfactual = counterfactual_raw if isinstance(counterfactual_raw, Mapping) else {}
    criteria_raw = counterfactual.get("checks")
    criteria = criteria_raw if isinstance(criteria_raw, Mapping) else {}

    qualified = observed.get("replay_counterfactual_passed") is True
    available = observed.get("replay_counterfactual_available") is True
    required = thresholds.get("require_counterfactual_passed") is not False
    sample_size = _safe_nonnegative_int(
        observed.get("replay_counterfactual_candidate_samples")
    )
    min_samples = _safe_nonnegative_int(
        observed.get("replay_counterfactual_required_samples")
    )
    net_edge_bps = _finite_float_or_none(
        observed.get("replay_counterfactual_candidate_net_edge_bps")
    )
    min_net_edge_bps = _finite_float_or_none(
        observed.get("replay_counterfactual_required_net_edge_bps")
    )
    failed_criteria = [
        str(name) for name, passed in criteria.items() if passed is not True
    ]
    if qualified:
        reason = "qualified"
    elif not available:
        reason = "counterfactual_unavailable"
    elif not required:
        reason = "not_required"
    elif min_samples and sample_size < min_samples:
        reason = "insufficient_samples"
    elif (
        net_edge_bps is not None
        and min_net_edge_bps is not None
        and net_edge_bps <= min_net_edge_bps
    ):
        reason = "net_edge_not_above_minimum"
    else:
        reason = "counterfactual_not_passed"
    edge_text = f"{net_edge_bps:.2f}" if net_edge_bps is not None else "n/a"
    min_edge_text = f"{min_net_edge_bps:.2f}" if min_net_edge_bps is not None else "n/a"
    summary = (
        f"Replay economics {'qualified' if qualified else 'not qualified'}: "
        f"{sample_size} of {min_samples} capped samples, net edge {edge_text} bps "
        f"(minimum {min_edge_text} bps)."
    )
    if withheld_attention_flag:
        summary += (
            " Paper mode: informational, not a parity failure, so the replay "
            "parity attention flag is not raised."
        )
    return {
        "check": REPLAY_ECONOMICS_CHECK,
        "required": required,
        "available": available,
        "qualified": qualified,
        "sample_size": sample_size,
        "min_samples": min_samples,
        "net_edge_bps": net_edge_bps,
        "min_net_edge_bps": min_net_edge_bps,
        "failed_criteria": failed_criteria,
        "reason": reason,
        "withheld_attention_flag": withheld_attention_flag,
        "summary": summary,
    }


_REPLAY_ECONOMICS_FAILURE_REASONS = frozenset(
    {"insufficient_samples", "net_edge_not_above_minimum", "counterfactual_not_passed"}
)
_PAPER_EXECUTION_MODES = frozenset({"paper", "sim", "simulation"})
_EXPLICIT_FALSE_TOKENS = frozenset({"", "0", "false", "no", "off"})


def _paper_mode_withhold_blockers(
    *, launch_profile: Any, launch_profile_name: str
) -> list[str]:
    """Reasons paper mode is not positively evidenced; any keeps the flag.

    Used only to decide whether the replay parity flag may be withheld.
    """

    blockers: list[str] = []
    if launch_profile is None or launch_profile_name in {"", "unknown"}:
        blockers.append("launch_profile_unresolved")
    elif launch_profile_name.startswith("live_"):
        blockers.append("launch_profile_live")
    try:
        # Raw variable only: the config default must not count as evidence.
        raw_mode = get_env("EXECUTION_MODE", None, cast=str, resolve_aliases=False)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        raw_mode = None
    if _normalized_token(raw_mode) not in _PAPER_EXECUTION_MODES:
        blockers.append("execution_mode_not_explicit_paper")
    try:
        from ai_trading.config.management import get_trading_config

        config_paper = getattr(get_trading_config(), "paper", None)
    except _HEALTH_FALLBACK_EXC:
        config_paper = None
    if config_paper is not True:
        blockers.append("runtime_paper_inference_not_paper")
    try:
        raw_enforce = get_env(
            "AI_TRADING_HEALTH_REPLAY_GATE_ENFORCE_IN_PAPER",
            None,
            cast=str,
            resolve_aliases=False,
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        raw_enforce = "unreadable"
    if _normalized_token(raw_enforce) not in _EXPLICIT_FALSE_TOKENS:
        blockers.append("paper_enforcement_setting_not_false")
    return blockers


def _runtime_replay_gate_withhold_blockers(
    record: Any,
    *,
    service_reason_normalized: str,
    health_replay_policy_hash: str,
) -> list[str]:
    """Reasons the trading cycle's own gate does not confirm economics-only.

    The health snapshot has no expected policy hash, so the cycle's result only
    counts for a replay artifact whose policy hash equals the hash the cycle
    expected. A later artifact with that same hash is still covered by
    health's own freshness and violation checks.
    """

    if not isinstance(record, Mapping):
        return ["runtime_gate_missing"]
    blockers: list[str] = []
    if record.get("error"):
        blockers.append("runtime_gate_error")
    if record.get("enabled") is not True or record.get("available") is not True:
        blockers.append("runtime_gate_unavailable")
    runtime_failed = record.get("failed_checks")
    if not isinstance(runtime_failed, list):
        blockers.append("runtime_failed_checks_unknown")
    elif runtime_failed != [REPLAY_ECONOMICS_CHECK]:
        blockers.append("runtime_gate_not_economics_only")
    if (
        service_reason_normalized == REPLAY_PARITY_ATTENTION_FLAG
        and _normalized_token(record.get("service_reason")) != REPLAY_PARITY_ATTENTION_FLAG
    ):
        blockers.append("runtime_gate_stale_for_service_reason")
    expected_hash = record.get("expected_policy_hash")
    if not isinstance(expected_hash, str) or not expected_hash.strip():
        blockers.append("runtime_expected_policy_hash_missing")
    elif health_replay_policy_hash != expected_hash.strip():
        blockers.append("replay_policy_hash_mismatch")
    return blockers


def _running_policy_hash_blockers(
    running: Any,
    *,
    expected_policy_hash: Any,
    health_replay_policy_hash: str,
) -> list[str]:
    """The running policy hash must match what the cycle and health checked.

    The running hash can be recompiled without a replay gate evaluation (for
    example while the market is closed, or while the cycle reuses its cached
    gate), so a cycle result only counts for the hash still running.
    """

    running_hash = running.get("policy_hash") if isinstance(running, Mapping) else None
    if not isinstance(running_hash, str) or not running_hash.strip():
        return ["running_policy_hash_missing"]
    running_hash = running_hash.strip()
    expected = expected_policy_hash.strip() if isinstance(expected_policy_hash, str) else ""
    if running_hash != expected or running_hash != health_replay_policy_hash:
        return ["running_policy_hash_mismatch"]
    return []


def _oms_lifecycle_limits_blockers() -> list[str]:
    """Health and the cycle must scan the same number of OMS lifecycle rows.

    Each limit is read exactly as its own call site reads it.
    """

    try:
        health_limit = int(
            get_env(
                "AI_TRADING_HEALTH_OMS_LIFECYCLE_PARITY_LIMIT",
                5000,
                cast=int,
                resolve_aliases=False,
            )
        )
        cycle_limit = int(
            get_env(
                "AI_TRADING_RUNTIME_PERF_OMS_LIFECYCLE_PARITY_LIMIT",
                5000,
                cast=int,
            )
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return ["oms_lifecycle_limits_unreadable"]
    return [] if health_limit == cycle_limit else ["oms_lifecycle_limits_differ"]


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
            except _HEALTH_FALLBACK_EXC as exc:  # pragma: no cover - defensive
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


def _decision_loop_snapshot() -> dict[str, Any]:
    try:
        from ai_trading.monitoring.decision_loop_watchdog import (
            get_decision_loop_snapshot,
        )

        return dict(get_decision_loop_snapshot())
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        logger.debug("DECISION_LOOP_SNAPSHOT_FAILED", extra={"error": str(exc)})
        return {"monitored": False, "stalled": False, "error": str(exc)}


def _database_readiness_snapshot() -> dict[str, Any]:
    enabled = _env_bool("AI_TRADING_HEALTH_DB_READINESS_ENABLED", True)
    if not enabled:
        return {"enabled": False}
    required = _env_bool("AI_TRADING_HEALTH_REQUIRE_DB_READY", False)

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
                "20260506_0001",
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
            "ok": not required,
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
    except _HEALTH_FALLBACK_EXC as exc:
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
            except _HEALTH_FALLBACK_EXC:
                pass


def _database_readiness_snapshot_cached() -> dict[str, Any]:
    if (
        _env_bool("AI_TRADING_HEALTH_REQUIRE_DB_READY", False)
        or not _health_snapshot_cache_enabled()
    ):
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
    except _HEALTH_FALLBACK_EXC as exc:
        return {
            "enabled": True,
            "available": False,
            "ok": False,
            "reason": "oms_invariants_unavailable",
            "error": str(exc),
        }


def _oms_invariants_snapshot_cached() -> dict[str, Any]:
    if (
        _env_bool(
            "AI_TRADING_HEALTH_REQUIRE_OMS_INVARIANTS",
            _default_fail_closed_outside_tests(),
        )
        or not _health_snapshot_cache_enabled()
    ):
        return _oms_invariants_snapshot()
    return _cached_background_snapshot(
        name="oms_invariants",
        ttl_seconds=_health_snapshot_ttl_seconds("oms_invariants", 300.0),
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
    except _HEALTH_FALLBACK_EXC as exc:
        return {
            "enabled": True,
            "available": False,
            "ok": False,
            "reason": "oms_lifecycle_parity_unavailable",
            "error": str(exc),
        }


def _oms_lifecycle_parity_snapshot_cached() -> dict[str, Any]:
    if (
        _env_bool(
            "AI_TRADING_HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY",
            _default_fail_closed_outside_tests(),
        )
        or not _health_snapshot_cache_enabled()
    ):
        return _oms_lifecycle_parity_snapshot()
    return _cached_background_snapshot(
        name="oms_lifecycle_parity",
        ttl_seconds=_health_snapshot_ttl_seconds("oms_lifecycle_parity", 300.0),
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
        result: dict[str, Any] = summarize_replay_live_parity_gate(
            oms_lifecycle_parity=oms_lifecycle_parity,
        )
        return result
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        return {"enabled": True, "available": False, "ok": False, "error": str(exc)}


def _replay_live_parity_gate_snapshot_cached(
    *,
    oms_lifecycle_parity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        _env_bool(
            "AI_TRADING_HEALTH_REQUIRE_REPLAY_LIVE_PARITY_GATE",
            _default_fail_closed_outside_tests(),
        )
        or not _health_snapshot_cache_enabled()
    ):
        return _replay_live_parity_gate_snapshot(
            oms_lifecycle_parity=oms_lifecycle_parity,
        )
    if isinstance(oms_lifecycle_parity, Mapping) and (
        bool(oms_lifecycle_parity.get("refreshing"))
        or str(oms_lifecycle_parity.get("reason") or "").strip().lower()
        == "warming_up"
        or (
            bool(oms_lifecycle_parity.get("enabled", False))
            and not bool(oms_lifecycle_parity.get("available", True))
        )
    ):
        return _replay_live_parity_gate_snapshot(
            oms_lifecycle_parity=oms_lifecycle_parity,
        )
    return _cached_background_snapshot(
        name="replay_live_parity_gate",
        ttl_seconds=_health_snapshot_ttl_seconds("replay_live_parity_gate", 300.0),
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


_TRADE_UPDATES_STREAM_UNAVAILABLE: dict[str, Any] = {
    "enabled": False,
    "connected": False,
    "degraded": False,
    "shadow_only": True,
    "last_event_at": None,
    "events_today": 0,
    "reconnects": 0,
    "parity": None,
    "reason": "trade_updates_stream_state_unavailable",
}


def _trade_updates_stream_snapshot() -> dict[str, Any]:
    """Cached in-memory stream state; no I/O on the request path."""

    snapshot = _safe_observe(observe_trade_updates_stream_state, None)
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    return dict(_TRADE_UPDATES_STREAM_UNAVAILABLE)


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
    path: Path = resolve_governance_base_path()
    return path


def _read_jsonl_tail(path: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        max_bytes = 256_000
        with path.open("rb") as handle:
            handle.seek(max(0, size - max_bytes))
            data = handle.read(max_bytes)
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
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
    trade_history = payload.get("trade_history")
    broker_open_position_snapshots = {}
    if isinstance(trade_history, Mapping):
        snapshots = trade_history.get("broker_open_position_snapshots")
        if isinstance(snapshots, Mapping):
            broker_open_position_snapshots = dict(snapshots)
    return {
        "available": bool(payload),
        "path": str(resolved),
        "go_no_go": go_no_go,
        "generated_at": payload.get("generated_at"),
        "source": payload.get("source"),
        "broker_open_position_snapshots": broker_open_position_snapshots,
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


def _latest_training_attempt_snapshot() -> dict[str, Any]:
    """Expose research freshness separately from governed serving authority."""

    try:
        configured = str(
            get_env(
                "AI_TRADING_AFTER_HOURS_REPORT_LATEST_PATH",
                "runtime/research_reports/after_hours_training_latest.json",
                cast=str,
                resolve_aliases=False,
            )
            or "runtime/research_reports/after_hours_training_latest.json"
        ).strip()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        configured = "runtime/research_reports/after_hours_training_latest.json"
    payload, resolved = _read_json_mapping_artifact(
        configured_path=configured,
        default_relative="runtime/research_reports/after_hours_training_latest.json",
    )
    report_raw = payload.get("report")
    report = dict(report_raw) if isinstance(report_raw, Mapping) else dict(payload)
    generated_at = payload.get("generated_at") or report.get("timestamp")
    return {
        "available": bool(report),
        "path": str(resolved),
        "generated_at": generated_at,
        "age_seconds": _timestamp_age_seconds(generated_at),
        "status": report.get("status") if report else "unavailable",
        "reason": report.get("reason") if report else "artifact_unavailable",
        "rows": report.get("rows"),
        "governance_status": report.get("governance_status"),
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
    }


def _execution_quality_governor_snapshot() -> dict[str, Any]:
    try:
        latest_path = str(
            get_env(
                "AI_TRADING_EXECUTION_QUALITY_GOVERNOR_REPORT_PATH",
                "runtime/execution_quality_governor_latest.json",
                cast=str,
                resolve_aliases=False,
            )
            or "runtime/execution_quality_governor_latest.json"
        ).strip()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        latest_path = "runtime/execution_quality_governor_latest.json"
    payload, resolved = _read_json_mapping_artifact(
        configured_path=latest_path,
        default_relative="runtime/execution_quality_governor_latest.json",
    )
    status_raw = payload.get("status") if isinstance(payload, Mapping) else {}
    observed_raw = payload.get("observed") if isinstance(payload, Mapping) else {}
    actions_raw = payload.get("actions") if isinstance(payload, Mapping) else {}
    window_raw = payload.get("window") if isinstance(payload, Mapping) else {}
    by_symbol_raw = payload.get("by_symbol") if isinstance(payload, Mapping) else []
    return {
        "available": bool(payload),
        "path": str(resolved),
        "schema_version": payload.get("schema_version"),
        "generated_at": payload.get("generated_at"),
        "status": dict(status_raw) if isinstance(status_raw, Mapping) else {},
        "observed": dict(observed_raw) if isinstance(observed_raw, Mapping) else {},
        "actions": dict(actions_raw) if isinstance(actions_raw, Mapping) else {},
        "window": dict(window_raw) if isinstance(window_raw, Mapping) else {},
        "by_symbol": (
            [dict(row) for row in by_symbol_raw if isinstance(row, Mapping)][:5]
            if isinstance(by_symbol_raw, list)
            else []
        ),
    }


def _live_cost_model_snapshot() -> dict[str, Any]:
    try:
        latest_path = str(
            get_env(
                "AI_TRADING_LIVE_COST_MODEL_PATH",
                "runtime/live_cost_model_latest.json",
                cast=str,
                resolve_aliases=False,
            )
            or "runtime/live_cost_model_latest.json"
        ).strip()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        latest_path = "runtime/live_cost_model_latest.json"
    payload, resolved = _read_json_mapping_artifact(
        configured_path=latest_path,
        default_relative="runtime/live_cost_model_latest.json",
    )
    status_raw = payload.get("status") if isinstance(payload, Mapping) else {}
    observed_raw = payload.get("observed") if isinstance(payload, Mapping) else {}
    window_raw = payload.get("window") if isinstance(payload, Mapping) else {}
    alerts_raw = payload.get("alerts") if isinstance(payload, Mapping) else {}
    sources_raw = payload.get("sources") if isinstance(payload, Mapping) else {}
    rows_raw = (
        payload.get("by_symbol_side_session") if isinstance(payload, Mapping) else []
    )
    rows = (
        [dict(row) for row in rows_raw if isinstance(row, Mapping)]
        if isinstance(rows_raw, list)
        else []
    )
    last_observed_values = [
        row.get("last_observed_at") for row in rows if row.get("last_observed_at")
    ]
    latest_observed = max((str(value) for value in last_observed_values), default=None)
    return {
        "available": bool(payload),
        "path": str(resolved),
        "schema_version": payload.get("schema_version"),
        "generated_at": payload.get("generated_at"),
        "age_seconds": _timestamp_age_seconds(payload.get("generated_at")),
        "last_observed_at": latest_observed,
        "last_sample_age_seconds": _timestamp_age_seconds(latest_observed),
        "status": dict(status_raw) if isinstance(status_raw, Mapping) else {},
        "observed": dict(observed_raw) if isinstance(observed_raw, Mapping) else {},
        "window": dict(window_raw) if isinstance(window_raw, Mapping) else {},
        "sources": dict(sources_raw) if isinstance(sources_raw, Mapping) else {},
        "alerts": dict(alerts_raw) if isinstance(alerts_raw, Mapping) else {},
        "by_symbol_side_session": rows[:8],
    }


def _symbol_universe_scorecard_snapshot() -> dict[str, Any]:
    try:
        latest_path = str(
            get_env(
                "AI_TRADING_SYMBOL_UNIVERSE_SCORECARD_PATH",
                "runtime/symbol_universe_scorecard_latest.json",
                cast=str,
                resolve_aliases=False,
            )
            or "runtime/symbol_universe_scorecard_latest.json"
        ).strip()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        latest_path = "runtime/symbol_universe_scorecard_latest.json"
    payload, resolved = _read_json_mapping_artifact(
        configured_path=latest_path,
        default_relative="runtime/symbol_universe_scorecard_latest.json",
    )
    status_raw = payload.get("status") if isinstance(payload, Mapping) else {}
    summary_raw = payload.get("summary") if isinstance(payload, Mapping) else {}
    thresholds_raw = payload.get("thresholds") if isinstance(payload, Mapping) else {}
    policy_raw = payload.get("policy") if isinstance(payload, Mapping) else {}
    symbols_raw = payload.get("symbols") if isinstance(payload, Mapping) else []
    symbols = (
        [dict(row) for row in symbols_raw if isinstance(row, Mapping)]
        if isinstance(symbols_raw, list)
        else []
    )
    bottom_symbols = sorted(
        symbols,
        key=lambda row: (
            str(row.get("effective_mode")) != "disabled",
            str(row.get("effective_mode")) != "shadow_only",
            float(row.get("quality_score") or 0.0),
            str(row.get("symbol") or ""),
        ),
    )[:8]
    top_symbols = sorted(
        symbols,
        key=lambda row: (
            float(row.get("quality_score") or 0.0),
            int(row.get("sample_count") or 0),
            str(row.get("symbol") or ""),
        ),
        reverse=True,
    )[:8]
    return {
        "available": bool(payload),
        "path": str(resolved),
        "schema_version": payload.get("schema_version"),
        "generated_at": payload.get("generated_at"),
        "status": dict(status_raw) if isinstance(status_raw, Mapping) else {},
        "summary": dict(summary_raw) if isinstance(summary_raw, Mapping) else {},
        "thresholds": dict(thresholds_raw) if isinstance(thresholds_raw, Mapping) else {},
        "policy": dict(policy_raw) if isinstance(policy_raw, Mapping) else {},
        "top_symbols": top_symbols,
        "bottom_symbols": bottom_symbols,
    }


def _runtime_decay_controls_snapshot() -> dict[str, Any]:
    try:
        latest_path = str(
            get_env(
                "AI_TRADING_RUNTIME_DECAY_CONTROLS_PATH",
                "runtime/runtime_decay_controls_latest.json",
                cast=str,
                resolve_aliases=False,
            )
            or "runtime/runtime_decay_controls_latest.json"
        ).strip()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        latest_path = "runtime/runtime_decay_controls_latest.json"
    payload, resolved = _read_json_mapping_artifact(
        configured_path=latest_path,
        default_relative="runtime/runtime_decay_controls_latest.json",
    )
    status_raw = payload.get("status") if isinstance(payload, Mapping) else {}
    observed_raw = payload.get("observed") if isinstance(payload, Mapping) else {}
    actions_raw = payload.get("actions") if isinstance(payload, Mapping) else {}
    recovery_raw = payload.get("recovery") if isinstance(payload, Mapping) else {}
    return {
        "available": bool(payload),
        "path": str(resolved),
        "schema_version": payload.get("schema_version"),
        "generated_at": payload.get("generated_at"),
        "age_seconds": _timestamp_age_seconds(payload.get("generated_at")),
        "status": dict(status_raw) if isinstance(status_raw, Mapping) else {},
        "observed": dict(observed_raw) if isinstance(observed_raw, Mapping) else {},
        "actions": dict(actions_raw) if isinstance(actions_raw, Mapping) else {},
        "recovery": dict(recovery_raw) if isinstance(recovery_raw, Mapping) else {},
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


def _drawdown_breaker_snapshot() -> dict[str, Any]:
    """Return the informational drawdown-breaker section; never raises."""

    unavailable: dict[str, Any] = {"available": False, "reason": "snapshot_failed"}
    try:
        from ai_trading.risk.circuit_breakers import drawdown_breaker_health_section
    except ImportError:
        emit_once(
            logger,
            "DRAWDOWN_BREAKER_HEALTH_UNAVAILABLE:import_failed",
            "warning",
            "DRAWDOWN_BREAKER_HEALTH_UNAVAILABLE",
            reason="import_failed",
        )
        return unavailable
    try:
        return drawdown_breaker_health_section()
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        # Once per day per process: this runs on every health poll.
        emit_once(
            logger,
            "DRAWDOWN_BREAKER_HEALTH_UNAVAILABLE:snapshot_failed",
            "warning",
            "DRAWDOWN_BREAKER_HEALTH_UNAVAILABLE",
            reason="snapshot_failed",
            error=f"{exc.__class__.__name__}: {exc}",
        )
        return unavailable


def _broker_account_configuration_snapshot() -> dict[str, Any]:
    """Return the cached read-only broker account-configuration check."""

    unavailable: dict[str, Any] = {
        "ok": None,
        "available": False,
        "reason": "snapshot_failed",
        "mismatches": [],
        "unreported_fields": [],
        "fetched_at": None,
    }
    try:
        from ai_trading.broker.account_configuration import health_section
    except ImportError:
        return unavailable
    section: dict[str, Any] = _safe_observe(health_section, unavailable)
    return section


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
        "preflight": {
            "available": True,
            "status": "ready",
            "failure_kind": None,
            "last_error": None,
            "last_failure_at": None,
            "retry_at": None,
            "retry_in_seconds": None,
            "consecutive_failures": 0,
        },
    }
    if isinstance(context, Mapping):
        for key, value in context.items():
            if key not in payload:
                continue
            if isinstance(payload[key], bool):
                payload[key] = bool(value)
            elif value is not None:
                payload[key] = value

    try:
        from ai_trading.alpaca_api import get_alpaca_service_status

        preflight = get_alpaca_service_status()
        if isinstance(preflight, Mapping):
            payload["preflight"] = dict(preflight)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        pass

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


def _operating_mode_snapshot() -> dict[str, Any]:
    """Report strategy authority without changing any readiness decision."""
    qualified = paper_sampling.qualified_strategy_available()
    result: dict[str, Any] = {
        "strategy_state": "qualified_strategy" if qualified else paper_sampling.NO_QUALIFIED_STRATEGY,
        "qualified_strategy": qualified,
        "rule_paper_sampling_enabled": False,
        "results_are_strategy_performance": False,
        "promotion_authority": False,
    }
    try:
        from ai_trading.config.management import get_trading_config
        from ai_trading.runtime.paper_sampling import rule_paper_sampling_allowed

        allowed = rule_paper_sampling_allowed(get_trading_config())
    except _HEALTH_FALLBACK_EXC:
        return {**result, "mode": "close_only",
                "summary": "No qualified strategy; paper rule sampling could not be verified from configuration."}
    if allowed:
        return {**result, "rule_paper_sampling_enabled": True, "mode": "paper_rule_sampling",
                "summary": "No qualified strategy; the rule scorer trades only as labelled paper sampling. These trades are not strategy performance."}
    return {**result, "mode": "close_only",
            "summary": "No qualified strategy; no new exposure. Held positions never grow; only the position backstop and exposure-reducing gates can reduce them."}


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
    decision_loop = _decision_loop_snapshot()
    database_readiness = _database_readiness_snapshot_cached()
    oms_invariants = _oms_invariants_snapshot_cached()
    oms_lifecycle_parity = _oms_lifecycle_parity_snapshot_cached()
    replay_live_parity_gate = _replay_live_parity_gate_snapshot_cached(
        oms_lifecycle_parity=oms_lifecycle_parity,
    )
    trade_updates_stream = _trade_updates_stream_snapshot()
    launch_profile = _safe_observe(resolve_launch_profile, None)
    launch_profile_status = (
        launch_profile_payload(launch_profile)
        if launch_profile is not None
        else {"name": "unknown", "error": "launch_profile_unavailable"}
    )
    provider_authority_ok, provider_authority = (
        provider_authority_allows(
            profile=launch_profile,
            provider_state=provider_state,
            quote_state=quote_state,
        )
        if launch_profile is not None
        else (False, {"reasons": ["launch_profile_unavailable"]})
    )
    execution_mode = _normalized_token(
        get_env("EXECUTION_MODE", "paper", cast=str)
    ) or "paper"
    launch_profile_name = _normalized_token(launch_profile_status.get("name"))
    live_execution_mode = bool(
        execution_mode == "live" or launch_profile_name.startswith("live_")
    )
    paper_evidence_mode = bool(
        execution_mode in {"paper", "sim", "simulation"}
        and not live_execution_mode
    )
    latest_training_attempt = _latest_training_attempt_snapshot()
    require_replay_live_parity_gate = _env_bool(
        "AI_TRADING_HEALTH_REQUIRE_REPLAY_LIVE_PARITY_GATE",
        _default_fail_closed_outside_tests(),
    )
    enforce_replay_gate_in_paper = _env_bool(
        "AI_TRADING_HEALTH_REPLAY_GATE_ENFORCE_IN_PAPER",
        False,
    )
    replay_gate_enabled = bool(replay_live_parity_gate.get("enabled", False))
    replay_gate_available = bool(replay_live_parity_gate.get("available", False))
    replay_gate_ok = bool(replay_live_parity_gate.get("ok"))
    replay_gate_required_failure = bool(
        require_replay_live_parity_gate
        and (
            not replay_gate_enabled
            or not replay_gate_available
            or not replay_gate_ok
        )
    )
    replay_gate_failure = bool(
        replay_gate_required_failure
        or (
            replay_gate_enabled
            and (not replay_gate_available or not replay_gate_ok)
        )
    )
    replay_gate_enforced_for_mode = bool(
        require_replay_live_parity_gate
        and (
            not paper_evidence_mode
            or enforce_replay_gate_in_paper
        )
    )
    replay_gate_monitor_only = bool(
        replay_gate_failure
        and paper_evidence_mode
        and not enforce_replay_gate_in_paper
    )
    replay_gate_readiness_failure = bool(
        replay_gate_enforced_for_mode
        and replay_gate_required_failure
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
    broker_fresh_raw = broker_state.get("fresh")
    broker_fresh = (
        bool(broker_fresh_raw)
        if broker_fresh_raw is not None
        else bool(
            broker_connected
            and broker_status_normalized in {"reachable", "ready", "connected"}
        )
    )
    if not broker_fresh:
        broker_connected = False
    open_orders_fresh_raw = broker_state.get("open_orders_fresh")
    open_orders_fresh = (
        bool(open_orders_fresh_raw)
        if open_orders_fresh_raw is not None
        else broker_fresh
    )
    positions_fresh_raw = broker_state.get("positions_fresh")
    positions_fresh = (
        bool(positions_fresh_raw)
        if positions_fresh_raw is not None
        else broker_fresh
    )
    failed_components_raw = broker_state.get("failed_components")
    if isinstance(failed_components_raw, (list, tuple, set)):
        failed_components = [
            str(component)
            for component in failed_components_raw
            if str(component).strip()
        ]
    elif failed_components_raw in (None, ""):
        failed_components = []
    else:
        failed_components = [str(failed_components_raw)]
    broker_payload = {
        "status": broker_status,
        "connected": broker_connected,
        "fresh": broker_fresh,
        "open_orders_fresh": open_orders_fresh,
        "positions_fresh": positions_fresh,
        "latency_ms": broker_state.get("latency_ms"),
        "last_error": broker_state.get("last_error"),
        "last_success_at": broker_state.get("last_success_at"),
        "failed_components": failed_components,
        "consecutive_failures": _safe_nonnegative_int(
            broker_state.get("consecutive_failures")
        ),
        "stale_age_s": broker_state.get("stale_age_s"),
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
    broker_degraded = bool(
        broker_status_normalized in {"degraded"}
        or not broker_fresh
        or not open_orders_fresh
        or not positions_fresh
    )
    broker_unknown = broker_status_normalized in {"", "unknown"}
    data_degraded = data_status_normalized in {"empty", "degraded"}
    service_degraded = service_status_normalized in {
        "degraded",
        "failed",
        "error",
        "halted",
        "stopped",
    }
    replay_only_service_degradation = bool(
        replay_gate_monitor_only
        and service_reason_normalized == "replay_live_parity_gate_failed"
    )
    service_degraded_for_health = bool(
        service_degraded and not replay_only_service_degradation
    )

    degraded = provider_disabled or provider_payload.get("using_backup") or (
        provider_status_normalized not in {"healthy", "ready"}
    )
    if broker_down or broker_degraded:
        degraded = True
    if provider_unknown or broker_unknown:
        degraded = True
    if data_degraded:
        degraded = True
    if service_degraded_for_health:
        degraded = True

    provider_healthy = provider_status_normalized in {"healthy", "ready"} and not data_degraded
    broker_healthy = bool(
        broker_status_normalized in {"reachable", "ready", "connected"}
        and broker_fresh
        and open_orders_fresh
        and positions_fresh
    )
    overall_ok = provider_healthy and broker_healthy
    if str(ok_mode).strip().lower() == "connectivity":
        provider_connectivity_ok = (
            provider_status_normalized in {"healthy", "ready", "degraded"}
            and not data_degraded
        )
        broker_connectivity_ok = bool(
            broker_status_normalized in {"reachable", "ready", "connected"}
            and broker_fresh
            and open_orders_fresh
            and positions_fresh
        )
        overall_ok = provider_connectivity_ok and broker_connectivity_ok
    offhours_market_closed_ready = (
        market_closed_mode
        and broker_healthy
        and not provider_unknown
        and not broker_down
        and not broker_degraded
        and not service_degraded_for_health
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
        and not provider_unknown
        and not data_degraded
        and not bool(provider_payload.get("using_backup"))
        and not broker_down
        and not broker_degraded
        and not service_degraded_for_health
        and broker_healthy
        and (
            service_phase_age_s is None
            or float(service_phase_age_s) <= float(warmup_fast_path_max_age_s)
        )
    )
    if offhours_market_closed_ready:
        overall_ok = True
        degraded = False
        service_payload["status"] = "ready"
        service_payload["reason"] = "market_closed"
    elif warmup_market_closed_ready:
        overall_ok = True
        degraded = False
        service_payload["status"] = "ready"
        service_payload["reason"] = "market_closed"
    if force_ok_for_pytest:
        overall_ok = True
    require_database_ready = _env_bool("AI_TRADING_HEALTH_REQUIRE_DB_READY", False)
    database_ok = bool(database_readiness.get("ok"))
    if require_database_ready and not database_ok:
        overall_ok = False
        degraded = True
    require_oms_invariants = _env_bool(
        "AI_TRADING_HEALTH_REQUIRE_OMS_INVARIANTS",
        _default_fail_closed_outside_tests(),
    )
    oms_invariants_failure = bool(
        oms_invariants.get("enabled", False)
        and not bool(oms_invariants.get("ok"))
    )
    if require_oms_invariants and oms_invariants_failure:
        overall_ok = False
        degraded = True
    require_oms_lifecycle_parity = _env_bool(
        "AI_TRADING_HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY",
        _default_fail_closed_outside_tests(),
    )
    oms_lifecycle_parity_failure = bool(
        oms_lifecycle_parity.get("enabled", False)
        and not bool(oms_lifecycle_parity.get("ok"))
    )
    if require_oms_lifecycle_parity and oms_lifecycle_parity_failure:
        overall_ok = False
        degraded = True
    if replay_gate_readiness_failure:
        overall_ok = False
        degraded = True
    if service_degraded_for_health:
        overall_ok = False
    if not overall_ok:
        degraded = True
    paper_evidence_operational_ok = bool(
        broker_healthy
        and provider_authority_ok
        and not service_degraded_for_health
        and not replay_gate_readiness_failure
        and (not require_database_ready or database_ok)
        and (not require_oms_invariants or not oms_invariants_failure)
        and (
            not require_oms_lifecycle_parity
            or not oms_lifecycle_parity_failure
        )
    )
    readiness_failures: list[str] = []
    if require_database_ready and not database_ok:
        readiness_failures.append("database_unhealthy")
    if require_oms_invariants and oms_invariants_failure:
        readiness_failures.append("oms_invariants_failed")
    if require_oms_lifecycle_parity and oms_lifecycle_parity_failure:
        readiness_failures.append("oms_lifecycle_parity_failed")
    if replay_gate_readiness_failure:
        readiness_failures.append("replay_live_parity_gate_failed")
    readiness_gates = {
        "oms_invariants": _build_contract_gate_status(
            oms_invariants,
            required=require_oms_invariants,
            failure_reason="oms_invariants_failed",
            attention_flag="oms_invariants_failed",
            action=(
                "Inspect OMS reconciliation invariant violations and repair "
                "intent, order, or broker-state drift before new openings."
            ),
        ),
        "oms_lifecycle_parity": _build_contract_gate_status(
            oms_lifecycle_parity,
            required=require_oms_lifecycle_parity,
            failure_reason="oms_lifecycle_parity_failed",
            attention_flag="oms_lifecycle_parity_failed",
            action=(
                "Inspect OMS lifecycle parity violations and reconcile missing "
                "or mismatched lifecycle events before new openings."
            ),
        ),
    }

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
        observe_oms_invariants=oms_invariants_failure,
        observe_oms_lifecycle_parity=oms_lifecycle_parity_failure,
        trade_updates_stream=trade_updates_stream,
        decision_loop=decision_loop,
    )
    broker_account_configuration = _broker_account_configuration_snapshot()
    if broker_account_configuration.get("available") and broker_account_configuration.get(
        "mismatches"
    ):
        attention_flags = _dedupe_flags(
            [*attention_flags, "broker_account_configuration_drift"]
        )

    failed_checks_raw = replay_live_parity_gate.get("failed_checks")
    failed_checks = (
        [str(item) for item in failed_checks_raw if str(item).strip()]
        if isinstance(failed_checks_raw, (list, tuple, set))
        else []
    )
    # Display only: in monitor-only paper mode, the parity attention flag is
    # withheld when both the health snapshot and the trading cycle's own gate
    # fail only the economics (counterfactual) check. The gate's
    # ok/failed_checks, readiness and entry control are computed as before.
    replay_parity_failed_checks = [
        check for check in failed_checks if check != REPLAY_ECONOMICS_CHECK
    ]
    replay_economics_only_failure = bool(
        replay_gate_enabled
        and replay_gate_available
        and not replay_gate_ok
        and failed_checks == [REPLAY_ECONOMICS_CHECK]
    )
    replay_live_parity_gate_payload: dict[str, Any] = replay_live_parity_gate
    if paper_evidence_mode and replay_gate_enabled:
        runtime_gate_record = _safe_observe(
            runtime_state.observe_replay_live_parity_runtime_gate, None
        )
        runtime_failed_raw = (
            runtime_gate_record.get("failed_checks")
            if isinstance(runtime_gate_record, Mapping)
            else None
        )
        runtime_failed_checks = (
            [str(item) for item in runtime_failed_raw]
            if isinstance(runtime_failed_raw, list)
            else None
        )
        economics_reason = _replay_economics_summary(
            replay_live_parity_gate, withheld_attention_flag=None
        )["reason"]
        withhold_blockers: list[str] = []
        if not replay_gate_monitor_only:
            withhold_blockers.append("replay_gate_not_monitor_only")
        if not replay_economics_only_failure:
            withhold_blockers.append("health_gate_not_economics_only")
        if economics_reason not in _REPLAY_ECONOMICS_FAILURE_REASONS:
            withhold_blockers.append(f"economics_{economics_reason}")
        withhold_blockers.extend(
            _paper_mode_withhold_blockers(
                launch_profile=launch_profile,
                launch_profile_name=launch_profile_name,
            )
        )
        health_observed_raw = replay_live_parity_gate.get("observed")
        health_replay_policy_hash = str(
            (
                health_observed_raw.get("replay_policy_hash")
                if isinstance(health_observed_raw, Mapping)
                else ""
            )
            or ""
        ).strip()
        runtime_blockers = _runtime_replay_gate_withhold_blockers(
            runtime_gate_record,
            service_reason_normalized=service_reason_normalized,
            health_replay_policy_hash=health_replay_policy_hash,
        )
        withhold_blockers.extend(runtime_blockers)
        running_policy_record = _safe_observe(
            runtime_state.observe_running_effective_policy_hash, None
        )
        running_policy_map = (
            running_policy_record if isinstance(running_policy_record, Mapping) else {}
        )
        running_blockers = _running_policy_hash_blockers(
            running_policy_record,
            expected_policy_hash=(
                runtime_gate_record.get("expected_policy_hash")
                if isinstance(runtime_gate_record, Mapping)
                else None
            ),
            health_replay_policy_hash=health_replay_policy_hash,
        )
        withhold_blockers.extend(running_blockers)
        withhold_blockers.extend(_oms_lifecycle_limits_blockers())
        replay_attention_flag_withheld = bool(
            not withhold_blockers
            and REPLAY_PARITY_ATTENTION_FLAG in attention_flags
        )
        if replay_attention_flag_withheld:
            attention_flags = [
                flag
                for flag in attention_flags
                if flag != REPLAY_PARITY_ATTENTION_FLAG
            ]
        parity_failed_union = list(replay_parity_failed_checks)
        runtime_union_checks = list(runtime_failed_checks or [])
        if (
            "replay_policy_hash_mismatch" in runtime_blockers
            or "running_policy_hash_mismatch" in running_blockers
        ):
            runtime_union_checks.append("replay_contract")
        for check in runtime_union_checks:
            if check != REPLAY_ECONOMICS_CHECK and check not in parity_failed_union:
                parity_failed_union.append(check)
        runtime_parity_ok = not running_blockers and not [
            blocker
            for blocker in runtime_blockers
            if blocker != "runtime_gate_not_economics_only"
        ] and not [
            check
            for check in runtime_failed_checks or []
            if check != REPLAY_ECONOMICS_CHECK
        ]
        runtime_record_map = (
            runtime_gate_record if isinstance(runtime_gate_record, Mapping) else {}
        )
        replay_live_parity_gate_payload = {
            **replay_live_parity_gate,
            "parity_ok": bool(
                replay_gate_available
                and (replay_gate_ok or replay_economics_only_failure)
                and runtime_parity_ok
            ),
            "parity_failed_checks": parity_failed_union,
            "runtime_failed_checks": runtime_failed_checks,
            "runtime_gate_evaluated_at": runtime_record_map.get("evaluated_at"),
            "runtime_gate_published_at": runtime_record_map.get("published_at"),
            "runtime_expected_policy_hash": runtime_record_map.get(
                "expected_policy_hash"
            ),
            "runtime_replay_policy_hash": runtime_record_map.get("replay_policy_hash"),
            "runtime_replay_artifact_ts": runtime_record_map.get("replay_artifact_ts"),
            "health_replay_policy_hash": health_replay_policy_hash or None,
            "running_policy_hash": running_policy_map.get("policy_hash"),
            "running_policy_hash_published_at": running_policy_map.get("published_at"),
            "attention_flag_withhold_blockers": withhold_blockers,
            "replay_economics": _replay_economics_summary(
                replay_live_parity_gate,
                withheld_attention_flag=(
                    REPLAY_PARITY_ATTENTION_FLAG
                    if replay_attention_flag_withheld
                    else None
                ),
            ),
        }
    if replay_gate_failure:
        entry_stage = "shadow"
    elif paper_evidence_mode:
        entry_stage = "paper"
    elif launch_profile_name == "live_canary":
        entry_stage = "canary"
    elif live_execution_mode:
        entry_stage = "full"
    else:
        entry_stage = execution_mode or "unknown"
    entry_reason = str(replay_live_parity_gate.get("reason") or "").strip()
    if not entry_reason:
        if replay_gate_failure:
            entry_reason = "replay_live_parity_gate_failed"
        elif not replay_gate_enabled:
            entry_reason = "replay_live_parity_gate_disabled"
        else:
            entry_reason = "replay_live_parity_gate_passed"
    entry_control = {
        "execution_mode": execution_mode,
        "stage": entry_stage,
        "replay_gate_ok": replay_gate_ok,
        "replay_gate_required": bool(require_replay_live_parity_gate),
        "replay_gate_enforced": replay_gate_enforced_for_mode,
        # Replay/model promotion failures must remain visible and keep live
        # capital disabled, but they are monitor-only for conservative paper
        # evidence.  Operational, provider, broker, DB, and OMS failures still
        # fail paper collection closed.
        "paper_evidence_allowed": bool(
            paper_evidence_mode and paper_evidence_operational_ok
        ),
        "live_new_exposure_allowed": bool(
            live_execution_mode
            and not replay_gate_failure
            and overall_ok
            and provider_authority_ok
        ),
        "risk_reducing_orders_allowed": True,
        "reason": entry_reason,
        "failed_checks": failed_checks,
    }

    if degraded:
        resolved_status = "degraded"
    elif offhours_market_closed_ready:
        resolved_status = "healthy"
    elif warmup_market_closed_ready:
        resolved_status = "healthy"
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
        "decision_loop": decision_loop,
        "operating_mode": _operating_mode_snapshot(),
        "latest_training_attempt": latest_training_attempt,
        "database": database_readiness,
        "oms_invariants": oms_invariants,
        "oms_lifecycle_parity": oms_lifecycle_parity,
        "replay_live_parity_gate": replay_live_parity_gate_payload,
        "trade_updates_stream": trade_updates_stream,
        "readiness_gates": readiness_gates,
        "readiness_failures": readiness_failures,
        "attention_flags": attention_flags,
        "entry_control": entry_control,
        "launch_profile": launch_profile_status,
        "provider_authority": provider_authority | {"ok": provider_authority_ok},
        "broker_account_configuration": broker_account_configuration,
        "drawdown_breaker": _drawdown_breaker_snapshot(),
    }
    if readiness_failures:
        payload["reason"] = readiness_failures[0]
    elif offhours_market_closed_ready:
        payload["reason"] = "market_closed"
    elif warmup_market_closed_ready:
        payload["reason"] = "market_closed"
    elif broker_unknown and service_phase_normalized == "warmup":
        payload["reason"] = "broker_status_unknown"
    elif provider_unknown:
        payload["reason"] = "provider_status_unknown"
    elif service_reason and not replay_only_service_degradation and not (
        broker_unknown and service_reason_normalized == "warmup_cycle"
    ):
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
    if broker_unknown and not payload.get("reason"):
        payload["reason"] = "broker_status_unknown"
    if broker_degraded and not broker_unknown and not payload.get("reason"):
        payload["reason"] = (
            broker_state.get("last_error")
            or "broker_snapshot_stale"
        )
    if service_degraded_for_health and not payload.get("reason"):
        payload["reason"] = "service_degraded"
    if require_database_ready and not database_ok and not payload.get("reason"):
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
    if replay_gate_readiness_failure and not payload.get("reason"):
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
    decision_loop = _decision_loop_snapshot()
    database_readiness = _database_readiness_snapshot()
    oms_invariants = _oms_invariants_snapshot()
    oms_lifecycle_parity = _oms_lifecycle_parity_snapshot()
    replay_live_parity_gate = _replay_live_parity_gate_snapshot(
        oms_lifecycle_parity=oms_lifecycle_parity,
    )
    runtime_performance = _runtime_performance_snapshot()
    execution_quality_governor = _execution_quality_governor_snapshot()
    live_cost_model = _live_cost_model_snapshot()
    symbol_universe_scorecard = _symbol_universe_scorecard_snapshot()
    runtime_decay_controls = _runtime_decay_controls_snapshot()
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
        trade_updates_stream=_trade_updates_stream_snapshot(),
        decision_loop=decision_loop,
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
            "governor": execution_quality_governor,
            "live_cost_model": live_cost_model,
            "symbol_universe_scorecard": symbol_universe_scorecard,
            "runtime_decay_controls": runtime_decay_controls,
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
        "decision_loop": decision_loop,
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
    alpaca_preflight = payload["alpaca"].get("preflight", {})
    preflight_failure = bool(
        isinstance(alpaca_preflight, Mapping)
        and alpaca_preflight.get("available") is False
        and alpaca_preflight.get("failure_kind")
    )
    if preflight_failure and not force_ok_for_pytest:
        failure_reason = "alpaca_preflight_unavailable"
        payload["ok"] = False
        payload["status"] = "degraded"
        payload["reason"] = failure_reason
        readiness_failures = payload.setdefault("readiness_failures", [])
        if failure_reason not in readiness_failures:
            readiness_failures.insert(0, failure_reason)
        attention_flags = payload.setdefault("attention_flags", [])
        if failure_reason not in attention_flags:
            attention_flags.insert(0, failure_reason)
        readiness_gates = payload.setdefault("readiness_gates", {})
        readiness_gates["alpaca_preflight"] = {
            "enabled": True,
            "required": True,
            "available": False,
            "ok": False,
            "status": "failed",
            "reason": failure_reason,
            "failure_kind": alpaca_preflight.get("failure_kind"),
            "retry_at": alpaca_preflight.get("retry_at"),
            "retry_in_seconds": alpaca_preflight.get("retry_in_seconds"),
            "consecutive_failures": alpaca_preflight.get(
                "consecutive_failures"
            ),
            "action": (
                "Wait for the bounded Alpaca preflight retry or investigate "
                "credentials and broker connectivity."
            ),
        }
        entry_control = payload.get("entry_control")
        if isinstance(entry_control, dict):
            entry_control["paper_evidence_allowed"] = False
            entry_control["live_new_exposure_allowed"] = False
            entry_control["reason"] = failure_reason

    env_error_text = str(env_error or "").strip()
    if env_error_text:
        if not payload.get("reason"):
            payload["reason"] = env_error_text
        existing_error = str(payload.get("error") or "").strip()
        errors = [error for error in (existing_error, env_error_text) if error]
        payload["error"] = "; ".join(dict.fromkeys(errors))
        payload["ok"] = False
        payload["status"] = "degraded"
    elif force_ok_for_pytest:
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
    env_error_text = str(env_error or "").strip()
    if env_error_text:
        payload["ok"] = False
        payload["status"] = "degraded"
        payload["error"] = env_error_text
    error_text = str(error or "").strip()
    if error_text:
        payload["ok"] = False
        payload["status"] = "degraded"
        existing_error = str(payload.get("error") or "").strip()
        errors = [error for error in (existing_error, error_text) if error]
        payload["error"] = "; ".join(dict.fromkeys(errors))
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


def _json_safe_value(value: Any, seen: set[int]) -> Any:
    """Return ``value`` with nested objects coerced to JSON-safe values."""

    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in seen:
            return "<recursive>"
        seen.add(value_id)
        try:
            return {str(key): _json_safe_value(nested, seen) for key, nested in value.items()}
        finally:
            seen.discard(value_id)
    if isinstance(value, list | tuple):
        value_id = id(value)
        if value_id in seen:
            return ["<recursive>"]
        seen.add(value_id)
        try:
            return [_json_safe_value(item, seen) for item in value]
        finally:
            seen.discard(value_id)
    if isinstance(value, set | frozenset):
        value_id = id(value)
        if value_id in seen:
            return ["<recursive>"]
        seen.add(value_id)
        try:
            return [_json_safe_value(item, seen) for item in sorted(value, key=str)]
        finally:
            seen.discard(value_id)
    try:
        json.dumps(value)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return str(value)
    return value


def sanitize_health_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-compatible copy of a health payload."""

    return {
        str(key): _json_safe_value(value, set())
        for key, value in dict(payload).items()
    }


def register_health_routes(
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

    def _health_handler() -> Any:
        status = 200
        try:
            payload = dict(payload_builder())
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
            if logger is not None:
                try:
                    logger.exception(error_event, exc_info=exc)
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    pass
            payload = build_health_exception_payload(exc, service_name=service_name)
            status = 503
        safe_payload = sanitize_health_payload(payload)
        if not bool(safe_payload.get("ok", True)):
            status = 503
        try:
            return response_builder(safe_payload, status)
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
            if logger is not None:
                try:
                    logger.exception(error_event, exc_info=exc)
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    pass
            fallback_payload = sanitize_health_payload(
                build_health_exception_payload(exc, service_name=service_name),
            )
            try:
                return response_builder(fallback_payload, 503)
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                return fallback_payload, 503

    if len(routes) == 1:
        route_name = routes[0].strip("/").replace("-", "_") or "health"
        _health_handler.__name__ = route_name
    else:
        _health_handler.__name__ = "health_routes"

    for route in routes:
        decorator = app.route(route, methods=list(methods))
        decorator(_health_handler)


__all__ = [
    "build_runtime_health_payload",
    "build_control_plane_snapshot",
    "build_alpaca_health_payload",
    "build_service_health_payload",
    "build_api_health_payload",
    "build_canonical_healthz_payload",
    "build_health_exception_payload",
    "sanitize_health_payload",
    "register_health_routes",
]
