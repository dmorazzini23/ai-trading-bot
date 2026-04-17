from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

_log = get_logger(__name__)


def _json_default(value: Any) -> str:
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            return str(value)
    return str(value)


def policy_ablation_state_path(*, for_write: bool = False) -> Path:
    configured = str(
        get_env(
            "AI_TRADING_POLICY_ABLATION_STATE_PATH",
            "runtime/policy_ablation_state.json",
            cast=str,
        )
        or ""
    ).strip()
    return cast(
        Path,
        resolve_runtime_artifact_path(
            configured or "runtime/policy_ablation_state.json",
            default_relative="runtime/policy_ablation_state.json",
            for_write=for_write,
        ),
    )


def policy_ablation_events_path(*, for_write: bool = False) -> Path:
    configured = str(
        get_env(
            "AI_TRADING_POLICY_ABLATION_EVENTS_PATH",
            "runtime/policy_ablation_events.jsonl",
            cast=str,
        )
        or ""
    ).strip()
    return cast(
        Path,
        resolve_runtime_artifact_path(
            configured or "runtime/policy_ablation_events.jsonl",
            default_relative="runtime/policy_ablation_events.jsonl",
            for_write=for_write,
        ),
    )


def policy_rollback_state_path(*, for_write: bool = False) -> Path:
    configured = str(
        get_env(
            "AI_TRADING_POLICY_ROLLBACK_STATE_PATH",
            "runtime/policy_rollback_state.json",
            cast=str,
        )
        or ""
    ).strip()
    return cast(
        Path,
        resolve_runtime_artifact_path(
            configured or "runtime/policy_rollback_state.json",
            default_relative="runtime/policy_rollback_state.json",
            for_write=for_write,
        ),
    )


def policy_runtime_toggles_path(*, for_write: bool = False) -> Path:
    configured = str(
        get_env(
            "AI_TRADING_POLICY_RUNTIME_TOGGLES_PATH",
            "runtime/policy_runtime_toggles.json",
            cast=str,
            resolve_aliases=False,
        )
        or ""
    ).strip()
    return cast(
        Path,
        resolve_runtime_artifact_path(
            configured or "runtime/policy_runtime_toggles.json",
            default_relative="runtime/policy_runtime_toggles.json",
            for_write=for_write,
        ),
    )


def _normalize_disabled_slices(values: Sequence[Any]) -> list[str]:
    return sorted({str(item).strip().upper() for item in values if str(item).strip()})


def build_policy_runtime_toggles_payload(
    *,
    disabled_slices: Sequence[Any],
    diagnostics: Mapping[str, Any] | None = None,
    updated_at: str | None = None,
    source_updated_at: str | None = None,
) -> dict[str, Any]:
    disabled = _normalize_disabled_slices(disabled_slices)
    disabled_set = set(disabled)
    return {
        "updated_at": updated_at or datetime.now(UTC).isoformat(),
        "source_updated_at": source_updated_at,
        "disabled_slices": disabled,
        "toggles": {
            "rankers": {
                "bandit_enabled": "RANKER:BANDIT" not in disabled_set,
                "counterfactual_enabled": "RANKER:COUNTERFACTUAL" not in disabled_set,
                "geometric_enabled": "RANKER:GEOMETRIC" not in disabled_set,
                "portfolio_log_growth_enabled": (
                    "RANKER:PORTFOLIO_LOG_GROWTH" not in disabled_set
                ),
            },
            "disabled_gate_roots": sorted(
                {
                    token.split(":", 1)[1].strip().upper()
                    for token in disabled
                    if token.startswith("GATE:") and ":" in token
                }
            ),
            "disabled_sleeves": sorted(
                {
                    token.split(":", 1)[1].strip().lower()
                    for token in disabled
                    if token.startswith("SLEEVE:") and ":" in token
                }
            ),
        },
        "diagnostics": dict(diagnostics) if isinstance(diagnostics, Mapping) else {},
    }


def load_policy_rollback_state() -> dict[str, Any]:
    path = policy_rollback_state_path()
    if not path.exists():
        return {"updated_at": None, "disabled_slices": [], "diagnostics": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _log.debug("POLICY_ROLLBACK_STATE_READ_FAILED", exc_info=True)
        return {"updated_at": None, "disabled_slices": [], "diagnostics": {}}
    if not isinstance(payload, Mapping):
        return {"updated_at": None, "disabled_slices": [], "diagnostics": {}}
    disabled_raw = payload.get("disabled_slices")
    disabled = (
        [
            str(item).strip().upper()
            for item in disabled_raw
            if str(item).strip()
        ]
        if isinstance(disabled_raw, Sequence)
        and not isinstance(disabled_raw, (str, bytes, bytearray))
        else []
    )
    diagnostics_raw = payload.get("diagnostics")
    diagnostics = (
        dict(diagnostics_raw) if isinstance(diagnostics_raw, Mapping) else {}
    )
    return {
        "updated_at": payload.get("updated_at"),
        "source_updated_at": payload.get("source_updated_at"),
        "disabled_slices": sorted(set(disabled)),
        "diagnostics": diagnostics,
        "toggle_changes": list(payload.get("toggle_changes", []))
        if isinstance(payload.get("toggle_changes"), Sequence)
        and not isinstance(payload.get("toggle_changes"), (str, bytes, bytearray))
        else [],
    }


def write_policy_rollback_state(payload: Mapping[str, Any]) -> Path:
    path = policy_rollback_state_path(for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def write_policy_runtime_toggles(
    *,
    disabled_slices: Sequence[Any],
    diagnostics: Mapping[str, Any] | None = None,
    updated_at: str | None = None,
    source_updated_at: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    payload = build_policy_runtime_toggles_payload(
        disabled_slices=disabled_slices,
        diagnostics=diagnostics,
        updated_at=updated_at,
        source_updated_at=source_updated_at,
    )
    path = policy_runtime_toggles_path(for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path, payload


def load_policy_runtime_toggles() -> dict[str, Any]:
    path = policy_runtime_toggles_path()
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _log.debug("POLICY_RUNTIME_TOGGLES_READ_FAILED", exc_info=True)
        else:
            if isinstance(payload, Mapping):
                disabled_raw = payload.get("disabled_slices")
                disabled = (
                    [
                        str(item).strip().upper()
                        for item in disabled_raw
                        if str(item).strip()
                    ]
                    if isinstance(disabled_raw, Sequence)
                    and not isinstance(disabled_raw, (str, bytes, bytearray))
                    else []
                )
                toggles_raw = payload.get("toggles")
                diagnostics_raw = payload.get("diagnostics")
                return {
                    "updated_at": payload.get("updated_at"),
                    "source_updated_at": payload.get("source_updated_at"),
                    "disabled_slices": sorted(set(disabled)),
                    "toggles": (
                        dict(toggles_raw)
                        if isinstance(toggles_raw, Mapping)
                        else {}
                    ),
                    "diagnostics": (
                        dict(diagnostics_raw)
                        if isinstance(diagnostics_raw, Mapping)
                        else {}
                    ),
                }
    rollback_payload = load_policy_rollback_state()
    fallback = build_policy_runtime_toggles_payload(
        disabled_slices=cast(
            Sequence[str],
            rollback_payload.get("disabled_slices", []),
        ),
        diagnostics=cast(Mapping[str, Any], rollback_payload.get("diagnostics", {})),
        updated_at=cast(str | None, rollback_payload.get("updated_at")),
        source_updated_at=cast(
            str | None,
            rollback_payload.get("source_updated_at"),
        ),
    )
    try:
        write_policy_runtime_toggles(
            disabled_slices=cast(Sequence[str], fallback.get("disabled_slices", [])),
            diagnostics=cast(Mapping[str, Any], fallback.get("diagnostics", {})),
            updated_at=cast(str | None, fallback.get("updated_at")),
            source_updated_at=cast(str | None, fallback.get("source_updated_at")),
        )
    except Exception as exc:
        _log.warning(
            "POLICY_RUNTIME_TOGGLES_WRITE_FAILED",
            extra={"path": str(policy_runtime_toggles_path(for_write=True)), "error": str(exc)},
        )
    return fallback


def ensure_policy_learning_artifacts(
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    ts = (now or datetime.now(UTC)).isoformat()
    state_path = policy_ablation_state_path(for_write=True)
    events_path = policy_ablation_events_path(for_write=True)
    runtime_toggles_path = policy_runtime_toggles_path(for_write=True)
    context: dict[str, Any] = {
        "state_path": str(state_path),
        "events_path": str(events_path),
        "runtime_toggles_path": str(runtime_toggles_path),
        "state_ready": bool(state_path.exists()),
        "events_ready": bool(events_path.exists()),
        "runtime_toggles_ready": bool(runtime_toggles_path.exists()),
        "state_created": False,
        "events_created": False,
        "runtime_toggles_created": False,
    }

    if not state_path.exists():
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps({"updated_at": None, "slices": {}}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            context["state_created"] = True
            context["state_ready"] = True
            _log.info(
                "POLICY_ABLATION_STATE_BOOTSTRAPPED",
                extra={"path": str(state_path), "ts": ts},
            )
        except Exception as exc:
            context["state_error"] = str(exc)
            _log.warning(
                "POLICY_ABLATION_STATE_BOOTSTRAP_FAILED",
                extra={"path": str(state_path), "error": str(exc)},
            )

    if not events_path.exists():
        try:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            events_path.touch(exist_ok=True)
            context["events_created"] = True
            context["events_ready"] = True
            _log.info(
                "POLICY_ABLATION_EVENTS_BOOTSTRAPPED",
                extra={"path": str(events_path), "ts": ts},
            )
        except Exception as exc:
            context["events_error"] = str(exc)
            _log.warning(
                "POLICY_ABLATION_EVENTS_BOOTSTRAP_FAILED",
                extra={"path": str(events_path), "error": str(exc)},
            )

    if not runtime_toggles_path.exists():
        rollback_payload = load_policy_rollback_state()
        try:
            _, payload = write_policy_runtime_toggles(
                disabled_slices=cast(
                    Sequence[str],
                    rollback_payload.get("disabled_slices", []),
                ),
                diagnostics=cast(Mapping[str, Any], rollback_payload.get("diagnostics", {})),
                updated_at=cast(str | None, rollback_payload.get("updated_at")),
                source_updated_at=cast(
                    str | None,
                    rollback_payload.get("source_updated_at"),
                ),
            )
            context["runtime_toggles_created"] = True
            context["runtime_toggles_ready"] = True
            context["runtime_toggles_payload"] = payload
            _log.info(
                "POLICY_RUNTIME_TOGGLES_BOOTSTRAPPED",
                extra={"path": str(runtime_toggles_path), "ts": ts},
            )
        except Exception as exc:
            context["runtime_toggles_error"] = str(exc)
            _log.warning(
                "POLICY_RUNTIME_TOGGLES_BOOTSTRAP_FAILED",
                extra={"path": str(runtime_toggles_path), "error": str(exc)},
            )
    return context


class RiskApprovalService:
    """Shared runtime service for policy toggles and manual override state."""

    def load_runtime_toggles(self) -> dict[str, Any]:
        return load_policy_runtime_toggles()

    def update_manual_overrides(
        self,
        *,
        disabled_slices: Sequence[Any],
        diagnostics: Mapping[str, Any] | None = None,
        source_updated_at: str | None = None,
    ) -> dict[str, Any]:
        path, payload = write_policy_runtime_toggles(
            disabled_slices=disabled_slices,
            diagnostics=diagnostics,
            source_updated_at=source_updated_at,
        )
        _log.info(
            "OPERATOR_MANUAL_OVERRIDES_UPDATED",
            extra={"path": str(path), "disabled_slice_count": len(payload["disabled_slices"])},
        )
        return {"available": True, "path": str(path), "state": payload}

    def clear_manual_overrides(self) -> dict[str, Any]:
        return self.update_manual_overrides(disabled_slices=[], diagnostics={})

    def ensure_policy_learning_artifacts(
        self,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        return ensure_policy_learning_artifacts(now=now)


__all__ = [
    "RiskApprovalService",
    "build_policy_runtime_toggles_payload",
    "ensure_policy_learning_artifacts",
    "load_policy_rollback_state",
    "load_policy_runtime_toggles",
    "policy_ablation_events_path",
    "policy_ablation_state_path",
    "policy_rollback_state_path",
    "policy_runtime_toggles_path",
    "write_policy_rollback_state",
    "write_policy_runtime_toggles",
]
