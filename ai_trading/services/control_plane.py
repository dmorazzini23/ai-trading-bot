from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from ai_trading.health_payload import build_control_plane_snapshot
from ai_trading.services.risk_approval import RiskApprovalService

_CONTROL_PLANE_SECTION_KEYS = {
    "rollout": "rollout",
    "broker-health": "broker_health",
    "broker_health": "broker_health",
    "positions": "positions",
    "open-orders": "open_orders",
    "open_orders": "open_orders",
    "execution-quality": "execution_quality",
    "execution_quality": "execution_quality",
    "circuit-breakers": "circuit_breakers",
    "circuit_breakers": "circuit_breakers",
    "liveness": "liveness",
    "manual-overrides": "manual_overrides",
    "manual_overrides": "manual_overrides",
    "governance": "governance",
    "services": "services",
}
class ControlPlaneService:
    """Service wrapper around canonical runtime control-plane views."""

    def __init__(self, *, service_name: str="ai-trading") -> None:
        self._service_name = str(service_name or "ai-trading")

    def snapshot(self) -> dict[str, Any]:
        snapshot = cast(
            dict[str, Any],
            build_control_plane_snapshot(service_name=self._service_name),
        )
        snapshot["services"] = self.service_boundaries(snapshot=snapshot)
        return snapshot

    def section(self, name: str) -> dict[str, Any]:
        key = _CONTROL_PLANE_SECTION_KEYS.get(str(name or "").strip().lower())
        if not key:
            raise KeyError(name)
        snapshot = self.snapshot()
        section = snapshot.get(key)
        if isinstance(section, Mapping):
            return dict(section)
        return {"value": section}

    def service_boundaries(
        self,
        *,
        snapshot: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        view = dict(snapshot) if isinstance(snapshot, Mapping) else self.snapshot()
        rollout = dict(view.get("rollout", {})) if isinstance(view.get("rollout"), Mapping) else {}
        broker_health = (
            dict(view.get("broker_health", {}))
            if isinstance(view.get("broker_health"), Mapping)
            else {}
        )
        data_provider = (
            dict(view.get("data_provider", {}))
            if isinstance(view.get("data_provider"), Mapping)
            else {}
        )
        positions = dict(view.get("positions", {})) if isinstance(view.get("positions"), Mapping) else {}
        execution_quality = (
            dict(view.get("execution_quality", {}))
            if isinstance(view.get("execution_quality"), Mapping)
            else {}
        )
        circuit_breakers = (
            dict(view.get("circuit_breakers", {}))
            if isinstance(view.get("circuit_breakers"), Mapping)
            else {}
        )
        governance = (
            dict(view.get("governance", {}))
            if isinstance(view.get("governance"), Mapping)
            else {}
        )
        manual_overrides = (
            dict(view.get("manual_overrides", {}))
            if isinstance(view.get("manual_overrides"), Mapping)
            else {}
        )
        return {
            "signal": {
                "owner": "ai_trading.services.signal",
                "status": data_provider.get("status") or rollout.get("status") or "unknown",
                "inputs": ["data_provider", "quotes"],
            },
            "portfolio": {
                "owner": "ai_trading.services.portfolio",
                "boundary_type": "facade",
                "canonical_runtime_owner": "ai_trading.portfolio.compute_portfolio_weights",
                "status": (
                    "ready"
                    if positions.get("reconciliation_available")
                    else "degraded"
                ),
                "inputs": ["positions"],
            },
            "risk_approval": {
                "owner": "ai_trading.services.risk_approval",
                "status": (
                    "blocked"
                    if not circuit_breakers.get("go_no_go_gate_passed", True)
                    else "ready"
                ),
                "inputs": ["circuit_breakers", "manual_overrides"],
                "manual_override_path": manual_overrides.get("path"),
            },
            "execution": {
                "owner": "ai_trading.services.execution",
                "boundary_type": "facade",
                "canonical_runtime_owner": [
                    "ai_trading.core.legacy_submit_runtime.submit_order_runtime",
                    "ai_trading.core.legacy_trade_cycle.execute_legacy_trade_logic",
                ],
                "status": broker_health.get("status") or rollout.get("status") or "unknown",
                "inputs": ["broker_health", "open_orders", "execution_quality"],
                "tca_available": execution_quality.get("oms_event_tca_available"),
            },
            "reconciliation": {
                "owner": "ai_trading.services.reconciliation",
                "boundary_type": "facade",
                "canonical_runtime_owner": [
                    "ai_trading.services.reconciliation.ReconciliationService.reconcile_position_targets",
                    "ai_trading.services.reconciliation.ReconciliationService.require_success",
                ],
                "status": (
                    "ready"
                    if positions.get("reconciliation_consistent")
                    else "degraded"
                ),
                "inputs": ["positions", "open_orders", "oms_invariants", "oms_lifecycle_parity"],
            },
            "governance": {
                "owner": "ai_trading.services.governance",
                "status": "ready" if governance else "degraded",
                "inputs": ["governance"],
            },
        }

    def get_manual_overrides(self) -> dict[str, Any]:
        return self.section("manual_overrides")

    def update_manual_overrides(
        self,
        *,
        disabled_slices: Sequence[Any],
        diagnostics: Mapping[str, Any] | None = None,
        source_updated_at: str | None = None,
    ) -> dict[str, Any]:
        return RiskApprovalService().update_manual_overrides(
            disabled_slices=disabled_slices,
            diagnostics=diagnostics,
            source_updated_at=source_updated_at,
        )

    def clear_manual_overrides(self) -> dict[str, Any]:
        return RiskApprovalService().clear_manual_overrides()


__all__ = ["ControlPlaneService"]
