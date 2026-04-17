from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ai_trading.health_payload import build_control_plane_snapshot


class GovernanceService:
    """Thin operator service for governance snapshots and actions."""

    def __init__(self, *, service_name: str="ai-trading", base_path: str="artifacts/governance") -> None:
        self._service_name = str(service_name or "ai-trading")
        self._base_path = str(base_path or "artifacts/governance")

    def _promotion(self) -> Any:
        from ai_trading.governance.promotion import ModelPromotion

        return ModelPromotion(base_path=self._base_path)

    def snapshot(self) -> dict[str, Any]:
        snapshot = build_control_plane_snapshot(service_name=self._service_name)
        governance = snapshot.get("governance")
        if isinstance(governance, Mapping):
            return dict(governance)
        return {}

    def record_approval(
        self,
        *,
        strategy: str,
        model_id: str,
        approver: str,
        decision: str = "approved",
        note: str | None = None,
        ticket: str | None = None,
    ) -> dict[str, Any]:
        promotion = self._promotion()
        output_path = promotion.record_promotion_approval(
            strategy=strategy,
            model_id=model_id,
            approver=approver,
            decision=decision,
            note=note,
            ticket=ticket,
        )
        return {
            "path": output_path,
            "approvals": promotion.list_recent_promotion_approvals(limit=5),
        }

    def rollback(
        self,
        *,
        strategy: str,
        reason: str,
        force: bool = True,
    ) -> dict[str, Any]:
        promotion = self._promotion()
        rolled_back = bool(
            promotion.rollback_to_previous_production(
                strategy=strategy,
                reason=reason,
                force=force,
            )
        )
        return {
            "ok": rolled_back,
            "rolled_back": rolled_back,
            "audit": promotion.list_recent_rollback_audit(limit=5),
        }


__all__ = ["GovernanceService"]
